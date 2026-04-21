"""
Google Gemini Provider — API Key & Gemini CLI OAuth via direct HTTP.

Uses PredaCore's in-house PredaCore SDK (``predacore_sdk.py``) — zero
external vendor-SDK dependency. POSTs directly to
``https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent``
with either an API key or a Gemini CLI OAuth token.

Auth strategies, in order:
  1. ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` env var
  2. ``config.api_key`` (unless it looks like an OpenAI-style key)
  3. Cached Gemini CLI credentials at ``~/.gemini/credentials.json``

The previous version of this file supported Google Application Default
Credentials via ``google-auth``; that path was removed as part of the
in-house SDK migration (vendor SDKs dropped). If you need ADC, export
an API key from Google AI Studio instead.

Tool use + text-based fallback follow the same pattern as the other
providers.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import predacore_sdk as psdk
from .base import LLMProvider
from .text_tool_adapter import build_tool_prompt, parse_tool_calls
from .types import AssistantResponse, Message, ToolCallRef, ToolResultRef

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(LLMProvider):
    """Google Gemini provider — raw httpx via PredaCore SDK."""

    name = "gemini"

    # ------------------------------------------------------------------
    # Tool-turn serialization (Phase A refactor — 2026-04-21).
    #
    # Gemini's wire format differs from OpenAI + Anthropic:
    #   - Roles: ``model`` (assistant) / ``user`` (everything else, incl. tool)
    #   - Turns have ``parts[]`` not ``content``
    #   - Tool calls: ``{"functionCall": {"name", "args"}}`` part
    #   - Tool results: ``{"functionResponse": {"name", "response"}}`` part
    #   - No native tool-call IDs — linkage by position; we synthesize
    #     stable IDs so the abstract ``ToolCallRef.id`` round-trips
    #
    # Both append_* methods override to stash wire-ready ``parts`` in
    # ``provider_extras["parts"]``. ``_build_payload`` prefers those parts
    # over reconstructing from ``content``.
    # ------------------------------------------------------------------

    def append_assistant_turn(
        self,
        messages: list["Message"],
        response: "AssistantResponse",
    ) -> None:
        import uuid

        # When the parser stashed the raw ``parts[]`` from the Gemini response
        # under provider_extras["content_parts"], replay them verbatim. This
        # preserves ``thoughtSignature`` (required on functionCall parts for
        # Gemini 2.5+/3.x reasoning models — omitting it causes HTTP 400
        # "Function call is missing a thought_signature") plus any other
        # per-part metadata Google may add later. Mirrors the Anthropic
        # ``content_blocks`` round-trip pattern.
        raw_parts = response.provider_extras.get("content_parts")
        use_raw = (
            isinstance(raw_parts, list)
            and any(
                isinstance(p, dict) and "functionCall" in p
                for p in raw_parts
            )
        )

        if use_raw:
            parts = list(raw_parts)
        else:
            # Fallback path — rebuild parts from typed fields. Used when the
            # response came via the text-tool fallback (no native functionCall
            # parts) or when a caller constructs AssistantResponse manually.
            # Loses thoughtSignature, but that only matters on the native path.
            parts = []
            if response.content:
                parts.append({"text": response.content})
            for tc in response.tool_calls:
                parts.append(
                    {"functionCall": {"name": tc.name, "args": tc.arguments}}
                )

        normalized_tool_calls: list[ToolCallRef] = []
        for i, tc in enumerate(response.tool_calls):
            # Gemini doesn't emit IDs — synthesize stable one if absent so the
            # downstream ``ToolResultRef.call_id`` has something to match.
            tc_id = tc.id or f"gemini_{i}_{uuid.uuid4().hex[:8]}"
            normalized_tool_calls.append(
                ToolCallRef(id=tc_id, name=tc.name, arguments=tc.arguments)
            )

        messages.append(
            Message(
                role="assistant",  # abstract — _build_payload maps to "model"
                content=response.content,
                tool_calls=normalized_tool_calls,
                provider_extras={
                    **response.provider_extras,
                    "parts": parts,
                },
            )
        )

    def append_tool_results_turn(
        self,
        messages: list["Message"],
        results: list["ToolResultRef"],
    ) -> None:
        if not results:
            return
        parts: list[dict[str, Any]] = []
        for r in results:
            response_obj: dict[str, Any] = {"result": r.result}
            if r.is_error:
                response_obj["is_error"] = True
            parts.append(
                {
                    "functionResponse": {
                        "name": r.name,
                        "response": response_obj,
                    }
                }
            )
        messages.append(
            Message(
                role="user",  # Gemini: tool results go in a user turn
                content="",  # Gemini ignores text content when parts[] is present
                tool_results=list(results),
                provider_extras={"parts": parts},
            )
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request to Gemini."""
        model_name = self.config.model or "gemini-2.0-flash"
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        api_key = (
            os.getenv("GEMINI_API_KEY", "")
            or os.getenv("GOOGLE_API_KEY", "")
        )
        if not api_key and self.config.api_key and not self.config.api_key.startswith("sk-"):
            api_key = self.config.api_key

        oauth_token = None
        if not api_key:
            oauth_token = _load_gemini_cli_token()

        if not api_key and not oauth_token:
            raise psdk.AuthenticationError(
                "No Gemini credentials found. Set GEMINI_API_KEY / GOOGLE_API_KEY, "
                "or run `gemini auth login` to cache credentials at "
                "~/.gemini/credentials.json."
            )

        payload = _build_payload(
            messages=messages,
            tools=tools,
            temperature=temp,
            max_tokens=max_tok,
        )

        endpoint, headers = _build_request_target(
            model_name=model_name,
            api_key=api_key,
            oauth_token=oauth_token,
        )
        max_retries = int(os.getenv("PREDACORE_API_MAX_RETRIES", "3"))

        async with psdk.make_client() as client:
            try:
                resp = await psdk.request_with_retry(
                    client,
                    "POST",
                    endpoint,
                    headers=headers,
                    json=payload,
                    max_retries=max_retries,
                )
                return _parse_response(resp.json(), model_name)
            except psdk.BadRequestError as exc:
                err_msg = str(exc).lower()
                is_tool_error = tools and ("tool" in err_msg or "function" in err_msg)
                if is_tool_error:
                    logger.warning(
                        "gemini: native tools rejected — text-tool fallback"
                    )
                    return await _fallback_text_tools(
                        client, endpoint, headers, payload, tools, model_name, max_retries,
                    )
                raise
            except psdk.LLMError as exc:
                logger.error("gemini error: %s", exc)
                raise


# ---------------------------------------------------------------------------
# Module-level helpers (stateless)
# ---------------------------------------------------------------------------


def _load_gemini_cli_token() -> str | None:
    """Read a cached OAuth token from ``~/.gemini/credentials.json`` if present."""
    try:
        cache_path = Path.home() / ".gemini" / "credentials.json"
        if not cache_path.exists():
            return None
        creds = json.loads(cache_path.read_text())
        token = creds.get("access_token")
        if token:
            logger.info("gemini: using CLI cached credentials")
            return token
    except (OSError, ValueError, KeyError) as exc:
        logger.debug("gemini CLI cache lookup failed: %s", exc)
    return None


def _build_request_target(
    *,
    model_name: str,
    api_key: str,
    oauth_token: str | None,
) -> tuple[str, dict[str, str]]:
    """Return (endpoint URL, headers) based on the auth mode.

    For API-key auth, we put the key in the ``x-goog-api-key`` header
    rather than the URL query string — matches the official SDK's
    behavior and keeps the key out of access logs.
    """
    endpoint = f"{GEMINI_BASE_URL}/models/{model_name}:generateContent"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["x-goog-api-key"] = api_key
    elif oauth_token:
        headers["Authorization"] = f"Bearer {oauth_token}"
    return endpoint, headers


def _build_payload(
    *,
    messages: list[dict[str, str]],
    tools: list[dict] | None,
    temperature: float | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    """Build the Gemini ``generateContent`` request payload.

    Gemini's wire format differs from OpenAI/Anthropic:
      * System prompts go in ``systemInstruction``, not ``contents``
      * User/assistant turns go in ``contents`` with role ``user``/``model``
      * Each turn's text is wrapped in ``parts: [{text: ...}]``
      * Tools are ``functionDeclarations`` under ``tools[0]``
    """
    contents: list[dict[str, Any]] = []
    system_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # NEW (Phase A): prefer pre-built parts from provider_extras when our
        # append_*_turn methods stashed wire-ready functionCall/functionResponse.
        # Falls back to text-based reconstruction for legacy callers.
        prebuilt_parts = msg.get("parts")
        # Tolerate list-of-blocks content (strip to text)
        if isinstance(content, list):
            content = "".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        if role == "system":
            if isinstance(prebuilt_parts, list):
                system_parts.extend(
                    p.get("text", "") for p in prebuilt_parts if isinstance(p, dict) and "text" in p
                )
            else:
                system_parts.append(content)
        elif role == "assistant":
            parts = prebuilt_parts if isinstance(prebuilt_parts, list) else [{"text": content}]
            contents.append({"role": "model", "parts": parts})
        else:
            # user OR tool (which we route into user turn per Gemini convention)
            parts = prebuilt_parts if isinstance(prebuilt_parts, list) else [{"text": content}]
            contents.append({"role": "user", "parts": parts})

    payload: dict[str, Any] = {"contents": contents}

    if system_parts:
        payload["systemInstruction"] = {
            "parts": [{"text": "\n".join(system_parts)}]
        }

    generation_config: dict[str, Any] = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = max_tokens
    if generation_config:
        payload["generationConfig"] = generation_config

    if tools:
        func_decls: list[dict[str, Any]] = []
        for t in tools:
            decl: dict[str, Any] = {
                "name": t["name"],
                "description": t.get("description", ""),
            }
            params = t.get("parameters", {})
            if isinstance(params, dict) and params.get("properties"):
                decl["parameters"] = params
            func_decls.append(decl)
        payload["tools"] = [{"functionDeclarations": func_decls}]

    return payload


async def _fallback_text_tools(
    client: Any,
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    tools: list[dict],
    model_name: str,
    max_retries: int,
) -> dict[str, Any]:
    """Strip native tools, embed them in systemInstruction as text, retry."""
    payload = dict(payload)
    payload.pop("tools", None)

    tool_prompt = build_tool_prompt(tools)
    if "systemInstruction" in payload:
        existing_text = payload["systemInstruction"]["parts"][0].get("text", "")
        payload["systemInstruction"]["parts"][0]["text"] = existing_text + tool_prompt
    else:
        payload["systemInstruction"] = {"parts": [{"text": tool_prompt}]}

    resp = await psdk.request_with_retry(
        client, "POST", endpoint, headers=headers, json=payload, max_retries=max_retries,
    )
    result = _parse_response(resp.json(), model_name)
    clean_text, text_tool_calls = parse_tool_calls(result["content"])
    if text_tool_calls:
        result["content"] = clean_text
        result["tool_calls"] = text_tool_calls
        result["finish_reason"] = "tool_calls"
    return result


def _parse_response(data: dict, model_name: str) -> dict[str, Any]:
    """Parse Gemini ``generateContent`` response into the router's standard shape.

    Handles three response shapes the original parser missed:
      * Thinking parts (``{"thought": true, "text": ...}`` or ``{"thinking": ...}``)
        emitted by reasoning models like ``gemini-3.1-pro-preview``. Reasoning
        is excluded from visible output.
      * Empty ``candidates`` — request was rejected before generation, usually
        by a safety filter. ``promptFeedback.blockReason`` surfaces the cause.
      * Non-empty ``candidates`` with no text parts and a non-stop
        ``finishReason`` (``SAFETY``, ``RECITATION``, ``MAX_TOKENS``, etc.) —
        model generated nothing usable. Surface the reason so the user
        sees actionable feedback instead of a blank message.
    """
    candidates = data.get("candidates") or []
    content_text = ""
    thinking_text = ""
    tool_calls_out: list[dict] = []
    finish_reason = "stop"

    if candidates:
        cand = candidates[0]
        parts = cand.get("content", {}).get("parts", [])
        for part in parts:
            if not isinstance(part, dict):
                continue
            if "functionCall" in part:
                fc = part["functionCall"] or {}
                tool_calls_out.append(
                    {
                        "name": fc.get("name", ""),
                        "arguments": fc.get("args", {}) or {},
                    }
                )
                finish_reason = "tool_calls"
            elif part.get("thought") is True or "thinking" in part:
                # Gemini thinking models return reasoning with either
                # ``{"thought": true, "text": ...}`` (current shape) or
                # ``{"thinking": "..."}`` (older preview shape). Either way,
                # internal reasoning is not sent to the user — skip it.
                thinking_text += part.get("text") or part.get("thinking") or ""
            elif "text" in part:
                content_text += part.get("text") or ""

        # Surface non-stop finish reasons when the model produced no output.
        if not content_text and not tool_calls_out:
            cand_finish = (cand.get("finishReason") or cand.get("finish_reason") or "").upper()
            if cand_finish == "SAFETY":
                content_text = "[Response blocked by Gemini safety filter. Try rephrasing or relax the prompt.]"
                finish_reason = "content_filter"
            elif cand_finish == "RECITATION":
                content_text = "[Response blocked: matched copyrighted content. Rephrase the prompt.]"
                finish_reason = "content_filter"
            elif cand_finish == "MAX_TOKENS":
                content_text = "[Response hit max_tokens before producing visible text. Raise max_tokens or lower reasoning effort.]"
                finish_reason = "length"
            elif cand_finish and cand_finish != "STOP":
                content_text = f"[Gemini returned no content (finishReason={cand_finish}).]"
                finish_reason = "stop"
    else:
        # No candidates = prompt rejected before generation.
        feedback = data.get("promptFeedback") or {}
        block_reason = feedback.get("blockReason") or ""
        if block_reason:
            content_text = f"[Prompt blocked by Gemini safety filter ({block_reason}). Rephrase the request.]"
            finish_reason = "content_filter"
        else:
            # Genuinely empty response — don't return "" (downstream code
            # treats empty as a silent success, so the user sees a blank turn).
            content_text = "[Empty response from Gemini — no candidates and no feedback. This usually means a transient API issue; try again.]"

    # Parse text-based tool calls if no native ones found
    if not tool_calls_out and content_text:
        clean_text, text_calls = parse_tool_calls(content_text)
        if text_calls:
            content_text = clean_text
            tool_calls_out = text_calls
            finish_reason = "tool_calls"

    usage = data.get("usageMetadata") or {}
    # Gemini 2.5+ and 3.x automatically cache repeated prefixes server-side
    # and report hits via `cachedContentTokenCount`. Older models + explicit
    # caching callers get the same field. We surface it so callers can track
    # cache hit rate — which directly drives cost.
    cached_tokens = int(usage.get("cachedContentTokenCount", 0) or 0)
    prompt_tokens = int(usage.get("promptTokenCount", 0) or 0)
    cache_hit_ratio = (cached_tokens / prompt_tokens) if prompt_tokens else 0.0
    if cached_tokens:
        logger.info(
            "gemini cache hit: %d / %d prompt tokens cached (%.0f%%)",
            cached_tokens, prompt_tokens, cache_hit_ratio * 100,
        )
    result: dict[str, Any] = {
        "content": content_text,
        "tool_calls": tool_calls_out,
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
            "thoughts_token_count": int(usage.get("thoughtsTokenCount", 0) or 0),
            # Implicit + explicit cache hits. Multiply by provider pricing to
            # get realised dollar savings.
            "cached_content_tokens": cached_tokens,
            "cache_hit_ratio": round(cache_hit_ratio, 3),
        },
        "thinking": thinking_text,
        "model": model_name,
    }

    # Preserve the raw ``parts[]`` from the response when it contains native
    # functionCall parts. Gemini 2.5+/3.x reasoning models attach a
    # ``thoughtSignature`` as a sibling field on each functionCall part; the
    # next request must echo that signature back verbatim or Gemini rejects
    # with HTTP 400 ("Function call is missing a thought_signature"). Rather
    # than thread thoughtSignature through the neutral type system, we stash
    # the whole parts list and let ``append_assistant_turn`` replay it.
    # Skipped for text-only responses and for the text-tool fallback path
    # (tool_calls_out would be empty from the native parse since the server
    # never emitted a functionCall part in that case).
    if tool_calls_out and candidates:
        raw_parts = candidates[0].get("content", {}).get("parts", [])
        if isinstance(raw_parts, list) and any(
            isinstance(p, dict) and "functionCall" in p for p in raw_parts
        ):
            result["content_parts"] = list(raw_parts)

    return result
