"""
Google Gemini Provider — API Key & Gemini CLI OAuth via direct HTTP.

Uses JARVIS's in-house Prometheus SDK (``prometheus_sdk.py``) — zero
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
from pathlib import Path
from typing import Any, Callable

from . import prometheus_sdk as psdk
from .base import LLMProvider
from .text_tool_adapter import build_tool_prompt, parse_tool_calls

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(LLMProvider):
    """Google Gemini provider — raw httpx via Prometheus SDK."""

    name = "gemini"

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
        max_retries = int(os.getenv("JARVIS_API_MAX_RETRIES", "3"))

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
        # Tolerate list-of-blocks content (strip to text)
        if isinstance(content, list):
            content = "".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})
        else:
            contents.append({"role": "user", "parts": [{"text": content}]})

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
    """Parse Gemini ``generateContent`` response into the router's standard shape."""
    candidates = data.get("candidates") or []
    content_text = ""
    tool_calls_out: list[dict] = []
    finish_reason = "stop"

    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            if isinstance(part, dict) and "functionCall" in part:
                fc = part["functionCall"] or {}
                tool_calls_out.append(
                    {
                        "name": fc.get("name", ""),
                        "arguments": fc.get("args", {}) or {},
                    }
                )
                finish_reason = "tool_calls"
            elif isinstance(part, dict) and "text" in part:
                content_text += part.get("text") or ""

    # Parse text-based tool calls if no native ones found
    if not tool_calls_out and content_text:
        clean_text, text_calls = parse_tool_calls(content_text)
        if text_calls:
            content_text = clean_text
            tool_calls_out = text_calls
            finish_reason = "tool_calls"

    usage = data.get("usageMetadata") or {}
    return {
        "content": content_text,
        "tool_calls": tool_calls_out,
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
        },
        "model": model_name,
    }
