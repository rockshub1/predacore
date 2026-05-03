"""
OpenAI Responses API adapter — POST /v1/responses.

OpenAI announced (2026-Q1) that Chat Completions support is being phased
out for ChatGPT-OAuth (Codex) tokens. The Responses API is the canonical
replacement and is structurally closer to Anthropic's content-blocks
shape: ``input[]`` and ``output[]`` are flat arrays of typed items
(``message`` / ``function_call`` / ``function_call_output`` / ``reasoning``).

This adapter targets that wire format. ``OpenAICodexProvider`` extends it
to swap in OAuth-bearer auth; the regular ``OpenAIProvider`` continues
to speak Chat Completions for now (the deprecation is OAuth-specific).

Wire format reference (key shape differences vs Chat Completions):

  Chat Completions  →  Responses API
  ─────────────────────────────────────────────────────────────
  POST /v1/chat/completions     POST /v1/responses
  body.messages: [...]          body.input: [...]
  message.tool_calls[].          input[type=function_call].
    {id, function.{name,           {call_id, name,
     arguments(JSON-string)}}     arguments(JSON-string)}
  role:"tool" message            input[type=function_call_output].
   with tool_call_id              {call_id, output}
  choices[0].message             output[type=message].content
  choices[0].finish_reason       output[type=function_call] +
                                  response.status
  usage.{prompt,completion}_     usage.{input,output}_tokens
   tokens
  body.max_tokens                body.max_output_tokens
  tools[type=function,           tools[type=function, name,
        function.{name,desc,            description, parameters]
        parameters}]             (flatter — no nested "function")
  SSE choices[].delta            SSE response.output_item.added/delta

Tool-flow validator: ``openai_validator`` is Chat-Completions-shaped, so
this module ships its own thin per-call check (orphan call_id detection)
inline in :func:`_validate_responses_input`. The full validator can be
layered on later if needed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable
from typing import Any

from . import predacore_sdk as psdk
from .base import LLMProvider

logger = logging.getLogger(__name__)


RESPONSES_API_BASE_URL = "https://api.openai.com/v1"
RESPONSES_API_PATH = "/responses"


class OpenAIResponsesProvider(LLMProvider):
    """OpenAI Responses API adapter — direct API key (or subclass for OAuth)."""

    name = "openai_responses"

    # ------------------------------------------------------------------
    # Tool-turn serialization is inherited from base — we DON'T override
    # append_assistant_turn / append_tool_results_turn. Reason: the
    # abstract Message shape (role + content + tool_calls + tool_results)
    # is sufficient; ``_serialize_messages_for_responses`` translates
    # that to Responses-API ``input[]`` items at request time.
    # ------------------------------------------------------------------

    def _resolve_endpoint_and_auth(self) -> tuple[str, str]:
        """Return (api_key, base_url). Subclasses override for OAuth."""
        api_key = os.getenv("OPENAI_API_KEY", "") or self.config.api_key
        if not api_key:
            raise psdk.AuthenticationError(
                "openai_responses: missing API key. "
                "Set OPENAI_API_KEY or configure api_key in ~/.predacore/config.yaml."
            )
        base_url = (self.config.api_base or RESPONSES_API_BASE_URL).rstrip("/")
        return api_key, base_url

    def _build_headers(self, auth_value: str) -> dict[str, str]:
        """Build request headers. Subclasses may override for OAuth bearer."""
        headers = {"Content-Type": "application/json"}
        if auth_value:
            headers["Authorization"] = f"Bearer {auth_value}"
        return headers

    @staticmethod
    def _serialize_messages_for_responses(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Translate abstract messages → Responses-API ``input[]`` items.

        Accepts the same dict shape as ``OpenAIProvider._serialize_messages_for_wire``
        produces — abstract role/content with optional ``tool_calls`` /
        ``tool_results`` keys. Emits typed Responses items.
        """
        out: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")

            # Assistant turn with tool_calls → emit message + function_call items
            if role == "assistant" and m.get("tool_calls"):
                content = m.get("content") or ""
                if content:
                    out.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": content,
                        }
                    )
                for tc in m["tool_calls"]:
                    args = tc.get("arguments", {})
                    args_str = args if isinstance(args, str) else json.dumps(args)
                    out.append(
                        {
                            "type": "function_call",
                            "call_id": tc.get("id", "") or "",
                            "name": tc.get("name", ""),
                            "arguments": args_str,
                        }
                    )
                continue

            # Tool result turn → function_call_output items
            if role == "tool" and m.get("tool_results"):
                for tr in m["tool_results"]:
                    out.append(
                        {
                            "type": "function_call_output",
                            "call_id": tr.get("call_id", "") or "",
                            "output": tr.get("result", ""),
                        }
                    )
                continue

            # Plain user / assistant / system message
            content = m.get("content")
            if content is None:
                continue
            out.append(
                {
                    "type": "message",
                    "role": role,
                    "content": content if isinstance(content, str) else str(content),
                }
            )

        return out

    @staticmethod
    def _serialize_tools_for_responses(
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """Translate abstract tool defs → Responses flat ``tools[]`` items.

        Responses API uses a flatter shape than Chat Completions:
          {"type": "function", "name", "description", "parameters"}
        instead of:
          {"type": "function", "function": {"name", "description", "parameters"}}
        """
        if not tools:
            return None
        out: list[dict[str, Any]] = []
        for t in tools:
            decl: dict[str, Any] = {
                "type": "function",
                "name": t.get("name", ""),
                "description": t.get("description", ""),
            }
            params = t.get("parameters")
            if isinstance(params, dict):
                decl["parameters"] = params
            out.append(decl)
        return out

    def repair_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate and repair the Responses-API ``input[]`` array.

        Lighter than the Chat-Completions validator since the wire shape is
        already typed-item-per-position. Drops orphan ``function_call_output``
        items (no preceding ``function_call`` with the same call_id).
        """
        return _repair_responses_input(messages)

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """Send a Responses-API request."""
        api_key, base_url = self._resolve_endpoint_and_auth()
        url = f"{base_url}{RESPONSES_API_PATH}"
        headers = self._build_headers(api_key)
        max_retries = int(os.getenv("PREDACORE_API_MAX_RETRIES", "3"))

        wire_input = self._serialize_messages_for_responses(list(messages))
        wire_input = self.repair_messages(wire_input)

        model = self.config.model or "gpt-5"
        body: dict[str, Any] = {
            "model": model,
            "input": wire_input,
            "temperature": (
                temperature if temperature is not None else self.config.temperature
            ),
            "max_output_tokens": (
                max_tokens if max_tokens is not None else self.config.max_tokens
            ),
        }
        wire_tools = self._serialize_tools_for_responses(tools)
        if wire_tools:
            body["tools"] = wire_tools
            body["tool_choice"] = "auto"

        async with psdk.make_client() as client:
            # ── Streaming path (text-only, no tools) ──
            if stream_fn and not tools:
                try:
                    return await _stream_responses(client, url, headers, body, stream_fn)
                except psdk.BadRequestError as stream_err:
                    if stream_err.status_code not in (400, 422):
                        raise
                    logger.info(
                        "openai_responses: stream rejected (%d) — non-streaming retry",
                        stream_err.status_code,
                    )
                    body.pop("stream", None)

            # ── Non-streaming path ──
            try:
                resp = await psdk.request_with_retry(
                    client, "POST", url,
                    headers=headers, json=body, max_retries=max_retries,
                )
                return _parse_responses(resp.json())
            except psdk.BadRequestError as exc:
                err_msg = str(exc).lower()
                if tools and ("tool" in err_msg or "function" in err_msg):
                    raise psdk.BadRequestError(
                        f"openai_responses: model '{model}' rejected native "
                        "tool calling. Pick a tool-tuned model (gpt-5, "
                        "gpt-5-codex, gpt-4o).",
                        status_code=exc.status_code,
                        request_id=exc.request_id,
                        response_body=exc.response_body,
                    ) from exc
                raise


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _repair_responses_input(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop orphan function_call_output items (no matching function_call).

    function_call → function_call_output linkage is by ``call_id``. We only
    enforce the orphan-output rule here; missing-output (function_call without
    a following output) is left alone because Responses tolerates it as long
    as the output arrives in the next turn (multi-turn tool flow).
    """
    seen_call_ids: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        itype = item.get("type")
        if itype == "function_call":
            cid = str(item.get("call_id", ""))
            if cid:
                seen_call_ids.add(cid)
            out.append(item)
        elif itype == "function_call_output":
            cid = str(item.get("call_id", ""))
            if not cid or cid not in seen_call_ids:
                logger.warning(
                    "openai_responses: dropping orphan function_call_output "
                    "(call_id=%r)",
                    cid,
                )
                continue
            out.append(item)
        else:
            out.append(item)
    return out


def _parse_responses(data: dict[str, Any]) -> dict[str, Any]:
    """Parse a Responses-API response into the router's standard shape."""
    output_items = data.get("output") or []
    content = ""
    tool_calls: list[dict[str, Any]] = []
    finish_reason = "stop"

    for item in output_items:
        itype = item.get("type")
        if itype == "message":
            # Responses API emits content as either a flat string or a list
            # of content blocks (typically [{type: "output_text", text: "..."}]).
            raw_content = item.get("content")
            if isinstance(raw_content, str):
                content += raw_content
            elif isinstance(raw_content, list):
                for block in raw_content:
                    if isinstance(block, dict) and "text" in block:
                        content += block.get("text") or ""
        elif itype == "function_call":
            args_raw = item.get("arguments") or ""
            try:
                args = json.loads(args_raw) if args_raw else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append(
                {
                    "id": str(item.get("call_id", "")),
                    "name": str(item.get("name", "")),
                    "arguments": args,
                }
            )
            finish_reason = "tool_calls"
        # 'reasoning' items are intentionally ignored — they're Responses-API's
        # equivalent of Anthropic thinking blocks. Future enhancement could
        # round-trip them via provider_extras.

    if "</think>" in content:
        content = content.split("</think>")[-1].strip()

    usage = data.get("usage") or {}
    return {
        "content": content,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "usage": {
            # Responses uses input_tokens/output_tokens; map to legacy keys
            # so downstream code (cost tracking, telemetry) keeps working.
            "prompt_tokens": int(
                usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
            ),
            "completion_tokens": int(
                usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
            ),
        },
    }


async def _stream_responses(
    client: Any,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    stream_fn: Callable,
) -> dict[str, Any]:
    """Streaming path — Responses SSE events. Text-only (no tool streaming)."""
    body = dict(body)
    body["stream"] = True

    full_content = ""
    finish_reason = "stop"
    prompt_tokens = 0
    completion_tokens = 0

    async with client.stream("POST", url, json=body, headers=headers) as resp:
        if resp.status_code >= 400:
            await resp.aread()
            psdk.raise_for_status(resp)

        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                if data_str == "[DONE]":
                    break
                continue
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            etype = event.get("type") or ""

            # Token deltas arrive on response.output_text.delta
            if etype == "response.output_text.delta":
                token = event.get("delta") or ""
                if token:
                    full_content += token
                    try:
                        result = stream_fn(token)
                        if asyncio.iscoroutine(result):
                            await result
                    except (TypeError, ValueError, OSError):
                        pass
            # Final response — pull usage + finish reason
            elif etype == "response.completed":
                resp_obj = event.get("response") or {}
                usage = resp_obj.get("usage") or {}
                prompt_tokens = int(
                    usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
                )
                completion_tokens = int(
                    usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
                )
                status = resp_obj.get("status") or ""
                if status == "incomplete":
                    finish_reason = "length"

    if "</think>" in full_content:
        full_content = full_content.split("</think>")[-1].strip()

    return {
        "content": full_content,
        "tool_calls": [],
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
