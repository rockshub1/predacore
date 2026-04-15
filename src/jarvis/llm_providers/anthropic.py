"""
Anthropic Provider — Claude Opus / Sonnet / Haiku via API-key HTTP.

Uses JARVIS's in-house Prometheus SDK (``prometheus_sdk.py``) — zero
external vendor-SDK dependency. POSTs directly to
``https://api.anthropic.com/v1/messages`` with an API key from
``console.anthropic.com``.

Per Anthropic's April 2026 policy, OAuth tokens from Free/Pro/Max
subscriptions are only valid inside Claude Code and other native
Anthropic applications. Developers building agents must use API keys
from https://console.anthropic.com. For the OAuth path, see ``claude_dev.py``.

All Anthropic wire-format logic (request body builder, response parser,
content_blocks round-trip) lives in ``_anthropic_wire.py`` and is shared
with ``claude_dev.py``. This file owns only auth, headers, and endpoint
concerns.

Features:
  - Extended thinking (adaptive on Opus 4.6 / Sonnet 4.6; auto-interleaved)
  - Effort parameter (``output_config.effort``: low/medium/high/max)
  - Prompt caching (top-level ephemeral + system breakpoint)
  - Server-side context compaction (``compact-2026-01-12`` beta)
  - Content-blocks round-trip (thinking signatures + tool_use ids survive)
  - Native tool use with text-based fallback
  - Retry with exponential backoff on 429/5xx/529 via Prometheus SDK
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

from . import _anthropic_wire as wire
from . import prometheus_sdk as psdk
from .base import LLMProvider
from .claude_models import DEFAULT_CLAUDE_MODEL, resolve_claude_model
from .message_validator import repair_tool_flow
from .text_tool_adapter import build_tool_prompt, parse_tool_calls

logger = logging.getLogger(__name__)

# Beta flags — compact-2026-01-12 enables server-side context compaction.
# Adaptive thinking auto-enables interleaved thinking on Opus 4.6, so no
# extra beta header needed for that.
_BETA_FLAGS = "compact-2026-01-12"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider — API-key auth via console.anthropic.com.

    Uses the Prometheus SDK for HTTP + retry + typed errors, and shares
    wire-format helpers with ``claude_dev.py`` via ``_anthropic_wire``.
    """

    name = "anthropic"

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request to Anthropic via API key."""
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key.startswith("sk-ant-"):
            raise psdk.AuthenticationError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY env var or "
                "configure api_key in ~/.prometheus/config.yaml. "
                "Get a key at https://console.anthropic.com."
            )

        model = resolve_claude_model(self.config.model, default=DEFAULT_CLAUDE_MODEL)
        max_tok = (
            max_tokens if max_tokens is not None else self.config.max_tokens or 8192
        )

        conv_messages = wire.build_conv_messages(messages)
        if any(isinstance(m.get("content"), list) for m in conv_messages):
            conv_messages = repair_tool_flow(conv_messages)

        system_blocks = wire.build_system_blocks(
            wire.extract_system_text(messages),
            identity_prefix=None,  # API-key path — no Claude Code identity needed
        )

        body = wire.build_request_body(
            model=model,
            max_tok=max_tok,
            conv_messages=conv_messages,
            system_blocks=system_blocks,
            tools=tools,
            temperature=temperature,
            reasoning_effort=self.config.reasoning_effort or "medium",
        )

        base_url = self.config.api_base or os.getenv("ANTHROPIC_BASE_URL", "")
        endpoint = (
            (base_url.rstrip("/") + "/v1/messages") if base_url else wire.ANTHROPIC_API_URL
        )
        headers = _build_headers(api_key)
        max_retries = int(os.getenv("JARVIS_API_MAX_RETRIES", "3"))

        async with psdk.make_client() as client:
            try:
                resp = await psdk.request_with_retry(
                    client,
                    "POST",
                    endpoint,
                    headers=headers,
                    json=body,
                    max_retries=max_retries,
                )
                return wire.parse_response(resp.json())
            except psdk.BadRequestError as exc:
                err_msg = str(exc).lower()
                is_tool_error = tools and ("tool" in err_msg or "function" in err_msg)
                if is_tool_error:
                    return await self._fallback_to_text_tools(
                        client, endpoint, headers, body, tools, max_retries,
                    )
                raise
            except psdk.LLMError as exc:
                logger.error("anthropic error: %s", exc)
                raise

    async def _fallback_to_text_tools(
        self,
        client: Any,
        endpoint: str,
        headers: dict[str, str],
        body: dict,
        tools: list[dict],
        max_retries: int,
    ) -> dict[str, Any]:
        """Native tool API rejected — retry with text-based tools in the prompt."""
        logger.warning("anthropic: native tools rejected — falling back to text tools")

        body = dict(body)
        body.pop("tools", None)
        body.pop("tool_choice", None)

        tool_prompt = build_tool_prompt(tools)
        system = body.get("system", [])
        if isinstance(system, list) and system:
            last = system[-1]
            if isinstance(last, dict) and last.get("type") == "text":
                last["text"] = last.get("text", "") + tool_prompt
            else:
                system.append({"type": "text", "text": tool_prompt})
        else:
            body["system"] = [{"type": "text", "text": tool_prompt}]

        resp = await psdk.request_with_retry(
            client, "POST", endpoint, headers=headers, json=body, max_retries=max_retries,
        )
        result = wire.parse_response(resp.json())
        clean_text, text_tool_calls = parse_tool_calls(result["content"])
        if text_tool_calls:
            result["content"] = clean_text
            result["tool_calls"] = text_tool_calls
            result["finish_reason"] = "tool_calls"
        return result


def _build_headers(api_key: str) -> dict[str, str]:
    """API-key auth headers for a Messages API call."""
    return {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": wire.ANTHROPIC_VERSION,
        "anthropic-beta": _BETA_FLAGS,
    }
