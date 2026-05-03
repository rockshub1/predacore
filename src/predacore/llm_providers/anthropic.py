"""
Anthropic Provider — Claude Opus / Sonnet / Haiku via API-key HTTP.

Uses PredaCore's in-house PredaCore SDK (``predacore_sdk.py``) — zero
external vendor-SDK dependency. POSTs directly to
``https://api.anthropic.com/v1/messages`` with an API key from
``console.anthropic.com``.

PredaCore no longer carries any legacy OAuth transport. Anthropic access
always goes through API keys and the in-house PredaCore SDK.

All Anthropic wire-format logic (request body builder, response parser,
content_blocks round-trip) lives in ``_anthropic_wire.py``. This file
owns only auth, headers, and endpoint concerns.

Features:
  - Extended thinking (adaptive on Opus 4.6 / Sonnet 4.6; auto-interleaved)
  - Effort parameter (``output_config.effort``: low/medium/high/max)
  - Prompt caching (top-level ephemeral + system breakpoint)
  - Server-side context compaction (``compact-2026-01-12`` beta)
  - Content-blocks round-trip (thinking signatures + tool_use ids survive)
  - Native tool use only (v1.5.0+: text fallback removed; pick a different
    model if your Anthropic deployment doesn't support tool calls)
  - Retry with exponential backoff on 429/5xx/529 via PredaCore SDK
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from . import _anthropic_wire as wire
from . import predacore_sdk as psdk
from .base import LLMProvider
from .claude_models import DEFAULT_CLAUDE_MODEL, resolve_claude_model
from .message_validator import repair_tool_flow
from .types import Message, ToolResultRef

logger = logging.getLogger(__name__)

# Beta flags — compact-2026-01-12 enables server-side context compaction.
# Adaptive thinking auto-enables interleaved thinking on Opus 4.6, so no
# extra beta header needed for that.
_BETA_FLAGS = "compact-2026-01-12"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider — API-key auth via console.anthropic.com.

    Uses the PredaCore SDK for HTTP + retry + typed errors.
    """

    name = "anthropic"

    # ------------------------------------------------------------------
    # Tool-turn serialization (Phase A refactor — 2026-04-21).
    #
    # Anthropic's API expects all tool_result blocks in a SINGLE user turn
    # whose content is a list of content blocks. This diverges from the
    # OpenAI-shaped default (one ``role="tool"`` turn per result), so we
    # override ``append_tool_results_turn`` to bundle.
    #
    # ``append_assistant_turn`` uses the default — it copies
    # ``response.provider_extras`` which carries ``content_blocks`` (thinking
    # signatures + tool_use ids) from ``wire.parse_response()``. The wire
    # serializer then round-trips those blocks verbatim, satisfying
    # Anthropic's extended-thinking contract.
    # ------------------------------------------------------------------

    def append_tool_results_turn(
        self,
        messages: list[Message],
        results: list[ToolResultRef],
    ) -> None:
        if not results:
            return
        # Pre-build wire-ready content_blocks so the existing
        # _anthropic_wire.build_conv_messages path finds them under
        # provider_extras["content_blocks"] and uses them verbatim. The
        # abstract ``tool_results`` field stays populated for any future
        # caller that prefers the typed form.
        wire_blocks: list[dict[str, Any]] = []
        for r in results:
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": r.call_id,
                "content": r.result,
            }
            if r.is_error:
                block["is_error"] = True
            wire_blocks.append(block)

        summary_lines = [
            f"[{r.name}] {'ERROR: ' if r.is_error else ''}{r.result[:200]}"
            for r in results
        ]
        messages.append(
            Message(
                role="user",
                content="\n".join(summary_lines),
                tool_results=list(results),
                provider_extras={"content_blocks": wire_blocks},
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
        """Send a chat completion request to Anthropic via API key."""
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key.startswith("sk-ant-"):
            raise psdk.AuthenticationError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY env var or "
                "configure api_key in ~/.predacore/config.yaml. "
                "Get a key at https://console.anthropic.com."
            )

        model = resolve_claude_model(self.config.model, default=DEFAULT_CLAUDE_MODEL)
        max_tok = (
            max_tokens if max_tokens is not None else self.config.max_tokens or 8192
        )

        conv_messages = wire.build_conv_messages(messages)
        conv_messages = self.repair_messages(conv_messages)

        system_blocks = wire.build_system_blocks(wire.extract_system_text(messages))

        body = wire.build_request_body(
            model=model,
            max_tok=max_tok,
            conv_messages=conv_messages,
            system_blocks=system_blocks,
            tools=tools,
            temperature=temperature,
            reasoning_effort=self.config.reasoning_effort or "medium",
        )

        base_url = getattr(self.config, "base_url", "") or os.getenv("ANTHROPIC_BASE_URL", "")
        endpoint = (
            (base_url.rstrip("/") + "/v1/messages") if base_url else wire.ANTHROPIC_API_URL
        )
        headers = _build_headers(api_key)
        max_retries = int(os.getenv("PREDACORE_API_MAX_RETRIES", "3"))

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
                # v1.5.0: text-tool fallback removed. If a Claude model
                # rejects native tools, surface a clear error instead of
                # silently degrading. (In practice no current Claude model
                # rejects tools — this branch only fires on misconfigured
                # custom-deployments / wrong model id.)
                err_msg = str(exc).lower()
                if tools and ("tool" in err_msg or "function" in err_msg):
                    raise psdk.BadRequestError(
                        f"anthropic: model '{model}' rejected native tool use. "
                        "Pick a Claude model that supports tools "
                        "(claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5).",
                        status_code=exc.status_code,
                        request_id=exc.request_id,
                        response_body=exc.response_body,
                    ) from exc
                raise
            except psdk.LLMError as exc:
                logger.error("anthropic error: %s", exc)
                raise

    # ------------------------------------------------------------------
    # repair_messages — Anthropic's wire validator (v1.5.0).
    #
    # Runs ``message_validator.repair_tool_flow`` only when the input has
    # structured (block-list) content. Flat-string conversations skip the
    # check (they have no tool_use blocks to validate).
    # ------------------------------------------------------------------

    def repair_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if any(isinstance(m.get("content"), list) for m in messages):
            return repair_tool_flow(messages)
        return messages


def _build_headers(api_key: str) -> dict[str, str]:
    """API-key auth headers for a Messages API call."""
    return {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": wire.ANTHROPIC_VERSION,
        "anthropic-beta": _BETA_FLAGS,
    }
