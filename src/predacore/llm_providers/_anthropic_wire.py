"""
Shared Anthropic Messages API wire format helpers.

The in-house Anthropic provider hits ``/v1/messages`` directly via
``httpx``. This module owns the stateless wire logic:

  * system/user/assistant message extraction
  * ``content_blocks`` round-trip (thinking signatures + tool_use ids)
  * request body builder (adaptive thinking, effort, compaction, caching,
    strict tools)
  * response parser (returns the router's standard dict shape)

The provider owns auth / headers / endpoint concerns; this module has no state.
"""
from __future__ import annotations

from typing import Any

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


# ---------------------------------------------------------------------------
# Message extraction + normalization
# ---------------------------------------------------------------------------


def extract_system_text(messages: list[dict]) -> str:
    """Concatenate all user-provided system messages into a single string.

    Tolerates both plain-string content and list-of-blocks content (pulls
    the text from text blocks, ignores other types).
    """
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        parts.append(content)
    return "\n".join(p for p in parts if p)


def build_conv_messages(messages: list[dict]) -> list[dict[str, Any]]:
    """Extract user/assistant messages, preserving content_blocks when set.

    If a message carries ``content_blocks`` (assistant turns with
    thinking/tool_use, user turns with tool_result), those are used
    verbatim. Anthropic requires thinking blocks to be replayed with
    signatures and tool_use/tool_result blocks to be id-matched across
    iterations — flat strings break the round trip.
    """
    conv: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            continue
        blocks = msg.get("content_blocks")
        if isinstance(blocks, list) and blocks:
            conv.append({"role": msg["role"], "content": blocks})
        else:
            conv.append({"role": msg["role"], "content": msg.get("content", "")})

    # Anthropic requires strict alternation starting with user.
    if conv and conv[0]["role"] != "user":
        conv.insert(0, {"role": "user", "content": "(conversation continues)"})

    # Empty-content guard — Anthropic rejects empty text blocks with 400.
    for msg in conv:
        content = msg.get("content")
        if isinstance(content, str) and not content.strip():
            msg["content"] = "(empty)"
        elif isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and not block.get("text", "").strip()
                ):
                    block["text"] = "(empty)"

    return conv


def build_system_blocks(
    user_system_text: str,
    *,
    identity_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Build the ``system`` field with an optional identity prefix block.

    When ``identity_prefix`` is given, it is emitted as the first system
    block. The user-provided system prompt goes into a second block with
    a cache breakpoint. When ``identity_prefix`` is omitted, the user
    system prompt is the only block, still with the cache breakpoint.

    A breakpoint on the last system text block caches tools + system
    together because tools render at position 0, before system.
    """
    blocks: list[dict[str, Any]] = []
    if identity_prefix:
        blocks.append({"type": "text", "text": identity_prefix})

    if user_system_text.strip():
        blocks.append(
            {
                "type": "text",
                "text": user_system_text.strip(),
                "cache_control": {"type": "ephemeral"},
            }
        )
    elif identity_prefix:
        # Put the cache breakpoint on the identity block so tools still cache
        blocks[0]["cache_control"] = {"type": "ephemeral"}

    return blocks


# ---------------------------------------------------------------------------
# Request body builder (all Tier 1 upgrades included)
# ---------------------------------------------------------------------------


def build_request_body(
    *,
    model: str,
    max_tok: int,
    conv_messages: list[dict[str, Any]],
    system_blocks: list[dict[str, Any]],
    tools: list[dict] | None,
    temperature: float | None,
    reasoning_effort: str,
) -> dict[str, Any]:
    """Assemble the full Messages API request body.

    Includes every current best-practice lever:
      * adaptive thinking (Opus 4.6 / Sonnet 4.6) with interleaved thinking
        auto-enabled
      * ``output_config.effort`` for cost/quality tuning
      * server-side ``context_management.compact_20260112`` compaction
      * top-level ``cache_control`` for auto prefix caching
      * ``strict: true`` on tools that opt in
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": conv_messages,
        "max_tokens": max_tok,
    }
    if system_blocks:
        body["system"] = system_blocks

    # Extended thinking
    model_lower = model.lower()
    is_opus = "opus" in model_lower
    is_sonnet_46 = "sonnet" in model_lower and ("4-6" in model_lower or "4.6" in model_lower)
    is_sonnet_37 = "sonnet" in model_lower and "3-7" in model_lower
    # Opus 4.7+ rejects non-default temperature / top_p / top_k with HTTP 400.
    # Detect by version suffix; future opus-4-8 / opus-5-* will inherit the
    # same restriction per Anthropic's migration guide.
    is_opus_47_or_later = is_opus and (
        "opus-4-7" in model_lower or "opus-4.7" in model_lower
        or "opus-4-8" in model_lower or "opus-4.8" in model_lower
        or "opus-5" in model_lower
    )

    if is_opus_47_or_later:
        # Opus 4.7+: adaptive-only, no sampling params.
        body["thinking"] = {"type": "adaptive"}
        # Deliberately do NOT set temperature/top_p/top_k — those 400.
    elif is_opus or (is_sonnet_46 and reasoning_effort in ("high", "medium", "max")):
        body["thinking"] = {"type": "adaptive"}
        body["temperature"] = 1.0  # adaptive thinking requires temp=1.0
    elif is_sonnet_37 and reasoning_effort in ("high", "medium"):
        budget = 4000 if reasoning_effort == "medium" else 16000
        budget = min(budget, max(1024, max_tok - 1024))
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}
        body["max_tokens"] = max(max_tok, budget + 1024)
        body["temperature"] = 1.0
    elif temperature is not None:
        body["temperature"] = temperature

    # Effort parameter — GA, no beta header needed. Controls thinking
    # depth + tool-call consolidation + preamble verbosity. `max` is
    # Opus-tier only.
    effort_allowed = is_opus or is_sonnet_46
    if effort_allowed and reasoning_effort in ("low", "medium", "high", "max"):
        body["output_config"] = {"effort": reasoning_effort}

    # Server-side compaction — summarizes earlier turns when context
    # approaches the window limit. Requires the ``compact-2026-01-12``
    # beta, which each caller sets in its own headers.
    body["context_management"] = {
        "edits": [{"type": "compact_20260112"}],
    }

    # Top-level cache_control: auto-places a breakpoint at the end of
    # the tools + system prefix. Combined with the explicit breakpoint
    # on the last system block, we get cache hits on the prefix AND the
    # second-to-last user message for multi-turn conversations.
    body["cache_control"] = {"type": "ephemeral"}

    # Tools (native Anthropic format)
    if tools:
        anthropic_tools: list[dict[str, Any]] = []
        for t in tools:
            tool_def: dict[str, Any] = {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", {}),
            }
            # Strict tool use — guarantees Claude emits schema-compliant
            # input. Requires additionalProperties:false on the schema
            # root; gated on the tool definition opting in.
            if t.get("strict"):
                tool_def["strict"] = True
            anthropic_tools.append(tool_def)
        body["tools"] = anthropic_tools
        body["tool_choice"] = {"type": "auto"}

    return body


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def parse_response(data: dict) -> dict[str, Any]:
    """Convert Anthropic JSON response to the router's standard shape.

    Emits ``content_blocks`` — the raw block list in replay-ready form —
    so the caller can round-trip thinking (with signatures) and tool_use
    (with ids) through subsequent turns. Required for extended thinking
    + tool use to work on Opus 4.6 / Sonnet 4.6.
    """
    content_text = ""
    thinking_text = ""
    tool_calls: list[dict] = []
    content_blocks: list[dict] = []

    for block in data.get("content", []):
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            content_text += text
            content_blocks.append({"type": "text", "text": text})
        elif btype == "thinking":
            thinking = block.get("thinking", "")
            thinking_text += thinking
            content_blocks.append(
                {
                    "type": "thinking",
                    "thinking": thinking,
                    "signature": block.get("signature", ""),
                }
            )
        elif btype == "redacted_thinking":
            content_blocks.append(
                {
                    "type": "redacted_thinking",
                    "data": block.get("data", ""),
                }
            )
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": block.get("input", {}),
                }
            )
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                }
            )
        elif btype == "compaction":
            # Compaction blocks must be preserved in subsequent turns —
            # they replace the compacted history when replayed.
            content_blocks.append(block)

    usage = data.get("usage", {}) or {}
    return {
        "content": content_text,
        "content_blocks": content_blocks,
        "tool_calls": tool_calls,
        "finish_reason": data.get("stop_reason") or "stop",
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
        },
        "thinking": thinking_text,
        "model": data.get("model", ""),
    }
