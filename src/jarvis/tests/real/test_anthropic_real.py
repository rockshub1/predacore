"""
Real integration tests — API-key path against console.anthropic.com.

Covers the contract-level concerns that mock-based tests cannot verify:

  1. Basic smoke — body builder produces fields Anthropic accepts
  2. System prompt + message round-trip
  3. Single-turn tool_use is emitted with a valid ``id``
  4. Multi-turn tool_use → tool_result round-trip with ``tool_use_id``
     linking (the exact contract that broke the production bug)
  5. Thinking block signatures survive across iterations — interleaved
     thinking contract
  6. ``compact-2026-01-12`` beta header is accepted (doesn't 400)
  7. ``output_config.effort: "max"`` is accepted on Opus 4.6
  8. ``effort: "max"`` is rejected on Haiku 4.5 (Opus-tier only)
  9. ``strict: true`` tool produces schema-compliant input

Every test hits live Anthropic. Gated behind ``pytest --real`` (root
conftest.py) and skipped if ``ANTHROPIC_API_KEY`` isn't set (local
conftest.py fixtures).
"""
from __future__ import annotations

from typing import Any

import pytest

pytestmark = [pytest.mark.real, pytest.mark.asyncio]


# ---------------------------------------------------------------------------
# 1-3. Smoke tests: body builder produces accepted fields
# ---------------------------------------------------------------------------


async def test_smoke_simple_message(haiku_anthropic_provider):
    """Bare-minimum: a single user message returns text. Validates auth,
    headers, beta flags, and body shape are all accepted by the server."""
    resp = await haiku_anthropic_provider.chat(
        messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
        max_tokens=16,
    )

    assert "content" in resp
    assert "PONG" in resp["content"]
    assert resp["finish_reason"] in ("end_turn", "stop", "max_tokens")
    assert resp["usage"]["prompt_tokens"] > 0
    assert resp["usage"]["completion_tokens"] > 0
    assert resp["model"].startswith("claude-haiku-4-5")


async def test_smoke_with_system_prompt(haiku_anthropic_provider):
    """System prompts go in the ``system`` array and are honored."""
    resp = await haiku_anthropic_provider.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a calculator. Reply with JUST the number, nothing else."
                ),
            },
            {"role": "user", "content": "2+2"},
        ],
        max_tokens=16,
    )
    assert "4" in resp["content"]


async def test_smoke_tool_use_single_turn(
    haiku_anthropic_provider, simple_read_tool: dict[str, Any]
):
    """Model emits a real tool_use block with a valid id + arguments."""
    resp = await haiku_anthropic_provider.chat(
        messages=[
            {
                "role": "user",
                "content": "Please read the file at /etc/hosts using the tool.",
            }
        ],
        tools=[simple_read_tool],
        max_tokens=512,
    )

    assert resp["finish_reason"] in ("tool_use", "tool_calls")
    assert len(resp["tool_calls"]) >= 1
    tc = resp["tool_calls"][0]
    assert tc["name"] == "read_file"
    assert tc["id"].startswith("toolu_"), f"tool_use_id missing: {tc}"
    assert "path" in tc["arguments"]

    # Content blocks round-trip: thinking + tool_use blocks should be present
    assert "content_blocks" in resp
    tool_use_blocks = [
        b for b in resp["content_blocks"] if b.get("type") == "tool_use"
    ]
    assert len(tool_use_blocks) >= 1
    assert tool_use_blocks[0]["id"] == tc["id"]


# ---------------------------------------------------------------------------
# 4. The bug: tool_use → tool_result round-trip with id linking
# ---------------------------------------------------------------------------


async def test_tool_use_id_round_trip_multi_turn(
    haiku_anthropic_provider, simple_read_tool: dict[str, Any]
):
    """The specific contract that broke in production.

    Iteration 1: user asks for a file read → model emits tool_use with id X.
    Iteration 2: we replay assistant content_blocks (with tool_use) and
    append a user turn containing a tool_result with tool_use_id=X.

    Anthropic hard-rejects if the ids don't match or the assistant content
    isn't replayed. If this test passes, the content_blocks round-trip is
    structurally correct.
    """
    # Iteration 1 — prompt the tool call
    first = await haiku_anthropic_provider.chat(
        messages=[
            {
                "role": "user",
                "content": "Use the read_file tool to read /tmp/nonexistent.txt.",
            }
        ],
        tools=[simple_read_tool],
        max_tokens=512,
    )
    assert first["finish_reason"] in ("tool_use", "tool_calls")
    assert first["tool_calls"], "iteration 1 must emit a tool call"
    tool_use_id = first["tool_calls"][0]["id"]

    # Iteration 2 — replay the assistant turn with its content_blocks,
    # then respond with a user turn carrying a tool_result block keyed
    # by the same tool_use_id.
    second = await haiku_anthropic_provider.chat(
        messages=[
            {
                "role": "user",
                "content": "Use the read_file tool to read /tmp/nonexistent.txt.",
            },
            {
                "role": "assistant",
                "content": first["content"],
                "content_blocks": first["content_blocks"],
            },
            {
                "role": "user",
                "content_blocks": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": "File not found: /tmp/nonexistent.txt",
                        "is_error": True,
                    }
                ],
                "content": "[Tool Result: file not found]",
            },
        ],
        tools=[simple_read_tool],
        max_tokens=256,
    )

    # If the round trip was malformed, Anthropic would have returned 400.
    # Reaching here means the contract is satisfied.
    assert second["finish_reason"] in (
        "end_turn", "stop", "max_tokens", "tool_use", "tool_calls",
    )
    # The model should acknowledge the error somehow in its reply.
    assert second["content"] or second["tool_calls"]


# ---------------------------------------------------------------------------
# 5. Thinking signatures preserved across iterations (interleaved thinking)
# ---------------------------------------------------------------------------


async def test_thinking_signatures_preserved_in_multi_turn_tool_use(
    anthropic_provider, simple_read_tool: dict[str, Any]
):
    """Opus 4.6 + adaptive thinking + tool use.

    Iteration 1: thinking block + tool_use (if the model decides to think)
    Iteration 2: we replay iteration 1's content_blocks verbatim, including
                 any thinking blocks with their signatures, plus a
                 tool_result. Anthropic hard-rejects if a thinking block
                 is missing its signature or is re-ordered. If it accepts,
                 the round trip is correct for interleaved thinking.
    """
    first = await anthropic_provider.chat(
        messages=[
            {
                "role": "user",
                "content": (
                    "Think about which file you want to read, then use the "
                    "read_file tool on /etc/hostname."
                ),
            }
        ],
        tools=[simple_read_tool],
        max_tokens=2048,
    )

    if not first["tool_calls"]:
        pytest.skip(
            "Model chose not to call a tool on iteration 1 — re-run the "
            "test or adjust the prompt. Not a contract failure."
        )

    tool_use_id = first["tool_calls"][0]["id"]

    # If thinking blocks are present, they must come back verbatim with
    # their signatures. If the model didn't think, the test still verifies
    # the tool_use path.
    thinking_blocks_in_first = [
        b for b in first["content_blocks"] if b.get("type") == "thinking"
    ]

    second = await anthropic_provider.chat(
        messages=[
            {
                "role": "user",
                "content": (
                    "Think about which file you want to read, then use the "
                    "read_file tool on /etc/hostname."
                ),
            },
            {
                "role": "assistant",
                "content": first["content"],
                "content_blocks": first["content_blocks"],
            },
            {
                "role": "user",
                "content_blocks": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": "jarvis-test-host",
                    }
                ],
                "content": "[Tool Result: jarvis-test-host]",
            },
        ],
        tools=[simple_read_tool],
        max_tokens=1024,
    )

    # Reaching here means Anthropic accepted the replay — thinking blocks
    # (if any) were valid with their signatures.
    assert second["content"] or second["tool_calls"]

    # If iteration 1 had thinking blocks, record that we successfully
    # replayed them (this is the specific scenario that broke in prod).
    if thinking_blocks_in_first:
        assert all(
            b.get("signature") for b in thinking_blocks_in_first
        ), "Iteration 1 thinking blocks must carry signatures"


# ---------------------------------------------------------------------------
# 6. compact-2026-01-12 beta is accepted
# ---------------------------------------------------------------------------


async def test_compact_beta_accepted(haiku_anthropic_provider):
    """The provider always sends ``anthropic-beta: compact-2026-01-12`` and
    ``context_management: {edits: [{type: "compact_20260112"}]}``. If the
    server 400s on the beta flag, this test surfaces it immediately."""
    resp = await haiku_anthropic_provider.chat(
        messages=[{"role": "user", "content": "Say 'ok'."}],
        max_tokens=16,
    )
    # Reaching here = the beta header + context_management body field were
    # accepted. No assertion beyond success.
    assert resp["content"]


# ---------------------------------------------------------------------------
# 7-8. effort parameter: accepted on Opus, rejected on Haiku
# ---------------------------------------------------------------------------


async def test_effort_max_accepted_on_opus_46(anthropic_provider):
    """``output_config: {effort: "max"}`` is Opus-tier only.

    We set ``reasoning_effort="max"`` on the provider, which makes the
    wire helper emit ``output_config: {effort: "max"}``. Anthropic should
    accept this on Opus 4.6 and return a normal response.
    """
    # Patch the provider config to use max effort
    anthropic_provider.config.reasoning_effort = "max"
    resp = await anthropic_provider.chat(
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=64,
    )
    assert "OK" in resp["content"]


async def test_effort_max_rejected_on_haiku_45(anthropic_api_key: str):
    """``output_config: {effort: "max"}`` is NOT supported on Haiku 4.5.

    The wire helper gates effort on ``is_opus or is_sonnet_46`` — so it
    should NOT send the field on Haiku. Verify Haiku still works by
    construction (no 400 from an unsupported field).
    """
    from jarvis.llm_providers.anthropic import AnthropicProvider
    from jarvis.llm_providers.base import ProviderConfig

    provider = AnthropicProvider(
        ProviderConfig(
            model="claude-haiku-4-5",
            api_key=anthropic_api_key,
            max_tokens=64,
            reasoning_effort="max",  # would be rejected if sent on Haiku
            extras={"provider": "anthropic"},
        )
    )
    resp = await provider.chat(
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=32,
    )
    # Haiku accepts the request because the wire helper correctly suppressed
    # output_config.effort for non-Opus/Sonnet-4.6 models.
    assert "OK" in resp["content"]


# ---------------------------------------------------------------------------
# 9. strict: true tool enforces schema compliance
# ---------------------------------------------------------------------------


async def test_strict_tool_accepted(
    haiku_anthropic_provider, strict_read_tool: dict[str, Any]
):
    """A tool definition with ``strict: true`` is accepted by Anthropic
    and the model's emitted input matches the declared schema.

    Note: strict mode validates the model's OUTPUT (its tool_use.input),
    not our tool definition. If Anthropic accepts the request with
    ``strict: true`` and the model's tool call is schema-compliant, the
    end-to-end contract is verified.
    """
    resp = await haiku_anthropic_provider.chat(
        messages=[
            {
                "role": "user",
                "content": "Use the read_file tool on /etc/hostname.",
            }
        ],
        tools=[strict_read_tool],
        max_tokens=256,
    )

    if not resp["tool_calls"]:
        pytest.skip("Model didn't call the tool this run — sampling variance")

    tc = resp["tool_calls"][0]
    assert tc["name"] == "read_file"
    # strict: true guarantees this is schema-compliant
    assert "path" in tc["arguments"]
    assert isinstance(tc["arguments"]["path"], str)
