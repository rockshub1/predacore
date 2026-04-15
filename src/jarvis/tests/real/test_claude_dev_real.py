"""
Real integration tests — OAuth path via Claude Code keychain.

These tests spoof Claude Code identity (``x-app: cli``, Claude Code
User-Agent, Claude Code identity system prompt, OAuth bearer from the
macOS Keychain) and hit Anthropic's production API. They verify the
OAuth enforcement path works end-to-end, which cannot be tested any
other way — Anthropic's server validates every piece of the OAuth
contract server-side.

Skipped unless:
  1. ``pytest --real`` is passed (top-level gate)
  2. Either ``CLAUDE_CODE_OAUTH_TOKEN`` env var is set OR macOS Keychain
     has a ``Claude Code-credentials`` entry (per-test gate)

Never run in CI — requires macOS Keychain access or an OAuth token that
belongs to a human's personal Max subscription. DEV-machine-only by
design; matches the "DEV ONLY, NOT FOR PUBLIC RELEASE" warning on
``claude_dev.py``.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.real, pytest.mark.asyncio]


async def test_oauth_smoke_simple_message(claude_dev_provider):
    """Bare OAuth round trip — auth, headers, beta flags, body shape all
    accepted by Anthropic's server-side OAuth enforcement.

    If this fails with a 429 or auth error, the Claude Code identity
    spoof isn't being accepted and ``_BETA_FLAGS``/``CLAUDE_CODE_IDENTITY``
    /``CLAUDE_CODE_USER_AGENT`` need to be bumped to match a current
    Claude Code release.
    """
    resp = await claude_dev_provider.chat(
        messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
        max_tokens=16,
    )
    assert "PONG" in resp["content"]
    assert resp["finish_reason"] in ("end_turn", "stop", "max_tokens")
    assert resp["usage"]["prompt_tokens"] > 0
    assert resp["usage"]["completion_tokens"] > 0


async def test_oauth_tool_call_with_content_blocks(
    claude_dev_provider, simple_read_tool
):
    """OAuth path emits tool_use blocks with ids, same as API-key path.

    If this passes, the shared ``_anthropic_wire`` helpers are working
    identically on both auth paths. Any divergence would surface here.
    """
    resp = await claude_dev_provider.chat(
        messages=[
            {
                "role": "user",
                "content": (
                    "Use the read_file tool to read /etc/hostname. "
                    "Do not describe, actually call the tool."
                ),
            }
        ],
        tools=[simple_read_tool],
        max_tokens=2048,
    )

    assert resp["finish_reason"] in ("tool_use", "tool_calls")
    assert resp["tool_calls"], "OAuth path should emit a real tool_use block"
    tc = resp["tool_calls"][0]
    assert tc["id"].startswith("toolu_")
    assert tc["name"] == "read_file"

    # content_blocks round-trip is the whole point
    assert "content_blocks" in resp
    tool_use_blocks = [
        b for b in resp["content_blocks"] if b.get("type") == "tool_use"
    ]
    assert len(tool_use_blocks) >= 1
    assert tool_use_blocks[0]["id"] == tc["id"]


async def test_oauth_interleaved_thinking_multi_turn(
    claude_dev_provider, simple_read_tool
):
    """OAuth path with adaptive thinking + tool use across two iterations.

    This is the exact pattern that broke JARVIS on Telegram. If it passes
    here, the OAuth path fully respects Anthropic's interleaved-thinking
    contract. If it fails with a 400 mentioning "thinking" or "tool_use",
    the content_blocks round-trip has regressed.
    """
    # Iteration 1 — prompt the model to think + call the tool
    first = await claude_dev_provider.chat(
        messages=[
            {
                "role": "user",
                "content": (
                    "Think about which file to read, then call read_file on "
                    "/etc/hostname."
                ),
            }
        ],
        tools=[simple_read_tool],
        max_tokens=2048,
    )
    if not first["tool_calls"]:
        pytest.skip("Model didn't call the tool this run — sampling variance")

    tool_use_id = first["tool_calls"][0]["id"]

    # Iteration 2 — replay iteration 1's content_blocks (with thinking
    # signatures if present) and append a tool_result turn.
    second = await claude_dev_provider.chat(
        messages=[
            {
                "role": "user",
                "content": (
                    "Think about which file to read, then call read_file on "
                    "/etc/hostname."
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
                        "content": "jarvis-dev-host",
                    }
                ],
                "content": "[Tool Result: jarvis-dev-host]",
            },
        ],
        tools=[simple_read_tool],
        max_tokens=1024,
    )

    # Reaching here = Anthropic's server accepted the replayed content_blocks.
    # The thinking signatures (if any) and tool_use_id linking all validated.
    assert second["content"] or second["tool_calls"]
