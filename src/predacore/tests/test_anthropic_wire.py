"""
Unit tests for predacore.llm_providers._anthropic_wire — shared Anthropic
wire format helpers used by the in-house Anthropic provider.

Tests are pure-Python (no mocks, no network) because the wire helpers
are stateless functions. They cover:

  * ``extract_system_text`` — concatenates system messages, tolerates lists
  * ``build_conv_messages`` — preserves content_blocks, strips system,
    enforces user-first alternation, empty-content guard
  * ``build_system_blocks`` — identity prefix (OAuth) vs none (API key),
    cache breakpoint placement
  * ``build_request_body`` — all Tier 1 upgrades in one place:
    adaptive thinking, effort parameter, compact-2026-01-12, cache_control,
    strict tools
  * ``parse_response`` — content_blocks with thinking signatures,
    tool_use ids, compaction blocks, cache metrics

These tests would have caught the content_blocks round-trip bug on day
one — every regression path is exercised here.
"""
from __future__ import annotations

from typing import Any

from predacore.llm_providers import _anthropic_wire as wire

# ---------------------------------------------------------------------------
# extract_system_text
# ---------------------------------------------------------------------------


class TestExtractSystemText:
    """extract_system_text — concatenates system messages."""

    def test_empty_messages_returns_empty(self):
        assert wire.extract_system_text([]) == ""

    def test_no_system_returns_empty(self):
        messages = [{"role": "user", "content": "hi"}]
        assert wire.extract_system_text(messages) == ""

    def test_single_system(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        assert wire.extract_system_text(messages) == "You are helpful."

    def test_multiple_systems_concatenated(self):
        messages = [
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
            {"role": "user", "content": "hi"},
        ]
        result = wire.extract_system_text(messages)
        assert "Rule 1" in result
        assert "Rule 2" in result

    def test_system_as_list_of_blocks(self):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "part 1"},
                    {"type": "text", "text": " part 2"},
                ],
            },
            {"role": "user", "content": "hi"},
        ]
        assert wire.extract_system_text(messages) == "part 1 part 2"

    def test_system_list_skips_non_text_blocks(self):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "keep"},
                    {"type": "image", "source": {}},  # ignored
                ],
            }
        ]
        assert wire.extract_system_text(messages) == "keep"


# ---------------------------------------------------------------------------
# build_conv_messages
# ---------------------------------------------------------------------------


class TestBuildConvMessages:
    """build_conv_messages — extracts user/assistant with content_blocks preservation."""

    def test_strips_system_messages(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        result = wire.build_conv_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_preserves_content_blocks_when_present(self):
        blocks = [
            {"type": "text", "text": "hi"},
            {
                "type": "thinking",
                "thinking": "considering",
                "signature": "sig_abc",
            },
            {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "/x"}},
        ]
        messages = [
            {"role": "user", "content": "read /x"},
            {
                "role": "assistant",
                "content": "hi",
                "content_blocks": blocks,
            },
        ]
        result = wire.build_conv_messages(messages)
        assert result[1]["content"] == blocks  # full blocks preserved

    def test_falls_back_to_flat_content_when_blocks_absent(self):
        messages = [{"role": "user", "content": "plain text"}]
        result = wire.build_conv_messages(messages)
        assert result[0]["content"] == "plain text"

    def test_empty_content_blocks_list_falls_back(self):
        messages = [
            {"role": "user", "content": "fallback", "content_blocks": []},
        ]
        result = wire.build_conv_messages(messages)
        assert result[0]["content"] == "fallback"

    def test_inserts_user_turn_if_assistant_first(self):
        """Anthropic requires first message to be user."""
        messages = [
            {"role": "assistant", "content": "continuing..."},
            {"role": "user", "content": "hi"},
        ]
        result = wire.build_conv_messages(messages)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "(conversation continues)"
        assert result[1]["role"] == "assistant"

    def test_empty_string_content_replaced(self):
        messages = [{"role": "user", "content": ""}]
        result = wire.build_conv_messages(messages)
        assert result[0]["content"] == "(empty)"

    def test_empty_text_block_in_list_replaced(self):
        messages = [
            {
                "role": "user",
                "content_blocks": [
                    {"type": "text", "text": ""},
                    {"type": "text", "text": "real"},
                ],
            }
        ]
        result = wire.build_conv_messages(messages)
        blocks = result[0]["content"]
        assert blocks[0]["text"] == "(empty)"
        assert blocks[1]["text"] == "real"

    def test_tool_use_round_trip_preserved(self):
        """The critical bug path: tool_use blocks with ids must survive."""
        messages = [
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": "sure",
                "content_blocks": [
                    {"type": "text", "text": "sure"},
                    {
                        "type": "tool_use",
                        "id": "toolu_xyz",
                        "name": "read_file",
                        "input": {"path": "/tmp/x"},
                    },
                ],
            },
            {
                "role": "user",
                "content": "[Tool Result]",
                "content_blocks": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_xyz",
                        "content": "file contents",
                    }
                ],
            },
        ]
        result = wire.build_conv_messages(messages)

        # Assistant turn has the tool_use block with its id
        assistant_blocks = result[1]["content"]
        tool_use = next(b for b in assistant_blocks if b["type"] == "tool_use")
        assert tool_use["id"] == "toolu_xyz"

        # Following user turn has the tool_result with matching id
        user_blocks = result[2]["content"]
        tool_result = next(b for b in user_blocks if b["type"] == "tool_result")
        assert tool_result["tool_use_id"] == "toolu_xyz"


# ---------------------------------------------------------------------------
# build_system_blocks
# ---------------------------------------------------------------------------


class TestBuildSystemBlocks:
    """build_system_blocks — identity prefix, cache breakpoint placement."""

    def test_empty_text_no_identity_returns_empty_list(self):
        assert wire.build_system_blocks("") == []

    def test_user_system_only_no_identity(self):
        result = wire.build_system_blocks("You are helpful")
        assert len(result) == 1
        assert result[0]["text"] == "You are helpful"
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_identity_prefix_without_user_system(self):
        """Prefix-only path — the first system block gets the cache breakpoint."""
        result = wire.build_system_blocks(
            "", identity_prefix="Pinned system prefix"
        )
        assert len(result) == 1
        assert result[0]["text"] == "Pinned system prefix"
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_identity_prefix_with_user_system(self):
        """Prefix + user system — prefix first, user system second with cache."""
        result = wire.build_system_blocks(
            "You are PredaCore",
            identity_prefix="Pinned system prefix",
        )
        assert len(result) == 2
        assert result[0]["text"] == "Pinned system prefix"
        assert "cache_control" not in result[0]  # identity has no breakpoint
        assert result[1]["text"] == "You are PredaCore"
        assert result[1]["cache_control"] == {"type": "ephemeral"}

    def test_whitespace_only_user_system_treated_as_empty(self):
        result = wire.build_system_blocks("   \n  \t  ")
        assert result == []


# ---------------------------------------------------------------------------
# build_request_body — the Tier 1 upgrade integration test
# ---------------------------------------------------------------------------


class TestBuildRequestBody:
    """build_request_body — all upgrades in one place."""

    def _call(self, **overrides) -> dict[str, Any]:
        defaults = {
            "model": "claude-opus-4-6",
            "max_tok": 8192,
            "conv_messages": [{"role": "user", "content": "hi"}],
            "system_blocks": [{"type": "text", "text": "sys"}],
            "tools": None,
            "temperature": None,
            "reasoning_effort": "medium",
        }
        defaults.update(overrides)
        return wire.build_request_body(**defaults)

    def test_core_fields_present(self):
        body = self._call()
        assert body["model"] == "claude-opus-4-6"
        assert body["max_tokens"] == 8192
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["system"] == [{"type": "text", "text": "sys"}]

    def test_no_system_blocks_omits_system(self):
        body = self._call(system_blocks=[])
        assert "system" not in body

    # ── adaptive thinking ─────────────────────────────────────────

    def test_opus_46_gets_adaptive_thinking(self):
        body = self._call(model="claude-opus-4-6")
        assert body["thinking"] == {"type": "adaptive"}
        assert body["temperature"] == 1.0

    def test_sonnet_46_gets_adaptive_thinking_on_medium(self):
        body = self._call(model="claude-sonnet-4-6", reasoning_effort="medium")
        assert body["thinking"] == {"type": "adaptive"}
        assert body["temperature"] == 1.0

    def test_sonnet_46_gets_adaptive_thinking_on_max(self):
        body = self._call(model="claude-sonnet-4-6", reasoning_effort="max")
        assert body["thinking"] == {"type": "adaptive"}

    def test_sonnet_37_gets_explicit_thinking_budget(self):
        body = self._call(
            model="claude-sonnet-3-7", reasoning_effort="medium", max_tok=16000
        )
        assert body["thinking"]["type"] == "enabled"
        assert body["thinking"]["budget_tokens"] == 4000
        assert body["temperature"] == 1.0

    def test_sonnet_37_high_gets_larger_budget(self):
        body = self._call(
            model="claude-sonnet-3-7", reasoning_effort="high", max_tok=32000
        )
        assert body["thinking"]["budget_tokens"] == 16000

    def test_haiku_no_thinking(self):
        body = self._call(model="claude-haiku-4-5", reasoning_effort="medium")
        assert "thinking" not in body

    def test_haiku_respects_explicit_temperature(self):
        body = self._call(
            model="claude-haiku-4-5", temperature=0.3, reasoning_effort="low"
        )
        assert body["temperature"] == 0.3

    # ── effort parameter ──────────────────────────────────────────

    def test_opus_effort_max(self):
        body = self._call(model="claude-opus-4-6", reasoning_effort="max")
        assert body["output_config"] == {"effort": "max"}

    def test_opus_effort_medium(self):
        body = self._call(model="claude-opus-4-6", reasoning_effort="medium")
        assert body["output_config"] == {"effort": "medium"}

    def test_sonnet_46_effort_high(self):
        body = self._call(model="claude-sonnet-4-6", reasoning_effort="high")
        assert body["output_config"] == {"effort": "high"}

    def test_haiku_omits_output_config(self):
        """Haiku doesn't support the effort parameter — must NOT be sent."""
        body = self._call(model="claude-haiku-4-5", reasoning_effort="medium")
        assert "output_config" not in body

    def test_sonnet_45_omits_output_config(self):
        body = self._call(model="claude-sonnet-4-5", reasoning_effort="max")
        assert "output_config" not in body

    # ── compaction ────────────────────────────────────────────────

    def test_compaction_field_always_present(self):
        body = self._call()
        assert body["context_management"] == {
            "edits": [{"type": "compact_20260112"}]
        }

    def test_compaction_present_on_haiku_too(self):
        body = self._call(model="claude-haiku-4-5")
        assert "context_management" in body

    # ── cache_control ─────────────────────────────────────────────

    def test_top_level_cache_control_always_present(self):
        body = self._call()
        assert body["cache_control"] == {"type": "ephemeral"}

    # ── tools ─────────────────────────────────────────────────────

    def test_no_tools_omits_tool_fields(self):
        body = self._call(tools=None)
        assert "tools" not in body
        assert "tool_choice" not in body

    def test_tools_converted_to_anthropic_format(self):
        tools = [
            {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }
        ]
        body = self._call(tools=tools)
        assert body["tools"][0]["name"] == "read_file"
        assert body["tools"][0]["description"] == "Read a file"
        assert body["tools"][0]["input_schema"]["properties"]["path"]["type"] == "string"
        assert body["tool_choice"] == {"type": "auto"}

    def test_strict_tool_opt_in(self):
        tools = [
            {
                "name": "t1",
                "description": "strict tool",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            },
            {
                "name": "t2",
                "description": "lenient tool",
                "parameters": {"type": "object", "properties": {}},
            },
        ]
        body = self._call(tools=tools)
        assert body["tools"][0].get("strict") is True
        assert "strict" not in body["tools"][1]


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    """parse_response — content_blocks round-trip, thinking, tool_use."""

    def test_text_only_response(self):
        data = {
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "claude-opus-4-6",
        }
        result = wire.parse_response(data)
        assert result["content"] == "hello"
        assert result["content_blocks"] == [{"type": "text", "text": "hello"}]
        assert result["tool_calls"] == []
        assert result["finish_reason"] == "end_turn"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5

    def test_thinking_block_preserves_signature(self):
        """The critical bug path — thinking blocks must carry signatures."""
        data = {
            "content": [
                {
                    "type": "thinking",
                    "thinking": "let me think",
                    "signature": "sig_xyz123",
                },
                {"type": "text", "text": "result"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 20},
            "model": "claude-opus-4-6",
        }
        result = wire.parse_response(data)
        assert result["content"] == "result"
        assert result["thinking"] == "let me think"

        thinking_block = next(
            b for b in result["content_blocks"] if b["type"] == "thinking"
        )
        assert thinking_block["signature"] == "sig_xyz123"

    def test_thinking_block_missing_signature_becomes_empty_string(self):
        data = {
            "content": [{"type": "thinking", "thinking": "no sig"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "model": "claude-opus-4-6",
        }
        result = wire.parse_response(data)
        thinking = result["content_blocks"][0]
        assert thinking["signature"] == ""

    def test_tool_use_preserves_id(self):
        """The critical bug path — tool_use blocks must carry ids."""
        data = {
            "content": [
                {"type": "text", "text": "calling"},
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "read_file",
                    "input": {"path": "/tmp/x"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "model": "claude-opus-4-6",
        }
        result = wire.parse_response(data)
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["id"] == "toolu_abc"
        assert tc["name"] == "read_file"
        assert tc["arguments"] == {"path": "/tmp/x"}

        # Raw content_blocks also carries the id
        tool_use = next(
            b for b in result["content_blocks"] if b["type"] == "tool_use"
        )
        assert tool_use["id"] == "toolu_abc"

    def test_redacted_thinking_block_preserved(self):
        data = {
            "content": [
                {"type": "redacted_thinking", "data": "encrypted_blob_xyz"},
                {"type": "text", "text": "result"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "model": "claude-opus-4-6",
        }
        result = wire.parse_response(data)
        redacted = result["content_blocks"][0]
        assert redacted["type"] == "redacted_thinking"
        assert redacted["data"] == "encrypted_blob_xyz"

    def test_compaction_block_preserved(self):
        data = {
            "content": [
                {"type": "compaction", "summary": "earlier turns compacted"},
                {"type": "text", "text": "new turn"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 10},
            "model": "claude-opus-4-6",
        }
        result = wire.parse_response(data)
        compaction = result["content_blocks"][0]
        assert compaction["type"] == "compaction"
        assert compaction["summary"] == "earlier turns compacted"

    def test_cache_metrics_exposed(self):
        data = {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 5000,
                "cache_creation_input_tokens": 200,
            },
            "model": "claude-opus-4-6",
        }
        result = wire.parse_response(data)
        assert result["usage"]["cache_read_input_tokens"] == 5000
        assert result["usage"]["cache_creation_input_tokens"] == 200

    def test_multiple_tool_calls(self):
        data = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "read_file",
                    "input": {"path": "/a"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_2",
                    "name": "read_file",
                    "input": {"path": "/b"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 50, "output_tokens": 20},
            "model": "claude-opus-4-6",
        }
        result = wire.parse_response(data)
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["id"] == "toolu_1"
        assert result["tool_calls"][1]["id"] == "toolu_2"

    def test_empty_response(self):
        data = {
            "content": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 0},
            "model": "claude-opus-4-6",
        }
        result = wire.parse_response(data)
        assert result["content"] == ""
        assert result["content_blocks"] == []
        assert result["tool_calls"] == []

    def test_missing_usage_defaults_to_zero(self):
        data = {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "model": "claude-opus-4-6",
        }
        result = wire.parse_response(data)
        assert result["usage"]["prompt_tokens"] == 0
        assert result["usage"]["completion_tokens"] == 0

    def test_missing_stop_reason_defaults_to_stop(self):
        data = {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "model": "claude-opus-4-6",
        }
        result = wire.parse_response(data)
        assert result["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# Full round-trip integration
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Build body, parse response, feed it back — verify the full cycle."""

    def test_response_parse_builds_replayable_history(self):
        """Simulate the exact multi-turn round trip that broke in production:
        iteration 1 returns content_blocks → iteration 2 replays them.
        """
        # Iteration 1: model responds with thinking + tool_use
        fake_iter1_data = {
            "content": [
                {
                    "type": "thinking",
                    "thinking": "I should read the file",
                    "signature": "sig_12345",
                },
                {"type": "text", "text": "Let me check that."},
                {
                    "type": "tool_use",
                    "id": "toolu_file_read",
                    "name": "read_file",
                    "input": {"path": "/etc/hosts"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "model": "claude-opus-4-6",
        }
        iter1 = wire.parse_response(fake_iter1_data)

        # Core.py builds iteration 2 messages list, replaying iter1's
        # content_blocks verbatim and appending a tool_result turn:
        iter2_messages = [
            {"role": "user", "content": "read /etc/hosts"},
            {
                "role": "assistant",
                "content": iter1["content"],
                "content_blocks": iter1["content_blocks"],
            },
            {
                "role": "user",
                "content": "[Tool Result]",
                "content_blocks": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_file_read",
                        "content": "127.0.0.1 localhost",
                    }
                ],
            },
        ]

        # build_conv_messages must preserve every structured block:
        conv = wire.build_conv_messages(iter2_messages)
        assert len(conv) == 3

        # Assistant turn — thinking with signature + tool_use with id
        assistant_blocks = conv[1]["content"]
        thinking = next(b for b in assistant_blocks if b["type"] == "thinking")
        assert thinking["signature"] == "sig_12345"
        tool_use = next(b for b in assistant_blocks if b["type"] == "tool_use")
        assert tool_use["id"] == "toolu_file_read"

        # Following user turn — tool_result keyed to same id
        user_blocks = conv[2]["content"]
        tool_result = next(b for b in user_blocks if b["type"] == "tool_result")
        assert tool_result["tool_use_id"] == "toolu_file_read"

        # Body builder accepts the conv_messages and emits the compaction field
        body = wire.build_request_body(
            model="claude-opus-4-6",
            max_tok=8192,
            conv_messages=conv,
            system_blocks=[],
            tools=None,
            temperature=None,
            reasoning_effort="medium",
        )
        assert body["messages"] == conv
        assert body["context_management"] == {
            "edits": [{"type": "compact_20260112"}]
        }
        assert body["thinking"] == {"type": "adaptive"}
