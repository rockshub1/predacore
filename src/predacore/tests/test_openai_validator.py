"""
Unit tests for predacore.llm_providers.openai_validator.

OpenAI's Chat Completions API rejects requests with malformed tool flow:
  * assistant ``tool_calls`` not followed by matching ``role:"tool"``
  * orphan ``role:"tool"`` messages with unknown tool_call_id
  * empty ``tool_calls: []`` arrays on assistant turns

This validator runs after ``OpenAIProvider._serialize_messages_for_wire``
and before the request hits the network.
"""
from __future__ import annotations

import pytest

from predacore.llm_providers import openai_validator as ov
from predacore.llm_providers.message_validator import ToolFlowInvariantError


def _assistant_with_calls(*ids_and_names: tuple[str, str]) -> dict:
    """Build an assistant turn with the given tool_calls."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
            for cid, name in ids_and_names
        ],
    }


def _tool_msg(call_id: str, content: str = "ok") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


# ---------------------------------------------------------------------------
# validate_wire — reports violations, doesn't mutate
# ---------------------------------------------------------------------------


class TestValidateWire:
    def test_empty_message_list_is_valid(self):
        assert ov.validate_wire([]) == []

    def test_plain_messages_no_tool_flow(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert ov.validate_wire(messages) == []

    def test_valid_tool_call_and_result(self):
        messages = [
            {"role": "user", "content": "do x"},
            _assistant_with_calls(("call_1", "x")),
            _tool_msg("call_1"),
            {"role": "assistant", "content": "done"},
        ]
        assert ov.validate_wire(messages) == []

    def test_orphan_tool_call_no_following_tool_msg(self):
        messages = [
            _assistant_with_calls(("call_1", "x")),
            {"role": "user", "content": "next"},
        ]
        issues = ov.validate_wire(messages)
        assert any("missing tool_result" in i for i in issues)

    def test_partial_tool_results(self):
        messages = [
            _assistant_with_calls(("call_1", "a"), ("call_2", "b")),
            _tool_msg("call_1"),  # call_2 missing
            {"role": "user", "content": "next"},
        ]
        issues = ov.validate_wire(messages)
        assert any("call_2" in i for i in issues)

    def test_orphan_tool_message(self):
        messages = [
            {"role": "user", "content": "hi"},
            _tool_msg("call_xyz"),  # no preceding assistant tool_calls
        ]
        issues = ov.validate_wire(messages)
        assert any("orphan" in i.lower() for i in issues)

    def test_empty_tool_calls_array(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": []},
        ]
        issues = ov.validate_wire(messages)
        assert any("empty tool_calls" in i for i in issues)


# ---------------------------------------------------------------------------
# repair_wire — auto-fix invariant violations
# ---------------------------------------------------------------------------


class TestRepairWire:
    def test_valid_flow_unchanged(self):
        messages = [
            {"role": "user", "content": "do x"},
            _assistant_with_calls(("call_1", "x")),
            _tool_msg("call_1"),
        ]
        out = ov.repair_wire(messages)
        # Same structural shape (note shallow copies are made)
        assert len(out) == len(messages)
        assert ov.validate_wire(out) == []

    def test_injects_synthetic_tool_msg_for_orphan_call(self):
        messages = [
            _assistant_with_calls(("call_1", "x")),
            {"role": "user", "content": "next"},
        ]
        out = ov.repair_wire(messages)
        # Synthetic tool msg should land between assistant and user
        assert out[1]["role"] == "tool"
        assert out[1]["tool_call_id"] == "call_1"
        assert "synthetic" in out[1]["content"].lower()
        assert ov.validate_wire(out) == []

    def test_fills_partial_tool_results(self):
        messages = [
            _assistant_with_calls(("call_1", "a"), ("call_2", "b")),
            _tool_msg("call_1", "got a"),
            # call_2 result missing
            {"role": "assistant", "content": "summary"},
        ]
        out = ov.repair_wire(messages)
        # Should now have both tool messages
        tool_msgs = [m for m in out if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        ids = {m["tool_call_id"] for m in tool_msgs}
        assert ids == {"call_1", "call_2"}
        assert ov.validate_wire(out) == []

    def test_drops_orphan_tool_msg(self):
        messages = [
            {"role": "user", "content": "hi"},
            _tool_msg("call_xyz"),  # orphaned
            {"role": "assistant", "content": "hello"},
        ]
        out = ov.repair_wire(messages)
        # Orphan dropped
        tool_msgs = [m for m in out if m.get("role") == "tool"]
        assert len(tool_msgs) == 0
        assert ov.validate_wire(out) == []

    def test_strips_empty_tool_calls_array(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok", "tool_calls": []},
        ]
        out = ov.repair_wire(messages)
        # tool_calls key should be gone
        assert "tool_calls" not in out[1]
        assert ov.validate_wire(out) == []

    def test_repair_idempotent(self):
        broken = [
            _assistant_with_calls(("call_1", "x")),
            {"role": "user", "content": "next"},
        ]
        once = ov.repair_wire(broken)
        twice = ov.repair_wire(once)
        # Same structural content the second time
        assert len(once) == len(twice)
        for a, b in zip(once, twice):
            assert a.get("role") == b.get("role")
            assert a.get("content") == b.get("content")
            assert a.get("tool_call_id") == b.get("tool_call_id")

    def test_preserves_input_when_clean(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        out = ov.repair_wire(messages)
        # Same content (shallow copies acceptable)
        assert len(out) == 2
        assert out[0]["content"] == "hi"
        assert out[1]["content"] == "hello"


# ---------------------------------------------------------------------------
# Env toggle integration (PREDACORE_REPAIR_TOOL_FLOW)
# ---------------------------------------------------------------------------


class TestEnvToggle:
    def test_strict_mode_raises_on_orphan_call(self, monkeypatch):
        monkeypatch.setenv("PREDACORE_REPAIR_TOOL_FLOW", "strict")
        broken = [
            _assistant_with_calls(("call_1", "x")),
            {"role": "user", "content": "next"},
        ]
        with pytest.raises(ToolFlowInvariantError):
            ov.repair_wire(broken)

    def test_off_mode_skips_repair(self, monkeypatch):
        monkeypatch.setenv("PREDACORE_REPAIR_TOOL_FLOW", "off")
        broken = [
            _assistant_with_calls(("call_1", "x")),
            {"role": "user", "content": "next"},
        ]
        out = ov.repair_wire(broken)
        # Same list reference, no synthetic injection
        assert out is broken
        # Still invalid
        assert ov.validate_wire(out) != []
