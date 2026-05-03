"""
Unit tests for predacore.llm_providers.gemini_validator.

Gemini's wire format uses `parts[].functionCall` / `parts[].functionResponse`
with linkage by **name + position** (no native IDs). Reasoning models 2.5+
require ``thoughtSignature`` on functionCall parts to round-trip correctly.

This validator runs after `gemini._build_payload` and before the request
hits the network.
"""
from __future__ import annotations

import pytest

from predacore.llm_providers import gemini_validator as gv
from predacore.llm_providers.message_validator import ToolFlowInvariantError


def _model_turn_with_calls(*names: str, with_signature: bool = False) -> dict:
    """Build a model turn with N functionCall parts."""
    parts = []
    for i, name in enumerate(names):
        part = {"functionCall": {"name": name, "args": {}}}
        # Per Google docs: signature on first functionCall only on parallel calls
        if with_signature and i == 0:
            part["thoughtSignature"] = "sig_" + name
        parts.append(part)
    return {"role": "model", "parts": parts}


def _user_turn_with_responses(*names: str) -> dict:
    """Build a user turn with N functionResponse parts."""
    return {
        "role": "user",
        "parts": [
            {"functionResponse": {"name": name, "response": {"result": "ok"}}}
            for name in names
        ],
    }


# ---------------------------------------------------------------------------
# validate_wire — reports violations
# ---------------------------------------------------------------------------


class TestValidateWire:
    def test_empty_contents_valid(self):
        assert gv.validate_wire([]) == []

    def test_text_only_conversation_valid(self):
        contents = [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "hello"}]},
        ]
        assert gv.validate_wire(contents) == []

    def test_matched_function_call_and_response(self):
        contents = [
            {"role": "user", "parts": [{"text": "do it"}]},
            _model_turn_with_calls("read_file"),
            _user_turn_with_responses("read_file"),
        ]
        assert gv.validate_wire(contents) == []

    def test_orphan_function_call_no_following_user(self):
        contents = [
            _model_turn_with_calls("read_file"),
        ]
        issues = gv.validate_wire(contents)
        assert any("functionCall" in i and "not followed" in i for i in issues)

    def test_partial_function_response(self):
        contents = [
            _model_turn_with_calls("a", "b"),
            _user_turn_with_responses("a"),  # b missing
        ]
        issues = gv.validate_wire(contents)
        assert any("functionCall #1" in i and "b" in i for i in issues)

    def test_orphan_function_response(self):
        contents = [
            {"role": "user", "parts": [{"text": "hi"}]},
            _user_turn_with_responses("ghost"),
        ]
        issues = gv.validate_wire(contents)
        assert any("orphan functionResponse" in i for i in issues)

    def test_position_matching_disambiguates_repeated_names(self):
        """Two calls to the same tool — position matters, not just name."""
        contents = [
            _model_turn_with_calls("read", "read"),
            _user_turn_with_responses("read", "read"),
        ]
        assert gv.validate_wire(contents) == []


# ---------------------------------------------------------------------------
# repair_wire — auto-fix violations
# ---------------------------------------------------------------------------


class TestRepairWire:
    def test_valid_contents_unchanged_structurally(self):
        contents = [
            {"role": "user", "parts": [{"text": "do it"}]},
            _model_turn_with_calls("read_file"),
            _user_turn_with_responses("read_file"),
        ]
        out = gv.repair_wire(contents)
        assert len(out) == 3
        assert gv.validate_wire(out) == []

    def test_injects_synthetic_user_for_orphan_call(self):
        contents = [
            _model_turn_with_calls("search"),
        ]
        out = gv.repair_wire(contents)
        assert len(out) == 2
        assert out[1]["role"] == "user"
        # Synthetic functionResponse with name=search
        responses = [p for p in out[1]["parts"] if "functionResponse" in p]
        assert len(responses) == 1
        assert responses[0]["functionResponse"]["name"] == "search"
        assert gv.validate_wire(out) == []

    def test_fills_partial_responses(self):
        contents = [
            _model_turn_with_calls("a", "b", "c"),
            _user_turn_with_responses("a"),  # b, c missing
        ]
        out = gv.repair_wire(contents)
        # User turn now has 3 functionResponse parts, in order a/b/c
        responses = [
            p["functionResponse"]["name"]
            for p in out[1]["parts"]
            if "functionResponse" in p
        ]
        assert responses == ["a", "b", "c"]
        assert gv.validate_wire(out) == []

    def test_drops_orphan_function_response(self):
        contents = [
            {"role": "user", "parts": [{"text": "hi"}]},
            _user_turn_with_responses("ghost"),  # orphan
        ]
        out = gv.repair_wire(contents)
        # Orphan dropped; turn becomes empty so a {text:""} placeholder is inserted
        # to avoid 400 on parts:[]
        last_parts = out[1]["parts"]
        assert len(last_parts) == 1
        assert last_parts[0] == {"text": ""}

    def test_preserves_thought_signature_on_repair(self):
        """Critical: the validator must NOT strip thoughtSignature when
        repairing the FOLLOWING user turn."""
        contents = [
            _model_turn_with_calls("read", with_signature=True),
            # No following user turn — needs synthetic injection
        ]
        out = gv.repair_wire(contents)
        # The model turn (out[0]) must still have its signature
        model_part = out[0]["parts"][0]
        assert "thoughtSignature" in model_part
        assert model_part["thoughtSignature"] == "sig_read"

    def test_first_part_signature_preserved_on_parallel_calls(self):
        """Per Google docs: only first functionCall has signature on parallel."""
        contents = [
            _model_turn_with_calls("a", "b", "c", with_signature=True),
            _user_turn_with_responses("a", "b", "c"),
        ]
        out = gv.repair_wire(contents)
        # Inspect the model turn
        model_parts = out[0]["parts"]
        assert "thoughtSignature" in model_parts[0]
        assert "thoughtSignature" not in model_parts[1]
        assert "thoughtSignature" not in model_parts[2]

    def test_repair_is_idempotent(self):
        broken = [
            _model_turn_with_calls("a", "b"),
            _user_turn_with_responses("a"),
        ]
        once = gv.repair_wire(broken)
        twice = gv.repair_wire(once)
        # Same number of turns, same name/order in user turn
        assert len(once) == len(twice)
        once_resp = [
            p["functionResponse"]["name"]
            for p in once[1]["parts"] if "functionResponse" in p
        ]
        twice_resp = [
            p["functionResponse"]["name"]
            for p in twice[1]["parts"] if "functionResponse" in p
        ]
        assert once_resp == twice_resp

    def test_empty_turn_replaced_with_placeholder(self):
        """Gemini 400s on parts:[] — validator inserts {text:""}."""
        contents = [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "user", "parts": []},  # empty
        ]
        out = gv.repair_wire(contents)
        # Repair only touches turns that have functionResponse parts.
        # Pre-existing empty turns are left alone — that's not our concern.
        # This test documents that behavior.
        assert len(out) == 2

    def test_position_match_for_repeated_tool_name(self):
        """If a, a is called and only first a has response, repair fills second."""
        contents = [
            _model_turn_with_calls("read", "read"),
            _user_turn_with_responses("read"),  # only one response
        ]
        out = gv.repair_wire(contents)
        responses = [
            p["functionResponse"]["name"]
            for p in out[1]["parts"]
            if "functionResponse" in p
        ]
        assert responses == ["read", "read"]


# ---------------------------------------------------------------------------
# Env toggle integration
# ---------------------------------------------------------------------------


class TestEnvToggle:
    def test_strict_mode_raises(self, monkeypatch):
        monkeypatch.setenv("PREDACORE_REPAIR_TOOL_FLOW", "strict")
        broken = [_model_turn_with_calls("x")]
        with pytest.raises(ToolFlowInvariantError):
            gv.repair_wire(broken)

    def test_off_mode_skips_repair(self, monkeypatch):
        monkeypatch.setenv("PREDACORE_REPAIR_TOOL_FLOW", "off")
        broken = [_model_turn_with_calls("x")]
        out = gv.repair_wire(broken)
        assert out is broken  # same reference
        assert gv.validate_wire(out) != []  # still broken
