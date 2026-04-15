"""
Unit tests for jarvis.llm_providers.message_validator.

Validator is the structural safety net that runs before every Anthropic
request. It catches tool_use/tool_result linkage bugs locally instead of
letting Anthropic 400 with a generic "connectivity issue".

Tests cover:
  * ``validate_tool_flow`` — reports violations without mutating
  * ``repair_tool_flow`` — auto-repairs common breakages:
      - Orphaned assistant tool_use (no following user turn)
      - Partially-satisfied tool_use (missing tool_result ids)
      - Orphaned user tool_result (no preceding tool_use)
  * Both pure-string and structured content variants are handled

Every repair emits a logger.warning — regressions stay visible in logs.
"""
from __future__ import annotations

from jarvis.llm_providers import message_validator as mv


# ---------------------------------------------------------------------------
# validate_tool_flow — reports violations, doesn't mutate
# ---------------------------------------------------------------------------


class TestValidateToolFlow:
    def test_empty_message_list_is_valid(self):
        assert mv.validate_tool_flow([]) == []

    def test_plain_string_messages_are_valid(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert mv.validate_tool_flow(messages) == []

    def test_valid_tool_use_and_tool_result(self):
        messages = [
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "sure"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "/x"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "contents",
                    }
                ],
            },
        ]
        assert mv.validate_tool_flow(messages) == []

    def test_orphaned_tool_use_no_following_user_turn(self):
        messages = [
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "/x"},
                    }
                ],
            },
        ]
        issues = mv.validate_tool_flow(messages)
        assert len(issues) == 1
        assert "not followed by a user turn" in issues[0]

    def test_orphaned_tool_use_followed_by_another_assistant(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "/x"},
                    }
                ],
            },
            {"role": "assistant", "content": "oops"},
        ]
        issues = mv.validate_tool_flow(messages)
        assert any("not followed by a user turn" in i for i in issues)

    def test_missing_tool_result_for_tool_use(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "/x"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "read_file",
                        "input": {"path": "/y"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "ok",
                    }
                    # toolu_2 is missing!
                ],
            },
        ]
        issues = mv.validate_tool_flow(messages)
        assert any("missing tool_result" in i and "toolu_2" in i for i in issues)

    def test_orphaned_user_tool_result_no_preceding_tool_use(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_ghost",
                        "content": "out of nowhere",
                    }
                ],
            },
        ]
        issues = mv.validate_tool_flow(messages)
        assert any(
            "orphaned tool_result" in i and "toolu_ghost" in i for i in issues
        )

    def test_mixed_content_assistant_with_thinking_and_tool_use(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "...", "signature": "sig"},
                    {"type": "text", "text": "reading"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "/x"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "ok",
                    }
                ],
            },
        ]
        assert mv.validate_tool_flow(messages) == []


# ---------------------------------------------------------------------------
# repair_tool_flow — auto-repairs violations
# ---------------------------------------------------------------------------


class TestRepairToolFlow:
    def test_valid_flow_unchanged(self):
        messages = [
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "/x"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "ok",
                    }
                ],
            },
        ]
        repaired = mv.repair_tool_flow(messages)
        assert len(repaired) == 3
        # Same structural shape after the repair
        assert mv.validate_tool_flow(repaired) == []

    def test_injects_synthetic_user_turn_for_orphaned_tool_use(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_orphan",
                        "name": "read_file",
                        "input": {"path": "/x"},
                    }
                ],
            },
        ]
        repaired = mv.repair_tool_flow(messages)
        assert len(repaired) == 2
        # Second message is the synthetic user turn
        assert repaired[1]["role"] == "user"
        blocks = repaired[1]["content"]
        assert isinstance(blocks, list)
        assert blocks[0]["type"] == "tool_result"
        assert blocks[0]["tool_use_id"] == "toolu_orphan"
        assert blocks[0]["is_error"] is True
        # After repair, validation passes
        assert mv.validate_tool_flow(repaired) == []

    def test_injects_missing_tool_result_into_existing_user_turn(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "/x"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "read_file",
                        "input": {"path": "/y"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "ok",
                    }
                    # toolu_2 missing
                ],
            },
        ]
        repaired = mv.repair_tool_flow(messages)
        assert len(repaired) == 2
        user_blocks = repaired[1]["content"]
        result_ids = [
            b["tool_use_id"]
            for b in user_blocks
            if b.get("type") == "tool_result"
        ]
        assert "toolu_1" in result_ids
        assert "toolu_2" in result_ids  # injected
        # After repair, validation passes
        assert mv.validate_tool_flow(repaired) == []

    def test_drops_orphaned_tool_result(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_ghost",
                        "content": "from nowhere",
                    }
                ],
            },
        ]
        repaired = mv.repair_tool_flow(messages)
        # The orphaned tool_result turn is replaced with a placeholder
        # since all its blocks were dropped
        last = repaired[-1]
        assert last["role"] == "user"
        blocks = last["content"]
        assert isinstance(blocks, list)
        # No tool_result blocks remain
        assert not any(b.get("type") == "tool_result" for b in blocks)

    def test_drops_only_orphaned_results_keeps_others_in_same_turn(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "/x"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "valid",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_ghost",
                        "content": "orphan",
                    },
                ],
            },
        ]
        repaired = mv.repair_tool_flow(messages)
        user_blocks = repaired[1]["content"]
        result_ids = {
            b["tool_use_id"]
            for b in user_blocks
            if b.get("type") == "tool_result"
        }
        assert "toolu_1" in result_ids
        assert "toolu_ghost" not in result_ids

    def test_repair_is_idempotent(self):
        """Running repair twice should produce the same result as once."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "/x"},
                    }
                ],
            },
        ]
        once = mv.repair_tool_flow(messages)
        twice = mv.repair_tool_flow(once)
        assert len(once) == len(twice)
        # Both should validate clean
        assert mv.validate_tool_flow(once) == []
        assert mv.validate_tool_flow(twice) == []

    def test_repair_preserves_plain_string_messages(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        repaired = mv.repair_tool_flow(messages)
        assert len(repaired) == 2
        assert repaired[0]["content"] == "hi"
        assert repaired[1]["content"] == "hello"

    def test_multi_turn_with_multiple_tool_uses_all_repaired(self):
        messages = [
            {"role": "user", "content": "q1"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_a",
                        "name": "read",
                        "input": {},
                    }
                ],
            },
            # missing user/tool_result turn here
            {"role": "user", "content": "q2"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_b",
                        "name": "read",
                        "input": {},
                    }
                ],
            },
            # another missing user/tool_result turn
        ]
        repaired = mv.repair_tool_flow(messages)
        assert mv.validate_tool_flow(repaired) == []
        # Synthetic tool_result turns should exist for both ids
        all_result_ids: set[str] = set()
        for msg in repaired:
            content = msg.get("content")
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        all_result_ids.add(b.get("tool_use_id", ""))
        assert "toolu_a" in all_result_ids
        assert "toolu_b" in all_result_ids

    def test_repair_simulates_the_production_bug_path(self):
        """
        Recreate the exact broken shape that caused the Telegram outage:
        assistant turn has tool_use, but the synthetic [Calling tool] text
        path replaced what should be a user tool_result turn with a bare
        string message.
        """
        messages = [
            {"role": "user", "content": "read /etc/hosts"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "reading"},
                    {
                        "type": "tool_use",
                        "id": "toolu_xyz",
                        "name": "read_file",
                        "input": {"path": "/etc/hosts"},
                    },
                ],
            },
            # What the old flat-string path produced:
            {
                "role": "user",
                "content": "[Tool Result: read_file]\n127.0.0.1 localhost",
            },
        ]
        # BEFORE REPAIR: this would hit Anthropic and 400 because the
        # user turn contains a plain string, not a structured tool_result.
        repaired = mv.repair_tool_flow(messages)
        # AFTER REPAIR: the validator detects the missing tool_result
        # because the following user turn has a string, so the assistant
        # turn's tool_use looks orphaned (no matching tool_result in blocks).
        # repair injects a synthetic user turn with the correct tool_use_id.
        validation = mv.validate_tool_flow(repaired)
        assert validation == [], f"still invalid after repair: {validation}"
