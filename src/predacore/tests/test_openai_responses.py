"""
Unit tests for predacore.llm_providers.openai_responses.

OpenAI's Responses API (POST /v1/responses) replaces Chat Completions for
ChatGPT-OAuth (Codex) tokens. This adapter targets the new wire format —
typed input[]/output[] items with function_call / function_call_output.
"""
from __future__ import annotations

from predacore.llm_providers import openai_responses as orsp


# ---------------------------------------------------------------------------
# Request shape — _serialize_messages_for_responses + _serialize_tools
# ---------------------------------------------------------------------------


class TestRequestShape:
    def test_plain_messages_emit_message_items(self):
        out = orsp.OpenAIResponsesProvider._serialize_messages_for_responses(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi back"},
            ]
        )
        assert out == [
            {"type": "message", "role": "user", "content": "hello"},
            {"type": "message", "role": "assistant", "content": "hi back"},
        ]

    def test_assistant_with_tool_calls_emits_function_call_items(self):
        out = orsp.OpenAIResponsesProvider._serialize_messages_for_responses(
            [
                {
                    "role": "assistant",
                    "content": "calling tool",
                    "tool_calls": [
                        {"id": "call_abc", "name": "do_thing", "arguments": {"x": 1}},
                    ],
                }
            ]
        )
        # assistant text → message item, tool_calls → function_call items
        assert len(out) == 2
        assert out[0]["type"] == "message"
        assert out[0]["role"] == "assistant"
        assert out[1]["type"] == "function_call"
        assert out[1]["call_id"] == "call_abc"
        assert out[1]["name"] == "do_thing"
        # arguments is JSON-string in Responses API
        assert isinstance(out[1]["arguments"], str)
        assert '"x": 1' in out[1]["arguments"] or '"x":1' in out[1]["arguments"]

    def test_tool_results_emit_function_call_output_items(self):
        out = orsp.OpenAIResponsesProvider._serialize_messages_for_responses(
            [
                {
                    "role": "tool",
                    "tool_results": [
                        {"call_id": "call_abc", "name": "do_thing", "result": "42"},
                    ],
                }
            ]
        )
        assert out == [
            {"type": "function_call_output", "call_id": "call_abc", "output": "42"},
        ]

    def test_tool_definitions_use_flat_shape(self):
        out = orsp.OpenAIResponsesProvider._serialize_tools_for_responses(
            [
                {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {}},
                }
            ]
        )
        assert out == [
            {
                "type": "function",
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    def test_no_tools_returns_none(self):
        assert orsp.OpenAIResponsesProvider._serialize_tools_for_responses(None) is None
        assert orsp.OpenAIResponsesProvider._serialize_tools_for_responses([]) is None


# ---------------------------------------------------------------------------
# Response parsing — _parse_responses
# ---------------------------------------------------------------------------


class TestResponseParser:
    def test_text_response(self):
        result = orsp._parse_responses(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Hello!"}
                        ],
                    }
                ],
                "usage": {"input_tokens": 12, "output_tokens": 3},
            }
        )
        assert result["content"] == "Hello!"
        assert result["tool_calls"] == []
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 12
        assert result["usage"]["completion_tokens"] == 3

    def test_function_call_response(self):
        result = orsp._parse_responses(
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_xyz",
                        "name": "search",
                        "arguments": '{"q": "hello"}',
                    }
                ],
                "usage": {"input_tokens": 50, "output_tokens": 20},
            }
        )
        assert result["tool_calls"] == [
            {"id": "call_xyz", "name": "search", "arguments": {"q": "hello"}}
        ]
        assert result["finish_reason"] == "tool_calls"

    def test_mixed_text_and_tool_call(self):
        result = orsp._parse_responses(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "I'll search."}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "search",
                        "arguments": "{}",
                    },
                ],
                "usage": {"input_tokens": 5, "output_tokens": 10},
            }
        )
        assert result["content"] == "I'll search."
        assert result["tool_calls"][0]["id"] == "call_1"
        assert result["finish_reason"] == "tool_calls"

    def test_reasoning_items_ignored(self):
        result = orsp._parse_responses(
            {
                "output": [
                    {"type": "reasoning", "id": "rs_1", "summary": [{"text": "thinking..."}]},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Done."}],
                    },
                ],
                "usage": {"input_tokens": 8, "output_tokens": 5},
            }
        )
        # Reasoning items don't contribute to content
        assert result["content"] == "Done."

    def test_string_content_field(self):
        # Some Responses replies use a flat string instead of content blocks
        result = orsp._parse_responses(
            {
                "output": [
                    {"type": "message", "role": "assistant", "content": "flat string"}
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )
        assert result["content"] == "flat string"


# ---------------------------------------------------------------------------
# Repair — orphan function_call_output dropped
# ---------------------------------------------------------------------------


class TestRepairResponsesInput:
    def test_orphan_function_call_output_dropped(self):
        items = [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "function_call_output", "call_id": "call_zzz", "output": "ghost"},
        ]
        out = orsp._repair_responses_input(items)
        # Orphan output dropped
        assert len(out) == 1
        assert out[0]["type"] == "message"

    def test_matched_function_call_output_kept(self):
        items = [
            {"type": "function_call", "call_id": "call_1", "name": "x", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        ]
        out = orsp._repair_responses_input(items)
        assert len(out) == 2

    def test_function_call_without_output_kept(self):
        # Lone function_call without output is fine — output may arrive next turn
        items = [
            {"type": "function_call", "call_id": "call_1", "name": "x", "arguments": "{}"},
        ]
        out = orsp._repair_responses_input(items)
        assert out == items
