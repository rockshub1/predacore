"""
Phase A refactor (2026-04-21) — per-provider tool-turn round-trip tests.

Locks in the core invariant: for every supported provider, a realistic
assistant→tool-result round-trip produces NO "[Calling tool: X]" or
"[Tool Result: X]" text stubs anywhere in the resulting message list.

The legacy flat path in ``core.py`` used to emit those stubs as assistant
content, which taught LLMs the bracket syntax via in-context learning —
a.k.a. context poisoning. The new provider-owned serialization eliminates
this at the source. These tests fail loudly if a regression reintroduces it.
"""
from __future__ import annotations

import json

import pytest

from predacore.llm_providers import (
    AssistantResponse,
    Message,
    ProviderConfig,
    ToolCallRef,
    ToolResultRef,
    message_from_dict,
    message_to_dict,
)
from predacore.llm_providers.anthropic import AnthropicProvider
from predacore.llm_providers.gemini import GeminiProvider, _build_payload
from predacore.llm_providers.gemini_cli import GeminiCLIProvider
from predacore.llm_providers.openai import OpenAIProvider
from predacore.llm_providers.text_tool_adapter import build_full_text_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_round_trip_messages(
    provider,
    assistant_resp: AssistantResponse,
    results: list[ToolResultRef],
    initial_messages: list[Message] | None = None,
) -> list[Message]:
    """Simulate core.py: take initial messages, append one tool round-trip."""
    msgs = list(initial_messages or [])
    provider.append_assistant_turn(msgs, assistant_resp)
    provider.append_tool_results_turn(msgs, results)
    return msgs


def _assert_no_poison(messages: list[Message]) -> None:
    """The one invariant that matters: no legacy text stubs anywhere."""
    for m in messages:
        content = m.content or ""
        assert "[Calling tool:" not in content, (
            f"POISON REGRESSION: [Calling tool:] stub leaked into {m.role!r} content: "
            f"{content[:100]!r}"
        )
        assert "[Tool Result:" not in content, (
            f"POISON REGRESSION: [Tool Result:] stub leaked into {m.role!r} content: "
            f"{content[:100]!r}"
        )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class TestAnthropicToolTurn:
    """AnthropicProvider bundles tool results into ONE user turn with
    content_blocks preserving thinking signatures and tool_use_id linkage."""

    def _provider(self) -> AnthropicProvider:
        return AnthropicProvider(
            config=ProviderConfig(model="claude-opus-4-6", api_key="sk-ant-test")
        )

    def test_thinking_signature_preserved(self):
        prov = self._provider()
        resp = AssistantResponse(
            content="Searching.",
            tool_calls=[ToolCallRef(id="toolu_01", name="grep", arguments={"p": "x"})],
            provider_extras={
                "content_blocks": [
                    {"type": "thinking", "thinking": "...", "signature": "sig_abc"},
                    {"type": "text", "text": "Searching."},
                    {"type": "tool_use", "id": "toolu_01", "name": "grep", "input": {"p": "x"}},
                ],
            },
        )
        msgs = _build_round_trip_messages(prov, resp, [
            ToolResultRef(call_id="toolu_01", name="grep", result="match"),
        ])
        # Signature survives on assistant turn
        asst = msgs[0]
        assert asst.role == "assistant"
        blocks = asst.provider_extras.get("content_blocks", [])
        assert any(b.get("signature") == "sig_abc" for b in blocks)

    def test_tool_results_bundled_into_one_user_turn(self):
        prov = self._provider()
        resp = AssistantResponse(
            content="",
            tool_calls=[
                ToolCallRef(id="toolu_1", name="grep", arguments={}),
                ToolCallRef(id="toolu_2", name="read", arguments={}),
                ToolCallRef(id="toolu_3", name="write", arguments={}),
            ],
        )
        msgs = _build_round_trip_messages(prov, resp, [
            ToolResultRef(call_id="toolu_1", name="grep", result="r1"),
            ToolResultRef(call_id="toolu_2", name="read", result="r2"),
            ToolResultRef(call_id="toolu_3", name="write", result="r3"),
        ])
        # 1 assistant + 1 user turn (with 3 result blocks)
        assert len(msgs) == 2
        user = msgs[1]
        assert user.role == "user"
        blocks = user.provider_extras.get("content_blocks", [])
        assert len(blocks) == 3
        assert all(b["type"] == "tool_result" for b in blocks)
        assert blocks[0]["tool_use_id"] == "toolu_1"

    def test_no_poison_stubs(self):
        prov = self._provider()
        resp = AssistantResponse(
            content="ok",
            tool_calls=[ToolCallRef(id="toolu_x", name="grep", arguments={})],
        )
        msgs = _build_round_trip_messages(prov, resp, [
            ToolResultRef(call_id="toolu_x", name="grep", result="ok"),
        ])
        _assert_no_poison(msgs)

    def test_is_error_propagates(self):
        prov = self._provider()
        resp = AssistantResponse(
            content="",
            tool_calls=[ToolCallRef(id="toolu_e", name="grep", arguments={})],
        )
        msgs = _build_round_trip_messages(prov, resp, [
            ToolResultRef(call_id="toolu_e", name="grep", result="bad", is_error=True),
        ])
        blocks = msgs[1].provider_extras.get("content_blocks", [])
        assert blocks[0].get("is_error") is True


# ---------------------------------------------------------------------------
# OpenAI (+ all 11 OpenAI-compatible providers via PROVIDER_ENDPOINTS)
# ---------------------------------------------------------------------------


class TestOpenAIToolTurn:
    """OpenAIProvider uses default OpenAI-shaped append_*_turn. The wire
    serializer (_serialize_messages_for_wire) translates to native
    tool_calls / role='tool' with tool_call_id at send time."""

    def _provider(self) -> OpenAIProvider:
        return OpenAIProvider(
            config=ProviderConfig(
                model="gpt-4o",
                api_key="sk-test",
                extras={"provider": "openai"},
            )
        )

    def test_assistant_turn_has_abstract_tool_calls(self):
        prov = self._provider()
        resp = AssistantResponse(
            content="searching",
            tool_calls=[ToolCallRef(id="call_1", name="grep", arguments={"p": "x"})],
        )
        msgs = _build_round_trip_messages(prov, resp, [
            ToolResultRef(call_id="call_1", name="grep", result="hit"),
        ])
        assert msgs[0].role == "assistant"
        assert len(msgs[0].tool_calls) == 1
        assert msgs[0].tool_calls[0].id == "call_1"

    def test_tool_result_turn_uses_tool_role_per_result(self):
        prov = self._provider()
        resp = AssistantResponse(
            content="",
            tool_calls=[
                ToolCallRef(id="call_a", name="grep", arguments={}),
                ToolCallRef(id="call_b", name="read", arguments={}),
            ],
        )
        msgs = _build_round_trip_messages(prov, resp, [
            ToolResultRef(call_id="call_a", name="grep", result="r1"),
            ToolResultRef(call_id="call_b", name="read", result="r2"),
        ])
        # 1 assistant + 2 role="tool" messages (OpenAI expects one per result)
        assert len(msgs) == 3
        assert msgs[1].role == "tool" and msgs[1].tool_results[0].call_id == "call_a"
        assert msgs[2].role == "tool" and msgs[2].tool_results[0].call_id == "call_b"

    def test_wire_translation_produces_native_openai_format(self):
        prov = self._provider()
        resp = AssistantResponse(
            content="searching",
            tool_calls=[ToolCallRef(id="call_1", name="grep", arguments={"p": "x"})],
        )
        msgs = _build_round_trip_messages(prov, resp, [
            ToolResultRef(call_id="call_1", name="grep", result="hit"),
        ])
        dict_msgs = [message_to_dict(m) for m in msgs]
        wire = OpenAIProvider._serialize_messages_for_wire(dict_msgs)
        # Assistant turn native OpenAI shape
        assert wire[0]["role"] == "assistant"
        tc = wire[0]["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "grep"
        # arguments MUST be JSON string per OpenAI spec
        assert isinstance(tc["function"]["arguments"], str)
        assert json.loads(tc["function"]["arguments"]) == {"p": "x"}
        # Tool result native shape
        assert wire[1]["role"] == "tool"
        assert wire[1]["tool_call_id"] == "call_1"
        assert wire[1]["content"] == "hit"

    def test_legacy_dict_passthrough(self):
        """Old-format callers (still producing [Calling tool: X] stubs)
        pass through unchanged so migration can happen incrementally."""
        legacy = [
            {"role": "system", "content": "You help."},
            {"role": "user", "content": "hi"},
        ]
        out = OpenAIProvider._serialize_messages_for_wire(legacy)
        assert out == legacy

    def test_no_poison_stubs(self):
        prov = self._provider()
        resp = AssistantResponse(
            content="ok",
            tool_calls=[ToolCallRef(id="x", name="grep", arguments={})],
        )
        msgs = _build_round_trip_messages(prov, resp, [
            ToolResultRef(call_id="x", name="grep", result="ok"),
        ])
        _assert_no_poison(msgs)


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


class TestGeminiToolTurn:
    """GeminiProvider uses functionCall / functionResponse parts and maps
    assistant role → 'model' at wire-serialization time. IDs are synthesized
    since Gemini doesn't emit them natively."""

    def _provider(self) -> GeminiProvider:
        return GeminiProvider(config=ProviderConfig(model="gemini-3-pro-preview"))

    def test_tool_call_ids_synthesized(self):
        prov = self._provider()
        resp = AssistantResponse(
            content="searching",
            tool_calls=[
                ToolCallRef(id="", name="grep", arguments={"p": "x"}),  # no id
                ToolCallRef(id="", name="read", arguments={"f": "a"}),  # no id
            ],
        )
        msgs = _build_round_trip_messages(prov, resp, [])
        asst = msgs[0]
        # Both IDs should be synthesized and unique
        ids = [tc.id for tc in asst.tool_calls]
        assert all(i.startswith("gemini_") for i in ids)
        assert ids[0] != ids[1]

    def test_function_call_parts_built(self):
        prov = self._provider()
        resp = AssistantResponse(
            content="searching",
            tool_calls=[ToolCallRef(id="g_0", name="grep", arguments={"p": "x"})],
        )
        msgs = _build_round_trip_messages(prov, resp, [])
        parts = msgs[0].provider_extras.get("parts", [])
        assert any(
            p.get("functionCall", {}).get("name") == "grep" for p in parts
        )

    def test_function_response_parts_built(self):
        prov = self._provider()
        msgs = [
            Message(role="user", content="find foo"),
        ]
        prov.append_tool_results_turn(msgs, [
            ToolResultRef(call_id="g_0", name="grep", result="hit", is_error=False),
        ])
        result_turn = msgs[-1]
        assert result_turn.role == "user"  # Gemini: results in user turn
        parts = result_turn.provider_extras.get("parts", [])
        assert any(
            p.get("functionResponse", {}).get("name") == "grep" for p in parts
        )

    def test_wire_payload_maps_roles_correctly(self):
        prov = self._provider()
        resp = AssistantResponse(
            content="ok",
            tool_calls=[ToolCallRef(id="g_0", name="grep", arguments={})],
        )
        msgs = _build_round_trip_messages(prov, resp, [
            ToolResultRef(call_id="g_0", name="grep", result="hit"),
        ])
        dict_msgs = [message_to_dict(m) for m in msgs]
        payload = _build_payload(
            messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}] + dict_msgs,
            tools=None,
            temperature=0.0,
            max_tokens=100,
        )
        # systemInstruction populated
        assert payload.get("systemInstruction")
        contents = payload["contents"]
        # Assistant → model
        roles = [c["role"] for c in contents]
        assert "model" in roles
        # Tool result → user (Gemini convention)
        last = contents[-1]
        assert last["role"] == "user"
        assert any("functionResponse" in p for p in last["parts"])

    def test_no_poison_stubs(self):
        prov = self._provider()
        resp = AssistantResponse(
            content="ok",
            tool_calls=[ToolCallRef(id="g_0", name="grep", arguments={})],
        )
        msgs = _build_round_trip_messages(prov, resp, [
            ToolResultRef(call_id="g_0", name="grep", result="hit"),
        ])
        _assert_no_poison(msgs)


# ---------------------------------------------------------------------------
# Gemini CLI
# ---------------------------------------------------------------------------


class TestGeminiCLIToolTurn:
    """GeminiCLIProvider serializes tool calls as inline <tool_call> XML
    and results as formatted text in user turns (CLI model doesn't
    understand the 'tool' role)."""

    def _provider(self) -> GeminiCLIProvider:
        return GeminiCLIProvider(config=ProviderConfig(model="gemini-2.5-flash"))

    def test_tool_call_reinjected_as_xml(self):
        prov = self._provider()
        resp = AssistantResponse(
            content="Let me search.",
            tool_calls=[ToolCallRef(id="cli_0", name="grep", arguments={"p": "x"})],
        )
        msgs = _build_round_trip_messages(prov, resp, [])
        assert msgs[0].role == "assistant"
        assert "<tool_call>" in msgs[0].content
        assert "grep" in msgs[0].content

    def test_tool_results_routed_to_user_role(self):
        prov = self._provider()
        resp = AssistantResponse(
            content="",
            tool_calls=[ToolCallRef(id="cli_0", name="grep", arguments={})],
        )
        msgs = _build_round_trip_messages(prov, resp, [
            ToolResultRef(call_id="cli_0", name="grep", result="hit"),
        ])
        # CLI doesn't know "tool" role — results go in user turn
        assert msgs[-1].role == "user"
        assert "[tool_result:" in msgs[-1].content

    def test_text_prompt_round_trip(self):
        prov = self._provider()
        resp = AssistantResponse(
            content="searching",
            tool_calls=[ToolCallRef(id="cli_0", name="grep", arguments={"p": "x"})],
        )
        msgs = _build_round_trip_messages(prov, resp, [
            ToolResultRef(call_id="cli_0", name="grep", result="hit"),
        ])
        dict_msgs = [message_to_dict(m) for m in msgs]
        prompt = build_full_text_prompt(
            [{"role": "system", "content": "help"}] + dict_msgs,
            tools=None,
        )
        assert "[ASSISTANT]" in prompt
        assert "<tool_call>" in prompt
        assert "[tool_result:" in prompt
        # Still no legacy poison
        assert "[Calling tool:" not in prompt
        assert "[Tool Result:" not in prompt

    def test_no_poison_stubs(self):
        prov = self._provider()
        resp = AssistantResponse(
            content="ok",
            tool_calls=[ToolCallRef(id="cli_0", name="grep", arguments={})],
        )
        msgs = _build_round_trip_messages(prov, resp, [
            ToolResultRef(call_id="cli_0", name="grep", result="hit"),
        ])
        _assert_no_poison(msgs)


# ---------------------------------------------------------------------------
# Cross-provider invariant — the load-bearing assertion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_factory", [
    lambda: AnthropicProvider(config=ProviderConfig(model="claude-opus-4-6", api_key="sk-ant-test")),
    lambda: OpenAIProvider(config=ProviderConfig(model="gpt-4o", api_key="sk-test", extras={"provider": "openai"})),
    lambda: GeminiProvider(config=ProviderConfig(model="gemini-3-pro-preview")),
    lambda: GeminiCLIProvider(config=ProviderConfig(model="gemini-2.5-flash")),
])
def test_no_legacy_stubs_any_provider(provider_factory):
    """For EVERY supported provider, a realistic tool round-trip produces
    zero '[Calling tool: X]' or '[Tool Result: X]' stubs. If this fails,
    context poisoning regressed."""
    prov = provider_factory()
    resp = AssistantResponse(
        content="Let me help.",
        tool_calls=[
            ToolCallRef(id="id_1", name="grep", arguments={"pattern": "foo"}),
            ToolCallRef(id="id_2", name="read_file", arguments={"path": "/tmp"}),
        ],
    )
    msgs = _build_round_trip_messages(prov, resp, [
        ToolResultRef(call_id="id_1", name="grep", result="match"),
        ToolResultRef(call_id="id_2", name="read_file", result="content"),
    ])
    _assert_no_poison(msgs)
