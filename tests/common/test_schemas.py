import pytest
"""
try:
    from jarvis._vendor.common import schemas  # test import
except ImportError:
    pytest.skip("Module not vendored", allow_module_level=True)

Tests for src/common/schemas.py — Pydantic v2 validated models.
"""

import pytest
from pydantic import ValidationError

from jarvis._vendor.common.schemas import (
    AgentTask,
    ChannelType,
    ChatMessage,
    IncomingMessage,
    LLMRequest,
    LLMResponse,
    MessageRole,
    OutgoingMessage,
    ServiceHealth,
    TaskStatus,
    ToolCall,
    ToolResult,
    TrustLevel,
)

# ── ChatMessage Tests ────────────────────────────────────────────────

class TestChatMessage:
    def test_basic_user_message(self):
        msg = ChatMessage(role="user", content="hello JARVIS")
        assert msg.role == "user"
        assert msg.content == "hello JARVIS"
        assert msg.timestamp > 0
        assert msg.metadata == {}
        assert msg.tool_calls == []

    def test_assistant_message(self):
        msg = ChatMessage(role="assistant", content="How can I help?")
        assert msg.role == "assistant"

    def test_system_message(self):
        msg = ChatMessage(role="system", content="You are JARVIS")
        assert msg.role == "system"

    def test_tool_message(self):
        msg = ChatMessage(role="tool", content="", tool_name="web_search",
                          tool_result='{"results": []}')
        assert msg.role == "tool"
        assert msg.tool_name == "web_search"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid_role", content="hello")

    def test_empty_user_content_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="")

    def test_empty_user_content_whitespace_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content="   ")

    def test_empty_assistant_content_allowed(self):
        # Assistant can have empty content (e.g., tool_calls only)
        msg = ChatMessage(role="assistant", content="")
        assert msg.content == ""

    def test_metadata(self):
        msg = ChatMessage(role="user", content="hi", metadata={"channel": "cli"})
        assert msg.metadata["channel"] == "cli"

    def test_tool_calls(self):
        tc = ToolCall(tool_name="web_search", arguments={"query": "weather"})
        msg = ChatMessage(role="assistant", content="", tool_calls=[tc])
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].tool_name == "web_search"

    def test_to_dict(self):
        msg = ChatMessage(role="user", content="hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"
        assert "timestamp" in d

    def test_from_dict(self):
        data = {"role": "user", "content": "hello", "timestamp": 1000.0}
        msg = ChatMessage.from_dict(data)
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.timestamp == 1000.0

    def test_to_llm_format(self):
        msg = ChatMessage(role="user", content="hello")
        fmt = msg.to_llm_format()
        assert fmt == {"role": "user", "content": "hello"}

    def test_to_llm_format_with_tool_calls(self):
        tc = ToolCall(tool_name="calculator", arguments={"expr": "2+2"})
        msg = ChatMessage(role="assistant", content="", tool_calls=[tc])
        fmt = msg.to_llm_format()
        assert "tool_calls" in fmt
        assert len(fmt["tool_calls"]) == 1

    def test_roundtrip_serialization(self):
        original = ChatMessage(
            role="user", content="hello",
            metadata={"key": "value"},
            tool_name="test_tool",
        )
        d = original.to_dict()
        restored = ChatMessage.from_dict(d)
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.metadata == original.metadata
        assert restored.tool_name == original.tool_name


# ── ToolCall / ToolResult Tests ──────────────────────────────────────

class TestToolModels:
    def test_tool_call_creation(self):
        tc = ToolCall(tool_name="web_search", arguments={"q": "test"})
        assert tc.tool_name == "web_search"
        assert tc.arguments == {"q": "test"}
        assert len(tc.tool_call_id) == 8

    def test_tool_call_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            ToolCall(tool_name="", arguments={})

    def test_tool_call_long_name_rejected(self):
        with pytest.raises(ValidationError):
            ToolCall(tool_name="x" * 129, arguments={})

    def test_tool_result(self):
        tr = ToolResult(
            tool_call_id="abc123",
            tool_name="web_search",
            output="results here",
            success=True,
            execution_time_ms=42.5,
        )
        assert tr.success is True
        assert tr.execution_time_ms == 42.5

    def test_tool_result_failure(self):
        tr = ToolResult(
            tool_call_id="abc123",
            tool_name="web_search",
            error="Connection timeout",
            success=False,
        )
        assert tr.success is False
        assert tr.error == "Connection timeout"


# ── IncomingMessage Tests ────────────────────────────────────────────

class TestIncomingMessage:
    def test_basic_creation(self):
        msg = IncomingMessage(channel="cli", user_id="shubham", text="hello")
        assert msg.channel == "cli"
        assert msg.user_id == "shubham"
        assert msg.text == "hello"
        assert len(msg.trace_id) == 16

    def test_invalid_channel_rejected(self):
        with pytest.raises(ValidationError):
            IncomingMessage(channel="carrier_pigeon", user_id="user1", text="hello")

    def test_empty_user_id_rejected(self):
        with pytest.raises(ValidationError):
            IncomingMessage(channel="cli", user_id="", text="hello")

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            IncomingMessage(channel="cli", user_id="user1", text="")

    def test_attachments(self):
        msg = IncomingMessage(
            channel="telegram", user_id="user1", text="check this",
            attachments=[{"type": "image", "url": "https://example.com/img.png"}],
        )
        assert len(msg.attachments) == 1

    def test_all_channel_types(self):
        for ch in ChannelType:
            msg = IncomingMessage(channel=ch.value, user_id="u1", text="test")
            assert msg.channel == ch.value


# ── OutgoingMessage Tests ────────────────────────────────────────────

class TestOutgoingMessage:
    def test_basic_creation(self):
        msg = OutgoingMessage(
            channel="cli", user_id="shubham",
            text="Hello!", session_id="sess123",
        )
        assert msg.channel == "cli"
        assert msg.session_id == "sess123"

    def test_with_thinking(self):
        msg = OutgoingMessage(
            channel="cli", user_id="u1",
            text="Done", session_id="s1",
            thinking="I analyzed the request...",
        )
        assert msg.thinking is not None


# ── LLMRequest Tests ─────────────────────────────────────────────────

class TestLLMRequest:
    def test_basic_request(self):
        req = LLMRequest(
            messages=[ChatMessage(role="user", content="hello")],
            model="gemini-pro",
        )
        assert req.model == "gemini-pro"
        assert req.temperature == 0.7
        assert req.max_tokens == 4096

    def test_empty_messages_rejected(self):
        with pytest.raises(ValidationError):
            LLMRequest(messages=[])

    def test_only_system_message_rejected(self):
        with pytest.raises(ValidationError):
            LLMRequest(
                messages=[ChatMessage(role="system", content="You are JARVIS")],
            )

    def test_system_plus_user_accepted(self):
        req = LLMRequest(
            messages=[
                ChatMessage(role="system", content="You are JARVIS"),
                ChatMessage(role="user", content="hello"),
            ],
        )
        assert len(req.messages) == 2

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            LLMRequest(
                messages=[ChatMessage(role="user", content="hi")],
                temperature=3.0,  # > 2.0
            )

    def test_max_tokens_bounds(self):
        with pytest.raises(ValidationError):
            LLMRequest(
                messages=[ChatMessage(role="user", content="hi")],
                max_tokens=0,
            )

    def test_trace_id_generated(self):
        req = LLMRequest(
            messages=[ChatMessage(role="user", content="hi")],
        )
        assert len(req.trace_id) == 16


# ── LLMResponse Tests ────────────────────────────────────────────────

class TestLLMResponse:
    def test_basic_response(self):
        resp = LLMResponse(
            content="Hello!",
            model="gemini-pro",
            provider="gemini",
            latency_ms=150.5,
        )
        assert resp.content == "Hello!"
        assert resp.latency_ms == 150.5

    def test_response_with_usage(self):
        resp = LLMResponse(
            content="Hi",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        assert resp.usage["total_tokens"] == 15

    def test_response_with_tool_calls(self):
        resp = LLMResponse(
            content="",
            tool_calls=[ToolCall(tool_name="search", arguments={"q": "test"})],
        )
        assert len(resp.tool_calls) == 1


# ── AgentTask Tests ──────────────────────────────────────────────────

class TestAgentTask:
    def test_task_creation(self):
        task = AgentTask(session_id="s1", description="Run web search")
        assert task.status == "pending"
        assert task.session_id == "s1"
        assert len(task.task_id) == 8
        assert len(task.trace_id) == 16

    def test_empty_description_rejected(self):
        with pytest.raises(ValidationError):
            AgentTask(session_id="s1", description="")

    def test_negative_timeout_rejected(self):
        with pytest.raises(ValidationError):
            AgentTask(session_id="s1", description="test", timeout=-1.0)


# ── ServiceHealth Tests ──────────────────────────────────────────────

class TestServiceHealth:
    def test_healthy(self):
        h = ServiceHealth(service="jarvis-core", status="healthy")
        assert h.status == "healthy"

    def test_degraded(self):
        h = ServiceHealth(
            service="memory", status="degraded",
            details={"sqlite": "ok", "embeddings": "unavailable"},
        )
        assert h.status == "degraded"

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            ServiceHealth(service="test", status="on_fire")

    def test_frozen(self):
        h = ServiceHealth(service="test")
        with pytest.raises(ValidationError):
            h.status = "unhealthy"  # type: ignore


# ── Enum Tests ───────────────────────────────────────────────────────

class TestEnums:
    def test_message_roles(self):
        assert set(MessageRole) == {
            MessageRole.USER, MessageRole.ASSISTANT,
            MessageRole.SYSTEM, MessageRole.TOOL,
        }

    def test_channel_types(self):
        assert "cli" in [ch.value for ch in ChannelType]
        assert "telegram" in [ch.value for ch in ChannelType]

    def test_trust_levels(self):
        assert TrustLevel.YOLO.value == "yolo"
        assert TrustLevel.NORMAL.value == "normal"
        assert TrustLevel.PARANOID.value == "paranoid"

    def test_task_statuses(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
