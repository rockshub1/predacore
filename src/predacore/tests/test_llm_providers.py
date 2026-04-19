"""
Comprehensive tests for predacore.llm_providers — LLM router, circuit breaker,
text tool adapter, base provider, and provider config.

Tests mock external APIs — no real LLM calls needed.
"""
from __future__ import annotations

import time

import pytest

from predacore.llm_providers.base import LLMProvider, ProviderConfig
from predacore.llm_providers.circuit_breaker import CircuitBreaker
from predacore.llm_providers.text_tool_adapter import (
    build_full_text_prompt,
    build_tool_prompt,
    parse_tool_calls,
)

# ── ProviderConfig ─────────────────────────────────────────────────


class TestProviderConfig:
    """Tests for the ProviderConfig dataclass."""

    def test_defaults(self):
        cfg = ProviderConfig()
        assert cfg.model == ""
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096
        assert cfg.reasoning_effort == "medium"
        assert cfg.api_key == ""
        assert cfg.extras == {}

    def test_custom_values(self):
        cfg = ProviderConfig(
            model="gpt-4",
            temperature=0.3,
            max_tokens=8192,
            api_key="sk-test",
        )
        assert cfg.model == "gpt-4"
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 8192
        assert cfg.api_key == "sk-test"

    def test_extras(self):
        cfg = ProviderConfig(extras={"provider": "anthropic", "session_id": "abc123"})
        assert cfg.extras["provider"] == "anthropic"
        assert cfg.extras["session_id"] == "abc123"

    def test_model_fallbacks(self):
        cfg = ProviderConfig(model_fallbacks=["gpt-4o-mini", "gpt-3.5-turbo"])
        assert len(cfg.model_fallbacks) == 2


# ── CircuitBreaker ─────────────────────────────────────────────────


class TestCircuitBreaker:
    """Tests for the per-provider circuit breaker."""

    def test_initial_state_closed(self):
        cb = CircuitBreaker()
        assert cb.is_open("openai") is False

    def test_single_failure_stays_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure("openai", Exception("error"))
        assert cb.is_open("openai") is False

    def test_trips_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, window_seconds=300)
        for i in range(3):
            cb.record_failure("openai", Exception(f"error {i}"))
        assert cb.is_open("openai") is True

    def test_different_providers_independent(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("openai", Exception("err"))
        cb.record_failure("openai", Exception("err"))
        assert cb.is_open("openai") is True
        assert cb.is_open("gemini") is False

    def test_cooldown_recovery(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure("openai", Exception("err"))
        assert cb.is_open("openai") is True
        time.sleep(0.02)
        assert cb.is_open("openai") is False

    def test_success_clears_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure("openai", Exception("err"))
        cb.record_failure("openai", Exception("err"))
        cb.record_success("openai")
        cb.record_failure("openai", Exception("err"))
        # Only 1 failure after success — should not trip
        assert cb.is_open("openai") is False

    def test_reset_single(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure("openai", Exception("err"))
        assert cb.is_open("openai") is True
        cb.reset("openai")
        assert cb.is_open("openai") is False

    def test_reset_all(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure("openai", Exception("err"))
        cb.record_failure("gemini", Exception("err"))
        assert cb.is_open("openai") is True
        assert cb.is_open("gemini") is True
        cb.reset_all()
        assert cb.is_open("openai") is False
        assert cb.is_open("gemini") is False

    def test_window_expiry(self):
        cb = CircuitBreaker(failure_threshold=3, window_seconds=0.01)
        cb.record_failure("openai", Exception("err"))
        cb.record_failure("openai", Exception("err"))
        time.sleep(0.02)
        # Old failures expired — new one doesn't trip
        cb.record_failure("openai", Exception("err"))
        assert cb.is_open("openai") is False

    def test_thread_safety(self):
        """Circuit breaker should be thread-safe."""
        import threading
        cb = CircuitBreaker(failure_threshold=100)
        errors = []

        def record_failures():
            try:
                for _ in range(50):
                    cb.record_failure("test", Exception("err"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_failures) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# ── Text Tool Adapter ──────────────────────────────────────────────


class TestBuildToolPrompt:
    """Tests for text-based tool prompt builder."""

    def test_empty_tools(self):
        assert build_tool_prompt(None) == ""
        assert build_tool_prompt([]) == ""

    def test_single_tool(self):
        tools = [
            {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            }
        ]
        prompt = build_tool_prompt(tools)
        assert "web_search" in prompt
        assert "Search the web" in prompt
        assert "query" in prompt
        assert "(required)" in prompt

    def test_multiple_tools(self):
        tools = [
            {"name": "tool_a", "description": "Desc A", "parameters": {}},
            {"name": "tool_b", "description": "Desc B", "parameters": {}},
        ]
        prompt = build_tool_prompt(tools)
        assert "tool_a" in prompt
        assert "tool_b" in prompt

    def test_enum_parameters(self):
        tools = [
            {
                "name": "set_mode",
                "description": "Set mode",
                "parameters": {
                    "properties": {
                        "mode": {
                            "type": "string",
                            "description": "Mode",
                            "enum": ["fast", "slow"],
                        }
                    }
                },
            }
        ]
        prompt = build_tool_prompt(tools)
        assert "fast" in prompt
        assert "slow" in prompt

    def test_contains_instructions(self):
        tools = [{"name": "test", "description": "desc", "parameters": {}}]
        prompt = build_tool_prompt(tools)
        assert "TOOL USE INSTRUCTIONS" in prompt
        assert "<tool_call>" in prompt


class TestParseToolCalls:
    """Tests for parsing <tool_call> blocks from LLM output."""

    def test_single_tool_call(self):
        text = 'Let me search. <tool_call>{"name": "web_search", "arguments": {"query": "test"}}</tool_call>'
        clean, calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "web_search"
        assert calls[0]["arguments"]["query"] == "test"
        assert "Let me search." in clean
        assert "<tool_call>" not in clean

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/a"}}</tool_call>'
            ' then '
            '<tool_call>{"name": "web_search", "arguments": {"query": "hello"}}</tool_call>'
        )
        clean, calls = parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["name"] == "read_file"
        assert calls[1]["name"] == "web_search"

    def test_no_tool_calls(self):
        text = "Just a normal response with no tools."
        clean, calls = parse_tool_calls(text)
        assert clean == text
        assert calls == []

    def test_malformed_json_skipped(self):
        text = '<tool_call>NOT VALID JSON</tool_call>'
        clean, calls = parse_tool_calls(text)
        assert calls == []

    def test_missing_name_skipped(self):
        text = '<tool_call>{"arguments": {"x": 1}}</tool_call>'
        clean, calls = parse_tool_calls(text)
        assert calls == []

    def test_multiline_tool_call(self):
        text = """<tool_call>
{
    "name": "run_command",
    "arguments": {"command": "ls -la"}
}
</tool_call>"""
        _, calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "run_command"

    def test_default_arguments(self):
        text = '<tool_call>{"name": "simple_tool"}</tool_call>'
        _, calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["arguments"] == {}


class TestBuildFullTextPrompt:
    """Tests for flattening messages + tools into a single text prompt."""

    def test_basic(self):
        messages = [
            {"role": "system", "content": "You are PredaCore"},
            {"role": "user", "content": "Hello"},
        ]
        prompt = build_full_text_prompt(messages, None)
        assert "[SYSTEM]" in prompt
        assert "You are PredaCore" in prompt
        assert "[USER]" in prompt
        assert "Hello" in prompt
        assert "[ASSISTANT]" in prompt

    def test_with_tools(self):
        messages = [{"role": "user", "content": "search"}]
        tools = [{"name": "web_search", "description": "Search", "parameters": {}}]
        prompt = build_full_text_prompt(messages, tools)
        assert "web_search" in prompt
        assert "TOOL USE INSTRUCTIONS" in prompt


# ── LLMProvider Base ───────────────────────────────────────────────


class TestLLMProviderBase:
    """Tests for the abstract LLMProvider base class."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            LLMProvider(ProviderConfig())

    def test_subclass_with_chat(self):
        class MockProvider(LLMProvider):
            name = "mock"
            async def chat(self, messages, tools=None, temperature=None,
                          max_tokens=None, stream_fn=None):
                return {"content": "mock response", "tool_calls": [], "usage": {}, "finish_reason": "stop"}

        provider = MockProvider(ProviderConfig())
        assert provider.name == "mock"
        assert provider.supports_native_tools is True
        assert provider.executes_tool_loop_internally is False

    def test_config_accessible(self):
        class MockProvider(LLMProvider):
            name = "mock"
            async def chat(self, messages, tools=None, temperature=None,
                          max_tokens=None, stream_fn=None):
                return {}

        cfg = ProviderConfig(model="test-model", temperature=0.5)
        provider = MockProvider(cfg)
        assert provider.config.model == "test-model"
        assert provider.config.temperature == 0.5


# ── LLMInterface Router (mocked) ──────────────────────────────────


class TestLLMInterfaceRouter:
    """Tests for LLMInterface router logic with mocked providers."""

    @pytest.fixture
    def mock_config(self):
        from predacore.config import LLMConfig, PredaCoreConfig
        cfg = PredaCoreConfig()
        cfg.llm = LLMConfig(
            provider="mock",
            fallback_providers=["mock_fallback"],
            auto_fallback=True,
            max_retries=1,
        )
        return cfg

    def test_provider_chain(self, mock_config):
        from predacore.llm_providers.router import LLMInterface
        router = LLMInterface(mock_config)
        assert router._provider_chain == ["mock", "mock_fallback"]

    def test_active_provider(self, mock_config):
        from predacore.llm_providers.router import LLMInterface
        router = LLMInterface(mock_config)
        assert router.active_provider == "mock"

    def test_set_active_model(self, mock_config):
        from predacore.llm_providers.router import LLMInterface
        router = LLMInterface(mock_config)
        router.set_active_model(provider="new_provider")
        assert router.active_provider == "new_provider"
        assert router._provider_chain[0] == "new_provider"

    def test_set_active_model_name(self, mock_config):
        from predacore.llm_providers.router import LLMInterface
        router = LLMInterface(mock_config)
        router.set_active_model(model="gpt-4o")
        assert router.active_model == "gpt-4o"

    def test_auto_fallback_property(self, mock_config):
        from predacore.llm_providers.router import LLMInterface
        router = LLMInterface(mock_config)
        assert router.auto_fallback is True

    def test_available_providers(self, mock_config):
        from predacore.llm_providers.router import LLMInterface
        router = LLMInterface(mock_config)
        assert "mock" in router.available_providers
        assert "mock_fallback" in router.available_providers


# ── Claude Models ──────────────────────────────────────────────────


class TestClaudeModels:
    """Tests for Claude model alias resolution."""

    def test_opus_alias(self):
        from predacore.llm_providers.claude_models import resolve_claude_model
        assert "opus" in resolve_claude_model("opus").lower()

    def test_sonnet_alias(self):
        from predacore.llm_providers.claude_models import resolve_claude_model
        assert "sonnet" in resolve_claude_model("sonnet").lower()

    def test_haiku_alias(self):
        from predacore.llm_providers.claude_models import resolve_claude_model
        assert "haiku" in resolve_claude_model("haiku").lower()

    def test_passthrough(self):
        from predacore.llm_providers.claude_models import resolve_claude_model
        assert resolve_claude_model("custom-model-123") == "custom-model-123"

    def test_whitespace_stripped(self):
        from predacore.llm_providers.claude_models import resolve_claude_model
        assert resolve_claude_model("  opus  ") == resolve_claude_model("opus")
