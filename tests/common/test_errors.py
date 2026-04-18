"""Tests for src/common/errors.py — Prometheus structured error hierarchy."""
import pytest

from predacore._vendor.common.errors import (
    AgentLifecycleError,
    AuthenticationError,
    AuthorizationError,
    ChannelConnectionError,
    ChannelError,
    ChannelMessageTooLongError,
    ChannelRateLimitError,
    ConfigurationError,
    EGMViolationError,
    LLMAllProvidersFailedError,
    LLMContextLengthError,
    LLMProviderError,
    LLMRateLimitError,
    MemoryNotFoundError,
    MemoryServiceError,
    PersistenceError,
    PlanningError,
    PrometheusError,
    SandboxError,
    SandboxNotAvailableError,
    SandboxTimeoutError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolTimeoutError,
)


class TestPrometheusErrorBase:
    """Tests for the base PrometheusError class."""

    def test_basic_creation(self):
        err = PrometheusError("something broke")
        assert str(err) == "[PREDACORE_ERROR] something broke"
        assert err.error_code == "PREDACORE_ERROR"
        assert err.context == {}
        assert err.recoverable is False

    def test_with_context(self):
        err = PrometheusError(
            "bad thing",
            context={"trace_id": "abc123", "tool": "shell_exec"},
        )
        s = str(err)
        assert "[PREDACORE_ERROR]" in s
        assert "trace_id='abc123'" in s
        assert "tool='shell_exec'" in s

    def test_override_error_code(self):
        err = PrometheusError("x", error_code="CUSTOM_CODE")
        assert err.error_code == "CUSTOM_CODE"
        assert "[CUSTOM_CODE]" in str(err)

    def test_override_recoverable(self):
        err = PrometheusError("x", recoverable=True)
        assert err.recoverable is True

    def test_is_exception(self):
        with pytest.raises(PrometheusError):
            raise PrometheusError("boom")

    def test_empty_message(self):
        err = PrometheusError()
        assert err.error_code == "PREDACORE_ERROR"


class TestToolErrors:
    """Tests for tool-related error classes."""

    def test_tool_execution_error(self):
        err = ToolExecutionError(
            "shell_exec failed",
            context={"command": "rm -rf /", "exit_code": 1},
        )
        assert err.error_code == "TOOL_EXEC_FAILED"
        assert err.recoverable is True
        assert isinstance(err, PrometheusError)

    def test_tool_not_found(self):
        err = ToolNotFoundError("nonexistent_tool")
        assert err.error_code == "TOOL_NOT_FOUND"
        assert err.recoverable is False

    def test_tool_permission(self):
        err = ToolPermissionError("shell_exec blocked by trust policy")
        assert err.error_code == "TOOL_PERMISSION_DENIED"
        assert err.recoverable is False

    def test_tool_timeout_inherits_from_execution(self):
        err = ToolTimeoutError("timed out after 30s")
        assert err.error_code == "TOOL_TIMEOUT"
        assert err.recoverable is True
        assert isinstance(err, ToolExecutionError)
        assert isinstance(err, PrometheusError)


class TestLLMErrors:
    """Tests for LLM provider error classes."""

    def test_provider_error(self):
        err = LLMProviderError("API returned 500")
        assert err.error_code == "LLM_PROVIDER_ERROR"
        assert err.recoverable is True

    def test_rate_limit(self):
        err = LLMRateLimitError("429 Too Many Requests")
        assert err.error_code == "LLM_RATE_LIMIT"
        assert err.recoverable is True
        assert isinstance(err, LLMProviderError)

    def test_context_length(self):
        err = LLMContextLengthError("message too long")
        assert err.error_code == "LLM_CONTEXT_TOO_LONG"
        assert err.recoverable is False

    def test_all_providers_failed(self):
        err = LLMAllProvidersFailedError("all 3 providers failed")
        assert err.error_code == "LLM_ALL_PROVIDERS_FAILED"
        assert err.recoverable is False


class TestSecurityErrors:
    def test_auth_error(self):
        err = AuthenticationError("expired token")
        assert err.error_code == "AUTH_FAILED"
        assert err.recoverable is False

    def test_authz_error(self):
        err = AuthorizationError("insufficient permissions")
        assert err.error_code == "AUTH_FORBIDDEN"
        assert err.recoverable is False


class TestSandboxErrors:
    def test_sandbox_base(self):
        err = SandboxError("container crashed")
        assert err.error_code == "SANDBOX_ERROR"
        assert err.recoverable is True

    def test_sandbox_not_available(self):
        err = SandboxNotAvailableError("Docker not running")
        assert err.error_code == "SANDBOX_NOT_AVAILABLE"
        assert err.recoverable is False
        assert isinstance(err, SandboxError)

    def test_sandbox_timeout(self):
        err = SandboxTimeoutError("execution exceeded 60s")
        assert err.error_code == "SANDBOX_TIMEOUT"
        assert err.recoverable is True


class TestMemoryErrors:
    def test_memory_service_error(self):
        err = MemoryServiceError("SQLite locked")
        assert err.error_code == "MEMORY_ERROR"
        assert err.recoverable is True

    def test_memory_not_found(self):
        err = MemoryNotFoundError("unknown UUID")
        assert err.error_code == "MEMORY_NOT_FOUND"
        assert err.recoverable is False

    def test_persistence_error(self):
        err = PersistenceError("disk full")
        assert err.error_code == "PERSISTENCE_ERROR"
        assert err.recoverable is True


class TestChannelErrors:
    def test_channel_error(self):
        assert ChannelError("x").error_code == "CHANNEL_ERROR"

    def test_channel_connection(self):
        assert ChannelConnectionError("x").error_code == "CHANNEL_CONNECT_FAILED"
        assert ChannelConnectionError("x").recoverable is True

    def test_channel_rate_limit(self):
        assert ChannelRateLimitError("x").error_code == "CHANNEL_RATE_LIMIT"

    def test_channel_msg_too_long(self):
        err = ChannelMessageTooLongError("4096 chars")
        assert err.error_code == "CHANNEL_MSG_TOO_LONG"
        assert err.recoverable is False


class TestMiscErrors:
    def test_planning_error(self):
        assert PlanningError("x").error_code == "PLANNING_ERROR"

    def test_agent_lifecycle(self):
        assert AgentLifecycleError("x").error_code == "AGENT_LIFECYCLE_ERROR"

    def test_egm_violation(self):
        err = EGMViolationError("action blocked")
        assert err.error_code == "EGM_VIOLATION"
        assert err.recoverable is False

    def test_config_error(self):
        assert ConfigurationError("x").error_code == "CONFIG_ERROR"
        assert ConfigurationError("x").recoverable is False


class TestErrorInheritanceChain:
    """Verify the full inheritance hierarchy so `except` blocks work correctly."""

    def test_catch_tool_timeout_as_execution_error(self):
        with pytest.raises(ToolExecutionError):
            raise ToolTimeoutError("timeout")

    def test_catch_tool_timeout_as_prometheus_error(self):
        with pytest.raises(PrometheusError):
            raise ToolTimeoutError("timeout")

    def test_catch_rate_limit_as_llm_error(self):
        with pytest.raises(LLMProviderError):
            raise LLMRateLimitError("429")

    def test_catch_sandbox_not_available_as_sandbox_error(self):
        with pytest.raises(SandboxError):
            raise SandboxNotAvailableError("Docker not found")

    def test_catch_memory_not_found_as_memory_error(self):
        with pytest.raises(MemoryServiceError):
            raise MemoryNotFoundError("UUID")
