"""
Comprehensive tests for jarvis.errors — structured error hierarchy.

Tests every error class: instantiation, attributes, inheritance,
string representation, and real-world construction patterns.
"""
from __future__ import annotations

import pytest

from jarvis.errors import (
    # Base
    JARVISError,
    # Tool errors
    ToolError,
    ToolNotFoundError,
    ToolTimeoutError,
    ToolPermissionError,
    ToolValidationError,
    # LLM errors
    LLMError,
    LLMRateLimitError,
    LLMConnectionError,
    LLMAllProvidersFailedError,
    # Memory errors
    MemoryStoreError,
    MemoryCapacityError,
    MemoryNotFoundError,
    # Session errors
    SessionError,
    SessionNotFoundError,
    SessionCorruptedError,
    # Security errors
    SecurityError,
    SSRFBlockedError,
    CommandInjectionError,
    PromptInjectionError,
    RateLimitExceededError,
    # Desktop errors
    DesktopError,
    DeviceNotConnectedError,
    AccessibilityPermissionError,
)


# ── Base Error ─────────────────────────────────────────────────────


class TestJARVISError:
    """Tests for the base JARVISError class."""

    def test_basic_instantiation(self):
        err = JARVISError("something broke")
        assert str(err) == "something broke"
        assert err.error_code == ""
        assert err.recoverable is False
        assert err.context == {}
        assert err.details == {}

    def test_with_error_code(self):
        err = JARVISError("fail", error_code="E001")
        assert str(err) == "[E001] fail"
        assert err.error_code == "E001"

    def test_with_recoverable(self):
        err = JARVISError("transient", recoverable=True)
        assert err.recoverable is True

    def test_with_context(self):
        ctx = {"file": "test.py", "line": 42}
        err = JARVISError("fail", context=ctx)
        assert err.context == ctx
        assert err.context["file"] == "test.py"

    def test_with_details(self):
        details = {"extra": "info"}
        err = JARVISError("fail", details=details)
        assert err.details == details

    def test_all_kwargs(self):
        err = JARVISError(
            "total failure",
            error_code="CRIT_001",
            recoverable=True,
            context={"a": 1},
            details={"b": 2},
        )
        assert str(err) == "[CRIT_001] total failure"
        assert err.recoverable is True
        assert err.context == {"a": 1}
        assert err.details == {"b": 2}

    def test_is_exception(self):
        err = JARVISError("test")
        assert isinstance(err, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(JARVISError, match="boom"):
            raise JARVISError("boom")

    def test_empty_error_code_no_brackets(self):
        err = JARVISError("clean message")
        assert "[" not in str(err)

    def test_context_defaults_to_empty_dict(self):
        err = JARVISError("test")
        assert err.context is not None
        assert isinstance(err.context, dict)

    def test_details_defaults_to_empty_dict(self):
        err = JARVISError("test")
        assert err.details is not None
        assert isinstance(err.details, dict)


# ── Tool Errors ────────────────────────────────────────────────────


class TestToolError:
    """Tests for tool error hierarchy."""

    def test_basic_tool_error(self):
        err = ToolError("tool failed", tool_name="web_search")
        assert isinstance(err, JARVISError)
        assert err.tool_name == "web_search"
        assert "tool failed" in str(err)

    def test_tool_error_default_tool_name(self):
        err = ToolError("fail")
        assert err.tool_name == ""

    def test_tool_error_with_details(self):
        err = ToolError("bad", tool_name="read_file", details={"path": "/etc/passwd"})
        assert err.details["path"] == "/etc/passwd"

    def test_tool_not_found(self):
        err = ToolNotFoundError("no such tool", tool_name="nonexistent")
        assert isinstance(err, ToolError)
        assert isinstance(err, JARVISError)
        assert err.tool_name == "nonexistent"

    def test_tool_timeout(self):
        err = ToolTimeoutError("web_search", 30.0)
        assert isinstance(err, ToolError)
        assert err.tool_name == "web_search"
        assert err.timeout_seconds == 30.0
        assert "timed out" in str(err)
        assert "30" in str(err)
        assert err.details["timeout_seconds"] == 30.0

    def test_tool_timeout_float_precision(self):
        err = ToolTimeoutError("slow_tool", 0.5)
        assert err.timeout_seconds == 0.5
        assert "0.5" in str(err)

    def test_tool_permission_error(self):
        err = ToolPermissionError("denied", tool_name="run_command")
        assert isinstance(err, ToolError)
        assert err.tool_name == "run_command"

    def test_tool_validation_error(self):
        err = ToolValidationError("bad args", tool_name="write_file")
        assert isinstance(err, ToolError)
        assert err.tool_name == "write_file"


# ── LLM Errors ─────────────────────────────────────────────────────


class TestLLMError:
    """Tests for LLM error hierarchy."""

    def test_basic_llm_error(self):
        err = LLMError("provider fail", provider="openai", model="gpt-4")
        assert isinstance(err, JARVISError)
        assert err.provider == "openai"
        assert err.model == "gpt-4"

    def test_llm_error_defaults(self):
        err = LLMError("fail")
        assert err.provider == ""
        assert err.model == ""

    def test_rate_limit_with_retry(self):
        err = LLMRateLimitError("openai", retry_after=30.0)
        assert isinstance(err, LLMError)
        assert err.provider == "openai"
        assert err.retry_after == 30.0
        assert "retry after 30.0s" in str(err)
        assert err.details["retry_after"] == 30.0

    def test_rate_limit_without_retry(self):
        err = LLMRateLimitError("anthropic")
        assert err.retry_after is None
        assert "retry after" not in str(err)
        assert "Rate limited by anthropic" in str(err)

    def test_connection_error(self):
        err = LLMConnectionError("timeout", provider="gemini")
        assert isinstance(err, LLMError)
        assert err.provider == "gemini"

    def test_all_providers_failed(self):
        errors = [("openai", "rate limited"), ("anthropic", "server error"), ("gemini", "timeout")]
        err = LLMAllProvidersFailedError(errors)
        assert isinstance(err, LLMError)
        assert err.provider_errors == errors
        assert len(err.provider_errors) == 3
        assert "openai: rate limited" in str(err)
        assert "anthropic: server error" in str(err)
        assert "gemini: timeout" in str(err)
        assert err.details["provider_errors"] == errors

    def test_all_providers_failed_single(self):
        errors = [("solo", "down")]
        err = LLMAllProvidersFailedError(errors)
        assert "solo: down" in str(err)

    def test_all_providers_failed_empty(self):
        err = LLMAllProvidersFailedError([])
        assert err.provider_errors == []
        assert "All LLM providers failed" in str(err)


# ── Memory Errors ──────────────────────────────────────────────────


class TestMemoryErrors:
    """Tests for memory error hierarchy."""

    def test_memory_store_error(self):
        err = MemoryStoreError("db locked")
        assert isinstance(err, JARVISError)

    def test_memory_capacity_error(self):
        err = MemoryCapacityError("store full")
        assert isinstance(err, MemoryStoreError)
        assert isinstance(err, JARVISError)

    def test_memory_not_found(self):
        err = MemoryNotFoundError("entry missing")
        assert isinstance(err, MemoryStoreError)


# ── Session Errors ─────────────────────────────────────────────────


class TestSessionErrors:
    """Tests for session error hierarchy."""

    def test_session_error(self):
        err = SessionError("session problem")
        assert isinstance(err, JARVISError)

    def test_session_not_found(self):
        err = SessionNotFoundError("no such session")
        assert isinstance(err, SessionError)

    def test_session_corrupted(self):
        err = SessionCorruptedError("bad json")
        assert isinstance(err, SessionError)


# ── Security Errors ────────────────────────────────────────────────


class TestSecurityErrors:
    """Tests for security error hierarchy."""

    def test_security_error(self):
        err = SecurityError("unauthorized")
        assert isinstance(err, JARVISError)

    def test_ssrf_blocked(self):
        err = SSRFBlockedError("internal URL blocked")
        assert isinstance(err, SecurityError)

    def test_command_injection(self):
        err = CommandInjectionError("shell injection detected")
        assert isinstance(err, SecurityError)

    def test_prompt_injection(self):
        err = PromptInjectionError("prompt manipulation attempt")
        assert isinstance(err, SecurityError)

    def test_rate_limit_exceeded(self):
        err = RateLimitExceededError("user-123", limit=30, window_seconds=60)
        assert isinstance(err, SecurityError)
        assert "user-123" in str(err)
        assert "30" in str(err)
        assert "60" in str(err)
        assert err.details["user_id"] == "user-123"
        assert err.details["limit"] == 30
        assert err.details["window_seconds"] == 60


# ── Desktop Errors ─────────────────────────────────────────────────


class TestDesktopErrors:
    """Tests for desktop/device error hierarchy."""

    def test_desktop_error(self):
        err = DesktopError("screen capture failed")
        assert isinstance(err, JARVISError)

    def test_device_not_connected(self):
        err = DeviceNotConnectedError("no ADB device")
        assert isinstance(err, DesktopError)

    def test_accessibility_permission(self):
        err = AccessibilityPermissionError("missing AX permission")
        assert isinstance(err, DesktopError)


# ── Inheritance Chain ──────────────────────────────────────────────


class TestInheritanceChain:
    """Verify the full inheritance tree is correct."""

    @pytest.mark.parametrize(
        "cls,parents",
        [
            (ToolError, (JARVISError,)),
            (ToolNotFoundError, (ToolError, JARVISError)),
            (ToolTimeoutError, (ToolError, JARVISError)),
            (ToolPermissionError, (ToolError, JARVISError)),
            (ToolValidationError, (ToolError, JARVISError)),
            (LLMError, (JARVISError,)),
            (LLMRateLimitError, (LLMError, JARVISError)),
            (LLMConnectionError, (LLMError, JARVISError)),
            (LLMAllProvidersFailedError, (LLMError, JARVISError)),
            (MemoryStoreError, (JARVISError,)),
            (MemoryCapacityError, (MemoryStoreError, JARVISError)),
            (MemoryNotFoundError, (MemoryStoreError, JARVISError)),
            (SessionError, (JARVISError,)),
            (SessionNotFoundError, (SessionError, JARVISError)),
            (SessionCorruptedError, (SessionError, JARVISError)),
            (SecurityError, (JARVISError,)),
            (SSRFBlockedError, (SecurityError, JARVISError)),
            (CommandInjectionError, (SecurityError, JARVISError)),
            (PromptInjectionError, (SecurityError, JARVISError)),
            (RateLimitExceededError, (SecurityError, JARVISError)),
            (DesktopError, (JARVISError,)),
            (DeviceNotConnectedError, (DesktopError, JARVISError)),
            (AccessibilityPermissionError, (DesktopError, JARVISError)),
        ],
    )
    def test_inheritance(self, cls, parents):
        for parent in parents:
            assert issubclass(cls, parent), f"{cls.__name__} should be subclass of {parent.__name__}"

    def test_all_errors_catchable_by_base(self):
        """Every custom error should be catchable with `except JARVISError`."""
        all_error_classes = [
            ToolError, ToolNotFoundError, ToolTimeoutError, ToolPermissionError,
            ToolValidationError, LLMError, LLMRateLimitError, LLMConnectionError,
            LLMAllProvidersFailedError, MemoryStoreError, MemoryCapacityError,
            MemoryNotFoundError, SessionError, SessionNotFoundError,
            SessionCorruptedError, SecurityError, SSRFBlockedError,
            CommandInjectionError, PromptInjectionError, RateLimitExceededError,
            DesktopError, DeviceNotConnectedError, AccessibilityPermissionError,
        ]
        for cls in all_error_classes:
            assert issubclass(cls, JARVISError)

    def test_total_error_count(self):
        """Verify we're testing all 23 error classes."""
        all_classes = [
            JARVISError,
            ToolError, ToolNotFoundError, ToolTimeoutError, ToolPermissionError,
            ToolValidationError,
            LLMError, LLMRateLimitError, LLMConnectionError, LLMAllProvidersFailedError,
            MemoryStoreError, MemoryCapacityError, MemoryNotFoundError,
            SessionError, SessionNotFoundError, SessionCorruptedError,
            SecurityError, SSRFBlockedError, CommandInjectionError,
            PromptInjectionError, RateLimitExceededError,
            DesktopError, DeviceNotConnectedError, AccessibilityPermissionError,
        ]
        assert len(all_classes) == 24  # 1 base + 23 specific


# ── Exception Handling Patterns ────────────────────────────────────


class TestExceptionHandlingPatterns:
    """Test real-world catch patterns that the codebase uses."""

    def test_catch_tool_errors_broadly(self):
        """Catching ToolError should catch all tool sub-errors."""
        with pytest.raises(ToolError):
            raise ToolNotFoundError("missing", tool_name="x")

        with pytest.raises(ToolError):
            raise ToolTimeoutError("y", 10)

    def test_catch_llm_errors_broadly(self):
        """Catching LLMError should catch all LLM sub-errors."""
        with pytest.raises(LLMError):
            raise LLMRateLimitError("openai", 5.0)

        with pytest.raises(LLMError):
            raise LLMAllProvidersFailedError([("a", "b")])

    def test_catch_security_errors_broadly(self):
        """Catching SecurityError should catch all security sub-errors."""
        with pytest.raises(SecurityError):
            raise SSRFBlockedError("blocked")

        with pytest.raises(SecurityError):
            raise RateLimitExceededError("u1", 10, 60)

    def test_catch_all_with_jarvis_error(self):
        """JARVISError should be the universal catch-all."""
        errors_to_test = [
            ToolTimeoutError("t", 5),
            LLMRateLimitError("p"),
            MemoryCapacityError("full"),
            SessionCorruptedError("bad"),
            SSRFBlockedError("no"),
            DeviceNotConnectedError("gone"),
        ]
        for err in errors_to_test:
            with pytest.raises(JARVISError):
                raise err
