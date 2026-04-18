"""
Tests for PredaCore Operators — Base, Mock, Retry, and Enums.

Covers:
  - BaseOperator contract (via mock implementations)
  - MockDesktopOperator and MockAndroidOperator
  - Macro execution with abort support
  - Retry decorator (sync and error classification)
  - OperatorError structure
"""
import time
import pytest

from predacore.operators.base import (
    BaseOperator,
    OperatorPlatform,
    OperatorError,
    MacroAbortToken,
    ActionCategory,
)
from predacore.operators.mock import MockDesktopOperator, MockAndroidOperator
from predacore.operators.retry import with_retry


# ═══════════════════════════════════════════════════════════════════
# OperatorError Tests
# ═══════════════════════════════════════════════════════════════════


class TestOperatorError:
    def test_basic_error(self):
        err = OperatorError("something broke", action="screenshot")
        assert "something broke" in str(err)
        assert err.action == "screenshot"
        assert err.recoverable is True

    def test_non_recoverable(self):
        err = OperatorError("fatal", recoverable=False)
        assert err.recoverable is False

    def test_to_dict(self):
        err = OperatorError("msg", action="tap", suggestion="try again")
        d = err.to_dict()
        assert d["error"] == "msg"
        assert d["action"] == "tap"
        assert d["suggestion"] == "try again"
        assert d["recoverable"] is True


# ═══════════════════════════════════════════════════════════════════
# MacroAbortToken Tests
# ═══════════════════════════════════════════════════════════════════


class TestMacroAbortToken:
    def test_initially_not_aborted(self):
        token = MacroAbortToken()
        assert not token.is_aborted
        assert token.reason == ""

    def test_abort_sets_flag(self):
        token = MacroAbortToken()
        token.abort("user cancelled")
        assert token.is_aborted
        assert token.reason == "user cancelled"

    def test_abort_default_reason(self):
        token = MacroAbortToken()
        token.abort()
        assert token.is_aborted
        assert token.reason == "Aborted"


# ═══════════════════════════════════════════════════════════════════
# MockDesktopOperator Tests
# ═══════════════════════════════════════════════════════════════════


class TestMockDesktopOperator:
    def test_platform(self):
        op = MockDesktopOperator()
        assert op.platform == OperatorPlatform.MACOS

    def test_available_by_default(self):
        op = MockDesktopOperator()
        assert op.available is True

    def test_available_configurable(self):
        op = MockDesktopOperator(available=False)
        assert op.available is False

    def test_execute_returns_default(self):
        op = MockDesktopOperator()
        result = op.execute("screenshot", {})
        assert result["ok"] is True
        assert result["action"] == "screenshot"

    def test_call_log(self):
        op = MockDesktopOperator()
        op.execute("type_text", {"text": "hello"})
        op.execute("press_key", {"key": "enter"})
        assert len(op.call_log) == 2
        assert op.call_log[0] == ("type_text", {"text": "hello"})
        assert op.call_log[1] == ("press_key", {"key": "enter"})

    def test_custom_response(self):
        op = MockDesktopOperator()
        op.set_response("frontmost_app", {"app_name": "Safari"})
        result = op.execute("frontmost_app", {})
        assert result["app_name"] == "Safari"

    def test_configured_failure(self):
        op = MockDesktopOperator()
        op.set_failure("open_app", "App not found")
        with pytest.raises(OperatorError, match="App not found"):
            op.execute("open_app", {"app_name": "fake"})

    def test_clear_failure(self):
        op = MockDesktopOperator()
        op.set_failure("open_app", "fail")
        op.clear_failure("open_app")
        result = op.execute("open_app", {"app_name": "ok"})
        assert result["ok"] is True

    def test_assert_called(self):
        op = MockDesktopOperator()
        op.execute("screenshot", {})
        op.assert_called("screenshot")
        op.assert_called("screenshot", times=1)

    def test_assert_called_fails(self):
        op = MockDesktopOperator()
        with pytest.raises(AssertionError):
            op.assert_called("never_called")

    def test_assert_not_called(self):
        op = MockDesktopOperator()
        op.assert_not_called("screenshot")

    def test_assert_not_called_fails(self):
        op = MockDesktopOperator()
        op.execute("screenshot", {})
        with pytest.raises(AssertionError):
            op.assert_not_called("screenshot")

    def test_reset(self):
        op = MockDesktopOperator()
        op.execute("screenshot", {})
        op.set_response("custom", {"data": 1})
        op.set_failure("fail", "err")
        op.reset()
        assert len(op.call_log) == 0

    def test_empty_action_raises(self):
        op = MockDesktopOperator()
        with pytest.raises(OperatorError, match="action is required"):
            op.execute("", {})

    def test_health_check(self):
        op = MockDesktopOperator()
        health = op.health_check()
        assert "accessibility_trusted" in health

    def test_supported_actions(self):
        op = MockDesktopOperator()
        actions = op.supported_actions
        assert "screenshot" in actions
        assert "type_text" in actions
        assert "run_macro" in actions

    def test_supports_method(self):
        op = MockDesktopOperator()
        assert op.supports("screenshot")
        assert not op.supports("nonexistent_action")

    def test_capabilities(self):
        op = MockDesktopOperator()
        caps = op.capabilities()
        assert caps["platform"] == "macos"
        assert caps["available"] is True
        assert isinstance(caps["actions"], list)


# ═══════════════════════════════════════════════════════════════════
# MockAndroidOperator Tests
# ═══════════════════════════════════════════════════════════════════


class TestMockAndroidOperator:
    def test_platform(self):
        op = MockAndroidOperator()
        assert op.platform == OperatorPlatform.ANDROID

    def test_execute_tap(self):
        op = MockAndroidOperator()
        result = op.execute("tap", {"x": 100, "y": 200})
        assert result["ok"] is True
        assert result["action"] == "tap"

    def test_health_check(self):
        op = MockAndroidOperator()
        health = op.health_check()
        assert health["adb_ok"] is True
        assert health["connected"] is True


# ═══════════════════════════════════════════════════════════════════
# Macro Execution Tests (via Mock Operators)
# ═══════════════════════════════════════════════════════════════════


class TestMacroExecution:
    def test_simple_macro(self):
        op = MockDesktopOperator()
        result = op.execute("run_macro", {
            "steps": [
                {"action": "open_app", "app_name": "Safari"},
                {"action": "type_text", "text": "hello"},
                {"action": "press_key", "key": "enter"},
            ],
        })
        assert result["ok"] is True
        assert result["completed_steps"] == 3

    def test_macro_stop_on_error(self):
        op = MockDesktopOperator()
        op.set_failure("type_text", "keyboard locked")
        result = op.execute("run_macro", {
            "steps": [
                {"action": "open_app", "app_name": "Safari"},
                {"action": "type_text", "text": "hello"},
                {"action": "press_key", "key": "enter"},
            ],
            "stop_on_error": True,
        })
        # Should have failed at step 2
        assert result["completed_steps"] == 1
        assert "failed_step" in result

    def test_macro_continue_on_error(self):
        op = MockDesktopOperator()
        op.set_failure("type_text", "keyboard locked")
        result = op.execute("run_macro", {
            "steps": [
                {"action": "open_app", "app_name": "Safari"},
                {"action": "type_text", "text": "hello"},
                {"action": "press_key", "key": "enter"},
            ],
            "stop_on_error": False,
        })
        assert result["completed_steps"] == 3
        # Step 2 should have an error
        assert "error" in result["results"][1]

    def test_macro_abort(self):
        op = MockDesktopOperator()
        token = MacroAbortToken()
        token.abort("user hit ESC")

        result = op.execute("run_macro", {
            "steps": [
                {"action": "open_app", "app_name": "Safari"},
                {"action": "type_text", "text": "hello"},
            ],
            "_abort_token": token,
        })
        assert result["aborted"] is True
        assert result["abort_reason"] == "user hit ESC"
        assert result["completed_steps"] == 0

    def test_macro_empty_steps_raises(self):
        op = MockDesktopOperator()
        with pytest.raises(OperatorError, match="non-empty steps"):
            op.execute("run_macro", {"steps": []})

    def test_macro_too_many_steps(self):
        op = MockDesktopOperator()
        steps = [{"action": "screenshot"} for _ in range(100)]
        with pytest.raises(OperatorError, match="limited to"):
            op.execute("run_macro", {"steps": steps})

    def test_macro_nesting_depth_limit(self):
        """Macro within a macro should hit depth limit."""
        op = MockDesktopOperator()
        # Set up a response that tries to run another macro
        # (simulated by calling execute_macro directly)
        # The base class tracks depth via _macro_depth
        nested = [{"action": "screenshot"}]
        # Direct call to test depth tracking
        op._macro_depth = 3  # Manually set to max
        with pytest.raises(OperatorError, match="nesting too deep"):
            op.execute_macro(nested)
        op._macro_depth = 0  # Reset


# ═══════════════════════════════════════════════════════════════════
# Retry Decorator Tests
# ═══════════════════════════════════════════════════════════════════


class TestRetryDecorator:
    def test_no_retry_on_success(self):
        call_count = 0

        @with_retry(max_attempts=3, retryable_errors=(RuntimeError,))
        def succeeds():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = succeeds()
        assert result == "ok"
        assert call_count == 1

    def test_retries_on_transient_error(self):
        call_count = 0

        @with_retry(
            max_attempts=3,
            backoff_base=0.01,
            retryable_errors=(RuntimeError,),
            retryable_messages=("transient",),
        )
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient error")
            return "success"

        result = flaky()
        assert result == "success"
        assert call_count == 3

    def test_no_retry_on_non_transient_error(self):
        call_count = 0

        @with_retry(
            max_attempts=3,
            retryable_errors=(RuntimeError,),
            retryable_messages=("transient",),
        )
        def permanent_fail():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("permanent error — unfixable")

        with pytest.raises(RuntimeError, match="permanent error"):
            permanent_fail()
        assert call_count == 1  # No retry

    def test_exhausts_retries(self):
        call_count = 0

        @with_retry(
            max_attempts=3,
            backoff_base=0.01,
            retryable_errors=(RuntimeError,),
            retryable_messages=("transient",),
        )
        def always_transient():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("transient forever")

        with pytest.raises(RuntimeError, match="transient forever"):
            always_transient()
        assert call_count == 3  # All attempts exhausted

    def test_non_retryable_exception_type(self):
        call_count = 0

        @with_retry(
            max_attempts=3,
            retryable_errors=(ValueError,),  # Only retry ValueError
            retryable_messages=("transient",),
        )
        def wrong_exception():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("transient but wrong type")

        with pytest.raises(RuntimeError):
            wrong_exception()
        assert call_count == 1  # No retry — wrong exception type

    def test_kax_error_is_transient(self):
        """macOS kAXErrorCannotComplete should trigger retry."""
        call_count = 0

        @with_retry(
            max_attempts=3,
            backoff_base=0.01,
            retryable_errors=(RuntimeError,),
        )
        def ax_flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("kAXErrorCannotComplete")
            return "recovered"

        result = ax_flaky()
        assert result == "recovered"
        assert call_count == 2


# ═══════════════════════════════════════════════════════════════════
# Platform Enum Tests
# ═══════════════════════════════════════════════════════════════════


class TestPlatformEnums:
    def test_operator_platforms(self):
        assert OperatorPlatform.MACOS == "macos"
        assert OperatorPlatform.ANDROID == "android"
        assert OperatorPlatform.LINUX == "linux"
        assert OperatorPlatform.WINDOWS == "windows"

    def test_action_categories(self):
        assert ActionCategory.APP_CONTROL == "app_control"
        assert ActionCategory.KEYBOARD == "keyboard"
        assert ActionCategory.MOUSE == "mouse"
        assert ActionCategory.SCREENSHOT == "screenshot"
