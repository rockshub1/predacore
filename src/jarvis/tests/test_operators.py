"""
Comprehensive tests for JARVIS operators system — Phase 4 Capabilities Layer.

Tests: operator enums, base operator ABC, mock operators, macro execution,
MacroAbortToken, with_retry decorator, and async_with_retry.

Target: 60+ tests covering all operator infrastructure.
"""
from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

# ── Imports under test ─────────────────────────────────────────────────

from jarvis.operators.enums import (
    DesktopAction,
    AndroidAction,
    VisionAction,
    ScreenshotQuality,
    SMART_ACTIONS,
    NATIVE_ONLY_ACTIONS,
    NATIVE_CAPABLE_ACTIONS,
)
from jarvis.operators.base import (
    BaseOperator,
    OperatorPlatform,
    ActionCategory,
    OperatorError,
    MacroAbortToken,
)
from jarvis.operators.mock import (
    MockDesktopOperator,
    MockAndroidOperator,
)
from jarvis.operators.retry import (
    with_retry,
    async_with_retry,
)


# ═══════════════════════════════════════════════════════════════════════
# 1. DesktopAction Enum
# ═══════════════════════════════════════════════════════════════════════


class TestDesktopAction:
    """Tests for DesktopAction enum — grouped frozensets and baseline members."""

    def test_member_count(self):
        """Sanity check: DesktopAction must have at least the core 31 members.
        New actions can be added without updating this test."""
        assert len(DesktopAction) >= 31

    @pytest.mark.parametrize(
        "member,value",
        [
            (DesktopAction.OPEN_APP, "open_app"),
            (DesktopAction.FOCUS_APP, "focus_app"),
            (DesktopAction.FRONTMOST_APP, "frontmost_app"),
            (DesktopAction.OPEN_URL, "open_url"),
            (DesktopAction.TYPE_TEXT, "type_text"),
            (DesktopAction.PRESS_KEY, "press_key"),
            (DesktopAction.MOUSE_MOVE, "mouse_move"),
            (DesktopAction.MOUSE_CLICK, "mouse_click"),
            (DesktopAction.MOUSE_DOUBLE_CLICK, "mouse_double_click"),
            (DesktopAction.MOUSE_RIGHT_CLICK, "mouse_right_click"),
            (DesktopAction.MOUSE_DRAG, "mouse_drag"),
            (DesktopAction.MOUSE_SCROLL, "mouse_scroll"),
            (DesktopAction.GET_MOUSE_POSITION, "get_mouse_position"),
            (DesktopAction.SCREENSHOT, "screenshot"),
            (DesktopAction.CLIPBOARD_READ, "clipboard_read"),
            (DesktopAction.CLIPBOARD_WRITE, "clipboard_write"),
            (DesktopAction.LIST_WINDOWS, "list_windows"),
            (DesktopAction.MOVE_WINDOW, "move_window"),
            (DesktopAction.RESIZE_WINDOW, "resize_window"),
            (DesktopAction.MINIMIZE_WINDOW, "minimize_window"),
            (DesktopAction.LIST_MONITORS, "list_monitors"),
            (DesktopAction.AX_QUERY, "ax_query"),
            (DesktopAction.AX_CLICK, "ax_click"),
            (DesktopAction.AX_SET_VALUE, "ax_set_value"),
            (DesktopAction.AX_REQUEST_ACCESS, "ax_request_access"),
            (DesktopAction.SMART_TYPE, "smart_type"),
            (DesktopAction.SMART_RUN_COMMAND, "smart_run_command"),
            (DesktopAction.SMART_CREATE_NOTE, "smart_create_note"),
            (DesktopAction.RUN_MACRO, "run_macro"),
            (DesktopAction.HEALTH_CHECK, "health_check"),
            (DesktopAction.SLEEP, "sleep"),
        ],
    )
    def test_member_values(self, member, value):
        """Each DesktopAction member should have the correct string value."""
        assert member.value == value

    def test_str_enum_semantics(self):
        """DesktopAction extends (str, Enum) so string comparison works."""
        assert DesktopAction.SCREENSHOT == "screenshot"
        assert "mouse_click" == DesktopAction.MOUSE_CLICK

    def test_lookup_by_value(self):
        assert DesktopAction("open_app") is DesktopAction.OPEN_APP

    def test_invalid_lookup_raises(self):
        with pytest.raises(ValueError):
            DesktopAction("nonexistent_action")


class TestSmartActions:
    """Tests for SMART_ACTIONS frozenset."""

    def test_smart_actions_contents(self):
        assert DesktopAction.SMART_TYPE in SMART_ACTIONS
        assert DesktopAction.SMART_RUN_COMMAND in SMART_ACTIONS
        assert DesktopAction.SMART_CREATE_NOTE in SMART_ACTIONS

    def test_smart_actions_size(self):
        assert len(SMART_ACTIONS) == 3

    def test_regular_actions_not_in_smart(self):
        assert DesktopAction.MOUSE_CLICK not in SMART_ACTIONS
        assert DesktopAction.SCREENSHOT not in SMART_ACTIONS


class TestNativeOnlyActions:
    """Tests for NATIVE_ONLY_ACTIONS frozenset."""

    def test_contains_ax_actions(self):
        assert DesktopAction.AX_QUERY in NATIVE_ONLY_ACTIONS
        assert DesktopAction.AX_CLICK in NATIVE_ONLY_ACTIONS
        assert DesktopAction.AX_SET_VALUE in NATIVE_ONLY_ACTIONS
        assert DesktopAction.AX_REQUEST_ACCESS in NATIVE_ONLY_ACTIONS

    def test_contains_clipboard(self):
        assert DesktopAction.CLIPBOARD_READ in NATIVE_ONLY_ACTIONS
        assert DesktopAction.CLIPBOARD_WRITE in NATIVE_ONLY_ACTIONS

    def test_contains_window_management(self):
        assert DesktopAction.LIST_WINDOWS in NATIVE_ONLY_ACTIONS
        assert DesktopAction.MOVE_WINDOW in NATIVE_ONLY_ACTIONS
        assert DesktopAction.RESIZE_WINDOW in NATIVE_ONLY_ACTIONS
        assert DesktopAction.MINIMIZE_WINDOW in NATIVE_ONLY_ACTIONS

    def test_mouse_drag_is_native_only(self):
        assert DesktopAction.MOUSE_DRAG in NATIVE_ONLY_ACTIONS

    def test_size(self):
        """Sanity check: at least the baseline 12 native-only actions exist."""
        assert len(NATIVE_ONLY_ACTIONS) >= 12


class TestNativeCapableActions:
    """Tests for NATIVE_CAPABLE_ACTIONS frozenset."""

    def test_is_superset_of_native_only(self):
        assert NATIVE_ONLY_ACTIONS.issubset(NATIVE_CAPABLE_ACTIONS)

    def test_contains_native_capable(self):
        assert DesktopAction.OPEN_APP in NATIVE_CAPABLE_ACTIONS
        assert DesktopAction.TYPE_TEXT in NATIVE_CAPABLE_ACTIONS
        assert DesktopAction.MOUSE_CLICK in NATIVE_CAPABLE_ACTIONS
        assert DesktopAction.FRONTMOST_APP in NATIVE_CAPABLE_ACTIONS

    def test_does_not_contain_non_native(self):
        # SCREENSHOT, RUN_MACRO, SMART_*, HEALTH_CHECK, SLEEP, OPEN_URL
        # are NOT in NATIVE_CAPABLE_ACTIONS
        assert DesktopAction.SCREENSHOT not in NATIVE_CAPABLE_ACTIONS
        assert DesktopAction.RUN_MACRO not in NATIVE_CAPABLE_ACTIONS
        assert DesktopAction.HEALTH_CHECK not in NATIVE_CAPABLE_ACTIONS
        assert DesktopAction.SLEEP not in NATIVE_CAPABLE_ACTIONS


# ═══════════════════════════════════════════════════════════════════════
# 2. AndroidAction Enum
# ═══════════════════════════════════════════════════════════════════════


class TestAndroidAction:
    """Tests for AndroidAction enum."""

    def test_member_count(self):
        """Sanity check: AndroidAction must have at least the baseline 27 members.
        New actions can be added without updating this test."""
        assert len(AndroidAction) >= 27

    @pytest.mark.parametrize(
        "member,value",
        [
            (AndroidAction.TAP, "tap"),
            (AndroidAction.LONG_PRESS, "long_press"),
            (AndroidAction.SWIPE, "swipe"),
            (AndroidAction.PINCH, "pinch"),
            (AndroidAction.TYPE_TEXT, "type_text"),
            (AndroidAction.PRESS_KEY, "press_key"),
            (AndroidAction.UI_DUMP, "ui_dump"),
            (AndroidAction.FIND_ELEMENT, "find_element"),
            (AndroidAction.FIND_AND_TAP, "find_and_tap"),
            (AndroidAction.FIND_AND_TYPE, "find_and_type"),
            (AndroidAction.WAIT_FOR_ELEMENT, "wait_for_element"),
            (AndroidAction.SCREENSHOT, "screenshot"),
            (AndroidAction.SCREEN_RECORD, "screen_record"),
            (AndroidAction.LAUNCH_APP, "launch_app"),
            (AndroidAction.STOP_APP, "stop_app"),
            (AndroidAction.CURRENT_APP, "current_app"),
            (AndroidAction.LIST_PACKAGES, "list_packages"),
            (AndroidAction.INSTALL_APK, "install_apk"),
            (AndroidAction.SHELL, "shell"),
            (AndroidAction.WAKE, "wake"),
            (AndroidAction.SLEEP_DEVICE, "sleep_device"),
            (AndroidAction.SCREEN_SIZE, "screen_size"),
            (AndroidAction.SCREEN_STATE, "screen_state"),
            (AndroidAction.PUSH_FILE, "push_file"),
            (AndroidAction.PULL_FILE, "pull_file"),
            (AndroidAction.HEALTH_CHECK, "health_check"),
            (AndroidAction.RUN_MACRO, "run_macro"),
        ],
    )
    def test_member_values(self, member, value):
        assert member.value == value

    def test_str_enum_semantics(self):
        assert AndroidAction.TAP == "tap"


# ═══════════════════════════════════════════════════════════════════════
# 3. VisionAction Enum
# ═══════════════════════════════════════════════════════════════════════


class TestVisionAction:
    """Tests for VisionAction enum — 9 members."""

    def test_member_count(self):
        assert len(VisionAction) == 9

    @pytest.mark.parametrize(
        "member,value",
        [
            (VisionAction.QUICK_SCAN, "quick_scan"),
            (VisionAction.SCAN, "scan"),
            (VisionAction.SCAN_WITH_VISION, "scan_with_vision"),
            (VisionAction.SCAN_WITH_OCR, "scan_with_ocr"),
            (VisionAction.FIND_AND_CLICK, "find_and_click"),
            (VisionAction.TYPE_INTO, "type_into"),
            (VisionAction.WAIT_FOR, "wait_for"),
            (VisionAction.READ_SCREEN_TEXT, "read_screen_text"),
            (VisionAction.DIFF, "diff"),
        ],
    )
    def test_member_values(self, member, value):
        assert member.value == value

    def test_str_enum_semantics(self):
        assert VisionAction.QUICK_SCAN == "quick_scan"


# ═══════════════════════════════════════════════════════════════════════
# 4. ScreenshotQuality Enum
# ═══════════════════════════════════════════════════════════════════════


class TestScreenshotQuality:
    """Tests for ScreenshotQuality enum."""

    def test_member_count(self):
        assert len(ScreenshotQuality) == 3

    def test_values(self):
        assert ScreenshotQuality.FULL == "full"
        assert ScreenshotQuality.FAST == "fast"
        assert ScreenshotQuality.THUMBNAIL == "thumbnail"


# ═══════════════════════════════════════════════════════════════════════
# 5. OperatorPlatform + ActionCategory
# ═══════════════════════════════════════════════════════════════════════


class TestOperatorPlatform:
    """Tests for OperatorPlatform enum."""

    def test_platforms_exist(self):
        assert OperatorPlatform.MACOS == "macos"
        assert OperatorPlatform.ANDROID == "android"
        assert OperatorPlatform.LINUX == "linux"
        assert OperatorPlatform.WINDOWS == "windows"


class TestActionCategory:
    """Tests for ActionCategory enum."""

    def test_categories(self):
        cats = {c.value for c in ActionCategory}
        expected = {
            "app_control", "keyboard", "mouse", "screenshot", "clipboard",
            "window", "accessibility", "macro", "smart_input", "device",
            "file_transfer",
        }
        assert cats == expected


# ═══════════════════════════════════════════════════════════════════════
# 6. OperatorError
# ═══════════════════════════════════════════════════════════════════════


class TestOperatorError:
    """Tests for OperatorError exception."""

    def test_basic_construction(self):
        err = OperatorError("something went wrong")
        assert str(err) == "something went wrong"
        assert err.action == ""
        assert err.recoverable is True
        assert err.suggestion == ""

    def test_full_construction(self):
        err = OperatorError(
            "AX error",
            action="ax_click",
            recoverable=False,
            suggestion="Grant accessibility",
        )
        assert err.action == "ax_click"
        assert err.recoverable is False
        assert err.suggestion == "Grant accessibility"

    def test_is_runtime_error(self):
        err = OperatorError("test")
        assert isinstance(err, RuntimeError)

    def test_to_dict(self):
        err = OperatorError("fail", action="screenshot", recoverable=True)
        d = err.to_dict()
        assert d["error"] == "fail"
        assert d["action"] == "screenshot"
        assert d["recoverable"] is True


# ═══════════════════════════════════════════════════════════════════════
# 7. MacroAbortToken
# ═══════════════════════════════════════════════════════════════════════


class TestMacroAbortToken:
    """Tests for thread-safe MacroAbortToken."""

    def test_initial_state(self):
        token = MacroAbortToken()
        assert token.is_aborted is False
        assert token.reason == ""

    def test_abort_sets_state(self):
        token = MacroAbortToken()
        token.abort("user cancelled")
        assert token.is_aborted is True
        assert token.reason == "user cancelled"

    def test_abort_default_reason(self):
        token = MacroAbortToken()
        token.abort()
        assert token.is_aborted is True
        assert token.reason == "Aborted"

    def test_thread_safety(self):
        """Abort from another thread should be visible."""
        token = MacroAbortToken()
        errors = []

        def abort_from_thread():
            try:
                time.sleep(0.01)
                token.abort("from thread")
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=abort_from_thread)
        t.start()
        t.join()
        assert not errors
        assert token.is_aborted is True
        assert token.reason == "from thread"

    def test_concurrent_aborts(self):
        """Multiple threads aborting simultaneously should not crash."""
        token = MacroAbortToken()
        errors = []

        def abort_repeatedly():
            try:
                for i in range(100):
                    token.abort(f"reason_{i}")
                    _ = token.is_aborted
                    _ = token.reason
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=abort_repeatedly) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert token.is_aborted is True


# ═══════════════════════════════════════════════════════════════════════
# 8. BaseOperator ABC Enforcement
# ═══════════════════════════════════════════════════════════════════════


class TestBaseOperatorABC:
    """Tests for BaseOperator abstract interface enforcement."""

    def test_cannot_instantiate_directly(self):
        """BaseOperator is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseOperator()

    def test_must_implement_abstract_methods(self):
        """Subclass without all abstract methods should fail."""
        class IncompleteOperator(BaseOperator):
            @property
            def platform(self):
                return OperatorPlatform.MACOS

        with pytest.raises(TypeError):
            IncompleteOperator()

    def test_complete_subclass_works(self):
        """Subclass with all abstract methods should instantiate."""
        class ConcreteOperator(BaseOperator):
            @property
            def platform(self):
                return OperatorPlatform.LINUX

            @property
            def available(self):
                return True

            @property
            def supported_actions(self):
                return {"test"}

            def execute(self, action, params):
                return {"ok": True, "action": action}

            def health_check(self, params=None):
                return {"ok": True}

        op = ConcreteOperator()
        assert op.platform == OperatorPlatform.LINUX
        assert op.available is True


# ═══════════════════════════════════════════════════════════════════════
# 9. BaseOperator.execute_macro
# ═══════════════════════════════════════════════════════════════════════


class _TestableOperator(BaseOperator):
    """Concrete operator for testing macro execution."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._results = {}  # action -> result dict

    @property
    def platform(self):
        return OperatorPlatform.MACOS

    @property
    def available(self):
        return True

    @property
    def supported_actions(self):
        return {"type_text", "press_key", "screenshot", "click", "fail_action"}

    def execute(self, action, params):
        if action == "fail_action":
            raise OperatorError("intentional failure", action=action)
        return self._results.get(action, {"ok": True, "action": action})

    def health_check(self, params=None):
        return {"ok": True}


class TestExecuteMacro:
    """Tests for BaseOperator.execute_macro."""

    def test_simple_macro(self):
        op = _TestableOperator()
        steps = [
            {"action": "type_text", "text": "hello"},
            {"action": "press_key", "key": "enter"},
        ]
        result = op.execute_macro(steps)
        assert result["completed_steps"] == 2
        assert len(result["results"]) == 2

    def test_empty_steps_raises(self):
        op = _TestableOperator()
        with pytest.raises(OperatorError, match="non-empty"):
            op.execute_macro([])

    def test_non_list_steps_raises(self):
        op = _TestableOperator()
        with pytest.raises(OperatorError, match="non-empty"):
            op.execute_macro("not_a_list")

    def test_max_steps_enforced(self):
        op = _TestableOperator()
        steps = [{"action": "click"}] * 51
        with pytest.raises(OperatorError, match="limited to"):
            op.execute_macro(steps)

    def test_stop_on_error(self):
        op = _TestableOperator()
        steps = [
            {"action": "type_text"},
            {"action": "fail_action"},
            {"action": "press_key"},
        ]
        result = op.execute_macro(steps, stop_on_error=True)
        assert result["completed_steps"] == 1
        assert result["failed_step"] == 2

    def test_continue_on_error(self):
        op = _TestableOperator()
        steps = [
            {"action": "type_text"},
            {"action": "fail_action"},
            {"action": "press_key"},
        ]
        result = op.execute_macro(steps, stop_on_error=False)
        assert result["completed_steps"] == 3
        # Second result should contain error
        assert "error" in result["results"][1]

    def test_abort_token(self):
        op = _TestableOperator()
        token = MacroAbortToken()
        token.abort("cancelled")

        steps = [
            {"action": "type_text"},
            {"action": "press_key"},
        ]
        result = op.execute_macro(steps, abort_token=token)
        assert result["aborted"] is True
        assert result["abort_reason"] == "cancelled"
        assert result["completed_steps"] == 0

    def test_missing_action_raises(self):
        op = _TestableOperator()
        with pytest.raises(OperatorError, match="missing step.action"):
            op.execute_macro([{"no_action": True}])

    def test_invalid_step_type_raises(self):
        op = _TestableOperator()
        with pytest.raises(OperatorError, match="invalid macro step"):
            op.execute_macro(["not_a_dict"])

    def test_nesting_depth_limit(self):
        """Nested macros should respect max depth."""
        op = _TestableOperator()
        # Manually set depth to max
        op._macro_depth = 3
        steps = [{"action": "click"}]
        with pytest.raises(OperatorError, match="nesting too deep"):
            op.execute_macro(steps)
        # Reset depth
        op._macro_depth = 0

    def test_delay_ms_bounded(self):
        op = _TestableOperator()
        steps = [{"action": "click"}]
        # Negative delay should be clamped to 0
        result = op.execute_macro(steps, delay_ms=-100)
        assert result["completed_steps"] == 1

    def test_macro_depth_decremented_on_error(self):
        """Macro depth should be decremented even if execute raises."""
        op = _TestableOperator()
        assert op._macro_depth == 0
        steps = [{"action": "fail_action"}]
        result = op.execute_macro(steps, stop_on_error=True)
        assert op._macro_depth == 0  # properly decremented


# ═══════════════════════════════════════════════════════════════════════
# 10. BaseOperator.telemetry + capabilities
# ═══════════════════════════════════════════════════════════════════════


class TestBaseOperatorTelemetry:
    """Tests for telemetry recording and capability discovery."""

    def test_record_and_retrieve_telemetry(self):
        op = _TestableOperator()
        op._record_telemetry("click", 10.0, True)
        op._record_telemetry("click", 20.0, False)
        tel = op.telemetry()
        assert "click" in tel
        assert tel["click"]["calls"] == 2

    def test_capabilities(self):
        op = _TestableOperator()
        caps = op.capabilities()
        assert caps["platform"] == "macos"
        assert caps["available"] is True
        assert "type_text" in caps["actions"]
        assert caps["max_macro_steps"] == 50
        assert caps["max_macro_depth"] == 3

    def test_supports(self):
        op = _TestableOperator()
        assert op.supports("type_text") is True
        assert op.supports("nonexistent") is False

    def test_telemetry_stats_accuracy(self):
        op = _TestableOperator()
        op._record_telemetry("act", 10.0, True)
        op._record_telemetry("act", 30.0, True)
        op._record_telemetry("act", 20.0, False)
        tel = op.telemetry()
        assert tel["act"]["calls"] == 3
        assert tel["act"]["min_ms"] == pytest.approx(10.0, abs=0.1)
        assert tel["act"]["max_ms"] == pytest.approx(30.0, abs=0.1)
        assert tel["act"]["avg_ms"] == pytest.approx(20.0, abs=0.1)


# ═══════════════════════════════════════════════════════════════════════
# 11. MockDesktopOperator
# ═══════════════════════════════════════════════════════════════════════


class TestMockDesktopOperator:
    """Tests for MockDesktopOperator."""

    def test_default_response(self):
        op = MockDesktopOperator()
        result = op.execute("screenshot", {})
        assert result["ok"] is True
        assert result["action"] == "screenshot"
        assert "path" in result

    def test_custom_response(self):
        op = MockDesktopOperator()
        op.set_response("frontmost_app", {"app_name": "Safari"})
        result = op.execute("frontmost_app", {})
        assert result["app_name"] == "Safari"

    def test_call_log(self):
        op = MockDesktopOperator()
        op.execute("screenshot", {"include_base64": True})
        assert len(op.call_log) == 1
        assert op.call_log[0] == ("screenshot", {"include_base64": True})

    def test_failure_injection(self):
        op = MockDesktopOperator()
        op.set_failure("open_app", "App not found")
        with pytest.raises(OperatorError, match="App not found"):
            op.execute("open_app", {"app_name": "fake"})

    def test_clear_failure(self):
        op = MockDesktopOperator()
        op.set_failure("open_app", "fail")
        op.clear_failure("open_app")
        result = op.execute("open_app", {"app_name": "test"})
        assert result["ok"] is True

    def test_assert_called(self):
        op = MockDesktopOperator()
        op.execute("screenshot", {})
        op.assert_called("screenshot")
        op.assert_called("screenshot", times=1)

    def test_assert_called_wrong_times(self):
        op = MockDesktopOperator()
        op.execute("screenshot", {})
        with pytest.raises(AssertionError, match="called 1 times, expected 5"):
            op.assert_called("screenshot", times=5)

    def test_assert_not_called(self):
        op = MockDesktopOperator()
        op.assert_not_called("screenshot")

    def test_assert_not_called_but_was(self):
        op = MockDesktopOperator()
        op.execute("screenshot", {})
        with pytest.raises(AssertionError):
            op.assert_not_called("screenshot")

    def test_assert_called_never_called(self):
        op = MockDesktopOperator()
        with pytest.raises(AssertionError, match="never called"):
            op.assert_called("screenshot")

    def test_reset(self):
        op = MockDesktopOperator()
        op.execute("screenshot", {})
        op.set_response("test", {"custom": True})
        op.set_failure("fail", "boom")
        op.reset()
        assert len(op.call_log) == 0
        assert len(op._custom_responses) == 0
        assert len(op._failures) == 0

    def test_platform(self):
        op = MockDesktopOperator()
        assert op.platform == OperatorPlatform.MACOS

    def test_available(self):
        op = MockDesktopOperator()
        assert op.available is True

    def test_unavailable(self):
        op = MockDesktopOperator(available=False)
        assert op.available is False

    def test_supported_actions(self):
        op = MockDesktopOperator()
        actions = op.supported_actions
        assert "screenshot" in actions
        assert "mouse_click" in actions
        assert "run_macro" in actions

    def test_health_check(self):
        op = MockDesktopOperator()
        health = op.health_check()
        assert health["accessibility_trusted"] is True

    def test_empty_action_raises(self):
        op = MockDesktopOperator()
        with pytest.raises(OperatorError, match="action is required"):
            op.execute("", {})

    def test_run_macro_via_execute(self):
        op = MockDesktopOperator()
        result = op.execute("run_macro", {
            "steps": [
                {"action": "type_text", "text": "hello"},
                {"action": "press_key", "key": "enter"},
            ],
        })
        assert result["ok"] is True
        assert result["completed_steps"] == 2

    def test_run_macro_with_abort(self):
        op = MockDesktopOperator()
        token = MacroAbortToken()
        token.abort("test abort")
        result = op.execute("run_macro", {
            "steps": [{"action": "type_text"}],
            "_abort_token": token,
        })
        assert result["aborted"] is True

    def test_unknown_action_returns_generic(self):
        op = MockDesktopOperator()
        result = op.execute("totally_unknown", {})
        assert result["ok"] is True
        assert result["action"] == "totally_unknown"

    def test_multiple_calls_tracked(self):
        op = MockDesktopOperator()
        op.execute("screenshot", {})
        op.execute("screenshot", {})
        op.execute("type_text", {"text": "hi"})
        assert op._execution_count["screenshot"] == 2
        assert op._execution_count["type_text"] == 1


# ═══════════════════════════════════════════════════════════════════════
# 12. MockAndroidOperator
# ═══════════════════════════════════════════════════════════════════════


class TestMockAndroidOperator:
    """Tests for MockAndroidOperator."""

    def test_default_response(self):
        op = MockAndroidOperator()
        result = op.execute("tap", {"x": 100, "y": 200})
        assert result["ok"] is True
        assert result["action"] == "tap"

    def test_custom_response(self):
        op = MockAndroidOperator()
        op.set_response("current_app", {"package": "com.test"})
        result = op.execute("current_app", {})
        assert result["package"] == "com.test"

    def test_call_log(self):
        op = MockAndroidOperator()
        op.execute("screenshot", {})
        assert len(op.call_log) == 1
        assert op.call_log[0][0] == "screenshot"

    def test_failure_injection(self):
        op = MockAndroidOperator()
        op.set_failure("launch_app", "Package not found")
        with pytest.raises(OperatorError, match="Package not found"):
            op.execute("launch_app", {"package": "com.fake"})

    def test_platform(self):
        op = MockAndroidOperator()
        assert op.platform == OperatorPlatform.ANDROID

    def test_supported_actions(self):
        op = MockAndroidOperator()
        actions = op.supported_actions
        assert "tap" in actions
        assert "ui_dump" in actions
        assert "run_macro" in actions
        assert "install_apk" in actions

    def test_health_check(self):
        op = MockAndroidOperator()
        health = op.health_check()
        assert health["adb_ok"] is True
        assert health["connected"] is True

    def test_reset(self):
        op = MockAndroidOperator()
        op.execute("tap", {"x": 100, "y": 200})
        op.set_failure("fail", "boom")
        op.reset()
        assert len(op.call_log) == 0
        assert len(op._failures) == 0

    def test_assert_called(self):
        op = MockAndroidOperator()
        op.execute("tap", {})
        op.assert_called("tap")

    def test_assert_not_called(self):
        op = MockAndroidOperator()
        op.assert_not_called("tap")

    def test_run_macro_via_execute(self):
        op = MockAndroidOperator()
        result = op.execute("run_macro", {
            "steps": [
                {"action": "tap", "x": 100, "y": 200},
                {"action": "type_text", "text": "hello"},
            ],
        })
        assert result["ok"] is True
        assert result["completed_steps"] == 2


# ═══════════════════════════════════════════════════════════════════════
# 13. with_retry Decorator
# ═══════════════════════════════════════════════════════════════════════


class TestWithRetry:
    """Tests for with_retry decorator."""

    def test_success_on_first_try(self):
        call_count = 0

        @with_retry(max_attempts=3)
        def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = fn()
        assert result == "ok"
        assert call_count == 1

    def test_retries_on_transient_error(self):
        call_count = 0

        @with_retry(max_attempts=3, backoff_base=0.001, retryable_messages=("transient",))
        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient error")
            return "ok"

        result = fn()
        assert result == "ok"
        assert call_count == 3

    def test_no_retry_on_non_matching_message(self):
        @with_retry(
            max_attempts=3,
            retryable_messages=("kAXErrorCannotComplete",),
        )
        def fn():
            raise RuntimeError("permanent failure")

        with pytest.raises(RuntimeError, match="permanent"):
            fn()

    def test_raises_after_max_attempts(self):
        call_count = 0

        @with_retry(max_attempts=2, backoff_base=0.001, retryable_messages=("transient",))
        def fn():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("transient error")

        with pytest.raises(RuntimeError, match="transient"):
            fn()
        assert call_count == 2

    def test_exponential_backoff(self):
        """Delays should increase exponentially."""
        start = time.time()
        call_count = 0

        @with_retry(max_attempts=3, backoff_base=0.05, backoff_max=1.0, retryable_messages=("retry",))
        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("retry me")
            return "ok"

        fn()
        elapsed = time.time() - start
        # 0.05 + 0.10 = 0.15 minimum
        assert elapsed >= 0.1

    def test_backoff_max_cap(self):
        call_count = 0

        @with_retry(max_attempts=5, backoff_base=1.0, backoff_max=0.01, retryable_messages=("retry",))
        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("retry me")
            return "ok"

        start = time.time()
        fn()
        elapsed = time.time() - start
        # backoff_max caps at 0.01, so retries are fast
        assert elapsed < 1.0

    def test_preserves_function_name(self):
        @with_retry()
        def my_func():
            return 42

        assert my_func.__name__ == "my_func"

    def test_non_retryable_error_type(self):
        """ValueError is not in retryable_errors by default."""

        @with_retry(max_attempts=3)
        def fn():
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            fn()

    def test_custom_retryable_errors(self):
        call_count = 0

        @with_retry(
            max_attempts=3,
            backoff_base=0.001,
            retryable_errors=(ValueError,),
            retryable_messages=("temp",),
        )
        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temp error")
            return "done"

        result = fn()
        assert result == "done"
        assert call_count == 3


class TestAsyncWithRetry:
    """Tests for async_with_retry function."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        async def coro():
            return "ok"

        result = await async_with_retry(coro, max_attempts=3)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_transient(self):
        call_count = 0

        async def coro():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("timed out")
            return "ok"

        result = await async_with_retry(
            coro, max_attempts=3, backoff_base=0.001,
        )
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self):
        async def coro():
            raise RuntimeError("connection reset forever")

        with pytest.raises(RuntimeError, match="connection reset"):
            await async_with_retry(coro, max_attempts=2, backoff_base=0.001)

    @pytest.mark.asyncio
    async def test_no_retry_on_non_transient(self):
        async def coro():
            raise RuntimeError("permanent failure")

        with pytest.raises(RuntimeError, match="permanent"):
            await async_with_retry(
                coro,
                max_attempts=3,
                retryable_messages=("kAXErrorCannotComplete",),
            )


# ═══════════════════════════════════════════════════════════════════════
# 14. Re-export verification
# ═══════════════════════════════════════════════════════════════════════


class TestEnumReExports:
    """Verify that tools/enums.py re-exports operator enums."""

    def test_desktop_action_reexport(self):
        from jarvis.tools.enums import DesktopAction as DAFromTools
        from jarvis.operators.enums import DesktopAction as DAFromOps
        assert DAFromTools is DAFromOps

    def test_android_action_reexport(self):
        from jarvis.tools.enums import AndroidAction as AAFromTools
        from jarvis.operators.enums import AndroidAction as AAFromOps
        assert AAFromTools is AAFromOps

    def test_vision_action_reexport(self):
        from jarvis.tools.enums import VisionAction as VAFromTools
        from jarvis.operators.enums import VisionAction as VAFromOps
        assert VAFromTools is VAFromOps

    def test_frozenset_reexport(self):
        from jarvis.tools.enums import SMART_ACTIONS as SA_Tools
        from jarvis.operators.enums import SMART_ACTIONS as SA_Ops
        assert SA_Tools is SA_Ops
