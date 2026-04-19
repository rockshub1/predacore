"""
PredaCore Mock Operators — Deterministic mocks for testing.

These operators record all calls and return configurable mock responses.
They never touch real hardware, ADB, or macOS APIs — perfect for:
  - Unit tests
  - Integration tests
  - CI environments (no macOS/Android needed)
  - Demos and documentation

Usage:
    from predacore.operators.mock import MockDesktopOperator, MockAndroidOperator

    desktop = MockDesktopOperator()
    result = desktop.execute("screenshot", {"include_base64": True})
    assert result["ok"] is True
    assert desktop.call_log[-1] == ("screenshot", {"include_base64": True})

    # Custom responses
    desktop.set_response("frontmost_app", {"app_name": "Safari"})
    result = desktop.execute("frontmost_app", {})
    assert result["app_name"] == "Safari"

    # Simulate failures
    desktop.set_failure("open_app", "App not found")
    try:
        desktop.execute("open_app", {"app_name": "fake"})
    except OperatorError:
        pass
"""
from __future__ import annotations

import logging
import time
from typing import Any

from .base import (
    BaseOperator,
    OperatorError,
    OperatorPlatform,
)

logger = logging.getLogger(__name__)


class MockDesktopOperator(BaseOperator):
    """Deterministic mock of MacDesktopOperator for testing.

    Records all calls. Returns configurable mock responses.
    Never touches real macOS APIs.
    """

    # Default mock responses for each action
    _DEFAULT_RESPONSES: dict[str, dict[str, Any]] = {
        "open_app": {"app_name": "MockApp", "new_instance": False},
        "focus_app": {"app_name": "MockApp"},
        "open_url": {"url": "https://example.com"},
        "type_text": {"typed_chars": 10, "app_name": None},
        "press_key": {"key": "enter", "key_code": None, "modifiers": []},
        "mouse_move": {"x": 100, "y": 200, "duration_seconds": 0.0},
        "mouse_click": {"x": 100, "y": 200, "clicks": 1, "button": "left"},
        "mouse_double_click": {"x": 100, "y": 200, "clicks": 2, "button": "left"},
        "mouse_right_click": {"x": 100, "y": 200, "clicks": 1, "button": "right"},
        "mouse_scroll": {"amount": 3},
        "get_mouse_position": {"x": 500, "y": 300},
        "screenshot": {"path": "/tmp/mock_screenshot.png", "size_bytes": 12345, "quality": "full"},
        "frontmost_app": {"app_name": "Terminal"},
        "health_check": {
            "platform": "Darwin",
            "accessibility_trusted": True,
            "osascript": {"ok": True, "error": None},
            "pyautogui": {"ok": True, "error": None},
            "native_backend": {"enabled": True, "fallback_enabled": True},
            "smart_input": {"enabled": True},
            "hints": [],
        },
        "sleep": {"seconds": 0.1},
        "smart_type": {"method": "mock", "verified": True, "typed_chars": 10},
        "smart_run_command": {"method": "mock", "command": "echo test"},
        "smart_create_note": {"method": "mock", "title": "Test", "body": "content"},
        "ax_query": {"elements": [], "total": 0},
        "ax_click": {"clicked": True, "element": "MockButton"},
        "ax_set_value": {"set": True, "value": "mock"},
        "clipboard_read": {"text": "mock clipboard content"},
        "clipboard_write": {"written": True},
        "list_windows": {"windows": [{"title": "Mock Window", "app": "MockApp", "id": 1}]},
        "move_window": {"moved": True},
        "resize_window": {"resized": True},
        "minimize_window": {"minimized": True},
        "list_monitors": {"monitors": [{"id": 1, "width": 1920, "height": 1080, "primary": True}]},
        "mouse_drag": {"from_x": 0, "from_y": 0, "to_x": 100, "to_y": 100},
    }

    def __init__(
        self,
        *,
        available: bool = True,
        log: logging.Logger | None = None,
    ) -> None:
        super().__init__(log=log)
        self._available = available
        self.call_log: list[tuple[str, dict[str, Any]]] = []
        self._custom_responses: dict[str, dict[str, Any]] = {}
        self._failures: dict[str, str] = {}  # action → error message
        self._execution_count: dict[str, int] = {}

    @property
    def platform(self) -> OperatorPlatform:
        """Return the mock platform identifier."""
        return OperatorPlatform.MACOS

    @property
    def available(self) -> bool:
        """Return True — mock is always available."""
        return self._available

    @property
    def supported_actions(self) -> set[str]:
        """Return the set of mock-supported actions."""
        return set(self._DEFAULT_RESPONSES.keys()) | {"run_macro"}

    def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Record the call and return mock response."""
        start = time.time()
        action = str(action or "").strip().lower()
        if not action:
            raise OperatorError("action is required", action="")

        # Record
        self.call_log.append((action, dict(params)))
        self._execution_count[action] = self._execution_count.get(action, 0) + 1

        # Handle macros via base class
        if action == "run_macro":
            steps = params.get("steps")
            abort_token = params.pop("_abort_token", None)
            result = self.execute_macro(
                steps,
                stop_on_error=bool(params.get("stop_on_error", True)),
                delay_ms=0,  # No delays in tests
                abort_token=abort_token,
            )
            elapsed_ms = int((time.time() - start) * 1000)
            result.update({"ok": True, "action": action, "elapsed_ms": elapsed_ms})
            return result

        # Check for configured failures
        if action in self._failures:
            raise OperatorError(
                self._failures[action],
                action=action,
                recoverable=True,
            )

        # Get response (custom > default > generic)
        response = (
            self._custom_responses.get(action)
            or self._DEFAULT_RESPONSES.get(action)
            or {}
        )
        result = dict(response)  # Copy to avoid mutation
        elapsed_ms = int((time.time() - start) * 1000)
        result.update({"ok": True, "action": action, "elapsed_ms": elapsed_ms})
        return result

    def health_check(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return mock health check status."""
        return dict(self._DEFAULT_RESPONSES["health_check"])

    # ── Test Configuration API ───────────────────────────────────

    def set_response(self, action: str, response: dict[str, Any]) -> None:
        """Set a custom mock response for a specific action."""
        self._custom_responses[action] = response

    def set_failure(self, action: str, error_message: str) -> None:
        """Configure an action to raise OperatorError when called."""
        self._failures[action] = error_message

    def clear_failure(self, action: str) -> None:
        """Remove a configured failure."""
        self._failures.pop(action, None)

    def reset(self) -> None:
        """Reset all state — call log, custom responses, failures."""
        self.call_log.clear()
        self._custom_responses.clear()
        self._failures.clear()
        self._execution_count.clear()

    def assert_called(self, action: str, times: int | None = None) -> None:
        """Assert that an action was called (optionally N times)."""
        count = self._execution_count.get(action, 0)
        if count == 0:
            raise AssertionError(f"Action '{action}' was never called")
        if times is not None and count != times:
            raise AssertionError(
                f"Action '{action}' called {count} times, expected {times}"
            )

    def assert_not_called(self, action: str) -> None:
        """Assert that an action was never called."""
        count = self._execution_count.get(action, 0)
        if count > 0:
            raise AssertionError(
                f"Action '{action}' was called {count} times, expected 0"
            )


class MockAndroidOperator(BaseOperator):
    """Deterministic mock of AndroidOperator for testing.

    Same pattern as MockDesktopOperator — records calls, returns mocks.
    """

    _DEFAULT_RESPONSES: dict[str, dict[str, Any]] = {
        "health_check": {
            "adb_path": "/usr/local/bin/adb",
            "adb_ok": True,
            "selected_serial": "emulator-5554",
            "devices": [{"serial": "emulator-5554", "state": "device", "model": "Pixel_6"}],
            "connected": True,
        },
        "ui_dump": {
            "total_elements": 3,
            "returned": 3,
            "elements": [
                {"resource_id": "com.app:id/button", "text": "Click Me", "bounds": [100, 200, 300, 250], "center": [200, 225], "clickable": True, "enabled": True, "scrollable": False},
            ],
        },
        "tap": {"x": 200, "y": 225},
        "long_press": {"x": 200, "y": 225, "duration_ms": 1000},
        "swipe": {"x1": 100, "y1": 500, "x2": 100, "y2": 200, "duration_ms": 300},
        "pinch": {"center_x": 540, "center_y": 960, "direction": "in", "distance": 200},
        "type_text": {"typed_chars": 5},
        "press_key": {"key": "home", "keycode": "KEYCODE_HOME"},
        "screenshot": {"path": "/tmp/mock_android.png", "size_bytes": 8000},
        "launch_app": {"package": "com.example.app", "activity": "(launcher)"},
        "stop_app": {"package": "com.example.app"},
        "current_app": {"package": "com.example.app", "activity": ".MainActivity"},
        "list_packages": {"packages": ["com.example.app", "com.android.settings"], "count": 2},
        "shell": {"command": "echo test", "output": "test"},
        "find_element": {"found": 1, "elements": [{"text": "Click Me", "center": [200, 225]}]},
        "find_and_tap": {"tapped": {"text": "Click Me"}, "x": 200, "y": 225},
        "find_and_type": {"field": {"text": ""}, "typed": "test"},
        "wait_for_element": {"found": True, "element": {"text": "Click Me"}},
        "screen_size": {"width": 1080, "height": 2400},
        "screen_state": {"screen_on": True, "raw": "Display Power: state=ON"},
        "wake": {"woke": True},
        "sleep_device": {"sleeping": True},
        "push_file": {"local": "/tmp/test.txt", "remote": "/sdcard/test.txt", "output": "pushed"},
        "pull_file": {"remote": "/sdcard/test.txt", "local": "/tmp/test.txt", "size_bytes": 100, "output": "pulled"},
        "screen_record": {"path": "/tmp/recording.mp4", "duration_seconds": 5, "size_bytes": 50000},
    }

    def __init__(
        self,
        *,
        available: bool = True,
        log: logging.Logger | None = None,
    ) -> None:
        super().__init__(log=log)
        self._available = available
        self.call_log: list[tuple[str, dict[str, Any]]] = []
        self._custom_responses: dict[str, dict[str, Any]] = {}
        self._failures: dict[str, str] = {}
        self._execution_count: dict[str, int] = {}

    @property
    def platform(self) -> OperatorPlatform:
        """Return the mock Android platform identifier."""
        return OperatorPlatform.ANDROID

    @property
    def available(self) -> bool:
        """Return True — mock Android is always available."""
        return self._available

    @property
    def supported_actions(self) -> set[str]:
        """Return the set of mock-supported Android actions."""
        return set(self._DEFAULT_RESPONSES.keys()) | {"run_macro", "install_apk"}

    def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a mock Android action and return canned result."""
        start = time.time()
        action = str(action or "").strip().lower()
        if not action:
            raise OperatorError("action is required", action="")

        self.call_log.append((action, dict(params)))
        self._execution_count[action] = self._execution_count.get(action, 0) + 1

        if action == "run_macro":
            steps = params.get("steps")
            abort_token = params.pop("_abort_token", None)
            result = self.execute_macro(
                steps,
                stop_on_error=bool(params.get("stop_on_error", True)),
                delay_ms=0,
                abort_token=abort_token,
            )
            elapsed_ms = int((time.time() - start) * 1000)
            result.update({"ok": True, "action": action, "elapsed_ms": elapsed_ms})
            return result

        if action in self._failures:
            raise OperatorError(self._failures[action], action=action)

        response = (
            self._custom_responses.get(action)
            or self._DEFAULT_RESPONSES.get(action)
            or {}
        )
        result = dict(response)
        elapsed_ms = int((time.time() - start) * 1000)
        result.update({"ok": True, "action": action, "elapsed_ms": elapsed_ms})
        return result

    def health_check(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return mock Android health status."""
        return dict(self._DEFAULT_RESPONSES["health_check"])

    def set_response(self, action: str, response: dict[str, Any]) -> None:
        """Set a canned response for a specific action."""
        self._custom_responses[action] = response

    def set_failure(self, action: str, error_message: str) -> None:
        """Configure the mock to raise an error for a specific action."""
        self._failures[action] = error_message

    def clear_failure(self, action: str) -> None:
        """Clear any configured failure for a specific action."""
        self._failures.pop(action, None)

    def reset(self) -> None:
        """Reset all canned responses, failures, and call history."""
        self.call_log.clear()
        self._custom_responses.clear()
        self._failures.clear()
        self._execution_count.clear()

    def assert_called(self, action: str, times: int | None = None) -> None:
        """Assert that a specific action was called at least once."""
        count = self._execution_count.get(action, 0)
        if count == 0:
            raise AssertionError(f"Action '{action}' was never called")
        if times is not None and count != times:
            raise AssertionError(f"Action '{action}' called {count} times, expected {times}")

    def assert_not_called(self, action: str) -> None:
        """Assert that a specific action was never called."""
        count = self._execution_count.get(action, 0)
        if count > 0:
            raise AssertionError(f"Action '{action}' was called {count} times, expected 0")
