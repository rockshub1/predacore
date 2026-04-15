"""
Real macOS desktop integration tests for native PyObjC backend.

NO MOCKS. Exercises real:
  - App launch/focus/quit
  - Keyboard input (CGEvent)
  - Mouse operations (CGEvent)
  - AX queries
  - Clipboard (NSPasteboard)
  - Window management
  - System info
  - Spotlight search
  - Screenshots

Auto-skips if:
  - Not macOS
  - PyObjC not installed
  - Accessibility not granted

Usage:
    cd project_prometheus/src
    python -m pytest jarvis/tests/test_desktop_native.py -v
"""
from __future__ import annotations

import asyncio
import os
import platform
import time

import pytest

IS_MACOS = platform.system() == "Darwin"

pytestmark = pytest.mark.skipif(not IS_MACOS, reason="macOS only")

# Check if PyObjC is available
try:
    import AppKit
    import Quartz
    HAS_PYOBJC = True
except ImportError:
    HAS_PYOBJC = False

if not HAS_PYOBJC:
    pytestmark = pytest.mark.skip(reason="PyObjC not installed")


@pytest.fixture(scope="module")
def native_service():
    """Create a MacDesktopNativeService instance."""
    from jarvis.operators.native_service import MacDesktopNativeService
    svc = MacDesktopNativeService()
    if not svc.available:
        pytest.skip(f"Native service unavailable: {svc.init_error}")
    return svc


@pytest.fixture(scope="module")
def desktop_operator():
    """Create a MacDesktopOperator with native-only backend."""
    from jarvis.operators.desktop import MacDesktopOperator
    # Force native-only (no legacy fallback)
    os.environ["JARVIS_DESKTOP_BACKEND"] = "native"
    op = MacDesktopOperator()
    if not op.available:
        pytest.skip("Desktop operator not available on this platform")
    return op


# ═══════════════════════════════════════════════════════════════════════
# Health & Connection
# ═══════════════════════════════════════════════════════════════════════


class TestHealth:
    def test_native_service_available(self, native_service):
        assert native_service.available

    def test_health_check(self, native_service):
        result = native_service.health_check()
        assert result["backend"] == "pyobjc_native"
        assert result["available"]

    def test_desktop_operator_health(self, desktop_operator):
        result = desktop_operator.health_check()
        assert result["platform"] == "Darwin"


# ═══════════════════════════════════════════════════════════════════════
# App Control
# ═══════════════════════════════════════════════════════════════════════


class TestAppControl:
    def test_frontmost_app(self, native_service):
        result = native_service.execute("frontmost_app", {})
        assert "app_name" in result
        assert len(result["app_name"]) > 0

    def test_list_apps(self, native_service):
        result = native_service.execute("list_apps", {})
        assert result["count"] > 0
        names = [a["name"] for a in result["apps"]]
        assert "Finder" in names  # Finder is always running

    def test_open_and_focus_app(self, native_service):
        # Open TextEdit (lightweight, always available)
        result = native_service.execute("open_app", {"app_name": "TextEdit"})
        assert "app_name" in result
        # Wait for app to register in NSWorkspace (cold start can be 5s+)
        found = False
        for _ in range(15):
            time.sleep(0.5)
            apps = native_service.execute("list_apps", {})
            if any(a["name"] == "TextEdit" for a in apps.get("apps", [])):
                found = True
                break
        if not found:
            pytest.skip("TextEdit failed to launch in time")
        # Focus it with launch_if_missing fallback
        result = native_service.execute("focus_app", {"app_name": "TextEdit", "launch_if_missing": True})
        assert "app_name" in result
        # Clean up
        time.sleep(0.5)
        try:
            native_service.execute("force_quit_app", {"app_name": "TextEdit"})
        except Exception:
            pass


class TestForceQuit:
    def test_force_quit_nonexistent(self, native_service):
        from jarvis.operators.native_service import DesktopNativeError
        with pytest.raises(DesktopNativeError, match="not found"):
            native_service.execute("force_quit_app", {"app_name": "NonexistentApp12345"})


# ═══════════════════════════════════════════════════════════════════════
# Clipboard
# ═══════════════════════════════════════════════════════════════════════


class TestClipboard:
    def test_clipboard_write_and_read(self, native_service):
        test_text = f"JARVIS desktop test {time.time()}"
        native_service.execute("clipboard_write", {"text": test_text})
        result = native_service.execute("clipboard_read", {})
        assert result["text"] == test_text


# ═══════════════════════════════════════════════════════════════════════
# Mouse & Keyboard
# ═══════════════════════════════════════════════════════════════════════


class TestMouse:
    def test_get_mouse_position(self, native_service):
        result = native_service.execute("get_mouse_position", {})
        assert "x" in result
        assert "y" in result
        assert isinstance(result["x"], int)

    def test_mouse_move(self, native_service):
        result = native_service.execute("mouse_move", {"x": 500, "y": 400})
        assert result["x"] == 500
        assert result["y"] == 400

    def test_mouse_click(self, native_service):
        # Click at a safe position (center of screen)
        result = native_service.execute("mouse_click", {"x": 500, "y": 400, "button": "left", "clicks": 1})
        assert result["clicks"] == 1


class TestKeyboard:
    def test_press_key(self, native_service):
        result = native_service.execute("press_key", {"key": "escape"})
        assert result["key"] == "escape"


# ═══════════════════════════════════════════════════════════════════════
# Windows
# ═══════════════════════════════════════════════════════════════════════


class TestWindows:
    def test_list_windows(self, native_service):
        result = native_service.execute("list_windows", {"all_apps": True})
        assert "windows" in result
        # At least one window should exist (this test window, Finder, etc.)

    def test_list_monitors(self, native_service):
        result = native_service.execute("list_monitors", {})
        assert result["count"] >= 1
        assert result["monitors"][0]["is_main"]


# ═══════════════════════════════════════════════════════════════════════
# Screenshots
# ═══════════════════════════════════════════════════════════════════════


class TestScreenshot:
    def test_screenshot(self, desktop_operator):
        result = desktop_operator.execute("screenshot", {})
        assert result["ok"]
        assert result["size_bytes"] > 0
        path = result.get("path", "")
        if path:
            from pathlib import Path
            Path(path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# System Info & Control
# ═══════════════════════════════════════════════════════════════════════


class TestSystemInfo:
    def test_get_system_info(self, native_service):
        result = native_service.execute("get_system_info", {})
        # Battery info may not be available on desktop Macs
        assert "screen_count" in result
        assert result["screen_count"] >= 1

    def test_get_dark_mode(self, native_service):
        result = native_service.execute("get_dark_mode", {})
        assert "dark_mode" in result
        assert isinstance(result["dark_mode"], bool)


class TestSpotlightSearch:
    def test_spotlight_search(self, native_service):
        result = native_service.execute("spotlight_search", {"query": "kind:app TextEdit", "limit": 5})
        assert "files" in result
        assert result["count"] >= 1


class TestRunningProcesses:
    def test_get_running_processes(self, native_service):
        result = native_service.execute("get_running_processes", {"limit": 10, "sort_by": "cpu"})
        assert result["count"] > 0
        proc = result["processes"][0]
        assert "pid" in proc
        assert "cpu_percent" in proc
        assert "command" in proc


# ═══════════════════════════════════════════════════════════════════════
# AX (Accessibility)
# ═══════════════════════════════════════════════════════════════════════


class TestAccessibility:
    def test_ax_request_access(self, native_service):
        result = native_service.execute("ax_request_access", {"prompt": False})
        assert "accessibility_trusted" in result

    def test_ax_query_focused_window(self, native_service):
        # This may fail if AX permissions aren't granted or no window is focused
        try:
            result = native_service.execute("ax_query", {
                "target": "focused_window",
                "max_depth": 1,
                "max_children": 10,
            })
            assert "snapshot" in result
        except Exception as e:
            err = str(e).lower()
            if "trust" in err or "accessibility" in err or "kaxerror" in err or "focused" in err:
                pytest.skip(f"AX unavailable: {e}")
            raise


# ═══════════════════════════════════════════════════════════════════════
# Volume
# ═══════════════════════════════════════════════════════════════════════


class TestVolume:
    def test_get_and_set_volume(self, native_service):
        # Read current volume
        original = native_service.execute("get_volume", {})
        assert "volume" in original
        # Set to a known value
        native_service.execute("set_volume", {"level": 25})
        result = native_service.execute("get_volume", {})
        assert result["volume"] == 25
        # Restore original
        native_service.execute("set_volume", {"level": original["volume"]})


# ═══════════════════════════════════════════════════════════════════════
# Desktop Operator (integration — dispatch through MacDesktopOperator)
# ═══════════════════════════════════════════════════════════════════════


class TestDesktopOperatorDispatch:
    def test_dispatch_frontmost_app(self, desktop_operator):
        result = desktop_operator.execute("frontmost_app", {})
        assert result["ok"]
        assert "app_name" in result

    def test_dispatch_list_apps(self, desktop_operator):
        result = desktop_operator.execute("list_apps", {})
        assert result["ok"]
        assert result["count"] > 0

    def test_dispatch_get_system_info(self, desktop_operator):
        result = desktop_operator.execute("get_system_info", {})
        assert result["ok"]

    def test_dispatch_spotlight_search(self, desktop_operator):
        result = desktop_operator.execute("spotlight_search", {"query": "kind:folder Desktop", "limit": 3})
        assert result["ok"]

    def test_dispatch_get_running_processes(self, desktop_operator):
        result = desktop_operator.execute("get_running_processes", {"limit": 5})
        assert result["ok"]
        assert result["count"] > 0
