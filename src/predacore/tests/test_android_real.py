"""
Real ADB integration tests for AndroidOperator.

NO MOCKS. Exercises real ADB commands against a connected Android device.
Auto-skips if no device is connected.

Usage:
    # Connect an Android device via USB or start an emulator, then:
    cd project_prometheus/src
    python -m pytest predacore/tests/test_android_real.py -v
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time

import pytest

# ── Skip if ADB not available or no device ───────────────────────────

ADB_PATH = shutil.which("adb")

def _device_connected() -> bool:
    if not ADB_PATH:
        return False
    try:
        out = subprocess.run([ADB_PATH, "devices"], capture_output=True, text=True, timeout=5)
        lines = [l.strip() for l in out.stdout.strip().split("\n")[1:] if l.strip()]
        return any("device" in l for l in lines)
    except (OSError, subprocess.SubprocessError):
        return False

HAS_DEVICE = _device_connected()

pytestmark = [
    pytest.mark.skipif(ADB_PATH is None, reason="ADB not installed"),
    pytest.mark.skipif(not HAS_DEVICE, reason="No Android device connected"),
]


@pytest.fixture(scope="module")
def operator():
    from predacore.operators.android import AndroidOperator
    op = AndroidOperator()
    if not op.available:
        pytest.skip("AndroidOperator not available")
    return op


# ═══════════════════════════════════════════════════════════════════════
# Health & Connection
# ═══════════════════════════════════════════════════════════════════════


class TestHealth:
    def test_health_check(self, operator):
        result = operator.execute("health_check", {})
        assert result["ok"]
        assert result["adb_ok"]
        assert result["connected"]

    def test_screen_size(self, operator):
        result = operator.execute("screen_size", {})
        assert result["ok"]
        assert result["width"] > 0
        assert result["height"] > 0

    def test_screen_state(self, operator):
        result = operator.execute("screen_state", {})
        assert result["ok"]
        assert "screen_on" in result


# ═══════════════════════════════════════════════════════════════════════
# Device Info (new)
# ═══════════════════════════════════════════════════════════════════════


class TestDeviceInfo:
    def test_get_device_info(self, operator):
        result = operator.execute("get_device_info", {})
        assert result["ok"]
        assert "model" in result
        assert "android_version" in result
        assert len(result["model"]) > 0

    def test_get_battery_info(self, operator):
        result = operator.execute("get_battery_info", {})
        assert result["ok"]
        assert "level" in result
        assert 0 <= result["level"] <= 100

    def test_get_wifi_info(self, operator):
        result = operator.execute("get_wifi_info", {})
        assert result["ok"]
        assert "connected" in result


# ═══════════════════════════════════════════════════════════════════════
# UI Interaction
# ═══════════════════════════════════════════════════════════════════════


class TestUI:
    def test_ui_dump(self, operator):
        result = operator.execute("ui_dump", {})
        assert result["ok"]
        assert result["total_elements"] > 0

    def test_screenshot(self, operator):
        result = operator.execute("screenshot", {})
        assert result["ok"]
        assert result["size_bytes"] > 0
        from pathlib import Path
        Path(result["path"]).unlink(missing_ok=True)

    def test_tap(self, operator):
        # Tap center of screen (safe area)
        size = operator.execute("screen_size", {})
        cx = size["width"] // 2
        cy = size["height"] // 2
        result = operator.execute("tap", {"x": cx, "y": cy})
        assert result["ok"]

    def test_double_tap(self, operator):
        size = operator.execute("screen_size", {})
        cx = size["width"] // 2
        cy = size["height"] // 2
        result = operator.execute("double_tap", {"x": cx, "y": cy})
        assert result["ok"]
        assert result["taps"] == 2

    def test_swipe(self, operator):
        size = operator.execute("screen_size", {})
        cx = size["width"] // 2
        result = operator.execute("swipe", {
            "x1": cx, "y1": size["height"] // 2,
            "x2": cx, "y2": size["height"] // 4,
            "duration_ms": 300,
        })
        assert result["ok"]

    def test_press_key(self, operator):
        result = operator.execute("press_key", {"key": "home"})
        assert result["ok"]


# ═══════════════════════════════════════════════════════════════════════
# Apps
# ═══════════════════════════════════════════════════════════════════════


class TestApps:
    def test_current_app(self, operator):
        result = operator.execute("current_app", {})
        assert result["ok"]

    def test_list_packages(self, operator):
        result = operator.execute("list_packages", {})
        assert result["ok"]
        assert result["count"] > 0

    def test_list_running_apps(self, operator):
        result = operator.execute("list_running_apps", {})
        assert result["ok"]

    def test_launch_and_stop_app(self, operator):
        # Launch Settings (always available)
        result = operator.execute("launch_app", {"package": "com.android.settings"})
        assert result["ok"]
        time.sleep(1)
        result = operator.execute("stop_app", {"package": "com.android.settings"})
        assert result["ok"]


# ═══════════════════════════════════════════════════════════════════════
# Notifications & System
# ═══════════════════════════════════════════════════════════════════════


class TestSystem:
    def test_get_notifications(self, operator):
        result = operator.execute("get_notifications", {})
        assert result["ok"]
        assert isinstance(result.get("notifications"), list)

    def test_get_logcat(self, operator):
        result = operator.execute("get_logcat", {"lines": 10})
        assert result["ok"]
        assert result["count"] > 0

    def test_open_settings(self, operator):
        result = operator.execute("open_settings", {"page": "about"})
        assert result["ok"]
        time.sleep(0.5)
        # Go back to home
        operator.execute("press_key", {"key": "home"})

    def test_open_url(self, operator):
        result = operator.execute("open_url", {"url": "https://example.com"})
        assert result["ok"]
        time.sleep(1)
        operator.execute("press_key", {"key": "home"})


# ═══════════════════════════════════════════════════════════════════════
# Screen Control
# ═══════════════════════════════════════════════════════════════════════


class TestScreenControl:
    def test_wake_and_sleep(self, operator):
        result = operator.execute("wake", {})
        assert result["ok"]
        # Don't actually sleep the device during tests

    def test_set_brightness(self, operator):
        result = operator.execute("set_brightness", {"level": 128})
        assert result["ok"]
        assert result["brightness"] == 128

    def test_rotate_screen(self, operator):
        # Rotate to landscape then back to portrait
        result = operator.execute("rotate_screen", {"orientation": "landscape"})
        assert result["ok"]
        time.sleep(0.5)
        result = operator.execute("rotate_screen", {"orientation": "portrait"})
        assert result["ok"]


# ═══════════════════════════════════════════════════════════════════════
# Clipboard
# ═══════════════════════════════════════════════════════════════════════


class TestClipboard:
    def test_set_and_get_clipboard(self, operator):
        result = operator.execute("set_clipboard", {"text": "PredaCore Android test"})
        assert result["ok"]
        result = operator.execute("get_clipboard", {})
        assert result["ok"]
        # Clipboard read may not work on all devices without u2
        assert isinstance(result.get("text"), str)


# ═══════════════════════════════════════════════════════════════════════
# Shell
# ═══════════════════════════════════════════════════════════════════════


class TestShell:
    def test_shell_command(self, operator):
        result = operator.execute("shell", {"command": "echo hello_predacore"})
        assert result["ok"]
        assert "hello_predacore" in result["output"]

    def test_dangerous_command_blocked(self, operator):
        from predacore.operators.android import ADBError
        with pytest.raises(ADBError, match="blocked"):
            operator.execute("shell", {"command": "reboot"})


# ═══════════════════════════════════════════════════════════════════════
# Action count verification
# ═══════════════════════════════════════════════════════════════════════


class TestActionCount:
    def test_total_actions(self, operator):
        from predacore.operators.enums import AndroidAction
        actions = [a.value for a in AndroidAction]
        assert len(actions) == 61, f"Expected 61 Android actions, got {len(actions)}: {actions}"
