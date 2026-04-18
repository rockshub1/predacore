"""
Tests for AndroidOperator's uiautomator2 backend integration.

Pillar #3 of the operator parity plan: replace `adb shell input` with
uiautomator2 for ~5-10x speedup. These tests verify the dispatch logic
without requiring uiautomator2 to be installed or a real device to be
connected — the u2 module and the u2 device handle are both mocked.

Coverage:
  - When u2 is unavailable, all methods fall back to shell with
    `backend == "shell"` in the result
  - When u2 is available, methods use the u2 path with `backend == "u2"`
  - When a u2 call raises, the operator falls back to shell mid-call
  - The `_get_u2()` lazy connector is single-shot (no retry storms)
  - `health_check` reports u2 status correctly
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from predacore.operators import android as android_mod
from predacore.operators.android import AndroidOperator


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _make_operator(adb_path: str = "/fake/adb", serial: str = "") -> AndroidOperator:
    """Build an AndroidOperator with adb_path forced and shell mocked."""
    op = AndroidOperator.__new__(AndroidOperator)
    # __init__ runs _find_adb() which we want to skip
    import logging
    import threading

    op._log = logging.getLogger("test")
    op._ops_cfg = None
    op._macro_depth = 0
    op._macro_lock = threading.Lock()
    op._action_stats = {}
    op._serial = serial
    op._adb_path = adb_path
    op._last_ui_dump = ""
    op._last_elements = []
    op._ui_cache_lock = threading.Lock()
    op._u2 = None
    op._u2_init_attempted = False
    op._u2_lock = threading.Lock()
    return op


def _fake_u2_device() -> MagicMock:
    """Return a MagicMock that pretends to be a uiautomator2 device."""
    dev = MagicMock(name="u2.Device")
    dev.info = {"screenOn": True, "displayWidth": 1080, "displayHeight": 1920}
    dev.window_size.return_value = (1080, 1920)
    dev.app_current.return_value = {
        "package": "com.example.app",
        "activity": ".MainActivity",
    }
    dev.dump_hierarchy.return_value = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<hierarchy rotation="0">'
        '<node text="hello" bounds="[0,0][100,100]" clickable="true"/>'
        "</hierarchy>"
    )
    return dev


# ─────────────────────────────────────────────────────────────────────
# Lazy lib import
# ─────────────────────────────────────────────────────────────────────


class TestU2LibImport:
    def setup_method(self) -> None:
        # Reset module-level cache between tests
        android_mod._U2_LIB = None

    def teardown_method(self) -> None:
        android_mod._U2_LIB = None

    def test_missing_lib_caches_negative(self) -> None:
        with patch.dict("sys.modules", {"uiautomator2": None}):
            assert android_mod._get_u2_lib() is None
        assert android_mod._U2_LIB is False

    def test_present_lib_returns_module(self) -> None:
        fake_lib = MagicMock(name="uiautomator2")
        with patch.dict("sys.modules", {"uiautomator2": fake_lib}):
            result = android_mod._get_u2_lib()
        assert result is fake_lib
        assert android_mod._U2_LIB is fake_lib


# ─────────────────────────────────────────────────────────────────────
# _get_u2 lazy connection
# ─────────────────────────────────────────────────────────────────────


class TestGetU2:
    def setup_method(self) -> None:
        android_mod._U2_LIB = None

    def teardown_method(self) -> None:
        android_mod._U2_LIB = None

    def test_no_lib_returns_none(self) -> None:
        op = _make_operator()
        with patch.object(android_mod, "_get_u2_lib", return_value=None):
            assert op._get_u2() is None
            assert op._u2 is None
            assert op._u2_init_attempted is True

    def test_no_retry_after_failed_connect(self) -> None:
        op = _make_operator()
        fake_lib = MagicMock()
        fake_lib.connect.side_effect = RuntimeError("device offline")
        with patch.object(android_mod, "_get_u2_lib", return_value=fake_lib):
            assert op._get_u2() is None
            assert op._get_u2() is None  # cached
        assert fake_lib.connect.call_count == 1

    def test_successful_connect_caches_device(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        fake_lib = MagicMock()
        fake_lib.connect.return_value = dev
        with patch.object(android_mod, "_get_u2_lib", return_value=fake_lib):
            assert op._get_u2() is dev
            assert op._get_u2() is dev  # cached, no second connect
        assert fake_lib.connect.call_count == 1

    def test_u2_active_property(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_get_u2", return_value=dev):
            assert op.u2_active is True
        with patch.object(op, "_get_u2", return_value=None):
            assert op.u2_active is False


# ─────────────────────────────────────────────────────────────────────
# Dual-path: input methods
# ─────────────────────────────────────────────────────────────────────


class TestTapShellFirst:
    """Shell wins on both USB and wireless — `input tap` is ~90ms native.
    u2 is only the fallback if shell raises ADBError.
    """

    def test_uses_shell_when_shell_works(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev) as u2_get, \
             patch.object(op, "_shell_cmd") as shell:
            result = op._tap({"x": 100, "y": 200})
        shell.assert_called_once_with("input tap 100 200", timeout=5)
        u2_get.assert_not_called()
        assert result == {"x": 100, "y": 200, "backend": "shell"}

    def test_falls_back_to_u2_when_shell_fails(self) -> None:
        from predacore.operators.android import ADBError
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd", side_effect=ADBError("offline")):
            result = op._tap({"x": 50, "y": 60})
        dev.click.assert_called_once_with(50, 60)
        assert result == {"x": 50, "y": 60, "backend": "u2"}

    def test_raises_when_both_fail(self) -> None:
        from predacore.operators.android import ADBError
        op = _make_operator()
        dev = _fake_u2_device()
        dev.click.side_effect = RuntimeError("agent dead")
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd", side_effect=ADBError("shell broken")):
            with pytest.raises(ADBError, match="shell broken"):
                op._tap({"x": 1, "y": 1})


class TestSwipeShellFirst:
    def test_uses_shell_when_shell_works(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev) as u2_get, \
             patch.object(op, "_shell_cmd") as shell:
            result = op._swipe({"x1": 1, "y1": 2, "x2": 3, "y2": 4, "duration_ms": 200})
        shell.assert_called_once_with("input swipe 1 2 3 4 200", timeout=10)
        u2_get.assert_not_called()
        assert result["backend"] == "shell"

    def test_falls_back_to_u2_when_shell_fails(self) -> None:
        from predacore.operators.android import ADBError
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd", side_effect=ADBError("offline")):
            result = op._swipe({"x1": 0, "y1": 0, "x2": 100, "y2": 100, "duration_ms": 500})
        dev.swipe.assert_called_once_with(0, 0, 100, 100, duration=0.5)
        assert result["backend"] == "u2"


class TestLongPressShellFirst:
    def test_uses_shell_when_shell_works(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev) as u2_get, \
             patch.object(op, "_shell_cmd") as shell:
            result = op._long_press({"x": 10, "y": 20, "duration_ms": 1500})
        shell.assert_called_once_with("input swipe 10 20 10 20 1500", timeout=10)
        u2_get.assert_not_called()
        assert result["backend"] == "shell"

    def test_falls_back_to_u2_when_shell_fails(self) -> None:
        from predacore.operators.android import ADBError
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd", side_effect=ADBError("offline")):
            result = op._long_press({"x": 10, "y": 20, "duration_ms": 1500})
        dev.long_click.assert_called_once_with(10, 20, duration=1.5)
        assert result["backend"] == "u2"


class TestTypeTextRouting:
    """Routing:
      - Simple ASCII → shell `input text` (native, ~130ms)
      - Unicode → u2 `send_keys` (FastInputIME, correct for unicode)
      - Unicode + no u2 → per-char shell clipboard fallback
    """

    def test_simple_ascii_uses_shell(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev) as u2_get, \
             patch.object(op, "_shell_cmd") as shell:
            result = op._type_text({"text": "abc"})
        shell.assert_called_once_with("input text abc", timeout=10)
        u2_get.assert_not_called()
        assert result == {"typed_chars": 3, "backend": "shell"}

    def test_unicode_uses_u2_when_available(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd") as shell:
            result = op._type_text({"text": "héllo wörld"})
        dev.send_keys.assert_called_once_with("héllo wörld")
        shell.assert_not_called()
        assert result == {"typed_chars": 11, "backend": "u2"}

    def test_unicode_falls_back_to_per_char_when_u2_unavailable(self) -> None:
        op = _make_operator()
        with patch.object(op, "_u2_or_none", return_value=None), \
             patch.object(op, "_shell_cmd") as shell:
            result = op._type_text({"text": "a ö"})
        # 'a' via `input text a`, ' ' via keyevent, 'ö' via clipboard paste
        assert shell.call_count == 3
        assert result == {"typed_chars": 3, "backend": "shell"}

    def test_simple_ascii_falls_back_to_u2_when_shell_fails(self) -> None:
        from predacore.operators.android import ADBError
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd", side_effect=ADBError("offline")):
            result = op._type_text({"text": "abc"})
        dev.send_keys.assert_called_once_with("abc")
        assert result == {"typed_chars": 3, "backend": "u2"}


class TestPressKeyShellFirst:
    """Shell wins by ~10x — u2.press() is a smart op that waits for
    UI transitions, ~500-1100ms vs shell's ~110ms.
    """

    def test_named_key_uses_shell(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev) as u2_get, \
             patch.object(op, "_shell_cmd") as shell:
            result = op._press_key({"key": "home"})
        shell.assert_called_once_with("input keyevent KEYCODE_HOME", timeout=5)
        u2_get.assert_not_called()
        assert result["backend"] == "shell"

    def test_numeric_keycode_uses_shell(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd") as shell:
            op._press_key({"keycode": 4})
        shell.assert_called_once_with("input keyevent 4", timeout=5)

    def test_falls_back_to_u2_when_shell_fails(self) -> None:
        from predacore.operators.android import ADBError
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd", side_effect=ADBError("offline")):
            result = op._press_key({"key": "back"})
        dev.press.assert_called_once_with("back")
        assert result["backend"] == "u2"


# ─────────────────────────────────────────────────────────────────────
# Dual-path: inspection methods
# ─────────────────────────────────────────────────────────────────────


class TestUiDumpDualPath:
    def test_uses_u2_dump_hierarchy(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd") as shell:
            result = op._ui_dump({})
        dev.dump_hierarchy.assert_called_once()
        shell.assert_not_called()
        assert result["backend"] == "u2"
        assert result["total_elements"] >= 1

    def test_falls_back_when_u2_dump_fails(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        dev.dump_hierarchy.side_effect = RuntimeError("agent dead")
        # Shell path returns valid XML for parsing
        shell_xml = (
            '<?xml version="1.0"?><hierarchy rotation="0">'
            '<node text="x" bounds="[0,0][1,1]"/></hierarchy>'
        )
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd", return_value=shell_xml) as shell:
            result = op._ui_dump({})
        # uiautomator dump → cat → rm = 3 shell calls
        assert shell.call_count == 3
        assert result["backend"] == "shell"


class TestScreenSizeShellFirst:
    """screen_size uses shell first because `wm size` is faster than u2.

    Empirical: shell ~115ms, u2 ~234ms on real OnePlus device. u2's HTTP
    round-trip overhead exceeds the cost of a single adb call.
    """

    def test_uses_shell_when_shell_works(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev) as u2_get, \
             patch.object(op, "_shell_cmd", return_value="Physical size: 1440x3120"):
            result = op._screen_size({})
        u2_get.assert_not_called()
        assert result == {"width": 1440, "height": 3120, "backend": "shell"}

    def test_falls_back_to_u2_when_shell_fails(self) -> None:
        from predacore.operators.android import ADBError
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd", side_effect=ADBError("device offline")):
            result = op._screen_size({})
        assert result == {"width": 1080, "height": 1920, "backend": "u2"}


class TestScreenStateShellFirst:
    """screen_state uses shell first — `dumpsys power` is a single fast call."""

    def test_uses_shell_when_shell_works(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev) as u2_get, \
             patch.object(op, "_shell_cmd", return_value="  mWakefulness=Awake"):
            result = op._screen_state({})
        u2_get.assert_not_called()
        assert result["screen_on"] is True
        assert result["backend"] == "shell"

    def test_shell_reports_asleep(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd", return_value="  mWakefulness=Asleep"):
            result = op._screen_state({})
        assert result["screen_on"] is False
        assert result["backend"] == "shell"

    def test_falls_back_to_u2_when_shell_fails(self) -> None:
        from predacore.operators.android import ADBError
        op = _make_operator()
        dev = _fake_u2_device()
        dev.info = {"screenOn": True}
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd", side_effect=ADBError("device offline")):
            result = op._screen_state({})
        assert result == {"screen_on": True, "backend": "u2"}


# ─────────────────────────────────────────────────────────────────────
# Dual-path: app management
# ─────────────────────────────────────────────────────────────────────


class TestLaunchAppShellFirst:
    def test_uses_shell_via_monkey_when_no_activity(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev) as u2_get, \
             patch.object(op, "_shell_cmd") as shell:
            result = op._launch_app({"package": "com.example.app"})
        shell.assert_called_once_with(
            "monkey -p com.example.app -c android.intent.category.LAUNCHER 1",
            timeout=10,
        )
        u2_get.assert_not_called()
        assert result["backend"] == "shell"

    def test_uses_shell_am_start_with_activity(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd") as shell:
            op._launch_app({"package": "com.example.app", "activity": ".MainActivity"})
        shell.assert_called_once_with("am start -n com.example.app/.MainActivity", timeout=10)

    def test_falls_back_to_u2_when_shell_fails(self) -> None:
        from predacore.operators.android import ADBError
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd", side_effect=ADBError("offline")):
            result = op._launch_app({"package": "com.example.app"})
        dev.app_start.assert_called_once_with("com.example.app", activity=None)
        assert result["backend"] == "u2"


class TestStopAppShellFirst:
    def test_uses_shell_when_shell_works(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev) as u2_get, \
             patch.object(op, "_shell_cmd") as shell:
            result = op._stop_app({"package": "com.example.app"})
        shell.assert_called_once_with("am force-stop com.example.app", timeout=5)
        u2_get.assert_not_called()
        assert result["backend"] == "shell"

    def test_falls_back_to_u2_when_shell_fails(self) -> None:
        from predacore.operators.android import ADBError
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd", side_effect=ADBError("offline")):
            result = op._stop_app({"package": "com.example.app"})
        dev.app_stop.assert_called_once_with("com.example.app")
        assert result["backend"] == "u2"


class TestCurrentAppShellFirst:
    """current_app uses shell first — u2's app_current is catastrophically
    slow on ColorOS (~5s vs shell's ~88ms). Shell is the fast path here.
    """

    def test_uses_shell_when_shell_works(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        # Real Android 14 ColorOS output format (relative activity name)
        real_output = "  ResumedActivity: ActivityRecord{abc u0 com.foo.app/.Main t5}"
        with patch.object(op, "_u2_or_none", return_value=dev) as u2_get, \
             patch.object(op, "_shell_cmd", return_value=real_output):
            result = op._current_app({})
        u2_get.assert_not_called()
        assert result["package"] == "com.foo.app"
        assert result["activity"] == ".Main"
        assert result["backend"] == "shell"

    def test_falls_back_to_u2_when_shell_fails(self) -> None:
        from predacore.operators.android import ADBError
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd", side_effect=ADBError("device offline")):
            result = op._current_app({})
        assert result == {
            "package": "com.example.app",
            "activity": ".MainActivity",
            "backend": "u2",
        }


class TestWakeSleepShellFirst:
    def test_wake_uses_shell(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev) as u2_get, \
             patch.object(op, "_shell_cmd") as shell:
            result = op._wake({})
        shell.assert_called_once_with("input keyevent KEYCODE_WAKEUP", timeout=5)
        u2_get.assert_not_called()
        assert result == {"woke": True, "backend": "shell"}

    def test_sleep_uses_shell(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev) as u2_get, \
             patch.object(op, "_shell_cmd") as shell:
            result = op._sleep_device({})
        shell.assert_called_once_with("input keyevent KEYCODE_SLEEP", timeout=5)
        u2_get.assert_not_called()
        assert result == {"sleeping": True, "backend": "shell"}

    def test_wake_falls_back_to_u2_when_shell_fails(self) -> None:
        from predacore.operators.android import ADBError
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd", side_effect=ADBError("offline")):
            result = op._wake({})
        dev.screen_on.assert_called_once()
        assert result == {"woke": True, "backend": "u2"}

    def test_sleep_falls_back_to_u2_when_shell_fails(self) -> None:
        from predacore.operators.android import ADBError
        op = _make_operator()
        dev = _fake_u2_device()
        with patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(op, "_shell_cmd", side_effect=ADBError("offline")):
            result = op._sleep_device({})
        dev.screen_off.assert_called_once()
        assert result == {"sleeping": True, "backend": "u2"}


# ─────────────────────────────────────────────────────────────────────
# Health check reports backend
# ─────────────────────────────────────────────────────────────────────


class TestHealthCheckReportsBackend:
    def setup_method(self) -> None:
        android_mod._U2_LIB = None

    def teardown_method(self) -> None:
        android_mod._U2_LIB = None

    def test_health_check_no_u2(self) -> None:
        op = _make_operator()
        with patch.object(op, "_adb", return_value="List of devices attached\n"), \
             patch.object(op, "_u2_or_none", return_value=None), \
             patch.object(android_mod, "_get_u2_lib", return_value=None):
            result = op.health_check()
        assert result["u2_installed"] is False
        assert result["u2_connected"] is False
        assert result["backend"] == "shell"

    def test_health_check_with_u2(self) -> None:
        op = _make_operator()
        dev = _fake_u2_device()
        fake_lib = MagicMock()
        with patch.object(op, "_adb", return_value="List of devices attached\nemulator-5554\tdevice\n"), \
             patch.object(op, "_u2_or_none", return_value=dev), \
             patch.object(android_mod, "_get_u2_lib", return_value=fake_lib):
            result = op.health_check()
        assert result["u2_installed"] is True
        assert result["u2_connected"] is True
        assert result["backend"] == "u2"
