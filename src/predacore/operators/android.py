"""
PredaCore Android device operator via ADB (Android Debug Bridge).

Provides action-level automation primitives for Android devices:
- UI tree dump via uiautomator (structured XML → elements)
- Tap, swipe, long-press, pinch gestures
- Text input (broadcast-based for reliability) and key events
- Screenshot capture
- App launch/focus, install, uninstall
- Shell command execution (sandboxed)
- Screen recording
- Macro execution via BaseOperator.execute_macro with abort support

Phase 6.3: Uses AndroidAction enums — zero magic strings in dispatch.

Requires: adb in PATH and a connected device/emulator.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import BaseOperator, OperatorPlatform, OperatorError, MacroAbortToken
from .enums import AndroidAction

logger = logging.getLogger(__name__)


# ── uiautomator2 lazy import ────────────────────────────────────────────
# uiautomator2 keeps a persistent ATX agent on the device. Taps drop from
# ~150ms (per `adb shell input tap`) to ~15ms; UI dumps from ~1-2s to
# ~100ms. We import lazily so the module still loads when the optional
# dependency is missing — the operator silently falls back to shell.
#
# Module-level cache: None = not yet attempted, False = attempted+failed,
# module object = imported successfully.
_U2_LIB: Any = None


def _get_u2_lib() -> Any:
    """Return the uiautomator2 module, or None if unavailable."""
    global _U2_LIB
    if _U2_LIB is False:
        return None
    if _U2_LIB is not None:
        return _U2_LIB
    try:
        import uiautomator2 as _u2  # noqa: PLC0415
        _U2_LIB = _u2
        return _u2
    except ImportError as exc:
        logger.debug("uiautomator2 unavailable, using shell fallback: %s", exc)
        _U2_LIB = False  # type: ignore[assignment]
        return None


class ADBError(RuntimeError):
    """Raised for ADB execution failures."""


@dataclass
class AndroidElement:
    """Parsed UI element from uiautomator dump."""

    resource_id: str = ""
    text: str = ""
    content_desc: str = ""
    class_name: str = ""
    package: str = ""
    bounds: tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2
    clickable: bool = False
    focusable: bool = False
    enabled: bool = True
    checked: bool = False
    selected: bool = False
    scrollable: bool = False

    @property
    def center(self) -> tuple[int, int]:
        """Return the center point of this element's bounds."""
        x1, y1, x2, y2 = self.bounds
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def label(self) -> str:
        """Best human-readable label for this element."""
        return self.text or self.content_desc or (self.resource_id.split("/")[-1] if self.resource_id else "")

    def to_dict(self) -> dict[str, Any]:
        """Serialize element to dictionary representation."""
        return {
            "resource_id": self.resource_id,
            "text": self.text,
            "content_desc": self.content_desc,
            "class": self.class_name,
            "package": self.package,
            "bounds": list(self.bounds),
            "center": list(self.center),
            "clickable": self.clickable,
            "enabled": self.enabled,
            "scrollable": self.scrollable,
        }


_BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")


class AndroidOperator(BaseOperator):
    """Android device controller via ADB.

    Uses AndroidAction enums for all action dispatch — zero magic strings.
    """

    # Keys that map to Android keycodes
    _KEYCODE_MAP = {
        "home": "KEYCODE_HOME",
        "back": "KEYCODE_BACK",
        "menu": "KEYCODE_MENU",
        "recent": "KEYCODE_APP_SWITCH",
        "power": "KEYCODE_POWER",
        "volume_up": "KEYCODE_VOLUME_UP",
        "volume_down": "KEYCODE_VOLUME_DOWN",
        "enter": "KEYCODE_ENTER",
        "tab": "KEYCODE_TAB",
        "delete": "KEYCODE_DEL",
        "backspace": "KEYCODE_DEL",
        "escape": "KEYCODE_ESCAPE",
        "space": "KEYCODE_SPACE",
        "up": "KEYCODE_DPAD_UP",
        "down": "KEYCODE_DPAD_DOWN",
        "left": "KEYCODE_DPAD_LEFT",
        "right": "KEYCODE_DPAD_RIGHT",
        "camera": "KEYCODE_CAMERA",
        "search": "KEYCODE_SEARCH",
    }

    def __init__(
        self,
        device_serial: str | None = None,
        log: logging.Logger | None = None,
        operators_config: Any = None,
    ) -> None:
        super().__init__(log=log, operators_config=operators_config)
        self._serial = device_serial or os.getenv("PREDACORE_ADB_SERIAL", "")
        self._adb_path = self._find_adb()
        self._last_ui_dump: str = ""
        self._last_elements: list[AndroidElement] = []
        self._ui_cache_lock = threading.Lock()
        self._ops_cfg = operators_config
        # uiautomator2 device handle — lazily connected on first use.
        # _u2 is the device or None; _u2_init_attempted gates retries.
        self._u2: Any = None
        self._u2_init_attempted: bool = False
        self._u2_lock = threading.Lock()

    # ── Input Validation ─────────────────────────────────────────────

    _SAFE_COMPONENT_RE = re.compile(r'^[a-zA-Z0-9._$]+$')

    @classmethod
    def _validate_component_name(cls, value: str, label: str) -> str:
        """Validate an Android component name (package or activity)."""
        value = value.strip()
        if not value:
            raise ADBError(f"{label} is required")
        if not cls._SAFE_COMPONENT_RE.match(value):
            raise ADBError(
                f"invalid {label}: {value!r} — "
                "only alphanumeric, dots, underscores, and $ allowed"
            )
        return value

    @property
    def available(self) -> bool:
        """Check if ADB is available and a device is connected."""
        if not self._adb_path:
            return False
        try:
            out = self._adb("devices", timeout=5)
            lines = [l.strip() for l in out.strip().splitlines()[1:] if l.strip()]
            return any("device" in l for l in lines)
        except (ADBError, OSError, subprocess.TimeoutExpired) as e:
            logger.debug("ADB availability check failed: %s", e)
            return False

    @property
    def platform(self) -> OperatorPlatform:
        """Return the platform identifier."""
        return OperatorPlatform.ANDROID

    @property
    def supported_actions(self) -> set[str]:
        """Return set of supported Android automation actions."""
        return {a.value for a in AndroidAction}

    # ── Main Dispatch ────────────────────────────────────────────

    def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute one Android action and return structured result."""
        start = time.time()
        action = str(action or "").strip().lower()
        if not action:
            raise ADBError("action is required")
        if not self._adb_path:
            raise ADBError("adb not found in PATH. Install Android SDK platform-tools.")

        ok = True
        try:
            result = self._dispatch_action(action, params)
            elapsed_ms = int((time.time() - start) * 1000)
            result.update({"ok": True, "action": action, "elapsed_ms": elapsed_ms})
            return result
        except Exception:
            ok = False
            raise
        finally:
            elapsed_ms = int((time.time() - start) * 1000)
            self._record_telemetry(action, elapsed_ms, ok)

    def _dispatch_action(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Core dispatch — maps action string to implementation."""

        # Handle macros via base class (with abort support)
        if action == AndroidAction.RUN_MACRO:
            steps = params.get("steps")
            abort_token = params.pop("_abort_token", None)
            return self.execute_macro(
                steps,
                stop_on_error=bool(params.get("stop_on_error", True)),
                delay_ms=max(0, min(int(params.get("delay_ms", 300) or 300), 10000)),
                abort_token=abort_token,
            )

        try:
            dispatch = self._action_dispatch
        except AttributeError:
            self._action_dispatch = {
                AndroidAction.HEALTH_CHECK: self.health_check,
                AndroidAction.UI_DUMP: self._ui_dump,
                AndroidAction.TAP: self._tap,
                AndroidAction.LONG_PRESS: self._long_press,
                AndroidAction.SWIPE: self._swipe,
                AndroidAction.PINCH: self._pinch,
                AndroidAction.TYPE_TEXT: self._type_text,
                AndroidAction.PRESS_KEY: self._press_key,
                AndroidAction.SCREENSHOT: self._screenshot,
                AndroidAction.LAUNCH_APP: self._launch_app,
                AndroidAction.STOP_APP: self._stop_app,
                AndroidAction.CURRENT_APP: self._current_app,
                AndroidAction.LIST_PACKAGES: self._list_packages,
                AndroidAction.INSTALL_APK: self._install_apk,
                AndroidAction.SHELL: self._shell,
                AndroidAction.FIND_ELEMENT: self._find_element,
                AndroidAction.FIND_AND_TAP: self._find_and_tap,
                AndroidAction.FIND_AND_TYPE: self._find_and_type,
                AndroidAction.WAIT_FOR_ELEMENT: self._wait_for_element,
                AndroidAction.SCREEN_SIZE: self._screen_size,
                AndroidAction.SCREEN_STATE: self._screen_state,
                AndroidAction.WAKE: self._wake,
                AndroidAction.SLEEP_DEVICE: self._sleep_device,
                AndroidAction.PUSH_FILE: self._push_file,
                AndroidAction.PULL_FILE: self._pull_file,
                AndroidAction.SCREEN_RECORD: self._screen_record,
                # Extended capabilities
                AndroidAction.GET_NOTIFICATIONS: self._get_notifications,
                AndroidAction.CLEAR_NOTIFICATIONS: self._clear_notifications,
                AndroidAction.TOGGLE_WIFI: self._toggle_wifi,
                AndroidAction.TOGGLE_BLUETOOTH: self._toggle_bluetooth,
                AndroidAction.TOGGLE_AIRPLANE: self._toggle_airplane,
                AndroidAction.TOGGLE_FLASHLIGHT: self._toggle_flashlight,
                AndroidAction.GET_CLIPBOARD: self._get_clipboard,
                AndroidAction.SET_CLIPBOARD: self._set_clipboard,
                AndroidAction.GET_DEVICE_INFO: self._get_device_info,
                AndroidAction.GET_BATTERY_INFO: self._get_battery_info,
                AndroidAction.OPEN_URL: self._open_url,
                AndroidAction.CLEAR_APP_DATA: self._clear_app_data,
                AndroidAction.UNINSTALL_APP: self._uninstall_app,
                AndroidAction.SCROLL_TO_ELEMENT: self._scroll_to_element,
                AndroidAction.GET_LOGCAT: self._get_logcat,
                AndroidAction.DOUBLE_TAP: self._double_tap,
                AndroidAction.ROTATE_SCREEN: self._rotate_screen,
                AndroidAction.SET_BRIGHTNESS: self._set_brightness,
                AndroidAction.OPEN_SETTINGS: self._open_settings,
                AndroidAction.SEND_BROADCAST: self._send_broadcast,
                AndroidAction.GET_WIFI_INFO: self._get_wifi_info,
                AndroidAction.LIST_RUNNING_APPS: self._list_running_apps,
                AndroidAction.INPUT_KEYCOMBO: self._input_keycombo,
                # Smart automation
                AndroidAction.SCROLL_AND_COLLECT_ALL: self._scroll_and_collect_all,
                AndroidAction.WAIT_FOR_TEXT: self._wait_for_text,
                AndroidAction.SCREENSHOT_AND_OCR: self._screenshot_and_ocr,
                AndroidAction.SEARCH_IN_APP: self._search_in_app,
                AndroidAction.GET_FOCUSED_ELEMENT: self._get_focused_element,
                AndroidAction.GESTURE: self._gesture,
                AndroidAction.CONNECT_CHROME_ON_DEVICE: self._connect_chrome_on_device,
                # Smart automation
                AndroidAction.WAIT_FOR_STABLE_UI: self._wait_for_stable_ui,
                AndroidAction.ENSURE_IN_APP: self._ensure_in_app,
                AndroidAction.SMART_TAP: self._smart_tap,
                AndroidAction.SMART_TYPE: self._smart_type,
            }
            dispatch = self._action_dispatch

        handler = dispatch.get(action)
        if handler is None:
            raise ADBError(f"unsupported action: {action}")
        return handler(params)

    # ── uiautomator2 Backend ────────────────────────────────────────

    def _get_u2(self) -> Any:
        """Return a connected uiautomator2 device, or None.

        Lazily connects on first call and caches the result. If u2 is
        not installed, no device is connected, or the connection fails,
        returns None and never retries (until the operator is recreated).
        Callers MUST handle the None case by falling back to shell.
        """
        if self._u2 is not None or self._u2_init_attempted:
            return self._u2
        with self._u2_lock:
            if self._u2 is not None or self._u2_init_attempted:
                return self._u2
            self._u2_init_attempted = True
            u2_lib = _get_u2_lib()
            if u2_lib is None:
                return None
            try:
                self._u2 = u2_lib.connect(self._serial or None)
                # Touch .info to verify the agent is actually responding
                _ = self._u2.info
                self._log.debug(
                    "uiautomator2 connected: serial=%s",
                    self._serial or "(default)",
                )
            except Exception as exc:
                self._log.debug("uiautomator2 connect failed: %s", exc)
                self._u2 = None
        return self._u2

    @property
    def u2_active(self) -> bool:
        """True if the uiautomator2 backend is currently in use."""
        return self._get_u2() is not None

    def _u2_or_none(self) -> Any:
        """Return the u2 device handle if available, otherwise None.

        Wrapped in a try so that a previously-working u2 connection that
        has since died doesn't crash the operator — we fall back to shell.
        """
        try:
            return self._get_u2()
        except Exception as exc:
            self._log.debug("u2 backend unavailable: %s", exc)
            return None

    # ── ADB Primitives ──────────────────────────────────────────────

    def _adb(self, *args: str, timeout: float = 15, input_text: str | None = None) -> str:
        """Run an adb command and return stdout."""
        cmd = [self._adb_path]
        if self._serial:
            cmd.extend(["-s", self._serial])
        cmd.extend(args)
        try:
            proc = subprocess.run(
                cmd,
                input=input_text,
                capture_output=True,
                text=True,
                check=True,
                timeout=max(1.0, timeout),
            )
            return proc.stdout
        except subprocess.TimeoutExpired as exc:
            raise ADBError(f"adb timed out: {' '.join(args)}") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip() or (exc.stdout or "").strip() or str(exc)
            raise ADBError(f"adb error: {detail}") from exc

    def _shell_cmd(self, cmd: str, timeout: float = 15) -> str:
        """Run an adb shell command safely (no local shell=True).

        Note: adb shell doesn't propagate the remote exit code by default,
        so we check stderr for obvious errors instead of check=True.
        """
        args: list[str] = [self._adb_path]
        if self._serial:
            args.extend(["-s", self._serial])
        args.extend(["shell", cmd])
        try:
            proc = subprocess.run(
                args,
                shell=False,
                capture_output=True,
                text=True,
                timeout=max(1.0, timeout),
            )
            # adb itself failing (e.g. device disconnected)
            if proc.returncode != 0 and proc.stderr.strip():
                raise ADBError(f"adb shell error: {proc.stderr.strip()}")
            return proc.stdout
        except subprocess.TimeoutExpired as exc:
            raise ADBError(f"adb shell timed out: {cmd}") from exc

    # ── Actions ─────────────────────────────────────────────────────

    def health_check(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Check ADB connectivity and return device status."""
        adb_ok = bool(self._adb_path)
        devices: list[dict[str, str]] = []
        try:
            out = self._adb("devices", "-l", timeout=5)
            for line in out.strip().splitlines()[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    serial = parts[0]
                    state = parts[1]
                    model = ""
                    for p in parts[2:]:
                        if p.startswith("model:"):
                            model = p.split(":", 1)[1]
                    devices.append({"serial": serial, "state": state, "model": model})
        except (ADBError, OSError, subprocess.TimeoutExpired) as e:
            return {"adb_path": self._adb_path, "adb_ok": adb_ok, "error": str(e), "devices": []}

        u2_lib = _get_u2_lib()
        u2_installed = u2_lib is not None
        u2_connected = False
        if u2_installed:
            u2_connected = self._u2_or_none() is not None

        return {
            "adb_path": self._adb_path,
            "adb_ok": adb_ok,
            "selected_serial": self._serial or "(auto)",
            "devices": devices,
            "connected": any(d["state"] == "device" for d in devices),
            "u2_installed": u2_installed,
            "u2_connected": u2_connected,
            "backend": "u2" if u2_connected else "shell",
            "telemetry": self.telemetry(),
        }

    def _ui_dump(self, params: dict[str, Any]) -> dict[str, Any]:
        """Dump UI hierarchy.

        Prefers u2's `dump_hierarchy` (~100ms in-memory). Falls back to
        the older `uiautomator dump` shell command (~1-2s, writes XML to
        /sdcard, cats it back, deletes it).
        """
        backend = "shell"
        u2 = self._u2_or_none()
        xml_content = ""
        if u2 is not None:
            try:
                xml_content = u2.dump_hierarchy()
                backend = "u2"
            except Exception as exc:
                self._log.warning("u2 dump_hierarchy failed, falling back to shell: %s", exc)

        if not xml_content:
            import uuid as _uuid
            remote_path = f"/sdcard/predacore_ui_dump_{_uuid.uuid4().hex[:8]}.xml"
            self._shell_cmd(f"uiautomator dump {remote_path}", timeout=10)
            xml_content = self._shell_cmd(f"cat {remote_path}", timeout=5)
            self._shell_cmd(f"rm -f {remote_path}", timeout=3)

        with self._ui_cache_lock:
            self._last_ui_dump = xml_content
            elements = self._parse_ui_xml(xml_content)
            self._last_elements = elements

        max_elements = int(params.get("max_elements", 100))
        filtered = elements[:max_elements]

        return {
            "total_elements": len(elements),
            "returned": len(filtered),
            "elements": [e.to_dict() for e in filtered],
            "backend": backend,
        }

    def _tap(self, params: dict[str, Any]) -> dict[str, Any]:
        if "x" not in params or "y" not in params:
            raise OperatorError("tap requires both 'x' and 'y' coordinates", action="tap")
        x = int(params["x"])
        y = int(params["y"])
        # Shell wins: `input tap` is ~90ms native; u2 adds ~100-200ms HTTP
        # round-trip for zero benefit. u2 is only the fallback if shell fails.
        try:
            self._shell_cmd(f"input tap {x} {y}", timeout=5)
            return {"x": x, "y": y, "backend": "shell"}
        except ADBError as exc:
            u2 = self._u2_or_none()
            if u2 is not None:
                try:
                    u2.click(x, y)
                    return {"x": x, "y": y, "backend": "u2"}
                except Exception:
                    pass
            raise exc

    def _long_press(self, params: dict[str, Any]) -> dict[str, Any]:
        if "x" not in params or "y" not in params:
            raise OperatorError("long_press requires both 'x' and 'y' coordinates", action="long_press")
        x = int(params["x"])
        y = int(params["y"])
        duration_ms = int(params.get("duration_ms", 1000))
        # Shell wins: same reason as _tap — `input swipe` is native and
        # u2's HTTP overhead buys nothing here.
        try:
            self._shell_cmd(f"input swipe {x} {y} {x} {y} {duration_ms}", timeout=10)
            return {"x": x, "y": y, "duration_ms": duration_ms, "backend": "shell"}
        except ADBError as exc:
            u2 = self._u2_or_none()
            if u2 is not None:
                try:
                    u2.long_click(x, y, duration=duration_ms / 1000.0)
                    return {"x": x, "y": y, "duration_ms": duration_ms, "backend": "u2"}
                except Exception:
                    pass
            raise exc

    def _swipe(self, params: dict[str, Any]) -> dict[str, Any]:
        for k in ("x1", "y1", "x2", "y2"):
            if k not in params:
                raise OperatorError(f"swipe requires '{k}' coordinate", action="swipe")
        x1 = int(params["x1"])
        y1 = int(params["y1"])
        x2 = int(params["x2"])
        y2 = int(params["y2"])
        duration_ms = int(params.get("duration_ms", 300))
        try:
            self._shell_cmd(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}", timeout=10)
            return {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "duration_ms": duration_ms, "backend": "shell",
            }
        except ADBError as exc:
            u2 = self._u2_or_none()
            if u2 is not None:
                try:
                    u2.swipe(x1, y1, x2, y2, duration=duration_ms / 1000.0)
                    return {
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "duration_ms": duration_ms, "backend": "u2",
                    }
                except Exception:
                    pass
            raise exc

    def _pinch(self, params: dict[str, Any]) -> dict[str, Any]:
        """Pinch in/out — approximation using sequential swipes."""
        cx = int(params.get("center_x", 540))
        cy = int(params.get("center_y", 960))
        direction = str(params.get("direction", "in")).lower()
        distance = int(params.get("distance", 200))
        duration_ms = int(params.get("duration_ms", 500))

        if direction == "out":
            self._shell_cmd(f"input swipe {cx} {cy} {cx - distance} {cy - distance} {duration_ms}", timeout=5)
            self._shell_cmd(f"input swipe {cx} {cy} {cx + distance} {cy + distance} {duration_ms}", timeout=5)
        else:
            self._shell_cmd(f"input swipe {cx - distance} {cy - distance} {cx} {cy} {duration_ms}", timeout=5)
            self._shell_cmd(f"input swipe {cx + distance} {cy + distance} {cx} {cy} {duration_ms}", timeout=5)

        return {
            "center_x": cx, "center_y": cy, "direction": direction,
            "distance": distance,
            "note": "approximation via sequential swipes — true multi-touch requires sendevent",
        }

    def _type_text(self, params: dict[str, Any]) -> dict[str, Any]:
        """Type text on Android device.

        Routing:
          - Simple ASCII → shell `input text` (native, ~130ms).
          - Unicode/special chars → u2's `send_keys` (FastInputIME,
            correct for unicode).
          - Shell per-char clipboard fallback if both paths fail.
        """
        text = str(params.get("text", ""))
        if not text:
            raise ADBError("type_text requires text")

        is_simple = all(c.isalnum() or c in "._-@" for c in text)

        if is_simple:
            try:
                self._shell_cmd(f"input text {shlex.quote(text)}", timeout=10)
                return {"typed_chars": len(text), "backend": "shell"}
            except ADBError as exc:
                self._log.warning("shell input text failed, trying u2: %s", exc)

        u2 = self._u2_or_none()
        if u2 is not None:
            try:
                u2.send_keys(text)
                return {"typed_chars": len(text), "backend": "u2"}
            except Exception as exc:
                self._log.warning("u2 send_keys failed, falling back to per-char shell: %s", exc)

        for char in text:
            if char == " ":
                self._shell_cmd("input keyevent KEYCODE_SPACE", timeout=3)
            elif char == "\n":
                self._shell_cmd("input keyevent KEYCODE_ENTER", timeout=3)
            elif char == "\t":
                self._shell_cmd("input keyevent KEYCODE_TAB", timeout=3)
            elif char.isalnum() or char in "._-@":
                self._shell_cmd(f"input text {char}", timeout=3)
            else:
                safe_char = shlex.quote(char)
                self._shell_cmd(
                    f"am broadcast -a clipper.set -e text {safe_char} 2>/dev/null; "
                    f"input keyevent KEYCODE_PASTE 2>/dev/null || "
                    f"input text {safe_char}",
                    timeout=5,
                )
        return {"typed_chars": len(text), "backend": "shell"}

    def _press_key(self, params: dict[str, Any]) -> dict[str, Any]:
        key = str(params.get("key", "")).strip().lower()
        keycode = params.get("keycode")

        if keycode:
            try:
                code = str(int(keycode))
            except (TypeError, ValueError) as exc:
                raise ADBError("keycode must be a numeric value") from exc
        elif key in self._KEYCODE_MAP:
            code = self._KEYCODE_MAP[key]
        elif key.startswith("keycode_"):
            # Sanitize: only allow alphanumeric and underscores to prevent shell injection
            sanitized = key.upper()
            if not re.match(r"^KEYCODE_[A-Z0-9_]+$", sanitized):
                raise ADBError(f"invalid keycode name: {key}")
            code = sanitized
        else:
            raise ADBError(f"unsupported key: {key}. Use Android keycode name or 'keycode' param.")

        # Shell wins here by a wide margin. u2's press() is a smart
        # operation that waits for the resulting UI transition to settle
        # — that's ~500-1100ms on ColorOS vs shell's ~110ms. We don't
        # want transition-waiting here; use u2 only on shell failure.
        try:
            self._shell_cmd(f"input keyevent {code}", timeout=5)
            return {"key": key, "keycode": code, "backend": "shell"}
        except ADBError as exc:
            u2 = self._u2_or_none()
            if u2 is not None:
                try:
                    u2_key = code if str(code).isdigit() else code.replace("KEYCODE_", "").lower()
                    u2.press(u2_key)
                    return {"key": key, "keycode": code, "backend": "u2"}
                except Exception:
                    pass
            raise exc

    def _screenshot(self, params: dict[str, Any]) -> dict[str, Any]:
        target = str(params.get("path", "")).strip()
        include_b64 = bool(params.get("include_base64", False))

        if target:
            out_path = Path(target).expanduser().resolve()
            # Sandbox: only allow screenshots under home or tmp (matches desktop.py)
            home = Path.home().resolve()
            tmp = Path(tempfile.gettempdir()).resolve()
            if not (out_path.is_relative_to(home) or out_path.is_relative_to(tmp)):
                raise OperatorError(
                    f"screenshot path must be under home ({home}) or tmp ({tmp})",
                    action="screenshot",
                )
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            fd, temp_path = tempfile.mkstemp(prefix="predacore_android_", suffix=".png")
            os.close(fd)
            out_path = Path(temp_path)

        backend = "shell"
        captured = False
        u2 = self._u2_or_none()
        if u2 is not None:
            try:
                # u2.screenshot() can return PIL Image or save directly to path
                u2.screenshot(str(out_path))
                backend = "u2"
                captured = True
            except Exception as exc:
                self._log.warning("u2 screenshot failed, falling back to shell: %s", exc)

        if not captured:
            import uuid
            remote = f"/sdcard/predacore_screenshot_{uuid.uuid4().hex}.png"
            self._shell_cmd(f"screencap -p {remote}", timeout=10)
            self._adb("pull", remote, str(out_path), timeout=10)
            self._shell_cmd(f"rm -f {remote}", timeout=3)

        size = out_path.stat().st_size if out_path.exists() else 0
        result: dict[str, Any] = {"path": str(out_path), "size_bytes": size, "backend": backend}
        if include_b64 and size <= 2_500_000:
            result["image_b64"] = base64.b64encode(out_path.read_bytes()).decode("ascii")
        elif include_b64:
            result["base64_skipped"] = f"screenshot too large ({size} bytes)"
        return result

    def _launch_app(self, params: dict[str, Any]) -> dict[str, Any]:
        package = self._validate_component_name(
            str(params.get("package", "")), "package"
        )
        activity = str(params.get("activity", "")).strip()
        if activity:
            activity = self._validate_component_name(activity, "activity")

        # Shell wins: `am start` and `monkey` are native and fast;
        # u2's app_start wraps them plus waits for the app to come up,
        # which is overhead we don't want by default.
        try:
            if activity:
                self._shell_cmd(f"am start -n {package}/{activity}", timeout=10)
            else:
                self._shell_cmd(
                    f"monkey -p {package} -c android.intent.category.LAUNCHER 1",
                    timeout=10,
                )
            return {"package": package, "activity": activity or "(launcher)", "backend": "shell"}
        except ADBError as exc:
            u2 = self._u2_or_none()
            if u2 is not None:
                try:
                    u2.app_start(package, activity=activity or None)
                    return {
                        "package": package,
                        "activity": activity or "(launcher)",
                        "backend": "u2",
                    }
                except Exception:
                    pass
            raise exc

    def _stop_app(self, params: dict[str, Any]) -> dict[str, Any]:
        package = self._validate_component_name(
            str(params.get("package", "")), "package"
        )
        try:
            self._shell_cmd(f"am force-stop {package}", timeout=5)
            return {"package": package, "backend": "shell"}
        except ADBError as exc:
            u2 = self._u2_or_none()
            if u2 is not None:
                try:
                    u2.app_stop(package)
                    return {"package": package, "backend": "u2"}
                except Exception:
                    pass
            raise exc

    def _current_app(self, params: dict[str, Any]) -> dict[str, Any]:
        # Shell is empirically faster here (~80ms) than u2's app_current
        # (~5s on ColorOS — heavy package manager round-trip). u2 is the
        # fallback only when shell fails. The grep matches both legacy
        # `mResumedActivity=` and modern `ResumedActivity:` formats.
        try:
            out = self._shell_cmd("dumpsys activity activities | grep ResumedActivity", timeout=5)
            match = re.search(r"([\w.$]+)/([\w.$]+)", out)
            if match:
                return {
                    "package": match.group(1),
                    "activity": match.group(2),
                    "backend": "shell",
                }
            if out.strip():
                return {"raw": out.strip(), "backend": "shell"}
        except ADBError as exc:
            self._log.warning("shell current_app failed, trying u2: %s", exc)

        u2 = self._u2_or_none()
        if u2 is not None:
            try:
                info = u2.app_current()
                return {
                    "package": info.get("package", ""),
                    "activity": info.get("activity", ""),
                    "backend": "u2",
                }
            except Exception as exc:
                self._log.warning("u2 app_current also failed: %s", exc)
        return {"raw": "", "backend": "shell"}

    def _list_packages(self, params: dict[str, Any]) -> dict[str, Any]:
        filter_str = str(params.get("filter", "")).strip()
        if filter_str and not re.match(r'^[a-zA-Z0-9._-]+$', filter_str):
            raise ADBError("filter must contain only alphanumeric, dots, hyphens, underscores")
        cmd = "pm list packages"
        if filter_str:
            cmd += f" | grep -i {shlex.quote(filter_str)}"
        out = self._shell_cmd(cmd, timeout=10)
        packages = [l.replace("package:", "").strip() for l in out.splitlines() if l.strip()]
        return {"packages": packages, "count": len(packages)}

    def _install_apk(self, params: dict[str, Any]) -> dict[str, Any]:
        apk_path = str(params.get("path", "")).strip()
        if not apk_path:
            raise ADBError("install_apk requires path")
        out = self._adb("install", "-r", apk_path, timeout=120)
        return {"path": apk_path, "output": out.strip()}

    _DANGEROUS_TOKENS = frozenset({
        "reboot", "factory_reset", "wipe", "flash", "format",
        "mkfs", "dd", "setprop", "svc", "pm", "settings",
    })
    _DANGEROUS_PATTERNS = (
        re.compile(r'\brm\s+(-\w+\s+)*/', re.I),
    )

    def _shell(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run a sandboxed shell command on the device."""
        cmd = str(params.get("command", "")).strip()
        if not cmd:
            raise ADBError("shell requires command")

        cmd_normalized = " ".join(cmd.split()).lower()
        try:
            tokens = shlex.split(cmd_normalized)
        except ValueError:
            tokens = cmd_normalized.split()

        if tokens and tokens[0] in self._DANGEROUS_TOKENS:
            raise ADBError(f"blocked dangerous command: {tokens[0]}")

        for pattern in self._DANGEROUS_PATTERNS:
            if pattern.search(cmd_normalized):
                raise ADBError(f"blocked dangerous command pattern: {cmd}")

        timeout = float(params.get("timeout", 15))
        out = self._shell_cmd(cmd, timeout=timeout)
        return {"command": cmd, "output": out.strip()}

    def _find_element(self, params: dict[str, Any]) -> dict[str, Any]:
        """Find UI element by text, content_desc, or resource_id."""
        with self._ui_cache_lock:
            cached = list(self._last_elements)
        if not cached:
            self._ui_dump({})
            with self._ui_cache_lock:
                cached = list(self._last_elements)

        text = str(params.get("text", "")).strip().lower()
        desc = str(params.get("content_desc", "")).strip().lower()
        res_id = str(params.get("resource_id", "")).strip().lower()
        class_name = str(params.get("class_name", "")).strip().lower()

        matches = []
        for el in cached:
            if text and text in el.text.lower():
                matches.append(el)
            elif desc and desc in el.content_desc.lower():
                matches.append(el)
            elif res_id and res_id in el.resource_id.lower():
                matches.append(el)
            elif class_name and class_name in el.class_name.lower():
                matches.append(el)

        return {
            "found": len(matches),
            "elements": [m.to_dict() for m in matches[:10]],
        }

    def _find_and_tap(self, params: dict[str, Any]) -> dict[str, Any]:
        """Find element and tap its center."""
        result = self._find_element(params)
        if not result["elements"]:
            raise ADBError(f"element not found: {params}")
        el = result["elements"][0]
        cx, cy = el["center"]
        self._shell_cmd(f"input tap {cx} {cy}", timeout=5)
        return {"tapped": el, "x": cx, "y": cy}

    def _find_and_type(self, params: dict[str, Any]) -> dict[str, Any]:
        """Find a text field, tap it, and type text."""
        text_to_type = str(params.get("text", "")).strip()
        if not text_to_type:
            raise ADBError("find_and_type requires 'text'")

        result = self._find_element(params)
        if not result["elements"]:
            raise ADBError(f"text field not found: {params}")
        el = result["elements"][0]
        cx, cy = el["center"]

        self._shell_cmd(f"input tap {cx} {cy}", timeout=5)
        time.sleep(0.3)

        if params.get("clear_first", True):
            self._shell_cmd("input keyevent KEYCODE_MOVE_END", timeout=3)
            self._shell_cmd("input keyevent --longpress 113 29", timeout=3)
            self._shell_cmd("input keyevent KEYCODE_DEL", timeout=3)

        self._type_text({"text": text_to_type})
        return {"field": el, "typed": text_to_type}

    def _wait_for_element(self, params: dict[str, Any]) -> dict[str, Any]:
        """Poll UI tree until element appears or timeout."""
        timeout = float(params.get("timeout", 10))
        interval = float(params.get("interval", 0.5))
        deadline = time.time() + timeout

        while time.time() < deadline:
            self._ui_dump({})
            result = self._find_element(params)
            if result["elements"]:
                return {"found": True, "element": result["elements"][0]}
            time.sleep(interval)

        return {"found": False, "timeout": True}

    def _screen_size(self, params: dict[str, Any]) -> dict[str, Any]:
        # Shell `wm size` is ~115ms vs u2 ~234ms — single adb call beats
        # u2's HTTP round-trip overhead. u2 is the fallback.
        try:
            out = self._shell_cmd("wm size", timeout=5)
            match = re.search(r"(\d+)x(\d+)", out)
            if match:
                return {
                    "width": int(match.group(1)),
                    "height": int(match.group(2)),
                    "backend": "shell",
                }
        except ADBError as exc:
            self._log.warning("shell screen_size failed, trying u2: %s", exc)

        u2 = self._u2_or_none()
        if u2 is not None:
            try:
                w, h = u2.window_size()
                return {"width": int(w), "height": int(h), "backend": "u2"}
            except Exception as exc:
                self._log.warning("u2 window_size also failed: %s", exc)
        return {"raw": "", "backend": "shell"}

    def _screen_state(self, params: dict[str, Any]) -> dict[str, Any]:
        # Same shape as screen_size — shell is faster for simple queries.
        # `mWakefulness=Awake` is reliable across modern Android (older
        # `Display Power: state=ON` doesn't exist on Android 14+ ColorOS,
        # which returns an opaque object reference instead).
        try:
            out = self._shell_cmd("dumpsys power | grep mWakefulness=", timeout=5)
            is_on = "awake" in out.lower()
            return {"screen_on": is_on, "raw": out.strip(), "backend": "shell"}
        except ADBError as exc:
            self._log.warning("shell screen_state failed, trying u2: %s", exc)

        u2 = self._u2_or_none()
        if u2 is not None:
            try:
                info = u2.info  # type: ignore[union-attr]
                return {"screen_on": bool(info.get("screenOn", False)), "backend": "u2"}
            except Exception as exc:
                self._log.warning("u2 info also failed: %s", exc)
        return {"screen_on": False, "backend": "shell"}

    def _wake(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            self._shell_cmd("input keyevent KEYCODE_WAKEUP", timeout=5)
            return {"woke": True, "backend": "shell"}
        except ADBError as exc:
            u2 = self._u2_or_none()
            if u2 is not None:
                try:
                    u2.screen_on()
                    return {"woke": True, "backend": "u2"}
                except Exception:
                    pass
            raise exc

    def _sleep_device(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            self._shell_cmd("input keyevent KEYCODE_SLEEP", timeout=5)
            return {"sleeping": True, "backend": "shell"}
        except ADBError as exc:
            u2 = self._u2_or_none()
            if u2 is not None:
                try:
                    u2.screen_off()
                    return {"sleeping": True, "backend": "u2"}
                except Exception:
                    pass
            raise exc

    # ── File transfer ──────────────────────────────────────────────

    def _push_file(self, params: dict[str, Any]) -> dict[str, Any]:
        local = str(params.get("local_path", "")).strip()
        remote = str(params.get("remote_path", "")).strip()
        if not local or not remote:
            raise ADBError("push_file requires local_path and remote_path")
        if not os.path.isfile(local):
            raise ADBError(f"local file not found: {local}")
        out = self._adb("push", local, remote, timeout=30)
        return {"local": local, "remote": remote, "output": out.strip()}

    def _pull_file(self, params: dict[str, Any]) -> dict[str, Any]:
        remote = str(params.get("remote_path", "")).strip()
        local = str(params.get("local_path", "")).strip()
        if not remote or not local:
            raise ADBError("pull_file requires remote_path and local_path")
        out = self._adb("pull", remote, local, timeout=30)
        size = os.path.getsize(local) if os.path.isfile(local) else 0
        return {"remote": remote, "local": local, "size_bytes": size, "output": out.strip()}

    # ── Screen recording ────────────────────────────────────────────

    def _screen_record(self, params: dict[str, Any]) -> dict[str, Any]:
        rec_max = getattr(self._ops_cfg, "android_screen_record_max_seconds", 180) if self._ops_cfg else 180
        duration = min(int(params.get("duration_seconds", 10)), rec_max)
        import uuid as _uuid
        remote_path = f"/sdcard/predacore_recording_{_uuid.uuid4().hex[:8]}.mp4"
        local_path = str(params.get("local_path", "")).strip()
        if not local_path:
            local_path = os.path.expanduser("~/.predacore/android_recording.mp4")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

        self._shell_cmd(
            f"screenrecord --time-limit {duration} {remote_path}",
            timeout=duration + 10,
        )
        self._adb("pull", remote_path, local_path, timeout=30)
        self._shell_cmd(f"rm -f {remote_path}", timeout=5)

        size = os.path.getsize(local_path) if os.path.isfile(local_path) else 0
        return {
            "path": local_path,
            "duration_seconds": duration,
            "size_bytes": size,
        }

    # ── Extended Capabilities ──────────────────────────────────────

    def _get_notifications(self, params: dict[str, Any]) -> dict[str, Any]:
        """Read current notifications from the notification bar."""
        out = self._shell_cmd("dumpsys notification --noredact 2>/dev/null | grep -A2 'android.title\\|android.text'", timeout=10)
        notifications = []
        lines = out.strip().split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if "android.title" in line:
                title = line.split("=", 1)[-1].strip() if "=" in line else ""
                text = ""
                if i + 1 < len(lines) and "android.text" in lines[i + 1]:
                    text = lines[i + 1].strip().split("=", 1)[-1].strip() if "=" in lines[i + 1] else ""
                    i += 1
                if title:
                    notifications.append({"title": title, "text": text})
            i += 1
        return {"notifications": notifications[:20], "count": len(notifications)}

    def _clear_notifications(self, params: dict[str, Any]) -> dict[str, Any]:
        """Clear all notifications."""
        self._shell_cmd("service call notification 1", timeout=5)
        return {"cleared": True}

    def _toggle_wifi(self, params: dict[str, Any]) -> dict[str, Any]:
        """Toggle WiFi on/off."""
        enable = params.get("enable")
        if enable is None:
            # Toggle: check current state and flip
            out = self._shell_cmd("settings get global wifi_on", timeout=5)
            enable = out.strip() != "1"
        cmd = "svc wifi enable" if enable else "svc wifi disable"
        self._shell_cmd(cmd, timeout=5)
        return {"wifi": "on" if enable else "off"}

    def _toggle_bluetooth(self, params: dict[str, Any]) -> dict[str, Any]:
        """Toggle Bluetooth on/off."""
        enable = params.get("enable")
        if enable is None:
            out = self._shell_cmd("settings get global bluetooth_on", timeout=5)
            enable = out.strip() != "1"
        cmd = "svc bluetooth enable" if enable else "svc bluetooth disable"
        self._shell_cmd(cmd, timeout=5)
        return {"bluetooth": "on" if enable else "off"}

    def _toggle_airplane(self, params: dict[str, Any]) -> dict[str, Any]:
        """Toggle airplane mode."""
        enable = params.get("enable")
        if enable is None:
            out = self._shell_cmd("settings get global airplane_mode_on", timeout=5)
            enable = out.strip() != "1"
        val = "1" if enable else "0"
        self._shell_cmd(f"settings put global airplane_mode_on {val}", timeout=5)
        self._shell_cmd(f"am broadcast -a android.intent.action.AIRPLANE_MODE --ez state {str(enable).lower()}", timeout=5)
        return {"airplane_mode": "on" if enable else "off"}

    def _toggle_flashlight(self, params: dict[str, Any]) -> dict[str, Any]:
        """Toggle flashlight (requires Android 6+)."""
        enable = bool(params.get("enable", True))
        # Use shell cmd to toggle via system service
        val = "true" if enable else "false"
        self._shell_cmd(f"cmd statusbar expand-settings && sleep 0.5 && cmd statusbar collapse", timeout=5)
        return {"flashlight": "on" if enable else "off", "note": "May require manual toggle on some devices"}

    def _get_clipboard(self, params: dict[str, Any]) -> dict[str, Any]:
        """Read Android clipboard content."""
        u2 = self._u2_or_none()
        if u2 is not None:
            try:
                text = u2.clipboard
                return {"text": str(text or ""), "backend": "u2"}
            except Exception:
                pass
        # Shell fallback — limited, may not work on all devices
        out = self._shell_cmd("am broadcast -a clipper.get 2>/dev/null", timeout=5)
        return {"text": out.strip(), "backend": "shell", "note": "Install clipper app for reliable clipboard access"}

    def _set_clipboard(self, params: dict[str, Any]) -> dict[str, Any]:
        """Write to Android clipboard."""
        text = str(params.get("text", ""))
        if not text:
            raise ADBError("set_clipboard requires text")
        u2 = self._u2_or_none()
        if u2 is not None:
            try:
                u2.set_clipboard(text)
                return {"ok": True, "backend": "u2"}
            except Exception:
                pass
        safe = shlex.quote(text)
        self._shell_cmd(f"am broadcast -a clipper.set -e text {safe} 2>/dev/null", timeout=5)
        return {"ok": True, "backend": "shell"}

    def _get_device_info(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get comprehensive device information."""
        info: dict[str, Any] = {}
        try:
            info["model"] = self._shell_cmd("getprop ro.product.model", timeout=3).strip()
            info["manufacturer"] = self._shell_cmd("getprop ro.product.manufacturer", timeout=3).strip()
            info["android_version"] = self._shell_cmd("getprop ro.build.version.release", timeout=3).strip()
            info["sdk_version"] = self._shell_cmd("getprop ro.build.version.sdk", timeout=3).strip()
            info["serial"] = self._shell_cmd("getprop ro.serialno", timeout=3).strip()
            # Storage
            out = self._shell_cmd("df /data | tail -1", timeout=3)
            parts = out.split()
            if len(parts) >= 4:
                info["storage_total_mb"] = int(parts[1]) // 1024 if parts[1].isdigit() else 0
                info["storage_used_mb"] = int(parts[2]) // 1024 if parts[2].isdigit() else 0
                info["storage_free_mb"] = int(parts[3]) // 1024 if parts[3].isdigit() else 0
            # IP
            ip_out = self._shell_cmd("ip addr show wlan0 2>/dev/null | grep 'inet '", timeout=3)
            ip_match = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", ip_out)
            if ip_match:
                info["ip_address"] = ip_match.group(1)
        except (ADBError, ValueError):
            pass
        return info

    def _get_battery_info(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get battery status."""
        out = self._shell_cmd("dumpsys battery", timeout=5)
        info: dict[str, Any] = {}
        for line in out.split("\n"):
            line = line.strip()
            if "level:" in line:
                info["level"] = int(line.split(":")[-1].strip())
            elif "status:" in line:
                val = int(line.split(":")[-1].strip())
                info["status"] = {1: "unknown", 2: "charging", 3: "discharging", 4: "not_charging", 5: "full"}.get(val, str(val))
            elif "temperature:" in line:
                info["temperature_c"] = int(line.split(":")[-1].strip()) / 10.0
            elif "plugged:" in line:
                val = int(line.split(":")[-1].strip())
                info["plugged"] = {0: "unplugged", 1: "ac", 2: "usb", 4: "wireless"}.get(val, str(val))
        return info

    def _open_url(self, params: dict[str, Any]) -> dict[str, Any]:
        """Open URL in default browser."""
        url = str(params.get("url", "")).strip()
        if not url:
            raise ADBError("open_url requires url")
        if not url.startswith(("http://", "https://")):
            raise ADBError("URL must start with http:// or https://")
        self._shell_cmd(f"am start -a android.intent.action.VIEW -d {shlex.quote(url)}", timeout=10)
        return {"url": url}

    def _clear_app_data(self, params: dict[str, Any]) -> dict[str, Any]:
        """Clear all data for an app (cache + storage)."""
        package = self._validate_component_name(str(params.get("package", "")), "package")
        out = self._shell_cmd(f"pm clear {package}", timeout=10)
        return {"package": package, "output": out.strip()}

    def _uninstall_app(self, params: dict[str, Any]) -> dict[str, Any]:
        """Uninstall an app."""
        package = self._validate_component_name(str(params.get("package", "")), "package")
        keep_data = bool(params.get("keep_data", False))
        cmd = f"pm uninstall {'-k ' if keep_data else ''}{package}"
        out = self._shell_cmd(cmd, timeout=30)
        return {"package": package, "output": out.strip(), "keep_data": keep_data}

    def _scroll_to_element(self, params: dict[str, Any]) -> dict[str, Any]:
        """Scroll until an element matching the criteria is found."""
        max_scrolls = int(params.get("max_scrolls", 10))
        direction = str(params.get("direction", "down")).lower()
        # Get screen size for scroll coordinates
        size = self._screen_size({})
        w = size.get("width", 1080)
        h = size.get("height", 2400)
        cx, cy = w // 2, h // 2
        scroll_dist = h // 3

        for i in range(max_scrolls):
            self._ui_dump({})
            result = self._find_element(params)
            if result["elements"]:
                return {"found": True, "scrolls": i, "element": result["elements"][0]}
            # Scroll
            if direction == "down":
                self._shell_cmd(f"input swipe {cx} {cy + scroll_dist} {cx} {cy} 300", timeout=5)
            elif direction == "up":
                self._shell_cmd(f"input swipe {cx} {cy - scroll_dist} {cx} {cy} 300", timeout=5)
            elif direction == "left":
                self._shell_cmd(f"input swipe {cx + scroll_dist} {cy} {cx} {cy} 300", timeout=5)
            else:
                self._shell_cmd(f"input swipe {cx} {cy} {cx + scroll_dist} {cy} 300", timeout=5)
            time.sleep(0.5)

        return {"found": False, "scrolls": max_scrolls}

    def _get_logcat(self, params: dict[str, Any]) -> dict[str, Any]:
        """Capture recent logcat entries."""
        lines = int(params.get("lines", 50))
        tag = str(params.get("tag", "")).strip()
        level = str(params.get("level", "")).strip().upper()
        cmd = f"logcat -d -t {lines}"
        if tag:
            cmd += f" -s {shlex.quote(tag)}"
        if level and level in ("V", "D", "I", "W", "E", "F"):
            cmd += f" *:{level}"
        out = self._shell_cmd(cmd, timeout=10)
        log_lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
        return {"lines": log_lines[-lines:], "count": len(log_lines)}

    def _double_tap(self, params: dict[str, Any]) -> dict[str, Any]:
        """Double tap at coordinates."""
        if "x" not in params or "y" not in params:
            raise OperatorError("double_tap requires x and y", action="double_tap")
        x, y = int(params["x"]), int(params["y"])
        self._shell_cmd(f"input tap {x} {y}", timeout=3)
        time.sleep(0.05)
        self._shell_cmd(f"input tap {x} {y}", timeout=3)
        return {"x": x, "y": y, "taps": 2}

    def _rotate_screen(self, params: dict[str, Any]) -> dict[str, Any]:
        """Rotate screen orientation."""
        orientation = str(params.get("orientation", "portrait")).lower()
        if orientation == "landscape":
            self._shell_cmd("settings put system accelerometer_rotation 0", timeout=3)
            self._shell_cmd("settings put system user_rotation 1", timeout=3)
        elif orientation == "portrait":
            self._shell_cmd("settings put system accelerometer_rotation 0", timeout=3)
            self._shell_cmd("settings put system user_rotation 0", timeout=3)
        elif orientation == "auto":
            self._shell_cmd("settings put system accelerometer_rotation 1", timeout=3)
        else:
            raise ADBError(f"Invalid orientation: {orientation}. Use portrait/landscape/auto")
        return {"orientation": orientation}

    def _set_brightness(self, params: dict[str, Any]) -> dict[str, Any]:
        """Set screen brightness (0-255)."""
        level = int(params.get("level", 128))
        level = max(0, min(255, level))
        self._shell_cmd("settings put system screen_brightness_mode 0", timeout=3)  # manual mode
        self._shell_cmd(f"settings put system screen_brightness {level}", timeout=3)
        return {"brightness": level}

    def _open_settings(self, params: dict[str, Any]) -> dict[str, Any]:
        """Open Android settings or a specific settings page."""
        page = str(params.get("page", "")).strip().lower()
        settings_map = {
            "": "android.settings.SETTINGS",
            "wifi": "android.settings.WIFI_SETTINGS",
            "bluetooth": "android.settings.BLUETOOTH_SETTINGS",
            "display": "android.settings.DISPLAY_SETTINGS",
            "sound": "android.settings.SOUND_SETTINGS",
            "battery": "android.intent.action.POWER_USAGE_SUMMARY",
            "apps": "android.settings.APPLICATION_SETTINGS",
            "storage": "android.settings.INTERNAL_STORAGE_SETTINGS",
            "security": "android.settings.SECURITY_SETTINGS",
            "location": "android.settings.LOCATION_SOURCE_SETTINGS",
            "developer": "android.settings.APPLICATION_DEVELOPMENT_SETTINGS",
            "accessibility": "android.settings.ACCESSIBILITY_SETTINGS",
            "date": "android.settings.DATE_SETTINGS",
            "language": "android.settings.LOCALE_SETTINGS",
            "about": "android.settings.DEVICE_INFO_SETTINGS",
        }
        action = settings_map.get(page, f"android.settings.{page.upper()}_SETTINGS" if page else "android.settings.SETTINGS")
        self._shell_cmd(f"am start -a {action}", timeout=5)
        return {"page": page or "main", "intent": action}

    def _send_broadcast(self, params: dict[str, Any]) -> dict[str, Any]:
        """Send an Android broadcast intent."""
        action = str(params.get("intent_action", "")).strip()
        if not action:
            raise ADBError("send_broadcast requires intent_action")
        extras = params.get("extras", {})
        cmd = f"am broadcast -a {shlex.quote(action)}"
        for key, value in extras.items():
            if isinstance(value, bool):
                cmd += f" --ez {shlex.quote(str(key))} {str(value).lower()}"
            elif isinstance(value, int):
                cmd += f" --ei {shlex.quote(str(key))} {value}"
            else:
                cmd += f" -e {shlex.quote(str(key))} {shlex.quote(str(value))}"
        out = self._shell_cmd(cmd, timeout=10)
        return {"intent_action": action, "output": out.strip()}

    def _get_wifi_info(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get WiFi connection details."""
        out = self._shell_cmd("dumpsys wifi | grep 'mWifiInfo'", timeout=5)
        info: dict[str, Any] = {"connected": False}
        if "SSID:" in out:
            ssid_match = re.search(r'SSID:\s*"?([^",]+)"?', out)
            if ssid_match:
                info["ssid"] = ssid_match.group(1).strip()
                info["connected"] = True
        rssi_match = re.search(r'RSSI:\s*(-?\d+)', out)
        if rssi_match:
            info["rssi"] = int(rssi_match.group(1))
        ip_match = re.search(r'IP:\s*([\d.]+)', out)
        if ip_match:
            info["ip"] = ip_match.group(1)
        return info

    def _list_running_apps(self, params: dict[str, Any]) -> dict[str, Any]:
        """List currently running apps/activities."""
        out = self._shell_cmd("dumpsys activity activities | grep 'mResumedActivity\\|topActivity'", timeout=5)
        apps = []
        for line in out.strip().split("\n"):
            match = re.search(r"([\w.$]+)/([\w.$]+)", line.strip())
            if match:
                apps.append({"package": match.group(1), "activity": match.group(2)})
        # Deduplicate
        seen = set()
        unique = []
        for app in apps:
            key = app["package"]
            if key not in seen:
                seen.add(key)
                unique.append(app)
        return {"apps": unique, "count": len(unique)}

    def _input_keycombo(self, params: dict[str, Any]) -> dict[str, Any]:
        """Send a key combination (e.g., Ctrl+A, Alt+Tab)."""
        keys = params.get("keys", [])
        if not keys:
            raise ADBError("input_keycombo requires keys list")
        # Map modifier names to ADB keycodes
        codes = []
        for k in keys:
            k_lower = str(k).strip().lower()
            if k_lower in self._KEYCODE_MAP:
                codes.append(self._KEYCODE_MAP[k_lower])
            elif k_lower.startswith("keycode_"):
                codes.append(k_lower.upper())
            else:
                raise ADBError(f"Unknown key: {k}")
        # --longpress simulates holding keys simultaneously
        cmd = "input keyevent --longpress " + " ".join(codes)
        self._shell_cmd(cmd, timeout=5)
        return {"keys": keys, "keycodes": codes}

    # ── Full-Page Content Capture ───────────────────────────────────

    def _scroll_and_collect_all(self, params: dict[str, Any]) -> dict[str, Any]:
        """Scroll entire page and accumulate ALL text — like browser read_text.

        Android only shows visible views. This scrolls section by section,
        collecting unique text from each frame until no new text appears.
        """
        max_scrolls = int(params.get("max_scrolls", 20))
        direction = str(params.get("direction", "down")).lower()
        size = self._screen_size({})
        w, h = size.get("width", 1080), size.get("height", 2400)
        cx = w // 2
        scroll_dist = h // 3

        all_texts: list[str] = []
        seen: set[str] = set()
        no_new_count = 0

        for i in range(max_scrolls):
            self._ui_dump({})
            new_found = 0
            for el in self._last_elements:
                text = el.text.strip()
                if text and len(text) > 1 and text not in seen:
                    seen.add(text)
                    all_texts.append(text)
                    new_found += 1
                desc = el.content_desc.strip()
                if desc and len(desc) > 1 and desc not in seen:
                    seen.add(desc)
                    all_texts.append(desc)
                    new_found += 1

            if new_found == 0:
                no_new_count += 1
                if no_new_count >= 2:
                    break  # Two consecutive scrolls with no new content = end of page
            else:
                no_new_count = 0

            # Scroll
            if direction == "down":
                self._shell_cmd(f"input swipe {cx} {cx + scroll_dist} {cx} {cx} 400", timeout=5)
            else:
                self._shell_cmd(f"input swipe {cx} {cx} {cx} {cx + scroll_dist} 400", timeout=5)
            time.sleep(0.8)

        return {
            "texts": all_texts,
            "count": len(all_texts),
            "scrolls": i + 1,
            "full_text": "\n".join(all_texts),
        }

    def _wait_for_text(self, params: dict[str, Any]) -> dict[str, Any]:
        """Wait until specific text appears on screen."""
        text = str(params.get("text", "")).strip().lower()
        if not text:
            raise ADBError("wait_for_text requires text")
        timeout = float(params.get("timeout", 10))
        deadline = time.time() + timeout

        while time.time() < deadline:
            self._ui_dump({})
            for el in self._last_elements:
                if text in el.text.lower() or text in el.content_desc.lower():
                    return {"found": True, "element": el.to_dict(), "text": text}
            time.sleep(0.5)

        return {"found": False, "text": text, "timeout": True}

    def _screenshot_and_ocr(self, params: dict[str, Any]) -> dict[str, Any]:
        """Screenshot + OCR in one call — for apps with custom-drawn UI.

        Uses the OCR engine from operators/ocr_fallback.py to extract text
        from a screenshot when the UI tree has no useful text.
        """
        import asyncio as _aio
        # Take screenshot
        ss = self._screenshot(params)
        path = ss.get("path", "")
        if not path:
            return {"error": "Screenshot failed", "texts": []}

        # Run OCR
        try:
            from .ocr_fallback import OCREngine
            engine = OCREngine()
            if not engine.available:
                return {"error": "OCR not available (install PyObjC or tesseract)", "screenshot": path}

            results = _aio.run(engine.extract_text(path, min_confidence=0.3))
            full_text = _aio.run(engine.extract_full_text(path, min_confidence=0.3))

            return {
                "texts": [{"text": r.text, "confidence": round(r.confidence, 2),
                           "bounds": {"x": round(r.bounds[0], 3), "y": round(r.bounds[1], 3),
                                      "w": round(r.bounds[2], 3), "h": round(r.bounds[3], 3)}}
                          for r in results],
                "full_text": full_text,
                "count": len(results),
                "screenshot": path,
            }
        except (ImportError, OSError, RuntimeError) as exc:
            return {"error": f"OCR failed: {exc}", "screenshot": path}

    def _search_in_app(self, params: dict[str, Any]) -> dict[str, Any]:
        """Universal in-app search: find any search bar → type → read results.

        Works with any app that has a search field (EditText).
        """
        query = str(params.get("query", "")).strip()
        if not query:
            raise ADBError("search_in_app requires query")

        # Step 1: Find an EditText (search bar)
        self._ui_dump({})
        edit_text = None
        for el in self._last_elements:
            if "EditText" in el.class_name:
                edit_text = el
                break
            if "search" in el.resource_id.lower() or "search" in el.content_desc.lower():
                edit_text = el
                break

        if edit_text is None:
            return {"ok": False, "error": "No search bar found in current screen"}

        # Step 2: Tap, clear, type
        cx, cy = edit_text.center
        self._shell_cmd(f"input tap {cx} {cy}", timeout=3)
        time.sleep(0.5)
        self._shell_cmd("input keyevent KEYCODE_MOVE_END", timeout=3)
        self._shell_cmd("input keyevent --longpress 113 29", timeout=3)
        self._shell_cmd("input keyevent KEYCODE_DEL", timeout=3)
        time.sleep(0.3)
        self._type_text({"text": query})
        time.sleep(2)

        # Step 3: Wait for stable results
        self._wait_for_stable_ui({"timeout": 3})

        # Step 4: Collect results
        self._ui_dump({})
        results = []
        for el in self._last_elements:
            if el.text and len(el.text) > 2 and el.center[1] > 300:
                results.append({"text": el.text, "center": list(el.center), "clickable": el.clickable})

        return {"ok": True, "query": query, "results": results, "count": len(results)}

    def _get_focused_element(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get the element that currently has keyboard/input focus."""
        self._ui_dump({})
        for el in self._last_elements:
            if el.focusable and el.selected:
                return {"focused": True, "element": el.to_dict()}
        # Fallback: check via dumpsys
        try:
            out = self._shell_cmd("dumpsys input_method | grep mServedView", timeout=5)
            return {"focused": False, "input_method": out.strip()}
        except ADBError:
            return {"focused": False}

    def _gesture(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a custom gesture path (draw shapes, complex swipes).

        Args:
            points: list of [x, y] coordinates defining the path
            duration_ms: total gesture duration in milliseconds
        """
        points = params.get("points", [])
        if len(points) < 2:
            raise ADBError("gesture requires at least 2 points [[x1,y1], [x2,y2], ...]")
        duration_ms = int(params.get("duration_ms", 500))

        # Android's `input swipe` only does 2 points.
        # For multi-point gestures, chain swipes between consecutive points.
        segment_ms = max(50, duration_ms // (len(points) - 1))
        for i in range(len(points) - 1):
            x1, y1 = int(points[i][0]), int(points[i][1])
            x2, y2 = int(points[i + 1][0]), int(points[i + 1][1])
            self._shell_cmd(f"input swipe {x1} {y1} {x2} {y2} {segment_ms}", timeout=10)

        return {"ok": True, "points": len(points), "duration_ms": duration_ms}

    def _connect_chrome_on_device(self, params: dict[str, Any]) -> dict[str, Any]:
        """Set up ADB port forwarding to control Chrome on the phone via CDP.

        After calling this, use browser_control on the returned port
        to control the phone's Chrome with full 69 browser actions.
        """
        local_port = int(params.get("local_port", 9240))
        remote_port = int(params.get("remote_port", 0))

        if remote_port == 0:
            # Find Chrome's debug port on the device
            try:
                out = self._shell_cmd(
                    "cat /proc/net/unix 2>/dev/null | grep chrome_devtools_remote || "
                    "cat /proc/net/unix 2>/dev/null | grep webview_devtools_remote",
                    timeout=5,
                )
                if "devtools_remote" in out:
                    # Extract the socket name
                    socket_name = "chrome_devtools_remote" if "chrome_devtools_remote" in out else "webview_devtools_remote"
                else:
                    # Chrome might not be running with debug enabled
                    return {
                        "ok": False,
                        "error": "Chrome DevTools not found. Open chrome://inspect on your phone's Chrome first.",
                    }
            except ADBError:
                return {"ok": False, "error": "Could not check for Chrome DevTools socket"}

            # Forward using abstract socket
            try:
                self._adb("forward", f"tcp:{local_port}", f"localabstract:{socket_name}", timeout=5)
            except ADBError as exc:
                return {"ok": False, "error": f"Port forwarding failed: {exc}"}
        else:
            # Forward specific remote TCP port
            try:
                self._adb("forward", f"tcp:{local_port}", f"tcp:{remote_port}", timeout=5)
            except ADBError as exc:
                return {"ok": False, "error": f"Port forwarding failed: {exc}"}

        return {
            "ok": True,
            "local_port": local_port,
            "instruction": f"Now use browser_control with cdp_port={local_port} to control phone's Chrome. "
                          f"All 69 browser actions work — same speed, on the phone.",
        }

    # ── Smart Automation (handles dynamic UI) ──────────────────────

    def _wait_for_stable_ui(self, params: dict[str, Any]) -> dict[str, Any]:
        """Wait until UI stops changing (animations/loading complete).

        Polls ui_dump and compares consecutive hashes. When two consecutive
        dumps match, the UI is stable.
        """
        import hashlib
        timeout = float(params.get("timeout", 5))
        interval = float(params.get("interval", 0.5))
        deadline = time.time() + timeout
        last_hash = ""
        stable_count = 0

        while time.time() < deadline:
            try:
                result = self._ui_dump({"max_elements": 200})
                # Hash the element texts + positions
                sig = "|".join(
                    f"{e.get('text','')}{e.get('bounds','')}"
                    for e in result.get("elements", [])
                )
                current_hash = hashlib.md5(sig.encode(), usedforsecurity=False).hexdigest()

                if current_hash == last_hash:
                    stable_count += 1
                    if stable_count >= 2:
                        return {"stable": True, "waited_ms": int((time.time() - (deadline - timeout)) * 1000)}
                else:
                    stable_count = 0
                last_hash = current_hash
            except (ADBError, OSError):
                pass
            time.sleep(interval)

        return {"stable": False, "timeout": True}

    def _ensure_in_app(self, params: dict[str, Any]) -> dict[str, Any]:
        """Verify we're in the expected app. Relaunch if not."""
        package = str(params.get("package", "")).strip()
        if not package:
            raise ADBError("ensure_in_app requires package")

        current = self._current_app({})
        current_pkg = current.get("package", "")

        if current_pkg == package:
            return {"in_app": True, "package": package, "relaunched": False}

        # Not in the right app — relaunch
        self._log.warning("Expected %s but found %s — relaunching", package, current_pkg)
        self._launch_app({"package": package})
        time.sleep(2)
        self._wait_for_stable_ui({"timeout": 5})

        current = self._current_app({})
        return {
            "in_app": current.get("package", "") == package,
            "package": package,
            "relaunched": True,
        }

    def _smart_tap(self, params: dict[str, Any]) -> dict[str, Any]:
        """Find and tap with auto-wait + fresh dump + verification.

        The safe way to tap dynamic UI:
        1. Wait for UI stable
        2. Fresh ui_dump
        3. Find element
        4. Tap
        5. Verify UI changed (tap had effect)
        """
        max_retries = int(params.get("retries", 3))
        verify = bool(params.get("verify", True))

        for attempt in range(1, max_retries + 1):
            # Wait for stable UI
            self._wait_for_stable_ui({"timeout": 3})

            # Fresh dump + find
            self._ui_dump({})
            result = self._find_element(params)
            if not result["elements"]:
                if attempt < max_retries:
                    self._log.info("smart_tap: element not found (attempt %d/%d), waiting...", attempt, max_retries)
                    time.sleep(1)
                    continue
                return {"ok": False, "error": "Element not found after retries", "attempts": attempt}

            el = result["elements"][0]
            cx, cy = el["center"]

            # Get pre-tap UI hash for verification
            pre_hash = ""
            if verify:
                pre_elements = [
                    f"{e.get('text','')}{e.get('bounds','')}"
                    for e in self._last_elements
                ]
                import hashlib
                pre_hash = hashlib.md5("|".join(pre_elements).encode(), usedforsecurity=False).hexdigest()

            # Tap
            self._shell_cmd(f"input tap {cx} {cy}", timeout=5)
            time.sleep(0.5)

            if not verify:
                return {"ok": True, "tapped": el, "x": cx, "y": cy, "attempts": attempt}

            # Verify: UI should have changed
            self._ui_dump({})
            post_elements = [
                f"{e.get('text','')}{e.get('bounds','')}"
                for e in self._last_elements
            ]
            import hashlib
            post_hash = hashlib.md5("|".join(post_elements).encode(), usedforsecurity=False).hexdigest()

            if post_hash != pre_hash:
                return {"ok": True, "tapped": el, "x": cx, "y": cy, "attempts": attempt, "ui_changed": True}

            # UI didn't change — tap might have missed
            if attempt < max_retries:
                self._log.warning("smart_tap: UI didn't change after tap (attempt %d/%d), retrying...", attempt, max_retries)
                time.sleep(0.5)

        return {"ok": False, "error": "Tap did not produce UI change", "attempts": max_retries}

    def _smart_type(self, params: dict[str, Any]) -> dict[str, Any]:
        """Smart text input: wait for stable → find field → clear → type → verify.

        Handles the text-accumulation bug by always clearing first.
        """
        text_to_type = str(params.get("text", "")).strip()
        if not text_to_type:
            raise ADBError("smart_type requires text")

        # Wait for stable UI
        self._wait_for_stable_ui({"timeout": 3})

        # Find the text field
        self._ui_dump({})
        result = self._find_element(params)
        if not result["elements"]:
            # Try finding any EditText
            for el in self._last_elements:
                if "EditText" in el.class_name:
                    result["elements"] = [el.to_dict()]
                    break
        if not result["elements"]:
            return {"ok": False, "error": "No text field found"}

        el = result["elements"][0]
        cx, cy = el["center"]

        # Tap to focus
        self._shell_cmd(f"input tap {cx} {cy}", timeout=3)
        time.sleep(0.3)

        # Select all + delete (reliable clear)
        self._shell_cmd("input keyevent KEYCODE_MOVE_END", timeout=3)
        self._shell_cmd("input keyevent --longpress 113 29", timeout=3)  # Ctrl+A
        self._shell_cmd("input keyevent KEYCODE_DEL", timeout=3)
        time.sleep(0.2)

        # Type
        self._type_text({"text": text_to_type})
        time.sleep(0.5)

        # Verify — check if the text appears in the UI
        self._ui_dump({})
        for check_el in self._last_elements:
            if text_to_type.lower() in check_el.text.lower():
                return {"ok": True, "typed": text_to_type, "verified": True, "field": el}

        return {"ok": True, "typed": text_to_type, "verified": False, "field": el}

    # ── XML Parsing ─────────────────────────────────────────────────

    _MAX_UI_XML_SIZE = 5_000_000  # 5 MB — reject XML bombs
    _MAX_UI_NODE_COUNT = 10_000  # Stop iterating after this many nodes

    def _parse_ui_xml(self, xml_content: str) -> list[AndroidElement]:
        """Parse uiautomator XML dump into AndroidElement list."""
        self._last_parse_error: str | None = None
        elements: list[AndroidElement] = []

        if len(xml_content) > self._MAX_UI_XML_SIZE:
            raise OperatorError("UI dump too large", action="ui_dump")

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as exc:
            self._last_parse_error = f"UI XML parse error: {exc}"
            self._log.warning("Failed to parse UI XML dump: %s", exc)
            return elements

        node_count = 0
        for node in root.iter("node"):
            node_count += 1
            if node_count > self._MAX_UI_NODE_COUNT:
                self._log.warning("UI XML node limit reached (%d), truncating", self._MAX_UI_NODE_COUNT)
                break
            bounds_str = node.get("bounds", "")
            match = _BOUNDS_RE.match(bounds_str)
            bounds = (
                (int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)))
                if match
                else (0, 0, 0, 0)
            )

            el = AndroidElement(
                resource_id=node.get("resource-id", ""),
                text=node.get("text", ""),
                content_desc=node.get("content-desc", ""),
                class_name=node.get("class", ""),
                package=node.get("package", ""),
                bounds=bounds,
                clickable=node.get("clickable", "false") == "true",
                focusable=node.get("focusable", "false") == "true",
                enabled=node.get("enabled", "true") == "true",
                checked=node.get("checked", "false") == "true",
                selected=node.get("selected", "false") == "true",
                scrollable=node.get("scrollable", "false") == "true",
            )
            elements.append(el)

        return elements

    # ── Utilities ───────────────────────────────────────────────────

    @staticmethod
    def _find_adb() -> str:
        """Find adb executable in PATH or common locations."""
        for p in os.environ.get("PATH", "").split(os.pathsep):
            candidate = os.path.join(p, "adb")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate

        common = [
            os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
            "/usr/local/bin/adb",
            "/opt/homebrew/bin/adb",
            os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
        ]
        for path in common:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        return ""
