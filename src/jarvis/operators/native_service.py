"""
Native macOS desktop control service built on PyObjC.

This backend uses AppKit + Quartz + ApplicationServices to provide:
- app launch/focus/url open
- keyboard/mouse input events
- Accessibility (AX) element queries and actions
"""
from __future__ import annotations

import logging
import platform
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DesktopNativeError(RuntimeError):
    """Raised when native desktop control fails."""


@dataclass
class _AXMatchResult:
    element: Any
    info: dict[str, Any]
    depth: int


class MacDesktopNativeService:
    """Native macOS desktop control/AX service."""

    _KEY_CODE_MAP = {
        "enter": 36,
        "return": 36,
        "tab": 48,
        "space": 49,
        "delete": 51,
        "backspace": 51,
        "escape": 53,
        "esc": 53,
        "left": 123,
        "right": 124,
        "down": 125,
        "up": 126,
    }

    _CHAR_KEY_CODE_MAP = {
        "a": 0,
        "s": 1,
        "d": 2,
        "f": 3,
        "h": 4,
        "g": 5,
        "z": 6,
        "x": 7,
        "c": 8,
        "v": 9,
        "b": 11,
        "q": 12,
        "w": 13,
        "e": 14,
        "r": 15,
        "y": 16,
        "t": 17,
        "1": 18,
        "2": 19,
        "3": 20,
        "4": 21,
        "6": 22,
        "5": 23,
        "=": 24,
        "9": 25,
        "7": 26,
        "-": 27,
        "8": 28,
        "0": 29,
        "]": 30,
        "o": 31,
        "u": 32,
        "[": 33,
        "i": 34,
        "p": 35,
        "l": 37,
        "j": 38,
        "'": 39,
        "k": 40,
        ";": 41,
        "\\": 42,
        ",": 43,
        "/": 44,
        "n": 45,
        "m": 46,
        ".": 47,
        "`": 50,
    }

    _MODIFIER_FLAG_NAMES = {
        "command": "kCGEventFlagMaskCommand",
        "cmd": "kCGEventFlagMaskCommand",
        "shift": "kCGEventFlagMaskShift",
        "option": "kCGEventFlagMaskAlternate",
        "alt": "kCGEventFlagMaskAlternate",
        "control": "kCGEventFlagMaskControl",
        "ctrl": "kCGEventFlagMaskControl",
        "fn": "kCGEventFlagMaskSecondaryFn",
    }

    _AX_ATTRIBUTE_KEYS = (
        ("role", "kAXRoleAttribute"),
        ("subrole", "kAXSubroleAttribute"),
        ("title", "kAXTitleAttribute"),
        ("description", "kAXDescriptionAttribute"),
        ("value", "kAXValueAttribute"),
        ("enabled", "kAXEnabledAttribute"),
        ("position", "kAXPositionAttribute"),
        ("size", "kAXSizeAttribute"),
    )

    def __init__(self, log: logging.Logger | None = None) -> None:
        self._log = log or logger
        self._is_macos = platform.system() == "Darwin"
        self._available = False
        self._init_error: str | None = None
        self._trust_cache_time: float = 0.0
        self._trust_cached: bool = False
        # App allowlist (mirrors desktop.py enforcement)
        import os as _os
        _raw = _os.environ.get("JARVIS_DESKTOP_ALLOWED_APPS", "").strip()
        self._allowed_apps: set[str] = (
            {a.strip().lower() for a in _raw.split(",") if a.strip()} if _raw else set()
        )
        self._quartz = None
        self._appkit = None
        self._foundation = None
        self._ax = None
        self._activate_ignore = 0
        self._ax_error_names: dict[int, str] = {}
        if not self._is_macos:
            self._init_error = "native desktop service is only available on macOS"
            return
        try:
            import AppKit  # type: ignore
            import Foundation  # type: ignore
            import Quartz  # type: ignore
            from ApplicationServices import (  # type: ignore
                AXIsProcessTrusted,
                AXIsProcessTrustedWithOptions,
                AXUIElementCopyAttributeValue,
                AXUIElementCreateApplication,
                AXUIElementCreateSystemWide,
                AXUIElementPerformAction,
                AXUIElementSetAttributeValue,
                AXValueCreate,
                kAXErrorSuccess,
                kAXTrustedCheckOptionPrompt,
                kAXValueTypeCGPoint,
                kAXValueTypeCGSize,
            )

            self._quartz = Quartz
            self._appkit = AppKit
            self._foundation = Foundation
            self._ax = {
                "AXIsProcessTrusted": AXIsProcessTrusted,
                "AXIsProcessTrustedWithOptions": AXIsProcessTrustedWithOptions,
                "AXUIElementCopyAttributeValue": AXUIElementCopyAttributeValue,
                "AXUIElementCreateApplication": AXUIElementCreateApplication,
                "AXUIElementCreateSystemWide": AXUIElementCreateSystemWide,
                "AXUIElementPerformAction": AXUIElementPerformAction,
                "AXUIElementSetAttributeValue": AXUIElementSetAttributeValue,
                "AXValueCreate": AXValueCreate,
                "kAXErrorSuccess": kAXErrorSuccess,
                "kAXTrustedCheckOptionPrompt": kAXTrustedCheckOptionPrompt,
                "kAXValueTypeCGPoint": kAXValueTypeCGPoint,
                "kAXValueTypeCGSize": kAXValueTypeCGSize,
            }
            self._activate_ignore = int(
                getattr(AppKit, "NSApplicationActivateIgnoringOtherApps", 0)
            )
            for name in dir(AppKit):
                if name.startswith("kAX"):
                    value = getattr(AppKit, name, None)
                    if isinstance(value, str):
                        self._ax[name] = value
            _app_services = __import__("ApplicationServices")
            for name in dir(_app_services):
                if name.startswith("kAX"):
                    value = getattr(_app_services, name, None)
                    if isinstance(value, str | int):
                        self._ax[name] = value
                if name.startswith("kAXError"):
                    value = getattr(_app_services, name, None)
                    if isinstance(value, int):
                        self._ax_error_names[value] = name
            self._available = True
        except (OSError, ImportError, RuntimeError, AttributeError) as exc:
            self._init_error = str(exc)
            self._available = False

    @property
    def available(self) -> bool:
        """Return True if the native PyObjC service is available."""
        return self._available

    @property
    def init_error(self) -> str | None:
        """Return the initialization error message, if any."""
        return self._init_error

    def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a native macOS action via PyObjC Accessibility API."""
        if not self._available:
            raise DesktopNativeError(
                self._init_error or "native desktop service unavailable"
            )
        action = str(action or "").strip().lower()
        if action == "open_app":
            return self._open_app(params)
        if action == "focus_app":
            return self._focus_app(params)
        if action == "open_url":
            return self._open_url(params)
        if action == "type_text":
            return self._type_text(params)
        if action == "press_key":
            return self._press_key(params)
        if action == "mouse_move":
            return self._mouse_move(params)
        if action == "mouse_click":
            return self._mouse_click(params, button="left", clicks=1)
        if action == "mouse_double_click":
            return self._mouse_click(params, button="left", clicks=2)
        if action == "mouse_right_click":
            return self._mouse_click(params, button="right", clicks=1)
        if action == "mouse_scroll":
            return self._mouse_scroll(params)
        if action == "get_mouse_position":
            return self._get_mouse_position()
        if action == "frontmost_app":
            return self._frontmost_app()
        if action == "ax_query":
            return self._ax_query(params)
        if action == "ax_click":
            return self._ax_click(params)
        if action == "ax_set_value":
            return self._ax_set_value(params)
        if action == "ax_request_access":
            return self._ax_request_access(params)
        if action == "health_check":
            return self.health_check(params)
        if action == "clipboard_read":
            return self._clipboard_read(params)
        if action == "clipboard_write":
            return self._clipboard_write(params)
        if action == "list_windows":
            return self._list_windows(params)
        if action == "move_window":
            return self._move_window(params)
        if action == "resize_window":
            return self._resize_window(params)
        if action == "minimize_window":
            return self._minimize_window(params)
        if action == "list_monitors":
            return self._list_monitors(params)
        if action == "mouse_drag":
            return self._mouse_drag(params)
        if action == "click_menu":
            return self._click_menu(params)
        if action == "set_volume":
            return self._set_volume(params)
        if action == "get_volume":
            return self._get_volume(params)
        if action == "set_brightness":
            return self._set_brightness(params)
        if action == "get_brightness":
            return self._get_brightness(params)
        if action == "move_to_monitor":
            return self._move_to_monitor(params)
        # System info & control actions
        if action == "list_apps":
            return self._list_apps(params)
        if action == "force_quit_app":
            return self._force_quit_app(params)
        if action == "get_system_info":
            return self._get_system_info(params)
        if action == "toggle_dark_mode":
            return self._toggle_dark_mode(params)
        if action == "get_dark_mode":
            return self._get_dark_mode(params)
        if action == "toggle_dnd":
            return self._toggle_dnd(params)
        if action == "spotlight_search":
            return self._spotlight_search(params)
        if action == "open_file":
            return self._open_file(params)
        if action == "get_finder_selection":
            return self._get_finder_selection(params)
        if action == "screen_record_start":
            return self._screen_record_start(params)
        if action == "screen_record_stop":
            return self._screen_record_stop(params)
        if action == "tile_windows":
            return self._tile_windows(params)
        if action == "app_switch":
            return self._app_switch(params)
        if action == "get_running_processes":
            return self._get_running_processes(params)
        raise DesktopNativeError(f"unsupported native action: {action}")

    def health_check(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return native service health diagnostics."""
        params = params or {}
        request_access = bool(params.get("request_access", False))
        trusted = False
        trust_error: str | None = None
        try:
            if request_access:
                trusted = bool(
                    self._ax["AXIsProcessTrustedWithOptions"](
                        {self._ax["kAXTrustedCheckOptionPrompt"]: True}
                    )
                )
            else:
                trusted = bool(self._ax["AXIsProcessTrusted"]())
        except (OSError, RuntimeError) as exc:
            trust_error = str(exc)
        return {
            "backend": "pyobjc_native",
            "available": self._available,
            "init_error": self._init_error,
            "accessibility_trusted": trusted,
            "accessibility_error": trust_error,
        }

    def _is_trusted(self) -> bool:
        now = time.monotonic()
        if now - self._trust_cache_time < 5.0:  # 5s TTL
            return self._trust_cached
        self._trust_cached = self.health_check({"request_access": False}).get("accessibility_trusted", False)
        self._trust_cache_time = now
        return self._trust_cached

    def _check_allowed_app(self, app_name: str) -> None:
        """Enforce JARVIS_DESKTOP_ALLOWED_APPS policy (mirrors desktop.py)."""
        if not self._allowed_apps:
            return
        if app_name.strip().lower() not in self._allowed_apps:
            raise DesktopNativeError(
                f"app '{app_name}' blocked by JARVIS_DESKTOP_ALLOWED_APPS policy"
            )

    def _open_app(self, params: dict[str, Any]) -> dict[str, Any]:
        app_name = str(params.get("app_name") or params.get("app") or "").strip()
        if not app_name:
            raise DesktopNativeError("open_app requires app_name")
        self._check_allowed_app(app_name)
        ws = self._appkit.NSWorkspace.sharedWorkspace()
        ok = bool(ws.launchApplication_(app_name))
        if not ok:
            raise DesktopNativeError(f"failed to launch app: {app_name}")
        return {"app_name": app_name}

    def _focus_app(self, params: dict[str, Any]) -> dict[str, Any]:
        app_name = str(params.get("app_name") or params.get("app") or "").strip()
        if not app_name:
            raise DesktopNativeError("focus_app requires app_name")
        self._check_allowed_app(app_name)
        ws = self._appkit.NSWorkspace.sharedWorkspace()
        matches = []
        for app in ws.runningApplications():
            name = str(app.localizedName() or "")
            if name.lower() == app_name.lower():
                matches.append(app)
        if not matches:
            if bool(params.get("launch_if_missing", False)):
                self._open_app({"app_name": app_name})
                wait = float(params.get("launch_wait_seconds") or 0.5)
                # Bounded retry — app may take a moment to register
                for _attempt in range(3):
                    time.sleep(wait)
                    for app in ws.runningApplications():
                        if str(app.localizedName() or "").lower() == app_name.lower():
                            app.activateWithOptions_(self._activate_ignore)
                            return {
                                "app_name": str(app.localizedName() or app_name),
                                "bundle_id": str(app.bundleIdentifier() or ""),
                                "pid": int(app.processIdentifier()),
                            }
                raise DesktopNativeError(
                    f"app launched but not found after retries: {app_name}"
                )
            raise DesktopNativeError(f"app not running: {app_name}")
        app = matches[0]
        app.activateWithOptions_(self._activate_ignore)
        return {
            "app_name": str(app.localizedName() or app_name),
            "bundle_id": str(app.bundleIdentifier() or ""),
            "pid": int(app.processIdentifier()),
        }

    def _open_url(self, params: dict[str, Any]) -> dict[str, Any]:
        url = str(params.get("url") or "").strip()
        if not url:
            raise DesktopNativeError("open_url requires url")
        # Validate URL scheme — only allow safe schemes (matches desktop.py)
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme and parsed.scheme.lower() not in ("http", "https", "mailto"):
            raise DesktopNativeError(
                f"blocked url scheme '{parsed.scheme}' — only http, https, mailto allowed"
            )
        ws = self._appkit.NSWorkspace.sharedWorkspace()
        url_obj = self._foundation.NSURL.URLWithString_(url)
        if url_obj is None:
            raise DesktopNativeError(f"invalid url: {url}")
        ok = bool(ws.openURL_(url_obj))
        if not ok:
            raise DesktopNativeError(f"failed to open url: {url}")
        return {"url": url}

    def _type_text(self, params: dict[str, Any]) -> dict[str, Any]:
        text = str(params.get("text") or "")
        if text == "":
            raise DesktopNativeError("type_text requires text")
        app_name = str(params.get("app_name") or "").strip()
        if app_name:
            self._focus_app({"app_name": app_name, "launch_if_missing": False})
            time.sleep(float(params.get("focus_delay_seconds") or 0.05))
        # Default 0.5ms delay between chars to avoid dropped keystrokes in some apps
        per_char_delay = float(params.get("per_char_delay_seconds") or 0.0005)
        for ch in text:
            down = self._quartz.CGEventCreateKeyboardEvent(None, 0, True)
            self._quartz.CGEventKeyboardSetUnicodeString(down, len(ch), ch)
            self._quartz.CGEventPost(self._quartz.kCGHIDEventTap, down)
            up = self._quartz.CGEventCreateKeyboardEvent(None, 0, False)
            self._quartz.CGEventKeyboardSetUnicodeString(up, len(ch), ch)
            self._quartz.CGEventPost(self._quartz.kCGHIDEventTap, up)
            if per_char_delay > 0:
                time.sleep(per_char_delay)
        return {"typed_chars": len(text), "app_name": app_name or None}

    def _press_key(self, params: dict[str, Any]) -> dict[str, Any]:
        key = str(params.get("key") or "").strip().lower()
        key_code_raw = params.get("key_code")
        if not key and key_code_raw is None:
            raise DesktopNativeError("press_key requires key or key_code")
        modifiers = params.get("modifiers") or []
        flags = self._modifier_flags(modifiers)

        if key_code_raw is not None:
            try:
                key_code = int(key_code_raw)
            except (TypeError, ValueError) as exc:
                raise DesktopNativeError("key_code must be an integer") from exc
        else:
            key_code = self._key_code_for_key(key)

        app_name = str(params.get("app_name") or "").strip()
        if app_name:
            self._focus_app({"app_name": app_name})
            time.sleep(float(params.get("focus_delay_seconds") or 0.05))

        down = self._quartz.CGEventCreateKeyboardEvent(None, key_code, True)
        if flags:
            self._quartz.CGEventSetFlags(down, flags)
        self._quartz.CGEventPost(self._quartz.kCGHIDEventTap, down)
        up = self._quartz.CGEventCreateKeyboardEvent(None, key_code, False)
        if flags:
            self._quartz.CGEventSetFlags(up, flags)
        self._quartz.CGEventPost(self._quartz.kCGHIDEventTap, up)
        return {"key": key or None, "key_code": key_code, "modifiers": modifiers}

    def _mouse_move(self, params: dict[str, Any]) -> dict[str, Any]:
        if params.get("x") is None or params.get("y") is None:
            raise DesktopNativeError("mouse_move requires x and y coordinates")
        x = int(params["x"])
        y = int(params["y"])
        duration = min(float(params.get("duration_seconds") or 0.0), 5.0)
        if duration <= 0:
            self._quartz.CGWarpMouseCursorPosition((x, y))
        else:
            current = self._mouse_location()
            steps = max(2, int(duration * 60))
            for step in range(1, steps + 1):
                ratio = step / steps
                px = current[0] + (x - current[0]) * ratio
                py = current[1] + (y - current[1]) * ratio
                self._quartz.CGWarpMouseCursorPosition((px, py))
                time.sleep(duration / steps)
        return {"x": x, "y": y, "duration_seconds": duration}

    def _mouse_click(
        self, params: dict[str, Any], *, button: str, clicks: int
    ) -> dict[str, Any]:
        x_raw = params.get("x")
        y_raw = params.get("y")
        if x_raw is None or y_raw is None:
            point = self._mouse_location()
        else:
            point = (float(x_raw), float(y_raw))
            self._quartz.CGWarpMouseCursorPosition(point)
        event_down, event_up, mouse_button = self._mouse_event_types(button)

        for i in range(clicks):
            down = self._quartz.CGEventCreateMouseEvent(
                None,
                event_down,
                point,
                mouse_button,
            )
            self._quartz.CGEventSetIntegerValueField(
                down, self._quartz.kCGMouseEventClickState, i + 1
            )
            self._quartz.CGEventPost(self._quartz.kCGHIDEventTap, down)

            up = self._quartz.CGEventCreateMouseEvent(
                None,
                event_up,
                point,
                mouse_button,
            )
            self._quartz.CGEventSetIntegerValueField(
                up, self._quartz.kCGMouseEventClickState, i + 1
            )
            self._quartz.CGEventPost(self._quartz.kCGHIDEventTap, up)
            time.sleep(0.02)
        return {
            "x": int(point[0]),
            "y": int(point[1]),
            "button": button,
            "clicks": clicks,
        }

    def _mouse_scroll(self, params: dict[str, Any]) -> dict[str, Any]:
        # Quartz: positive = scroll up, negative = scroll down.
        # Our API convention: positive amount = scroll down (natural).
        amount = int(params.get("amount") or 0)
        ev = self._quartz.CGEventCreateScrollWheelEvent(
            None,
            self._quartz.kCGScrollEventUnitLine,
            1,
            -amount,  # Invert: positive input = scroll down = negative Quartz value
        )
        self._quartz.CGEventPost(self._quartz.kCGHIDEventTap, ev)
        return {"amount": amount}

    def _get_mouse_position(self) -> dict[str, Any]:
        x, y = self._mouse_location()
        return {"x": int(x), "y": int(y)}

    def _frontmost_app(self) -> dict[str, Any]:
        ws = self._appkit.NSWorkspace.sharedWorkspace()
        app = ws.frontmostApplication()
        if app is None:
            raise DesktopNativeError("unable to resolve frontmost app")
        return {
            "app_name": str(app.localizedName() or ""),
            "bundle_id": str(app.bundleIdentifier() or ""),
            "pid": int(app.processIdentifier()),
        }

    def _ax_query(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._is_trusted():
            raise DesktopNativeError(
                "Accessibility not trusted for this process; use ax_request_access"
            )
        root = self._ax_root(params)
        max_depth = max(0, int(params.get("max_depth") or 2))
        max_children = max(1, int(params.get("max_children") or 25))
        snapshot = self._ax_snapshot(
            root, depth=0, max_depth=max_depth, max_children=max_children
        )
        return {
            "root_target": str(params.get("target") or "focused_window"),
            "snapshot": snapshot,
            "max_depth": max_depth,
            "max_children": max_children,
        }

    def _ax_click(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._is_trusted():
            raise DesktopNativeError(
                "Accessibility not trusted for this process; use ax_request_access"
            )
        match = params.get("match")
        # Accept target string as shorthand — convert to match dict
        if not match and params.get("target"):
            match = {"AXTitle": params["target"]}
        if not isinstance(match, dict) or not match:
            raise DesktopNativeError("ax_click requires match object or target string")
        params["match"] = match
        # Retry on transient AX errors during app transitions
        import time as _time
        for attempt in range(3):
            try:
                root = self._ax_root(params)
                break
            except Exception as e:
                if "kAXErrorCannotComplete" in str(e) and attempt < 2:
                    _time.sleep(0.3)
                    continue
                raise
        max_depth = max(0, int(params.get("max_depth") or 6))
        max_children = max(1, int(params.get("max_children") or 50))
        found = self._ax_find_first(
            root,
            match=match,
            depth=0,
            max_depth=max_depth,
            max_children=max_children,
        )
        if found is None:
            raise DesktopNativeError("no AX element matched query")
        err = self._ax["AXUIElementPerformAction"](
            found.element, self._ax.get("kAXPressAction", "AXPress")
        )
        if int(err) != int(self._ax["kAXErrorSuccess"]):
            raise DesktopNativeError(
                f"AX press failed with {self._ax_error_label(int(err))}"
            )
        return {
            "matched": found.info,
            "depth": found.depth,
            "action": "AXPress",
        }

    def _ax_set_value(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._is_trusted():
            raise DesktopNativeError(
                "Accessibility not trusted for this process; use ax_request_access"
            )
        match = params.get("match")
        if not isinstance(match, dict) or not match:
            raise DesktopNativeError("ax_set_value requires match object")
        if "value" not in params:
            raise DesktopNativeError("ax_set_value requires value")
        value = str(params.get("value"))
        root = self._ax_root(params)
        max_depth = max(0, int(params.get("max_depth") or 6))
        max_children = max(1, int(params.get("max_children") or 50))
        found = self._ax_find_first(
            root,
            match=match,
            depth=0,
            max_depth=max_depth,
            max_children=max_children,
        )
        if found is None:
            raise DesktopNativeError("no AX element matched query")
        err = self._ax["AXUIElementSetAttributeValue"](
            found.element,
            self._ax.get("kAXValueAttribute", "AXValue"),
            value,
        )
        if int(err) != int(self._ax["kAXErrorSuccess"]):
            raise DesktopNativeError(
                f"AX set value failed with {self._ax_error_label(int(err))}"
            )
        return {"matched": found.info, "depth": found.depth, "value": value}

    def _ax_request_access(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.health_check({"request_access": bool(params.get("prompt", True))})

    def _click_menu(self, params: dict[str, Any]) -> dict[str, Any]:
        """Click a menu item via AX menu bar traversal (e.g. File > Save)."""
        if not self._is_trusted():
            raise DesktopNativeError(
                "Accessibility not trusted for this process; use ax_request_access"
            )
        menu_name = str(params.get("menu") or "").strip()
        item_name = str(params.get("item") or "").strip()
        if not menu_name:
            raise DesktopNativeError("click_menu requires 'menu' (e.g. 'File')")
        if not item_name:
            raise DesktopNativeError("click_menu requires 'item' (e.g. 'Save')")

        menu_bar = self._ax_menu_bar()
        # Find the top-level menu (e.g. "File")
        top_menu = self._ax_find_first(
            menu_bar,
            match={"role": "AXMenuBarItem", "title": menu_name},
            depth=0,
            max_depth=1,
            max_children=30,
        )
        if top_menu is None:
            raise DesktopNativeError(f"menu '{menu_name}' not found in menu bar")

        # Press the top-level menu to open it
        err = self._ax["AXUIElementPerformAction"](
            top_menu.element, self._ax.get("kAXPressAction", "AXPress")
        )
        if int(err) != int(self._ax["kAXErrorSuccess"]):
            raise DesktopNativeError(
                f"failed to open menu '{menu_name}': {self._ax_error_label(int(err))}"
            )
        time.sleep(0.15)  # Allow menu to render

        # Find the menu item within the expanded menu
        menu_item = self._ax_find_first(
            top_menu.element,
            match={"role": "AXMenuItem", "title": item_name},
            depth=0,
            max_depth=3,
            max_children=50,
        )
        if menu_item is None:
            raise DesktopNativeError(
                f"item '{item_name}' not found in menu '{menu_name}'"
            )

        # Click the menu item
        err = self._ax["AXUIElementPerformAction"](
            menu_item.element, self._ax.get("kAXPressAction", "AXPress")
        )
        if int(err) != int(self._ax["kAXErrorSuccess"]):
            raise DesktopNativeError(
                f"failed to click '{menu_name} > {item_name}': "
                f"{self._ax_error_label(int(err))}"
            )
        return {"menu": menu_name, "item": item_name, "action": "AXPress"}

    def _ax_root(self, params: dict[str, Any]) -> Any:
        target = str(params.get("target") or "focused_window").strip().lower()
        if target == "focused_app":
            return self._ax_focused_application()
        if target == "focused_element":
            return self._ax_focused_element()
        if target == "system":
            return self._ax["AXUIElementCreateSystemWide"]()
        if target == "menu_bar":
            return self._ax_menu_bar()
        return self._ax_focused_window()

    def _ax_focused_application(self) -> Any:
        system = self._ax["AXUIElementCreateSystemWide"]()
        err, app = self._ax["AXUIElementCopyAttributeValue"](
            system,
            self._ax.get("kAXFocusedApplicationAttribute", "AXFocusedApplication"),
            None,
        )
        if int(err) != int(self._ax["kAXErrorSuccess"]) or app is None:
            raise DesktopNativeError(
                f"unable to read focused application ({self._ax_error_label(int(err))})"
            )
        return app

    def _ax_focused_window(self) -> Any:
        app = self._ax_focused_application()
        err, win = self._ax["AXUIElementCopyAttributeValue"](
            app, self._ax.get("kAXFocusedWindowAttribute", "AXFocusedWindow"), None
        )
        if int(err) != int(self._ax["kAXErrorSuccess"]) or win is None:
            raise DesktopNativeError(
                f"unable to read focused window ({self._ax_error_label(int(err))})"
            )
        return win

    def _ax_focused_element(self) -> Any:
        system = self._ax["AXUIElementCreateSystemWide"]()
        err, elem = self._ax["AXUIElementCopyAttributeValue"](
            system,
            self._ax.get("kAXFocusedUIElementAttribute", "AXFocusedUIElement"),
            None,
        )
        if int(err) != int(self._ax["kAXErrorSuccess"]) or elem is None:
            raise DesktopNativeError(
                f"unable to read focused element ({self._ax_error_label(int(err))})"
            )
        return elem

    def _ax_menu_bar(self) -> Any:
        """Return the AXMenuBar element of the focused application."""
        app = self._ax_focused_application()
        err, menu_bar = self._ax["AXUIElementCopyAttributeValue"](
            app,
            self._ax.get("kAXMenuBarAttribute", "AXMenuBar"),
            None,
        )
        if int(err) != int(self._ax["kAXErrorSuccess"]) or menu_bar is None:
            raise DesktopNativeError(
                f"unable to read menu bar ({self._ax_error_label(int(err))})"
            )
        return menu_bar

    def _ax_snapshot(
        self,
        element: Any,
        *,
        depth: int,
        max_depth: int,
        max_children: int,
    ) -> dict[str, Any]:
        info = self._ax_element_info(element)
        info["depth"] = depth
        if depth >= max_depth:
            return info
        children = self._ax_children(element)
        info["child_count"] = len(children)
        if children:
            info["children"] = [
                self._ax_snapshot(
                    child,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_children=max_children,
                )
                for child in children[:max_children]
            ]
        return info

    def _ax_find_first(
        self,
        element: Any,
        *,
        match: dict[str, Any],
        depth: int,
        max_depth: int,
        max_children: int,
    ) -> _AXMatchResult | None:
        info = self._ax_element_info(element)
        if self._ax_matches(info, match):
            return _AXMatchResult(element=element, info=info, depth=depth)
        if depth >= max_depth:
            return None
        for child in self._ax_children(element)[:max_children]:
            found = self._ax_find_first(
                child,
                match=match,
                depth=depth + 1,
                max_depth=max_depth,
                max_children=max_children,
            )
            if found is not None:
                return found
        return None

    def _ax_children(self, element: Any) -> list[Any]:
        err, children = self._ax["AXUIElementCopyAttributeValue"](
            element,
            self._ax.get("kAXChildrenAttribute", "AXChildren"),
            None,
        )
        if int(err) != int(self._ax["kAXErrorSuccess"]) or children is None:
            return []
        if isinstance(children, list | tuple):
            return list(children)
        return []

    def _ax_element_info(self, element: Any) -> dict[str, Any]:
        info: dict[str, Any] = {"element_ref": self._element_ref(element)}
        for key, attr_name in self._AX_ATTRIBUTE_KEYS:
            attr = self._ax.get(attr_name)
            if not attr:
                continue
            try:
                err, value = self._ax["AXUIElementCopyAttributeValue"](
                    element, attr, None
                )
                if int(err) == int(self._ax["kAXErrorSuccess"]) and value is not None:
                    info[key] = self._json_safe(value)
            except (OSError, RuntimeError, KeyError, TypeError):
                continue  # Expected: some AX attributes unavailable on certain elements
        # Also query available actions so the LLM knows what's possible
        try:
            action_names_attr = self._ax.get("kAXActionNamesAttribute")
            if action_names_attr:
                err, actions = self._ax["AXUIElementCopyAttributeValue"](
                    element, action_names_attr, None
                )
                if int(err) == int(self._ax["kAXErrorSuccess"]) and actions:
                    info["actions"] = [str(a) for a in actions]
        except (OSError, RuntimeError) as e:
            logger.debug("Failed to read AX action names: %s", e)
        return info

    @staticmethod
    def _ax_matches(info: dict[str, Any], match: dict[str, Any]) -> bool:
        role = str(match.get("role") or "").strip()
        if role and str(info.get("role") or "") != role:
            return False
        title = str(match.get("title") or "").strip()
        if title and str(info.get("title") or "") != title:
            return False
        title_contains = str(match.get("title_contains") or "").strip().lower()
        if (
            title_contains
            and title_contains not in str(info.get("title") or "").lower()
        ):
            return False
        desc_contains = str(match.get("description_contains") or "").strip().lower()
        if (
            desc_contains
            and desc_contains not in str(info.get("description") or "").lower()
        ):
            return False
        value_contains = str(match.get("value_contains") or "").strip().lower()
        if (
            value_contains
            and value_contains not in str(info.get("value") or "").lower()
        ):
            return False
        return True

    def _modifier_flags(self, modifiers: Any) -> int:
        if not modifiers:
            return 0
        if not isinstance(modifiers, list):
            raise DesktopNativeError("modifiers must be an array of strings")
        flags = 0
        for raw in modifiers:
            key = str(raw).strip().lower()
            flag_name = self._MODIFIER_FLAG_NAMES.get(key)
            if flag_name is None:
                raise DesktopNativeError(f"unsupported modifier: {raw}")
            flags |= int(getattr(self._quartz, flag_name))
        return flags

    def _key_code_for_key(self, key: str) -> int:
        if key in self._KEY_CODE_MAP:
            return self._KEY_CODE_MAP[key]
        if len(key) == 1 and key.lower() in self._CHAR_KEY_CODE_MAP:
            return self._CHAR_KEY_CODE_MAP[key.lower()]
        raise DesktopNativeError(f"unsupported key: {key}")

    def _mouse_location(self) -> tuple[float, float]:
        event = self._quartz.CGEventCreate(None)
        point = self._quartz.CGEventGetLocation(event)
        return (float(point.x), float(point.y))

    def _mouse_event_types(self, button: str) -> tuple[int, int, int]:
        button = button.strip().lower()
        if button == "right":
            return (
                int(self._quartz.kCGEventRightMouseDown),
                int(self._quartz.kCGEventRightMouseUp),
                int(self._quartz.kCGMouseButtonRight),
            )
        return (
            int(self._quartz.kCGEventLeftMouseDown),
            int(self._quartz.kCGEventLeftMouseUp),
            int(self._quartz.kCGMouseButtonLeft),
        )

    def _ax_error_label(self, code: int) -> str:
        return self._ax_error_names.get(code, f"AXError({code})")

    @staticmethod
    def _element_ref(element: Any) -> str:
        text = repr(element)
        return text[:128]

    # ── Clipboard ──────────────────────────────────────────────────────

    def _clipboard_read(self, params: dict[str, Any]) -> dict[str, Any]:
        pb = self._appkit.NSPasteboard.generalPasteboard()
        text = pb.stringForType_(self._appkit.NSPasteboardTypeString)
        return {"text": str(text or "")}

    def _clipboard_write(self, params: dict[str, Any]) -> dict[str, Any]:
        text = str(params.get("text", ""))
        if not text:
            raise DesktopNativeError("clipboard_write requires text")
        pb = self._appkit.NSPasteboard.generalPasteboard()
        pb.clearContents()
        pb.setString_forType_(text, self._appkit.NSPasteboardTypeString)
        return {"written_chars": len(text)}

    # ── Window management ────────────────────────────────────────────

    def _list_windows(self, params: dict[str, Any]) -> dict[str, Any]:
        all_apps = bool(params.get("all_apps", True))
        windows_attr = self._ax.get("kAXWindowsAttribute")
        if not windows_attr:
            raise DesktopNativeError("AXWindows attribute not available")

        if all_apps:
            # List windows from ALL running apps
            workspace = self._appkit.NSWorkspace.sharedWorkspace()
            running_apps = workspace.runningApplications()
            result = []
            for app in running_apps:
                if app.activationPolicy() != 0:  # 0 = regular app (has UI)
                    continue
                pid = app.processIdentifier()
                app_name = str(app.localizedName() or "")
                try:
                    app_el = self._ax["AXUIElementCreateApplication"](pid)
                    err, windows = self._ax["AXUIElementCopyAttributeValue"](
                        app_el, windows_attr, None
                    )
                    if int(err) != int(self._ax["kAXErrorSuccess"]) or not windows:
                        continue
                    for win in windows:
                        info = self._ax_element_info(win)
                        result.append({
                            "title": info.get("title", ""),
                            "position": info.get("position"),
                            "size": info.get("size"),
                            "role": info.get("role", ""),
                            "app": app_name,
                            "pid": pid,
                        })
                except (OSError, RuntimeError) as e:
                    logger.debug("Failed to list windows for app %s: %s", app_name, e)
                    continue
            return {"windows": result, "count": len(result)}
        else:
            # Focused app only
            app_el = self._ax_focused_application()
            err, windows = self._ax["AXUIElementCopyAttributeValue"](
                app_el, windows_attr, None
            )
            if int(err) != int(self._ax["kAXErrorSuccess"]) or not windows:
                return {"windows": [], "count": 0}
            result = []
            for win in windows:
                info = self._ax_element_info(win)
                result.append({
                    "title": info.get("title", ""),
                    "position": info.get("position"),
                    "size": info.get("size"),
                    "role": info.get("role", ""),
                })
            return {"windows": result, "count": len(result)}

    def _move_window(self, params: dict[str, Any]) -> dict[str, Any]:
        x = params.get("x")
        y = params.get("y")
        if x is None or y is None:
            raise DesktopNativeError("move_window requires x and y")
        win = self._find_window(params.get("window_title"))
        pos_attr = self._ax.get("kAXPositionAttribute")
        if not pos_attr:
            raise DesktopNativeError("AXPosition attribute not available")
        point = self._quartz.CGPointMake(float(x), float(y))
        value = self._ax["AXValueCreate"](self._ax["kAXValueTypeCGPoint"], point)
        err = self._ax["AXUIElementSetAttributeValue"](win, pos_attr, value)
        if int(err) != int(self._ax["kAXErrorSuccess"]):
            raise DesktopNativeError(f"move_window failed: {self._ax_error_label(int(err))}")
        return {"x": int(x), "y": int(y)}

    def _resize_window(self, params: dict[str, Any]) -> dict[str, Any]:
        w = params.get("width")
        h = params.get("height")
        if w is None or h is None:
            raise DesktopNativeError("resize_window requires width and height")
        win = self._find_window(params.get("window_title"))
        size_attr = self._ax.get("kAXSizeAttribute")
        if not size_attr:
            raise DesktopNativeError("AXSize attribute not available")
        size = self._quartz.CGSizeMake(float(w), float(h))
        value = self._ax["AXValueCreate"](self._ax["kAXValueTypeCGSize"], size)
        err = self._ax["AXUIElementSetAttributeValue"](win, size_attr, value)
        if int(err) != int(self._ax["kAXErrorSuccess"]):
            raise DesktopNativeError(f"resize_window failed: {self._ax_error_label(int(err))}")
        return {"width": int(w), "height": int(h)}

    def _minimize_window(self, params: dict[str, Any]) -> dict[str, Any]:
        minimize_others = bool(params.get("others", False))
        minimized_attr = self._ax.get("kAXMinimizedAttribute") or "AXMinimized"

        if minimize_others:
            # Minimize all windows from OTHER apps, keep frontmost app's windows
            workspace = self._appkit.NSWorkspace.sharedWorkspace()
            front_app = workspace.frontmostApplication()
            front_pid = front_app.processIdentifier() if front_app else -1
            windows_attr = self._ax.get("kAXWindowsAttribute")
            if not windows_attr:
                raise DesktopNativeError("AXWindows attribute not available")

            minimized_count = 0
            for app in workspace.runningApplications():
                if app.activationPolicy() != 0:  # regular app only
                    continue
                pid = app.processIdentifier()
                if pid == front_pid:
                    continue
                try:
                    app_el = self._ax["AXUIElementCreateApplication"](pid)
                    err, windows = self._ax["AXUIElementCopyAttributeValue"](
                        app_el, windows_attr, None
                    )
                    if int(err) != int(self._ax["kAXErrorSuccess"]) or not windows:
                        continue
                    for win in windows:
                        try:
                            self._ax["AXUIElementSetAttributeValue"](win, minimized_attr, True)
                            minimized_count += 1
                        except (OSError, RuntimeError) as e:
                            logger.debug("Failed to minimize window in app (pid %d): %s", pid, e)
                except (OSError, RuntimeError, KeyError) as e:
                    logger.debug("Failed to enumerate windows for app (pid %d): %s", pid, e)
                    continue
            return {"minimized": True, "count": minimized_count}
        else:
            win = self._find_window(params.get("window_title"))
            err = self._ax["AXUIElementSetAttributeValue"](win, minimized_attr, True)
            if int(err) != int(self._ax["kAXErrorSuccess"]):
                raise DesktopNativeError(f"minimize_window failed: {self._ax_error_label(int(err))}")
            return {"minimized": True}

    def _find_window(self, title: str | None = None) -> Any:
        """Find window by title (across all apps) or return the focused window."""
        if not title:
            return self._ax_focused_window()
        windows_attr = self._ax.get("kAXWindowsAttribute")
        if not windows_attr:
            raise DesktopNativeError("AXWindows attribute not available")

        # Search across all running apps
        workspace = self._appkit.NSWorkspace.sharedWorkspace()
        title_lower = str(title).lower()
        for app in workspace.runningApplications():
            if app.activationPolicy() != 0:
                continue
            try:
                app_el = self._ax["AXUIElementCreateApplication"](app.processIdentifier())
                err, windows = self._ax["AXUIElementCopyAttributeValue"](
                    app_el, windows_attr, None
                )
                if int(err) != int(self._ax["kAXErrorSuccess"]) or not windows:
                    continue
                for win in windows:
                    info = self._ax_element_info(win)
                    win_title = str(info.get("title", "")).lower()
                    if title_lower in win_title:
                        return win
            except (OSError, RuntimeError) as e:
                logger.debug("Error searching windows in app: %s", e)
                continue
        raise DesktopNativeError(f"no window matching '{title}'")

    # ── Multi-monitor ────────────────────────────────────────────────

    def _list_monitors(self, params: dict[str, Any]) -> dict[str, Any]:
        screens = self._appkit.NSScreen.screens()
        monitors = []
        for i, screen in enumerate(screens or []):
            frame = screen.frame()
            monitors.append({
                "index": i,
                "x": int(frame.origin.x),
                "y": int(frame.origin.y),
                "width": int(frame.size.width),
                "height": int(frame.size.height),
                "is_main": (i == 0),
            })
        return {"monitors": monitors, "count": len(monitors)}

    # ── Mouse drag ───────────────────────────────────────────────────

    def _mouse_drag(self, params: dict[str, Any]) -> dict[str, Any]:
        start_x = float(params.get("start_x", params.get("x", 0)))
        start_y = float(params.get("start_y", params.get("y", 0)))
        end_x = float(params.get("end_x", 0))
        end_y = float(params.get("end_y", 0))
        duration = min(float(params.get("duration_seconds", 0.5)), 5.0)
        quartz = self._quartz

        # Mouse down at start
        down_event = quartz.CGEventCreateMouseEvent(
            None, quartz.kCGEventLeftMouseDown, (start_x, start_y),
            quartz.kCGMouseButtonLeft,
        )
        quartz.CGEventPost(quartz.kCGHIDEventTap, down_event)

        # Smooth drag to end
        steps = max(int(duration * 60), 5)
        for i in range(1, steps + 1):
            t = i / steps
            cx = start_x + (end_x - start_x) * t
            cy = start_y + (end_y - start_y) * t
            drag_event = quartz.CGEventCreateMouseEvent(
                None, quartz.kCGEventLeftMouseDragged, (cx, cy),
                quartz.kCGMouseButtonLeft,
            )
            quartz.CGEventPost(quartz.kCGHIDEventTap, drag_event)
            time.sleep(duration / steps)

        # Mouse up at end
        up_event = quartz.CGEventCreateMouseEvent(
            None, quartz.kCGEventLeftMouseUp, (end_x, end_y),
            quartz.kCGMouseButtonLeft,
        )
        quartz.CGEventPost(quartz.kCGHIDEventTap, up_event)

        return {
            "start": {"x": int(start_x), "y": int(start_y)},
            "end": {"x": int(end_x), "y": int(end_y)},
        }

    # ── Volume / Brightness ─────────────────────────────────────────

    def _set_volume(self, params: dict[str, Any]) -> dict[str, Any]:
        """Set system output volume (0-100) via AppleScript."""
        level = int(params.get("level", 50))
        level = max(0, min(100, level))
        result = subprocess.run(
            ["osascript", "-e", f"set volume output volume {level}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            raise DesktopNativeError(
                f"failed to set volume: {result.stderr.strip() or 'osascript exited non-zero'}"
            )
        return {"volume": level}

    def _get_volume(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get current system output volume (0-100) via AppleScript."""
        result = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            capture_output=True, text=True, timeout=5,
        )
        try:
            volume = int(result.stdout.strip())
        except (ValueError, TypeError):
            raise DesktopNativeError(
                f"failed to read volume: {result.stderr.strip() or result.stdout.strip()}"
            )
        return {"volume": volume}

    def _set_brightness(self, params: dict[str, Any]) -> dict[str, Any]:
        """Set display brightness (0.0-1.0) via CoreDisplay private framework.

        NOTE: This requires the CoreDisplay framework. Raises DesktopNativeError
        if it's unavailable — there is no fallback.
        """
        level = float(params.get("level", 0.5))
        level = max(0.0, min(1.0, level))
        try:
            import ctypes
            import ctypes.util
            core_display = ctypes.CDLL("/System/Library/Frameworks/CoreDisplay.framework/CoreDisplay")
            core_display.CoreDisplay_Display_SetUserBrightness.argtypes = [ctypes.c_uint32, ctypes.c_double]
            core_display.CoreDisplay_Display_SetUserBrightness(0, level)
        except (OSError, AttributeError):
            raise DesktopNativeError(
                "brightness control requires CoreDisplay framework (IOKit); "
                "use System Preferences or keyboard brightness keys instead"
            )
        return {"brightness": round(level, 2)}

    def _get_brightness(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get current display brightness (0.0-1.0) via CoreDisplay."""
        try:
            import ctypes
            core_display = ctypes.CDLL("/System/Library/Frameworks/CoreDisplay.framework/CoreDisplay")
            core_display.CoreDisplay_Display_GetUserBrightness.restype = ctypes.c_double
            core_display.CoreDisplay_Display_GetUserBrightness.argtypes = [ctypes.c_uint32]
            brightness = core_display.CoreDisplay_Display_GetUserBrightness(0)
            return {"brightness": round(float(brightness), 2)}
        except (OSError, AttributeError):
            raise DesktopNativeError(
                "brightness read requires CoreDisplay framework (IOKit)"
            )

    # ── Move window to monitor ──────────────────────────────────────

    def _move_to_monitor(self, params: dict[str, Any]) -> dict[str, Any]:
        """Move the focused (or title-matched) window to a specific monitor.

        Args (in params):
            monitor: 0-based monitor index
            window_title: optional window title to match (defaults to focused)
        """
        monitor_idx = int(params.get("monitor", 0))
        monitors_info = self._list_monitors({})
        monitors = monitors_info.get("monitors", [])
        if not monitors:
            raise DesktopNativeError("no monitors detected")
        if monitor_idx < 0 or monitor_idx >= len(monitors):
            raise DesktopNativeError(
                f"monitor index {monitor_idx} out of range (0-{len(monitors) - 1})"
            )
        target = monitors[monitor_idx]
        # Center the window on the target monitor
        center_x = target["x"] + target["width"] // 4
        center_y = target["y"] + target["height"] // 4
        self._move_window({
            "x": center_x,
            "y": center_y,
            "window_title": params.get("window_title"),
        })
        return {
            "monitor": monitor_idx,
            "position": {"x": center_x, "y": center_y},
            "monitor_info": target,
        }

    # ── System Info & Control ───────────────────────────────────────

    def _list_apps(self, params: dict[str, Any]) -> dict[str, Any]:
        """List all running GUI applications via NSWorkspace."""
        ws = self._appkit.NSWorkspace.sharedWorkspace()
        apps = []
        for app in ws.runningApplications():
            if app.activationPolicy() != 0:  # 0 = regular GUI app
                continue
            apps.append({
                "name": str(app.localizedName() or ""),
                "bundle_id": str(app.bundleIdentifier() or ""),
                "pid": int(app.processIdentifier()),
                "active": bool(app.isActive()),
                "hidden": bool(app.isHidden()),
            })
        return {"apps": apps, "count": len(apps)}

    def _force_quit_app(self, params: dict[str, Any]) -> dict[str, Any]:
        """Force quit an application by name or bundle ID."""
        app_name = str(params.get("app_name", "")).strip()
        bundle_id = str(params.get("bundle_id", "")).strip()
        if not app_name and not bundle_id:
            raise DesktopNativeError("force_quit_app requires app_name or bundle_id")
        ws = self._appkit.NSWorkspace.sharedWorkspace()
        for app in ws.runningApplications():
            name = str(app.localizedName() or "")
            bid = str(app.bundleIdentifier() or "")
            if (app_name and name.lower() == app_name.lower()) or (bundle_id and bid == bundle_id):
                terminated = app.forceTerminate()
                return {"terminated": terminated, "app_name": name, "pid": int(app.processIdentifier())}
        raise DesktopNativeError(f"App not found: {app_name or bundle_id}")

    def _get_system_info(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get battery, display, and basic system info."""
        import subprocess as _sp
        info: dict[str, Any] = {}
        # Battery
        try:
            out = _sp.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=3)
            lines = out.stdout.strip().split("\n")
            for line in lines:
                if "%" in line:
                    import re
                    m = re.search(r"(\d+)%", line)
                    if m:
                        info["battery_percent"] = int(m.group(1))
                    info["battery_charging"] = "charging" in line.lower()
                    info["battery_on_ac"] = "ac power" in lines[0].lower() if lines else False
        except (OSError, _sp.SubprocessError):
            pass
        # WiFi
        try:
            out = _sp.run(
                ["/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport", "-I"],
                capture_output=True, text=True, timeout=3
            )
            for line in out.stdout.split("\n"):
                line = line.strip()
                if line.startswith("SSID:"):
                    info["wifi_ssid"] = line.split(":", 1)[1].strip()
                elif line.startswith("RSSI:"):
                    info["wifi_rssi"] = int(line.split(":", 1)[1].strip())
        except (OSError, _sp.SubprocessError, ValueError):
            pass
        # Screen count
        try:
            screens = self._appkit.NSScreen.screens()
            info["screen_count"] = len(screens or [])
            if screens:
                main = screens[0].frame()
                info["main_screen"] = {"width": int(main.size.width), "height": int(main.size.height)}
        except (AttributeError, TypeError):
            pass
        return info

    def _toggle_dark_mode(self, params: dict[str, Any]) -> dict[str, Any]:
        """Toggle macOS dark mode via AppleScript (visually applies immediately)."""
        import subprocess as _sp
        current = self._get_dark_mode({})
        is_dark = current.get("dark_mode", False)
        # AppleScript toggle — this actually changes the UI immediately
        _sp.run(
            ["osascript", "-e",
             'tell application "System Events" to tell appearance preferences to set dark mode to '
             + ("false" if is_dark else "true")],
            capture_output=True, timeout=5
        )
        return {"dark_mode": not is_dark, "toggled": True}

    def _get_dark_mode(self, params: dict[str, Any]) -> dict[str, Any]:
        """Check if dark mode is currently active."""
        import subprocess as _sp
        try:
            out = _sp.run(["defaults", "read", "-g", "AppleInterfaceStyle"],
                          capture_output=True, text=True, timeout=3)
            is_dark = "dark" in out.stdout.strip().lower()
            return {"dark_mode": is_dark}
        except (OSError, _sp.SubprocessError):
            return {"dark_mode": False}

    def _toggle_dnd(self, params: dict[str, Any]) -> dict[str, Any]:
        """Toggle Do Not Disturb / Focus mode."""
        import subprocess as _sp
        # macOS Monterey+: use Focus via shortcuts
        try:
            # Check current state
            out = _sp.run(["defaults", "read", "com.apple.controlcenter", "NSStatusItem Visible FocusModes"],
                          capture_output=True, text=True, timeout=3)
            # Toggle via keyboard shortcut (most reliable cross-version method)
            self._press_key({"key": "moon", "key_code": None, "modifiers": []})
            return {"toggled": True, "note": "Focus mode toggled"}
        except (OSError, _sp.SubprocessError, DesktopNativeError):
            return {"toggled": False, "error": "Could not toggle DND"}

    def _spotlight_search(self, params: dict[str, Any]) -> dict[str, Any]:
        """Search files via mdfind (Spotlight CLI) — instant, no AppleScript."""
        import subprocess as _sp
        query = str(params.get("query", "")).strip()
        if not query:
            raise DesktopNativeError("spotlight_search requires query")
        limit = min(int(params.get("limit", 20)), 100)
        try:
            out = _sp.run(["mdfind", "-limit", str(limit), query],
                          capture_output=True, text=True, timeout=10)
            files = [f.strip() for f in out.stdout.strip().split("\n") if f.strip()]
            return {"query": query, "files": files, "count": len(files)}
        except (OSError, _sp.SubprocessError) as exc:
            raise DesktopNativeError(f"Spotlight search failed: {exc}")

    def _open_file(self, params: dict[str, Any]) -> dict[str, Any]:
        """Open a file with the default app, or a specific app."""
        file_path = str(params.get("path", params.get("file_path", ""))).strip()
        if not file_path:
            raise DesktopNativeError("open_file requires path")
        app_name = str(params.get("app_name", "")).strip()
        ws = self._appkit.NSWorkspace.sharedWorkspace()
        url = self._foundation.NSURL.fileURLWithPath_(file_path)
        if app_name:
            # Open with specific app
            ok = ws.openFile_withApplication_(file_path, app_name)
        else:
            ok = ws.openURL_(url)
        return {"opened": bool(ok), "path": file_path, "app": app_name or "(default)"}

    def _get_finder_selection(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get currently selected files in Finder via AppleScript (fast, single call)."""
        import subprocess as _sp
        try:
            out = _sp.run(
                ["osascript", "-e", 'tell application "Finder" to get POSIX path of (selection as alias list)'],
                capture_output=True, text=True, timeout=5
            )
            if out.returncode != 0:
                # Try single selection
                out = _sp.run(
                    ["osascript", "-e", 'tell application "Finder" to get POSIX path of (selection as alias)'],
                    capture_output=True, text=True, timeout=5
                )
            paths = [p.strip() for p in out.stdout.strip().split(", ") if p.strip()]
            return {"files": paths, "count": len(paths)}
        except (OSError, _sp.SubprocessError):
            return {"files": [], "count": 0}

    def _screen_record_start(self, params: dict[str, Any]) -> dict[str, Any]:
        """Start screen recording via screencapture."""
        import subprocess as _sp
        path = str(params.get("path", "")).strip()
        if not path:
            import tempfile
            fd, path = tempfile.mkstemp(prefix="jarvis_screenrec_", suffix=".mov")
            import os; os.close(fd)
        try:
            proc = _sp.Popen(
                ["screencapture", "-v", "-k", path],
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
            )
            self._screen_record_proc = proc
            self._screen_record_path = path
            return {"recording": True, "path": path, "pid": proc.pid}
        except (OSError, _sp.SubprocessError) as exc:
            return {"recording": False, "error": str(exc)}

    def _screen_record_stop(self, params: dict[str, Any]) -> dict[str, Any]:
        """Stop screen recording."""
        import os, signal
        proc = getattr(self, "_screen_record_proc", None)
        path = getattr(self, "_screen_record_path", "")
        if proc is None:
            return {"stopped": False, "error": "No recording in progress"}
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=5)
        except (OSError, _sp.SubprocessError):
            proc.kill()
        self._screen_record_proc = None
        size = os.path.getsize(path) if os.path.exists(path) else 0
        return {"stopped": True, "path": path, "size_bytes": size}

    def _tile_windows(self, params: dict[str, Any]) -> dict[str, Any]:
        """Tile two windows side by side (split view)."""
        left_app = str(params.get("left_app", "")).strip()
        right_app = str(params.get("right_app", "")).strip()
        if not left_app or not right_app:
            raise DesktopNativeError("tile_windows requires left_app and right_app")
        screens = self._appkit.NSScreen.screens()
        if not screens:
            raise DesktopNativeError("No screens detected")
        frame = screens[0].visibleFrame()
        half_w = int(frame.size.width / 2)
        h = int(frame.size.height)
        x = int(frame.origin.x)
        y = int(frame.origin.y)
        # Position left app
        self._focus_app({"app_name": left_app})
        self._move_window({"x": x, "y": y})
        self._resize_window({"width": half_w, "height": h})
        # Position right app
        self._focus_app({"app_name": right_app})
        self._move_window({"x": x + half_w, "y": y})
        self._resize_window({"width": half_w, "height": h})
        return {"tiled": True, "left": left_app, "right": right_app}

    def _app_switch(self, params: dict[str, Any]) -> dict[str, Any]:
        """Switch to an app — faster than focus_app (uses NSRunningApplication.activate)."""
        app_name = str(params.get("app_name", "")).strip()
        if not app_name:
            raise DesktopNativeError("app_switch requires app_name")
        return self._focus_app({"app_name": app_name, "launch_if_missing": True})

    def _get_running_processes(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get running processes with CPU/memory usage via ps."""
        import subprocess as _sp
        limit = min(int(params.get("limit", 20)), 100)
        sort_by = str(params.get("sort_by", "cpu")).strip().lower()
        sort_flag = "-m" if sort_by == "memory" else "-r"
        try:
            out = _sp.run(
                ["ps", sort_flag, "-eo", "pid,pcpu,pmem,comm"],
                capture_output=True, text=True, timeout=5
            )
            lines = out.stdout.strip().split("\n")[1:]  # skip header
            processes = []
            for line in lines[:limit]:
                parts = line.split(None, 3)
                if len(parts) >= 4:
                    processes.append({
                        "pid": int(parts[0]),
                        "cpu_percent": float(parts[1]),
                        "memory_percent": float(parts[2]),
                        "command": parts[3].split("/")[-1],  # basename
                    })
            return {"processes": processes, "count": len(processes)}
        except (OSError, _sp.SubprocessError, ValueError) as exc:
            return {"processes": [], "error": str(exc)}

    # ── Helpers ──────────────────────────────────────────────────────

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, list | tuple):
            return [cls._json_safe(v) for v in value[:50]]
        # Handle AXValue objects (wrap CGPoint/CGSize) — need AXValueGetValue
        val_str = str(value)
        if "kAXValueCGPointType" in val_str:
            try:
                from ApplicationServices import AXValueGetValue, kAXValueTypeCGPoint  # type: ignore
                ok, point = AXValueGetValue(value, kAXValueTypeCGPoint, None)
                if ok and hasattr(point, "x"):
                    return {"x": float(point.x), "y": float(point.y)}
            except (ImportError, OSError, TypeError, ValueError):
                pass  # Expected: value may not be CGPoint type
        if "kAXValueCGSizeType" in val_str:
            try:
                from ApplicationServices import AXValueGetValue, kAXValueTypeCGSize  # type: ignore
                ok, size = AXValueGetValue(value, kAXValueTypeCGSize, None)
                if ok and hasattr(size, "width"):
                    return {"width": float(size.width), "height": float(size.height)}
            except (ImportError, OSError, TypeError, ValueError):
                pass  # Expected: value may not be CGSize type
        if hasattr(value, "x") and hasattr(value, "y"):
            try:
                return {"x": float(value.x), "y": float(value.y)}
            except (TypeError, ValueError, AttributeError):
                return str(value)
        if hasattr(value, "width") and hasattr(value, "height"):
            try:
                return {"width": float(value.width), "height": float(value.height)}
            except (TypeError, ValueError, AttributeError):
                return str(value)
        text = val_str
        return text[:400] if len(text) > 400 else text
