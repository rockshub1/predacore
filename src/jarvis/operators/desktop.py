"""
High-power macOS desktop operator for JARVIS.

Provides action-level automation primitives:
- App launch/focus and URL open
- Keyboard input (type text, key presses with modifiers)
- Mouse operations (move/click/scroll) when pyautogui is available
- Screenshots via macOS screencapture (multi-quality: full/fast/thumbnail)
- Macro execution via BaseOperator.execute_macro with abort support
- **Smart input** (smart_type, smart_run_command, smart_create_note)
  via the SmartInputEngine — bulletproof input with auto-fallback
- **Async bridge** — dedicated worker thread for safe sync-to-async calls
- **Retry decorator** — automatic retry on transient macOS failures

Phase 6.3: Uses DesktopAction enums — zero magic strings in dispatch.
Phase 6.4: Retry decorator on flaky AppleScript/screenshot methods.
"""
from __future__ import annotations

import asyncio
import atexit
import base64
import concurrent.futures
import logging
import os
import platform
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

try:
    from jarvis.operators.native_service import (
        DesktopNativeError,
        MacDesktopNativeService,
    )
except ImportError:  # pragma: no cover - fallback when imported as src.jarvis.*
    from src.jarvis.operators.native_service import (  # type: ignore
        DesktopNativeError,
        MacDesktopNativeService,
    )

try:
    from jarvis.operators.smart_input import SmartInputEngine
except ImportError:  # pragma: no cover
    try:
        from src.jarvis.operators.smart_input import SmartInputEngine  # type: ignore
    except ImportError:
        SmartInputEngine = None  # type: ignore[assignment, misc]

from .base import BaseOperator, OperatorPlatform, OperatorError, MacroAbortToken
from .enums import (
    DesktopAction,
    ScreenshotQuality,
    SMART_ACTIONS,
    NATIVE_ONLY_ACTIONS,
    NATIVE_CAPABLE_ACTIONS,
)
from .retry import with_retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async Bridge — dedicated thread+loop to prevent deadlocks
# ---------------------------------------------------------------------------


class AsyncBridge:
    """Dedicated thread + event loop for running async from sync context.

    Avoids the deadlock-prone pattern of run_coroutine_threadsafe on the
    caller's own loop. The bridge has its own loop on its own daemon thread.
    """

    _instance: "AsyncBridge | None" = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="jarvis-async-bridge"
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro, timeout: float = 30):
        """Submit a coroutine and block until it completes (or timeout)."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def stop(self):
        """Stop the event loop and join the worker thread."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    @classmethod
    def get(cls) -> "AsyncBridge":
        """Get or create the singleton async bridge."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    atexit.register(cls._instance.stop)
        return cls._instance


class DesktopControlError(RuntimeError):
    """Raised for desktop control execution failures."""


class MacDesktopOperator(BaseOperator):
    """macOS desktop controller with safe, structured actions.

    Uses DesktopAction enums for all action dispatch — zero magic strings.
    Flaky methods (AppleScript, screenshots) auto-retry on transient failures.
    """

    _ACTION_ALIASES = {
        "key_press": DesktopAction.PRESS_KEY,
        "keypress": DesktopAction.PRESS_KEY,
    }

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
        # Modifier keys (standalone press)
        "shift": 56,
        "control": 59,
        "ctrl": 59,
        "option": 58,
        "alt": 58,
        "command": 55,
        "cmd": 55,
        "fn": 63,
        # Function keys
        "f1": 122, "f2": 120, "f3": 99, "f4": 118,
        "f5": 96, "f6": 97, "f7": 98, "f8": 100,
        "f9": 101, "f10": 109, "f11": 103, "f12": 111,
        # Additional common keys
        "home": 115,
        "end": 119,
        "pageup": 116,
        "pagedown": 121,
        "capslock": 57,
        "forward_delete": 117,
    }

    _MODIFIER_MAP = {
        "command": "command down",
        "cmd": "command down",
        "shift": "shift down",
        "option": "option down",
        "alt": "option down",
        "control": "control down",
        "ctrl": "control down",
        "fn": "fn down",
    }

    def __init__(self, log: logging.Logger | None = None, operators_config: Any = None) -> None:
        super().__init__(log=log, operators_config=operators_config)
        self._is_macos = platform.system() == "Darwin"
        self._pyautogui = None
        self._native_service: MacDesktopNativeService | None = None
        self._native_fallback_enabled = True
        self._ops_cfg = operators_config
        self._smart_input: SmartInputEngine | None = None

        allow_raw = str(os.getenv("JARVIS_DESKTOP_ALLOWED_APPS", "")).strip()
        self._allowed_apps = {
            part.strip().lower() for part in allow_raw.split(",") if part.strip()
        }
        # Default: native ONLY. Legacy AppleScript is 100x slower (spawns
        # osascript per call). Set JARVIS_DESKTOP_BACKEND=native,legacy to
        # re-enable the slow fallback if native PyObjC is unavailable.
        backend_pref = str(os.getenv("JARVIS_DESKTOP_BACKEND", "native")).strip()
        prefs = {p.strip().lower() for p in backend_pref.split(",") if p.strip()}
        self._native_fallback_enabled = "legacy" in prefs
        if "native" in prefs and self._is_macos:
            try:
                service = MacDesktopNativeService(log=self._log)
                if service.available:
                    self._native_service = service
                else:
                    self._log.warning(
                        "Native desktop backend unavailable: %s",
                        service.init_error,
                    )
            except (OSError, ImportError, RuntimeError) as exc:
                self._log.warning(
                    "Failed to initialize native desktop backend: %s", exc
                )

        # Initialize Smart Input Engine
        if SmartInputEngine is not None and self._is_macos:
            try:
                self._smart_input = SmartInputEngine(
                    desktop_operator=self,
                    max_retries=3,
                    verify_timeout=1.0,
                    focus_settle_ms=300,
                    ax_ready_timeout=3.0,
                )
                self._log.info("Smart Input Engine initialized")
            except (OSError, RuntimeError, TypeError) as exc:
                self._log.warning("Failed to initialize Smart Input Engine: %s", exc)

    @property
    def available(self) -> bool:
        """Return True if desktop automation is available on this platform."""
        return self._is_macos

    @property
    def platform(self) -> OperatorPlatform:
        """Return the platform identifier."""
        return OperatorPlatform.MACOS

    @property
    def supported_actions(self) -> set[str]:
        """Return set of supported macOS desktop actions."""
        return {a.value for a in DesktopAction}

    @property
    def smart_input(self) -> SmartInputEngine | None:
        """Access the Smart Input Engine directly (for async callers)."""
        return self._smart_input

    # ── Main Dispatch ────────────────────────────────────────────

    def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute one desktop action and return structured result."""
        start = time.time()
        action = str(action or "").strip().lower()
        # Resolve aliases
        alias = self._ACTION_ALIASES.get(action)
        if alias:
            action = alias if isinstance(alias, str) else alias.value
        if not action:
            raise DesktopControlError("action is required")
        if not self._is_macos:
            raise DesktopControlError("desktop_control is only available on macOS")

        ok = True
        try:
            result = self._dispatch_action(action, params)
            elapsed_ms = int((time.time() - start) * 1000)
            result.update({"ok": True, "action": action, "elapsed_ms": elapsed_ms})
            return result
        except Exception:  # catch-all: top-level boundary
            ok = False
            raise
        finally:
            elapsed_ms = int((time.time() - start) * 1000)
            self._record_telemetry(action, elapsed_ms, ok)

    def _dispatch_action(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Core dispatch — maps action string to implementation."""

        # Handle smart actions (via async bridge)
        if action in SMART_ACTIONS:
            return self._execute_smart_action(action, params)

        # Handle macros via base class (with abort support)
        if action == DesktopAction.RUN_MACRO:
            steps = params.get("steps")
            abort_token = params.pop("_abort_token", None)
            return self.execute_macro(
                steps,
                stop_on_error=bool(params.get("stop_on_error", True)),
                delay_ms=max(0, min(int(params.get("delay_ms", 0) or 0), 10000)),
                abort_token=abort_token,
            )

        # Try native backend first for capable actions
        native_error: str | None = None
        if action in NATIVE_CAPABLE_ACTIONS and self._native_service is not None:
            try:
                return self._native_service.execute(action=action, params=params)
            except DesktopNativeError as exc:
                native_error = str(exc)
                if action in NATIVE_ONLY_ACTIONS:
                    raise DesktopControlError(native_error) from exc
                if not self._native_fallback_enabled:
                    raise DesktopControlError(native_error) from exc
            except (OSError, RuntimeError) as exc:
                native_error = str(exc)
                if action in NATIVE_ONLY_ACTIONS or not self._native_fallback_enabled:
                    raise DesktopControlError(native_error) from exc

        # Legacy fallback dispatch (cached on instance after first call)
        try:
            dispatch = self._legacy_dispatch
        except AttributeError:
            self._legacy_dispatch = {
                DesktopAction.OPEN_APP: self._open_app,
                DesktopAction.FOCUS_APP: self._focus_app,
                DesktopAction.OPEN_URL: self._open_url,
                DesktopAction.TYPE_TEXT: self._type_text,
                DesktopAction.PRESS_KEY: self._press_key,
                DesktopAction.MOUSE_MOVE: self._mouse_move,
                DesktopAction.MOUSE_SCROLL: self._mouse_scroll,
                DesktopAction.GET_MOUSE_POSITION: lambda _: self._get_mouse_position(),
                DesktopAction.SCREENSHOT: self._screenshot,
                DesktopAction.FRONTMOST_APP: lambda _: self._frontmost_app(),
                DesktopAction.HEALTH_CHECK: lambda _: self.health_check(),
                DesktopAction.SLEEP: self._sleep,
            }
            dispatch = self._legacy_dispatch

        handler = dispatch.get(action)
        if handler is not None:
            result = handler(params)
            if native_error:
                result["native_fallback_error"] = native_error
            return result

        # Mouse click variants (share implementation)
        if action == DesktopAction.MOUSE_CLICK:
            result = self._mouse_click(params, clicks=1)
        elif action == DesktopAction.MOUSE_DOUBLE_CLICK:
            result = self._mouse_click(params, clicks=2)
        elif action == DesktopAction.MOUSE_RIGHT_CLICK:
            result = self._mouse_click(params, clicks=1, button="right")
        elif action in NATIVE_ONLY_ACTIONS:
            raise DesktopControlError(
                "native desktop backend required for AX actions. "
                "Set JARVIS_DESKTOP_BACKEND=native,legacy and grant Accessibility access."
            )
        else:
            raise DesktopControlError(f"unsupported action: {action}")

        if native_error:
            result["native_fallback_error"] = native_error
        return result

    # ── Smart Input Actions (via AsyncBridge) ────────────────────

    def _execute_smart_action(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Bridge sync execute() to async SmartInputEngine methods."""
        if self._smart_input is None:
            raise DesktopControlError(
                "Smart Input Engine not available. "
                "Ensure SmartInputEngine is importable and macOS is the platform."
            )

        bridge = AsyncBridge.get()
        try:
            return bridge.run(
                self._dispatch_smart_async(action, params),
                timeout=30,
            )
        except concurrent.futures.TimeoutError as exc:
            raise DesktopControlError(
                f"Smart action '{action}' timed out after 30s"
            ) from exc

    async def _dispatch_smart_async(
        self, action: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Dispatch to the correct SmartInputEngine method."""
        assert self._smart_input is not None

        if action == DesktopAction.SMART_TYPE:
            text = str(params.get("text") or "")
            if not text:
                raise DesktopControlError("smart_type requires text")
            result = await self._smart_input.smart_type(
                text=text,
                app_name=str(params.get("app_name") or ""),
                press_enter=bool(params.get("press_enter", False)),
                field_label=str(params.get("field_label") or ""),
                verify=bool(params.get("verify", True)),
            )
            return result.to_dict()

        elif action == DesktopAction.SMART_RUN_COMMAND:
            command = str(params.get("command") or params.get("text") or "")
            if not command:
                raise DesktopControlError("smart_run_command requires command")
            result = await self._smart_input.smart_run_in_terminal(
                command=command,
                app_name=str(params.get("app_name") or "Terminal"),
                new_tab=bool(params.get("new_tab", True)),
            )
            return result.to_dict()

        elif action == DesktopAction.SMART_CREATE_NOTE:
            title = str(params.get("title") or "")
            body = str(params.get("body") or params.get("text") or "")
            if not body:
                raise DesktopControlError("smart_create_note requires body/text")
            result = await self._smart_input.smart_create_note(
                title=title,
                body=body,
                app_name=str(params.get("app_name") or "Notes"),
            )
            return result.to_dict()

        raise DesktopControlError(f"Unknown smart action: {action}")

    # ── Action Implementations ───────────────────────────────────

    def _open_app(self, params: dict[str, Any]) -> dict[str, Any]:
        app = str(params.get("app_name") or params.get("app") or "").strip()
        if not app:
            raise DesktopControlError("open_app requires app_name")
        self._check_allowed_app(app)
        new_instance = bool(params.get("new_instance", False))
        cmd = ["open", "-na" if new_instance else "-a", app]
        self._run(cmd, timeout=float(params.get("timeout_seconds") or 10))
        return {"app_name": app, "new_instance": new_instance}

    def _focus_app(self, params: dict[str, Any]) -> dict[str, Any]:
        app = str(params.get("app_name") or params.get("app") or "").strip()
        if not app:
            raise DesktopControlError("focus_app requires app_name")
        self._check_allowed_app(app)
        script = f'tell application "{self._escape_osascript(app)}" to activate'
        self._run_osascript(script, timeout=float(params.get("timeout_seconds") or 10))
        return {"app_name": app}

    def _open_url(self, params: dict[str, Any]) -> dict[str, Any]:
        url = str(params.get("url") or "").strip()
        if not url:
            raise DesktopControlError("open_url requires url")
        from urllib.parse import urlparse as _urlparse

        parsed = _urlparse(url)
        _allowed = {"http", "https", "mailto"}
        if not parsed.scheme or parsed.scheme.lower() not in _allowed:
            raise DesktopControlError(
                f"URL scheme '{parsed.scheme or '(none)'}' not allowed; only http/https/mailto"
            )
        self._run(["open", url], timeout=float(params.get("timeout_seconds") or 10))
        return {"url": url}

    def _type_text(self, params: dict[str, Any]) -> dict[str, Any]:
        text = str(params.get("text") or "")
        if text == "":
            raise DesktopControlError("type_text requires text")
        app = str(params.get("app_name") or "").strip()
        if app:
            self._focus_app({"app_name": app})
            time.sleep(float(params.get("focus_delay_seconds") or 0.05))

        safe_text = self._escape_osascript(text)
        script = (
            'tell application "System Events"\n'
            f'    keystroke "{safe_text}"\n'
            "end tell"
        )
        self._run_osascript(script, timeout=float(params.get("timeout_seconds") or 15))
        return {"typed_chars": len(text), "app_name": app or None}

    def _press_key(self, params: dict[str, Any]) -> dict[str, Any]:
        key = str(params.get("key") or "").strip().lower()
        key_code_raw = params.get("key_code")
        modifiers = params.get("modifiers") or []
        if not key and key_code_raw is None:
            raise DesktopControlError("press_key requires key or key_code")

        mod_clause = self._modifiers_clause(modifiers)
        app = str(params.get("app_name") or "").strip()
        if app:
            self._focus_app({"app_name": app})
            time.sleep(float(params.get("focus_delay_seconds") or 0.05))

        if key_code_raw is not None:
            try:
                key_code = int(key_code_raw)
            except (TypeError, ValueError) as exc:
                raise DesktopControlError("key_code must be an integer") from exc
            command = f"key code {key_code}{mod_clause}"
        elif len(key) == 1:
            command = f'keystroke "{self._escape_osascript(key)}"{mod_clause}'
        else:
            mapped = self._KEY_CODE_MAP.get(key)
            if mapped is None:
                raise DesktopControlError(f"unsupported key: {key}")
            command = f"key code {mapped}{mod_clause}"

        script = 'tell application "System Events"\n' f"    {command}\n" "end tell"
        self._run_osascript(script, timeout=float(params.get("timeout_seconds") or 10))
        return {"key": key or None, "key_code": key_code_raw, "modifiers": modifiers}

    def _mouse_move(self, params: dict[str, Any]) -> dict[str, Any]:
        pg = self._require_pyautogui()
        x_raw = params.get("x")
        y_raw = params.get("y")
        if x_raw is None or y_raw is None:
            raise DesktopControlError("mouse_move requires x and y coordinates")
        try:
            x = int(x_raw)
            y = int(y_raw)
        except (TypeError, ValueError) as exc:
            raise DesktopControlError("x and y must be integers") from exc
        if x < 0 or y < 0:
            raise DesktopControlError("x and y must be non-negative")
        duration = max(0.0, float(params.get("duration_seconds") or 0.0))
        pg.moveTo(x, y, duration=duration)
        return {"x": x, "y": y, "duration_seconds": duration}

    _VALID_MOUSE_BUTTONS = {"left", "right", "middle"}

    def _mouse_click(
        self,
        params: dict[str, Any],
        *,
        clicks: int = 1,
        button: str = "left",
    ) -> dict[str, Any]:
        pg = self._require_pyautogui()
        if button not in self._VALID_MOUSE_BUTTONS:
            raise DesktopControlError(f"invalid mouse button: {button}")
        x = params.get("x")
        y = params.get("y")
        if x is not None and y is not None:
            try:
                ix, iy = int(x), int(y)
            except (TypeError, ValueError) as exc:
                raise DesktopControlError("x and y must be integers") from exc
            if ix < 0 or iy < 0:
                raise DesktopControlError("x and y must be non-negative")
            pg.moveTo(ix, iy)
        pg.click(clicks=max(1, min(int(clicks), 5)), button=button)
        pos = pg.position()
        return {"x": int(pos[0]), "y": int(pos[1]), "clicks": clicks, "button": button}

    def _mouse_scroll(self, params: dict[str, Any]) -> dict[str, Any]:
        pg = self._require_pyautogui()
        try:
            amount = int(params.get("amount") or 0)
        except (TypeError, ValueError) as exc:
            raise DesktopControlError("scroll amount must be an integer") from exc
        scroll_max = getattr(self._ops_cfg, "scroll_max_amount", 100) if self._ops_cfg else 100
        amount = max(-scroll_max, min(amount, scroll_max))
        pg.scroll(amount)
        return {"amount": amount}

    def _get_mouse_position(self) -> dict[str, Any]:
        pg = self._require_pyautogui()
        pos = pg.position()
        return {"x": int(pos[0]), "y": int(pos[1])}

    @with_retry(
        max_attempts=3,
        backoff_base=0.3,
        retryable_errors=(DesktopControlError,),
        retryable_messages=(
            "timed out",
            "connection reset",
            "resource busy",
            "screencapture",
            "cannot complete",
        ),
    )
    def _screenshot(self, params: dict[str, Any]) -> dict[str, Any]:
        """Capture screenshot with quality modes: full (PNG), fast (JPEG), thumbnail (JPEG+resize).

        Auto-retries on transient screencapture failures (up to 3 attempts).
        """
        target = str(params.get("path") or "").strip()
        include_b64 = bool(params.get("include_base64", False))
        quality_str = str(params.get("quality") or "full").strip().lower()

        # Validate quality via enum
        try:
            quality = ScreenshotQuality(quality_str)
        except ValueError:
            quality = ScreenshotQuality.FULL

        use_jpeg = quality in (ScreenshotQuality.FAST, ScreenshotQuality.THUMBNAIL)
        suffix = ".jpg" if use_jpeg else ".png"

        ss_max = getattr(self._ops_cfg, "screenshot_max_b64_bytes", 10_000_000) if self._ops_cfg else 10_000_000
        max_b64_bytes = max(0, min(int(params.get("max_base64_bytes") or 2_500_000), ss_max))

        if target:
            out_path = Path(target).expanduser().resolve()
            home = Path.home().resolve()
            tmp = Path(tempfile.gettempdir()).resolve()
            if not (out_path.is_relative_to(home) or out_path.is_relative_to(tmp)):
                raise DesktopControlError(
                    f"screenshot path must be under home ({home}) or tmp ({tmp})"
                )
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            fd, temp_path = tempfile.mkstemp(prefix="jarvis_screen_", suffix=suffix)
            os.close(fd)
            out_path = Path(temp_path)

        cmd = ["screencapture", "-x"]
        if use_jpeg:
            cmd.extend(["-t", "jpg"])

        # Region capture
        if all(k in params for k in ("x", "y", "width", "height")):
            x = int(params["x"])
            y = int(params["y"])
            w = int(params["width"])
            h = int(params["height"])
            if x < 0 or y < 0:
                raise DesktopControlError("screenshot region x/y must be non-negative")
            if w <= 0 or h <= 0:
                raise DesktopControlError("screenshot region width/height must be positive")
            cmd.extend(["-R", f"{x},{y},{w},{h}"])

        cmd.append(str(out_path))
        try:
            self._run(cmd, timeout=float(params.get("timeout_seconds") or 15))
        except Exception:
            if not target:
                Path(out_path).unlink(missing_ok=True)
            raise

        # Re-resolve after capture to guard against symlink race (TOCTOU)
        final_real = Path(out_path).resolve()
        if not any(str(final_real).startswith(str(r)) for r in [Path.home().resolve(), Path(tempfile.gettempdir()).resolve()]):
            raise DesktopControlError("Screenshot path changed after capture")

        # Resize for thumbnail mode (PIL in-process, avoids sips subprocess)
        if quality == ScreenshotQuality.THUMBNAIL and out_path.exists():
            try:
                from PIL import Image
                img = Image.open(out_path)
                img.thumbnail((960, 960))
                img.save(out_path, "JPEG", quality=75)
            except ImportError:
                pass  # Skip resize if PIL not available
            except (OSError, ValueError) as e:
                logger.debug("Thumbnail resize failed (non-critical): %s", e)

        size = out_path.stat().st_size if out_path.exists() else 0
        result: dict[str, Any] = {"path": str(out_path), "size_bytes": size, "quality": quality.value}

        if include_b64:
            if size > max_b64_bytes:
                result["base64_skipped"] = (
                    f"screenshot too large for inline base64 ({size} bytes)"
                )
            else:
                result["image_b64"] = base64.b64encode(out_path.read_bytes()).decode("ascii")

        # Clean up temp file when no explicit target path was given
        if not target and out_path.exists():
            out_path.unlink()

        return result

    def _frontmost_app(self) -> dict[str, Any]:
        script = (
            'tell application "System Events"\n'
            "    name of first application process whose frontmost is true\n"
            "end tell"
        )
        out = self._run_osascript(script, timeout=5)
        return {"app_name": out.strip()}

    def health_check(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Check Accessibility API status and return health diagnostics."""
        osascript_ok = False
        osascript_error: str | None = None
        try:
            out = self._run_osascript('return "ok"', timeout=3)
            osascript_ok = out.strip().lower() == "ok"
            if not osascript_ok:
                osascript_error = f"unexpected osascript output: {out!r}"
        except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
            osascript_error = str(exc)

        pyautogui_ok = False
        pyautogui_error: str | None = None
        try:
            if self._pyautogui is None:
                import pyautogui  # type: ignore

                self._pyautogui = pyautogui
            pyautogui_ok = True
        except ImportError as exc:
            pyautogui_error = str(exc)

        hints: list[str] = []
        if not osascript_ok:
            hints.append(
                "Grant Terminal/JARVIS Automation + Accessibility permissions in "
                "System Settings > Privacy & Security."
            )
        if not pyautogui_ok:
            hints.append("Install mouse automation dependency: pip install pyautogui")

        native: dict[str, Any] = {
            "enabled": self._native_service is not None,
            "fallback_enabled": self._native_fallback_enabled,
        }
        if self._native_service is not None:
            try:
                native_check = self._native_service.health_check()
                native.update(native_check)
                if not bool(native_check.get("accessibility_trusted", False)):
                    hints.append(
                        "For AX actions, enable Accessibility for the Python/JARVIS process "
                        "in System Settings > Privacy & Security > Accessibility."
                    )
            except (OSError, RuntimeError) as exc:
                native["error"] = str(exc)
                hints.append(f"Native backend check failed: {exc}")

        smart_input: dict[str, Any] = {
            "enabled": self._smart_input is not None,
        }
        if self._smart_input is not None:
            smart_input["supported_actions"] = sorted(SMART_ACTIONS)
            smart_input["input_methods"] = [
                "applescript_do_script",
                "ax_set_value",
                "keystroke_verified",
                "keystroke_unverified",
            ]

        return {
            "platform": platform.system(),
            "osascript": {"ok": osascript_ok, "error": osascript_error},
            "pyautogui": {"ok": pyautogui_ok, "error": pyautogui_error},
            "native_backend": native,
            "smart_input": smart_input,
            "async_bridge": {"active": AsyncBridge._instance is not None},
            "allowed_apps": sorted(self._allowed_apps),
            "telemetry": self.telemetry(),
            "hints": hints,
        }

    def _sleep(self, params: dict[str, Any]) -> dict[str, Any]:
        duration = float(params.get("seconds") or 0.0)
        sleep_max = getattr(self._ops_cfg, "sleep_max_seconds", 60.0) if self._ops_cfg else 60.0
        if duration < 0 or duration > sleep_max:
            raise DesktopControlError(f"sleep.seconds must be between 0 and {sleep_max}")
        time.sleep(duration)
        return {"seconds": duration}

    def _check_allowed_app(self, app_name: str) -> None:
        if not self._allowed_apps:
            return
        if app_name.strip().lower() not in self._allowed_apps:
            raise DesktopControlError(
                f"app '{app_name}' blocked by JARVIS_DESKTOP_ALLOWED_APPS policy"
            )

    _pyautogui_lock = threading.Lock()

    def _require_pyautogui(self):
        if self._pyautogui is None:
            with self._pyautogui_lock:
                if self._pyautogui is None:  # double-check after lock
                    try:
                        import pyautogui  # type: ignore
                    except ImportError as exc:  # pragma: no cover - env dependent
                        raise DesktopControlError(
                            "pyautogui is required for mouse actions; install with `pip install pyautogui`"
                        ) from exc
                    self._pyautogui = pyautogui
        return self._pyautogui

    @staticmethod
    def _escape_osascript(text: str) -> str:
        """Escape text for use inside AppleScript double-quoted strings."""
        # Canonical impl lives in smart_input._escape_applescript;
        # duplicated here to avoid circular import in the legacy path.
        text = str(text)
        text = text.replace("\\", "\\\\")
        text = text.replace('"', '\\"')
        text = text.replace("\n", "\\n")
        text = text.replace("\r", "\\r")
        text = text.replace("\t", "\\t")
        return text

    @with_retry(
        max_attempts=3,
        backoff_base=0.2,
        retryable_errors=(DesktopControlError,),
        retryable_messages=(
            "kAXErrorCannotComplete",
            "timed out",
            "connection reset",
            "resource busy",
            "execution error",
        ),
    )
    def _run_osascript(self, script: str, timeout: float = 10) -> str:
        """Execute AppleScript with automatic retry on transient failures.

        Common transient errors: kAXErrorCannotComplete, osascript timeout,
        System Events resource busy. Retries up to 3 times with 0.2s→0.4s backoff.
        """
        proc = self._run(
            ["osascript", "-"],
            timeout=timeout,
            input_text=script,
        )
        return proc.stdout.strip()

    def _run(
        self,
        cmd: list[str],
        *,
        timeout: float = 10,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                cmd,
                input=input_text,
                capture_output=True,
                text=True,
                check=True,
                timeout=max(1.0, float(timeout)),
            )
        except subprocess.TimeoutExpired as exc:
            raise DesktopControlError(f"command timed out: {' '.join(cmd)}") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            detail = stderr or stdout or str(exc)
            raise DesktopControlError(detail) from exc

    def _modifiers_clause(self, modifiers: Any) -> str:
        if not modifiers:
            return ""
        if not isinstance(modifiers, list):
            raise DesktopControlError("modifiers must be an array of strings")
        mapped: list[str] = []
        for item in modifiers:
            key = str(item).strip().lower()
            mod = self._MODIFIER_MAP.get(key)
            if mod is None:
                raise DesktopControlError(f"unsupported modifier: {item}")
            mapped.append(mod)
        if not mapped:
            return ""
        return " using {" + ", ".join(mapped) + "}"
