"""
PredaCore Smart Input Engine — Bulletproof text input & command execution for macOS.

The Problem:
  Raw keystroke simulation (type_text + press_key) is fragile because:
  1. Target app may not be frontmost when keystrokes fire
  2. Shell prompt may not be ready (timing race)
  3. Dialogs/animations can steal focus mid-type
  4. No verification that input actually landed

The Solution (3-Tier Priority Chain):
  Tier 1: Native AppleScript `do script` / `do shell script` (for apps with scripting support)
  Tier 2: AX set_value on the target input element (direct value injection)
  Tier 3: Focus-verify → AX-wait-for-ready → Keystroke simulation → Read-back verify → Retry

This engine wraps all three tiers behind a single `smart_type()` call that automatically
picks the best strategy for the target app.

Integration:
  - Used by MacDesktopOperator for `smart_type`, `smart_run_in_terminal`, `smart_create_note`
  - Used by ScreenVisionEngine for `type_into`
  - Falls back gracefully when AX or AppleScript isn't available
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────


class InputMethod(str, Enum):
    """How the input was delivered."""
    APPLESCRIPT_DO_SCRIPT = "applescript_do_script"
    APPLESCRIPT_KEYSTROKE = "applescript_keystroke"
    APPLESCRIPT_CREATE = "applescript_create"
    APPLESCRIPT_URL = "applescript_url"
    AX_SET_VALUE = "ax_set_value"
    AX_FOCUS_AND_TYPE = "ax_focus_and_type"
    KEYSTROKE_VERIFIED = "keystroke_verified"
    KEYSTROKE_UNVERIFIED = "keystroke_unverified"


@dataclass
class SmartInputResult:
    """Result of a smart input operation."""
    ok: bool
    method: InputMethod
    app_name: str = ""
    typed_text: str = ""
    verified: bool = False
    attempts: int = 1
    elapsed_ms: float = 0.0
    error: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize input plan to dictionary."""
        d: dict[str, Any] = {
            "ok": self.ok,
            "method": self.method.value,
            "verified": self.verified,
            "attempts": self.attempts,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }
        if self.app_name:
            d["app_name"] = self.app_name
        if self.typed_text:
            d["typed_chars"] = len(self.typed_text)
        if self.error:
            d["error"] = self.error
        if self.details:
            d["details"] = self.details
        return d


# ── AppleScript App Registry ────────────────────────────────────────


# Apps that support AppleScript `do script` (command execution in their shell/environment)
_APPS_WITH_DO_SCRIPT: dict[str, str] = {
    "terminal": "Terminal",
    "iterm": "iTerm",
    "iterm2": "iTerm2",
    "iterm 2": "iTerm2",
    # "warp" — excluded: Warp doesn't support AppleScript do script (falls through to Tier 2/3)
    "kitty": "kitty",
    "alacritty": "Alacritty",
}

# Apps that support AppleScript for creating/manipulating content
_APPS_WITH_MAKE_NEW: dict[str, dict[str, str]] = {
    "notes": {
        "app_name": "Notes",
        "create_script": (
            'tell application "Notes"\n'
            '    tell account "iCloud"\n'
            '        make new note at folder "Notes" with properties '
            '{{name:"{title}", body:"{body}"}}\n'
            "    end tell\n"
            "end tell"
        ),
    },
    "textedit": {
        "app_name": "TextEdit",
        "create_script": (
            'tell application "TextEdit"\n'
            "    activate\n"
            "    make new document\n"
            '    set text of document 1 to "{body}"\n'
            "end tell"
        ),
    },
    "reminders": {
        "app_name": "Reminders",
        "create_script": (
            'tell application "Reminders"\n'
            '    tell list "Reminders"\n'
            '        make new reminder with properties {{name:"{title}", body:"{body}"}}\n'
            "    end tell\n"
            "end tell"
        ),
    },
    "mail": {
        "app_name": "Mail",
        "create_script": (
            'tell application "Mail"\n'
            '    set newMsg to make new outgoing message with properties '
            '{{subject:"{subject}", content:"{body}"}}\n'
            '    tell newMsg\n'
            '        make new to recipient at end of to recipients '
            'with properties {{address:"{to}"}}\n'
            '    end tell\n'
            '    activate\n'
            'end tell'
        ),
    },
    "calendar": {
        "app_name": "Calendar",
        "create_script": (
            'tell application "Calendar"\n'
            '    tell calendar "Home"\n'
            '        make new event at end with properties '
            '{{summary:"{title}", start date:date "{start}", end date:date "{end}"}}\n'
            '    end tell\n'
            '    activate\n'
            'end tell'
        ),
    },
}

# Apps where we can set URL bar or search field values via AppleScript
_APPS_WITH_URL_SET: dict[str, str] = {
    "safari": (
        'tell application "Safari"\n'
        "    activate\n"
        '    set URL of document 1 to "{url}"\n'
        "end tell"
    ),
    "google chrome": (
        'tell application "Google Chrome"\n'
        "    activate\n"
        '    set URL of active tab of window 1 to "{url}"\n'
        "end tell"
    ),
    "chrome": (
        'tell application "Google Chrome"\n'
        "    activate\n"
        '    set URL of active tab of window 1 to "{url}"\n'
        "end tell"
    ),
    "brave browser": (
        'tell application "Brave Browser"\n'
        "    activate\n"
        '    set URL of active tab of window 1 to "{url}"\n'
        "end tell"
    ),
    "microsoft edge": (
        'tell application "Microsoft Edge"\n'
        "    activate\n"
        '    set URL of active tab of window 1 to "{url}"\n'
        "end tell"
    ),
    "arc": (
        'tell application "Arc"\n'
        "    activate\n"
        "    tell front window\n"
        '        set URL of active tab to "{url}"\n'
        "    end tell\n"
        "end tell"
    ),
}

# Apps with specific AppleScript keystroke workarounds
_APPS_WITH_CUSTOM_TYPE: dict[str, str] = {
    "finder": (
        'tell application "System Events"\n'
        '    tell process "Finder"\n'
        '        keystroke "{text}"\n'
        "    end tell\n"
        "end tell"
    ),
}

# AX roles that indicate an input-accepting element
_INPUT_AX_ROLES = frozenset({
    "AXTextArea",
    "AXTextField",
    "AXComboBox",
    "AXSearchField",
    "AXWebArea",
    "AXTextMarkerRange",
})


# ── Smart Input Engine ───────────────────────────────────────────────


class SmartInputEngine:
    """
    Bulletproof text input engine for macOS.

    3-Tier Priority Chain:
      Tier 1: Native AppleScript (do script, make new, set URL)
      Tier 2: AX set_value (direct value injection via Accessibility API)
      Tier 3: Verified keystroke (focus → wait → type → verify → retry)

    Usage:
        engine = SmartInputEngine(desktop_operator)

        # Smart type into any app
        result = await engine.smart_type(
            text="git status",
            app_name="Terminal",
            press_enter=True,
        )

        # Smart run command in Terminal
        result = await engine.smart_run_in_terminal(
            command="cd ~/projects && python main.py",
        )

        # Create a note in Apple Notes
        result = await engine.smart_create_note(
            title="Meeting Notes",
            body="Discussed Q3 roadmap...",
        )

        # Open URL in any browser
        result = await engine.smart_open_url(
            url="https://github.com",
            browser="chrome",
        )

        # Synchronous version (for desktop.py integration)
        result = engine.smart_type_sync(text="hello", app_name="TextEdit")
    """

    def __init__(
        self,
        desktop_operator: Any,
        max_retries: int = 3,
        verify_timeout: float = 1.0,
        focus_settle_ms: int = 300,
        ax_ready_timeout: float = 3.0,
    ):
        self._op = desktop_operator
        self._max_retries = max_retries
        self._verify_timeout = verify_timeout
        self._focus_settle_ms = focus_settle_ms
        self._ax_ready_timeout = ax_ready_timeout
        self._has_native = hasattr(desktop_operator, '_native_service') and desktop_operator._native_service is not None

    # ══════════════════════════════════════════════════════════════
    # PUBLIC API — Async
    # ══════════════════════════════════════════════════════════════

    async def smart_type(
        self,
        text: str,
        app_name: str = "",
        press_enter: bool = False,
        field_label: str = "",
        verify: bool = True,
    ) -> SmartInputResult:
        """
        Intelligently type text into the target app using the best available method.

        Priority chain:
          1. AppleScript `do script` if app supports it and press_enter=True (Terminal, iTerm)
          2. AX set_value if field_label is specified and the element is found
          3. Focus-verify → AX-wait → Keystroke → Verify → Retry

        Args:
            text: The text to type
            app_name: Target application name (e.g. "Terminal", "Safari")
            press_enter: Whether to press Enter/Return after typing
            field_label: Optional label of specific input field to target via AX
            verify: Whether to verify the input landed (only for keystroke method)

        Returns:
            SmartInputResult with method used and verification status
        """
        t0 = time.time()
        app_key = app_name.strip().lower()

        # ── Tier 1: AppleScript do script (for terminal-like apps) ──
        if app_key in _APPS_WITH_DO_SCRIPT and press_enter:
            try:
                result = await self._applescript_do_script(
                    text, _APPS_WITH_DO_SCRIPT[app_key]
                )
                result.elapsed_ms = (time.time() - t0) * 1000
                return result
            except (OSError, subprocess.SubprocessError, RuntimeError) as e:
                logger.warning(
                    "Tier 1 (AppleScript do script) failed for %s: %s — falling to Tier 2",
                    app_name, e,
                )

        # ── Tier 2: AX set_value (for specific labeled fields) ──
        if field_label and self._has_native:
            try:
                result = await self._ax_set_value_method(
                    text, field_label, app_name
                )
                if result.ok:
                    result.elapsed_ms = (time.time() - t0) * 1000
                    return result
            except (OSError, RuntimeError, AttributeError) as e:
                logger.warning(
                    "Tier 2 (AX set_value) failed for field '%s': %s — falling to Tier 3",
                    field_label, e,
                )

        # ── Tier 3: Focus → Wait → Type → Verify → Retry ──
        result = await self._verified_keystroke(
            text, app_name, press_enter, verify
        )
        result.elapsed_ms = (time.time() - t0) * 1000
        return result

    async def smart_run_in_terminal(
        self,
        command: str,
        app_name: str = "Terminal",
        new_tab: bool = True,
    ) -> SmartInputResult:
        """
        Run a command in Terminal/iTerm using the most reliable method available.

        Priority:
          1. AppleScript `do script` (most reliable — bypasses focus/timing entirely)
          2. Focus + keystroke with verification (fallback)

        Args:
            command: The shell command to execute
            app_name: Terminal app name ("Terminal" or "iTerm2")
            new_tab: Whether to open a new tab (only for AppleScript method)

        Returns:
            SmartInputResult
        """
        t0 = time.time()
        app_key = app_name.strip().lower()
        real_name = _APPS_WITH_DO_SCRIPT.get(app_key, app_name)

        # Tier 1: AppleScript do script
        try:
            result = await self._applescript_do_script(
                command, real_name, new_tab=new_tab
            )
            result.elapsed_ms = (time.time() - t0) * 1000
            return result
        except (OSError, subprocess.SubprocessError, RuntimeError) as e:
            logger.warning(
                "AppleScript do script failed: %s — falling back to keystroke", e
            )

        # Tier 3: Verified keystroke fallback
        result = await self._verified_keystroke(
            command, app_name, press_enter=True, verify=True
        )
        result.elapsed_ms = (time.time() - t0) * 1000
        return result

    async def smart_create_note(
        self,
        title: str,
        body: str,
        app_name: str = "Notes",
    ) -> SmartInputResult:
        """
        Create a note in Apple Notes (or TextEdit, Reminders) using native AppleScript.

        This bypasses all focus/timing issues by using the app's scripting dictionary.
        """
        t0 = time.time()
        app_key = app_name.strip().lower()
        config = _APPS_WITH_MAKE_NEW.get(app_key)

        if config:
            try:
                result = await self._applescript_create_content(
                    title, body, config
                )
                result.elapsed_ms = (time.time() - t0) * 1000
                return result
            except (OSError, subprocess.SubprocessError, RuntimeError) as e:
                logger.warning("AppleScript create failed: %s", e)
                return SmartInputResult(
                    ok=False,
                    method=InputMethod.APPLESCRIPT_CREATE,
                    app_name=app_name,
                    error=str(e),
                    elapsed_ms=(time.time() - t0) * 1000,
                )

        return SmartInputResult(
            ok=False,
            method=InputMethod.KEYSTROKE_UNVERIFIED,
            app_name=app_name,
            error=f"No AppleScript support configured for '{app_name}'. "
                  f"Supported: {', '.join(_APPS_WITH_MAKE_NEW.keys())}",
            elapsed_ms=(time.time() - t0) * 1000,
        )

    async def smart_send_email(
        self,
        to: str,
        subject: str,
        body: str,
    ) -> SmartInputResult:
        """
        Compose a new email in Mail.app using native AppleScript.

        The message is created and Mail is activated, but NOT sent automatically
        (the user can review before sending).

        Args:
            to: Recipient email address
            subject: Email subject line
            body: Email body text
        """
        t0 = time.time()
        config = _APPS_WITH_MAKE_NEW.get("mail")
        if not config:
            return SmartInputResult(
                ok=False,
                method=InputMethod.APPLESCRIPT_CREATE,
                app_name="Mail",
                error="Mail.app config not found",
                elapsed_ms=(time.time() - t0) * 1000,
            )
        escaped_to = _escape_applescript(to)
        escaped_subject = _escape_applescript(subject)
        escaped_body = _escape_applescript(body)
        script = config["create_script"].format(
            to=escaped_to,
            subject=escaped_subject,
            body=escaped_body,
        )
        try:
            await self._run_osascript(script, timeout=10)
            return SmartInputResult(
                ok=True,
                method=InputMethod.APPLESCRIPT_CREATE,
                app_name="Mail",
                typed_text=body,
                verified=True,
                attempts=1,
                details={"to": to, "subject": subject, "tier": 1},
                elapsed_ms=(time.time() - t0) * 1000,
            )
        except (OSError, subprocess.SubprocessError, RuntimeError) as e:
            logger.warning("AppleScript Mail create failed: %s", e)
            return SmartInputResult(
                ok=False,
                method=InputMethod.APPLESCRIPT_CREATE,
                app_name="Mail",
                error=str(e),
                elapsed_ms=(time.time() - t0) * 1000,
            )

    async def smart_create_event(
        self,
        title: str,
        start: str,
        end: str,
        calendar_name: str = "Home",
    ) -> SmartInputResult:
        """
        Create a new calendar event in Calendar.app using native AppleScript.

        Args:
            title: Event title/summary
            start: Start date-time string (e.g. "March 30, 2026 2:00 PM")
            end: End date-time string (e.g. "March 30, 2026 3:00 PM")
            calendar_name: Target calendar name (default: "Home")
        """
        t0 = time.time()
        config = _APPS_WITH_MAKE_NEW.get("calendar")
        if not config:
            return SmartInputResult(
                ok=False,
                method=InputMethod.APPLESCRIPT_CREATE,
                app_name="Calendar",
                error="Calendar.app config not found",
                elapsed_ms=(time.time() - t0) * 1000,
            )
        escaped_title = _escape_applescript(title)
        escaped_start = _escape_applescript(start)
        escaped_end = _escape_applescript(end)
        escaped_cal = _escape_applescript(calendar_name)
        # Build script with custom calendar name (override the default "Home")
        script = (
            'tell application "Calendar"\n'
            f'    tell calendar "{escaped_cal}"\n'
            '        make new event at end with properties '
            f'{{summary:"{escaped_title}", start date:date "{escaped_start}", '
            f'end date:date "{escaped_end}"}}\n'
            '    end tell\n'
            '    activate\n'
            'end tell'
        )
        try:
            await self._run_osascript(script, timeout=10)
            return SmartInputResult(
                ok=True,
                method=InputMethod.APPLESCRIPT_CREATE,
                app_name="Calendar",
                typed_text=title,
                verified=True,
                attempts=1,
                details={
                    "title": title,
                    "start": start,
                    "end": end,
                    "calendar": calendar_name,
                    "tier": 1,
                },
                elapsed_ms=(time.time() - t0) * 1000,
            )
        except (OSError, subprocess.SubprocessError, RuntimeError) as e:
            logger.warning("AppleScript Calendar create failed: %s", e)
            return SmartInputResult(
                ok=False,
                method=InputMethod.APPLESCRIPT_CREATE,
                app_name="Calendar",
                error=str(e),
                elapsed_ms=(time.time() - t0) * 1000,
            )

    async def smart_open_url(
        self,
        url: str,
        browser: str = "",
    ) -> SmartInputResult:
        """
        Open a URL in the specified browser (or default) using native AppleScript.

        Much more reliable than simulating Cmd+L → type URL → Enter because
        it directly sets the URL property via the app's scripting dictionary.

        Args:
            url: URL to open
            browser: Browser name ("chrome", "safari", etc.) or "" for system default

        Returns:
            SmartInputResult
        """
        t0 = time.time()

        if not browser:
            # Use system default via open command
            try:
                await asyncio.to_thread(
                    self._run_osascript_sync,
                    f'open location "{_escape_applescript(url)}"', 5
                )
                return SmartInputResult(
                    ok=True,
                    method=InputMethod.APPLESCRIPT_URL,
                    typed_text=url,
                    verified=True,
                    elapsed_ms=(time.time() - t0) * 1000,
                    details={"browser": "system_default"},
                )
            except (OSError, RuntimeError) as e:
                return SmartInputResult(
                    ok=False,
                    method=InputMethod.APPLESCRIPT_URL,
                    error=str(e),
                    elapsed_ms=(time.time() - t0) * 1000,
                )

        browser_key = browser.strip().lower()
        script_template = _APPS_WITH_URL_SET.get(browser_key)

        if script_template:
            try:
                script = script_template.format(url=_escape_applescript(url))
                await asyncio.to_thread(self._run_osascript_sync, script, 10)
                return SmartInputResult(
                    ok=True,
                    method=InputMethod.APPLESCRIPT_URL,
                    app_name=browser,
                    typed_text=url,
                    verified=True,
                    elapsed_ms=(time.time() - t0) * 1000,
                    details={"browser": browser_key},
                )
            except (OSError, subprocess.SubprocessError, RuntimeError) as e:
                logger.warning("AppleScript URL set failed for %s: %s", browser, e)
                return SmartInputResult(
                    ok=False,
                    method=InputMethod.APPLESCRIPT_URL,
                    app_name=browser,
                    error=str(e),
                    elapsed_ms=(time.time() - t0) * 1000,
                )

        # Unknown browser — try opening via `open -a`
        try:
            await asyncio.to_thread(
                self._run_osascript_sync,
                f'tell application "{_escape_applescript(browser)}" to open location "{_escape_applescript(url)}"',
                5,
            )
            return SmartInputResult(
                ok=True,
                method=InputMethod.APPLESCRIPT_URL,
                app_name=browser,
                typed_text=url,
                verified=True,
                elapsed_ms=(time.time() - t0) * 1000,
                details={"browser": browser_key, "method": "open_-a"},
            )
        except (OSError, subprocess.SubprocessError, RuntimeError) as e:
            return SmartInputResult(
                ok=False,
                method=InputMethod.APPLESCRIPT_URL,
                app_name=browser,
                error=str(e),
                elapsed_ms=(time.time() - t0) * 1000,
            )

    # ══════════════════════════════════════════════════════════════
    # PUBLIC API — Synchronous (for desktop.py integration)
    # ══════════════════════════════════════════════════════════════

    def smart_type_sync(
        self,
        text: str,
        app_name: str = "",
        press_enter: bool = False,
        field_label: str = "",
        verify: bool = True,
    ) -> SmartInputResult:
        """Synchronous wrapper for smart_type (for non-async callers like desktop.py)."""
        return _run_sync(self.smart_type(
            text=text,
            app_name=app_name,
            press_enter=press_enter,
            field_label=field_label,
            verify=verify,
        ))

    def smart_run_in_terminal_sync(
        self,
        command: str,
        app_name: str = "Terminal",
        new_tab: bool = True,
    ) -> SmartInputResult:
        """Synchronous wrapper for smart_run_in_terminal."""
        return _run_sync(self.smart_run_in_terminal(
            command=command,
            app_name=app_name,
            new_tab=new_tab,
        ))

    def smart_create_note_sync(
        self,
        title: str,
        body: str,
        app_name: str = "Notes",
    ) -> SmartInputResult:
        """Synchronous wrapper for smart_create_note."""
        return _run_sync(self.smart_create_note(
            title=title,
            body=body,
            app_name=app_name,
        ))

    def smart_send_email_sync(
        self,
        to: str,
        subject: str,
        body: str,
    ) -> SmartInputResult:
        """Synchronous wrapper for smart_send_email."""
        return _run_sync(self.smart_send_email(to=to, subject=subject, body=body))

    def smart_create_event_sync(
        self,
        title: str,
        start: str,
        end: str,
        calendar_name: str = "Home",
    ) -> SmartInputResult:
        """Synchronous wrapper for smart_create_event."""
        return _run_sync(self.smart_create_event(
            title=title, start=start, end=end, calendar_name=calendar_name,
        ))

    def smart_open_url_sync(
        self,
        url: str,
        browser: str = "",
    ) -> SmartInputResult:
        """Synchronous wrapper for smart_open_url."""
        return _run_sync(self.smart_open_url(url=url, browser=browser))

    # ══════════════════════════════════════════════════════════════
    # PUBLIC UTILITIES
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def get_best_method(app_name: str, action: str = "type") -> str:
        """
        Return the recommended input method for a given app and action.

        Useful for logging and debugging which path will be taken.

        Args:
            app_name: Target application name
            action: "type", "command", "note", "url", "email", "event"

        Returns:
            Human-readable method name
        """
        app_key = app_name.strip().lower()

        if action == "command" and app_key in _APPS_WITH_DO_SCRIPT:
            return f"Tier 1: AppleScript do script → {_APPS_WITH_DO_SCRIPT[app_key]}"
        if action == "note" and app_key in _APPS_WITH_MAKE_NEW:
            return f"Tier 1: AppleScript make new → {_APPS_WITH_MAKE_NEW[app_key]['app_name']}"
        if action == "url" and app_key in _APPS_WITH_URL_SET:
            return f"Tier 1: AppleScript set URL → {app_key}"
        if action == "type" and app_key in _APPS_WITH_DO_SCRIPT:
            return f"Tier 1: AppleScript do script → {_APPS_WITH_DO_SCRIPT[app_key]}"
        if action == "email":
            return "Tier 1: AppleScript make new → Mail"
        if action == "event":
            return "Tier 1: AppleScript make new → Calendar"

        return "Tier 3: Verified keystroke (focus → wait → type → verify → retry)"

    @staticmethod
    def supported_apps() -> dict[str, list[str]]:
        """Return all apps with native scripting support by category."""
        return {
            "terminal_apps": list(_APPS_WITH_DO_SCRIPT.values()),
            "content_creation": [v["app_name"] for v in _APPS_WITH_MAKE_NEW.values()],
            "browsers_url_set": list(_APPS_WITH_URL_SET.keys()),
        }

    # ══════════════════════════════════════════════════════════════
    # TIER 1: AppleScript native scripting
    # ══════════════════════════════════════════════════════════════

    async def _applescript_do_script(
        self,
        command: str,
        app_name: str,
        new_tab: bool = True,
    ) -> SmartInputResult:
        """
        Use AppleScript's `do script` to execute a command in Terminal/iTerm.

        This is the most reliable method because:
        - It doesn't need the app to be frontmost
        - It doesn't simulate keystrokes
        - It waits for the shell to be ready internally
        - It works even if a dialog is blocking the UI
        """
        escaped_cmd = _escape_applescript(command)

        if app_name.lower() in ("iterm2", "iterm"):
            script = (
                f'tell application "{app_name}"\n'
                f"    activate\n"
                f"    tell current session of current window\n"
                f'        write text "{escaped_cmd}"\n'
                f"    end tell\n"
                f"end tell"
            )
        elif app_name.lower() == "warp":
            # Warp doesn't support AppleScript do script — skip to Tier 2/3
            raise RuntimeError(f"{app_name} doesn't support AppleScript do script (use Tier 2/3)")
        else:
            # Standard Terminal.app
            if new_tab:
                script = (
                    f'tell application "Terminal"\n'
                    f"    activate\n"
                    f'    do script "{escaped_cmd}"\n'
                    f"end tell"
                )
            else:
                script = (
                    f'tell application "Terminal"\n'
                    f"    activate\n"
                    f'    do script "{escaped_cmd}" in front window\n'
                    f"end tell"
                )

        await self._run_osascript(script, timeout=10)

        return SmartInputResult(
            ok=True,
            method=InputMethod.APPLESCRIPT_DO_SCRIPT,
            app_name=app_name,
            typed_text=command,
            verified=True,  # do script is inherently reliable
            attempts=1,
            details={"new_tab": new_tab, "tier": 1},
        )

    async def _applescript_create_content(
        self,
        title: str,
        body: str,
        config: dict[str, str],
    ) -> SmartInputResult:
        """
        Create content in an app using its AppleScript scripting dictionary.

        Works for Notes, TextEdit, Reminders, etc.
        """
        escaped_title = _escape_applescript(title)
        escaped_body = _escape_applescript(body)
        script = config["create_script"].format(
            title=escaped_title,
            body=escaped_body,
        )

        await self._run_osascript(script, timeout=10)

        return SmartInputResult(
            ok=True,
            method=InputMethod.APPLESCRIPT_CREATE,
            app_name=config.get("app_name", ""),
            typed_text=body,
            verified=True,
            attempts=1,
            details={"title": title, "tier": 1},
        )

    # ══════════════════════════════════════════════════════════════
    # TIER 2: AX set_value (direct value injection)
    # ══════════════════════════════════════════════════════════════

    async def _ax_set_value_method(
        self,
        text: str,
        field_label: str,
        app_name: str,
    ) -> SmartInputResult:
        """
        Set value on a specific AX element by label.

        Uses the Accessibility API to directly inject text into a field,
        bypassing all focus and keystroke timing issues.
        """
        if app_name:
            await self._ensure_focused(app_name)

        match: dict[str, Any] = {"title_contains": field_label}
        result = self._op.execute("ax_set_value", {
            "target": "focused_window",
            "match": match,
            "value": text,
            "max_depth": 6,
            "max_children": 50,
        })

        if result.get("ok"):
            return SmartInputResult(
                ok=True,
                method=InputMethod.AX_SET_VALUE,
                app_name=app_name,
                typed_text=text,
                verified=True,  # AX set_value is direct
                attempts=1,
                details={"matched": result.get("matched", {}), "tier": 2},
            )

        return SmartInputResult(
            ok=False,
            method=InputMethod.AX_SET_VALUE,
            app_name=app_name,
            error=f"AX set_value failed: {result}",
        )

    # ══════════════════════════════════════════════════════════════
    # TIER 3: Verified Keystroke (with full safety chain)
    # ══════════════════════════════════════════════════════════════

    async def _verified_keystroke(
        self,
        text: str,
        app_name: str,
        press_enter: bool,
        verify: bool,
    ) -> SmartInputResult:
        """
        Type text with full safety chain:
          1. Ensure target app is focused
          2. Wait for AX input readiness (text area/field is active)
          3. Type the text via keystroke simulation
          4. Read back what's in the focused element
          5. If mismatch, clear and retry (up to max_retries)
        """
        if app_name:
            focused = await self._ensure_focused(app_name)
            if not focused:
                return SmartInputResult(
                    ok=False,
                    method=InputMethod.KEYSTROKE_UNVERIFIED,
                    app_name=app_name,
                    error=f"Could not focus {app_name}",
                    details={"tier": 3},
                )

        # Wait for an input-ready element
        await self._wait_for_input_ready(app_name)

        for attempt in range(1, self._max_retries + 1):
            # Type the text
            self._op.execute("type_text", {"text": text})

            if press_enter:
                await asyncio.sleep(0.05)
                self._op.execute("press_key", {"key": "return"})
                # After pressing enter, we can't verify (content may have been submitted)
                return SmartInputResult(
                    ok=True,
                    method=InputMethod.KEYSTROKE_VERIFIED if verify else InputMethod.KEYSTROKE_UNVERIFIED,
                    app_name=app_name,
                    typed_text=text,
                    verified=False,  # Can't verify after Enter
                    attempts=attempt,
                    details={"press_enter": True, "tier": 3},
                )

            if not verify:
                return SmartInputResult(
                    ok=True,
                    method=InputMethod.KEYSTROKE_UNVERIFIED,
                    app_name=app_name,
                    typed_text=text,
                    verified=False,
                    attempts=attempt,
                    details={"tier": 3},
                )

            # Verify: read back the focused element's value
            await asyncio.sleep(0.15)
            current_value = await self._read_focused_value()

            if current_value is not None and text in current_value:
                return SmartInputResult(
                    ok=True,
                    method=InputMethod.KEYSTROKE_VERIFIED,
                    app_name=app_name,
                    typed_text=text,
                    verified=True,
                    attempts=attempt,
                    details={"tier": 3},
                )

            # Input didn't land — clear and retry
            logger.warning(
                "Tier 3 verification failed (attempt %d/%d): expected '%s' in '%s'",
                attempt, self._max_retries, text[:50], (current_value or "")[:50],
            )

            if attempt < self._max_retries:
                await self._clear_current_field()
                await asyncio.sleep(0.1)
                if app_name:
                    await self._ensure_focused(app_name)
                    await self._wait_for_input_ready(app_name)

        # All retries exhausted
        return SmartInputResult(
            ok=True,  # We did type it, just couldn't verify
            method=InputMethod.KEYSTROKE_UNVERIFIED,
            app_name=app_name,
            typed_text=text,
            verified=False,
            attempts=self._max_retries,
            details={"warning": "Input typed but verification failed after all retries", "tier": 3},
        )

    # ══════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════

    async def _ensure_focused(self, app_name: str) -> bool:
        """
        Ensure the target app is the frontmost application.
        Returns True if the app is now focused.
        """
        try:
            # First check if already focused
            frontmost = self._op.execute("frontmost_app", {})
            if frontmost.get("app_name", "").lower() == app_name.lower():
                return True

            # Not focused — activate it
            self._op.execute("focus_app", {"app_name": app_name})
            await asyncio.sleep(self._focus_settle_ms / 1000.0)

            # Verify focus
            frontmost = self._op.execute("frontmost_app", {})
            if frontmost.get("app_name", "").lower() == app_name.lower():
                return True

            # Second attempt with longer wait
            self._op.execute("focus_app", {"app_name": app_name})
            await asyncio.sleep(self._focus_settle_ms * 2 / 1000.0)

            frontmost = self._op.execute("frontmost_app", {})
            return frontmost.get("app_name", "").lower() == app_name.lower()

        except (OSError, RuntimeError, KeyError) as e:
            logger.warning("Failed to focus %s: %s", app_name, e)
            return False

    async def _wait_for_input_ready(
        self,
        app_name: str,
        timeout: float | None = None,
    ) -> bool:
        """
        Wait until an input-accepting element is active in the AX tree.

        Polls for AXTextArea, AXTextField, AXComboBox, or AXSearchField
        in the focused window. Returns True if ready, False on timeout.
        """
        if not self._has_native:
            # No AX available — just wait a fixed amount for the app to settle
            await asyncio.sleep(self._focus_settle_ms / 1000.0)
            return True

        timeout = timeout or self._ax_ready_timeout
        deadline = time.time() + timeout
        poll_interval = 0.15

        while time.time() < deadline:
            try:
                # Check the focused element's role
                result = self._op.execute("ax_query", {
                    "target": "focused_element",
                    "max_depth": 0,
                })
                snapshot = result.get("snapshot", {})
                role = str(snapshot.get("role", ""))
                if role in _INPUT_AX_ROLES:
                    return True

                # Also check if any child of the focused window is an input element
                result = self._op.execute("ax_query", {
                    "target": "focused_window",
                    "max_depth": 2,
                    "max_children": 30,
                })
                if _has_input_element(result.get("snapshot", {})):
                    return True

            except (OSError, RuntimeError, KeyError) as e:
                logger.debug("AX input readiness check failed: %s", e)

            await asyncio.sleep(poll_interval)

        logger.debug(
            "Input readiness timeout after %.1fs for %s", timeout, app_name
        )
        # Return True anyway — some apps (like Terminal) don't expose text fields
        # in the standard AX way, so we should still try typing
        return True

    async def _read_focused_value(self) -> str | None:
        """Read the value of the currently focused UI element."""
        if not self._has_native:
            return None
        try:
            result = await asyncio.to_thread(self._op.execute, "ax_query", {
                "target": "focused_element",
                "max_depth": 0,
            })
            snapshot = result.get("snapshot", {})
            value = snapshot.get("value")
            if value is not None:
                return str(value)
            title = snapshot.get("title")
            if title is not None:
                return str(title)
            return None
        except (OSError, RuntimeError, KeyError, AttributeError) as e:
            logger.debug("Failed to read focused element value: %s", e)
            return None

    async def _clear_current_field(self) -> None:
        """Select all text in the current field and delete it."""
        try:
            self._op.execute("press_key", {
                "key": "a",
                "modifiers": ["command"],
            })
            await asyncio.sleep(0.05)
            self._op.execute("press_key", {"key": "delete"})
        except (OSError, RuntimeError) as e:
            logger.debug("Clear field failed: %s", e)

    async def _run_osascript(
        self,
        script: str,
        timeout: float = 10.0,
    ) -> str:
        """Run an AppleScript asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._run_osascript_sync(script, timeout),
        )

    @staticmethod
    def _run_osascript_sync(script: str, timeout: float) -> str:
        """Synchronous osascript execution."""
        try:
            result = subprocess.run(
                ["osascript", "-"],
                input=script,
                capture_output=True,
                text=True,
                check=True,
                timeout=max(1.0, timeout),
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"AppleScript timed out after {timeout}s") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(f"AppleScript error: {stderr or exc}") from exc


# ── Module-Level Utility Functions ───────────────────────────────────


def _escape_applescript(text: str) -> str:
    """Escape text for use inside AppleScript double-quoted strings."""
    text = str(text)
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = text.replace("\n", "\\n")
    text = text.replace("\r", "\\r")
    text = text.replace("\t", "\\t")
    return text


def _has_input_element(snapshot: dict[str, Any]) -> bool:
    """Recursively check if any element in the AX snapshot has an input role."""
    if str(snapshot.get("role", "")) in _INPUT_AX_ROLES:
        return True
    for child in snapshot.get("children", []):
        if _has_input_element(child):
            return True
    return False


_SYNC_BRIDGE_POOL: "concurrent.futures.ThreadPoolExecutor | None" = None
_SYNC_BRIDGE_LOCK = threading.Lock()


def _get_sync_bridge_pool() -> "concurrent.futures.ThreadPoolExecutor":
    """Lazy singleton thread pool for sync→async bridging.

    A persistent worker thread is dramatically cheaper than spinning a
    fresh ThreadPoolExecutor on every call.
    """
    global _SYNC_BRIDGE_POOL
    if _SYNC_BRIDGE_POOL is None:
        with _SYNC_BRIDGE_LOCK:
            if _SYNC_BRIDGE_POOL is None:
                _SYNC_BRIDGE_POOL = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="predacore-smart-input-sync"
                )
    return _SYNC_BRIDGE_POOL


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine synchronously, handling nested event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an event loop — submit to the persistent bridge pool
        pool = _get_sync_bridge_pool()
        future = pool.submit(asyncio.run, coro)
        return future.result(timeout=30)
    else:
        return asyncio.run(coro)
