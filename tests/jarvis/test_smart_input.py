"""
Tests for JARVIS Smart Input Engine.

Covers all three input tiers:
  Tier 1: AppleScript do script (Terminal/iTerm)
  Tier 2: AX set_value (direct value injection)
  Tier 3: Verified keystroke (focus → wait → type → verify → retry)
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.jarvis.operators.smart_input import (
    InputMethod,
    SmartInputEngine,
    SmartInputResult,
    _escape_applescript,
)
from src.jarvis.operators.desktop import DesktopControlError, MacDesktopOperator


# ── Fixtures ─────────────────────────────────────────────────────────


class FakeDesktopOperator:
    """Mock desktop operator for testing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._frontmost_app = "Terminal"
        self._focused_value = ""
        self._ax_query_result: dict[str, Any] = {
            "snapshot": {"role": "AXTextArea", "value": ""},
        }
        self._native_service = True  # Satisfy _has_native check

    def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((action, params))

        if action == "frontmost_app":
            return {"ok": True, "app_name": self._frontmost_app}
        if action == "focus_app":
            self._frontmost_app = params.get("app_name", "")
            return {"ok": True, "app_name": self._frontmost_app}
        if action == "type_text":
            self._focused_value += params.get("text", "")
            return {"ok": True, "typed_chars": len(params.get("text", ""))}
        if action == "press_key":
            return {"ok": True, "key": params.get("key")}
        if action == "ax_query":
            return {"ok": True, **self._ax_query_result}
        if action == "ax_set_value":
            return {
                "ok": True,
                "matched": {"role": "AXTextField", "title": "Search"},
            }
        if action == "ax_click":
            return {"ok": True, "matched": {"role": "AXButton"}}
        return {"ok": True}


@pytest.fixture
def fake_op() -> FakeDesktopOperator:
    return FakeDesktopOperator()


@pytest.fixture
def engine(fake_op: FakeDesktopOperator) -> SmartInputEngine:
    return SmartInputEngine(
        desktop_operator=fake_op,
        max_retries=2,
        verify_timeout=0.5,
        focus_settle_ms=50,
        ax_ready_timeout=0.5,
    )


# ── Unit Tests: Escape Function ──────────────────────────────────────


def test_escape_applescript_basic():
    assert _escape_applescript('hello "world"') == 'hello \\"world\\"'


def test_escape_applescript_backslash():
    assert _escape_applescript("path\\to\\file") == "path\\\\to\\\\file"


def test_escape_applescript_newlines():
    assert _escape_applescript("line1\nline2\r") == "line1\\nline2\\r"


def test_escape_applescript_tabs():
    assert _escape_applescript("col1\tcol2") == "col1\\tcol2"


def test_escape_applescript_combined():
    result = _escape_applescript('say "hi"\nnext\\line')
    assert result == 'say \\"hi\\"\\nnext\\\\line'


# ── Unit Tests: SmartInputEngine static methods ──────────────────────


def test_get_best_method_terminal():
    result = SmartInputEngine.get_best_method("Terminal", action="command")
    assert "Tier 1" in result
    assert "AppleScript" in result


def test_get_best_method_iterm():
    result = SmartInputEngine.get_best_method("iTerm2", action="command")
    assert "Tier 1" in result


def test_get_best_method_unknown_app():
    result = SmartInputEngine.get_best_method("RandomApp", action="type")
    assert "Tier 3" in result
    assert "keystroke" in result.lower()


def test_get_best_method_notes():
    result = SmartInputEngine.get_best_method("Notes", action="note")
    assert "Tier 1" in result
    assert "make new" in result.lower()


def test_supported_apps_returns_categories():
    apps = SmartInputEngine.supported_apps()
    assert "terminal_apps" in apps
    assert "content_creation" in apps
    assert "Terminal" in apps["terminal_apps"]


# ── Unit Tests: SmartInputResult ─────────────────────────────────────


def test_result_to_dict_basic():
    r = SmartInputResult(
        ok=True,
        method=InputMethod.APPLESCRIPT_DO_SCRIPT,
        app_name="Terminal",
        typed_text="ls -la",
        verified=True,
    )
    d = r.to_dict()
    assert d["ok"] is True
    assert d["method"] == "applescript_do_script"
    assert d["verified"] is True
    assert d["typed_chars"] == 6
    assert d["app_name"] == "Terminal"
    assert "error" not in d


def test_result_to_dict_with_error():
    r = SmartInputResult(
        ok=False,
        method=InputMethod.KEYSTROKE_UNVERIFIED,
        error="focus lost",
    )
    d = r.to_dict()
    assert d["ok"] is False
    assert d["error"] == "focus lost"
    assert "app_name" not in d
    assert "typed_chars" not in d


# ── Async Tests: Tier 1 — AppleScript do script ─────────────────────


@pytest.mark.asyncio
async def test_smart_type_terminal_uses_applescript(engine: SmartInputEngine):
    """When app is Terminal and press_enter=True, should use AppleScript do script."""
    with patch.object(engine, "_run_osascript", new_callable=AsyncMock) as mock_osa:
        mock_osa.return_value = ""
        result = await engine.smart_type(
            text="git status",
            app_name="Terminal",
            press_enter=True,
        )
        assert result.ok is True
        assert result.method == InputMethod.APPLESCRIPT_DO_SCRIPT
        assert result.verified is True
        assert "git status" in result.typed_text
        mock_osa.assert_called_once()
        script = mock_osa.call_args[0][0]
        assert "git status" in script
        assert "Terminal" in script


@pytest.mark.asyncio
async def test_smart_run_in_terminal_uses_applescript(engine: SmartInputEngine):
    """smart_run_in_terminal should use AppleScript do script."""
    with patch.object(engine, "_run_osascript", new_callable=AsyncMock) as mock_osa:
        mock_osa.return_value = ""
        result = await engine.smart_run_in_terminal(
            command="python main.py",
            app_name="Terminal",
        )
        assert result.ok is True
        assert result.method == InputMethod.APPLESCRIPT_DO_SCRIPT
        assert result.typed_text == "python main.py"
        mock_osa.assert_called_once()


@pytest.mark.asyncio
async def test_smart_run_iterm_uses_write_text(engine: SmartInputEngine):
    """iTerm2 should use its own scripting model (write text)."""
    with patch.object(engine, "_run_osascript", new_callable=AsyncMock) as mock_osa:
        mock_osa.return_value = ""
        result = await engine.smart_run_in_terminal(
            command="npm start",
            app_name="iTerm2",
        )
        assert result.ok is True
        assert result.method == InputMethod.APPLESCRIPT_DO_SCRIPT
        script = mock_osa.call_args[0][0]
        assert "write text" in script
        assert "iTerm2" in script


@pytest.mark.asyncio
async def test_applescript_do_script_new_tab_flag(engine: SmartInputEngine):
    """new_tab=False should use 'in front window' syntax."""
    with patch.object(engine, "_run_osascript", new_callable=AsyncMock) as mock_osa:
        mock_osa.return_value = ""
        result = await engine.smart_run_in_terminal(
            command="ls",
            app_name="Terminal",
            new_tab=False,
        )
        assert result.ok is True
        script = mock_osa.call_args[0][0]
        assert "in front window" in script


@pytest.mark.asyncio
async def test_applescript_escapes_quotes(engine: SmartInputEngine):
    """Commands with quotes should be properly escaped."""
    with patch.object(engine, "_run_osascript", new_callable=AsyncMock) as mock_osa:
        mock_osa.return_value = ""
        await engine.smart_run_in_terminal(
            command='echo "hello world"',
            app_name="Terminal",
        )
        script = mock_osa.call_args[0][0]
        # Should have escaped quotes
        assert '\\"hello world\\"' in script


# ── Async Tests: Tier 2 — AX set_value ──────────────────────────────


@pytest.mark.asyncio
async def test_smart_type_with_field_label_uses_ax(
    engine: SmartInputEngine, fake_op: FakeDesktopOperator
):
    """When field_label is specified and AX works, should use ax_set_value."""
    result = await engine.smart_type(
        text="python docs",
        app_name="Safari",
        field_label="Search",
    )
    assert result.ok is True
    assert result.method == InputMethod.AX_SET_VALUE
    assert result.verified is True

    # Check that ax_set_value was called
    ax_calls = [c for c in fake_op.calls if c[0] == "ax_set_value"]
    assert len(ax_calls) == 1
    assert ax_calls[0][1]["value"] == "python docs"


@pytest.mark.asyncio
async def test_ax_set_value_falls_through_on_failure(
    engine: SmartInputEngine, fake_op: FakeDesktopOperator
):
    """If AX set_value fails, should fall through to Tier 3 keystroke."""
    # Make ax_set_value raise
    original_execute = fake_op.execute

    def patched_execute(action: str, params: dict) -> dict:
        if action == "ax_set_value":
            raise RuntimeError("AX not available")
        return original_execute(action, params)

    fake_op.execute = patched_execute
    fake_op._focused_value = ""

    result = await engine.smart_type(
        text="hello",
        app_name="Safari",
        field_label="Search",
    )
    # Should have fallen through to keystroke
    assert result.ok is True
    assert result.method in (
        InputMethod.KEYSTROKE_VERIFIED,
        InputMethod.KEYSTROKE_UNVERIFIED,
    )


# ── Async Tests: Tier 3 — Verified Keystroke ─────────────────────────


@pytest.mark.asyncio
async def test_verified_keystroke_types_and_verifies(
    engine: SmartInputEngine, fake_op: FakeDesktopOperator
):
    """Tier 3: should type text and verify it landed."""
    fake_op._frontmost_app = "TextEdit"

    # The AX query for focused element returns the typed value
    original_execute = fake_op.execute
    typed_text = ""

    def patched_execute(action: str, params: dict) -> dict:
        nonlocal typed_text
        if action == "type_text":
            typed_text += params.get("text", "")
            return {"ok": True, "typed_chars": len(params.get("text", ""))}
        if action == "ax_query" and params.get("target") == "focused_element":
            return {
                "ok": True,
                "snapshot": {"role": "AXTextArea", "value": typed_text},
            }
        return original_execute(action, params)

    fake_op.execute = patched_execute

    result = await engine.smart_type(
        text="verified input",
        app_name="TextEdit",
        verify=True,
    )
    assert result.ok is True
    assert result.verified is True
    assert result.attempts == 1


@pytest.mark.asyncio
async def test_keystroke_retries_on_verification_failure(
    engine: SmartInputEngine, fake_op: FakeDesktopOperator
):
    """If verification fails, should clear and retry."""
    fake_op._frontmost_app = "TextEdit"
    attempt_count = 0

    original_execute = fake_op.execute

    def patched_execute(action: str, params: dict) -> dict:
        nonlocal attempt_count
        if action == "type_text":
            attempt_count += 1
            return {"ok": True, "typed_chars": len(params.get("text", ""))}
        if action == "ax_query" and params.get("target") == "focused_element":
            # Always return empty — verification always fails
            return {
                "ok": True,
                "snapshot": {"role": "AXTextArea", "value": ""},
            }
        return original_execute(action, params)

    fake_op.execute = patched_execute

    result = await engine.smart_type(
        text="will not verify",
        app_name="TextEdit",
        verify=True,
    )
    # Should have retried max_retries times (2)
    assert attempt_count == 2
    assert result.ok is True  # Still ok, just unverified
    assert result.verified is False
    assert result.method == InputMethod.KEYSTROKE_UNVERIFIED


@pytest.mark.asyncio
async def test_press_enter_skips_verification(
    engine: SmartInputEngine, fake_op: FakeDesktopOperator
):
    """When press_enter=True (non-terminal app), can't verify after Enter."""
    fake_op._frontmost_app = "Spotlight"

    result = await engine.smart_type(
        text="query",
        app_name="Spotlight",
        press_enter=True,
    )
    assert result.ok is True
    assert result.verified is False

    # Should have called press_key with return
    key_calls = [c for c in fake_op.calls if c[0] == "press_key"]
    assert len(key_calls) == 1
    assert key_calls[0][1]["key"] == "return"


@pytest.mark.asyncio
async def test_focus_verified_before_typing(
    engine: SmartInputEngine, fake_op: FakeDesktopOperator
):
    """Should verify focus on the target app before typing."""
    fake_op._frontmost_app = "Finder"  # Different app initially

    result = await engine.smart_type(
        text="test",
        app_name="TextEdit",
        verify=False,
    )
    assert result.ok is True

    # Should have called focus_app to switch to TextEdit
    focus_calls = [c for c in fake_op.calls if c[0] == "focus_app"]
    assert len(focus_calls) >= 1
    assert focus_calls[0][1]["app_name"] == "TextEdit"


@pytest.mark.asyncio
async def test_focus_failure_returns_error(
    engine: SmartInputEngine, fake_op: FakeDesktopOperator
):
    """If we can't focus the target app, should return error."""
    original_execute = fake_op.execute

    def patched_execute(action: str, params: dict) -> dict:
        if action == "focus_app":
            return {"ok": True}  # Claims success but...
        if action == "frontmost_app":
            return {"ok": True, "app_name": "SomeOtherApp"}  # Never actually focused
        return original_execute(action, params)

    fake_op.execute = patched_execute

    result = await engine.smart_type(
        text="test",
        app_name="TextEdit",
        verify=False,
    )
    assert result.ok is False
    assert "focus" in result.error.lower()


# ── Async Tests: smart_create_note ───────────────────────────────────


@pytest.mark.asyncio
async def test_smart_create_note_uses_applescript(engine: SmartInputEngine):
    """smart_create_note should use AppleScript for Apple Notes."""
    with patch.object(engine, "_run_osascript", new_callable=AsyncMock) as mock_osa:
        mock_osa.return_value = ""
        result = await engine.smart_create_note(
            title="Test Note",
            body="Hello World",
            app_name="Notes",
        )
        assert result.ok is True
        assert result.method in (InputMethod.APPLESCRIPT_CREATE, InputMethod.APPLESCRIPT_KEYSTROKE)
        mock_osa.assert_called_once()
        script = mock_osa.call_args[0][0]
        assert "Notes" in script
        assert "Test Note" in script
        assert "Hello World" in script


@pytest.mark.asyncio
async def test_smart_create_note_unsupported_app(engine: SmartInputEngine):
    """Unsupported apps should return error."""
    result = await engine.smart_create_note(
        title="Test",
        body="Content",
        app_name="RandomApp",
    )
    assert result.ok is False
    assert "no applescript" in result.error.lower() or "not supported" in result.error.lower() or "no" in result.error.lower()


# ── Async Tests: Internal Helpers ────────────────────────────────────


@pytest.mark.asyncio
async def test_wait_for_input_ready_finds_text_area(
    engine: SmartInputEngine, fake_op: FakeDesktopOperator
):
    """Should detect AXTextArea as input-ready."""
    fake_op._ax_query_result = {
        "snapshot": {"role": "AXTextArea", "value": ""},
    }
    ready = await engine._wait_for_input_ready("Terminal", timeout=0.5)
    assert ready is True


@pytest.mark.asyncio
async def test_wait_for_input_ready_timeout_returns_true(
    engine: SmartInputEngine, fake_op: FakeDesktopOperator
):
    """On timeout, should still return True (some apps don't expose standard fields)."""
    fake_op._ax_query_result = {
        "snapshot": {"role": "AXGroup"},  # Not an input element
    }
    ready = await engine._wait_for_input_ready("Terminal", timeout=0.3)
    assert ready is True  # Returns True even on timeout (permissive)


@pytest.mark.asyncio
async def test_read_focused_value(
    engine: SmartInputEngine, fake_op: FakeDesktopOperator
):
    """Should read value from focused element."""
    fake_op._ax_query_result = {
        "snapshot": {"role": "AXTextField", "value": "hello world"},
    }
    value = await engine._read_focused_value()
    assert value == "hello world"


@pytest.mark.asyncio
async def test_read_focused_value_returns_title_if_no_value(
    engine: SmartInputEngine, fake_op: FakeDesktopOperator
):
    """Should fall back to title if value is None."""
    fake_op._ax_query_result = {
        "snapshot": {"role": "AXButton", "title": "Save"},
    }
    value = await engine._read_focused_value()
    assert value == "Save"


@pytest.mark.asyncio
async def test_clear_current_field(
    engine: SmartInputEngine, fake_op: FakeDesktopOperator
):
    """Should send Cmd+A then Delete."""
    await engine._clear_current_field()
    key_calls = [c for c in fake_op.calls if c[0] == "press_key"]
    assert len(key_calls) == 2
    assert key_calls[0][1]["key"] == "a"
    assert key_calls[0][1]["modifiers"] == ["command"]
    assert key_calls[1][1]["key"] == "delete"


# ── Integration Test: Desktop Operator Smart Actions ─────────────────


def _mk_operator(monkeypatch: pytest.MonkeyPatch) -> MacDesktopOperator:
    op = MacDesktopOperator()
    monkeypatch.setattr(op, "_is_macos", True)
    op._native_service = None
    op._native_fallback_enabled = True
    return op


def test_desktop_operator_has_smart_input(monkeypatch: pytest.MonkeyPatch):
    """Desktop operator should expose smart_input property."""
    op = _mk_operator(monkeypatch)
    assert hasattr(op, "smart_input")


def test_health_check_includes_smart_input(monkeypatch: pytest.MonkeyPatch):
    """Health check should report smart input status."""
    op = _mk_operator(monkeypatch)
    monkeypatch.setattr(op, "_run_osascript", lambda script, timeout=3: "ok")
    op._pyautogui = object()

    result = op.execute("health_check", {})
    assert "smart_input" in result
    assert "enabled" in result["smart_input"]


def test_smart_type_action_requires_text(monkeypatch: pytest.MonkeyPatch):
    """smart_type without text should raise DesktopControlError."""
    op = _mk_operator(monkeypatch)
    if op._smart_input is None:
        pytest.skip("SmartInputEngine not available")

    with pytest.raises(DesktopControlError, match="requires text"):
        op.execute("smart_type", {})


def test_smart_run_command_requires_command(monkeypatch: pytest.MonkeyPatch):
    """smart_run_command without command should raise DesktopControlError."""
    op = _mk_operator(monkeypatch)
    if op._smart_input is None:
        pytest.skip("SmartInputEngine not available")

    with pytest.raises(DesktopControlError, match="requires command"):
        op.execute("smart_run_command", {})


def test_smart_create_note_requires_body(monkeypatch: pytest.MonkeyPatch):
    """smart_create_note without body should raise DesktopControlError."""
    op = _mk_operator(monkeypatch)
    if op._smart_input is None:
        pytest.skip("SmartInputEngine not available")

    with pytest.raises(DesktopControlError, match="requires body"):
        op.execute("smart_create_note", {"title": "Test"})


# ── Edge Cases ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_text_handled_gracefully(engine: SmartInputEngine):
    """Empty text should be handled without crash."""
    with patch.object(engine, "_run_osascript", new_callable=AsyncMock) as mock_osa:
        mock_osa.return_value = ""
        result = await engine.smart_type(
            text="",
            app_name="Terminal",
            press_enter=True,
        )
        # AppleScript do script with empty string is valid
        assert result.ok is True


@pytest.mark.asyncio
async def test_very_long_text(engine: SmartInputEngine, fake_op: FakeDesktopOperator):
    """Very long text should still work via keystroke."""
    long_text = "x" * 10000
    fake_op._frontmost_app = "TextEdit"

    result = await engine.smart_type(
        text=long_text,
        app_name="TextEdit",
        verify=False,
    )
    assert result.ok is True
    assert len(result.typed_text) == 10000


@pytest.mark.asyncio
async def test_special_characters_in_command(engine: SmartInputEngine):
    """Special characters should be escaped in AppleScript."""
    with patch.object(engine, "_run_osascript", new_callable=AsyncMock) as mock_osa:
        mock_osa.return_value = ""
        await engine.smart_run_in_terminal(
            command="echo 'test' && cd ~/dir\twith\ttabs",
        )
        script = mock_osa.call_args[0][0]
        # Should have escaped tabs
        assert "\\t" in script


# ── Sync wrappers ────────────────────────────────────────────────────


def test_sync_wrappers_exist(engine: SmartInputEngine):
    """Sync wrappers should be available on the engine."""
    assert hasattr(engine, "smart_type_sync")
    assert hasattr(engine, "smart_run_in_terminal_sync")
    assert hasattr(engine, "smart_create_note_sync")
    assert callable(engine.smart_type_sync)
    assert callable(engine.smart_run_in_terminal_sync)
    assert callable(engine.smart_create_note_sync)
