from __future__ import annotations

import base64
import subprocess
from pathlib import Path

import pytest

from src.jarvis.config import JARVISConfig, LLMConfig, MemoryConfig, SecurityConfig
from src.jarvis.core import ToolExecutor
from src.jarvis.operators.desktop import DesktopControlError, MacDesktopOperator


def _mk_operator(monkeypatch: pytest.MonkeyPatch) -> MacDesktopOperator:
    op = MacDesktopOperator()
    monkeypatch.setattr(op, "_is_macos", True)
    op._native_service = None
    op._native_fallback_enabled = True
    return op


def test_open_app_uses_open_command(monkeypatch: pytest.MonkeyPatch):
    op = _mk_operator(monkeypatch)
    calls = []

    def fake_run(cmd, timeout=10, input_text=None):
        calls.append((cmd, timeout, input_text))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(op, "_run", fake_run)
    result = op.execute("open_app", {"app_name": "Safari", "timeout_seconds": 7})

    assert result["ok"] is True
    assert result["app_name"] == "Safari"
    assert calls[0][0] == ["open", "-a", "Safari"]
    assert calls[0][1] == 7.0


def test_open_app_respects_allowlist(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JARVIS_DESKTOP_ALLOWED_APPS", "Safari,Visual Studio Code")
    op = _mk_operator(monkeypatch)
    with pytest.raises(DesktopControlError, match="blocked"):
        op.execute("open_app", {"app_name": "Terminal"})


def test_press_key_builds_modifier_chord(monkeypatch: pytest.MonkeyPatch):
    op = _mk_operator(monkeypatch)
    scripts = []

    def fake_osascript(script, timeout=10):
        scripts.append((script, timeout))
        return ""

    monkeypatch.setattr(op, "_run_osascript", fake_osascript)
    result = op.execute(
        "press_key",
        {"key": "k", "modifiers": ["command", "shift"], "timeout_seconds": 9},
    )

    assert result["ok"] is True
    assert result["key"] == "k"
    assert "keystroke \"k\" using {command down, shift down}" in scripts[0][0]
    assert scripts[0][1] == 9.0


def test_key_press_alias_maps_to_press_key(monkeypatch: pytest.MonkeyPatch):
    op = _mk_operator(monkeypatch)
    scripts = []

    def fake_osascript(script, timeout=10):
        scripts.append((script, timeout))
        return ""

    monkeypatch.setattr(op, "_run_osascript", fake_osascript)
    result = op.execute("key_press", {"key": "enter"})

    assert result["ok"] is True
    assert result["action"] == "press_key"
    assert "key code 36" in scripts[0][0]


def test_macro_stops_on_error(monkeypatch: pytest.MonkeyPatch):
    op = _mk_operator(monkeypatch)
    result = op.execute(
        "run_macro",
        {
            "steps": [
                {"action": "sleep", "seconds": 0},
                {"action": "does_not_exist"},
            ],
            "stop_on_error": True,
        },
    )

    assert result["ok"] is True
    assert result["completed_steps"] == 1
    assert result["failed_step"] == 2
    assert result["results"][1]["action"] == "does_not_exist"


def test_screenshot_can_inline_base64(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    op = _mk_operator(monkeypatch)

    def fake_run(cmd, timeout=10, input_text=None):
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(op, "_run", fake_run)
    out_file = tmp_path / "shot.png"
    result = op.execute(
        "screenshot",
        {"path": str(out_file), "include_base64": True, "timeout_seconds": 4},
    )

    assert result["ok"] is True
    assert result["path"] == str(out_file)
    assert result["image_b64"] == base64.b64encode(b"png").decode("ascii")


def test_health_check_reports_ready_components(monkeypatch: pytest.MonkeyPatch):
    op = _mk_operator(monkeypatch)
    monkeypatch.setattr(op, "_run_osascript", lambda script, timeout=3: "ok")
    op._pyautogui = object()

    result = op.execute("health_check", {})

    assert result["ok"] is True
    assert result["osascript"]["ok"] is True
    assert result["pyautogui"]["ok"] is True
    assert result["hints"] == []


def test_native_backend_is_preferred(monkeypatch: pytest.MonkeyPatch):
    op = _mk_operator(monkeypatch)

    class StubNative:
        available = True

        def execute(self, action: str, params: dict):
            return {"backend": "native", "action": action}

        def health_check(self):
            return {"backend": "pyobjc_native", "accessibility_trusted": True}

    op._native_service = StubNative()  # type: ignore[assignment]
    result = op.execute("frontmost_app", {})
    assert result["ok"] is True
    assert result["backend"] == "native"


def test_native_only_action_requires_native_backend(monkeypatch: pytest.MonkeyPatch):
    op = _mk_operator(monkeypatch)
    op._native_service = None
    with pytest.raises(DesktopControlError, match="native desktop backend required"):
        op.execute("ax_query", {})


@pytest.fixture
def config(tmp_path: Path) -> JARVISConfig:
    home = tmp_path / ".prometheus"
    (home / "sessions").mkdir(parents=True)
    (home / "skills").mkdir(parents=True)
    (home / "logs").mkdir(parents=True)
    (home / "memory").mkdir(parents=True)
    return JARVISConfig(
        home_dir=str(home),
        sessions_dir=str(home / "sessions"),
        skills_dir=str(home / "skills"),
        logs_dir=str(home / "logs"),
        llm=LLMConfig(provider="gemini-cli"),
        security=SecurityConfig(trust_level="normal"),
        memory=MemoryConfig(persistence_dir=str(home / "memory")),
    )


@pytest.mark.asyncio
async def test_tool_executor_desktop_control_dispatch(config: JARVISConfig):
    executor = ToolExecutor(config)

    class StubDesktop:
        def execute(self, action: str, params: dict):
            return {"action_echo": action, "received": params.get("marker")}

    executor._desktop_operator = StubDesktop()  # type: ignore[assignment]
    out = await executor.execute(
        "desktop_control",
        {"action": "frontmost_app", "marker": "ok"},
    )

    assert '"action_echo": "frontmost_app"' in out
    assert '"received": "ok"' in out


@pytest.mark.asyncio
async def test_tool_executor_desktop_control_disabled(config: JARVISConfig):
    executor = ToolExecutor(config)
    executor._desktop_operator = None

    out = await executor.execute("desktop_control", {"action": "frontmost_app"})
    assert out == "[Desktop control is not available — Suggestion: Check that Desktop control is properly configured and initialized]"
