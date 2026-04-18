"""
``predacore bootstrap`` — the one command between ``pip install`` and chat.

Idempotent, safe to rerun. It:

- Verifies the Rust kernel (``predacore_core``) imports.
- Pre-warms the BGE-small-en-v1.5 embedding model (first download is ~133 MB).
- Installs Playwright's Chromium binary if Playwright is available.
- Detects system binaries (Docker, sox, ffmpeg, adb, Chrome) and prints the
  exact per-OS command to install any missing ones.
- Warns if Docker is missing (only needed for the optional ``execute_code`` tool).
- Writes ``~/.predacore/.bootstrapped`` so subsequent runs skip the heavy work.

Output is a beautiful glass-panel status table consistent with the rest of
the PredaCore CLI UX. No arguments required.
"""
from __future__ import annotations

import asyncio
import logging
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# ── Marker file ──────────────────────────────────────────────────────

_HOME_DIR = Path.home() / ".predacore"
_MARKER_FILE = _HOME_DIR / ".bootstrapped"
_MARKER_VERSION = 1  # bump when bootstrap picks up a new required step


# ── Platform-aware install hints ─────────────────────────────────────

_INSTALL_HINTS: dict[str, dict[str, str]] = {
    "docker": {
        "darwin": "download Docker Desktop: https://docker.com/products/docker-desktop",
        "linux":  "curl -fsSL https://get.docker.com | sh",
        "win32":  "download Docker Desktop: https://docker.com/products/docker-desktop",
    },
    "sox": {
        "darwin": "brew install sox",
        "linux":  "sudo apt install sox  (or dnf install sox)",
        "win32":  "scoop install sox  (or choco install sox)",
    },
    "ffmpeg": {
        "darwin": "brew install ffmpeg",
        "linux":  "sudo apt install ffmpeg",
        "win32":  "winget install ffmpeg",
    },
    "adb": {
        "darwin": "brew install --cask android-platform-tools",
        "linux":  "sudo apt install android-tools-adb",
        "win32":  "scoop install adb",
    },
    "chrome": {
        "darwin": "download Chrome: https://google.com/chrome",
        "linux":  "sudo apt install google-chrome-stable  (or use Chromium)",
        "win32":  "download Chrome: https://google.com/chrome",
    },
}


# ── Status model ─────────────────────────────────────────────────────


@dataclass
class _StepResult:
    name: str
    ok: bool
    detail: str = ""
    hint: str = ""
    severity: str = "ok"  # "ok" | "warn" | "err"

    @property
    def icon(self) -> str:
        return {"ok": "◆", "warn": "◇", "err": "✗"}.get(self.severity, "◇")


@dataclass
class _BootstrapReport:
    steps: list[_StepResult] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def add(self, result: _StepResult) -> None:
        self.steps.append(result)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    @property
    def ok_count(self) -> int:
        return sum(1 for s in self.steps if s.ok)

    @property
    def warn_count(self) -> int:
        return sum(1 for s in self.steps if not s.ok and s.severity == "warn")

    @property
    def err_count(self) -> int:
        return sum(1 for s in self.steps if s.severity == "err")


# ── Individual bootstrap steps ───────────────────────────────────────


def _check_rust_kernel() -> _StepResult:
    """Verify ``predacore_core`` imports and reports a sensible embedding dim."""
    try:
        import predacore_core  # type: ignore[import-not-found]
    except ImportError as exc:
        return _StepResult(
            name="Rust kernel (predacore_core)",
            ok=False,
            severity="err",
            detail="not importable — required for embeddings + vector search",
            hint=(
                "maturin develop --release "
                "--manifest-path src/predacore_core_crate/Cargo.toml"
            ),
        )
    try:
        dim = int(predacore_core.embedding_dim())
    except (AttributeError, RuntimeError) as exc:
        return _StepResult(
            name="Rust kernel (predacore_core)",
            ok=False,
            severity="err",
            detail=f"imported but embedding_dim() failed: {exc}",
            hint="rebuild the Rust kernel (see INSTALL.md)",
        )
    return _StepResult(
        name="Rust kernel (predacore_core)",
        ok=True,
        detail=f"BGE-small-en-v1.5 · {dim}-dim embeddings",
    )


def _check_bge_model() -> _StepResult:
    """Trigger a BGE embed call to pre-warm the HF model cache (~133 MB)."""
    try:
        import predacore_core  # type: ignore[import-not-found]
    except ImportError:
        return _StepResult(
            name="BGE embedding model",
            ok=False,
            severity="err",
            detail="Rust kernel not loaded — cannot pre-warm",
        )
    try:
        t0 = time.time()
        _ = predacore_core.embed("PredaCore bootstrap warmup")
        elapsed = time.time() - t0
    except Exception as exc:
        return _StepResult(
            name="BGE embedding model",
            ok=False,
            severity="warn",
            detail=f"warmup failed: {exc}",
            hint="check network access to huggingface.co",
        )
    return _StepResult(
        name="BGE embedding model",
        ok=True,
        detail=f"cached + ready ({elapsed:.1f}s warmup)",
    )


def _check_playwright_chromium() -> _StepResult:
    """Install Playwright's Chromium binary if Playwright is importable."""
    try:
        import playwright  # noqa: F401
    except ImportError:
        return _StepResult(
            name="Playwright (browser_control)",
            ok=False,
            severity="warn",
            detail="not installed — browser_control tool unavailable",
            hint="pip install predacore[full]  (Playwright is in core now)",
        )

    # Try a headless launch to see if the Chromium binary is already there.
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import]
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            browser.close()
        return _StepResult(
            name="Playwright (browser_control)",
            ok=True,
            detail="Chromium binary cached and launchable",
        )
    except Exception:
        pass

    # Missing binary — install it. This downloads ~180 MB.
    cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return _StepResult(
            name="Playwright (browser_control)",
            ok=False,
            severity="warn",
            detail=f"Chromium install failed: {exc}",
            hint="python -m playwright install chromium",
        )
    if proc.returncode != 0:
        return _StepResult(
            name="Playwright (browser_control)",
            ok=False,
            severity="warn",
            detail="Chromium install exited non-zero",
            hint="python -m playwright install chromium --with-deps",
        )
    return _StepResult(
        name="Playwright (browser_control)",
        ok=True,
        detail="Chromium downloaded + installed",
    )


def _check_spacy_model() -> _StepResult:
    """If spaCy is installed, ensure ``en_core_web_sm`` is available."""
    try:
        import spacy  # type: ignore[import]
    except ImportError:
        return _StepResult(
            name="spaCy (goal parsing)",
            ok=True,  # optional — not an error if absent
            severity="ok",
            detail="not installed (optional — pip install predacore[ml] to enable)",
        )
    try:
        spacy.load("en_core_web_sm")
    except (OSError, ValueError):
        cmd = [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                return _StepResult(
                    name="spaCy (goal parsing)",
                    ok=False,
                    severity="warn",
                    detail="model download failed",
                    hint="python -m spacy download en_core_web_sm",
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return _StepResult(
                name="spaCy (goal parsing)",
                ok=False,
                severity="warn",
                detail="download command failed",
                hint="python -m spacy download en_core_web_sm",
            )
    return _StepResult(
        name="spaCy (goal parsing)",
        ok=True,
        detail="en_core_web_sm loaded",
    )


def _binary_check(name: str, description: str, severity: str = "warn") -> _StepResult:
    found = shutil.which(name)
    if found:
        return _StepResult(name=description, ok=True, detail=found)
    hint = _INSTALL_HINTS.get(name, {}).get(sys.platform, f"install {name}")
    return _StepResult(
        name=description,
        ok=False,
        severity=severity,
        detail=f"{name} not found in PATH",
        hint=hint,
    )


def _check_docker() -> _StepResult:
    result = _binary_check("docker", "Docker (sandboxed code exec)")
    if not result.ok:
        return result
    # Check the daemon is actually up — `docker info` exits non-zero otherwise.
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0:
            version = proc.stdout.strip() or "running"
            return _StepResult(
                name="Docker (sandboxed code exec)",
                ok=True,
                detail=f"daemon up · v{version}",
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return _StepResult(
        name="Docker (sandboxed code exec)",
        ok=False,
        severity="warn",
        detail="docker CLI present but daemon not running",
        hint="start Docker Desktop (or `systemctl start docker`)",
    )


def _pull_sandbox_image() -> _StepResult:
    """If Docker is up, pull the unified sandbox image."""
    if not shutil.which("docker"):
        return _StepResult(
            name="Sandbox image (predacore/sandbox)",
            ok=True,
            severity="ok",
            detail="skipped — Docker not installed",
        )
    try:
        proc = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=5,
        )
        if proc.returncode != 0:
            return _StepResult(
                name="Sandbox image (predacore/sandbox)",
                ok=True,
                severity="ok",
                detail="skipped — Docker daemon not running",
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return _StepResult(
            name="Sandbox image (predacore/sandbox)",
            ok=True,
            severity="ok",
            detail="skipped — Docker not reachable",
        )
    try:
        proc = subprocess.run(
            ["docker", "pull", "predacore/sandbox:latest"],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode == 0:
            return _StepResult(
                name="Sandbox image (predacore/sandbox)",
                ok=True,
                detail="pulled + cached",
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return _StepResult(
        name="Sandbox image (predacore/sandbox)",
        ok=True,
        severity="ok",
        detail="skipped — build locally if you need execute_code",
        hint="cd docker/sandbox && docker build -t predacore/sandbox:latest .",
    )


def _check_macos_permissions_note() -> _StepResult:
    """On macOS, remind about Accessibility + Screen Recording permissions."""
    if sys.platform != "darwin":
        return _StepResult(
            name="macOS permissions",
            ok=True,
            detail=f"skipped (platform={sys.platform})",
        )
    return _StepResult(
        name="macOS permissions",
        ok=True,
        severity="ok",
        detail=(
            "grant Accessibility + Screen Recording when first desktop "
            "action prompts (System Settings → Privacy & Security)"
        ),
    )


def _ensure_home_dir() -> None:
    _HOME_DIR.mkdir(parents=True, exist_ok=True)


def _write_marker(report: _BootstrapReport) -> None:
    """Record that bootstrap completed so later runs can skip."""
    import json
    _ensure_home_dir()
    _MARKER_FILE.write_text(
        json.dumps(
            {
                "version": _MARKER_VERSION,
                "timestamp": time.time(),
                "platform": sys.platform,
                "python": platform.python_version(),
                "errors": report.err_count,
                "warnings": report.warn_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def is_bootstrapped() -> bool:
    """Return True if a prior bootstrap completed with the current version."""
    if not _MARKER_FILE.exists():
        return False
    try:
        import json
        data = json.loads(_MARKER_FILE.read_text(encoding="utf-8"))
        return int(data.get("version", 0)) >= _MARKER_VERSION
    except (OSError, ValueError):
        return False


# ── Orchestrator ─────────────────────────────────────────────────────


def run_bootstrap(
    console=None,
    *,
    force: bool = False,
    step_printer: Callable[[_StepResult], None] | None = None,
) -> _BootstrapReport:
    """Run all bootstrap steps. Returns a report. Writes the marker on success.

    Args:
        console: Rich Console for pretty output. If None, logs via ``logging``.
        force: Re-run even if the marker already exists.
        step_printer: Optional callback invoked after each step.
    """
    _ensure_home_dir()
    if is_bootstrapped() and not force:
        logger.debug("Bootstrap already completed; skipping.")
        return _BootstrapReport(steps=[
            _StepResult(name="Bootstrap", ok=True, detail="already completed (use --force to rerun)")
        ])

    report = _BootstrapReport()

    steps: list[tuple[str, Callable[[], _StepResult]]] = [
        ("Rust kernel",        _check_rust_kernel),
        ("BGE model",          _check_bge_model),
        ("Playwright",         _check_playwright_chromium),
        ("spaCy",              _check_spacy_model),
        ("Docker",             _check_docker),
        ("Sandbox image",      _pull_sandbox_image),
        ("sox (voice)",        lambda: _binary_check("sox",    "sox (voice recording)",     "warn")),
        ("ffmpeg (audio)",     lambda: _binary_check("ffmpeg", "ffmpeg (audio conversion)", "warn")),
        ("adb (Android)",      lambda: _binary_check("adb",    "adb (Android control)",     "warn")),
        ("Chrome/Chromium",    lambda: _binary_check("google-chrome", "Chrome (browser_control)", "warn")
                               if shutil.which("google-chrome")
                               else _binary_check("chromium", "Chromium (browser_control)", "warn")),
        ("macOS permissions",  _check_macos_permissions_note),
    ]

    for label, fn in steps:
        if console is not None:
            with console.status(f"[glass.cyan]{label}[/glass.cyan]", spinner="dots"):
                result = fn()
        else:
            result = fn()
        report.add(result)
        if step_printer:
            step_printer(result)

    # Only write the marker if no hard errors — warnings are fine.
    if report.err_count == 0:
        _write_marker(report)

    return report
