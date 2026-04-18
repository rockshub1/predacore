"""
PredaCore CLI — Terminal interface for the PredaCore agent framework.

Commands:
    predacore chat [--session ID]   — Interactive chat in terminal
    predacore start [--daemon]      — Start the agent (foreground or daemon)
    predacore setup                 — Guided onboarding wizard
    predacore status                — Show agent status
    predacore sessions              — List recent sessions
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import select
import subprocess as _subprocess
import sys
import termios
import time
import tty
import uuid
from pathlib import Path
from typing import Any, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

try:
    from predacore._vendor.common.logging_config import setup_logging
except ImportError:  # pragma: no cover - fallback
    def setup_logging(level: str = "INFO", **kwargs) -> None:
        """Minimal fallback when common.logging_config is unavailable."""
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

logger = logging.getLogger(__name__)

# ── Rich Console Setup ────────────────────────────────────────────────

_THEME = Theme(
    {
        # ── Legacy tokens (kept for backwards compatibility) ──
        "banner": "bold bright_cyan",
        "banner.sub": "dim cyan",
        "user": "bold bright_green",
        "agent": "bold bright_cyan",
        "tool.name": "bold bright_yellow",
        "tool.result": "dim white",
        "tool.icon": "bright_yellow",
        "stat": "dim",
        "stat.value": "bold bright_white",
        "error": "bold bright_red",
        "warning": "bright_yellow",
        "success": "bold bright_green",
        "muted": "dim",
        "heading": "bold bright_cyan",
        "key": "dim cyan",
        "provider.ag": "bold bright_magenta",
        "accent": "bold bright_blue",
        "select.active": "bold bright_cyan on grey23",
        "select.inactive": "dim white",

        # ── Glassmorphism palette ──
        # Soft pastels inspired by iOS-style translucent panels.
        # Hex where Rich supports truecolor; named fallbacks elsewhere.
        "glass.cyan": "#7DD3FC",
        "glass.violet": "#C4B5FD",
        "glass.pink": "#F9A8D4",
        "glass.mint": "#86EFAC",
        "glass.amber": "#FDE68A",
        "glass.rose": "#FCA5A5",
        "glass.border": "grey35",
        "glass.border.soft": "grey23",
        "glass.border.accent": "#7DD3FC",
        "glass.surface": "grey11",
        "glass.text": "grey85",
        "glass.text.muted": "grey58",
        "glass.text.dim": "grey42",
        "glass.hint": "grey62",
        # Step badges — pink/violet/cyan by position in flow
        "step.badge.1": "bold white on #EC4899",  # pink
        "step.badge.2": "bold white on #A855F7",  # violet
        "step.badge.3": "bold white on #6366F1",  # indigo
        "step.badge.4": "bold white on #0EA5E9",  # sky
        "step.badge.5": "bold white on #10B981",  # emerald
        "step.title": "bold #E9D5FF",
        # Status indicators
        "status.ok": "bold #86EFAC",
        "status.warn": "bold #FDE68A",
        "status.err": "bold #FCA5A5",
        "status.skip": "dim grey58",
        "status.pending": "dim #C4B5FD",
    }
)

console = Console(theme=_THEME)


# ── Glassmorphism UI Helpers ──────────────────────────────────────────


def _glass_panel(
    content,
    title: str | None = None,
    subtitle: str | None = None,
    border_style: str = "glass.border",
    padding: tuple[int, int] = (1, 3),
    width: int | None = None,
) -> Panel:
    """Rounded-border panel matching the glassmorphism aesthetic.

    Uses box.ROUNDED for soft corners, generous padding for breathing room,
    and a theme-coordinated border color. Titles render on the top border.
    """
    return Panel(
        content,
        title=title,
        subtitle=subtitle,
        border_style=border_style,
        box=box.ROUNDED,
        padding=padding,
        width=width,
        title_align="left",
        subtitle_align="right",
    )


def _step_rule(step: int, total: int, title: str) -> Rule:
    """A horizontal section divider with a step badge + title centered inline.

    Rendered as: ─── [ 1/5 ] System check ─────────────────────
    The step badge uses a per-step color (pink → violet → indigo → sky → emerald).
    """
    badge_style = f"step.badge.{min(max(step, 1), 5)}"
    label = Text.from_markup(
        f"[{badge_style}] {step}/{total} [/{badge_style}]  [step.title]{title}[/step.title]"
    )
    return Rule(label, style="glass.border.soft")


def _gradient_text(text: str, colors: list[str] | None = None) -> Text:
    """Render text with a soft color gradient across characters.

    Default palette: pink → violet → indigo → sky → cyan (classic glass).
    Not a true gradient (terminals don't interpolate), but the chunked
    color runs create the visual effect.
    """
    if colors is None:
        colors = ["#F9A8D4", "#DDA0DD", "#C4B5FD", "#A5B4FC", "#7DD3FC"]
    result = Text()
    n = len(text)
    if n == 0:
        return result
    chunk = max(1, n // len(colors))
    for i, ch in enumerate(text):
        idx = min(i // chunk, len(colors) - 1)
        result.append(ch, style=colors[idx])
    return result


def _status_pill(label: str, kind: str = "ok") -> Text:
    """A compact status pill: [ READY ] / [ SKIP ] / [ WARN ] / [ ERR ]."""
    label_map = {
        "ok":      ("status.ok",      "READY"),
        "warn":    ("status.warn",    "SETUP"),
        "err":     ("status.err",     "ERROR"),
        "skip":    ("status.skip",    "SKIP "),
        "pending": ("status.pending", "• • •"),
    }
    style, default_label = label_map.get(kind, ("status.ok", label))
    text = label or default_label
    return Text.assemble(
        ("[ ", "glass.text.dim"),
        (text.upper(), style),
        (" ]", "glass.text.dim"),
    )


# ── Generation Controller ──────────────────────────────────────────


# ── Interactive Arrow-Key Selector ─────────────────────────────────


def _interactive_select(
    title: str,
    options: list[tuple[str, str, str]],  # (label, sublabel, value)
    current_value: str = "",
    max_visible: int = 12,
) -> str | None:
    """
    Interactive arrow-key selector with scrolling, type-to-filter, and smooth visuals.
    Returns the selected value or None if cancelled (Esc/q).

    Uses os.read(fd) for raw byte reading — avoids conflicts with prompt_toolkit's
    stdin buffering layer.
    """
    if not options:
        return None

    all_options = list(options)
    filtered = list(all_options)
    filter_text = ""

    # Find initial cursor position
    cursor = 0
    for i, (_, _, val) in enumerate(filtered):
        if val == current_value:
            cursor = i
            break

    fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
    except termios.error:
        return None

    # ANSI
    C = "\033[96m"       # cyan
    G = "\033[92m"       # green
    Y = "\033[93m"       # yellow
    D = "\033[2m"        # dim
    B = "\033[1m"        # bold
    R = "\033[0m"        # reset
    BG = "\033[48;5;236m"  # dark bg
    W = "\033[97m"       # white

    scroll_offset = 0
    rendered_lines = 0

    def _read_byte() -> bytes:
        """Read exactly one byte from stdin fd (bypasses Python buffering)."""
        return os.read(fd, 1)

    def _has_input(timeout: float = 0.0) -> bool:
        """Check if stdin fd has data ready."""
        r, _, _ = select.select([fd], [], [], timeout)
        return bool(r)

    def _render():
        nonlocal rendered_lines, scroll_offset
        total = len(filtered)
        visible = min(total, max_visible)

        if cursor < scroll_offset:
            scroll_offset = cursor
        elif cursor >= scroll_offset + visible:
            scroll_offset = cursor - visible + 1

        out: list[str] = []
        out.append("\033[?25l")  # hide cursor
        out.append(f"  {C}{B}\u250c\u2500 {title} \u2500\u2500\u2500{R}")

        if filter_text:
            out.append(f"  {C}\u2502{R}  {Y}Filter:{R} {filter_text}\u2581")
        out.append(f"  {C}\u2502{R}")

        if scroll_offset > 0:
            out.append(f"  {C}\u2502{R}  {D}\u25b2 {scroll_offset} more above{R}")

        for i in range(scroll_offset, min(scroll_offset + visible, total)):
            label, sublabel, val = filtered[i]
            is_cur = val == current_value
            if i == cursor:
                tag = f" {D}\u2713{R}" if is_cur else ""
                ln = f"  {C}\u2502{R} {G}{B}\u276f{R} {BG}{W}{B} {label:<28}{R}{tag}"
                if sublabel:
                    ln += f"  {D}{sublabel}{R}"
            else:
                tag = f" {D}\u2713{R}" if is_cur else "  "
                ln = f"  {C}\u2502{R}   {D} {label:<28}{R}{tag}"
                if sublabel:
                    ln += f"  {D}{sublabel}{R}"
            out.append(ln)

        remaining = total - scroll_offset - visible
        if remaining > 0:
            out.append(f"  {C}\u2502{R}  {D}\u25bc {remaining} more below{R}")

        out.append(f"  {C}\u2502{R}")
        cnt = f"  {D}{total} options{R}" if filter_text else ""
        out.append(
            f"  {C}\u2514\u2500{R} {D}\u2191\u2193 move \u00b7 Enter select"
            f" \u00b7 Esc cancel \u00b7 type to filter{R}{cnt}"
        )

        sys.stdout.write("\n".join(out) + "\n")
        sys.stdout.flush()
        rendered_lines = len(out)

    def _wipe():
        for _ in range(rendered_lines):
            sys.stdout.write("\033[A\033[2K")
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()

    def _apply_filter():
        nonlocal filtered, cursor, scroll_offset
        if not filter_text:
            filtered = list(all_options)
        else:
            ft = filter_text.lower()
            filtered = [
                (l, s, v) for l, s, v in all_options
                if ft in l.lower() or ft in s.lower() or ft in v.lower()
            ]
        cursor = 0
        scroll_offset = 0

    try:
        # cbreak mode: keystroke-at-a-time input but output still works normally.
        # (setraw would break output rendering — no newlines, no echo)
        tty.setcbreak(fd)

        # Drain leftover bytes
        while _has_input(0.1):
            os.read(fd, 64)
        time.sleep(0.05)
        while _has_input(0.05):
            os.read(fd, 64)

        _render()

        while True:
            b = _read_byte()
            if not b:
                continue
            ch = b[0]  # integer byte value

            if ch == 0x1b:  # ESC — could be arrow key or plain Esc
                # Try to read the rest of the escape sequence
                if _has_input(0.3):
                    b2 = _read_byte()
                    prefix = b2[0] if b2 else 0
                    if prefix in (0x5b, 0x4f):  # '[' or 'O'
                        if _has_input(0.3):
                            b3 = _read_byte()
                            arrow = b3[0] if b3 else 0
                            if arrow == 0x41 and filtered:  # A = Up
                                cursor = (cursor - 1) % len(filtered)
                            elif arrow == 0x42 and filtered:  # B = Down
                                cursor = (cursor + 1) % len(filtered)
                            # else: ignore (right/left/other)
                        # else: partial sequence, ignore
                    # else: Esc + unknown char — ignore
                else:
                    # Plain Esc — cancel
                    _wipe()
                    return None
            elif ch in (0x03, 0x04):  # Ctrl+C / Ctrl+D
                _wipe()
                return None
            elif ch in (0x0d, 0x0a):  # Enter
                _wipe()
                return filtered[cursor][2] if filtered else None
            elif ch == 0x71 and not filter_text:  # 'q'
                _wipe()
                return None
            elif ch in (0x7f, 0x08):  # Backspace
                if filter_text:
                    filter_text = filter_text[:-1]
                    _apply_filter()
                else:
                    continue
            elif ch == 0x15:  # Ctrl+U
                filter_text = ""
                _apply_filter()
            elif 0x20 <= ch < 0x7f:  # Printable ASCII
                filter_text += chr(ch)
                _apply_filter()
            else:
                continue

            _wipe()
            _render()

    except (OSError, ValueError):
        return None
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
        except termios.error:
            pass


def _banner() -> str:
    """Glass banner: `✦ ＰｒｅｄａＣｏｒｅ ✦` in fullwidth Unicode + subtitle.

    Title is rendered in fullwidth characters (each ~2 cells wide in a
    monospace terminal) so it visually doubles in size without ASCII
    art. Letters get a 9-color gradient across P-r-e-d-a-C-o-r-e,
    sparkles flank both sides, italic violet subtitle on line 2.
    """
    pink   = "#F9A8D4"
    cyan   = "#7DD3FC"
    violet = "#C4B5FD"

    # Fullwidth Unicode — each char is ~2 terminal cells wide
    fullwidth = ["Ｐ", "ｒ", "ｅ", "ｄ", "ａ", "Ｃ", "ｏ", "ｒ", "ｅ"]
    grad = [
        "#F9A8D4", "#E8A8D8", "#D7A8DC", "#C6A8E0", "#B5A8E4",
        "#A4A8E8", "#93A8EC", "#82A8F0", "#7DD3FC",
    ]

    title = "".join(
        f"[bold {grad[i]}]{fullwidth[i]}[/bold {grad[i]}]"
        for i in range(len(fullwidth))
    )

    line1 = f"  [{pink}]✦[/{pink}]  {title}  [{cyan}]✦[/{cyan}]"
    line2 = f"  [italic {violet}]the apex autonomous agent[/italic {violet}]"

    return "\n" + line1 + "\n" + line2 + "\n"


def _autoload_env_for_cli(config_path: str | None = None) -> Path | None:
    """
    Auto-load `.env` files for CLI usage.

    Loads ALL found files (override=False means first value wins):
      1) ~/.predacore/.env          — API keys (always loaded first = highest priority)
      2) PREDACORE_ENV_FILE         — explicit override
      3) .env next to --config   — project-specific
      4) cwd/.env                — local overrides
      5) repository-root/.env    — repo defaults
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    candidates = []

    # ~/.predacore/.env is ALWAYS first — single source of truth for keys
    candidates.append(Path.home() / ".predacore" / ".env")

    explicit = os.getenv("PREDACORE_ENV_FILE", "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())

    if config_path:
        candidates.append(Path(config_path).expanduser().resolve().parent / ".env")

    candidates.append(Path.cwd() / ".env")
    candidates.append(Path(__file__).resolve().parents[2] / ".env")

    loaded: list[Path] = []
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists() and candidate.is_file():
            load_dotenv(dotenv_path=candidate, override=False)
            loaded.append(candidate)

    if loaded:
        logger.info("Loaded environment from: %s", ", ".join(str(p) for p in loaded))
        return loaded[0]
    return None


# ── Interactive Chat ─────────────────────────────────────────────────


class _AsyncSpinner:
    """Async braille-dot spinner with a cycling cyan hue.

    Start with ``.start(label)`` and stop with ``await .stop()``. Safe to
    start/stop repeatedly across a single chat turn (thinking → tool →
    thinking → stream). No-ops cleanly when stdout isn't a TTY.
    """

    _FRAMES = "⣾⣽⣻⢿⡿⣟⣯⣷"

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._stop: asyncio.Event = asyncio.Event()
        self._label: str = ""

    async def _run(self) -> None:
        i = 0
        try:
            while not self._stop.is_set():
                frame = self._FRAMES[i % len(self._FRAMES)]
                hue = 39 + (i % 6)  # cycling cyan shades
                sys.stdout.write(
                    f"\r  \x1b[38;5;{hue}m{frame}\x1b[0m \x1b[2m{self._label}\x1b[0m"
                )
                sys.stdout.flush()
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=0.08)
                except asyncio.TimeoutError:
                    pass
                i += 1
        finally:
            sys.stdout.write("\r\x1b[K")
            sys.stdout.flush()

    def start(self, label: str) -> None:
        if self._task and not self._task.done():
            self._label = label  # just update the caption
            return
        if not sys.stdout.isatty():
            return
        self._label = label
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="chat-spinner")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass
        self._task = None


class _ChatRenderer:
    """Stateful renderer for WebSocket events — keeps streaming clean,
    interleaves tool rows, and closes each turn with a stats pill."""

    _TOOL_ICONS = {
        "run_command": "\U0001f4bb", "read_file": "\U0001f4c4", "write_file": "\u270f\ufe0f",
        "list_directory": "\U0001f4c1", "web_search": "\U0001f310", "web_scrape": "\U0001f50d",
        "strategic_plan": "\U0001f5fa\ufe0f", "multi_agent": "\U0001f916", "speak": "\U0001f50a",
        "memory_store": "\U0001f9e0", "memory_recall": "\U0001f9e0",
        "git_context": "\U0001f4ca", "git_diff_summary": "\U0001f4dd",
    }

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self._spinner = _AsyncSpinner()
        self._streaming: bool = False
        self._last_stats: dict | None = None

    async def on_event(self, data: dict) -> bool:
        """Render a single WS event. Returns True when the turn is complete."""
        t = data.get("type")
        if t == "thinking":
            if not self._streaming:
                self._spinner.start("thinking")
            return False

        if t == "tool_start":
            await self._spinner.stop()
            name = data.get("name", "")
            icon = self._TOOL_ICONS.get(name, "\u25c6")
            console.print(
                f"  [glass.violet]├─[/glass.violet]  {icon}  "
                f"[bold #C4B5FD]{name}[/bold #C4B5FD]"
            )
            self._spinner.start(f"running {name}")
            return False

        if t == "tool_end":
            await self._spinner.stop()
            ms = int(data.get("duration_ms", 0))
            preview = (data.get("result_preview") or "").replace("\n", " ").strip()
            if len(preview) > 72:
                preview = preview[:69] + "..."
            tail = f"[glass.text.dim]{preview}[/glass.text.dim]  " if preview else ""
            console.print(
                f"  [glass.violet]│[/glass.violet]    "
                f"[glass.mint]\u21b3[/glass.mint]  {tail}"
                f"[glass.text.dim]({ms}ms)[/glass.text.dim]"
            )
            return False

        if t == "stream":
            if not self._streaming:
                await self._spinner.stop()
                console.print()  # blank line between tool rows and response
                console.print(
                    f"  [#F9A8D4]\u2726[/#F9A8D4]  "
                    f"[bold #7DD3FC]{self.agent_name}[/bold #7DD3FC]"
                )
                self._streaming = True
            chunk = data.get("content", "")
            sys.stdout.write(chunk)
            sys.stdout.flush()
            return False

        if t == "response_stats":
            self._last_stats = data
            return False

        if t == "message":
            await self._spinner.stop()
            if self._streaming:
                print()  # close streamed line
            else:
                # Non-streaming provider — the assistant text arrives here
                text = (data.get("content") or "").rstrip()
                if text:
                    console.print()
                    console.print(
                        f"  [#F9A8D4]\u2726[/#F9A8D4]  "
                        f"[bold #7DD3FC]{self.agent_name}[/bold #7DD3FC]"
                    )
                    for line in text.splitlines() or [""]:
                        console.print(f"  {line}")
            self._print_footer()
            self._streaming = False
            self._last_stats = None
            return True

        if t == "error":
            await self._spinner.stop()
            self._streaming = False
            console.print(
                f"\n  [glass.rose]\u2717[/glass.rose]  "
                f"[error]{data.get('content', 'Unknown error')}[/error]\n"
            )
            return True

        # typing / system / stats_update — silent
        return False

    def _print_footer(self) -> None:
        stats = self._last_stats or {}
        elapsed = float(stats.get("elapsed_s") or 0)
        tokens = int(stats.get("tokens") or 0)
        tools = int(stats.get("tools_used") or 0)
        provider = (stats.get("provider") or "").strip()
        parts: list[str] = []
        if elapsed:
            parts.append(f"[#7DD3FC]{elapsed:.1f}s[/#7DD3FC]")
        if tokens:
            parts.append(f"[#86EFAC]{tokens}[/#86EFAC][glass.text.dim] tok[/glass.text.dim]")
        if tools:
            word = "tool" if tools == 1 else "tools"
            parts.append(f"[#C4B5FD]{tools}[/#C4B5FD][glass.text.dim] {word}[/glass.text.dim]")
        if provider:
            parts.append(f"[glass.text.dim]{provider}[/glass.text.dim]")
        if not parts:
            console.print()
            return
        pill = " [glass.text.dim]\u00b7[/glass.text.dim] ".join(parts)
        console.print(
            f"  [glass.text.dim]\u2570\u2500[/glass.text.dim] {pill}"
        )
        console.print()

    async def close(self) -> None:
        await self._spinner.stop()


async def _run_chat(
    session_id: str | None = None,
    config_path: str | None = None,
    profile: str | None = None,
) -> None:
    """Thin terminal client for the running daemon's webchat WebSocket.

    The daemon owns the engine. Terminal chat, browser webchat, and
    channel adapters all talk to the same gateway — one source of truth
    for sessions, memory, and identity. If the daemon isn't running,
    print a hint and exit rather than spawning a second engine.
    """
    import aiohttp

    from .config import load_config

    config = load_config(config_path, profile_override=profile)
    port = _resolve_webchat_port(config)

    _animated_banner_reveal()

    if not await _probe_daemon_port(port):
        _print_daemon_missing_panel(port)
        return

    _print_chat_banner(config, port)

    history_path = Path.home() / ".predacore" / "chat_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_session = PromptSession(
        history=FileHistory(str(history_path)),
        multiline=False,
        auto_suggest=AutoSuggestFromHistory(),
        enable_history_search=True,
    )

    cid = (session_id or uuid.uuid4().hex)[:32]
    url = f"ws://127.0.0.1:{port}/ws?cid={cid}"

    async with aiohttp.ClientSession() as http:
        try:
            with console.status(
                "[glass.cyan]connecting to daemon[/glass.cyan]",
                spinner="dots",
                spinner_style="glass.cyan",
            ):
                ws = await http.ws_connect(url, heartbeat=30.0, max_msg_size=0)
        except (aiohttp.ClientError, OSError) as exc:
            console.print(
                f"\n  [glass.rose]\u2717[/glass.rose]  "
                f"[error]Couldn't connect to daemon:[/error] {exc}\n"
            )
            return

        response_done = asyncio.Event()
        response_done.set()
        renderer = _ChatRenderer(agent_name=config.name)
        recv_task = asyncio.create_task(
            _chat_recv_loop(ws, response_done, renderer),
            name="chat-recv",
        )

        try:
            while True:
                await response_done.wait()
                try:
                    user_input = await prompt_session.prompt_async(
                        HTML(
                            "<ansibrightcyan>\u2726</ansibrightcyan> "
                            "<ansigreen><b>you</b></ansigreen>"
                            "<ansigray> \u25b8 </ansigray>"
                        ),
                    )
                except (KeyboardInterrupt, EOFError):
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input in ("/exit", "/quit"):
                    break
                if user_input == "/clear":
                    console.clear()
                    continue
                if user_input == "/help":
                    _print_chat_help()
                    continue

                response_done.clear()
                try:
                    await ws.send_json({"content": user_input})
                except (ConnectionError, aiohttp.ClientError) as exc:
                    console.print(
                        f"\n  [glass.rose]\u2717[/glass.rose]  "
                        f"[error]Daemon disconnected: {exc}[/error]\n"
                    )
                    break

        finally:
            recv_task.cancel()
            try:
                await recv_task
            except (asyncio.CancelledError, Exception):
                pass
            await renderer.close()
            if not ws.closed:
                await ws.close()

    console.print(
        "\n  [#F9A8D4]\u273d[/#F9A8D4]  "
        "[glass.text.dim]until next time.[/glass.text.dim]\n"
    )


async def _chat_recv_loop(
    ws,
    response_done: asyncio.Event,
    renderer: _ChatRenderer,
) -> None:
    """Drive the renderer from WebSocket events.

    Sets `response_done` when the turn is complete so the main loop can
    re-prompt without racing the stream.
    """
    from aiohttp import WSMsgType

    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                if msg.type in (WSMsgType.CLOSED, WSMsgType.CLOSE, WSMsgType.ERROR):
                    break
                continue
            try:
                data = json.loads(msg.data)
            except (json.JSONDecodeError, ValueError):
                continue
            try:
                turn_done = await renderer.on_event(data)
            except Exception:
                logger.debug("Renderer error", exc_info=True)
                turn_done = False
            if turn_done:
                response_done.set()
    finally:
        response_done.set()


async def _probe_daemon_port(port: int, timeout: float = 0.5) -> bool:
    """Return True if the daemon's webchat port accepts a TCP connection."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port),
            timeout=timeout,
        )
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass
        return True
    except (OSError, asyncio.TimeoutError):
        return False


def _resolve_webchat_port(config) -> int:
    """Read the webchat port from config (default 3000)."""
    wc = getattr(config.channels, "webchat", None)
    if isinstance(wc, dict):
        return int(wc.get("port", 3000))
    if wc is not None:
        return int(getattr(wc, "port", 3000))
    return 3000


def _animated_banner_reveal() -> None:
    """Print the glass banner with a gentle line-by-line reveal.

    No-ops into a plain print when stdout isn't a TTY so piped output
    and CI logs stay clean.
    """
    content = _banner()
    if not sys.stdout.isatty():
        console.print(content)
        return
    for line in content.split("\n"):
        if line.strip():
            console.print(line)
            time.sleep(0.06)
        else:
            console.print()
    time.sleep(0.04)


def _print_daemon_missing_panel(port: int) -> None:
    """Beautiful 'daemon not running' panel with next-step hint."""
    body = Group(
        Text("The daemon isn't running yet.", style="bold #FCA5A5"),
        Text(""),
        Text("Start it in another terminal:", style="glass.text.dim"),
        Text("  predacore start --daemon", style="bold #7DD3FC"),
        Text(""),
        Text(
            f"Chat attaches to the daemon on 127.0.0.1:{port}",
            style="glass.text.dim",
        ),
    )
    console.print()
    console.print(_glass_panel(
        body,
        title=(
            "[glass.rose]\u2717[/glass.rose] "
            "[bold grey85]Cannot connect[/bold grey85]"
        ),
        border_style="glass.rose",
        width=64,
    ))
    console.print()


def _print_chat_banner(config, port: int) -> None:
    """Compact session panel — daemon URL, provider, model, trust level."""
    trust_style_map = {
        "yolo": "#FCA5A5", "normal": "#7DD3FC", "paranoid": "#86EFAC",
    }
    trust_style = trust_style_map.get(config.security.trust_level, "glass.text")

    info = Table(show_header=False, box=None, padding=(0, 2), expand=False)
    info.add_column("key", style="glass.text.muted", width=10, justify="right")
    info.add_column("value", style="glass.text")
    info.add_row("Daemon",   Text(f"ws://127.0.0.1:{port}", style="#7DD3FC"))
    info.add_row("Provider", Text(config.llm.provider, style="bold #7DD3FC"))
    info.add_row("Model",    Text(config.llm.model or "auto", style="#C4B5FD"))
    info.add_row("Trust",    Text(config.security.trust_level, style=f"bold {trust_style}"))

    console.print()
    console.print(_glass_panel(
        info,
        title=(
            "[#F9A8D4]\u2726[/#F9A8D4]  "
            "[bold grey85]Session[/bold grey85]  "
            "[#7DD3FC]\u2726[/#7DD3FC]"
        ),
        border_style="glass.border",
        padding=(1, 2),
        width=64,
    ))
    console.print()
    console.print(
        "  [glass.text.dim]/help[/glass.text.dim]  "
        "[glass.text.dim]\u00b7[/glass.text.dim]  "
        "[glass.text.dim]/clear[/glass.text.dim]  "
        "[glass.text.dim]\u00b7[/glass.text.dim]  "
        "[glass.text.dim]/exit[/glass.text.dim]  "
        "[glass.text.dim]\u00b7[/glass.text.dim]  "
        "[glass.text.dim]Ctrl+D to quit[/glass.text.dim]"
    )
    console.print(Rule(style="glass.border.soft"))
    console.print()


def _print_chat_help() -> None:
    """Print the in-chat slash-command reference."""
    console.print()
    console.print("  [bold grey85]In-chat commands[/bold grey85]")
    console.print("  [glass.cyan]/help[/glass.cyan]    show this")
    console.print("  [glass.cyan]/clear[/glass.cyan]   clear the screen")
    console.print("  [glass.cyan]/exit[/glass.cyan]    leave  [glass.text.dim](Ctrl+D or /quit also work)[/glass.text.dim]")
    console.print()
    console.print(
        "  [glass.text.dim]For status, tools, sessions, doctor — use the top-level CLI:[/glass.text.dim]"
    )
    console.print(
        "  [glass.cyan]predacore status[/glass.cyan]   "
        "[glass.cyan]predacore doctor[/glass.cyan]   "
        "[glass.cyan]predacore logs[/glass.cyan]"
    )
    console.print()


# ── Setup Wizard ─────────────────────────────────────────────────────


async def _run_setup() -> None:
    """Guided onboarding wizard — welcoming, complete, 5 steps, ~90 seconds."""
    import platform
    import shutil
    import subprocess
    import sys

    from .config import DEFAULT_CONFIG_FILE, DEFAULT_HOME, save_default_config

    # ── Welcome panel ──
    console.print()
    console.print(Align.center(_gradient_text("✦  PREDACORE  ✦", colors=[
        "#F9A8D4", "#F0A5D4", "#DDA0DD", "#C4B5FD", "#A5B4FC", "#8AB4F8", "#7DD3FC"
    ])))
    console.print(Align.center(Text("the apex autonomous agent", style="glass.text.dim italic")))
    console.print()

    welcome_body = Text.assemble(
        ("You're about to set up your own AI agent. Takes about 90 seconds.\n", "glass.text"),
        ("Everything lives in ", "glass.text.muted"),
        ("~/.predacore/", "glass.cyan"),
        (" — nothing touches the rest of your system.\n\n", "glass.text.muted"),
        ("What you're building:\n", "bold grey85"),
        ("  ◆  ", "glass.cyan"),
        ("An agent that uses tools — files, browser, terminal, Android, memory\n", "glass.text"),
        ("  ◆  ", "glass.violet"),
        ("A relationship that grows — it learns about you through conversation\n", "glass.text"),
        ("  ◆  ", "glass.pink"),
        ("Channels — talk to it from terminal, web, Telegram, Discord\n", "glass.text"),
        ("  ◆  ", "glass.mint"),
        ("Total privacy — runs locally, you choose what leaves the machine", "glass.text"),
    )
    console.print(Align.center(
        _glass_panel(
            welcome_body,
            title="[glass.pink]✨[/glass.pink]  [bold glass.text]Welcome[/bold glass.text]  [glass.cyan]✨[/glass.cyan]",
            border_style="glass.border.accent",
            padding=(1, 3),
            width=78,
        )
    ))
    console.print()

    # Check if config already exists
    if DEFAULT_CONFIG_FILE.exists():
        console.print(Align.center(Text.assemble(
            ("  Existing config found at ", "glass.text.muted"),
            (str(DEFAULT_CONFIG_FILE), "glass.cyan"),
        )))
        overwrite = console.input(
            "  [glass.violet]❯[/glass.violet]  [glass.text]Overwrite and start fresh?[/glass.text] [glass.text.dim][y/N]:[/glass.text.dim] "
        ).strip().lower()
        if overwrite != "y":
            console.print()
            console.print(Align.center(Text("Setup cancelled. Your existing config is untouched.", style="glass.text.muted italic")))
            return
        console.print()

    # ── Step 1: System check ──
    console.print(_step_rule(1, 5, "System check"))
    console.print()

    sys_table = Table(show_header=False, box=None, padding=(0, 2), expand=False)
    sys_table.add_column("icon", width=2)
    sys_table.add_column("item", style="glass.text", min_width=28)
    sys_table.add_column("status", justify="right")
    sys_table.add_column("detail", style="glass.text.dim")

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 11)
    if py_ok:
        sys_table.add_row("[glass.cyan]◆[/glass.cyan]", f"Python {py_ver}", _status_pill("ready", "ok"), "")
    else:
        sys_table.add_row("[glass.rose]✗[/glass.rose]", f"Python {py_ver}", _status_pill("need 3.11+", "err"), "upgrade required")

    # OS
    os_name = platform.system()
    if os_name == "Darwin":
        os_label = f"macOS {platform.mac_ver()[0] or ''}".strip()
    else:
        os_label = f"{os_name} {platform.release()}".strip()
    sys_table.add_row("[glass.cyan]◆[/glass.cyan]", os_label, _status_pill("ready", "ok"), "")

    # Chrome
    chrome_paths = [
        "/Applications/Google Chrome.app",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
    ]
    chrome_found = shutil.which("google-chrome") or shutil.which("chromium") or any(
        Path(p).exists() for p in chrome_paths
    )
    if chrome_found:
        sys_table.add_row("[glass.cyan]◆[/glass.cyan]", "Chrome", _status_pill("ready", "ok"), "browser_control enabled")
    else:
        sys_table.add_row("[glass.amber]◇[/glass.amber]", "Chrome", _status_pill("missing", "warn"), "browser_control disabled")

    # ADB
    if shutil.which("adb"):
        sys_table.add_row("[glass.cyan]◆[/glass.cyan]", "ADB", _status_pill("ready", "ok"), "Android control enabled")
    else:
        sys_table.add_row("[glass.violet]◇[/glass.violet]", "ADB", _status_pill("optional", "skip"), "brew install --cask android-platform-tools")

    console.print(_glass_panel(
        sys_table,
        title="[glass.cyan]⚙[/glass.cyan]  [bold glass.text]Environment[/bold glass.text]",
        padding=(1, 2),
        width=78,
    ))

    if not py_ok:
        console.print()
        console.print("    [glass.rose]✗[/glass.rose]  [glass.text]Upgrade Python before continuing:[/glass.text] [glass.cyan]https://www.python.org/downloads/[/glass.cyan]")
        return

    if os_name == "Darwin":
        console.print()
        console.print(Padding(
            Text.from_markup(
                "  [glass.violet]ⓘ[/glass.violet]  [glass.text.muted]macOS note: on the first desktop/screen action, "
                "macOS will request Accessibility\n     and Screen Recording permissions. "
                "Grant both in [glass.text]System Settings → Privacy[/glass.text].[/glass.text.muted]"
            ),
            (0, 2),
        ))

    console.print()

    # ── Step 2: Auto-detect LLM providers ──
    console.print(_step_rule(2, 5, "Pick your LLM provider"))
    console.print()

    detected: dict[str, dict[str, str]] = {}

    # Check Gemini CLI
    gemini_bin = shutil.which("gemini")
    if gemini_bin:
        detected["gemini-cli"] = {"status": "ready", "detail": "CLI installed, no API key needed"}
    else:
        detected["gemini-cli"] = {"status": "missing", "detail": "Install: npm i -g @google/gemini-cli"}

    # Check Gemini API
    gemini_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    if gemini_key:
        detected["gemini"] = {"status": "ready", "detail": "GEMINI_API_KEY=****configured"}
    else:
        detected["gemini"] = {"status": "no_key", "detail": "Set GEMINI_API_KEY"}

    # Check OpenAI
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        detected["openai"] = {"status": "ready", "detail": "OPENAI_API_KEY=****configured"}
    else:
        detected["openai"] = {"status": "no_key", "detail": "Set OPENAI_API_KEY"}

    # Check Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        detected["anthropic"] = {"status": "ready", "detail": "ANTHROPIC_API_KEY=****configured"}
    else:
        detected["anthropic"] = {"status": "no_key", "detail": "Set ANTHROPIC_API_KEY"}

    # Check OpenRouter — one key, 100+ models (Claude, GPT, Gemini, Llama, etc.)
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if openrouter_key:
        detected["openrouter"] = {"status": "ready", "detail": "OPENROUTER_API_KEY=****configured"}
    else:
        detected["openrouter"] = {"status": "no_key", "detail": "Get a free key at openrouter.ai/keys"}

    # Check Ollama
    ollama_bin = shutil.which("ollama")
    ollama_running = False
    if ollama_bin:
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                models = [
                    line.split()[0]
                    for line in result.stdout.strip().split("\n")[1:]
                    if line.strip()
                ]
                if models:
                    detected["ollama"] = {
                        "status": "ready",
                        "detail": f"Running, models: {', '.join(models[:3])}",
                    }
                    ollama_running = True
                else:
                    detected["ollama"] = {
                        "status": "no_models",
                        "detail": "Running but no models. Run: ollama pull llama3.2",
                    }
            else:
                detected["ollama"] = {
                    "status": "not_running",
                    "detail": "Installed but not running. Run: ollama serve",
                }
        except (OSError, subprocess.SubprocessError):
            detected["ollama"] = {"status": "not_running", "detail": "Installed but not responding"}
    else:
        detected["ollama"] = {"status": "missing", "detail": "Install: brew install ollama"}

    # Display detected providers (ordered by recommendation priority)
    provider_labels = {
        "gemini-cli":  ("Gemini CLI",  "Free, no API key — uses the gemini CLI binary"),
        "gemini":      ("Gemini API",  "Google AI Studio, generous free tier"),
        "openrouter":  ("OpenRouter",  "One key → 100+ models (Claude, GPT, Gemini, Llama). Free tier available."),
        "anthropic":   ("Anthropic",   "Claude Opus 4.6, Claude Sonnet — highest quality"),
        "openai":      ("OpenAI",      "GPT-4o, GPT-5"),
        "ollama":      ("Ollama",      "Local models, fully private, no internet needed"),
    }

    ready_providers: list[str] = []
    provider_menu: dict[str, str] = {}

    provider_table = Table(
        show_header=True,
        header_style="bold grey58",
        box=None,
        padding=(0, 2),
        expand=False,
    )
    provider_table.add_column("#", style="glass.text.dim", width=2, justify="right")
    provider_table.add_column("Provider", style="glass.text", min_width=12)
    provider_table.add_column("Status", justify="center", width=16)
    provider_table.add_column("Tagline", style="glass.text")
    provider_table.add_column("Details", style="glass.text.dim")

    idx = 1
    for pname, (label, tagline) in provider_labels.items():
        info = detected[pname]
        if info["status"] == "ready":
            status_cell = _status_pill("ready", "ok")
            ready_providers.append(pname)
        elif info["status"] in ("no_key", "no_models", "not_running"):
            status_cell = _status_pill("setup", "warn")
        else:
            status_cell = _status_pill("missing", "skip")
        provider_menu[str(idx)] = pname
        provider_table.add_row(
            str(idx),
            Text(label, style="bold grey85"),
            status_cell,
            tagline,
            info["detail"],
        )
        idx += 1

    console.print(_glass_panel(
        provider_table,
        title="[glass.violet]✦[/glass.violet]  [bold glass.text]Available Providers[/bold glass.text]",
        padding=(1, 2),
        width=100,
    ))
    console.print()

    # Recommend the best ready provider — free/no-setup first, then meta, then paid, then local
    preference_order = ["gemini-cli", "gemini", "openrouter", "anthropic", "openai", "ollama"]
    default_provider = "gemini-cli"
    for p in preference_order:
        if p in ready_providers:
            default_provider = p
            break
    default_idx = next(k for k, v in provider_menu.items() if v == default_provider)
    default_label = provider_labels[default_provider][0]

    choice = console.input(
        f"  [glass.violet]❯[/glass.violet]  [glass.text]Choose provider[/glass.text] "
        f"[glass.text.dim]\\[{default_idx} = {default_label}]:[/glass.text.dim] "
    ).strip() or default_idx
    provider = provider_menu.get(choice, default_provider)
    provider_display = provider_labels.get(provider, (provider,))[0]

    # If provider needs an API key and we don't have one, ask for it
    api_key = ""
    key_env_map = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    if provider in key_env_map and detected[provider]["status"] == "no_key":
        env_var = key_env_map[provider]
        console.print()
        console.print(Padding(
            Text.from_markup(
                f"  [glass.amber]⚠[/glass.amber]  [glass.text]{provider_display} needs an API key.[/glass.text]  "
                f"[glass.text.dim]It will be saved to ~/.predacore/.env with chmod 600.[/glass.text.dim]"
            ),
            (0, 1),
        ))
        api_key = console.input(
            f"  [glass.violet]❯[/glass.violet]  [glass.text]Paste your[/glass.text] "
            f"[glass.cyan]{env_var}[/glass.cyan][glass.text.dim]:[/glass.text.dim] "
        ).strip()
        if not api_key:
            console.print("    [glass.text.dim]Skipped. Set the env var later to enable this provider.[/glass.text.dim]")

    # ── Step 3: Trust level ──
    console.print()
    console.print(_step_rule(3, 5, "Trust level"))
    console.print()

    def _trust_card(num: str, name: str, accent: str, headline: str, tagline: str) -> Panel:
        body = Text.assemble(
            (f"{num}  ", f"bold {accent}"),
            (name.upper(), f"bold {accent}"),
            ("\n\n", ""),
            (headline, "glass.text"),
            ("\n", ""),
            (tagline, "glass.text.dim"),
        )
        return Panel(
            body,
            border_style=accent,
            box=box.ROUNDED,
            padding=(1, 2),
            width=34,
        )

    trust_cards = [
        _trust_card(
            "1", "yolo", "glass.rose",
            "Auto-run everything.",
            "Maximum speed, minimum friction. Irreversible ops still confirm.",
        ),
        _trust_card(
            "2", "normal", "glass.cyan",
            "Auto-run reads. Confirm writes.",
            "Recommended default. Good balance of speed and safety.",
        ),
        _trust_card(
            "3", "paranoid", "glass.mint",
            "Confirm every action.",
            "For sensitive environments or when you're auditing behavior.",
        ),
    ]
    console.print(Columns(trust_cards, equal=True, expand=False, padding=(0, 1)))
    console.print()
    trust_choice = console.input(
        "  [glass.violet]❯[/glass.violet]  [glass.text]Choose trust level[/glass.text] "
        "[glass.text.dim]\\[2 = normal]:[/glass.text.dim] "
    ).strip() or "2"
    trust_map = {"1": "yolo", "2": "normal", "3": "paranoid"}
    trust_level = trust_map.get(trust_choice, "normal")

    # ── Step 4: Channels ──
    console.print()
    console.print(_step_rule(4, 5, "Channels"))
    console.print()

    channels_table = Table(show_header=False, box=None, padding=(0, 2), expand=False)
    channels_table.add_column("idx", style="glass.text.dim", width=2, justify="right")
    channels_table.add_column("name", min_width=12)
    channels_table.add_column("desc", style="glass.text")
    channels_table.add_row(
        "●", Text("terminal (cli)", style="bold #86EFAC"),
        Text("Always on — you can always chat from a shell", style="glass.text"),
    )
    channels_table.add_row(
        "1", Text("webchat", style="bold #7DD3FC"),
        "Browser UI at [glass.cyan]http://localhost:3000[/glass.cyan] [glass.text.dim](recommended)[/glass.text.dim]",
    )
    channels_table.add_row(
        "2", Text("telegram", style="bold #C4B5FD"),
        "Chat from Telegram [glass.text.dim](needs bot token from @BotFather)[/glass.text.dim]",
    )
    channels_table.add_row(
        "3", Text("discord", style="bold #F9A8D4"),
        "Chat from Discord [glass.text.dim](needs bot token from Developer Portal)[/glass.text.dim]",
    )
    console.print(_glass_panel(
        channels_table,
        title="[glass.cyan]📡[/glass.cyan]  [bold glass.text]Available channels[/bold glass.text]",
        padding=(1, 2),
        width=90,
    ))
    console.print()
    ch_choice = console.input(
        "  [glass.violet]❯[/glass.violet]  [glass.text]Enable channels[/glass.text] "
        "[glass.text.dim]\\[1 = webchat, comma-separated e.g. '1,2']:[/glass.text.dim] "
    ).strip() or "1"
    ch_map = {"1": "webchat", "2": "telegram", "3": "discord"}
    selected_channels = ["cli"]
    for token in ch_choice.split(","):
        token = token.strip()
        if token in ch_map:
            ch_name = ch_map[token]
            if ch_name not in selected_channels:
                selected_channels.append(ch_name)

    # Collect channel tokens inline so the user doesn't have to figure out .env later
    channel_secrets: dict[str, str] = {}
    if "telegram" in selected_channels:
        console.print()
        console.print(_glass_panel(
            Text.assemble(
                ("Get your bot token from ", "glass.text"),
                ("@BotFather", "bold #C4B5FD"),
                (" on Telegram:\n", "glass.text"),
                ("  1.  ", "glass.text.dim"),
                ("Message @BotFather\n", "glass.text"),
                ("  2.  ", "glass.text.dim"),
                ("/newbot\n", "glass.cyan"),
                ("  3.  ", "glass.text.dim"),
                ("Copy the token it gives you", "glass.text"),
            ),
            title="[glass.violet]✈[/glass.violet]  [bold glass.text]Telegram bot setup[/bold glass.text]",
            border_style="glass.violet",
            padding=(1, 2),
            width=78,
        ))
        tg_token = console.input(
            "  [glass.violet]❯[/glass.violet]  [glass.text]Paste[/glass.text] "
            "[glass.cyan]TELEGRAM_BOT_TOKEN[/glass.cyan] [glass.text.dim](Enter to skip):[/glass.text.dim] "
        ).strip()
        if tg_token:
            channel_secrets["TELEGRAM_BOT_TOKEN"] = tg_token
        else:
            console.print("    [glass.text.dim]Skipped. Add the token to ~/.predacore/.env later to enable Telegram.[/glass.text.dim]")

    if "discord" in selected_channels:
        console.print()
        console.print(_glass_panel(
            Text.assemble(
                ("Get your bot token from ", "glass.text"),
                ("discord.com/developers/applications", "bold #F9A8D4"),
                (":\n", "glass.text"),
                ("  1.  ", "glass.text.dim"),
                ("New Application\n", "glass.text"),
                ("  2.  ", "glass.text.dim"),
                ("Bot tab\n", "glass.text"),
                ("  3.  ", "glass.text.dim"),
                ("Reset Token → Copy", "glass.text"),
            ),
            title="[glass.pink]◈[/glass.pink]  [bold glass.text]Discord bot setup[/bold glass.text]",
            border_style="glass.pink",
            padding=(1, 2),
            width=78,
        ))
        dc_token = console.input(
            "  [glass.violet]❯[/glass.violet]  [glass.text]Paste[/glass.text] "
            "[glass.cyan]DISCORD_BOT_TOKEN[/glass.cyan] [glass.text.dim](Enter to skip):[/glass.text.dim] "
        ).strip()
        if dc_token:
            channel_secrets["DISCORD_BOT_TOKEN"] = dc_token
        else:
            console.print("    [glass.text.dim]Skipped. Add the token to ~/.predacore/.env later to enable Discord.[/glass.text.dim]")

    # ── Step 5: Save + test ──
    console.print()
    console.print(_step_rule(5, 5, "Save & verify"))
    console.print()
    config_path = save_default_config(
        provider=provider,
        trust_level=trust_level,
        channels=selected_channels,
    )
    console.print(f"  [glass.mint]◆[/glass.mint]  Config written to [glass.cyan]{config_path}[/glass.cyan]")

    # Consolidated .env writing — LLM API key + any channel tokens, all at once
    env_writes: dict[str, str] = {}
    if api_key and provider in key_env_map:
        env_writes[key_env_map[provider]] = api_key
    env_writes.update(channel_secrets)

    if env_writes:
        env_file = DEFAULT_HOME / ".env"
        existing_content = env_file.read_text() if env_file.exists() else ""
        with open(env_file, "a") as f:
            for env_var, secret_val in env_writes.items():
                if env_var not in existing_content:
                    f.write(f"{env_var}={secret_val}\n")
        os.chmod(str(env_file), 0o600)
        secret_count = len(env_writes)
        plural = "s" if secret_count != 1 else ""
        console.print(f"  [glass.mint]◆[/glass.mint]  {secret_count} secret{plural} written to [glass.cyan]{env_file}[/glass.cyan] [glass.text.dim](chmod 600)[/glass.text.dim]")
        for env_var, secret_val in env_writes.items():
            os.environ[env_var] = secret_val

    # Test the LLM connection with a tiny handshake
    console.print(f"  [glass.violet]◆[/glass.violet]  [glass.text.muted]Testing connection to {provider_display}...[/glass.text.muted]")
    try:
        from .config import load_config
        from .llm_providers.router import LLMInterface

        config = load_config(str(config_path))
        llm = LLMInterface(config)
        result = await llm.chat(
            messages=[{"role": "user", "content": "Say 'PredaCore online' in exactly two words."}],
            max_tokens=20,
        )
        reply = result.get("content", "").strip()
        provider_used = result.get("provider_used", provider)
        latency = result.get("latency_ms", 0)
        console.print(f"  [glass.mint]◆[/glass.mint]  Connected to [bold glass.text]{provider_used}[/bold glass.text] in [glass.cyan]{latency:.0f}ms[/glass.cyan]")
        console.print(f"      [glass.text.dim]Response: \"{reply}\"[/glass.text.dim]")
    except (OSError, ConnectionError, ValueError, KeyError) as e:
        err = str(e)[:120]
        console.print(f"  [glass.amber]◇[/glass.amber]  [glass.text]Connection test failed:[/glass.text] [glass.text.dim]{err}[/glass.text.dim]")
        console.print("      [glass.text.dim]Not fatal — fix the config later and run: [glass.cyan]predacore doctor[/glass.cyan][/glass.text.dim]")

    # ── Configuration summary card ──
    console.print()
    summary_table = Table(show_header=False, box=None, padding=(0, 2), expand=False)
    summary_table.add_column("key", style="glass.text.muted", width=10, justify="right")
    summary_table.add_column("value", style="glass.text")
    summary_table.add_row("Provider", Text.assemble(
        (provider_display, "bold #7DD3FC"),
        (f"  ({provider})", "glass.text.dim"),
    ))
    trust_style = {"yolo": "glass.rose", "normal": "glass.cyan", "paranoid": "glass.mint"}.get(trust_level, "glass.text")
    summary_table.add_row("Trust", Text(trust_level, style=f"bold {trust_style}"))
    summary_table.add_row("Channels", Text(", ".join(selected_channels), style="glass.violet"))
    summary_table.add_row("Config", Text(str(config_path), style="glass.text.dim"))
    summary_table.add_row("Home", Text(str(DEFAULT_HOME), style="glass.text.dim"))

    console.print(Align.center(_glass_panel(
        summary_table,
        title="[glass.mint]✓[/glass.mint]  [bold glass.text]Configuration Summary[/bold glass.text]",
        border_style="glass.mint",
        padding=(1, 2),
        width=78,
    )))
    console.print()

    # ── Bootstrap preview — the "one thing to know" panel ──
    preview_body = Text.assemble(
        ("On your very first message, your agent will greet you with something like:\n\n", "glass.text.muted"),
        ("    ", ""),
        ('"yooo what\'s up, what\'s your name?"', "bold #F9A8D4"),
        ("\n\n", ""),
        ("It doesn't have a name, a personality, or a memory yet —\n", "glass.text.muted"),
        ("you'll discover those together in the first conversation.\n", "glass.text.muted"),
        ("That's not a bug. That's how it's born.\n\n", "glass.text.muted"),
        ("Just talk naturally. It's listening.", "italic grey85"),
    )
    console.print(Align.center(_glass_panel(
        preview_body,
        title="[glass.pink]◉[/glass.pink]  [bold glass.text]One thing to know before you start[/bold glass.text]",
        border_style="glass.pink",
        padding=(1, 3),
        width=78,
    )))
    console.print()

    # ── Next step cards: daemon vs chat ──
    daemon_card = Panel(
        Text.assemble(
            ("predacore start --daemon\n\n", "bold #7DD3FC"),
            ("Full experience — channels, cron,\n", "glass.text"),
            ("memory, reflection, background work.\n\n", "glass.text"),
            ("◆ Recommended for daily use.", "glass.mint"),
        ),
        title="[glass.cyan]▸[/glass.cyan]  [bold glass.text]Production[/bold glass.text]",
        border_style="glass.cyan",
        box=box.ROUNDED,
        padding=(1, 2),
        width=38,
    )
    chat_card = Panel(
        Text.assemble(
            ("predacore chat\n\n", "bold #C4B5FD"),
            ("Quick terminal-only session.\n", "glass.text"),
            ("No daemon, no channels, no persistence.\n\n", "glass.text"),
            ("◇ Good for testing or one-offs.", "glass.text.muted"),
        ),
        title="[glass.violet]▹[/glass.violet]  [bold glass.text]Quick Chat[/bold glass.text]",
        border_style="glass.violet",
        box=box.ROUNDED,
        padding=(1, 2),
        width=38,
    )
    console.print(Align.center(Text("Pick how you want to run it:", style="bold grey85")))
    console.print()
    console.print(Align.center(Columns([daemon_card, chat_card], equal=True, padding=(0, 2))))
    console.print()

    # ── Useful-later footer ──
    footer_table = Table(show_header=False, box=None, padding=(0, 2))
    footer_table.add_column("cmd", style="glass.cyan")
    footer_table.add_column("desc", style="glass.text.muted")
    footer_table.add_row("predacore doctor", "Check system + operator health")
    footer_table.add_row("predacore status", "Show current daemon state")
    footer_table.add_row("predacore stop",   "Gracefully stop the daemon")
    footer_table.add_row("predacore setup",  "Re-run this wizard")
    console.print(Align.center(_glass_panel(
        footer_table,
        title="[glass.amber]✦[/glass.amber]  [bold glass.text]Useful later[/bold glass.text]",
        padding=(1, 2),
        width=78,
    )))
    console.print()
    console.print(Align.center(_gradient_text(
        "✦  Your agent is waiting.  ✦",
        colors=["#F9A8D4", "#C4B5FD", "#7DD3FC", "#C4B5FD", "#F9A8D4"],
    )))
    console.print()


# ── Status Command ───────────────────────────────────────────────────


def _show_status(config_path: str | None = None, profile: str | None = None) -> None:
    """Show PredaCore status summary."""
    from .config import load_config

    console.print(_banner())

    try:
        config = load_config(config_path, profile_override=profile)

        tbl = Table(show_header=False, box=None, padding=(0, 1), expand=False)
        tbl.add_column("key", style="muted", width=16)
        tbl.add_column("value")

        tbl.add_row("Mode", config.mode)
        tbl.add_row("Profile", config.launch.profile)
        tbl.add_row("Provider", config.llm.provider)
        tbl.add_row("Model", config.llm.model or "auto")
        tbl.add_row("Trust", config.security.trust_level)
        tbl.add_row("Home", str(config.home_dir))
        tbl.add_row("Channels", ", ".join(config.channels.enabled))
        tbl.add_row("Approvals", "on" if config.launch.approvals_required else "off")
        tbl.add_row("EGM Mode", config.launch.egm_mode)
        tbl.add_row(
            "Code Network", "on" if config.launch.default_code_network else "off"
        )
        tbl.add_row("OpenClaw", "on" if config.launch.enable_openclaw_bridge else "off")

        if config.launch.enable_openclaw_bridge:
            bridge_url = config.openclaw.base_url or "(unset)"
            tbl.add_row("  Bridge URL", bridge_url)
            tbl.add_row("  Bridge Status", config.openclaw.status_path)
            tbl.add_row(
                "  Bridge Retry",
                f"retries={config.openclaw.max_retries}, backoff={config.openclaw.retry_backoff_seconds}s",
            )
            tbl.add_row(
                "  Bridge Poll",
                f"interval={config.openclaw.poll_interval_seconds}s, max={config.openclaw.max_poll_seconds}s",
            )
            ledger_path = os.getenv(
                "PREDACORE_ACTION_LEDGER_PATH",
                str(Path(config.logs_dir) / "openclaw_actions.jsonl"),
            )
            tbl.add_row("  Bridge Ledger", ledger_path)

        tbl.add_row(
            "Marketplace", "on" if config.launch.enable_plugin_marketplace else "off"
        )
        if config.launch.enable_plugin_marketplace:
            skills_dir = (
                os.getenv("PREDACORE_OPENCLAW_SKILLS_DIR")
                or os.getenv("OPENCLAW_SKILLS_DIR")
                or config.openclaw.skills_dir
                or "auto-detect (installed openclaw)"
            )
            tbl.add_row(
                "  OpenClaw Skills",
                "on" if config.openclaw.auto_import_skills else "off",
            )
            tbl.add_row("  Skills Path", str(skills_dir))

        tbl.add_row(
            "Self-Evolve", "on" if config.launch.enable_self_evolution else "off"
        )
        tbl.add_row(
            "Spawn Limits",
            f"depth={config.launch.max_spawn_depth}, fanout={config.launch.max_spawn_fanout}",
        )
        tbl.add_row("Tool Loop Cap", str(config.launch.max_tool_iterations))

        console.print(
            Panel(
                tbl,
                title="[heading]PredaCore Status[/heading]",
                border_style="cyan",
                width=70,
            )
        )

    except (OSError, ValueError, KeyError) as e:
        console.print(f"  [error]Error loading config: {e}[/error]")
        console.print("  [muted]Run `predacore setup` to create config[/muted]")
    console.print()


# ── Main Entry Point ────────────────────────────────────────────────


def main() -> None:
    """Main CLI entry point."""
    import argparse

    cli_prog = Path(sys.argv[0]).name or "prometheus"
    if cli_prog in {"python", "python3", "python3.10", "python3.11", "python3.12"}:
        cli_prog = "predacore"

    parser = argparse.ArgumentParser(
        prog=cli_prog,
        description="PredaCore — PredaCore AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  chat      Interactive terminal chat
  start     Start PredaCore (foreground or daemon)
  stop      Stop the PredaCore daemon
  install   Install as macOS launchd service
  logs      Tail daemon logs
  setup     Guided onboarding wizard
  status    Show PredaCore status
  doctor    Run system health diagnostics
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive terminal chat")
    chat_parser.add_argument("--session", "-s", help="Resume a specific session")
    chat_parser.add_argument("--config", "-c", help="Path to config file")
    chat_parser.add_argument(
        "--profile",
        choices=["balanced", "public_beast", "enterprise_lockdown"],
        help="Launch profile override",
    )

    # Start command
    start_parser = subparsers.add_parser("start", help="Start PredaCore")
    start_parser.add_argument(
        "--daemon", "-d", action="store_true", help="Run as background daemon"
    )
    start_parser.add_argument(
        "--foreground",
        "-f",
        action="store_true",
        help="Run daemon in foreground (no fork)",
    )
    start_parser.add_argument("--config", "-c", help="Path to config file")
    start_parser.add_argument(
        "--profile",
        choices=["balanced", "public_beast", "enterprise_lockdown"],
        help="Launch profile override",
    )
    start_parser.add_argument(
        "--public",
        action="store_true",
        help="Shortcut for --profile public_beast",
    )

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the PredaCore daemon")
    stop_parser.add_argument("--config", "-c", help="Path to config file")

    # Restart command
    restart_parser = subparsers.add_parser("restart", help="Restart the PredaCore daemon")
    restart_parser.add_argument("--config", "-c", help="Path to config file")

    # Install command
    install_parser = subparsers.add_parser(
        "install", help="Install as macOS launchd service"
    )
    install_parser.add_argument("--config", "-c", help="Path to config file")

    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Tail daemon logs")
    logs_parser.add_argument(
        "--lines",
        "-n",
        type=int,
        default=50,
        help="Number of lines to show (default: 50)",
    )
    logs_parser.add_argument(
        "--follow", "-f", action="store_true", help="Follow log output"
    )
    logs_parser.add_argument("--config", "-c", help="Path to config file")

    # Setup command
    subparsers.add_parser("setup", help="Guided onboarding wizard")

    # Bootstrap command — post-install one-time setup
    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        help="Post-install checks + pre-warm (Rust kernel, BGE model, Playwright, Docker)",
    )
    bootstrap_parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if bootstrap already completed",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show PredaCore status")
    status_parser.add_argument("--config", "-c", help="Path to config file")
    status_parser.add_argument(
        "--profile",
        choices=["balanced", "public_beast", "enterprise_lockdown"],
        help="Launch profile override",
    )

    # Doctor command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation benchmarks")
    eval_parser.add_argument(
        "suite",
        choices=["agent", "code", "load"],
        help="Evaluation suite: agent (ALE-bench), code (SWE-bench), load (stress test)",
    )
    eval_parser.add_argument("--config", "-c", help="Path to config file")
    eval_parser.add_argument(
        "--threshold", type=float, default=0.7, help="Pass threshold (default: 0.7)"
    )

    doctor_parser = subparsers.add_parser(
        "doctor", help="Run system health diagnostics"
    )
    doctor_parser.add_argument("--config", "-c", help="Path to config file")
    doctor_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed diagnostic output"
    )
    doctor_parser.add_argument(
        "--profile",
        choices=["balanced", "public_beast", "enterprise_lockdown"],
        help="Launch profile override",
    )

    args = parser.parse_args()

    # Load project/user environment before command execution.
    _autoload_env_for_cli(getattr(args, "config", None))

    # Configure structured logging
    log_level = os.getenv("PREDACORE_LOG_LEVEL", "WARNING").upper()
    json_logs = os.getenv("PREDACORE_LOG_JSON", "").lower() in ("1", "true", "yes")
    setup_logging(level=log_level, json_mode=json_logs)

    if args.command == "chat":
        asyncio.run(
            _run_chat(
                session_id=args.session,
                config_path=args.config,
                profile=getattr(args, "profile", None),
            )
        )
    elif args.command == "setup":
        asyncio.run(_run_setup())
    elif args.command == "bootstrap":
        _run_bootstrap(force=getattr(args, "force", False))
    elif args.command == "start":
        _run_start(args)
    elif args.command == "stop":
        _run_stop(config_path=getattr(args, "config", None))
    elif args.command == "restart":
        _run_restart(config_path=getattr(args, "config", None))
    elif args.command == "install":
        _run_install(config_path=getattr(args, "config", None))
    elif args.command == "logs":
        _run_logs(
            lines=args.lines,
            follow=args.follow,
            config_path=getattr(args, "config", None),
        )
    elif args.command == "status":
        _show_status(
            config_path=getattr(args, "config", None),
            profile=getattr(args, "profile", None),
        )
    elif args.command == "doctor":
        _run_doctor(
            config_path=getattr(args, "config", None),
            verbose=getattr(args, "verbose", False),
            profile=getattr(args, "profile", None),
        )
    elif args.command == "eval":
        asyncio.run(
            _run_eval(
                suite=args.suite,
                config_path=getattr(args, "config", None),
                threshold=getattr(args, "threshold", 0.7),
            )
        )
    else:
        # No subcommand → the "just install and go" path.
        # bootstrap (if never run) → setup (if no config) → daemon → chat.
        _run_zero_args_flow()


def _run_zero_args_flow() -> None:
    """Runs the shortest path from a fresh ``pip install`` to first message.

    1. If never bootstrapped, run ``predacore bootstrap`` (prewarms the stack).
    2. If no config file exists, run ``predacore setup`` (asks for LLM key).
    3. Make sure the daemon is running (starts it if not).
    4. Drop into ``predacore chat``.

    Any failure short-circuits gracefully with a clear next-step hint.
    """
    from .bootstrap import is_bootstrapped
    from .config import DEFAULT_CONFIG_FILE

    # Step 1 — bootstrap
    if not is_bootstrapped():
        console.print()
        console.print(
            "  [glass.text.dim]First run detected — warming up the stack...[/glass.text.dim]"
        )
        _run_bootstrap(force=False)

    # Step 2 — setup wizard if no config
    if not Path(DEFAULT_CONFIG_FILE).exists():
        console.print()
        console.print(
            "  [glass.text.dim]No config yet — let's pick an LLM provider.[/glass.text.dim]"
        )
        asyncio.run(_run_setup())

    # Step 3 — make sure the daemon is up (so chat's thin client can connect)
    from .config import load_config
    from .services.daemon import PIDManager

    cfg = load_config(None)
    pid_manager = PIDManager(str(Path(cfg.home_dir) / "predacore.pid"))
    if not pid_manager.is_running():
        console.print()
        console.print(
            "  [glass.text.dim]Starting the daemon in the background...[/glass.text.dim]"
        )
        # Reuse the existing start flow in daemon mode
        class _Args:
            daemon = True
            config = None
            profile = None
        _run_start(_Args())
        # Give the daemon a moment to bind the webchat port
        time.sleep(2.0)

    # Step 4 — chat
    asyncio.run(_run_chat(
        session_id=None,
        config_path=None,
        profile=None,
    ))


def _run_bootstrap(force: bool = False) -> None:
    """Render the bootstrap report with the glass table UX."""
    from .bootstrap import run_bootstrap, is_bootstrapped

    console.print()
    console.print(Align.center(_gradient_text(
        "✦  BOOTSTRAP  ✦",
        colors=["#F9A8D4", "#C4B5FD", "#7DD3FC", "#86EFAC"],
    )))
    console.print()
    console.print(Align.center(Text(
        "Pre-warming the stack so your first message is ready.",
        style="glass.text.dim",
    )))
    console.print()

    if is_bootstrapped() and not force:
        console.print(_glass_panel(
            Text.assemble(
                ("Bootstrap already completed.  ", "glass.text"),
                ("Rerun with ", "glass.text.dim"),
                ("--force", "bold #7DD3FC"),
                (" to repeat the checks.", "glass.text.dim"),
            ),
            title="[glass.mint]✓[/glass.mint]  [bold grey85]Skipping[/bold grey85]",
            border_style="glass.mint",
            padding=(1, 2),
            width=72,
        ))
        console.print()
        return

    # Build a live-updating table as steps complete.
    results_table = Table(show_header=False, box=None, padding=(0, 2), expand=False)
    results_table.add_column("icon", width=2)
    results_table.add_column("step", style="glass.text", min_width=32)
    results_table.add_column("status", justify="right")
    results_table.add_column("hint", style="glass.text.dim")

    def _render_step(result) -> None:
        icon_style = {"ok": "#86EFAC", "warn": "#FDE68A", "err": "#FCA5A5"}.get(
            result.severity, "glass.text"
        )
        icon = Text(result.icon, style=f"bold {icon_style}")
        status_pill = _status_pill(
            "ready" if result.ok else ("partial" if result.severity == "warn" else "missing"),
            "ok" if result.ok else ("warn" if result.severity == "warn" else "err"),
        )
        detail = result.detail or ""
        hint = result.hint or ""
        display_hint = hint if hint else Text(detail, style="glass.text.dim")
        results_table.add_row(icon, result.name, status_pill, display_hint)

    report = run_bootstrap(console=console, force=force, step_printer=_render_step)

    console.print(_glass_panel(
        results_table,
        title="[glass.cyan]⚙[/glass.cyan]  [bold grey85]Capability Check[/bold grey85]",
        border_style="glass.border",
        padding=(1, 2),
        width=92,
    ))
    console.print()

    # Summary pill
    if report.err_count:
        summary_style = "glass.rose"
        summary_icon = "✗"
        summary_text = (
            f"{report.err_count} blocker(s) — install the items above, then rerun "
            "[glass.cyan]predacore bootstrap[/glass.cyan]."
        )
    elif report.warn_count:
        summary_style = "glass.amber"
        summary_icon = "◇"
        summary_text = (
            f"{report.ok_count} ready, {report.warn_count} optional missing — "
            "PredaCore works; fill gaps for full power."
        )
    else:
        summary_style = "glass.mint"
        summary_icon = "✓"
        summary_text = "Every capability is ready. You're set for first message."

    console.print(Align.center(_glass_panel(
        Text.from_markup(
            f"  [{summary_style}]{summary_icon}[/{summary_style}]  {summary_text}\n"
            f"  [glass.text.dim]elapsed {report.elapsed:.1f}s  ·  "
            f"next: [glass.cyan]predacore setup[/glass.cyan] (if not configured) "
            f"or [glass.cyan]predacore[/glass.cyan] to start chatting[/glass.text.dim]"
        ),
        title=f"[{summary_style}]{summary_icon}[/{summary_style}]  [bold grey85]Bootstrap complete[/bold grey85]",
        border_style=summary_style,
        padding=(1, 2),
        width=84,
    )))
    console.print()


def _run_doctor(
    config_path: str | None = None, verbose: bool = False, profile: str | None = None
) -> None:
    """Run comprehensive system health diagnostics."""
    import platform
    import shutil
    import sys

    console.print()
    console.print("[heading]PredaCore System Health Check[/heading]")
    console.rule(style="dim")
    console.print()

    checks_passed = 0
    checks_failed = 0
    checks_warned = 0

    def ok(msg: str, detail: str = ""):
        nonlocal checks_passed
        checks_passed += 1
        console.print(f"  [success]OK[/success] {msg}")
        if detail and verbose:
            console.print(f"     [muted]{detail}[/muted]")

    def warn(msg: str, detail: str = ""):
        nonlocal checks_warned
        checks_warned += 1
        console.print(f"  [warning]WARN[/warning] {msg}")
        if detail:
            console.print(f"     [muted]{detail}[/muted]")

    def fail(msg: str, detail: str = ""):
        nonlocal checks_failed
        checks_failed += 1
        console.print(f"  [error]FAIL[/error] {msg}")
        if detail:
            console.print(f"     [muted]{detail}[/muted]")

    # ── System ──
    console.print("[bold]System[/bold]")
    ok(f"Python {sys.version.split()[0]}", f"Path: {sys.executable}")
    ok(f"Platform: {platform.system()} {platform.machine()}")

    # ── Docker ──
    console.print("\n[bold]Docker[/bold]")
    docker_path = shutil.which("docker")
    if docker_path:
        try:
            import subprocess

            result = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                ok(f"Docker engine v{result.stdout.strip()}", f"Path: {docker_path}")
            else:
                warn("Docker installed but not running", "Start Docker Desktop")
        except (OSError, subprocess.SubprocessError):
            warn("Docker installed but unreachable")
    else:
        warn("Docker not installed", "Install for sandbox code execution")

    # ── Playwright ──
    console.print("\n[bold]Browser Automation[/bold]")
    try:
        import playwright

        ok("Playwright installed")
    except ImportError:
        warn("Playwright not installed", "pip install playwright && playwright install")

    # ── Config ──
    console.print("\n[bold]Configuration[/bold]")
    try:
        from .config import load_config

        config = load_config(config_path, profile_override=profile)
        ok("Config loaded", f"Home: {config.home_dir}")
    except (OSError, ValueError, KeyError) as e:
        fail(f"Config load failed: {e}")
        config = None

    # ── LLM Providers ──
    console.print("\n[bold]LLM Providers[/bold]")
    found_llm = False

    # Check Gemini CLI (the default provider)
    gemini_cli_path = shutil.which("gemini")
    if gemini_cli_path:
        ok("Gemini CLI available (no API key needed)", f"Path: {gemini_cli_path}")
        found_llm = True
    elif verbose:
        console.print("  [muted]— Gemini CLI: not installed[/muted]")

    # Check API key providers
    env_keys = {
        "GEMINI_API_KEY": "Gemini API",
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
    }
    for key, name in env_keys.items():
        val = os.environ.get(key)
        if val:
            ok(f"{name} key configured", f"{key}=****configured")
            found_llm = True
        elif verbose:
            console.print(f"  [muted]— {name}: not set[/muted]")

    # Check Ollama
    ollama_path = shutil.which("ollama")
    if ollama_path:
        ok("Ollama available (local LLM)", f"Path: {ollama_path}")
        found_llm = True
    elif verbose:
        console.print("  [muted]— Ollama: not installed[/muted]")

    if not found_llm:
        warn(
            "No LLM provider configured",
            "Run: predacore setup"
        )

    # ── Channels ──
    console.print("\n[bold]Channels[/bold]")
    channel_configs = {
        "Telegram": "TELEGRAM_BOT_TOKEN",
        "Discord": "DISCORD_BOT_TOKEN",
    }
    for ch_name, env_var in channel_configs.items():
        if os.environ.get(env_var):
            ok(f"{ch_name} token configured")
        elif verbose:
            console.print(f"  [muted]— {ch_name}: not configured[/muted]")

    # ── Memory ──
    console.print("\n[bold]Memory & Storage[/bold]")
    try:
        from src.predacore.memory import UnifiedMemoryStore  # type: ignore
        ok("UnifiedMemoryStore available")
    except ImportError:
        try:
            from predacore._vendor.common.memory_service import MemoryService  # type: ignore
            ok("MemoryService (legacy) available")
        except ImportError:
            warn("Memory system not importable")

    # ── Plugins ──
    console.print("\n[bold]Plugins[/bold]")
    try:
        from .services.plugins import PluginRegistry

        ok("Plugin SDK available")
    except ImportError:
        warn("Plugin SDK not available")

    # ── Summary ──
    console.print()
    console.rule(style="dim")
    total = checks_passed + checks_failed + checks_warned
    parts = []
    if checks_passed:
        parts.append(f"[success]{checks_passed} passed[/success]")
    if checks_warned:
        parts.append(f"[warning]{checks_warned} warnings[/warning]")
    if checks_failed:
        parts.append(f"[error]{checks_failed} failed[/error]")
    console.print(f"  {', '.join(parts)} ({total} checks)")

    if checks_failed == 0:
        console.print("\n  [success]PredaCore is healthy![/success]")
    elif checks_failed <= 2:
        console.print(
            "\n  [warning]PredaCore has minor issues — check warnings above[/warning]"
        )
    else:
        console.print(
            "\n  [error]PredaCore has critical issues — fix errors above[/error]"
        )
    console.print()


def _run_start(args) -> None:
    """Start PredaCore in foreground or daemon mode."""
    from .config import load_config
    from .services.daemon import PredaCoreDaemon, PIDManager

    profile = "public_beast" if args.public else args.profile
    config = load_config(args.config, profile_override=profile)
    daemon = PredaCoreDaemon(config)
    pid_manager = PIDManager(str(Path(config.home_dir) / "predacore.pid"))

    # Fast pre-flight check in parent process to avoid misleading fork output.
    if pid_manager.is_running():
        existing_pid = pid_manager.read()
        console.print(f"[error]PredaCore is already running (PID {existing_pid})[/error]")
        console.print(
            f"   [muted]Run 'predacore stop' first, or remove {pid_manager.pid_path}[/muted]"
        )
        return

    if args.daemon and not args.foreground:
        # Background daemon mode — spawn a detached subprocess
        import subprocess
        import sys

        log_dir = Path(config.logs_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "daemon.log"

        # Spawn the daemon as a detached subprocess in foreground mode
        # (it handles its own logging via --foreground + file logging)
        # Use the same module path resolution as the current process
        # __name__ may be "__main__" when invoked via -m; fall back to known package path
        if __name__ == "__main__":
            module_name = "src.predacore.cli"
        else:
            module_name = __name__.rsplit(".", 1)[0] + ".cli"
        with open(log_file, "a") as log_fh:
            proc = subprocess.Popen(
                [sys.executable, "-m", module_name, "start",
                 "--daemon", "--foreground",
                 "--config", str(Path(config.home_dir) / "config.yaml")],
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ},
            )
        console.print("[success]Starting PredaCore daemon...[/success]")
        console.print(f"   [muted]Logs: {log_file}[/muted]")
        console.print(f"   [muted]PID file: {config.home_dir}/predacore.pid[/muted]")
        console.print("   [muted]Run `predacore status` to check[/muted]")
        console.print("   [muted]Run `predacore stop` to stop[/muted]")
        console.print(f"   [muted]PID: {proc.pid}[/muted]")
    else:
        # Foreground mode — set up file logging if running as daemon subprocess
        if args.daemon:
            log_dir = Path(config.logs_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "daemon.log"
            setup_logging(level="INFO", json_mode=True, log_file=str(log_file))
        else:
            console.print("[success]Starting PredaCore in foreground mode...[/success]")
            console.print("   [muted]Press Ctrl+C to stop[/muted]")
        asyncio.run(daemon.start())


def _run_stop(config_path: str | None = None) -> None:
    """Stop the PredaCore daemon."""
    from .config import load_config
    from .services.daemon import PIDManager, stop_daemon

    config = load_config(config_path)
    pid_manager = PIDManager(str(Path(config.home_dir) / "predacore.pid"))
    if not pid_manager.is_running():
        console.print("[warning]No running PredaCore daemon found[/warning]")
        return

    if stop_daemon(config):
        console.print("[success]PredaCore daemon stopped[/success]")
    else:
        current_pid = pid_manager.read()
        console.print("[error]Failed to stop PredaCore daemon gracefully[/error]")
        if current_pid:
            console.print(
                f"   [muted]PID {current_pid} is still running. "
                "Stop it manually and remove ~/.predacore/predacore.pid if needed.[/muted]"
            )


def _run_restart(config_path: str | None = None) -> None:
    """Restart the PredaCore daemon seamlessly."""
    console.print("[info]Restarting PredaCore...[/info]")
    _run_stop(config_path=config_path)
    time.sleep(2)

    # Build start command
    cmd = [sys.executable, "-m", "src.predacore.cli", "start", "--daemon"]
    if config_path:
        cmd.extend(["--config", config_path])

    # Start new daemon (detached)
    _subprocess.Popen(
        cmd,
        stdout=_subprocess.DEVNULL,
        stderr=_subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(3)

    # Verify
    from .config import load_config
    from .services.daemon import PIDManager
    config = load_config(config_path)
    pid_manager = PIDManager(str(Path(config.home_dir) / "predacore.pid"))
    if pid_manager.is_running():
        console.print(f"[success]PredaCore restarted (PID {pid_manager.read()})[/success]")
    else:
        console.print("[error]Restart failed — check logs[/error]")


def _run_install(config_path: str | None = None) -> None:
    """Install PredaCore as a macOS launchd service."""
    import platform

    if platform.system() != "Darwin":
        console.print("[warning]launchd is macOS-only. Use systemd on Linux.[/warning]")
        return

    from .config import load_config
    from .services.daemon import install_launchd

    config = load_config(config_path)
    plist_path = install_launchd(config)

    console.print("[success]Installed launchd service[/success]")
    console.print(f"   [muted]Plist: {plist_path}[/muted]")
    console.print()
    console.print("To enable auto-start on login:")
    console.print(f"  [banner]launchctl load {plist_path}[/banner]")
    console.print()
    console.print("To disable:")
    console.print(f"  [banner]launchctl unload {plist_path}[/banner]")


def _run_logs(lines: int = 50, follow: bool = False, config_path: str | None = None) -> None:
    """Tail the daemon log file."""
    from .config import load_config

    # Clamp lines to a safe positive range
    lines = max(1, min(lines, 10000))

    config = load_config(config_path)
    log_file = Path(config.logs_dir) / "daemon.log"

    if not log_file.exists():
        console.print(
            "[warning]No daemon log found. Has the daemon been started?[/warning]"
        )
        console.print(f"   [muted]Expected: {log_file}[/muted]")
        return

    import subprocess as _sp

    if follow:
        # Use tail -f for live following
        _sp.run(["tail", "-f", "-n", str(lines), str(log_file)])
    else:
        # Just show last N lines
        _sp.run(["tail", "-n", str(lines), str(log_file)])


async def _run_eval(
    suite: str, config_path: str | None = None, threshold: float = 0.7
) -> None:
    """Run evaluation benchmarks."""
    from .config import load_config

    load_config(config_path)

    console.print(
        f"[banner]Running {suite} evaluation (threshold={threshold})...[/banner]"
    )

    if suite == "agent":
        from .evals.ale_bench import AgentEvalReport, AgentEvalRunner

        runner = AgentEvalRunner(pass_threshold=threshold)
        report: AgentEvalReport = await runner.run_default()
        console.print("\n[success]Agent Evaluation Results[/success]")
        console.print(f"  Pass rate: {report.pass_rate:.1%}  "
                      f"({report.passed}/{report.total} scenarios)")
        console.print(f"  Total latency: {report.total_latency_ms:.0f}ms")
        for cat, stats in report.by_category().items():
            console.print(
                f"  {cat}: {stats['passed']}/{stats['total']} "
                f"(avg score {stats['avg_score']:.2f})"
            )
    elif suite == "code":
        from .evals.swe_bench import EvalReport, EvalRunner

        runner = EvalRunner(pass_threshold=threshold)
        report: EvalReport = await runner.run_default()
        console.print("\n[success]Code Evaluation Results[/success]")
        console.print(f"  Pass rate: {report.pass_rate:.1%}  "
                      f"({report.passed}/{report.total} tasks)")
        console.print(f"  Avg score: {report.avg_score:.2f}")
        console.print(f"  Total latency: {report.total_latency_ms:.0f}ms")
    elif suite == "load":
        from .evals.load_test import LoadTestRunner

        runner = LoadTestRunner()
        report = await runner.run_default()
        console.print("\n[success]Load Test Results[/success]")
        console.print(f"  Total requests: {report.total_requests}  "
                      f"({report.successful} ok, {report.failed} err)")
        console.print(
            f"  p50: {report.latency_percentile(50):.0f}ms  "
            f"p95: {report.latency_percentile(95):.0f}ms  "
            f"p99: {report.latency_percentile(99):.0f}ms"
        )
        console.print(f"  Throughput: {report.requests_per_second:.1f} req/s")

    console.print("\n[muted]Done.[/muted]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        sys.exit(0)
    except Exception as exc:
        logging.getLogger(__name__).error("Fatal error: %s", exc, exc_info=True)
        print(f"\n[Fatal error] {exc}\nCheck logs for details.", file=sys.stderr)
        sys.exit(1)
