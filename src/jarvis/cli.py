"""
JARVIS CLI — Terminal interface for Project Prometheus.

Commands:
    prometheus chat [--session ID]  — Interactive chat in terminal
    prometheus start [--daemon]     — Start JARVIS (foreground or daemon)
    prometheus setup                — Guided onboarding wizard
    prometheus auth                 — Set up a dedicated Claude account for JARVIS
    prometheus status               — Show JARVIS status
    prometheus sessions             — List recent sessions
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
import threading
import time
import tty
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

try:
    from jarvis._vendor.common.logging_config import setup_logging
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
        "jarvis": "bold bright_cyan",
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


class GenerationController:
    """Controls the lifecycle of a single LLM generation — enables Esc to stop."""

    def __init__(self):
        self._cancel = asyncio.Event()
        self._partial_response = ""
        self._is_generating = False

    def reset(self):
        self._cancel.clear()
        self._partial_response = ""
        self._is_generating = False

    def cancel(self):
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def accumulate(self, token: str):
        self._partial_response += token


# ── Smart Completer ───────────────────────────────────────────────


class JARVISCompleter(Completer):
    """Context-aware completer: commands after /, models after /model."""

    COMMANDS = [
        "/exit", "/quit", "/status", "/tools", "/sessions",
        "/clear", "/help", "/model", "/compact",
        "/retry", "/copy", "/new", "/cost", "/history",
    ]

    def __init__(self, get_models_fn: Callable[[], list[str]] | None = None):
        self._get_models = get_models_fn

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/model "):
            prefix = text[7:]
            if self._get_models:
                for model in self._get_models():
                    if model.lower().startswith(prefix.lower()):
                        yield Completion(model, start_position=-len(prefix))
        elif text.startswith("/"):
            for cmd in self.COMMANDS:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))


# ── Status Bar ────────────────────────────────────────────────────


class StatusBar:
    """Live-updating bottom toolbar for prompt_toolkit."""

    def __init__(self):
        self.model = ""
        self.provider = ""
        self.trust = ""
        self.session_id = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.turn_count = 0

    def __call__(self):
        parts = []
        if self.provider:
            model_label = self.model or "auto"
            parts.append(f"<b>{self.provider}</b>/{model_label}")
        if self.trust:
            parts.append(f"trust:{self.trust}")
        parts.append(
            f"\u25b2{self._fmt(self.prompt_tokens)} "
            f"\u25bc{self._fmt(self.completion_tokens)}"
        )
        if self.turn_count:
            parts.append(f"turns:{self.turn_count}")
        if self.session_id:
            parts.append(f"session:{self.session_id[:8]}")
        return HTML(" <ansigray>\u2502</ansigray> ".join(parts))

    @staticmethod
    def _fmt(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1000:
            return f"{n / 1000:.1f}k"
        return str(n)


# ── Escape Key Listener ──────────────────────────────────────────


async def _listen_for_escape(gen_ctrl: GenerationController):
    """Non-blocking stdin listener for Esc key during generation.

    Uses os.read() on the raw fd to avoid conflicts with prompt_toolkit's
    buffered TextIOWrapper on sys.stdin.
    """
    loop = asyncio.get_running_loop()
    fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
    except termios.error:
        return  # Not a real terminal (e.g. piped input)
    try:
        tty.setcbreak(fd)  # cbreak keeps output visible (unlike raw)
        while not gen_ctrl.cancelled and gen_ctrl._is_generating:
            ready = await loop.run_in_executor(
                None, lambda: select.select([fd], [], [], 0.15)[0]
            )
            if ready:
                b = os.read(fd, 1)
                if not b:
                    continue
                ch = b[0]
                if ch == 0x1b:  # Escape — graceful stop
                    gen_ctrl.cancel()
                    break
                elif ch == 0x03:  # Ctrl+C — hard cancel
                    gen_ctrl.cancel()
                    break
    except (OSError, ValueError):
        pass  # stdin closed or not a tty
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except termios.error:
            pass


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


# ── Animated Banner ───────────────────────────────────────────────


def _animated_banner():
    """Print the fullwidth `✦ ＪＡＲＶＩＳ ✦` banner with a gentle 2-line reveal.

    Reuses `_banner()` content and prints it line-by-line with a small
    delay between lines. No cursor manipulation — keeps Rich happy and
    works consistently whether stdout is a terminal or a pipe.
    """
    content = _banner()
    for line in content.split("\n"):
        if line.strip():
            console.print(line)
            time.sleep(0.08)
        else:
            console.print()
    time.sleep(0.05)


# ── LiveChatRenderer ────────────────────────────────────────────────


class LiveChatRenderer:
    """
    Rich-powered UI renderer — shows real-time feedback during JARVIS processing.

    Displays thinking spinners, tool call visibility, result previews,
    and a stats footer after each response.
    """

    TOOL_ICONS = {
        "run_command": "💻",
        "read_file": "📄",
        "write_file": "✏️",
        "list_directory": "📁",
        "web_search": "🌐",
        "web_fetch": "🌐",
        "web_scrape": "🔍",
        "strategic_plan": "🗺️",
        "multi_agent": "🤖",
        "speak": "🔊",
        "desktop_control": "🖥️",
        "memory_store": "🧠",
        "memory_recall": "🧠",
        "git_context": "📊",
        "git_diff_summary": "📝",
        "git_commit_suggest": "💬",
    }

    def __init__(self, name: str = "JARVIS", status_bar: StatusBar | None = None):
        self.name = name
        self._spinner_active = False
        self._spinner_thread: threading.Thread | None = None
        self._stop_spinner = threading.Event()
        self._tool_count = 0
        self._current_line = ""
        self._status_bar = status_bar

    def _clear_line(self):
        """Clear the current terminal line."""
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _start_spinner(self, message: str):
        """Start an animated spinner with a message."""
        self._stop_spinner.clear()
        self._spinner_active = True

        frames = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]

        def _spin():
            i = 0
            while not self._stop_spinner.is_set():
                frame = frames[i % len(frames)]
                self._clear_line()
                color = f"\033[38;5;{39 + (i % 6)}m"  # Cycling blue shades
                sys.stdout.write(f"  {color}{frame}\033[0m \033[2m{message}\033[0m")
                sys.stdout.flush()
                i += 1
                self._stop_spinner.wait(0.06)
            self._clear_line()

        self._spinner_thread = threading.Thread(target=_spin, daemon=True)
        self._spinner_thread.start()

    def _stop_spinner_now(self):
        """Stop the spinner."""
        if self._spinner_active:
            self._stop_spinner.set()
            if self._spinner_thread:
                self._spinner_thread.join(timeout=0.5)
            self._spinner_active = False
            self._clear_line()

    def on_event(self, event_type: str, data: dict[str, Any]):
        """Handle a live UI event from the agent loop."""
        if event_type == "thinking":
            iteration = data.get("iteration", 1)
            if iteration == 1:
                self._tool_count = 0
                self._start_spinner("Thinking...")
            else:
                self._stop_spinner_now()
                self._start_spinner("Composing response...")

        elif event_type == "tool_start":
            self._stop_spinner_now()
            self._tool_count += 1
            name = data.get("name", "unknown")
            args = data.get("args", {})
            icon = self.TOOL_ICONS.get(name, "\u25c6")  # ◆ fallback

            # Format args preview in dim glass text
            args_preview = ""
            if args:
                args_str = json.dumps(args, default=str)
                if len(args_str) > 70:
                    args_str = args_str[:67] + "..."
                args_preview = f" [glass.text.dim]{args_str}[/glass.text.dim]"

            console.print(
                f"  [glass.violet]\u2502[/glass.violet]  {icon}  "
                f"[bold #C4B5FD]{name}[/bold #C4B5FD]{args_preview}"
            )
            self._start_spinner(f"Running {name}...")

        elif event_type == "tool_end":
            self._stop_spinner_now()
            name = data.get("name", "unknown")
            duration = data.get("duration_ms", 0)
            result_preview = data.get("result", "")
            is_error = data.get("error", False) or "error" in result_preview.lower()[:40]
            result_preview = result_preview.replace("\n", " ").strip()
            if len(result_preview) > 80:
                result_preview = result_preview[:77] + "..."

            arrow_color = "glass.rose" if is_error else "glass.mint"
            console.print(
                f"  [glass.violet]\u2502[/glass.violet]    "
                f"[{arrow_color}]\u21b3[/{arrow_color}]  "
                f"[glass.text.dim]{result_preview}[/glass.text.dim]  "
                f"[glass.text.dim]({duration:.0f}ms)[/glass.text.dim]"
            )

        elif event_type == "response":
            self._stop_spinner_now()
            content = data.get("content", "")
            tokens = data.get("tokens", 0)
            tools_used = data.get("tools_used", 0)
            elapsed = data.get("elapsed_s", 0)
            provider = data.get("provider", "")
            usage = data.get("usage", {})
            streamed = data.get("streamed", False)

            # Update status bar with token counts
            if self._status_bar:
                self._status_bar.completion_tokens += tokens
                self._status_bar.prompt_tokens += usage.get("prompt_tokens", 0)
                self._status_bar.turn_count += 1

            # Build the stats footer as a Rich Text — used as Panel subtitle
            # when we render, or as a standalone footer line when we don't.
            parts = []
            if elapsed:
                parts.append(f"[#7DD3FC]{elapsed:.1f}s[/#7DD3FC]")
            if tokens:
                parts.append(f"[#86EFAC]{tokens}[/#86EFAC] [glass.text.dim]tok[/glass.text.dim]")
            if tools_used:
                tool_word = "tool" if tools_used == 1 else "tools"
                parts.append(f"[#C4B5FD]{tools_used}[/#C4B5FD] [glass.text.dim]{tool_word}[/glass.text.dim]")
            if provider:
                parts.append(f"[glass.text.dim]{provider}[/glass.text.dim]")
            cost = data.get("cost_usd", 0)
            if cost:
                parts.append(f"[glass.text.dim]${cost:.4f}[/glass.text.dim]")
            footer = " [glass.text.dim]\u00b7[/glass.text.dim] ".join(parts) if parts else ""

            if streamed:
                # Provider natively streamed tokens to stdout — don't re-render.
                # Just drop a soft footer line below the live stream.
                sys.stdout.write("\n")
                sys.stdout.flush()
                if footer:
                    console.print(f"  [glass.cyan]\u2570\u2500[/glass.cyan] {footer}")
                console.print()
            else:
                # Non-streaming provider — render the full content inside a
                # rounded glass panel with a gradient title and a stats subtitle
                # on the bottom border.
                console.print()
                try:
                    body = Markdown(content)
                except (ValueError, TypeError):
                    body = Text(content, style="glass.text")

                title_markup = (
                    "[#F9A8D4]\u25c6[/#F9A8D4]  "
                    f"[bold #7DD3FC]{self.name}[/bold #7DD3FC]"
                )
                subtitle_markup = footer or None

                panel = Panel(
                    body,
                    title=title_markup,
                    subtitle=subtitle_markup,
                    border_style="glass.cyan",
                    box=box.ROUNDED,
                    padding=(1, 2),
                    title_align="left",
                    subtitle_align="right",
                )
                console.print(panel)
                console.print()


def _banner() -> str:
    """Glass banner: `✦ ＪＡＲＶＩＳ ✦` in fullwidth Unicode + subtitle.

    Same 2-line layout as the original `✦ JARVIS ✦` but the title is
    rendered in fullwidth characters (each ~2 cells wide in a monospace
    terminal) so it visually doubles in size without ASCII art. Letters
    get the 6-color gradient across J-A-R-V-I-S, sparkles flank both
    sides, italic violet subtitle on line 2.
    """
    pink   = "#F9A8D4"
    cyan   = "#7DD3FC"
    violet = "#C4B5FD"

    # Fullwidth Unicode — each char is ~2 terminal cells wide
    fullwidth = ["Ｊ", "Ａ", "Ｒ", "Ｖ", "Ｉ", "Ｓ"]
    grad = ["#F9A8D4", "#DDA0DD", "#C4B5FD", "#A5B4FC", "#8AB4F8", "#7DD3FC"]

    title = "".join(
        f"[bold {grad[i]}]{fullwidth[i]}[/bold {grad[i]}]"
        for i in range(len(fullwidth))
    )

    line1 = f"  [{pink}]✦[/{pink}]  {title}  [{cyan}]✦[/{cyan}]"
    line2 = f"  [italic {violet}]your self-evolving AI agent[/italic {violet}]"

    return "\n" + line1 + "\n" + line2 + "\n"


def _autoload_env_for_cli(config_path: str | None = None) -> Path | None:
    """
    Auto-load `.env` files for CLI usage.

    Loads ALL found files (override=False means first value wins):
      1) ~/.prometheus/.env          — API keys (always loaded first = highest priority)
      2) JARVIS_ENV_FILE         — explicit override
      3) .env next to --config   — project-specific
      4) cwd/.env                — local overrides
      5) repository-root/.env    — repo defaults
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    candidates = []

    # ~/.prometheus/.env is ALWAYS first — single source of truth for keys
    candidates.append(Path.home() / ".prometheus" / ".env")

    explicit = os.getenv("JARVIS_ENV_FILE", "").strip()
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


async def _run_chat(
    session_id: str | None = None,
    config_path: str | None = None,
    profile: str | None = None,
) -> None:
    """Run interactive chat session with prompt_toolkit, streaming, and Esc-to-stop."""
    from .config import load_config
    from .core import JARVISCore
    from .gateway import Gateway, IncomingMessage

    # Load config
    config = load_config(config_path, profile_override=profile)
    core = JARVISCore(config)
    gateway = Gateway(config, core.process)
    # Share core's OutcomeStore so gateway can record failures without a second DB connection
    if core._outcome_store is not None:
        gateway._outcome_store = core._outcome_store

    # Fresh session unless explicitly resuming
    if not session_id:
        session_id = f"cli-{uuid.uuid4().hex[:8]}"

    # Animated banner
    _animated_banner()

    # Build the glass-style Session Info panel — color-coded, rounded, calm.
    info = Table(show_header=False, box=None, padding=(0, 2), expand=False)
    info.add_column("key", style="glass.text.muted", width=10, justify="right")
    info.add_column("value", style="glass.text")

    trust_style_map = {"yolo": "#FCA5A5", "normal": "#7DD3FC", "paranoid": "#86EFAC"}
    trust_style = trust_style_map.get(config.security.trust_level, "glass.text")

    info.add_row("Provider", Text(config.llm.provider, style="bold #7DD3FC"))
    info.add_row("Model",    Text(config.llm.model or "auto", style="#C4B5FD"))
    info.add_row("Trust",    Text(config.security.trust_level, style=f"bold {trust_style}"))
    info.add_row("Tools",    Text.assemble(
        (f"{len(core.get_tool_list())}", "bold #86EFAC"),
        (" available", "glass.text.dim"),
    ))
    info.add_row("Session",  Text(session_id, style="glass.text.dim"))

    console.print(_glass_panel(
        info,
        title="[#F9A8D4]✦[/#F9A8D4]  [bold grey85]Session[/bold grey85]  [#7DD3FC]✦[/#7DD3FC]",
        border_style="glass.border",
        padding=(1, 2),
        width=64,
    ))

    console.print()
    console.print(
        "  [glass.text.dim]Enter[/glass.text.dim] [glass.text.muted]send[/glass.text.muted]  "
        "[glass.text.dim]\u00b7[/glass.text.dim]  "
        "[glass.text.dim]Alt+Enter[/glass.text.dim] [glass.text.muted]newline[/glass.text.muted]  "
        "[glass.text.dim]\u00b7[/glass.text.dim]  "
        "[glass.text.dim]Tab[/glass.text.dim] [glass.text.muted]complete[/glass.text.muted]  "
        "[glass.text.dim]\u00b7[/glass.text.dim]  "
        "[glass.text.dim]Esc[/glass.text.dim] [glass.text.muted]stop[/glass.text.muted]"
    )
    console.print(
        "  [glass.text.dim]Commands:[/glass.text.dim]  "
        "[glass.cyan]/help[/glass.cyan]  [glass.cyan]/status[/glass.cyan]  "
        "[glass.cyan]/model[/glass.cyan]  [glass.cyan]/tools[/glass.cyan]  "
        "[glass.cyan]/sessions[/glass.cyan]  [glass.cyan]/new[/glass.cyan]  "
        "[glass.cyan]/retry[/glass.cyan]  [glass.cyan]/copy[/glass.cyan]  "
        "[glass.cyan]/cost[/glass.cyan]  [glass.cyan]/clear[/glass.cyan]  "
        "[glass.cyan]/exit[/glass.cyan]"
    )
    console.print()
    console.print(Rule(style="glass.border.soft"))
    console.print()

    # ── Status bar ──
    status_bar = StatusBar()
    status_bar.provider = config.llm.provider
    status_bar.model = config.llm.model or "auto"
    status_bar.trust = config.security.trust_level
    status_bar.session_id = session_id

    # ── State ──
    user_id = os.getenv("USER", "default")
    renderer = LiveChatRenderer(name=config.name, status_bar=status_bar)
    gen_ctrl = GenerationController()
    last_user_message = ""
    last_response_text = ""
    _tool_decisions: dict[str, bool] = {}  # per-session tool allow/deny

    # ── prompt_toolkit setup ──
    kb = KeyBindings()

    @kb.add("escape", "enter")  # Alt+Enter inserts newline
    def _newline(event):
        event.current_buffer.insert_text("\n")

    @kb.add("c-l")  # Ctrl+L clears screen
    def _clear_screen(event):
        event.app.renderer.clear()

    history_path = Path.home() / ".prometheus" / "chat_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    completer = JARVISCompleter()

    prompt_session = PromptSession(
        history=FileHistory(str(history_path)),
        multiline=False,
        key_bindings=kb,
        auto_suggest=AutoSuggestFromHistory(),
        enable_history_search=True,
        mouse_support=False,
        complete_while_typing=True,
        completer=completer,
        bottom_toolbar=status_bar,
    )

    # ── Tool confirmation callback ──
    async def _confirm_tool(tool_name: str, arguments: dict, ctx=None) -> bool:
        """Rich approval prompt for tool execution."""
        if config.security.trust_level == "yolo":
            return True
        if tool_name in _tool_decisions:
            return _tool_decisions[tool_name]

        icon = LiveChatRenderer.TOOL_ICONS.get(tool_name, "\U0001f527")
        args_preview = json.dumps(arguments, default=str)
        if len(args_preview) > 100:
            args_preview = args_preview[:97] + "..."

        risk_color = "yellow"
        risk_reason = ""
        if ctx:
            risk_level = getattr(ctx, "risk_level", "medium")
            risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(
                risk_level, "yellow"
            )
            risk_reason = getattr(ctx, "risk_reason", "")

        console.print(f"\n  {icon} [bold]{tool_name}[/bold]")
        console.print(f"    [muted]{args_preview}[/muted]")
        if risk_reason:
            console.print(f"    [{risk_color}]{risk_reason}[/{risk_color}]")

        try:
            answer = await prompt_session.prompt_async(
                HTML("  <ansiyellow>Allow?</ansiyellow> <ansigray>[y/N/always] </ansigray>"),
            )
        except (KeyboardInterrupt, EOFError):
            return False

        answer = answer.strip().lower()
        if answer in ("a", "always"):
            _tool_decisions[tool_name] = True
            return True
        approved = answer in ("y", "yes")
        return approved

    # ── Main loop ──
    _last_ctrl_c: float = 0.0
    while True:
        try:
            user_input = await prompt_session.prompt_async(
                HTML("<ansigreen><b>You</b></ansigreen> <ansigray>\u25b8 </ansigray>"),
                prompt_continuation="  ... ",
            )
            _last_ctrl_c = 0.0  # Reset on successful input
        except KeyboardInterrupt:
            now = time.time()
            if now - _last_ctrl_c < 1.5:
                # Double Ctrl+C — exit
                renderer._stop_spinner_now()
                console.print("\n[jarvis]Goodbye![/jarvis]")
                break
            _last_ctrl_c = now
            console.print("\n  [muted]Press Ctrl+C again to exit, or Ctrl+D[/muted]")
            continue
        except EOFError:
            renderer._stop_spinner_now()
            console.print("\n[jarvis]Goodbye![/jarvis]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            handled = await _handle_command(
                user_input, core, gateway, session_id, status_bar,
                last_user_message, last_response_text, prompt_session,
            )
            if handled == "exit":
                break
            if handled == "retry" and last_user_message:
                user_input = last_user_message
            elif isinstance(handled, str) and handled.startswith("resend:"):
                user_input = handled[7:]
            elif isinstance(handled, str) and handled.startswith("new_session:"):
                session_id = handled[12:]
                continue
            else:
                continue

        last_user_message = user_input
        print()  # Blank line before response

        # ── Generate with Esc-to-stop ──
        gen_ctrl.reset()
        gen_ctrl._is_generating = True

        try:
            incoming = IncomingMessage(
                channel="cli",
                user_id=user_id,
                text=user_input,
                session_id=session_id,
            )

            # Stream callback — prints tokens as they arrive
            async def _stream_token(token: str):
                if gen_ctrl.cancelled:
                    return
                # Stop spinner when first token arrives
                renderer._stop_spinner_now()
                # Print token without newline for streaming effect
                sys.stdout.write(token)
                sys.stdout.flush()
                gen_ctrl.accumulate(token)

            gen_task = asyncio.create_task(
                gateway.handle_message(
                    incoming,
                    event_fn=renderer.on_event,
                    stream_fn=_stream_token,
                    confirm_fn=_confirm_tool,
                )
            )
            esc_task = asyncio.create_task(_listen_for_escape(gen_ctrl))

            done, pending = await asyncio.wait(
                {gen_task, esc_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            if gen_ctrl.cancelled:
                # gen_task already cancelled in the pending loop above;
                # just ensure spinner stops and partial response is shown
                renderer._stop_spinner_now()
                if gen_ctrl._partial_response:
                    console.print()
                    console.print(Markdown(gen_ctrl._partial_response))
                console.print("  [muted]\u2500\u2500\u2500 stopped \u2500\u2500\u2500[/muted]\n")
                last_response_text = gen_ctrl._partial_response
            else:
                response = gen_task.result()
                session_id = response.session_id
                last_response_text = response.text
                status_bar.session_id = session_id

        except KeyboardInterrupt:
            renderer._stop_spinner_now()
            console.print("\n  [warning]Interrupted.[/warning]\n")
        except Exception as e:
            renderer._stop_spinner_now()
            err_msg = str(e)
            if len(err_msg) > 150:
                err_msg = err_msg[:150] + "..."
            console.print(f"\n  [error]Error: {err_msg}[/error]\n")
            logger.debug("Chat error detail: %s", e, exc_info=True)
        finally:
            gen_ctrl._is_generating = False

    # Cleanup
    await gateway.stop()


def _build_model_options() -> list[tuple[str, str]]:
    """Build list of (provider, model) tuples for all available models."""
    from .llm_providers.openai import PROVIDER_ENDPOINTS
    import shutil

    available: list[tuple[str, str]] = []

    # Gemini CLI models (free via 'gemini auth login')
    if shutil.which("gemini"):
        for m in ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-flash-preview"]:
            available.append(("gemini-cli", m))

    # SambaNova models (free tier)
    samba_models = [
        "DeepSeek-V3.1", "Qwen3-235B", "Llama-4-Maverick-17B-128E-Instruct",
        "Meta-Llama-3.3-70B-Instruct", "DeepSeek-R1",
    ]
    for m in samba_models:
        available.append(("sambanova", m))

    # Other providers with keys configured
    for pname, ep in PROVIDER_ENDPOINTS.items():
        if pname == "sambanova":
            continue
        if os.getenv(ep["env_key"], ""):
            available.append((pname, ep["default_model"]))

    # Gemini / Anthropic
    if os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""):
        available.append(("gemini", "gemini-2.5-flash"))
    if os.getenv("ANTHROPIC_API_KEY", ""):
        available.append(("anthropic", "claude-sonnet-4-6"))

    return available


# ── Interactive Command Selector ─────────────────────────────────

# Command definitions: (name, description, icon)
_COMMAND_DEFS: list[tuple[str, str, str]] = [
    ("/model",    "Switch LLM model/provider",    "\U0001f9e0"),
    ("/status",   "Show JARVIS status + tokens",   "\U0001f4ca"),
    ("/tools",    "List available tools",           "\U0001f527"),
    ("/sessions", "List recent sessions",           "\U0001f4c1"),
    ("/new",      "Start a fresh session",          "\u2728"),
    ("/retry",    "Resend last message",            "\U0001f504"),
    ("/copy",     "Copy last response to clipboard","\U0001f4cb"),
    ("/cost",     "Show token usage this session",  "\U0001f4b0"),
    ("/history",  "Show conversation history",      "\U0001f4dc"),
    ("/compact",  "Compress conversation context",  "\U0001f5dc\ufe0f"),
    ("/clear",    "Clear screen",                   "\U0001f9f9"),
    ("/help",     "Show all commands & shortcuts",  "\u2753"),
    ("/exit",     "Exit chat",                      "\U0001f44b"),
]


def _select_command() -> str | None:
    """Show interactive command picker with arrow keys."""
    options = [
        (f"{icon}  {name}", desc, name)
        for name, desc, icon in _COMMAND_DEFS
    ]
    return _interactive_select("Commands", options)


async def _handle_command(
    cmd: str,
    core,
    gateway,
    session_id: str = "",
    status_bar: StatusBar | None = None,
    last_user_message: str = "",
    last_response_text: str = "",
    prompt_session: PromptSession | None = None,
) -> str | None:
    """Handle CLI slash commands.

    Returns:
        'exit'       — quit the chat loop
        'retry'      — resend last user message
        'resend:...' — resend with edited text
        'new_session:ID' — switch to new session
        None         — command handled, continue loop
    """
    parts = cmd.strip().split(None, 1)
    base = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    # Bare "/" — open interactive command picker
    if base == "/":
        selected = _select_command()
        if selected:
            return await _handle_command(
                selected, core, gateway, session_id, status_bar,
                last_user_message, last_response_text, prompt_session,
            )
        console.print("  [muted]Cancelled[/muted]\n")
        return None

    if base in ("/exit", "/quit", "/q"):
        console.print("\n[jarvis]Goodbye![/jarvis]")
        return "exit"

    elif base == "/status":
        status = core.get_status()
        stats = gateway.get_stats()

        tbl = Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column("key", style="muted", width=14)
        tbl.add_column("value", style="stat.value")
        tbl.add_row("Provider", status["provider"])
        tbl.add_row("Model", status["model"])
        tbl.add_row("Trust", status["trust_level"])
        tbl.add_row("Tools", str(status.get("tools_count", len(status["tools"]))))
        persistent = status.get("persistent_memory_items")
        session_mem = status.get("session_memory_items", status["memory_items"])
        mem_str = f"session={session_mem}"
        if persistent is not None:
            mem_str += f", persistent={persistent}"
        tbl.add_row("Memory", mem_str)
        tbl.add_row(
            "Messages", f"\u2193{stats['messages_received']} \u2191{stats['messages_sent']}"
        )
        tbl.add_row("Errors", str(stats["errors"]))
        if status_bar:
            tbl.add_row(
                "Tokens",
                f"\u25b2{status_bar.prompt_tokens:,} \u25bc{status_bar.completion_tokens:,}",
            )
            tbl.add_row("Turns", str(status_bar.turn_count))

        console.print()
        console.print(
            Panel(
                tbl,
                title="[heading]JARVIS Status[/heading]",
                border_style="cyan",
                width=50,
            )
        )
        console.print()

    elif base == "/tools":
        tools = core.get_tool_list()
        tbl = Table(
            show_header=True, header_style="bold cyan", box=None, padding=(0, 1)
        )
        tbl.add_column("#", style="muted", width=4)
        tbl.add_column("Tool", style="tool.name")
        for i, t in enumerate(tools, 1):
            icon = LiveChatRenderer.TOOL_ICONS.get(t, "\U0001f527")
            tbl.add_row(str(i), f"{icon} {t}")
        console.print()
        console.print(
            Panel(
                tbl,
                title="[heading]Available Tools[/heading]",
                border_style="cyan",
                width=50,
            )
        )
        console.print()

    elif base == "/sessions":
        sessions = gateway.session_store.list_sessions(limit=10)
        console.print()
        if sessions:
            tbl = Table(
                show_header=True, header_style="bold cyan", box=None, padding=(0, 1)
            )
            tbl.add_column("Session", style="stat.value", width=20)
            tbl.add_column("Title", style="muted")
            tbl.add_column("Age", style="muted", justify="right")
            for s in sessions:
                age = time.time() - s.updated_at
                age_str = (
                    f"{int(age / 60)}m ago" if age < 3600 else f"{int(age / 3600)}h ago"
                )
                tbl.add_row(s.session_id[:18], s.title[:35], age_str)
            console.print(
                Panel(
                    tbl,
                    title="[heading]Recent Sessions[/heading]",
                    border_style="cyan",
                    width=60,
                )
            )
        else:
            console.print("  [muted]No sessions yet[/muted]")
        console.print()

    elif base == "/clear":
        console.clear()

    # ── New commands ──────────────────────────────────────────────

    elif base == "/model":
        status = core.get_status()
        available = _build_model_options()
        current_val = f"{status['provider']}/{status['model']}"

        if not arg:
            # Interactive arrow-key selector
            options = [
                (mdl, prov, f"{prov}/{mdl}")
                for prov, mdl in available
            ]
            selected = _interactive_select(
                f"Switch Model  (current: {current_val})",
                options,
                current_value=current_val,
            )
            if selected and selected != current_val:
                provider, model = selected.split("/", 1)
                try:
                    core.set_model(provider=provider, model=model)
                    new_status = core.get_status()
                    if status_bar:
                        status_bar.provider = new_status["provider"]
                        status_bar.model = new_status["model"]
                    console.print(
                        f"  [success]\u2713 Switched to {new_status['provider']}/{new_status['model']}[/success]\n"
                    )
                except (ValueError, KeyError, OSError) as e:
                    console.print(f"  [error]Failed to switch model: {e}[/error]\n")
            elif selected:
                console.print(f"  [muted]Already using {current_val}[/muted]\n")
            else:
                console.print("  [muted]Cancelled[/muted]\n")
        else:
            # Direct: /model provider/model or /model name
            if "/" in arg:
                provider, model = arg.split("/", 1)
            else:
                # Try to find a matching model in available list
                provider = None
                model = arg
                for prov, mdl in available:
                    if mdl.lower() == arg.lower() or arg.lower() in mdl.lower():
                        provider = prov
                        model = mdl
                        break

            try:
                core.set_model(provider=provider, model=model)
                new_status = core.get_status()
                if status_bar:
                    status_bar.provider = new_status["provider"]
                    status_bar.model = new_status["model"]
                console.print(
                    f"  [success]\u2713 Switched to {new_status['provider']}/{new_status['model']}[/success]"
                )
            except (ValueError, KeyError, OSError) as e:
                console.print(f"  [error]Failed to switch model: {e}[/error]")
            console.print()

    elif base == "/retry":
        if last_user_message:
            console.print("  [muted]Retrying last message...[/muted]")
            return "retry"
        else:
            console.print("  [warning]No previous message to retry[/warning]")
            console.print()

    elif base == "/copy":
        if last_response_text:
            try:
                proc = _subprocess.run(
                    ["pbcopy"], input=last_response_text.encode(), check=True
                )
                console.print("  [success]Copied to clipboard[/success]")
            except (OSError, _subprocess.CalledProcessError):
                console.print("  [warning]pbcopy not available (macOS only)[/warning]")
        else:
            console.print("  [warning]No response to copy[/warning]")
        console.print()

    elif base == "/cost":
        console.print()
        if status_bar:
            total = status_bar.prompt_tokens + status_bar.completion_tokens
            console.print(f"  [bold]Prompt tokens:[/bold]     {status_bar.prompt_tokens:,}")
            console.print(f"  [bold]Completion tokens:[/bold] {status_bar.completion_tokens:,}")
            console.print(f"  [bold]Total:[/bold]             {total:,}")
            console.print(f"  [bold]Turns:[/bold]             {status_bar.turn_count}")
        else:
            console.print("  [muted]No token data available[/muted]")
        console.print()

    elif base == "/new":
        new_sid = f"cli-{uuid.uuid4().hex[:8]}"
        if status_bar:
            status_bar.session_id = new_sid
            status_bar.prompt_tokens = 0
            status_bar.completion_tokens = 0
            status_bar.turn_count = 0
        console.print(f"  [success]New session: {new_sid}[/success]")
        console.print()
        return f"new_session:{new_sid}"

    elif base == "/compact":
        console.print("  [muted]Compacting conversation history...[/muted]")
        return "resend:Summarize our entire conversation so far into a concise context block. Keep key decisions, code changes, and open items. Drop redundant detail."

    elif base == "/history":
        try:
            session = gateway.session_store.get(session_id)
            if session:
                msgs = session.get_llm_messages(max_messages=20)
                console.print()
                for msg in msgs:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        continue
                    if content.startswith("[Tool Result:") or content.startswith("[Calling tool:"):
                        continue
                    prefix = "You" if role == "user" else "JARVIS"
                    style = "user" if role == "user" else "jarvis"
                    text = content[:120] + ("..." if len(content) > 120 else "")
                    text = text.replace("\n", " ")
                    console.print(f"  [{style}]{prefix}[/{style}]: {text}")
                console.print()
            else:
                console.print("  [muted]No history for this session[/muted]\n")
        except (OSError, KeyError, ValueError):
            console.print("  [muted]Could not retrieve history[/muted]\n")

    elif base == "/help":
        tbl = Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column("cmd", style="bold cyan", width=12)
        tbl.add_column("desc", style="muted")
        tbl.add_row("/status", "Show JARVIS status + token usage")
        tbl.add_row("/model", "Show or switch model (/model Qwen3-235B)")
        tbl.add_row("/tools", "List available tools")
        tbl.add_row("/sessions", "List recent sessions")
        tbl.add_row("/new", "Start a fresh session")
        tbl.add_row("/retry", "Resend last message")
        tbl.add_row("/copy", "Copy last response to clipboard")
        tbl.add_row("/cost", "Show token usage this session")
        tbl.add_row("/history", "Show conversation history")
        tbl.add_row("/compact", "Compress conversation context")
        tbl.add_row("/clear", "Clear screen")
        tbl.add_row("/exit", "Exit chat")

        console.print()
        console.print(
            Panel(
                tbl, title="[heading]Commands[/heading]", border_style="cyan", width=55
            )
        )
        console.print()
        console.print("  [muted]Shortcuts: Enter=send \u00b7 Alt+Enter=newline \u00b7 Esc=stop generation[/muted]")
        console.print("  [muted]          Ctrl+R=search history \u00b7 Tab=complete \u00b7 Ctrl+L=clear[/muted]")
        console.print()

    else:
        console.print(f"  [warning]Unknown command: {base}[/warning]")
        console.print("  [muted]Type /help for available commands[/muted]")
        console.print()

    return None


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
    console.print(Align.center(_gradient_text("✦  PROMETHEUS  ✦", colors=[
        "#F9A8D4", "#F0A5D4", "#DDA0DD", "#C4B5FD", "#A5B4FC", "#8AB4F8", "#7DD3FC"
    ])))
    console.print(Align.center(Text("your self-evolving AI agent", style="glass.text.dim italic")))
    console.print()

    welcome_body = Text.assemble(
        ("You're about to set up your own AI agent. Takes about 90 seconds.\n", "glass.text"),
        ("Everything lives in ", "glass.text.muted"),
        ("~/.prometheus/", "glass.cyan"),
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
                f"[glass.text.dim]It will be saved to ~/.prometheus/.env with chmod 600.[/glass.text.dim]"
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
            console.print("    [glass.text.dim]Skipped. Add the token to ~/.prometheus/.env later to enable Telegram.[/glass.text.dim]")

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
            console.print("    [glass.text.dim]Skipped. Add the token to ~/.prometheus/.env later to enable Discord.[/glass.text.dim]")

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
            messages=[{"role": "user", "content": "Say 'JARVIS online' in exactly two words."}],
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
        console.print("      [glass.text.dim]Not fatal — fix the config later and run: [glass.cyan]prometheus doctor[/glass.cyan][/glass.text.dim]")

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
            ("prometheus start --daemon\n\n", "bold #7DD3FC"),
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
            ("prometheus chat\n\n", "bold #C4B5FD"),
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
    footer_table.add_row("prometheus doctor", "Check system + operator health")
    footer_table.add_row("prometheus status", "Show current daemon state")
    footer_table.add_row("prometheus stop",   "Gracefully stop the daemon")
    footer_table.add_row("prometheus setup",  "Re-run this wizard")
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
    """Show JARVIS status summary."""
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
                "JARVIS_ACTION_LEDGER_PATH",
                str(Path(config.logs_dir) / "openclaw_actions.jsonl"),
            )
            tbl.add_row("  Bridge Ledger", ledger_path)

        tbl.add_row(
            "Marketplace", "on" if config.launch.enable_plugin_marketplace else "off"
        )
        if config.launch.enable_plugin_marketplace:
            skills_dir = (
                os.getenv("JARVIS_OPENCLAW_SKILLS_DIR")
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
                title="[heading]JARVIS Status[/heading]",
                border_style="cyan",
                width=70,
            )
        )

    except (OSError, ValueError, KeyError) as e:
        console.print(f"  [error]Error loading config: {e}[/error]")
        console.print("  [muted]Run `prometheus setup` to create config[/muted]")
    console.print()


# ── Auth Command ─────────────────────────────────────────────────────


async def _run_jarvis_auth(action: str = "login") -> None:
    """Set up a dedicated Claude Code account for JARVIS.

    This creates a separate config directory (~/.prometheus/claude-config/)
    so JARVIS has its own Claude account and rate limits, independent
    from the user's CLI session.
    """
    import subprocess
    from pathlib import Path

    jarvis_config_dir = Path.home() / ".prometheus" / "claude-config"

    console.print(_banner())
    console.print("  [bold]JARVIS Dedicated Claude Account Setup[/bold]")
    console.print()

    if action == "status":
        if jarvis_config_dir.exists():
            # Check if there's a valid auth
            result = subprocess.run(
                ["claude", "auth", "status"],
                capture_output=True, text=True, timeout=10,
                env={**os.environ, "CLAUDE_CONFIG_DIR": str(jarvis_config_dir)},
            )
            if result.returncode == 0:
                console.print(f"  [success]JARVIS has its own Claude account[/success]")
                console.print(f"  Config: {jarvis_config_dir}")
                console.print(f"  {result.stdout.strip()}")
            else:
                console.print(f"  [warning]Config dir exists but no valid auth[/warning]")
                console.print(f"  Run: prometheus auth --jarvis")
        else:
            console.print("  [warning]No dedicated account set up[/warning]")
            console.print("  JARVIS shares rate limits with your CLI session.")
            console.print("  Run: prometheus auth --jarvis")
        console.print()
        return

    if action == "logout":
        if jarvis_config_dir.exists():
            import shutil
            shutil.rmtree(jarvis_config_dir, ignore_errors=True)
            console.print("  [success]JARVIS Claude account removed[/success]")
            console.print("  JARVIS will now share your CLI account.")
        else:
            console.print("  [muted]No dedicated account to remove[/muted]")
        console.print()
        return

    # Login — create isolated config dir and run claude login there
    jarvis_config_dir.mkdir(parents=True, exist_ok=True)

    console.print("  This will open a browser to log in a SEPARATE Claude account")
    console.print("  for JARVIS. Use a different account than your main CLI session")
    console.print("  so they don't share rate limits.")
    console.print()
    console.print(f"  [muted]Config dir: {jarvis_config_dir}[/muted]")
    console.print()

    try:
        import shutil
        claude_bin = shutil.which("claude")
        if not claude_bin:
            console.print("  [error]'claude' CLI not found. Install Claude Code first.[/error]")
            console.print()
            return

        env = {**os.environ, "CLAUDE_CONFIG_DIR": str(jarvis_config_dir)}

        console.print("  Launching claude login (interactive)...")
        console.print()

        # Use os.execvpe to replace this process with claude login.
        # This gives claude full TTY access for OAuth browser flow.
        os.execvpe(claude_bin, [claude_bin, "login"], env)
        # Note: os.execvpe never returns — process is replaced.
        # After login, user re-runs prometheus to continue.

    except FileNotFoundError:
        console.print("  [error]'claude' CLI not found. Install Claude Code first.[/error]")
    console.print()


# ── Main Entry Point ────────────────────────────────────────────────


def main() -> None:
    """Main CLI entry point."""
    import argparse

    cli_prog = Path(sys.argv[0]).name or "prometheus"
    if cli_prog in {"python", "python3", "python3.10", "python3.11", "python3.12"}:
        cli_prog = "prometheus"

    parser = argparse.ArgumentParser(
        prog=cli_prog,
        description="JARVIS — Project Prometheus AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  chat      Interactive terminal chat
  start     Start JARVIS (foreground or daemon)
  stop      Stop the JARVIS daemon
  install   Install as macOS launchd service
  logs      Tail daemon logs
  setup     Guided onboarding wizard
  auth      Set up dedicated Claude account for JARVIS
  status    Show JARVIS status
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
    start_parser = subparsers.add_parser("start", help="Start JARVIS")
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
    stop_parser = subparsers.add_parser("stop", help="Stop the JARVIS daemon")
    stop_parser.add_argument("--config", "-c", help="Path to config file")

    # Restart command
    restart_parser = subparsers.add_parser("restart", help="Restart the JARVIS daemon")
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

    # Auth command — sets up a dedicated Claude account for JARVIS
    auth_parser = subparsers.add_parser(
        "auth", help="Set up a dedicated Claude account for JARVIS"
    )
    auth_parser.add_argument(
        "action",
        nargs="?",
        default="login",
        choices=["login", "logout", "status"],
        help="Auth action (default: login)",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show JARVIS status")
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
    log_level = os.getenv("JARVIS_LOG_LEVEL", "WARNING").upper()
    json_logs = os.getenv("JARVIS_LOG_JSON", "").lower() in ("1", "true", "yes")
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
    elif args.command == "auth":
        asyncio.run(_run_jarvis_auth(action=args.action))
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
        parser.print_help()


def _run_doctor(
    config_path: str | None = None, verbose: bool = False, profile: str | None = None
) -> None:
    """Run comprehensive system health diagnostics."""
    import platform
    import shutil
    import sys

    console.print()
    console.print("[heading]JARVIS System Health Check[/heading]")
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
            "Run: prometheus setup"
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
        from src.jarvis.memory import UnifiedMemoryStore  # type: ignore
        ok("UnifiedMemoryStore available")
    except ImportError:
        try:
            from jarvis._vendor.common.memory_service import MemoryService  # type: ignore
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
        console.print("\n  [success]JARVIS is healthy![/success]")
    elif checks_failed <= 2:
        console.print(
            "\n  [warning]JARVIS has minor issues — check warnings above[/warning]"
        )
    else:
        console.print(
            "\n  [error]JARVIS has critical issues — fix errors above[/error]"
        )
    console.print()


def _run_start(args) -> None:
    """Start JARVIS in foreground or daemon mode."""
    from .config import load_config
    from .services.daemon import JARVISDaemon, PIDManager

    profile = "public_beast" if args.public else args.profile
    config = load_config(args.config, profile_override=profile)
    daemon = JARVISDaemon(config)
    pid_manager = PIDManager(str(Path(config.home_dir) / "jarvis.pid"))

    # Fast pre-flight check in parent process to avoid misleading fork output.
    if pid_manager.is_running():
        existing_pid = pid_manager.read()
        console.print(f"[error]JARVIS is already running (PID {existing_pid})[/error]")
        console.print(
            f"   [muted]Run 'prometheus stop' first, or remove {pid_manager.pid_path}[/muted]"
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
            module_name = "src.jarvis.cli"
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
        console.print("[success]Starting JARVIS daemon...[/success]")
        console.print(f"   [muted]Logs: {log_file}[/muted]")
        console.print(f"   [muted]PID file: {config.home_dir}/jarvis.pid[/muted]")
        console.print("   [muted]Run `prometheus status` to check[/muted]")
        console.print("   [muted]Run `prometheus stop` to stop[/muted]")
        console.print(f"   [muted]PID: {proc.pid}[/muted]")
    else:
        # Foreground mode — set up file logging if running as daemon subprocess
        if args.daemon:
            log_dir = Path(config.logs_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "daemon.log"
            setup_logging(level="INFO", json_mode=True, log_file=str(log_file))
        else:
            console.print("[success]Starting JARVIS in foreground mode...[/success]")
            console.print("   [muted]Press Ctrl+C to stop[/muted]")
        asyncio.run(daemon.start())


def _run_stop(config_path: str | None = None) -> None:
    """Stop the JARVIS daemon."""
    from .config import load_config
    from .services.daemon import PIDManager, stop_daemon

    config = load_config(config_path)
    pid_manager = PIDManager(str(Path(config.home_dir) / "jarvis.pid"))
    if not pid_manager.is_running():
        console.print("[warning]No running JARVIS daemon found[/warning]")
        return

    if stop_daemon(config):
        console.print("[success]JARVIS daemon stopped[/success]")
    else:
        current_pid = pid_manager.read()
        console.print("[error]Failed to stop JARVIS daemon gracefully[/error]")
        if current_pid:
            console.print(
                f"   [muted]PID {current_pid} is still running. "
                "Stop it manually and remove ~/.prometheus/jarvis.pid if needed.[/muted]"
            )


def _run_restart(config_path: str | None = None) -> None:
    """Restart the JARVIS daemon seamlessly."""
    console.print("[info]Restarting JARVIS...[/info]")
    _run_stop(config_path=config_path)
    time.sleep(2)

    # Build start command
    cmd = [sys.executable, "-m", "src.jarvis.cli", "start", "--daemon"]
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
    pid_manager = PIDManager(str(Path(config.home_dir) / "jarvis.pid"))
    if pid_manager.is_running():
        console.print(f"[success]JARVIS restarted (PID {pid_manager.read()})[/success]")
    else:
        console.print("[error]Restart failed — check logs[/error]")


def _run_install(config_path: str | None = None) -> None:
    """Install JARVIS as a macOS launchd service."""
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
        console.print(f"  Pass rate: {report.pass_rate:.1%}")
        console.print(f"  Total: {report.total_scenarios} scenarios")
        for cat, score in report.category_scores.items():
            console.print(f"  {cat}: {score:.2f}")
    elif suite == "code":
        from .evals.swe_bench import EvalReport, EvalRunner

        runner = EvalRunner(pass_threshold=threshold)
        report: EvalReport = await runner.run_default()
        console.print("\n[success]Code Evaluation Results[/success]")
        console.print(f"  Pass rate: {report.pass_rate:.1%}")
        console.print(f"  Total: {report.total_tasks} tasks")
    elif suite == "load":
        from .evals.load_test import LoadTestRunner

        runner = LoadTestRunner()
        report = await runner.run_default()
        console.print("\n[success]Load Test Results[/success]")
        console.print(f"  Total requests: {report.total_requests}")
        console.print(
            f"  p50: {report.p50_ms:.0f}ms  p95: {report.p95_ms:.0f}ms  p99: {report.p99_ms:.0f}ms"
        )
        console.print(f"  Throughput: {report.throughput_rps:.1f} req/s")

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
