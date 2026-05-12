"""
First-time setup — single game-style progress bar for model downloads.

When ``predacore start`` is invoked for the first time on a machine, the
core ML weights (embedder + reranker) aren't yet in ~/.cache/huggingface/.
This module shows a clean "Loading memory and browser..." screen with one
unified progress bar covering all required downloads.

Subsequent starts: this module runs ``needs_first_time_setup()`` which
returns False (everything cached), and the daemon boots silently.

Design choices (Wave 12.5):
  - **Single progress bar** — total bytes across all downloads, not
    per-model. Users don't need to see model internals.
  - **Opaque labels** — "Loading memory and browser..." not
    "Downloading Qwen3-Reranker-0.6B-q4". Game-loader UX.
  - **Defer the niche stuff** — TinyClick (canvas-vision, ~540MB) is
    only useful on Figma/Miro/canvas-app pages. Most users never hit
    one. Stays lazy-loaded on first canvas page.
  - **Daemon-mode safe** — TTY-detection bypasses the visible bar when
    running from launchd / systemd / non-interactive shells; downloads
    proceed via log lines instead.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Models we eagerly download on first start. TinyClick (vision) is
# deliberately NOT in this list — it's lazy on first canvas-app page.
# Chromium (Playwright) is handled separately via `playwright install`.
_ESSENTIAL_REPOS: dict[str, str] = {
    # repo_id  →  friendly stage label shown rotating on the progress bar.
    # Names are intentionally vague — users don't need to see model internals.
    "BAAI/bge-small-en-v1.5":      "Loading memory",
    "Qwen/Qwen3-Reranker-0.6B":    "Loading recall layer",
}

# Total bytes estimate for the essential set. Used to seed the progress
# bar before the HF API returns actual file sizes. ~1.4 GB conservative.
_ESTIMATED_TOTAL_BYTES = 1_400_000_000


def _hf_cache_root() -> Path:
    """Path to the HuggingFace hub cache root.

    Resolution order matches what ``huggingface_hub`` itself uses internally:
      1. ``HUGGINGFACE_HUB_CACHE`` env (the hub subdir directly) — wins
      2. ``HF_HOME`` env (the base dir) + "/hub" — HF appends /hub itself
      3. ``~/.cache/huggingface/hub`` (default)

    Earlier bug (Wave-12.5 live-test catch): treating HF_HOME as the hub
    dir directly would miss caches when ``HF_HOME`` was set to a base
    dir (since HF actually stores under ``<HF_HOME>/hub/``). Now it's
    correctly mapped.
    """
    hub_env = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_env:
        return Path(hub_env)
    base_env = os.environ.get("HF_HOME")
    if base_env:
        return Path(base_env) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_cache_dir(repo_id: str) -> Path:
    """The HF cache dir for a given repo_id (matches HF's naming scheme)."""
    safe = "models--" + repo_id.replace("/", "--")
    return _hf_cache_root() / safe


def _is_repo_cached(repo_id: str) -> bool:
    """True when at least one full snapshot of the repo is present.

    HF cache layout: ``<root>/models--<org>--<name>/snapshots/<sha>/<files>``.
    We accept any non-empty snapshot directory as "cached" — the resolver
    will pick the right revision at load time.
    """
    cache_dir = _repo_cache_dir(repo_id)
    if not cache_dir.is_dir():
        return False
    snap_dir = cache_dir / "snapshots"
    if not snap_dir.is_dir():
        return False
    for sub in snap_dir.iterdir():
        if sub.is_dir() and any(sub.iterdir()):
            return True
    return False


def needs_first_time_setup() -> bool:
    """True if any essential model is missing from the local cache."""
    return any(not _is_repo_cached(r) for r in _ESSENTIAL_REPOS)


def _compute_actual_download_size(missing_repos: list[str]) -> int:
    """Query HuggingFace Hub for the real total byte size of files we'll
    download. Falls back to the conservative estimate if the API is
    unreachable (offline, rate-limited, etc.).

    Caveat fix: previously the progress bar used a hardcoded 1.4GB
    estimate, so the bar would slightly under- or over-shoot 100%. Now
    the bar's max value is the real sum of remote file sizes.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        total = 0
        for repo_id in missing_repos:
            try:
                info = api.repo_info(repo_id, files_metadata=True)
                for sibling in (info.siblings or []):
                    size = getattr(sibling, "size", None) or 0
                    total += size
            except Exception as exc:  # noqa: BLE001 — network failure path
                logger.debug("Couldn't sum sizes for %s: %s", repo_id, exc)
                # Fall back to estimate for THIS repo
                total += _ESTIMATED_TOTAL_BYTES // max(len(missing_repos), 1)
        return total if total > 0 else _ESTIMATED_TOTAL_BYTES
    except Exception:  # noqa: BLE001
        return _ESTIMATED_TOTAL_BYTES


def _is_interactive_tty() -> bool:
    """True when we have a real TTY for rich progress rendering.

    Daemon mode (launchd / systemd / `predacore start --daemon` redirected
    to a logfile) gets non-TTY behavior: log lines, no progress bar.
    """
    return sys.stdout.isatty() and not os.environ.get("PREDACORE_NO_TTY")


def run_first_time_setup() -> bool:
    """Download missing essential models with a single progress bar.

    Returns True on success, False if downloads failed (caller should
    decide whether to proceed with degraded behavior or abort).

    No-op when ``needs_first_time_setup()`` is False.
    """
    missing = [r for r in _ESSENTIAL_REPOS if not _is_repo_cached(r)]
    if not missing:
        return True

    # ── Daemon mode: log-only, no fancy UI ──────────────────────────────
    if not _is_interactive_tty():
        logger.info(
            "First-time setup — downloading %d model%s in background",
            len(missing), "" if len(missing) == 1 else "s",
        )
        return _download_with_logs(missing)

    # ── Interactive TTY: game-style single progress bar ─────────────────
    try:
        from rich.console import Console
        from rich.progress import (
            BarColumn, Progress, TextColumn, TimeRemainingColumn,
        )
    except ImportError:
        # rich is in core deps but if it's somehow missing, fall back
        return _download_with_logs(missing)

    console = Console()
    console.print()
    console.print("🚀 [bold cyan]Predacore[/bold cyan]")
    console.print()
    console.print("[dim]First-time setup — this happens once.[/dim]")
    console.print()

    # Single bar covering all downloads. Real byte total (queried from HF)
    # so the bar lands cleanly at 100%. Stage label rotates per-repo
    # (e.g. "Loading memory" → "Loading recall layer" → "Almost ready").
    total_bytes = _compute_actual_download_size(missing)
    with Progress(
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("Loading memory and browser", total=total_bytes)
        success = _download_with_progress(missing, progress, task_id, total_bytes)

    if success:
        console.print()
        console.print("[green]✓[/green] Ready.")
        console.print()
    else:
        console.print()
        console.print(
            "[yellow]⚠[/yellow] Setup incomplete — some features may "
            "download on first use instead."
        )
        console.print()
    return success


def _download_with_progress(
    repos: list[str], progress, task_id, total_bytes: int,
) -> bool:
    """Download each missing repo, feeding cumulative bytes into the
    shared progress bar via a custom tqdm hook + rotating the bar's
    description text per-repo stage.

    Hugging Face Hub uses tqdm internally for download bars; we hook it
    by passing ``tqdm_class`` to ``snapshot_download``. Our custom class
    forwards every ``update(n)`` call into the rich.progress task,
    accumulating across all repos for the single-bar UX. The bar's
    description text rotates through stage labels as we move repo to
    repo ("Loading memory" → "Loading recall layer" → "Almost ready").
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed — cannot prefetch models")
        return False

    cumulative = {"bytes": 0}

    # Subclass tqdm so we inherit class-level methods (get_lock, set_lock)
    # that HF Hub's concurrent download path calls. A bare ad-hoc class
    # broke when HF Hub's `thread_map` tried `tqdm_class.get_lock()`.
    # Subclassing keeps all the lock + display machinery; we override
    # `update()` to relay bytes into the rich.progress task.
    try:
        from tqdm import tqdm as _tqdm_base
    except ImportError:
        # tqdm is a transitive dep of huggingface_hub — should be present
        # whenever snapshot_download is. If somehow missing, error out
        # cleanly rather than crashing the daemon.
        logger.error("tqdm not installed — cannot relay download progress")
        return False

    class _RelayTqdm(_tqdm_base):
        """tqdm subclass that relays .update(n) into our single rich bar."""
        def __init__(self, *args, **kwargs):
            # Suppress tqdm's own bar rendering; we have rich.progress for that.
            kwargs.setdefault("disable", True)
            super().__init__(*args, **kwargs)

        def update(self, n: int = 1) -> None:
            super().update(n)
            if n <= 0:
                return
            cumulative["bytes"] += n
            # Cap displayed progress at 99% until all repos finish; the
            # final 1% is the "done" flip below. Prevents the bar from
            # hitting 100% mid-download if HF returns slightly more
            # data than the metadata sum predicted.
            target = min(cumulative["bytes"], int(total_bytes * 0.99))
            progress.update(task_id, completed=target)

    all_ok = True
    repo_count = len(repos)
    for idx, repo_id in enumerate(repos):
        # Rotate the bar's label to the current stage. Final stage label
        # is "Almost ready" so the user sees the wind-down even if a
        # repo's last few MB take a while to flush.
        if idx == repo_count - 1 and repo_count > 1:
            stage_label = "Almost ready"
        else:
            stage_label = _ESSENTIAL_REPOS.get(repo_id, f"Loading {idx + 1}/{repo_count}")
        progress.update(task_id, description=stage_label)

        try:
            snapshot_download(
                repo_id=repo_id,
                tqdm_class=_RelayTqdm,
                allow_patterns=None,
            )
        except Exception as exc:  # noqa: BLE001 — broad: download UX must not crash daemon
            logger.warning("Prefetch failed for %s: %s", repo_id, exc)
            all_ok = False

    # Flip to 100% on success + final friendly label
    if all_ok:
        progress.update(task_id, completed=total_bytes, description="Ready")
    return all_ok


def _download_with_logs(repos: list[str]) -> bool:
    """Fallback path for non-TTY mode — same downloads, log-only UX."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed — cannot prefetch models")
        return False

    all_ok = True
    for repo_id in repos:
        logger.info("Prefetching %s ...", repo_id)
        try:
            snapshot_download(repo_id=repo_id)
            logger.info("  ✓ %s ready", repo_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  ✗ %s failed: %s", repo_id, exc)
            all_ok = False
    return all_ok


__all__ = [
    "needs_first_time_setup",
    "run_first_time_setup",
]
