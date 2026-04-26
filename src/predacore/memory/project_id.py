"""Project-id detection for memory tagging.

Memories carry a ``project_id`` tag so cross-repo pollution doesn't
leak: a query in repo A shouldn't surface notes from repo B unless
the caller opts in. This module is the single source of truth for
"what project am I currently working in?", used by:

  - tools/handlers/file_ops.py        (auto-reindex on Write)
  - tools/handlers/shell.py           (auto-sync after git mutations)
  - tools/handlers/memory.py          (memory_store / memory_recall handlers)
  - tools/subsystem_init.py           (defaults at boot)

Resolution precedence (mirrors lab mcp_server.py:_default_project):

  1. ``PREDACORE_MEMORY_PROJECT`` environment variable (explicit override)
  2. ``git rev-parse --show-toplevel`` basename (the repo name)
  3. cwd basename (when not in a git repo)
  4. ``"default"`` (last-resort fallback)

The git lookup is bounded to ~1.5s so it can never hang the caller.
Repeated lookups against the same cwd hit a small in-process cache.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

# Per-cwd cache: {cwd_str: (deadline_monotonic, project_id)}.
# 60-second TTL is plenty — `cd` events between cwds are rare in an
# agent loop, and re-resolving on every reindex is wasteful.
_PROJECT_CACHE: dict[str, tuple[float, str]] = {}
_CACHE_TTL_SECONDS = 60.0
_GIT_TIMEOUT_SECONDS = 1.5

# Sentinel meaning "don't filter on project_id" — handlers accept this
# from agents that explicitly want cross-project recall.
ALL_PROJECTS = "all"


def default_project(cwd: str | Path | None = None) -> str:
    """Resolve the current project_id with the standard precedence.

    Args:
        cwd: Working directory to resolve from. None → process cwd.

    Returns:
        Project identifier (always a non-empty string).
    """
    env = os.environ.get("PREDACORE_MEMORY_PROJECT", "").strip()
    if env:
        return env

    target_path = Path(cwd) if cwd else None
    cache_key = str(target_path.resolve()) if target_path else "_cwd_"
    cached = _PROJECT_CACHE.get(cache_key)
    if cached is not None:
        deadline, value = cached
        if time.monotonic() < deadline:
            return value

    project = _resolve_uncached(target_path)
    _PROJECT_CACHE[cache_key] = (
        time.monotonic() + _CACHE_TTL_SECONDS,
        project,
    )
    return project


def _resolve_uncached(cwd: Path | None) -> str:
    """Run the git → cwd → 'default' fallback chain."""
    cwd_str = str(cwd) if cwd else None
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            cwd=cwd_str,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return Path(proc.stdout.strip()).name
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    try:
        target = cwd if cwd else Path.cwd()
        return target.resolve().name or "default"
    except OSError:
        return "default"


def clear_cache() -> None:
    """Invalidate the per-cwd project_id cache. Useful for tests + after
    git checkouts that move us into a different repo."""
    _PROJECT_CACHE.clear()
