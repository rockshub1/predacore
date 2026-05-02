"""Workspace tracking + bulk-index state for PredaCore.

Three concepts live here:

  - **AbortToken** — a thread-safe flag that a long-running bulk-index can
    poll to honor user "stop" requests. Stays minimal: a flag + a reason.
  - **MemoryIndexStatus** — global state describing what the bulk-index is
    doing right now (idle / running / done / failed) plus per-file progress.
    Read by ``predacore status``, the ``memory_index_status`` tool, and any
    UI surface that wants to show "indexing 89/212 files (42%)".
  - **WorkspaceTracker** — detects when the active project changes. Other
    code subscribes to ``on_change`` callbacks; the tracker's ``check()``
    is cheap enough to call from the dispatcher per-tool-call.

This module is **state container only** — it doesn't decide policy. The
tools/handlers + daemon decide WHEN to prompt, WHEN to bulk, WHEN to abort.
This file gives them the primitives.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


# ── AbortToken ───────────────────────────────────────────────────────────


@dataclass
class AbortToken:
    """Cancel a long-running operation cooperatively.

    The bulk-index walker polls ``token.is_aborted`` between files (cheap;
    just a boolean read). When the user wants to stop, the tool handler
    flips ``request_abort(reason)`` and the walker exits at its next
    file boundary, returning whatever it indexed so far (partial state
    is fine — chunks are already on disk, the index is just incomplete).
    """

    _aborted: bool = field(default=False)
    _reason: str = field(default="")
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def is_aborted(self) -> bool:
        with self._lock:
            return self._aborted

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def request_abort(self, reason: str = "user requested") -> None:
        with self._lock:
            self._aborted = True
            self._reason = reason
        logger.info("AbortToken: abort requested (%s)", reason)

    def reset(self) -> None:
        """Clear the abort flag so the token can be reused."""
        with self._lock:
            self._aborted = False
            self._reason = ""


# ── MemoryIndexStatus ────────────────────────────────────────────────────


_State = Literal["idle", "indexing", "ready", "failed", "aborted"]


@dataclass
class MemoryIndexStatus:
    """Singleton-style global state for the active bulk-index.

    Read by ``predacore status``, ``memory_index_status`` tool, channel
    adapters that want to surface progress. There's exactly one of these
    per process — daemons / CLI sessions hold a reference and update it
    via ``begin()`` / ``progress()`` / ``finish()``.
    """

    state: _State = "idle"
    project_id: str = ""
    root: str = ""
    files_total: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    chunks_added: int = 0
    started_at: float | None = None
    completed_at: float | None = None
    last_error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def progress_pct(self) -> float:
        with self._lock:
            if self.files_total <= 0:
                return 100.0 if self.state == "ready" else 0.0
            return round(
                100.0 * (self.files_indexed + self.files_skipped) / self.files_total,
                1,
            )

    @property
    def elapsed_sec(self) -> float | None:
        with self._lock:
            if self.started_at is None:
                return None
            end = self.completed_at if self.completed_at else time.time()
            return round(end - self.started_at, 1)

    def begin(self, *, project_id: str, root: str, files_total: int) -> None:
        with self._lock:
            self.state = "indexing"
            self.project_id = project_id
            self.root = root
            self.files_total = files_total
            self.files_indexed = 0
            self.files_skipped = 0
            self.files_failed = 0
            self.chunks_added = 0
            self.started_at = time.time()
            self.completed_at = None
            self.last_error = None

    def update(
        self,
        *,
        indexed_delta: int = 0,
        skipped_delta: int = 0,
        failed_delta: int = 0,
        chunks_delta: int = 0,
    ) -> None:
        with self._lock:
            self.files_indexed += indexed_delta
            self.files_skipped += skipped_delta
            self.files_failed += failed_delta
            self.chunks_added += chunks_delta

    def finish(self, *, success: bool = True, error: str | None = None) -> None:
        with self._lock:
            self.state = "ready" if success else "failed"
            if error:
                self.last_error = error
            self.completed_at = time.time()

    def mark_aborted(self, reason: str) -> None:
        with self._lock:
            self.state = "aborted"
            self.last_error = f"aborted: {reason}"
            self.completed_at = time.time()

    def to_dict(self) -> dict:
        # Compute progress + elapsed INLINE rather than calling the @property
        # variants — those would re-acquire self._lock and deadlock on the
        # non-reentrant ``threading.Lock``.
        with self._lock:
            if self.files_total <= 0:
                pct = 100.0 if self.state == "ready" else 0.0
            else:
                pct = round(
                    100.0 * (self.files_indexed + self.files_skipped) / self.files_total,
                    1,
                )
            if self.started_at is None:
                elapsed: float | None = None
            else:
                end = self.completed_at if self.completed_at else time.time()
                elapsed = round(end - self.started_at, 1)
            return {
                "state": self.state,
                "project_id": self.project_id,
                "root": self.root,
                "files_total": self.files_total,
                "files_indexed": self.files_indexed,
                "files_skipped": self.files_skipped,
                "files_failed": self.files_failed,
                "chunks_added": self.chunks_added,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "elapsed_sec": elapsed,
                "progress_pct": pct,
                "last_error": self.last_error,
            }


# Process-level singleton. Imports across the codebase share this instance
# so ``predacore status`` and the index handler look at the same state.
_GLOBAL_STATUS = MemoryIndexStatus()
_GLOBAL_ABORT_TOKEN = AbortToken()


def get_global_status() -> MemoryIndexStatus:
    return _GLOBAL_STATUS


def get_global_abort_token() -> AbortToken:
    return _GLOBAL_ABORT_TOKEN


# ── WorkspaceTracker ─────────────────────────────────────────────────────


class WorkspaceTracker:
    """Detects project_id changes; fires registered callbacks.

    The dispatcher calls ``check_change()`` after every tool call (cheap:
    one cwd resolution + dict lookup). When the project changes, listeners
    fire — the daemon uses this to surface a first-touch prompt for projects
    we haven't bulk-indexed yet.

    NOT a singleton on import — daemon constructs one and shares the
    instance with anything that needs subscriptions.
    """

    def __init__(self) -> None:
        self._current: str | None = None
        self._listeners: list[Callable[[str | None, str], None]] = []
        self._lock = threading.Lock()

    @property
    def current(self) -> str | None:
        with self._lock:
            return self._current

    def on_change(self, cb: Callable[[str | None, str], None]) -> None:
        """Register ``cb(old_project, new_project)`` to fire on each change."""
        with self._lock:
            self._listeners.append(cb)

    def check_change(self, cwd: str | Path | None = None) -> str | None:
        """Resolve current project; return new project name if it changed.

        Returns ``None`` if no change. Cheap enough to call per-tool-call.
        """
        # Lazy import to avoid circulars at module load.
        from .project_id import default_project
        new = default_project(cwd, refresh=True) if cwd else default_project(refresh=True)
        with self._lock:
            if new == self._current:
                return None
            old = self._current
            self._current = new
            listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(old, new)
            except Exception as exc:  # noqa: BLE001 — never let a listener crash dispatch
                logger.warning("WorkspaceTracker listener failed: %s", exc)
        return new


# ── First-touch marker file ──────────────────────────────────────────────


def _markers_dir(home_dir: str | None = None) -> Path:
    base = Path(home_dir or "~/.predacore").expanduser()
    return base / "data" / "projects"


def has_been_bulk_indexed(project_id: str, *, home_dir: str | None = None) -> bool:
    """Return True if a marker file exists for ``project_id``.

    The marker is written after a successful bulk so subsequent daemon
    starts don't re-prompt the user. Wipe the file (or call
    ``mark_bulk_unindexed``) to force a re-prompt next time.
    """
    if not project_id or project_id == "all":
        return False
    marker = _markers_dir(home_dir) / project_id / ".bulk_indexed"
    return marker.is_file()


def mark_bulk_indexed(
    project_id: str,
    *,
    files_indexed: int,
    chunks_added: int,
    home_dir: str | None = None,
) -> None:
    """Write the first-touch marker file for ``project_id``.

    Stores enough metadata that ``predacore status`` can show "last indexed
    on YYYY-MM-DD" without re-walking. Callers should invoke this after a
    SUCCESSFUL bulk run (not on partial / aborted ones).
    """
    if not project_id or project_id == "all":
        return
    target_dir = _markers_dir(home_dir) / project_id
    target_dir.mkdir(parents=True, exist_ok=True)
    marker = target_dir / ".bulk_indexed"
    payload = {
        "project_id": project_id,
        "indexed_at": time.time(),
        "files_indexed": files_indexed,
        "chunks_added": chunks_added,
    }
    import json
    marker.write_text(json.dumps(payload, indent=2))


def mark_bulk_unindexed(project_id: str, *, home_dir: str | None = None) -> None:
    """Remove the marker so the next workspace touch re-prompts the user."""
    if not project_id:
        return
    marker = _markers_dir(home_dir) / project_id / ".bulk_indexed"
    try:
        marker.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Failed to remove marker %s: %s", marker, exc)
