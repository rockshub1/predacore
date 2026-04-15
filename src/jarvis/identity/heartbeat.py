"""
JARVIS HeartbeatService — proactive background behaviors.

The heartbeat system runs periodic tasks that make the agent feel alive
between conversations: journal pruning, identity staleness detection,
and integration with the existing memory consolidation pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import IdentityEngine

logger = logging.getLogger(__name__)

_JOURNAL_LOCK = asyncio.Lock()


@dataclass
class HeartbeatTask:
    """A registered periodic background task."""

    name: str
    interval_seconds: float
    handler: Callable[[], Coroutine[Any, Any, None]]
    last_run: float = 0.0
    run_count: int = 0
    active: bool = True
    _task: asyncio.Task | None = field(default=None, repr=False)


class HeartbeatService:
    """Proactive background behaviors — the agent's initiative."""

    def __init__(self, engine: IdentityEngine):
        self.engine = engine
        self._tasks: dict[str, HeartbeatTask] = {}
        self._running = False

    def register(
        self,
        name: str,
        interval_seconds: float,
        handler: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        """Register a periodic heartbeat task."""
        self._tasks[name] = HeartbeatTask(
            name=name,
            interval_seconds=interval_seconds,
            handler=handler,
        )
        logger.info(
            "Heartbeat task registered: %s (every %.0fs)", name, interval_seconds
        )

    async def start(self) -> None:
        """Start all registered heartbeat tasks as background coroutines."""
        if self._running:
            logger.warning("HeartbeatService already running")
            return

        self._running = True
        for task_def in self._tasks.values():
            task_def._task = asyncio.create_task(
                self._run_loop(task_def),
                name=f"heartbeat:{task_def.name}",
            )
        logger.info("HeartbeatService started with %d tasks", len(self._tasks))

    async def stop(self) -> None:
        """Gracefully stop all heartbeat tasks."""
        self._running = False
        for task_def in self._tasks.values():
            if task_def._task and not task_def._task.done():
                task_def._task.cancel()
                try:
                    await task_def._task
                except asyncio.CancelledError:
                    pass
            task_def.active = False
        logger.info("HeartbeatService stopped")

    async def _run_loop(self, task_def: HeartbeatTask) -> None:
        """Run a single heartbeat task in a loop."""
        while self._running and task_def.active:
            try:
                await asyncio.sleep(task_def.interval_seconds)
                if not self._running:
                    break
                await task_def.handler()
                task_def.last_run = time.time()
                task_def.run_count += 1
                logger.debug(
                    "Heartbeat task '%s' completed (run #%d)",
                    task_def.name,
                    task_def.run_count,
                )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error(
                    "Heartbeat task '%s' failed", task_def.name, exc_info=True
                )
                # Don't crash the loop — wait and retry
                await asyncio.sleep(min(task_def.interval_seconds, 60))

    def get_stats(self) -> dict[str, Any]:
        """Return heartbeat service statistics."""
        return {
            "running": self._running,
            "tasks": {
                name: {
                    "interval_seconds": t.interval_seconds,
                    "last_run": t.last_run or None,
                    "run_count": t.run_count,
                    "active": t.active,
                }
                for name, t in self._tasks.items()
            },
        }


# ---------------------------------------------------------------------------
# Default heartbeat tasks
# ---------------------------------------------------------------------------


async def journal_prune_task(engine: IdentityEngine, max_age_days: int = 30) -> None:
    """Trim old journal entries into a summary section.

    Entries older than max_age_days are condensed into a single
    "Archive" block at the top of JOURNAL.md.
    """
    async with _JOURNAL_LOCK:
        await _journal_prune_impl(engine, max_age_days)


async def _journal_prune_impl(engine: IdentityEngine, max_age_days: int = 30) -> None:
    """Inner implementation — called with _JOURNAL_LOCK held."""
    path = engine.workspace / "JOURNAL.md"
    if not path.exists():
        return

    content = path.read_text(encoding="utf-8")
    entries = re.split(r"(?=^## \d{4}-\d{2}-\d{2})", content, flags=re.MULTILINE)

    if len(entries) <= 25:
        return  # Not enough entries to warrant pruning

    header = entries[0] if not entries[0].strip().startswith("## 20") else ""
    dated_entries = [e for e in entries if e.strip().startswith("## 20")]

    now = datetime.now(timezone.utc)
    recent = []
    old = []

    for entry in dated_entries:
        # Extract date from ## YYYY-MM-DD header
        date_match = re.match(r"## (\d{4}-\d{2}-\d{2})", entry.strip())
        if date_match:
            try:
                entry_date = datetime.strptime(
                    date_match.group(1), "%Y-%m-%d"
                ).replace(tzinfo=timezone.utc)
                age_days = (now - entry_date).days
                if age_days > max_age_days:
                    old.append(entry.strip())
                else:
                    recent.append(entry.strip())
            except ValueError:
                recent.append(entry.strip())
        else:
            recent.append(entry.strip())

    if not old:
        return  # Nothing to prune

    archive_summary = (
        f"## Archive (pruned {len(old)} entries older than {max_age_days} days)\n\n"
        f"_{len(old)} entries from early journal history were condensed on "
        f"{now.strftime('%Y-%m-%d')}._\n"
    )

    new_content = header.strip() + "\n\n" if header.strip() else ""
    new_content += archive_summary + "\n\n"
    new_content += "\n\n".join(recent)
    new_content += "\n"

    path.write_text(new_content, encoding="utf-8")
    logger.info("Journal pruned: %d old entries archived", len(old))


async def identity_staleness_check(
    engine: IdentityEngine, max_stale_days: int = 14
) -> None:
    """Check if SOUL.md hasn't been updated recently and log a reminder."""
    soul_path = engine.workspace / "SOUL.md"
    if not soul_path.exists():
        return

    mtime = soul_path.stat().st_mtime
    age_days = (time.time() - mtime) / 86400

    if age_days > max_stale_days:
        async with _JOURNAL_LOCK:
            engine.append_journal(
                f"Identity staleness notice: SOUL.md hasn't been updated in "
                f"{int(age_days)} days. Consider reflecting on whether your "
                f"personality has evolved since last update."
            )
        logger.info("Identity staleness detected: SOUL.md is %d days old", int(age_days))


def create_default_heartbeat(engine: IdentityEngine) -> HeartbeatService:
    """Create a HeartbeatService with default tasks registered."""
    service = HeartbeatService(engine)

    # Journal pruning — daily (86400 seconds)
    service.register(
        "journal_prune",
        86400,
        lambda: journal_prune_task(engine),
    )

    # Identity staleness check — weekly (604800 seconds)
    service.register(
        "identity_staleness",
        604800,
        lambda: identity_staleness_check(engine),
    )

    return service
