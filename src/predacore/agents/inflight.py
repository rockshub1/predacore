"""In-flight orchestration dedup.

Two flavors:
  1. Orchestration-level: same user prompt typed twice in 2 s. Don't
     spawn a second orchestrator — return the first's result when ready.
  2. Tool-level (within a run): two subagents both call
     web_search("Q3 earnings") at t=0. Don't run the API call twice;
     second caller awaits the first's result.

Tool-level dedup is implemented in services/memory_server.py:_InFlightRegistry
because it must be visible across processes (DAF workers + main).

Orchestration-level dedup lives here as a per-process registry. If you
want it cross-process (e.g. user has two CLI sessions in parallel),
escalate to the daemon side later — single-user CLI rarely needs it.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


def orchestration_key(*, user_id: str, task: str) -> str:
    """Stable key for an orchestration. Hashes user+task content.

    A fresh suffix is NOT added — that's the point: identical asks should
    collide so the dedup wins.
    """
    payload = f"{user_id}:{task.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class _InFlightEntry:
    started_at: float
    future: asyncio.Future[Any] = field(default_factory=lambda: asyncio.get_running_loop().create_future())


class OrchestrationInFlight:
    """Process-local registry for orchestration-level dedup.

    Use:
        ifr = OrchestrationInFlight(window_seconds=10)
        existing = ifr.try_claim(key)
        if existing is None:
            # I'm the first — run the orchestration, then publish
            try:
                result = await orchestrator.run(...)
                ifr.publish(key, result)
            except Exception as exc:
                ifr.fail(key, exc)
        else:
            # Duplicate — await the first caller's result
            result = await existing
    """

    def __init__(self, *, window_seconds: int = 10) -> None:
        self._window = window_seconds
        self._inflight: dict[str, _InFlightEntry] = {}
        self._lock = asyncio.Lock()

    async def try_claim(self, key: str) -> asyncio.Future[Any] | None:
        """Returns None if first caller (claimed), else returns the
        existing in-flight Future to await."""
        now = time.monotonic()
        async with self._lock:
            self._gc(now)
            if key in self._inflight:
                return self._inflight[key].future
            self._inflight[key] = _InFlightEntry(
                started_at=now,
                future=asyncio.get_running_loop().create_future(),
            )
            return None

    async def publish(self, key: str, result: Any) -> None:
        async with self._lock:
            entry = self._inflight.pop(key, None)
            if entry is not None and not entry.future.done():
                entry.future.set_result(result)

    async def fail(self, key: str, exc: BaseException) -> None:
        async with self._lock:
            entry = self._inflight.pop(key, None)
            if entry is not None and not entry.future.done():
                entry.future.set_exception(exc)

    def _gc(self, now: float) -> None:
        """Drop stale entries (called under lock)."""
        cutoff = now - self._window
        stale = [k for k, e in self._inflight.items() if e.started_at < cutoff]
        for k in stale:
            entry = self._inflight.pop(k, None)
            if entry is not None and not entry.future.done():
                entry.future.cancel()
