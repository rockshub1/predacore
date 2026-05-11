"""
Async-safe circuit breaker for LLM provider failover.

All methods are async and use ``asyncio.Lock`` to coordinate state under
concurrent access from the same event loop. The lock-held sections only
do dict operations (no I/O), so contention is bounded.

Extracted from LLMInterface to be reusable and testable independently.
"""
from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Per-provider circuit breaker with failure counting and cooldown.

    When a provider accumulates ``failure_threshold`` failures within
    ``window_seconds``, the circuit trips and the provider is skipped
    for ``cooldown_seconds``.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        window_seconds: float = 300,
        cooldown_seconds: float = 120,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds

        # asyncio.Lock instead of threading.Lock — this breaker is only
        # touched from async paths (router.chat). Using a sync lock would
        # block the event loop under contention; using asyncio.Lock makes
        # the design honest about its concurrency model.
        self._lock = asyncio.Lock()
        self._failures: dict[str, list[tuple[float, str]]] = {}
        self._tripped_until: dict[str, float] = {}

    async def is_open(self, provider: str) -> bool:
        """Check if provider's circuit is tripped (should be skipped).

        L37 (Wave 12) — pure read. Cooldown expiration was previously
        side-effectful here (cleared `_tripped_until` and `_failures`),
        which meant a probing caller could accidentally reset state.
        Now: read-only check against `_tripped_until`. Failure-count
        evaluation and trip-on-overflow happen inside `record_failure`
        where the state mutation belongs. Expired cooldowns are cleaned
        up lazily by `record_failure` and the next `record_success`.
        """
        async with self._lock:
            tripped = self._tripped_until.get(provider, 0)
            return tripped > time.time()

    async def record_failure(self, provider: str, error: Exception | str) -> None:
        """Record a provider failure. Trips the circuit on threshold overflow.

        L37 (Wave 12) — this method now owns the trip decision. The check
        used to live in `is_open`, which made a "read" method mutate
        `_tripped_until`. Now writes go where writes belong: an actual
        failure is the trigger for evaluating the threshold.
        """
        now = time.time()
        async with self._lock:
            # Clear expired cooldowns (lazy GC).
            if 0 < self._tripped_until.get(provider, 0) <= now:
                self._tripped_until.pop(provider, None)
                self._failures.pop(provider, None)
                logger.info("Circuit breaker reset for provider: %s", provider)

            self._failures.setdefault(provider, []).append((now, str(error)))
            # Prune to the window.
            cutoff = now - self.window_seconds
            self._failures[provider] = [
                (ts, err) for ts, err in self._failures[provider] if ts > cutoff
            ]
            recent_count = len(self._failures[provider])
            if recent_count >= self.failure_threshold:
                self._tripped_until[provider] = now + self.cooldown_seconds
                logger.warning(
                    "Circuit breaker TRIPPED for '%s' (%d failures in %ds) — "
                    "cooling down for %ds",
                    provider,
                    recent_count,
                    int(self.window_seconds),
                    int(self.cooldown_seconds),
                )
        logger.warning("Provider '%s' failed: %s", provider, error)

    async def record_success(self, provider: str) -> None:
        """Record a provider success — clears failure history for faster recovery."""
        async with self._lock:
            if provider in self._failures:
                self._failures[provider].clear()

    async def reset(self, provider: str) -> None:
        """Manually reset a provider's circuit breaker."""
        async with self._lock:
            self._failures.pop(provider, None)
            self._tripped_until.pop(provider, None)

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        async with self._lock:
            self._failures.clear()
            self._tripped_until.clear()
