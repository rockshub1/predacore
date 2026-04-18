"""
Thread-safe circuit breaker for LLM provider failover.

Extracted from LLMInterface to be reusable and testable independently.
"""
from __future__ import annotations

import logging
import threading
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

        self._lock = threading.Lock()
        self._failures: dict[str, list[tuple[float, str]]] = {}
        self._tripped_until: dict[str, float] = {}

    def is_open(self, provider: str) -> bool:
        """Check if provider's circuit is tripped (should be skipped)."""
        with self._lock:
            now = time.time()
            tripped = self._tripped_until.get(provider, 0)

            if tripped > now:
                return True
            elif tripped > 0:
                # Cooldown expired — reset
                self._tripped_until.pop(provider, None)
                self._failures.pop(provider, None)
                logger.info("Circuit breaker reset for provider: %s", provider)
                return False

            # Check failure count in window
            failures = self._failures.get(provider, [])
            cutoff = now - self.window_seconds
            recent = [ts for ts, _ in failures if ts > cutoff]
            self._failures[provider] = [
                (ts, err) for ts, err in failures if ts > cutoff
            ]

            if len(recent) >= self.failure_threshold:
                self._tripped_until[provider] = now + self.cooldown_seconds
                logger.warning(
                    "Circuit breaker TRIPPED for '%s' (%d failures in %ds) — "
                    "cooling down for %ds",
                    provider,
                    len(recent),
                    int(self.window_seconds),
                    int(self.cooldown_seconds),
                )
                return True

            return False

    def record_failure(self, provider: str, error: Exception | str) -> None:
        """Record a provider failure."""
        with self._lock:
            if provider not in self._failures:
                self._failures[provider] = []
            self._failures[provider].append((time.time(), str(error)))
        logger.warning("Provider '%s' failed: %s", provider, error)

    def record_success(self, provider: str) -> None:
        """Record a provider success — clears failure history for faster recovery."""
        with self._lock:
            if provider in self._failures:
                self._failures[provider].clear()

    def reset(self, provider: str) -> None:
        """Manually reset a provider's circuit breaker."""
        with self._lock:
            self._failures.pop(provider, None)
            self._tripped_until.pop(provider, None)

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            self._failures.clear()
            self._tripped_until.clear()
