"""
Channel Health Monitor — tracks uptime, message rates, error rates per channel.

Provides real-time health metrics for all registered channel adapters,
supporting:
  - Per-channel uptime tracking
  - Message throughput (msgs/sec rolling window)
  - Error rate tracking with auto-circuit-breaker
  - Reconnection management with exponential backoff
  - Health check endpoint data for /status

Usage:
    monitor = ChannelHealthMonitor()
    monitor.register("telegram")
    monitor.record_message("telegram")
    monitor.record_error("telegram", "connection_timeout")
    health = monitor.get_health_report()
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Optional

logger = logging.getLogger(__name__)


# ── Health Status ────────────────────────────────────────────────────


class ChannelStatus(str, Enum):
    """Health status of a channel adapter."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"  # High error rate but still working
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


# ── Rate-Limited Message Queue ───────────────────────────────────────


class RateLimiter:
    """
    Token-bucket rate limiter for outgoing messages.

    Prevents hitting platform API limits (e.g., Telegram: 30 msgs/sec,
    Discord: 5 msgs/sec per channel).
    """

    def __init__(self, rate: float, burst: int = 1):
        """
        Args:
            rate: Messages per second allowed.
            burst: Maximum burst size (tokens).
        """
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume one."""
        wait_time = 0.0
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / self._rate
            else:
                self._tokens -= 1.0

        # Sleep OUTSIDE the lock so other coroutines aren't blocked
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            async with self._lock:
                self._tokens = 0.0
                self._last_refill = time.monotonic()


# ── Exponential Backoff Reconnection Manager ────────────────────────


class ReconnectionManager:
    """
    Manages reconnection with exponential backoff + jitter.

    Usage:
        mgr = ReconnectionManager()
        while not connected:
            delay = mgr.next_delay()
            await asyncio.sleep(delay)
            try_connect()
        mgr.reset()
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 300.0,
        multiplier: float = 2.0,
        jitter: float = 0.1,
    ):
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._multiplier = multiplier
        self._jitter = jitter
        self._attempt = 0

    def next_delay(self) -> float:
        """Calculate the next backoff delay with jitter."""
        import random

        delay = min(
            self._base_delay * (self._multiplier**self._attempt),
            self._max_delay,
        )
        jitter = delay * self._jitter * (2 * random.random() - 1)
        self._attempt += 1
        return max(0.1, delay + jitter)

    def reset(self) -> None:
        """Reset backoff after successful reconnection."""
        self._attempt = 0

    @property
    def attempt(self) -> int:
        return self._attempt


# ── Per-Channel Health Record ───────────────────────────────────────


@dataclass
class ChannelHealthRecord:
    """Tracks health metrics for a single channel."""

    channel_name: str
    status: ChannelStatus = ChannelStatus.UNKNOWN
    started_at: float = 0.0
    last_message_at: float = 0.0
    last_error_at: float = 0.0
    last_error_msg: str = ""

    # Rolling windows (last 60 seconds of timestamps)
    _message_times: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    _error_times: deque[float] = field(default_factory=lambda: deque(maxlen=500))

    # Counters
    total_messages_in: int = 0
    total_messages_out: int = 0
    total_errors: int = 0

    # Reconnection state
    reconnect_count: int = 0

    # Rate limiter (set per-channel)
    rate_limiter: RateLimiter | None = None

    # Reconnection manager
    reconnection: ReconnectionManager = field(default_factory=ReconnectionManager)

    def record_message(self, direction: str = "in") -> None:
        """Record a message event (incoming or outgoing)."""
        now = time.time()
        self.last_message_at = now
        self._message_times.append(now)
        if direction == "in":
            self.total_messages_in += 1
        else:
            self.total_messages_out += 1

    def record_error(self, error_msg: str = "") -> None:
        """Record an error event."""
        now = time.time()
        self.last_error_at = now
        self.last_error_msg = error_msg
        self.total_errors += 1
        self._error_times.append(now)

    def messages_per_minute(self, window: float = 60.0) -> float:
        """Calculate messages per minute over a rolling window."""
        cutoff = time.time() - window
        recent = sum(1 for t in self._message_times if t > cutoff)
        return recent * (60.0 / window)

    def error_rate(self, window: float = 300.0) -> float:
        """Calculate error rate (errors / total events) over a rolling window."""
        cutoff = time.time() - window
        recent_errors = sum(1 for t in self._error_times if t > cutoff)
        recent_messages = sum(1 for t in self._message_times if t > cutoff)
        total = recent_errors + recent_messages
        return recent_errors / total if total > 0 else 0.0

    @property
    def uptime_seconds(self) -> float:
        """Time since the channel was started."""
        if self.started_at:
            return time.time() - self.started_at
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize health record for API responses."""
        return {
            "channel": self.channel_name,
            "status": self.status.value,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "messages_per_minute": round(self.messages_per_minute(), 2),
            "total_messages_in": self.total_messages_in,
            "total_messages_out": self.total_messages_out,
            "total_errors": self.total_errors,
            "error_rate_5m": round(self.error_rate(300.0), 4),
            "last_message_age_seconds": round(time.time() - self.last_message_at, 1)
            if self.last_message_at
            else None,
            "last_error": self.last_error_msg or None,
            "reconnect_count": self.reconnect_count,
        }


# ── Channel Health Monitor ──────────────────────────────────────────

# Default rate limits per platform (msgs/sec)
DEFAULT_RATE_LIMITS: dict[str, float] = {
    "telegram": 30.0,  # Telegram: 30 msgs/sec global
    "discord": 5.0,  # Discord: 5 msgs/sec per channel
    "whatsapp": 80.0,  # WhatsApp Business: 80 msgs/sec
    "webchat": 100.0,  # WebSocket — no platform limit, set a sane max
    "cli": 1000.0,  # CLI — effectively unlimited
}

# Error rate threshold for "degraded" status
DEGRADED_ERROR_RATE = 0.10  # 10% errors = degraded
CIRCUIT_BREAK_ERROR_RATE = 0.50  # 50% errors = stop sending


class ChannelHealthMonitor:
    """
    Central health monitor for all channel adapters.

    Features:
        - Per-channel health tracking
        - Rate limiting per platform
        - Auto status detection (healthy → degraded → disconnected)
        - Health report generation for /status endpoint
        - Reconnection coordination
    """

    def __init__(self):
        self._channels: dict[str, ChannelHealthRecord] = {}
        self._callbacks: list[Callable] = []
        self._monitor_task: asyncio.Task | None = None

    def register(
        self,
        channel_name: str,
        rate_limit: float | None = None,
    ) -> ChannelHealthRecord:
        """
        Register a channel for health monitoring.

        Args:
            channel_name: Channel identifier (e.g., "telegram")
            rate_limit: Custom rate limit (msgs/sec). Uses platform default if None.

        Returns:
            The ChannelHealthRecord for further interaction.
        """
        effective_rate = rate_limit or DEFAULT_RATE_LIMITS.get(channel_name, 100.0)
        record = ChannelHealthRecord(
            channel_name=channel_name,
            rate_limiter=RateLimiter(
                rate=effective_rate, burst=max(1, int(effective_rate))
            ),
        )
        self._channels[channel_name] = record
        logger.info(
            "Health monitor registered channel: %s (rate limit: %.1f/sec)",
            channel_name,
            effective_rate,
        )
        return record

    def mark_started(self, channel_name: str) -> None:
        """Mark a channel as started and healthy."""
        record = self._channels.get(channel_name)
        if record:
            record.started_at = time.time()
            record.status = ChannelStatus.HEALTHY
            record.reconnection.reset()
            logger.info("Channel %s marked healthy", channel_name)

    def mark_stopped(self, channel_name: str) -> None:
        """Mark a channel as stopped."""
        record = self._channels.get(channel_name)
        if record:
            record.status = ChannelStatus.STOPPED

    def mark_disconnected(self, channel_name: str, reason: str = "") -> None:
        """Mark a channel as disconnected."""
        record = self._channels.get(channel_name)
        if record:
            record.status = ChannelStatus.DISCONNECTED
            if reason:
                record.record_error(reason)
            logger.warning("Channel %s disconnected: %s", channel_name, reason)

    def mark_reconnecting(self, channel_name: str) -> None:
        """Mark a channel as reconnecting."""
        record = self._channels.get(channel_name)
        if record:
            record.status = ChannelStatus.RECONNECTING
            record.reconnect_count += 1

    def record_message(self, channel_name: str, direction: str = "in") -> None:
        """Record a message event for a channel."""
        record = self._channels.get(channel_name)
        if record:
            record.record_message(direction)
            # Auto-promote to HEALTHY if was degraded and error rate dropped
            if record.status == ChannelStatus.DEGRADED:
                if record.error_rate() < DEGRADED_ERROR_RATE:
                    record.status = ChannelStatus.HEALTHY

    def record_error(self, channel_name: str, error_msg: str = "") -> None:
        """Record an error event for a channel."""
        record = self._channels.get(channel_name)
        if record:
            record.record_error(error_msg)
            # Auto-degrade if error rate is too high
            if record.error_rate() >= DEGRADED_ERROR_RATE:
                if record.status == ChannelStatus.HEALTHY:
                    record.status = ChannelStatus.DEGRADED
                    logger.warning(
                        "Channel %s degraded (error rate: %.1f%%)",
                        channel_name,
                        record.error_rate() * 100,
                    )

    async def acquire_rate_limit(self, channel_name: str) -> bool:
        """
        Wait for rate limit token before sending a message.

        Returns:
            True if allowed, False if circuit breaker is tripped.
        """
        record = self._channels.get(channel_name)
        if not record:
            return True

        # Circuit breaker: refuse to send if error rate is too high
        if record.error_rate() >= CIRCUIT_BREAK_ERROR_RATE:
            logger.error(
                "Circuit breaker tripped for %s (error rate: %.1f%%)",
                channel_name,
                record.error_rate() * 100,
            )
            return False

        if record.rate_limiter:
            await record.rate_limiter.acquire()

        return True

    def get_channel_health(self, channel_name: str) -> dict[str, Any] | None:
        """Get health data for a specific channel."""
        record = self._channels.get(channel_name)
        return record.to_dict() if record else None

    def get_health_report(self) -> dict[str, Any]:
        """
        Get a full health report across all channels.

        Returns a dict suitable for JSON serialization and /status endpoint.
        """
        channels = {name: record.to_dict() for name, record in self._channels.items()}

        # Compute aggregate health
        statuses = [r.status for r in self._channels.values()]
        if not statuses:
            overall = "no_channels"
        elif all(s == ChannelStatus.HEALTHY for s in statuses):
            overall = "healthy"
        elif any(
            s in (ChannelStatus.DISCONNECTED, ChannelStatus.STOPPED) for s in statuses
        ):
            overall = "degraded"
        elif any(s == ChannelStatus.DEGRADED for s in statuses):
            overall = "degraded"
        else:
            overall = "unknown"

        return {
            "overall_status": overall,
            "channels": channels,
            "total_channels": len(self._channels),
            "healthy_channels": sum(1 for s in statuses if s == ChannelStatus.HEALTHY),
            "timestamp": time.time(),
        }

    async def start_monitoring(self, interval: float = 30.0) -> None:
        """Start a background task that periodically checks channel health."""

        async def _monitor_loop():
            while True:
                await asyncio.sleep(interval)
                self._auto_health_check()

        self._monitor_task = asyncio.create_task(_monitor_loop())
        logger.info("Health monitor started (interval: %.0fs)", interval)

    async def stop_monitoring(self) -> None:
        """Stop the background health monitor."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    def _auto_health_check(self) -> None:
        """
        Automatically check channel health based on message activity.

        If no messages received in the last 5 minutes for a "started"
        channel, mark it as degraded.
        """
        now = time.time()
        for name, record in self._channels.items():
            if record.status in (ChannelStatus.STOPPED, ChannelStatus.UNKNOWN):
                continue

            # Check staleness: no messages for 5 minutes while "healthy"
            if record.status == ChannelStatus.HEALTHY and record.last_message_at:
                age = now - record.last_message_at
                if age > 300:  # 5 minutes
                    logger.info("Channel %s went idle (last msg %.0fs ago)", name, age)

            # Check error rate
            error_rate = record.error_rate()
            if error_rate >= CIRCUIT_BREAK_ERROR_RATE:
                record.status = ChannelStatus.DISCONNECTED
            elif error_rate >= DEGRADED_ERROR_RATE:
                record.status = ChannelStatus.DEGRADED
            elif (
                record.status == ChannelStatus.DEGRADED
                and error_rate < DEGRADED_ERROR_RATE
            ):
                record.status = ChannelStatus.HEALTHY
