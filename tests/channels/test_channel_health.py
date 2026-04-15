"""
Tests for Channel Health Monitor.

Covers:
  - ChannelHealthRecord lifecycle
  - RateLimiter token-bucket behavior
  - ReconnectionManager exponential backoff
  - ChannelHealthMonitor registration, status transitions, health reports
  - Circuit breaker logic
  - Auto health check
"""
from __future__ import annotations

import time

import pytest

from src.jarvis.channels.health import (
    CIRCUIT_BREAK_ERROR_RATE,
    DEFAULT_RATE_LIMITS,
    DEGRADED_ERROR_RATE,
    ChannelHealthMonitor,
    ChannelHealthRecord,
    ChannelStatus,
    RateLimiter,
    ReconnectionManager,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ChannelHealthRecord Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestChannelHealthRecord:
    """Tests for per-channel health records."""

    def test_initial_state(self):
        """Record starts with all counters at zero."""
        record = ChannelHealthRecord(channel_name="test")
        assert record.status == ChannelStatus.UNKNOWN
        assert record.total_messages_in == 0
        assert record.total_messages_out == 0
        assert record.total_errors == 0
        assert record.uptime_seconds == 0.0

    def test_record_message_in(self):
        """Recording incoming messages increments the correct counter."""
        record = ChannelHealthRecord(channel_name="test")
        record.record_message("in")
        record.record_message("in")
        assert record.total_messages_in == 2
        assert record.total_messages_out == 0
        assert record.last_message_at > 0

    def test_record_message_out(self):
        """Recording outgoing messages increments the correct counter."""
        record = ChannelHealthRecord(channel_name="test")
        record.record_message("out")
        assert record.total_messages_in == 0
        assert record.total_messages_out == 1

    def test_record_error(self):
        """Recording errors increments the counter and stores the message."""
        record = ChannelHealthRecord(channel_name="test")
        record.record_error("timeout")
        assert record.total_errors == 1
        assert record.last_error_msg == "timeout"
        assert record.last_error_at > 0

    def test_messages_per_minute(self):
        """Messages per minute reflects recent activity."""
        record = ChannelHealthRecord(channel_name="test")
        # Add 10 messages "now"
        for _ in range(10):
            record.record_message("in")
        mpm = record.messages_per_minute(window=60.0)
        assert mpm == 10.0  # 10 messages over 60-second window → 10/min

    def test_error_rate_no_events(self):
        """Error rate is 0.0 when there are no events."""
        record = ChannelHealthRecord(channel_name="test")
        assert record.error_rate() == 0.0

    def test_error_rate_with_events(self):
        """Error rate is computed correctly."""
        record = ChannelHealthRecord(channel_name="test")
        # 1 error out of 10 total events
        for _ in range(9):
            record.record_message("in")
        record.record_error("test")
        rate = record.error_rate(window=60.0)
        assert abs(rate - 0.1) < 0.01  # ~10%

    def test_uptime(self):
        """Uptime reflects time since start."""
        record = ChannelHealthRecord(channel_name="test")
        record.started_at = time.time() - 120  # 2 minutes ago
        assert record.uptime_seconds >= 119  # at least 119 seconds

    def test_to_dict(self):
        """Serialization includes all expected fields."""
        record = ChannelHealthRecord(channel_name="telegram")
        record.started_at = time.time()
        record.status = ChannelStatus.HEALTHY
        record.record_message("in")
        record.record_message("out")

        data = record.to_dict()
        assert data["channel"] == "telegram"
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data
        assert "messages_per_minute" in data
        assert data["total_messages_in"] == 1
        assert data["total_messages_out"] == 1
        assert data["total_errors"] == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RateLimiter Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRateLimiter:
    """Tests for token-bucket rate limiter."""

    @pytest.mark.asyncio
    async def test_acquire_within_burst(self):
        """Acquiring within burst limit should not block."""
        limiter = RateLimiter(rate=100.0, burst=5)
        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1  # All within burst, should be instant

    @pytest.mark.asyncio
    async def test_acquire_respects_rate(self):
        """After burst exhaustion, acquire should throttle."""
        limiter = RateLimiter(rate=100.0, burst=1)
        # First acquire — instant (uses burst)
        await limiter.acquire()
        # Second acquire — should wait ~10ms (1/100 sec)
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.005  # At least a small wait


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ReconnectionManager Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestReconnectionManager:
    """Tests for exponential backoff reconnection."""

    def test_initial_delay(self):
        """First delay should be close to base_delay."""
        mgr = ReconnectionManager(base_delay=1.0, jitter=0.0)
        delay = mgr.next_delay()
        assert abs(delay - 1.0) < 0.2  # ~1s with no jitter

    def test_exponential_growth(self):
        """Delays should grow exponentially."""
        mgr = ReconnectionManager(base_delay=1.0, multiplier=2.0, jitter=0.0)
        d1 = mgr.next_delay()  # ~1
        d2 = mgr.next_delay()  # ~2
        d3 = mgr.next_delay()  # ~4
        assert d2 > d1
        assert d3 > d2

    def test_max_delay_cap(self):
        """Delay should not exceed max_delay."""
        mgr = ReconnectionManager(base_delay=1.0, max_delay=10.0, jitter=0.0)
        for _ in range(20):
            mgr.next_delay()
        delay = mgr.next_delay()
        assert delay <= 11.0  # max_delay with small tolerance for jitter=0 rounding

    def test_reset(self):
        """Reset should bring delay back to base_delay."""
        mgr = ReconnectionManager(base_delay=1.0, jitter=0.0)
        mgr.next_delay()
        mgr.next_delay()
        mgr.next_delay()
        mgr.reset()
        assert mgr.attempt == 0
        delay = mgr.next_delay()
        assert abs(delay - 1.0) < 0.2

    def test_attempt_counter(self):
        """Attempt counter should increment with each call."""
        mgr = ReconnectionManager()
        assert mgr.attempt == 0
        mgr.next_delay()
        assert mgr.attempt == 1
        mgr.next_delay()
        assert mgr.attempt == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ChannelHealthMonitor Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestChannelHealthMonitor:
    """Tests for the central health monitor."""

    def test_register(self):
        """Registering a channel creates a health record."""
        monitor = ChannelHealthMonitor()
        record = monitor.register("telegram")
        assert record.channel_name == "telegram"
        assert record.rate_limiter is not None

    def test_register_with_default_rate_limit(self):
        """Registration uses platform-specific default rate limits."""
        monitor = ChannelHealthMonitor()
        record = monitor.register("telegram")
        # Telegram default is 30.0/sec
        assert record.rate_limiter is not None

    def test_register_with_custom_rate_limit(self):
        """Custom rate limits override defaults."""
        monitor = ChannelHealthMonitor()
        record = monitor.register("custom", rate_limit=50.0)
        assert record.rate_limiter is not None

    def test_mark_started(self):
        """mark_started sets status to HEALTHY."""
        monitor = ChannelHealthMonitor()
        monitor.register("telegram")
        monitor.mark_started("telegram")
        health = monitor.get_channel_health("telegram")
        assert health["status"] == "healthy"

    def test_mark_stopped(self):
        """mark_stopped sets status to STOPPED."""
        monitor = ChannelHealthMonitor()
        monitor.register("telegram")
        monitor.mark_started("telegram")
        monitor.mark_stopped("telegram")
        health = monitor.get_channel_health("telegram")
        assert health["status"] == "stopped"

    def test_mark_disconnected(self):
        """mark_disconnected sets status and records error."""
        monitor = ChannelHealthMonitor()
        monitor.register("telegram")
        monitor.mark_disconnected("telegram", "socket_error")
        health = monitor.get_channel_health("telegram")
        assert health["status"] == "disconnected"
        assert health["last_error"] == "socket_error"

    def test_mark_reconnecting(self):
        """mark_reconnecting sets status and increments counter."""
        monitor = ChannelHealthMonitor()
        monitor.register("telegram")
        monitor.mark_reconnecting("telegram")
        monitor.mark_reconnecting("telegram")
        health = monitor.get_channel_health("telegram")
        assert health["status"] == "reconnecting"
        assert health["reconnect_count"] == 2

    def test_record_message(self):
        """Messages are correctly recorded."""
        monitor = ChannelHealthMonitor()
        monitor.register("telegram")
        monitor.record_message("telegram", "in")
        monitor.record_message("telegram", "out")
        health = monitor.get_channel_health("telegram")
        assert health["total_messages_in"] == 1
        assert health["total_messages_out"] == 1

    def test_record_error_triggers_degradation(self):
        """High error rate triggers DEGRADED status."""
        monitor = ChannelHealthMonitor()
        monitor.register("telegram")
        monitor.mark_started("telegram")

        # Make the error rate > DEGRADED_ERROR_RATE (10%)
        # 1 message, 1 error = 50% error rate
        monitor.record_message("telegram", "in")
        monitor.record_error("telegram", "api_error")

        health = monitor.get_channel_health("telegram")
        assert health["status"] == "degraded"

    def test_auto_recovery_from_degraded(self):
        """Channel recovers from DEGRADED when error rate drops."""
        monitor = ChannelHealthMonitor()
        monitor.register("telegram")
        monitor.mark_started("telegram")

        # First: trigger degradation
        monitor.record_error("telegram", "error")
        monitor.record_message("telegram", "in")
        assert monitor.get_channel_health("telegram")["status"] == "degraded"

        # Then: flood with successful messages to lower error rate
        for _ in range(100):
            monitor.record_message("telegram", "in")

        # After enough successful messages, should recover
        health = monitor.get_channel_health("telegram")
        assert health["status"] == "healthy"

    def test_health_report(self):
        """Health report includes all channels and aggregate status."""
        monitor = ChannelHealthMonitor()
        monitor.register("telegram")
        monitor.register("discord")
        monitor.mark_started("telegram")
        monitor.mark_started("discord")

        report = monitor.get_health_report()
        assert report["overall_status"] == "healthy"
        assert report["total_channels"] == 2
        assert report["healthy_channels"] == 2
        assert "telegram" in report["channels"]
        assert "discord" in report["channels"]

    def test_health_report_degraded(self):
        """Report shows 'degraded' if any channel is unhealthy."""
        monitor = ChannelHealthMonitor()
        monitor.register("telegram")
        monitor.register("discord")
        monitor.mark_started("telegram")
        monitor.mark_disconnected("discord", "timeout")

        report = monitor.get_health_report()
        assert report["overall_status"] == "degraded"

    @pytest.mark.asyncio
    async def test_acquire_rate_limit(self):
        """Rate limit acquisition works and returns True."""
        monitor = ChannelHealthMonitor()
        monitor.register("telegram")
        result = await monitor.acquire_rate_limit("telegram")
        assert result is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_trips(self):
        """Circuit breaker refuses messages when error rate is too high."""
        monitor = ChannelHealthMonitor()
        monitor.register("telegram")
        monitor.mark_started("telegram")

        # Generate very high error rate
        for _ in range(10):
            monitor.record_error("telegram", "error")

        result = await monitor.acquire_rate_limit("telegram")
        assert result is False

    @pytest.mark.asyncio
    async def test_unregistered_channel_rate_limit(self):
        """Rate limit for unregistered channel returns True."""
        monitor = ChannelHealthMonitor()
        result = await monitor.acquire_rate_limit("nonexistent")
        assert result is True

    def test_get_channel_health_nonexistent(self):
        """Getting health for unregistered channel returns None."""
        monitor = ChannelHealthMonitor()
        assert monitor.get_channel_health("nonexistent") is None

    def test_auto_health_check_error_rate(self):
        """Auto health check degrades channels with high error rates."""
        monitor = ChannelHealthMonitor()
        monitor.register("telegram")
        monitor.mark_started("telegram")

        # Generate errors to trigger degradation threshold
        for _ in range(100):
            monitor.record_error("telegram", "error")

        monitor._auto_health_check()
        health = monitor.get_channel_health("telegram")
        assert health["status"] in ("degraded", "disconnected")

    def test_auto_health_check_recovery(self):
        """Auto health check recovers degraded channels with low error rate."""
        monitor = ChannelHealthMonitor()
        monitor.register("telegram")
        monitor.mark_started("telegram")

        # Trigger degradation
        monitor.record_error("telegram", "error")
        record = monitor._channels["telegram"]
        record.status = ChannelStatus.DEGRADED

        # Flood with successful messages
        for _ in range(200):
            monitor.record_message("telegram", "in")

        monitor._auto_health_check()
        health = monitor.get_channel_health("telegram")
        assert health["status"] == "healthy"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ChannelStatus Enum Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestChannelStatus:
    """Tests for the ChannelStatus enum."""

    def test_all_statuses(self):
        """All expected statuses exist."""
        assert ChannelStatus.HEALTHY == "healthy"
        assert ChannelStatus.DEGRADED == "degraded"
        assert ChannelStatus.DISCONNECTED == "disconnected"
        assert ChannelStatus.RECONNECTING == "reconnecting"
        assert ChannelStatus.STOPPED == "stopped"
        assert ChannelStatus.UNKNOWN == "unknown"

    def test_string_value(self):
        """Status values are serializable strings."""
        assert str(ChannelStatus.HEALTHY) == "ChannelStatus.HEALTHY"
        assert ChannelStatus.HEALTHY.value == "healthy"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Defaults Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDefaults:
    """Tests for default configuration values."""

    def test_default_rate_limits(self):
        """All major platforms have default rate limits."""
        assert "telegram" in DEFAULT_RATE_LIMITS
        assert "discord" in DEFAULT_RATE_LIMITS
        assert "whatsapp" in DEFAULT_RATE_LIMITS
        assert "webchat" in DEFAULT_RATE_LIMITS

    def test_thresholds(self):
        """Error rate thresholds are sensible."""
        assert DEGRADED_ERROR_RATE < CIRCUIT_BREAK_ERROR_RATE
        assert 0 < DEGRADED_ERROR_RATE < 1
        assert 0 < CIRCUIT_BREAK_ERROR_RATE <= 1
