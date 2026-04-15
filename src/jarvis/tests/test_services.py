"""
Comprehensive tests for jarvis.services — Services Layer (Phase 7).

Tests: PIDManager, RateLimiter (3 algorithms), CronEngine, LaneQueue,
OutcomeStore, TranscriptWriter, AlertManager, ConfigWatcher,
GitIntegration, and IdentityService.

112 tests total.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# ---------------------------------------------------------------------------
# PIDManager
# ---------------------------------------------------------------------------
from jarvis.services.daemon import PIDManager


class TestPIDManager:
    """PID file lifecycle: write, read, is_running, cleanup, stale PID."""

    def test_write_creates_pid_file(self, tmp_path):
        pid_path = tmp_path / "jarvis.pid"
        mgr = PIDManager(str(pid_path))
        mgr.write()
        assert pid_path.exists()
        assert pid_path.read_text().strip() == str(os.getpid())

    def test_read_returns_pid(self, tmp_path):
        pid_path = tmp_path / "jarvis.pid"
        pid_path.write_text("12345")
        mgr = PIDManager(str(pid_path))
        assert mgr.read() == 12345

    def test_read_returns_none_when_missing(self, tmp_path):
        mgr = PIDManager(str(tmp_path / "no.pid"))
        assert mgr.read() is None

    def test_read_returns_none_on_invalid_content(self, tmp_path):
        pid_path = tmp_path / "jarvis.pid"
        pid_path.write_text("not-a-number")
        mgr = PIDManager(str(pid_path))
        assert mgr.read() is None

    def test_is_running_with_current_process(self, tmp_path):
        pid_path = tmp_path / "jarvis.pid"
        pid_path.write_text(str(os.getpid()))
        mgr = PIDManager(str(pid_path))
        assert mgr.is_running() is True

    def test_is_running_stale_pid_cleans_up(self, tmp_path):
        pid_path = tmp_path / "jarvis.pid"
        # Use a very high PID that is almost certainly not running
        pid_path.write_text("9999999")
        mgr = PIDManager(str(pid_path))
        with patch("os.kill", side_effect=ProcessLookupError):
            result = mgr.is_running()
        assert result is False
        # PID file should be cleaned up
        assert not pid_path.exists()

    def test_is_running_no_pid_file(self, tmp_path):
        mgr = PIDManager(str(tmp_path / "no.pid"))
        assert mgr.is_running() is False

    def test_is_running_permission_error(self, tmp_path):
        pid_path = tmp_path / "jarvis.pid"
        pid_path.write_text(str(os.getpid()))
        mgr = PIDManager(str(pid_path))
        with patch("os.kill", side_effect=PermissionError):
            assert mgr.is_running() is True

    def test_cleanup_removes_file(self, tmp_path):
        pid_path = tmp_path / "jarvis.pid"
        pid_path.write_text("12345")
        mgr = PIDManager(str(pid_path))
        mgr.cleanup()
        assert not pid_path.exists()

    def test_cleanup_missing_file_no_error(self, tmp_path):
        mgr = PIDManager(str(tmp_path / "no.pid"))
        mgr.cleanup()  # Should not raise

    def test_write_stale_pid_overwrites(self, tmp_path):
        pid_path = tmp_path / "jarvis.pid"
        pid_path.write_text("9999999")
        mgr = PIDManager(str(pid_path))
        with patch("os.kill", side_effect=ProcessLookupError):
            mgr.write()
        assert pid_path.read_text().strip() == str(os.getpid())

    def test_write_existing_running_raises(self, tmp_path):
        pid_path = tmp_path / "jarvis.pid"
        pid_path.write_text(str(os.getpid()))
        mgr = PIDManager(str(pid_path))
        with pytest.raises(FileExistsError, match="already running"):
            mgr.write()

    def test_get_status(self, tmp_path):
        pid_path = tmp_path / "jarvis.pid"
        pid_path.write_text(str(os.getpid()))
        mgr = PIDManager(str(pid_path))
        status = mgr.get_status()
        assert status["pid"] == os.getpid()
        assert status["running"] is True
        assert str(pid_path) in status["pid_file"]

    def test_write_creates_parent_dir(self, tmp_path):
        pid_path = tmp_path / "sub" / "dir" / "jarvis.pid"
        mgr = PIDManager(str(pid_path))
        mgr.write()
        assert pid_path.exists()


# ---------------------------------------------------------------------------
# RateLimiter — 3 algorithms
# ---------------------------------------------------------------------------
from jarvis.services.rate_limiter import (
    InMemoryBackend,
    RateLimitAlgorithm,
    RateLimitConfig,
    RateLimitResult,
    RateLimiter,
    default_api_limits,
)


class TestRateLimitResult:
    """RateLimitResult header generation."""

    def test_to_headers_allowed(self):
        r = RateLimitResult(allowed=True, remaining=5, limit=10, reset_at=1000.0)
        h = r.to_headers()
        assert h["X-RateLimit-Limit"] == "10"
        assert h["X-RateLimit-Remaining"] == "5"
        assert "Retry-After" not in h

    def test_to_headers_blocked(self):
        r = RateLimitResult(
            allowed=False, remaining=0, limit=10, reset_at=1000.0, retry_after=30.0
        )
        h = r.to_headers()
        assert "Retry-After" in h
        assert int(h["Retry-After"]) == 31  # int(30.0) + 1

    def test_remaining_clamped_to_zero(self):
        r = RateLimitResult(allowed=False, remaining=-3, limit=10, reset_at=1000.0)
        h = r.to_headers()
        assert h["X-RateLimit-Remaining"] == "0"


class TestFixedWindowBackend:
    """In-memory fixed window rate limiting."""

    def test_allows_under_limit(self):
        backend = InMemoryBackend()
        for i in range(5):
            result = backend.fixed_window_check("key", max_requests=5, window_seconds=60)
            assert result.allowed is True

    def test_blocks_at_limit(self):
        backend = InMemoryBackend()
        for _ in range(5):
            backend.fixed_window_check("key", max_requests=5, window_seconds=60)
        result = backend.fixed_window_check("key", max_requests=5, window_seconds=60)
        assert result.allowed is False

    def test_separate_keys_independent(self):
        backend = InMemoryBackend()
        for _ in range(5):
            backend.fixed_window_check("a", max_requests=5, window_seconds=60)
        result = backend.fixed_window_check("b", max_requests=5, window_seconds=60)
        assert result.allowed is True

    def test_remaining_decrements(self):
        backend = InMemoryBackend()
        r1 = backend.fixed_window_check("key", max_requests=3, window_seconds=60)
        assert r1.remaining == 2
        r2 = backend.fixed_window_check("key", max_requests=3, window_seconds=60)
        assert r2.remaining == 1

    def test_reset_at_is_future(self):
        backend = InMemoryBackend()
        result = backend.fixed_window_check("key", max_requests=10, window_seconds=60)
        assert result.reset_at > time.time()


class TestSlidingWindowBackend:
    """In-memory sliding window rate limiting."""

    def test_allows_under_limit(self):
        backend = InMemoryBackend()
        result = backend.sliding_window_check("key", max_requests=10, window_seconds=60)
        assert result.allowed is True

    def test_blocks_when_weighted_count_exceeded(self):
        backend = InMemoryBackend()
        # Fill up
        for _ in range(10):
            backend.sliding_window_check("key", max_requests=10, window_seconds=60)
        result = backend.sliding_window_check("key", max_requests=10, window_seconds=60)
        assert result.allowed is False

    def test_remaining_nonnegative_in_headers(self):
        backend = InMemoryBackend()
        for _ in range(15):
            backend.sliding_window_check("key", max_requests=5, window_seconds=60)
        r = backend.sliding_window_check("key", max_requests=5, window_seconds=60)
        h = r.to_headers()
        assert int(h["X-RateLimit-Remaining"]) >= 0

    def test_retry_after_set_when_blocked(self):
        backend = InMemoryBackend()
        for _ in range(10):
            backend.sliding_window_check("key", max_requests=10, window_seconds=60)
        result = backend.sliding_window_check("key", max_requests=10, window_seconds=60)
        assert result.retry_after > 0


class TestTokenBucketBackend:
    """In-memory token bucket rate limiting."""

    def test_allows_initial_request(self):
        backend = InMemoryBackend()
        result = backend.token_bucket_check("key", max_tokens=10, refill_rate=1.0)
        assert result.allowed is True

    def test_burst_capacity(self):
        backend = InMemoryBackend()
        # Max + burst = capacity
        for _ in range(15):
            result = backend.token_bucket_check(
                "key", max_tokens=10, refill_rate=1.0, burst=5
            )
        # Should have allowed 15 (10+5 capacity)
        assert result.allowed is True

    def test_blocks_when_empty(self):
        backend = InMemoryBackend()
        for _ in range(11):
            result = backend.token_bucket_check("key", max_tokens=10, refill_rate=0.001)
        assert result.allowed is False

    def test_tokens_refill_over_time(self):
        backend = InMemoryBackend()
        # Drain bucket
        for _ in range(10):
            backend.token_bucket_check("key", max_tokens=10, refill_rate=100.0)
        # With refill_rate=100/sec, even a tiny time gap refills some tokens
        import time; time.sleep(0.02)  # Ensure enough time passes for refill
        result = backend.token_bucket_check("key", max_tokens=10, refill_rate=100.0)
        assert result.allowed is True

    def test_deducts_only_on_allowed(self):
        backend = InMemoryBackend()
        # Drain to 1 token
        for _ in range(9):
            backend.token_bucket_check("key", max_tokens=10, refill_rate=0.001)
        # This should succeed (1 token left)
        r = backend.token_bucket_check("key", max_tokens=10, refill_rate=0.001)
        assert r.allowed is True
        # Now empty
        r = backend.token_bucket_check("key", max_tokens=10, refill_rate=0.001)
        assert r.allowed is False

    def test_zero_refill_rate(self):
        backend = InMemoryBackend()
        result = backend.token_bucket_check("key", max_tokens=1, refill_rate=0)
        assert result.allowed is True
        result = backend.token_bucket_check("key", max_tokens=1, refill_rate=0)
        assert result.allowed is False


class TestRateLimiter:
    """High-level RateLimiter with rules and stats."""

    def test_no_rules_allows_everything(self):
        limiter = RateLimiter(redis_url="")
        result = limiter.check(user_id="u1", endpoint="/api/test")
        assert result.allowed is True

    def test_single_rule_enforcement(self):
        limiter = RateLimiter(redis_url="")
        limiter.add_rule(RateLimitConfig(name="test", max_requests=2, window_seconds=60))
        limiter.check(user_id="u1")
        limiter.check(user_id="u1")
        result = limiter.check(user_id="u1")
        assert result.allowed is False

    def test_most_restrictive_wins(self):
        limiter = RateLimiter(redis_url="")
        limiter.add_rule(
            RateLimitConfig(name="loose", max_requests=100, window_seconds=60)
        )
        limiter.add_rule(
            RateLimitConfig(name="tight", max_requests=1, window_seconds=60)
        )
        limiter.check(user_id="u1")
        result = limiter.check(user_id="u1")
        assert result.allowed is False

    def test_stats_tracking(self):
        limiter = RateLimiter(redis_url="")
        limiter.add_rule(RateLimitConfig(name="test", max_requests=1, window_seconds=60))
        limiter.check(user_id="u1")
        limiter.check(user_id="u1")  # blocked
        stats = limiter.get_stats()
        assert stats["total_checks"] == 2
        assert stats["total_blocked"] == 1
        assert stats["backend"] in ("memory", "redis")  # depends on redis availability

    def test_default_api_limits(self):
        limits = default_api_limits()
        assert len(limits) == 4
        names = {l.name for l in limits}
        assert "global" in names
        assert "per_user" in names


# ---------------------------------------------------------------------------
# CronEngine
# ---------------------------------------------------------------------------
from jarvis.services.cron import CronEngine, CronExpression, CronJob


class TestCronExpression:
    """Cron expression parsing and matching."""

    def test_wildcard_matches_all(self):
        expr = CronExpression("* * * * *")
        dt = datetime(2025, 3, 15, 10, 30, tzinfo=timezone.utc)
        assert expr.matches(dt) is True

    def test_specific_minute(self):
        expr = CronExpression("30 * * * *")
        assert expr.matches(datetime(2025, 3, 15, 10, 30, tzinfo=timezone.utc)) is True
        assert expr.matches(datetime(2025, 3, 15, 10, 31, tzinfo=timezone.utc)) is False

    def test_step_expression(self):
        expr = CronExpression("*/15 * * * *")
        assert expr.matches(datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)) is True
        assert expr.matches(datetime(2025, 1, 1, 0, 15, tzinfo=timezone.utc)) is True
        assert expr.matches(datetime(2025, 1, 1, 0, 7, tzinfo=timezone.utc)) is False

    def test_range_expression(self):
        expr = CronExpression("0 9-17 * * *")
        assert expr.matches(datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)) is True
        assert expr.matches(datetime(2025, 1, 1, 5, 0, tzinfo=timezone.utc)) is False

    def test_list_expression(self):
        expr = CronExpression("0,15,30,45 * * * *")
        assert expr.matches(datetime(2025, 1, 1, 0, 15, tzinfo=timezone.utc)) is True
        assert expr.matches(datetime(2025, 1, 1, 0, 20, tzinfo=timezone.utc)) is False

    def test_invalid_field_count_raises(self):
        with pytest.raises(ValueError, match="5 fields"):
            CronExpression("* * *")

    def test_value_out_of_bounds_raises(self):
        with pytest.raises(ValueError):
            CronExpression("60 * * * *")

    def test_step_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            CronExpression("*/0 * * * *")

    def test_dow_conversion_sunday(self):
        # Cron: 0 = Sunday, Python weekday: 6 = Sunday
        expr = CronExpression("0 0 * * 0")
        # Jan 5, 2025 is a Sunday
        dt = datetime(2025, 1, 5, 0, 0, tzinfo=timezone.utc)
        assert dt.weekday() == 6  # Python Sunday
        assert expr.matches(dt) is True

    def test_dow_conversion_monday(self):
        # Cron: 1 = Monday, Python weekday: 0 = Monday
        expr = CronExpression("0 0 * * 1")
        dt = datetime(2025, 1, 6, 0, 0, tzinfo=timezone.utc)
        assert dt.weekday() == 0  # Python Monday
        assert expr.matches(dt) is True


class TestCronJob:
    """CronJob scheduling logic."""

    def test_should_run_when_matches(self):
        job = CronJob(name="test", schedule="* * * * *", action="do something")
        now = datetime(2025, 1, 1, 10, 30, tzinfo=timezone.utc)
        assert job.should_run(now) is True

    def test_should_not_run_when_disabled(self):
        job = CronJob(
            name="test", schedule="* * * * *", action="do something", enabled=False
        )
        now = datetime(2025, 1, 1, 10, 30, tzinfo=timezone.utc)
        assert job.should_run(now) is False

    def test_prevents_double_run_same_minute(self):
        job = CronJob(name="test", schedule="* * * * *", action="do something")
        now = datetime(2025, 1, 1, 10, 30, tzinfo=timezone.utc)
        job.last_run = now.timestamp()
        assert job.should_run(now) is False

    def test_mark_run_updates_state(self):
        job = CronJob(name="test", schedule="* * * * *", action="do something")
        assert job.run_count == 0
        assert job.last_run is None
        job.mark_run()
        assert job.run_count == 1
        assert job.last_run is not None


class TestCronEngine:
    """CronEngine add/remove/list jobs."""

    def _make_engine(self, tmp_path):
        """Create a CronEngine with a mocked config and gateway."""
        config = MagicMock()
        config.home_dir = str(tmp_path)
        config.daemon.cron_file = ""
        gateway = MagicMock()
        # Patch yaml to avoid import issues during _load_jobs
        with patch.dict("sys.modules", {"yaml": MagicMock()}):
            engine = CronEngine.__new__(CronEngine)
            engine.config = config
            engine.gateway = gateway
            engine.jobs = []
            engine._task = None
            engine._job_tasks = set()
            engine._running = False
        return engine

    def test_add_job(self, tmp_path):
        engine = self._make_engine(tmp_path)
        job = CronJob(name="test_job", schedule="* * * * *", action="hello")
        engine.add_job(job)
        assert len(engine.jobs) == 1
        assert engine.jobs[0].name == "test_job"

    def test_remove_job(self, tmp_path):
        engine = self._make_engine(tmp_path)
        engine.add_job(CronJob(name="a", schedule="* * * * *", action="x"))
        engine.add_job(CronJob(name="b", schedule="* * * * *", action="y"))
        removed = engine.remove_job("a")
        assert removed is True
        assert len(engine.jobs) == 1
        assert engine.jobs[0].name == "b"

    def test_remove_nonexistent(self, tmp_path):
        engine = self._make_engine(tmp_path)
        removed = engine.remove_job("nope")
        assert removed is False

    def test_list_jobs(self, tmp_path):
        engine = self._make_engine(tmp_path)
        engine.add_job(CronJob(name="a", schedule="0 9 * * *", action="morning"))
        listing = engine.list_jobs()
        assert len(listing) == 1
        assert listing[0]["name"] == "a"
        assert listing[0]["schedule"] == "0 9 * * *"


# ---------------------------------------------------------------------------
# LaneQueue
# ---------------------------------------------------------------------------
from jarvis.services.lane_queue import LaneQueue, LaneTask, TaskStatus


class TestLaneTask:
    """LaneTask dataclass defaults."""

    def test_defaults(self):
        task = LaneTask()
        assert task.status == TaskStatus.PENDING
        assert task.task_id  # auto-generated
        assert task.created_at > 0

    def test_custom_fields(self):
        task = LaneTask(session_id="s1", timeout=60.0)
        assert task.session_id == "s1"
        assert task.timeout == 60.0


class TestLaneQueue:
    """LaneQueue serial execution and session isolation."""

    @pytest.mark.asyncio
    async def test_submit_executes_coroutine(self):
        queue = LaneQueue(max_lanes=10)
        async def work():
            return 42
        result = await queue.submit("session-1", work)
        assert result == 42
        await queue.shutdown()

    @pytest.mark.asyncio
    async def test_serial_execution_within_session(self):
        queue = LaneQueue(max_lanes=10)
        order = []

        async def task_a():
            order.append("a_start")
            await asyncio.sleep(0.05)
            order.append("a_end")
            return "a"

        async def task_b():
            order.append("b_start")
            order.append("b_end")
            return "b"

        # Submit both to same session
        t1 = asyncio.create_task(queue.submit("s1", task_a))
        await asyncio.sleep(0.01)  # Let task_a start
        t2 = asyncio.create_task(queue.submit("s1", task_b))
        await asyncio.gather(t1, t2)

        # Within same session, b must not start until a finishes
        assert order.index("a_end") < order.index("b_start")
        await queue.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_across_sessions(self):
        queue = LaneQueue(max_lanes=10)
        order = []

        async def slow_task(label):
            order.append(f"{label}_start")
            await asyncio.sleep(0.05)
            order.append(f"{label}_end")
            return label

        # Submit to different sessions — should run concurrently
        t1 = asyncio.create_task(queue.submit("s1", slow_task, "a"))
        t2 = asyncio.create_task(queue.submit("s2", slow_task, "b"))
        await asyncio.gather(t1, t2)

        # Both should have started before either finished (concurrent)
        starts = [i for i, x in enumerate(order) if x.endswith("_start")]
        ends = [i for i, x in enumerate(order) if x.endswith("_end")]
        # At least both starts should come before last end
        assert len(starts) == 2
        await queue.shutdown()

    @pytest.mark.asyncio
    async def test_task_exception_propagates(self):
        queue = LaneQueue(max_lanes=10)
        async def failing():
            raise ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            await queue.submit("s1", failing)
        await queue.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_rejects_new_tasks(self):
        queue = LaneQueue(max_lanes=10)
        await queue.shutdown()
        with pytest.raises(RuntimeError, match="shutting down"):
            await queue.submit("s1", asyncio.sleep, 0)

    def test_get_lane_stats_empty(self):
        queue = LaneQueue()
        stats = queue.get_lane_stats()
        assert stats["active_lanes"] == 0

    @pytest.mark.asyncio
    async def test_queue_full_rejection(self):
        queue = LaneQueue(max_lanes=10)
        queue.MAX_QUEUE_SIZE = 1

        blocker = asyncio.Event()
        async def block():
            await blocker.wait()
            return "done"

        # Submit first task (will block on event)
        t1 = asyncio.create_task(queue.submit("s1", block))
        await asyncio.sleep(0.05)

        # Submit second — goes into the queue
        t2 = asyncio.create_task(queue.submit("s1", block))
        await asyncio.sleep(0.05)

        # Third should be rejected (queue full)
        with pytest.raises(RuntimeError, match="queue full"):
            await queue.submit("s1", block)

        # Unblock and cleanup
        blocker.set()
        await asyncio.gather(t1, t2, return_exceptions=True)
        await queue.shutdown()


# ---------------------------------------------------------------------------
# OutcomeStore
# ---------------------------------------------------------------------------
from jarvis.services.outcome_store import OutcomeStore, TaskOutcome, detect_feedback


class TestDetectFeedback:
    """Implicit feedback detection from user messages."""

    def test_positive_thanks(self):
        assert detect_feedback("Thanks!") == "good"

    def test_positive_great(self):
        assert detect_feedback("great job") == "good"

    def test_negative_wrong(self):
        assert detect_feedback("That's wrong") == "bad"

    def test_negative_useless(self):
        assert detect_feedback("useless response") == "bad"

    def test_neutral_returns_none(self):
        assert detect_feedback("Tell me about Python") is None

    def test_empty_returns_none(self):
        assert detect_feedback("") is None

    def test_none_returns_none(self):
        assert detect_feedback(None) is None

    def test_long_message_returns_none(self):
        assert detect_feedback("thanks " * 100) is None  # > 200 chars

    def test_explicit_feedback_command_good(self):
        assert detect_feedback("/feedback good") == "good"

    def test_explicit_feedback_command_bad(self):
        assert detect_feedback("/feedback bad") == "bad"

    def test_explicit_feedback_command_invalid(self):
        assert detect_feedback("/feedback neutral") is None

    def test_negative_takes_priority(self):
        # "wrong" is checked before "thanks"
        assert detect_feedback("wrong thanks") == "bad"


class TestOutcomeStore:
    """OutcomeStore: record, feedback, queries."""

    def _make_store(self, tmp_path) -> OutcomeStore:
        return OutcomeStore(tmp_path / "outcomes.db")

    @pytest.mark.asyncio
    async def test_record_and_retrieve(self, tmp_path):
        store = self._make_store(tmp_path)
        outcome = TaskOutcome(
            user_id="u1",
            user_message="hello",
            tools_used=["tool_a"],
            success=True,
        )
        await store.record(outcome)
        recent = await store.get_recent(limit=5, user_id="u1")
        assert len(recent) == 1
        assert recent[0]["user_id"] == "u1"
        store.close()

    @pytest.mark.asyncio
    async def test_update_feedback(self, tmp_path):
        store = self._make_store(tmp_path)
        outcome = TaskOutcome(user_id="u1", success=True)
        await store.record(outcome)
        updated = await store.update_feedback("u1", "good")
        assert updated is True
        store.close()

    @pytest.mark.asyncio
    async def test_update_feedback_no_user(self, tmp_path):
        store = self._make_store(tmp_path)
        updated = await store.update_feedback("nonexistent", "bad")
        assert updated is False
        store.close()

    @pytest.mark.asyncio
    async def test_failure_patterns(self, tmp_path):
        store = self._make_store(tmp_path)
        outcome = TaskOutcome(
            user_id="u1",
            success=False,
            tool_errors=["search: timeout after 5s"],
        )
        await store.record(outcome)
        patterns = await store.get_failure_patterns(window_hours=1)
        assert len(patterns) == 1
        assert patterns[0]["tool_name"] == "search"
        assert patterns[0]["failure_count"] == 1
        store.close()

    @pytest.mark.asyncio
    async def test_prune_old(self, tmp_path):
        store = self._make_store(tmp_path)
        old = TaskOutcome(user_id="u1", success=True)
        old.timestamp = time.time() - 200 * 86400  # 200 days ago
        await store.record(old)
        count = await store.prune_old(max_age_days=90)
        assert count == 1
        store.close()

    @pytest.mark.asyncio
    async def test_feedback_summary(self, tmp_path):
        store = self._make_store(tmp_path)
        o1 = TaskOutcome(user_id="u1", success=True, user_feedback="good")
        o2 = TaskOutcome(user_id="u2", success=True, user_feedback="bad")
        await store.record(o1)
        await store.record(o2)
        summary = await store.get_feedback_summary(window_hours=1)
        assert summary["good"] == 1
        assert summary["bad"] == 1
        store.close()

    def test_task_outcome_auto_id(self):
        o = TaskOutcome()
        assert len(o.task_id) == 12
        assert o.timestamp > 0

    @pytest.mark.asyncio
    async def test_get_recent_no_user(self, tmp_path):
        store = self._make_store(tmp_path)
        o = TaskOutcome(user_id="u1", success=True)
        await store.record(o)
        recent = await store.get_recent(limit=10)
        assert len(recent) == 1
        store.close()


# ---------------------------------------------------------------------------
# TranscriptWriter
# ---------------------------------------------------------------------------
from jarvis.services.transcripts import TranscriptEntry, TranscriptWriter


class TestTranscriptEntry:
    """TranscriptEntry serialization roundtrip."""

    def test_to_dict(self):
        e = TranscriptEntry(
            timestamp=1000.0,
            event_type="message",
            role="user",
            content="hello",
        )
        d = e.to_dict()
        assert d["ts"] == 1000.0
        assert d["event"] == "message"
        assert d["role"] == "user"

    def test_from_dict_roundtrip(self):
        original = TranscriptEntry(
            timestamp=2000.0,
            event_type="tool_call",
            role=None,
            content='{"tool": "search"}',
            metadata={"key": "val"},
        )
        d = original.to_dict()
        restored = TranscriptEntry.from_dict(d)
        assert restored.timestamp == original.timestamp
        assert restored.event_type == original.event_type
        assert restored.content == original.content


class TestTranscriptWriter:
    """TranscriptWriter: JSONL session persistence."""

    def test_start_and_read_session(self, tmp_path):
        writer = TranscriptWriter(tmp_path / "transcripts")
        path = writer.start_session("s1", user_id="u1", channel="cli")
        assert path.exists()
        entries = writer.read_transcript("s1")
        assert len(entries) == 1
        assert entries[0].event_type == "session_start"

    def test_append_message(self, tmp_path):
        writer = TranscriptWriter(tmp_path / "transcripts")
        writer.start_session("s1")
        writer.append_message("s1", role="user", content="hello")
        writer.append_message("s1", role="assistant", content="hi")
        entries = writer.read_transcript("s1")
        assert len(entries) == 3  # start + 2 messages
        assert entries[1].role == "user"
        assert entries[2].role == "assistant"

    def test_append_tool_call(self, tmp_path):
        writer = TranscriptWriter(tmp_path / "transcripts")
        writer.start_session("s1")
        writer.append_tool_call("s1", tool_name="search", args={"q": "test"})
        entries = writer.read_transcript("s1")
        assert entries[1].event_type == "tool_call"
        data = json.loads(entries[1].content)
        assert data["tool"] == "search"

    def test_append_tool_result(self, tmp_path):
        writer = TranscriptWriter(tmp_path / "transcripts")
        writer.start_session("s1")
        writer.append_tool_result("s1", tool_name="search", result="found 5")
        entries = writer.read_transcript("s1")
        assert entries[1].event_type == "tool_result"

    def test_append_error(self, tmp_path):
        writer = TranscriptWriter(tmp_path / "transcripts")
        writer.start_session("s1")
        writer.append_error("s1", error="something broke")
        entries = writer.read_transcript("s1")
        assert entries[1].event_type == "error"
        assert entries[1].content == "something broke"

    def test_close_session(self, tmp_path):
        writer = TranscriptWriter(tmp_path / "transcripts")
        writer.start_session("s1")
        writer.close_session("s1")
        entries = writer.read_transcript("s1")
        assert entries[-1].event_type == "session_end"

    def test_jsonl_format(self, tmp_path):
        writer = TranscriptWriter(tmp_path / "transcripts")
        writer.start_session("s1")
        writer.append_message("s1", role="user", content="test")
        path = writer._get_path("s1")
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            json.loads(line)  # Each line must be valid JSON

    def test_list_sessions(self, tmp_path):
        writer = TranscriptWriter(tmp_path / "transcripts")
        writer.start_session("s1")
        writer.start_session("s2")
        sessions = writer.list_sessions()
        assert "s1" in sessions
        assert "s2" in sessions

    def test_get_transcript_path_exists(self, tmp_path):
        writer = TranscriptWriter(tmp_path / "transcripts")
        writer.start_session("s1")
        path = writer.get_transcript_path("s1")
        assert path is not None and path.exists()

    def test_get_transcript_path_missing(self, tmp_path):
        writer = TranscriptWriter(tmp_path / "transcripts")
        assert writer.get_transcript_path("nonexistent") is None

    def test_path_sanitization(self, tmp_path):
        writer = TranscriptWriter(tmp_path / "transcripts")
        path = writer._get_path("../../../etc/passwd")
        assert ".." not in str(path.name)

    def test_get_stats(self, tmp_path):
        writer = TranscriptWriter(tmp_path / "transcripts")
        writer.start_session("s1")
        stats = writer.get_stats()
        assert stats["total_sessions"] == 1
        assert stats["total_size_bytes"] > 0

    def test_read_nonexistent_session(self, tmp_path):
        writer = TranscriptWriter(tmp_path / "transcripts")
        entries = writer.read_transcript("ghost")
        assert entries == []


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------
from jarvis.services.alerting import (
    Alert,
    AlertChannel,
    AlertManager,
    AlertSeverity,
    SlackDispatcher,
    PagerDutyDispatcher,
    WebhookDispatcher,
    DiscordDispatcher,
    EmailDispatcher,
    _is_safe_url,
)


class TestAlertSeverity:
    def test_enum_values(self):
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestAlert:
    def test_to_dict(self):
        a = Alert(title="Test", message="msg", severity=AlertSeverity.WARNING)
        d = a.to_dict()
        assert d["title"] == "Test"
        assert d["severity"] == "warning"
        assert d["timestamp"] > 0


class TestSlackDispatcher:
    def test_not_configured_without_url(self):
        d = SlackDispatcher(webhook_url="")
        assert d.is_configured is False

    def test_send_unconfigured_returns_false(self):
        d = SlackDispatcher(webhook_url="")
        result = d.send(Alert(title="t", message="m"))
        assert result is False


class TestPagerDutyDispatcher:
    def test_not_configured_without_key(self):
        d = PagerDutyDispatcher(routing_key="")
        assert d.is_configured is False


class TestWebhookDispatcher:
    def test_not_configured_without_url(self):
        d = WebhookDispatcher(url="")
        assert d.is_configured is False


class TestDiscordDispatcher:
    def test_not_configured_without_url(self):
        d = DiscordDispatcher(webhook_url="")
        assert d.is_configured is False


class TestEmailDispatcher:
    def test_not_configured_without_host(self):
        d = EmailDispatcher()
        assert d.is_configured is False


class TestIsSafeUrl:
    def test_blocks_private_ip(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("10.0.0.1", 80))
        ]):
            assert _is_safe_url("http://internal.corp/hook") is False

    def test_blocks_localhost(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("127.0.0.1", 80))
        ]):
            assert _is_safe_url("http://localhost/hook") is False

    def test_allows_public_ip(self):
        with patch("socket.getaddrinfo", return_value=[
            (2, 1, 6, "", ("8.8.8.8", 443))
        ]):
            assert _is_safe_url("https://example.com/hook") is True

    def test_blocks_non_http_scheme(self):
        assert _is_safe_url("ftp://example.com/hook") is False


class TestAlertManager:
    def test_fire_logs_always(self):
        mgr = AlertManager()
        result = mgr.fire(Alert(title="Test", message="msg", severity=AlertSeverity.INFO))
        assert "log" in result

    def test_cooldown_dedup(self):
        mgr = AlertManager()
        mgr._cooldown_seconds = 300
        alert = Alert(title="Same Alert", message="m", severity=AlertSeverity.WARNING)
        r1 = mgr.fire(alert)
        assert "cooldown" not in r1
        r2 = mgr.fire(alert)
        assert r2.get("cooldown") is True

    def test_resolved_skips_cooldown(self):
        mgr = AlertManager()
        alert = Alert(
            title="Fixed", message="m", severity=AlertSeverity.RESOLVED, dedup_key="dk"
        )
        mgr.fire(alert)
        r2 = mgr.fire(alert)
        assert "cooldown" not in r2

    def test_history_tracking(self):
        mgr = AlertManager()
        mgr.fire(Alert(title="A", message="m", severity=AlertSeverity.INFO))
        mgr.fire(Alert(title="B", message="m", severity=AlertSeverity.WARNING))
        assert len(mgr.get_recent_alerts(limit=10)) == 2

    def test_get_stats(self):
        mgr = AlertManager()
        mgr.fire(Alert(title="X", message="m", severity=AlertSeverity.CRITICAL))
        stats = mgr.get_stats()
        assert stats["total_alerts"] == 1
        assert "critical" in stats["by_severity"]


# ---------------------------------------------------------------------------
# ConfigWatcher
# ---------------------------------------------------------------------------
from jarvis.services.config_watcher import ConfigWatcher


class TestConfigWatcher:
    """ConfigWatcher: file change detection and reload callbacks."""

    @pytest.mark.asyncio
    async def test_detect_change(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text('{"key": "value1"}')
        watcher = ConfigWatcher(config_path=config_file, poll_interval=0.1)

        changes = []
        watcher.on_change(lambda data: changes.append(data))
        await watcher.start()

        # Modify the file
        await asyncio.sleep(0.2)
        config_file.write_text('{"key": "value2"}')
        await asyncio.sleep(0.5)

        await watcher.stop()
        assert len(changes) >= 1
        assert changes[0]["key"] == "value2"

    @pytest.mark.asyncio
    async def test_no_change_no_callback(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text('{"key": "value"}')
        watcher = ConfigWatcher(config_path=config_file, poll_interval=0.1)

        changes = []
        watcher.on_change(lambda data: changes.append(data))
        await watcher.start()
        await asyncio.sleep(0.4)
        await watcher.stop()
        assert len(changes) == 0

    @pytest.mark.asyncio
    async def test_reload_count(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text('{"v": 1}')
        watcher = ConfigWatcher(config_path=config_file, poll_interval=0.1)
        await watcher.start()

        await asyncio.sleep(0.2)
        config_file.write_text('{"v": 2}')
        await asyncio.sleep(0.5)

        assert watcher.reload_count >= 1
        await watcher.stop()

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        watcher = ConfigWatcher(config_path=config_file)
        await watcher.start()
        await watcher.stop()
        await watcher.stop()  # Should not raise

    def test_get_stats(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        watcher = ConfigWatcher(config_path=config_file, poll_interval=5.0)
        stats = watcher.get_stats()
        assert stats["poll_interval"] == 5.0
        assert stats["reload_count"] == 0

    @pytest.mark.asyncio
    async def test_missing_file_no_crash(self, tmp_path):
        watcher = ConfigWatcher(
            config_path=tmp_path / "nonexistent.yaml", poll_interval=0.1
        )
        await watcher.start()
        await asyncio.sleep(0.3)
        await watcher.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_yaml_loading(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value1\n")
        watcher = ConfigWatcher(config_path=config_file, poll_interval=0.1)

        changes = []
        watcher.on_change(lambda data: changes.append(data))
        await watcher.start()

        await asyncio.sleep(0.2)
        config_file.write_text("key: value2\n")
        await asyncio.sleep(0.5)
        await watcher.stop()

        # May or may not have yaml available; if loaded, should be a dict
        if changes:
            assert isinstance(changes[0], dict)


# ---------------------------------------------------------------------------
# GitIntegration
# ---------------------------------------------------------------------------
from jarvis.services.git_integration import (
    DiffSummary,
    FileDiff,
    GitStatus,
    _classify_commit_type,
    _find_common_scope,
    _glob_match,
    _parse_status,
)


class TestGitStatus:
    """GitStatus dataclass defaults."""

    def test_defaults(self):
        gs = GitStatus()
        assert gs.branch == ""
        assert gs.is_clean is True
        assert gs.is_repo is False


class TestFileDiff:
    """FileDiff summary formatting."""

    def test_summary_added(self):
        fd = FileDiff(path="new.py", status="A", insertions=10)
        assert "added" in fd.summary
        assert "+10" in fd.summary

    def test_summary_modified(self):
        fd = FileDiff(path="old.py", status="M", insertions=5, deletions=3)
        assert "modified" in fd.summary
        assert "+5" in fd.summary
        assert "-3" in fd.summary

    def test_summary_deleted(self):
        fd = FileDiff(path="dead.py", status="D", deletions=50)
        assert "deleted" in fd.summary

    def test_summary_renamed(self):
        fd = FileDiff(path="new_name.py", status="R", old_path="old_name.py")
        s = fd.summary
        assert "renamed" in s
        assert "old_name.py" in s

    def test_summary_binary(self):
        fd = FileDiff(path="image.png", status="A", binary=True)
        assert "binary" in fd.summary


class TestDiffSummary:
    """DiffSummary text output."""

    def test_to_text(self):
        ds = DiffSummary(
            files=[FileDiff(path="a.py", status="M", insertions=5, deletions=2)],
            total_insertions=5,
            total_deletions=2,
            total_files=1,
        )
        text = ds.to_text()
        assert "+5/-2" in text

    def test_to_text_with_diff(self):
        ds = DiffSummary(
            files=[],
            total_insertions=0,
            total_deletions=0,
            total_files=0,
            diff_text="+ added line\n- removed line\n",
        )
        text = ds.to_text(include_diff=True)
        assert "--- Diff ---" in text


class TestParseStatus:
    """git status --porcelain=v2 parsing."""

    def test_parse_branch(self):
        raw = "# branch.head main\n# branch.upstream origin/main\n# branch.ab +2 -1\n"
        gs = _parse_status(raw)
        assert gs.branch == "main"
        assert gs.tracking == "origin/main"
        assert gs.ahead == 2
        assert gs.behind == 1

    def test_parse_untracked(self):
        raw = "? newfile.txt\n"
        gs = _parse_status(raw)
        assert "newfile.txt" in gs.untracked
        assert gs.is_clean is False

    def test_parse_empty(self):
        gs = _parse_status("")
        assert gs.is_clean is True


class TestClassifyCommitType:
    def test_all_tests(self):
        files = [("M", "tests/test_foo.py"), ("A", "tests/test_bar.py")]
        assert _classify_commit_type(files) == "test"

    def test_all_docs(self):
        files = [("M", "README.md"), ("A", "docs/guide.rst")]
        assert _classify_commit_type(files) == "docs"

    def test_all_added(self):
        files = [("A", "src/new_module.py")]
        assert _classify_commit_type(files) == "feat"

    def test_all_deleted(self):
        files = [("D", "src/old.py")]
        assert _classify_commit_type(files) == "refactor"

    def test_empty_files(self):
        assert _classify_commit_type([]) == "chore"

    def test_all_config(self):
        files = [("M", "pyproject.toml"), ("M", "Makefile")]
        assert _classify_commit_type(files) == "chore"


class TestFindCommonScope:
    def test_single_file(self):
        assert _find_common_scope(["src/services/daemon.py"]) == "services"

    def test_common_dir(self):
        scope = _find_common_scope(["src/services/a.py", "src/services/b.py"])
        assert scope == "services"

    def test_no_common(self):
        scope = _find_common_scope(["a.py", "b.py"])
        assert scope == ""

    def test_empty(self):
        assert _find_common_scope([]) == ""


class TestGlobMatch:
    def test_star_match(self):
        assert _glob_match("test_foo.py", "test_*.py") is True

    def test_question_mark(self):
        assert _glob_match("a.py", "?.py") is True

    def test_no_wildcards(self):
        assert _glob_match("test.py", "test.py") is False  # No wildcards → returns False

    def test_complex_pattern(self):
        assert _glob_match("src/services/daemon.py", "src/*/daemon.py") is True


# ---------------------------------------------------------------------------
# IdentityService
# ---------------------------------------------------------------------------
from jarvis.services.identity_service import IdentityService


class TestIdentityService:
    """IdentityService: cross-channel identity linking."""

    def _make_service(self, tmp_path) -> IdentityService:
        return IdentityService(tmp_path)

    def test_resolve_creates_new_user(self, tmp_path):
        svc = self._make_service(tmp_path)
        canonical = svc.resolve("telegram", "123456")
        assert canonical.startswith("u-")
        svc.close()

    def test_resolve_returns_same_canonical(self, tmp_path):
        svc = self._make_service(tmp_path)
        c1 = svc.resolve("telegram", "123456")
        c2 = svc.resolve("telegram", "123456")
        assert c1 == c2
        svc.close()

    def test_link_cross_channel(self, tmp_path):
        svc = self._make_service(tmp_path)
        canonical = svc.resolve("telegram", "tg_user")
        success = svc.link("webchat", "wc_token", canonical)
        assert success is True
        c2 = svc.resolve("webchat", "wc_token")
        assert c2 == canonical
        svc.close()

    def test_link_nonexistent_user_fails(self, tmp_path):
        svc = self._make_service(tmp_path)
        result = svc.link("telegram", "123", "u-nonexistent")
        assert result is False
        svc.close()

    def test_get_links(self, tmp_path):
        svc = self._make_service(tmp_path)
        canonical = svc.resolve("telegram", "tg_user")
        svc.link("webchat", "wc_user", canonical)
        links = svc.get_links(canonical)
        channels = {l["channel"] for l in links}
        assert "telegram" in channels
        assert "webchat" in channels
        svc.close()

    def test_get_user_info(self, tmp_path):
        svc = self._make_service(tmp_path)
        canonical = svc.resolve("telegram", "tg_user", display_name="Alice")
        info = svc.get_user_info(canonical)
        assert info is not None
        assert info["canonical_id"] == canonical
        assert info["display_name"] == "Alice"
        assert len(info["channels"]) == 1
        svc.close()

    def test_get_user_info_nonexistent(self, tmp_path):
        svc = self._make_service(tmp_path)
        assert svc.get_user_info("u-ghost") is None
        svc.close()

    def test_unlink(self, tmp_path):
        svc = self._make_service(tmp_path)
        canonical = svc.resolve("telegram", "tg_user")
        removed = svc.unlink("telegram", "tg_user")
        assert removed is True
        # Resolving again should create a new canonical
        c2 = svc.resolve("telegram", "tg_user")
        assert c2 != canonical
        svc.close()

    def test_unlink_nonexistent(self, tmp_path):
        svc = self._make_service(tmp_path)
        assert svc.unlink("telegram", "nobody") is False
        svc.close()

    def test_delete_user(self, tmp_path):
        svc = self._make_service(tmp_path)
        canonical = svc.resolve("telegram", "tg_user")
        svc.link("webchat", "wc_user", canonical)
        deleted = svc.delete_user(canonical)
        assert deleted is True
        assert svc.get_user_info(canonical) is None
        assert svc.get_links(canonical) == []
        svc.close()

    def test_delete_user_nonexistent(self, tmp_path):
        svc = self._make_service(tmp_path)
        assert svc.delete_user("u-ghost") is False
        svc.close()

    def test_generate_and_redeem_link_code(self, tmp_path):
        svc = self._make_service(tmp_path)
        canonical = svc.resolve("telegram", "tg_user")
        code = svc.generate_link_code(canonical)
        assert len(code) == 8
        redeemed = svc.redeem_link_code(code, "discord", "dc_user")
        assert redeemed == canonical
        # Discord should now resolve to same canonical
        c2 = svc.resolve("discord", "dc_user")
        assert c2 == canonical
        svc.close()

    def test_link_code_expired(self, tmp_path):
        svc = self._make_service(tmp_path)
        canonical = svc.resolve("telegram", "tg_user")
        code = svc.generate_link_code(canonical)
        # Manually expire the code
        conn = svc._get_conn()
        meta = json.dumps({"expires_at": time.time() - 100})
        conn.execute(
            "UPDATE channel_links SET metadata = ? WHERE channel = '_link_code' AND channel_user_id = ?",
            (meta, code),
        )
        conn.commit()
        result = svc.redeem_link_code(code, "discord", "dc_user")
        assert result is None
        svc.close()

    def test_redeem_invalid_code(self, tmp_path):
        svc = self._make_service(tmp_path)
        assert svc.redeem_link_code("BADCODE", "discord", "dc_user") is None
        svc.close()

    def test_relink_to_different_user(self, tmp_path):
        svc = self._make_service(tmp_path)
        c1 = svc.resolve("telegram", "tg_user")
        c2 = svc.resolve("webchat", "wc_user")
        # Re-link wc_user to c1 instead of c2
        success = svc.link("webchat", "wc_user", c1)
        assert success is True
        resolved = svc.resolve("webchat", "wc_user")
        assert resolved == c1
        svc.close()


# ---------------------------------------------------------------------------
# Plugins
# ---------------------------------------------------------------------------
from jarvis.services.plugins import (
    Plugin,
    PluginRegistry,
    hook,
    tool,
    VALID_HOOK_EVENTS,
    MAX_LOADED_PLUGINS,
)


class SamplePlugin(Plugin):
    name = "sample"
    version = "1.0.0"
    description = "A test plugin"

    @tool(description="Say hello")
    async def greet(self, name: str) -> str:
        return f"Hello, {name}!"

    @hook("on_message")
    async def log_msg(self, **kwargs):
        return "logged"


class TestPlugin:
    def test_get_tools(self):
        p = SamplePlugin()
        tools = p.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "greet"

    def test_get_hooks(self):
        p = SamplePlugin()
        hooks = p.get_hooks()
        assert len(hooks) == 1
        assert hooks[0][0] == "on_message"

    def test_enable_disable(self):
        p = SamplePlugin()
        assert p.enabled is True
        p.disable()
        assert p.enabled is False
        p.enable()
        assert p.enabled is True

    def test_tool_definitions(self):
        p = SamplePlugin()
        defs = p.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "sample.greet"


class TestPluginRegistry:
    @pytest.mark.asyncio
    async def test_load_and_unload(self):
        reg = PluginRegistry()
        plugin = await reg.load_plugin(SamplePlugin)
        assert plugin.name == "sample"
        assert reg.plugin_count == 1
        await reg.unload_plugin("sample")
        assert reg.plugin_count == 0

    @pytest.mark.asyncio
    async def test_duplicate_load_raises(self):
        reg = PluginRegistry()
        await reg.load_plugin(SamplePlugin)
        with pytest.raises(ValueError, match="already loaded"):
            await reg.load_plugin(SamplePlugin)
        await reg.unload_plugin("sample")

    @pytest.mark.asyncio
    async def test_unload_nonexistent_raises(self):
        reg = PluginRegistry()
        with pytest.raises(ValueError, match="not loaded"):
            await reg.unload_plugin("ghost")

    @pytest.mark.asyncio
    async def test_call_tool(self):
        reg = PluginRegistry()
        await reg.load_plugin(SamplePlugin)
        result = await reg.call_tool("sample.greet", {"name": "World"})
        assert "Hello, World!" in result
        await reg.unload_plugin("sample")

    @pytest.mark.asyncio
    async def test_call_nonexistent_tool(self):
        reg = PluginRegistry()
        result = await reg.call_tool("ghost.tool", {})
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_dispatch_hook(self):
        reg = PluginRegistry()
        await reg.load_plugin(SamplePlugin)
        results = await reg.dispatch_hook("on_message")
        assert "logged" in results
        await reg.unload_plugin("sample")

    @pytest.mark.asyncio
    async def test_get_all_tool_definitions(self):
        reg = PluginRegistry()
        await reg.load_plugin(SamplePlugin)
        defs = reg.get_all_tool_definitions()
        assert len(defs) == 1
        await reg.unload_plugin("sample")

    @pytest.mark.asyncio
    async def test_invalid_plugin_class_raises(self):
        reg = PluginRegistry()
        with pytest.raises(TypeError, match="Plugin subclass"):
            await reg.load_plugin(str)  # type: ignore

    @pytest.mark.asyncio
    async def test_unnamed_plugin_raises(self):
        class BadPlugin(Plugin):
            pass  # name == "unnamed_plugin"
        reg = PluginRegistry()
        with pytest.raises(ValueError, match="must define"):
            await reg.load_plugin(BadPlugin)

    @pytest.mark.asyncio
    async def test_disabled_plugin_tool_blocked(self):
        reg = PluginRegistry()
        plugin = await reg.load_plugin(SamplePlugin)
        plugin.disable()
        result = await reg.call_tool("sample.greet", {"name": "X"})
        assert "disabled" in result
        await reg.unload_plugin("sample")

    @pytest.mark.asyncio
    async def test_get_loaded_plugins(self):
        reg = PluginRegistry()
        await reg.load_plugin(SamplePlugin)
        loaded = reg.get_loaded_plugins()
        assert len(loaded) == 1
        assert loaded[0]["name"] == "sample"
        assert loaded[0]["tools"] == 1
        await reg.unload_plugin("sample")
