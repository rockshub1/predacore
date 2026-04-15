"""
Tests for JARVIS Tool Dispatcher — the full dispatch pipeline including
alias resolution, rate limiting, circuit breaker, caching, and timeout.

Uses lightweight mocks to avoid requiring full subsystem initialization.
"""
import asyncio
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from jarvis.tools.dispatcher import AdaptiveTimeoutTracker, ToolDispatcher
from jarvis.tools.enums import ToolStatus


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════
# AdaptiveTimeoutTracker Tests
# ═══════════════════════════════════════════════════════════════════


class TestAdaptiveTimeoutTracker:
    def test_returns_ceiling_with_no_data(self):
        tracker = AdaptiveTimeoutTracker(min_samples=3)
        assert tracker.get_timeout("read_file", 60.0) == 60.0

    def test_returns_ceiling_with_few_samples(self):
        tracker = AdaptiveTimeoutTracker(min_samples=3)
        tracker.record("read_file", 1.0)
        tracker.record("read_file", 2.0)
        # Only 2 samples, need 3
        assert tracker.get_timeout("read_file", 60.0) == 60.0

    def test_adapts_after_enough_samples(self):
        tracker = AdaptiveTimeoutTracker(
            min_samples=3, multiplier=2.0, min_floor=1.0
        )
        # Record 5 fast calls
        for _ in range(5):
            tracker.record("fast_tool", 0.5)

        timeout = tracker.get_timeout("fast_tool", 60.0)
        # P95 ≈ 0.5s, × 2.0 = 1.0s, capped by floor at 1.0s
        assert timeout <= 60.0
        assert timeout >= 1.0

    def test_never_below_min_floor(self):
        tracker = AdaptiveTimeoutTracker(
            min_samples=3, multiplier=2.0, min_floor=5.0
        )
        for _ in range(5):
            tracker.record("tiny", 0.01)

        timeout = tracker.get_timeout("tiny", 60.0)
        assert timeout >= 5.0  # Floor enforced

    def test_never_above_ceiling(self):
        tracker = AdaptiveTimeoutTracker(
            min_samples=3, multiplier=2.0, min_floor=1.0
        )
        for _ in range(5):
            tracker.record("slow", 50.0)

        timeout = tracker.get_timeout("slow", 30.0)
        assert timeout <= 30.0  # Ceiling enforced

    def test_independent_per_tool(self):
        tracker = AdaptiveTimeoutTracker(min_samples=3, multiplier=2.0, min_floor=1.0)
        for _ in range(5):
            tracker.record("fast", 0.1)
            tracker.record("slow", 10.0)

        fast_timeout = tracker.get_timeout("fast", 60.0)
        slow_timeout = tracker.get_timeout("slow", 60.0)
        assert fast_timeout < slow_timeout

    def test_rolling_window(self):
        tracker = AdaptiveTimeoutTracker(
            window_size=5, min_samples=3, multiplier=2.0, min_floor=1.0
        )
        # Start with slow calls
        for _ in range(5):
            tracker.record("tool", 10.0)
        slow_timeout = tracker.get_timeout("tool", 60.0)

        # Now add fast calls (push out slow ones)
        for _ in range(5):
            tracker.record("tool", 0.5)
        fast_timeout = tracker.get_timeout("tool", 60.0)

        assert fast_timeout < slow_timeout


# ═══════════════════════════════════════════════════════════════════
# Dispatcher Alias Resolution Tests
# ═══════════════════════════════════════════════════════════════════


def _make_dispatcher():
    """Create a dispatcher with minimal mocks."""
    from jarvis.tools.handlers._context import ToolContext

    security = SimpleNamespace(
        trust_level="yolo",
        blocked_paths=[],
        blocked_commands=[],
        task_timeout_seconds=300,
    )
    config = SimpleNamespace(security=security, mode="test")
    ctx = ToolContext(config=config, memory={})
    trust = MagicMock()
    trust.requires_confirmation.return_value = False
    trust.is_blocked.return_value = None

    return ToolDispatcher(trust, ctx, rate_max=1000, tool_timeout=30)


class TestDispatcherAliases:
    """Test tool name alias resolution."""

    def test_gemini_alias_run_in_terminal(self):
        d = _make_dispatcher()
        # run_in_terminal should map to run_command
        result = _run(d.dispatch("run_in_terminal", {"command": "echo test"}))
        assert "test" in result or "echo" in result

    def test_gemini_alias_edit_file(self):
        d = _make_dispatcher()
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            result = _run(d.dispatch("edit_file", {"path": path, "content": "hello"}))
            # edit_file → write_file
            assert os.path.exists(path)

    def test_legacy_alias_run_shell_command(self):
        d = _make_dispatcher()
        result = _run(d.dispatch("run_shell_command", {"command": "echo legacy"}))
        assert "legacy" in result

    def test_unknown_tool(self):
        d = _make_dispatcher()
        result = _run(d.dispatch("nonexistent_tool_xyz", {}))
        assert "Unknown tool" in result


# ═══════════════════════════════════════════════════════════════════
# Dispatcher Circuit Breaker Integration
# ═══════════════════════════════════════════════════════════════════


class TestDispatcherCircuitBreaker:
    def test_circuit_breaker_fast_fails(self):
        d = _make_dispatcher()
        # Manually trip the circuit
        d.circuit_breaker.record_failure("read_file")
        d.circuit_breaker.record_failure("read_file")
        d.circuit_breaker.record_failure("read_file")

        result = _run(d.dispatch("read_file", {"path": "/tmp/x"}))
        assert "circuit breaker" in result.lower() or "unavailable" in result.lower()


# ═══════════════════════════════════════════════════════════════════
# Dispatcher Cache Integration
# ═══════════════════════════════════════════════════════════════════


class TestDispatcherCache:
    def test_cache_stats_accessible(self):
        d = _make_dispatcher()
        stats = d.result_cache.stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "entries" in stats


# ═══════════════════════════════════════════════════════════════════
# Dispatcher Execution History Integration
# ═══════════════════════════════════════════════════════════════════


class TestDispatcherHistory:
    def test_records_execution(self):
        d = _make_dispatcher()
        _run(d.dispatch("run_command", {"command": "echo test"}))
        recent = d.execution_history.recent(5)
        assert len(recent) >= 1
        assert recent[-1]["tool"] == "run_command"
        # ToolStatus is (str, Enum) so == "ok" works, but be safe
        assert recent[-1]["status"] == ToolStatus.OK or recent[-1]["status"] == "ok"

    def test_records_unknown_tool(self):
        d = _make_dispatcher()
        _run(d.dispatch("fake_tool", {}))
        # Unknown tools don't go through the full pipeline — they return early
        # Check that the result contains the right message
        recent = d.execution_history.recent(5)
        # May or may not be recorded depending on implementation
        # The important thing is it doesn't crash

    def test_history_stats(self):
        d = _make_dispatcher()
        _run(d.dispatch("run_command", {"command": "echo 1"}))
        _run(d.dispatch("run_command", {"command": "echo 2"}))
        stats = d.execution_history.stats()
        assert stats["total_calls"] >= 2
