"""
Tests for PredaCore Tool Resilience — Circuit Breaker, Result Cache, Execution History.

Covers:
  - Circuit breaker state machine (CLOSED → OPEN → HALF_OPEN → CLOSED)
  - Result cache (LRU eviction, TTL expiry, cache invalidation on writes)
  - Execution history (recording, stats, per-tool filtering)
"""
import time
import pytest
from predacore.tools.resilience import (
    CircuitState,
    ExecutionHistory,
    ToolCircuitBreaker,
    ToolResultCache,
)


# ═══════════════════════════════════════════════════════════════════
# Circuit Breaker Tests
# ═══════════════════════════════════════════════════════════════════


class TestToolCircuitBreaker:
    """Test the per-tool circuit breaker."""

    def test_initial_state_is_closed(self):
        cb = ToolCircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        assert cb.state("web_search") == CircuitState.CLOSED
        assert not cb.is_open("web_search")

    def test_opens_after_threshold_failures(self):
        cb = ToolCircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        assert cb.state("web_search") == CircuitState.CLOSED  # 2 < 3
        cb.record_failure("web_search")
        assert cb.state("web_search") == CircuitState.OPEN  # 3 >= 3
        assert cb.is_open("web_search")

    def test_success_resets_failure_count(self):
        cb = ToolCircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        cb.record_success("web_search")  # Reset
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        assert cb.state("web_search") == CircuitState.CLOSED  # Only 2 after reset

    def test_half_open_after_cooldown(self):
        cb = ToolCircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)
        cb.record_failure("test_tool")
        cb.record_failure("test_tool")
        assert cb.state("test_tool") == CircuitState.OPEN

        time.sleep(0.15)  # Wait for cooldown
        assert cb.state("test_tool") == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self):
        cb = ToolCircuitBreaker(failure_threshold=2, cooldown_seconds=0.05)
        cb.record_failure("t")
        cb.record_failure("t")
        assert cb.is_open("t")

        time.sleep(0.06)
        assert cb.state("t") == CircuitState.HALF_OPEN
        cb.record_success("t")
        assert cb.state("t") == CircuitState.CLOSED

    def test_half_open_failure_reopens_circuit(self):
        cb = ToolCircuitBreaker(failure_threshold=2, cooldown_seconds=0.05)
        cb.record_failure("t")
        cb.record_failure("t")
        time.sleep(0.06)
        assert cb.state("t") == CircuitState.HALF_OPEN

        cb.record_failure("t")  # Probe failed
        assert cb.state("t") == CircuitState.OPEN

    def test_independent_per_tool(self):
        cb = ToolCircuitBreaker(failure_threshold=2, cooldown_seconds=60)
        cb.record_failure("tool_a")
        cb.record_failure("tool_a")
        assert cb.is_open("tool_a")
        assert not cb.is_open("tool_b")  # Unaffected

    def test_status_report(self):
        cb = ToolCircuitBreaker(failure_threshold=2, cooldown_seconds=60)
        cb.record_failure("x")
        cb.record_failure("x")
        status = cb.status()
        assert "x" in status
        assert status["x"]["state"] == "open"
        assert status["x"]["failures"] == 2


# ═══════════════════════════════════════════════════════════════════
# Result Cache Tests
# ═══════════════════════════════════════════════════════════════════


class TestToolResultCache:
    """Test the LRU + TTL result cache."""

    def test_uncacheable_tool_returns_none(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("write_file", {"path": "/tmp/x"}, "wrote ok")
        assert cache.get("write_file", {"path": "/tmp/x"}) is None

    def test_cache_hit(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("read_file", {"path": "/tmp/x"}, "file content here")
        result = cache.get("read_file", {"path": "/tmp/x"})
        assert result == "file content here"

    def test_cache_miss_different_args(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("read_file", {"path": "/tmp/a"}, "content a")
        assert cache.get("read_file", {"path": "/tmp/b"}) is None

    def test_cache_ttl_expiry(self):
        cache = ToolResultCache(
            max_entries=10,
            ttl_map={"read_file": 0.1},  # 100ms TTL
        )
        cache.put("read_file", {"path": "/tmp/x"}, "content")
        assert cache.get("read_file", {"path": "/tmp/x"}) == "content"
        time.sleep(0.15)
        assert cache.get("read_file", {"path": "/tmp/x"}) is None  # Expired

    def test_lru_eviction(self):
        cache = ToolResultCache(max_entries=2)
        cache.put("read_file", {"path": "a"}, "A")
        cache.put("read_file", {"path": "b"}, "B")
        cache.put("read_file", {"path": "c"}, "C")  # Evicts "a"
        assert cache.get("read_file", {"path": "a"}) is None
        assert cache.get("read_file", {"path": "b"}) == "B"
        assert cache.get("read_file", {"path": "c"}) == "C"

    def test_invalidate_specific_tool(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("read_file", {"path": "a"}, "A")
        cache.put("web_search", {"query": "test"}, "results")
        removed = cache.invalidate("read_file")
        assert removed >= 1
        assert cache.get("read_file", {"path": "a"}) is None
        assert cache.get("web_search", {"query": "test"}) == "results"

    def test_invalidate_all(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("read_file", {"path": "a"}, "A")
        cache.put("web_search", {"query": "q"}, "R")
        count = cache.invalidate()
        assert count == 2

    def test_invalidate_on_write_file(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("read_file", {"path": "a"}, "A")
        cache.put("list_directory", {"path": "/"}, "dir listing")
        cache.invalidate_on_write("write_file", {"path": "a"})
        assert cache.get("read_file", {"path": "a"}) is None
        assert cache.get("list_directory", {"path": "/"}) is None

    def test_invalidate_on_run_command(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("git_context", {}, "context")
        cache.invalidate_on_write("run_command", {"command": "git add ."})
        assert cache.get("git_context", {}) is None

    def test_does_not_cache_error_results(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("read_file", {"path": "x"}, "[Tool error: file not found]")
        assert cache.get("read_file", {"path": "x"}) is None

    def test_stats(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("read_file", {"path": "a"}, "content")
        cache.get("read_file", {"path": "a"})  # hit
        cache.get("read_file", {"path": "b"})  # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1

    def test_is_cacheable(self):
        cache = ToolResultCache(max_entries=10)
        assert cache.is_cacheable("read_file")
        assert cache.is_cacheable("web_search")
        assert not cache.is_cacheable("write_file")
        assert not cache.is_cacheable("run_command")


# ═══════════════════════════════════════════════════════════════════
# Execution History Tests
# ═══════════════════════════════════════════════════════════════════


class TestExecutionHistory:
    """Test the ring buffer execution history."""

    def test_record_and_recent(self):
        history = ExecutionHistory(max_entries=100)
        history.record("read_file", {"path": "/tmp"}, "content", "ok", 15.0)
        recent = history.recent(5)
        assert len(recent) == 1
        assert recent[0]["tool"] == "read_file"
        assert recent[0]["status"] == "ok"

    def test_ring_buffer_eviction(self):
        history = ExecutionHistory(max_entries=3)
        for i in range(5):
            history.record(f"tool_{i}", {}, f"result_{i}", "ok", 10.0)
        recent = history.recent(10)
        assert len(recent) == 3
        assert recent[0]["tool"] == "tool_2"  # Oldest surviving

    def test_for_tool_filtering(self):
        history = ExecutionHistory(max_entries=100)
        history.record("read_file", {"path": "a"}, "A", "ok", 10.0)
        history.record("web_search", {"q": "x"}, "results", "ok", 200.0)
        history.record("read_file", {"path": "b"}, "B", "ok", 15.0)

        rf = history.for_tool("read_file")
        assert len(rf) == 2
        assert all(r["tool"] == "read_file" for r in rf)

    def test_stats_counts(self):
        history = ExecutionHistory(max_entries=100)
        history.record("tool_a", {}, "ok", "ok", 10.0)
        history.record("tool_a", {}, "ok", "ok", 20.0)
        history.record("tool_b", {}, "fail", "error", 100.0)
        history.record("tool_c", {}, "timeout", "timeout", 5000.0)

        stats = history.stats()
        assert stats["total_calls"] == 4
        assert stats["total_errors"] == 2  # error + timeout
        assert stats["top_tools"]["tool_a"] == 2

    def test_stats_avg_latency(self):
        history = ExecutionHistory(max_entries=100)
        history.record("t", {}, "r", "ok", 100.0)
        history.record("t", {}, "r", "ok", 200.0)
        stats = history.stats()
        assert stats["avg_latency_ms"] == 150.0

    def test_stats_slowest_call(self):
        history = ExecutionHistory(max_entries=100)
        history.record("fast", {}, "r", "ok", 10.0)
        history.record("slow", {}, "r", "ok", 5000.0)
        stats = history.stats()
        assert stats["slowest_call"]["tool"] == "slow"

    def test_clear(self):
        history = ExecutionHistory(max_entries=100)
        history.record("t", {}, "r", "ok", 10.0)
        history.clear()
        assert history.stats()["total_calls"] == 0
        assert len(history.recent(10)) == 0

    def test_origin_tracking(self):
        history = ExecutionHistory(max_entries=100)
        history.record("t", {}, "r", "ok", 10.0, origin="llm")
        recent = history.recent(1)
        assert recent[0]["origin"] == "llm"

    def test_result_preview_truncation(self):
        history = ExecutionHistory(max_entries=100)
        long_result = "x" * 1000
        history.record("t", {}, long_result, "ok", 10.0)
        recent = history.recent(1)
        assert len(recent[0]["result_preview"]) <= 300  # to_dict truncates to 300
