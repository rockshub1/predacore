"""Tests for predacore.tools.resilience — Circuit Breaker, Cache, ExecutionHistory."""
import time
import pytest

from predacore.tools.resilience import (
    CircuitState,
    ExecutionHistory,
    ToolCircuitBreaker,
    ToolResultCache,
)


# ── Circuit Breaker ──────────────────────────────────────────────────


class TestToolCircuitBreaker:
    def test_initial_state_is_closed(self):
        cb = ToolCircuitBreaker(failure_threshold=3, cooldown_seconds=10)
        assert cb.state("web_search") == CircuitState.CLOSED
        assert not cb.is_open("web_search")

    def test_opens_after_threshold_failures(self):
        cb = ToolCircuitBreaker(failure_threshold=3, cooldown_seconds=10)
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        assert cb.state("web_search") == CircuitState.CLOSED
        cb.record_failure("web_search")
        assert cb.state("web_search") == CircuitState.OPEN
        assert cb.is_open("web_search")

    def test_success_resets_failure_count(self):
        cb = ToolCircuitBreaker(failure_threshold=3, cooldown_seconds=10)
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        cb.record_success("web_search")
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        # 2 failures, not 3 (reset happened)
        assert cb.state("web_search") == CircuitState.CLOSED

    def test_half_open_after_cooldown(self):
        cb = ToolCircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)
        cb.record_failure("t")
        cb.record_failure("t")
        assert cb.state("t") == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state("t") == CircuitState.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        cb = ToolCircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)
        cb.record_failure("t")
        cb.record_failure("t")
        time.sleep(0.15)
        assert cb.state("t") == CircuitState.HALF_OPEN
        cb.record_success("t")
        assert cb.state("t") == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self):
        cb = ToolCircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)
        cb.record_failure("t")
        cb.record_failure("t")
        time.sleep(0.15)
        assert cb.state("t") == CircuitState.HALF_OPEN
        cb.record_failure("t")
        assert cb.state("t") == CircuitState.OPEN

    def test_different_tools_independent(self):
        cb = ToolCircuitBreaker(failure_threshold=2, cooldown_seconds=10)
        cb.record_failure("a")
        cb.record_failure("a")
        assert cb.is_open("a")
        assert not cb.is_open("b")

    def test_status_returns_all_tools(self):
        cb = ToolCircuitBreaker(failure_threshold=3, cooldown_seconds=10)
        cb.record_failure("a")
        cb.record_success("b")
        status = cb.status()
        assert "a" in status
        assert "b" in status
        assert status["a"]["failures"] == 1
        assert status["b"]["state"] == "closed"


# ── Tool Result Cache ────────────────────────────────────────────────


class TestToolResultCache:
    def test_cacheable_tools(self):
        cache = ToolResultCache(max_entries=50)
        assert cache.is_cacheable("read_file")
        assert cache.is_cacheable("web_search")
        assert not cache.is_cacheable("write_file")
        assert not cache.is_cacheable("run_command")

    def test_put_and_get(self):
        cache = ToolResultCache(max_entries=50)
        cache.put("read_file", {"path": "/tmp/test.txt"}, "hello world")
        result = cache.get("read_file", {"path": "/tmp/test.txt"})
        assert result == "hello world"

    def test_miss_on_different_args(self):
        cache = ToolResultCache(max_entries=50)
        cache.put("read_file", {"path": "/tmp/a.txt"}, "aaa")
        result = cache.get("read_file", {"path": "/tmp/b.txt"})
        assert result is None

    def test_ttl_expiration(self):
        cache = ToolResultCache(
            max_entries=50,
            ttl_map={"read_file": 0},  # 0-second TTL = instant expire
        )
        cache.put("read_file", {"path": "/tmp/test.txt"}, "hello")
        time.sleep(0.01)
        result = cache.get("read_file", {"path": "/tmp/test.txt"})
        assert result is None

    def test_lru_eviction(self):
        cache = ToolResultCache(max_entries=2)
        cache.put("read_file", {"path": "a"}, "aaa")
        cache.put("read_file", {"path": "b"}, "bbb")
        cache.put("read_file", {"path": "c"}, "ccc")  # Evicts "a"
        assert cache.get("read_file", {"path": "a"}) is None
        assert cache.get("read_file", {"path": "b"}) == "bbb"
        assert cache.get("read_file", {"path": "c"}) == "ccc"

    def test_does_not_cache_errors(self):
        cache = ToolResultCache(max_entries=50)
        cache.put("read_file", {"path": "x"}, "[Tool error: file not found]")
        result = cache.get("read_file", {"path": "x"})
        assert result is None

    def test_invalidate_specific_tool(self):
        cache = ToolResultCache(max_entries=50)
        cache.put("read_file", {"path": "a"}, "aaa")
        cache.put("web_search", {"query": "x"}, "results")
        count = cache.invalidate("read_file")
        assert count == 1
        assert cache.get("read_file", {"path": "a"}) is None
        assert cache.get("web_search", {"query": "x"}) == "results"

    def test_invalidate_all(self):
        cache = ToolResultCache(max_entries=50)
        cache.put("read_file", {"path": "a"}, "aaa")
        cache.put("web_search", {"query": "x"}, "results")
        count = cache.invalidate()
        assert count == 2

    def test_invalidate_on_write(self):
        cache = ToolResultCache(max_entries=50)
        cache.put("read_file", {"path": "a"}, "aaa")
        cache.put("git_context", {}, "status")
        cache.invalidate_on_write("write_file", {"path": "a"})
        assert cache.get("read_file", {"path": "a"}) is None

    def test_stats(self):
        cache = ToolResultCache(max_entries=50)
        cache.put("read_file", {"path": "a"}, "aaa")
        cache.get("read_file", {"path": "a"})  # hit
        cache.get("read_file", {"path": "b"})  # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1


# ── Execution History ────────────────────────────────────────────────


class TestExecutionHistory:
    def test_record_and_recent(self):
        h = ExecutionHistory(max_entries=100)
        h.record("read_file", {"path": "a"}, "content", "ok", 50.0)
        h.record("web_search", {"query": "x"}, "results", "ok", 200.0)
        recent = h.recent(10)
        assert len(recent) == 2
        assert recent[0]["tool"] == "read_file"
        assert recent[1]["tool"] == "web_search"

    def test_for_tool_filtering(self):
        h = ExecutionHistory(max_entries=100)
        h.record("read_file", {}, "a", "ok", 10)
        h.record("web_search", {}, "b", "ok", 20)
        h.record("read_file", {}, "c", "ok", 30)
        tool_history = h.for_tool("read_file")
        assert len(tool_history) == 2
        assert all(r["tool"] == "read_file" for r in tool_history)

    def test_ring_buffer_eviction(self):
        h = ExecutionHistory(max_entries=3)
        for i in range(5):
            h.record(f"tool_{i}", {}, f"result_{i}", "ok", 10)
        recent = h.recent(10)
        assert len(recent) == 3
        assert recent[0]["tool"] == "tool_2"

    def test_stats_calculation(self):
        h = ExecutionHistory(max_entries=100)
        h.record("a", {}, "ok", "ok", 100)
        h.record("b", {}, "fail", "error", 200)
        h.record("a", {}, "ok", "ok", 50)
        stats = h.stats()
        assert stats["total_calls"] == 3
        assert stats["total_errors"] == 1
        assert stats["top_tools"]["a"] == 2
        assert stats["avg_latency_ms"] > 0

    def test_clear(self):
        h = ExecutionHistory(max_entries=100)
        h.record("a", {}, "ok", "ok", 10)
        h.clear()
        assert h.stats()["total_calls"] == 0
        assert h.recent(10) == []
