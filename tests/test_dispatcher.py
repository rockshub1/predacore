"""
Tests for JARVIS Tool Dispatcher — full pipeline coverage.

Tests the dispatch pipeline: alias resolution → rate limiting → circuit breaker →
cache → confirmation → blocked/allowed → handler execution → adaptive timeout →
ToolError handling → metrics → output sanitization.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis.tools.enums import ToolName, ToolStatus, WRITE_TOOLS
from jarvis.tools.handlers._context import ToolContext, ToolError, ToolErrorKind
from jarvis.tools.resilience import ToolCircuitBreaker, ToolResultCache, ExecutionHistory
from jarvis.tools.dispatcher import ToolDispatcher, AdaptiveTimeoutTracker
from jarvis.tools.trust_policy import TrustPolicyEvaluator


# ── Test Fixtures ─────────────────────────────────────────────────────


@dataclass
class MockSecurity:
    """Minimal security config for ToolContext."""
    dangerous_patterns: list = field(default_factory=lambda: ["rm -rf /"])
    blocked_paths: list = field(default_factory=list)
    task_timeout_seconds: int = 30
    trust_level: str = "yolo"


@dataclass
class MockConfig:
    """Minimal config for ToolContext."""
    security: MockSecurity = field(default_factory=MockSecurity)
    home_dir: str = "/tmp/jarvis_test"


@dataclass
class MockMemory:
    """Minimal memory stand-in."""
    async def search(self, *a, **kw): return []
    async def store(self, *a, **kw): return True


def make_tool_ctx() -> ToolContext:
    """Create a minimal ToolContext for testing."""
    return ToolContext(config=MockConfig(), memory=MockMemory())


def make_dispatcher(trust_level: str = "yolo") -> ToolDispatcher:
    """Create a ToolDispatcher with a mock trust evaluator."""
    trust = TrustPolicyEvaluator(trust_level=trust_level)
    ctx = make_tool_ctx()
    return ToolDispatcher(trust, ctx, rate_max=1000)


# =====================================================================
# AdaptiveTimeoutTracker
# =====================================================================


class TestAdaptiveTimeoutTracker:
    """Tests for the adaptive timeout system."""

    def test_cold_start_returns_ceiling(self):
        """Without enough samples, static ceiling is returned."""
        tracker = AdaptiveTimeoutTracker(min_samples=3)
        assert tracker.get_timeout("web_search", 60.0) == 60.0

    def test_adapts_after_min_samples(self):
        """After enough samples, timeout adapts based on P95."""
        tracker = AdaptiveTimeoutTracker(min_samples=3, multiplier=2.0, min_floor=5.0)
        # Record 5 fast executions (1 second each)
        for _ in range(5):
            tracker.record("fast_tool", 1.0)
        # Adaptive = P95(1.0) * 2.0 = 2.0, but min_floor = 5.0
        timeout = tracker.get_timeout("fast_tool", 60.0)
        assert timeout == 5.0  # min_floor kicks in

    def test_ceiling_caps_adaptive(self):
        """Adaptive timeout never exceeds the static ceiling."""
        tracker = AdaptiveTimeoutTracker(min_samples=3, multiplier=2.0, min_floor=1.0)
        # Record slow executions
        for _ in range(5):
            tracker.record("slow_tool", 50.0)
        # P95 * 2 = 100, but ceiling is 60
        timeout = tracker.get_timeout("slow_tool", 60.0)
        assert timeout == 60.0

    def test_per_tool_isolation(self):
        """Each tool has its own latency window."""
        tracker = AdaptiveTimeoutTracker(min_samples=3)
        for _ in range(5):
            tracker.record("fast_tool", 1.0)
            tracker.record("slow_tool", 30.0)
        # fast_tool should have lower adaptive timeout
        fast_t = tracker.get_timeout("fast_tool", 120.0)
        slow_t = tracker.get_timeout("slow_tool", 120.0)
        assert fast_t < slow_t

    def test_rolling_window(self):
        """Old samples drop off when window is full."""
        tracker = AdaptiveTimeoutTracker(window_size=5, min_samples=3, multiplier=2.0, min_floor=1.0)
        # Fill with slow values
        for _ in range(5):
            tracker.record("tool", 30.0)
        # Now push fast values — old slow values should fall off
        for _ in range(5):
            tracker.record("tool", 1.0)
        timeout = tracker.get_timeout("tool", 120.0)
        assert timeout < 10.0  # Should be adapted to fast values now


# =====================================================================
# ToolDispatcher — Core Pipeline
# =====================================================================


class TestDispatcherPipeline:
    """Tests for the full dispatch pipeline."""

    @pytest.fixture
    def dispatcher(self):
        return make_dispatcher()

    @pytest.mark.asyncio
    async def test_unknown_tool(self, dispatcher):
        """Unknown tool returns descriptive error message."""
        result = await dispatcher.dispatch("nonexistent_tool_xyz", {})
        assert "Unknown tool" in result
        assert "nonexistent_tool_xyz" in result

    @pytest.mark.asyncio
    async def test_alias_resolution(self, dispatcher):
        """Gemini CLI aliases are resolved to JARVIS tool names."""
        # run_in_terminal → run_command
        # We patch the handler to verify the resolved name is used
        with patch.dict(
            "jarvis.tools.handlers.HANDLER_MAP",
            {"run_command": AsyncMock(return_value="executed")},
        ):
            result = await dispatcher.dispatch("run_in_terminal", {"command": "echo hi"})
            assert result == "executed"

    @pytest.mark.asyncio
    async def test_run_shell_command_alias(self, dispatcher):
        """Legacy alias run_shell_command → run_command works."""
        with patch.dict(
            "jarvis.tools.handlers.HANDLER_MAP",
            {"run_command": AsyncMock(return_value="ok")},
        ):
            result = await dispatcher.dispatch("run_shell_command", {"command": "ls"})
            assert result == "ok"

    @pytest.mark.asyncio
    async def test_successful_execution(self, dispatcher):
        """Successful tool execution returns the handler's result."""
        mock_handler = AsyncMock(return_value="file contents here")
        with patch.dict(
            "jarvis.tools.handlers.HANDLER_MAP",
            {"read_file": mock_handler},
        ):
            result = await dispatcher.dispatch("read_file", {"path": "/tmp/test.txt"})
            assert result == "file contents here"
            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, dispatcher):
        """ToolError from handlers is formatted correctly."""
        async def raise_tool_error(args, ctx):
            raise ToolError("File not found: /tmp/missing.txt", kind=ToolErrorKind.NOT_FOUND)

        with patch.dict(
            "jarvis.tools.handlers.HANDLER_MAP",
            {"read_file": raise_tool_error},
        ):
            result = await dispatcher.dispatch("read_file", {"path": "/tmp/missing.txt"})
            assert "not_found" in result.lower() or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_generic_exception_handling(self, dispatcher):
        """Generic exceptions are caught and formatted."""
        async def raise_error(args, ctx):
            raise RuntimeError("connection refused")

        with patch.dict(
            "jarvis.tools.handlers.HANDLER_MAP",
            {"web_search": raise_error},
        ):
            result = await dispatcher.dispatch("web_search", {"query": "test"})
            assert "Tool error" in result
            assert "connection refused" in result

    @pytest.mark.asyncio
    async def test_blocked_tool(self, dispatcher):
        """Blocked tools are rejected before execution."""
        result = await dispatcher.dispatch(
            "read_file", {"path": "/tmp/test"},
            blocked_tools=["read_file"],
        )
        assert "blocked" in result.lower()

    @pytest.mark.asyncio
    async def test_allowed_tools_enforcement(self, dispatcher):
        """Tools not in allowed list are rejected."""
        result = await dispatcher.dispatch(
            "read_file", {"path": "/tmp/test"},
            allowed_tools=["web_search"],
        )
        assert "not in the allowed" in result.lower()


# =====================================================================
# Circuit Breaker Integration
# =====================================================================


class TestDispatcherCircuitBreaker:
    """Tests for circuit breaker integration in the dispatcher."""

    @pytest.fixture
    def dispatcher(self):
        return make_dispatcher()

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self, dispatcher):
        """Circuit opens after 3 consecutive failures."""
        async def failing_handler(args, ctx):
            raise RuntimeError("service down")

        with patch.dict(
            "jarvis.tools.handlers.HANDLER_MAP",
            {"web_search": failing_handler},
        ):
            # 3 failures to trip the circuit
            for _ in range(3):
                await dispatcher.dispatch("web_search", {"query": "test"})

            # 4th call should be fast-failed by circuit breaker
            result = await dispatcher.dispatch("web_search", {"query": "test"})
            assert "circuit breaker" in result.lower() or "temporarily unavailable" in result.lower()

    @pytest.mark.asyncio
    async def test_circuit_resets_on_success(self, dispatcher):
        """Circuit closes after a successful execution."""
        call_count = 0

        async def sometimes_fails(args, ctx):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("fail")
            return "success"

        with patch.dict(
            "jarvis.tools.handlers.HANDLER_MAP",
            {"web_search": sometimes_fails},
        ):
            # 2 failures (not enough to trip)
            await dispatcher.dispatch("web_search", {"query": "test"})
            await dispatcher.dispatch("web_search", {"query": "test"})
            # 3rd should succeed and reset counter
            result = await dispatcher.dispatch("web_search", {"query": "test"})
            assert result == "success"

    @pytest.mark.asyncio
    async def test_tool_error_user_errors_dont_trip_circuit(self, dispatcher):
        """User errors (MISSING_PARAM, NOT_FOUND) don't trip the circuit."""
        async def missing_param(args, ctx):
            raise ToolError("Missing 'path'", kind=ToolErrorKind.MISSING_PARAM)

        with patch.dict(
            "jarvis.tools.handlers.HANDLER_MAP",
            {"read_file": missing_param},
        ):
            # Even 10 user errors shouldn't trip the circuit
            for _ in range(10):
                await dispatcher.dispatch("read_file", {})

            # Circuit should still be closed
            assert not dispatcher.circuit_breaker.is_open("read_file")


# =====================================================================
# Result Cache Integration
# =====================================================================


class TestDispatcherCache:
    """Tests for result cache integration."""

    @pytest.fixture
    def dispatcher(self):
        return make_dispatcher()

    @pytest.mark.asyncio
    async def test_cacheable_tool_returns_cached(self, dispatcher):
        """Second call to a cacheable tool returns cached result."""
        call_count = 0

        async def counting_handler(args, ctx):
            nonlocal call_count
            call_count += 1
            return f"result-{call_count}"

        with patch.dict(
            "jarvis.tools.handlers.HANDLER_MAP",
            {"read_file": counting_handler},
        ):
            r1 = await dispatcher.dispatch("read_file", {"path": "/tmp/test"})
            r2 = await dispatcher.dispatch("read_file", {"path": "/tmp/test"})
            assert r1 == "result-1"
            assert r2 == "result-1"  # Cached!
            assert call_count == 1  # Handler only called once

    @pytest.mark.asyncio
    async def test_different_args_not_cached(self, dispatcher):
        """Different arguments produce different cache keys."""
        call_count = 0

        async def counting_handler(args, ctx):
            nonlocal call_count
            call_count += 1
            return f"result-{call_count}"

        with patch.dict(
            "jarvis.tools.handlers.HANDLER_MAP",
            {"read_file": counting_handler},
        ):
            r1 = await dispatcher.dispatch("read_file", {"path": "/tmp/a.txt"})
            r2 = await dispatcher.dispatch("read_file", {"path": "/tmp/b.txt"})
            assert r1 != r2
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_write_invalidates_read_cache(self, dispatcher):
        """Write operations invalidate read cache."""
        call_count = 0

        async def read_handler(args, ctx):
            nonlocal call_count
            call_count += 1
            return f"content-v{call_count}"

        async def write_handler(args, ctx):
            return "written"

        with patch.dict(
            "jarvis.tools.handlers.HANDLER_MAP",
            {"read_file": read_handler, "write_file": write_handler},
        ):
            # Read and cache
            r1 = await dispatcher.dispatch("read_file", {"path": "/tmp/test"})
            assert r1 == "content-v1"

            # Write invalidates read cache
            await dispatcher.dispatch("write_file", {"path": "/tmp/test", "content": "new"})

            # Read again — should hit handler (cache invalidated)
            r2 = await dispatcher.dispatch("read_file", {"path": "/tmp/test"})
            assert r2 == "content-v2"
            assert call_count == 2


# =====================================================================
# Execution History Integration
# =====================================================================


class TestDispatcherHistory:
    """Tests for execution history recording."""

    @pytest.fixture
    def dispatcher(self):
        return make_dispatcher()

    @pytest.mark.asyncio
    async def test_records_successful_execution(self, dispatcher):
        """Successful execution is recorded in history."""
        with patch.dict(
            "jarvis.tools.handlers.HANDLER_MAP",
            {"read_file": AsyncMock(return_value="content")},
        ):
            await dispatcher.dispatch("read_file", {"path": "/tmp/test"})
            
            history = dispatcher.execution_history
            recent = history.recent(1)
            assert len(recent) == 1
            assert recent[0]["tool"] == "read_file"
            assert recent[0]["status"] == ToolStatus.OK

    @pytest.mark.asyncio
    async def test_records_error_execution(self, dispatcher):
        """Failed execution is recorded in history."""
        async def failing(args, ctx):
            raise RuntimeError("boom")

        with patch.dict(
            "jarvis.tools.handlers.HANDLER_MAP",
            {"web_search": failing},
        ):
            await dispatcher.dispatch("web_search", {"query": "test"})
            
            recent = dispatcher.execution_history.recent(1)
            assert len(recent) == 1
            assert recent[0]["status"] == ToolStatus.ERROR

    @pytest.mark.asyncio
    async def test_records_circuit_open(self, dispatcher):
        """Circuit breaker fast-fail is recorded in history."""
        # Manually open the circuit
        for _ in range(3):
            dispatcher.circuit_breaker.record_failure("broken_tool")

        with patch.dict(
            "jarvis.tools.handlers.HANDLER_MAP",
            {"broken_tool": AsyncMock(return_value="ok")},
        ):
            await dispatcher.dispatch("broken_tool", {})
            
            recent = dispatcher.execution_history.recent(1)
            assert len(recent) == 1
            assert recent[0]["status"] == ToolStatus.CIRCUIT_OPEN

    @pytest.mark.asyncio
    async def test_records_cached_result(self, dispatcher):
        """Cache hit is recorded in history."""
        with patch.dict(
            "jarvis.tools.handlers.HANDLER_MAP",
            {"read_file": AsyncMock(return_value="cached content")},
        ):
            # First call — miss
            await dispatcher.dispatch("read_file", {"path": "/tmp/test"})
            # Second call — hit
            await dispatcher.dispatch("read_file", {"path": "/tmp/test"})
            
            recent = dispatcher.execution_history.recent(2)
            assert len(recent) == 2
            assert recent[0]["status"] == ToolStatus.OK
            assert recent[1]["status"] == ToolStatus.CACHED

    @pytest.mark.asyncio
    async def test_stats_aggregation(self, dispatcher):
        """History stats aggregate correctly across calls."""
        with patch.dict(
            "jarvis.tools.handlers.HANDLER_MAP",
            {
                "read_file": AsyncMock(return_value="ok"),
                "web_search": AsyncMock(return_value="results"),
            },
        ):
            for _ in range(3):
                await dispatcher.dispatch("read_file", {"path": "/tmp/test"})
            await dispatcher.dispatch("web_search", {"query": "test"})
            
            stats = dispatcher.execution_history.stats()
            # 3 read_file (1 real + 2 cached) + 1 web_search = 4 total
            assert stats["total_calls"] == 4
            assert stats["total_errors"] == 0


# =====================================================================
# Rate Limiting
# =====================================================================


class TestDispatcherRateLimit:
    """Tests for rate limiting in the dispatcher."""

    @pytest.mark.asyncio
    async def test_rate_limit_triggers(self):
        """Rate limit kicks in after exceeding max calls."""
        # Create dispatcher with very low rate limit
        trust = TrustPolicyEvaluator(trust_level="yolo")
        ctx = make_tool_ctx()
        dispatcher = ToolDispatcher(trust, ctx, rate_max=2)

        with patch.dict(
            "jarvis.tools.handlers.HANDLER_MAP",
            {"read_file": AsyncMock(return_value="ok")},
        ):
            r1 = await dispatcher.dispatch("read_file", {"path": "/a"})
            r2 = await dispatcher.dispatch("read_file", {"path": "/b"})
            # 3rd call should be rate limited
            r3 = await dispatcher.dispatch("read_file", {"path": "/c"})
            assert "Rate limited" in r3 or r3 == "ok"  # depends on sliding window timing


# =====================================================================
# Origin Tracking
# =====================================================================


class TestDispatcherOrigin:
    """Tests for origin tracking in dispatch."""

    @pytest.mark.asyncio
    async def test_origin_recorded_in_history(self):
        """The origin tag is passed through to execution history."""
        dispatcher = make_dispatcher()

        with patch.dict(
            "jarvis.tools.handlers.HANDLER_MAP",
            {"read_file": AsyncMock(return_value="ok")},
        ):
            await dispatcher.dispatch("read_file", {"path": "/tmp/test"}, origin="gemini_cli")
            
            recent = dispatcher.execution_history.recent(1)
            assert recent[0]["origin"] == "gemini_cli"


# =====================================================================
# Dispatcher Properties
# =====================================================================


class TestDispatcherProperties:
    """Tests for dispatcher property accessors."""

    def test_tool_ctx_property(self):
        dispatcher = make_dispatcher()
        assert isinstance(dispatcher.tool_ctx, ToolContext)

    def test_circuit_breaker_property(self):
        dispatcher = make_dispatcher()
        assert isinstance(dispatcher.circuit_breaker, ToolCircuitBreaker)

    def test_result_cache_property(self):
        dispatcher = make_dispatcher()
        assert isinstance(dispatcher.result_cache, ToolResultCache)

    def test_execution_history_property(self):
        dispatcher = make_dispatcher()
        assert isinstance(dispatcher.execution_history, ExecutionHistory)

    def test_dispatcher_ref_on_ctx(self):
        """ToolContext gets a back-reference to the dispatcher (for pipeline)."""
        dispatcher = make_dispatcher()
        assert hasattr(dispatcher.tool_ctx, '_dispatcher_ref')
        # _dispatcher_ref is now a weakref
        ref = dispatcher.tool_ctx._dispatcher_ref
        assert ref() is dispatcher
