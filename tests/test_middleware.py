"""
Tests for JARVIS Tool Middleware — pluggable pre/post dispatch hooks.

Covers:
  - MiddlewareContext creation and fields
  - MiddlewareStack ordering (before: low→high, after: high→low)
  - LoggingMiddleware structured logging
  - MetricsMiddleware counters, latency percentiles, reset
  - AuditTrailMiddleware append-only log, search, eviction
  - InputSanitizerMiddleware null bytes, whitespace, truncation
  - OutputTruncationMiddleware oversized output handling
  - PerToolRateLimitMiddleware per-tool throttling
  - create_default_stack factory
  - Error isolation (failing middleware doesn't crash pipeline)
  - skip_execution control flow
"""
from __future__ import annotations

import asyncio
import pytest

from jarvis.tools.middleware import (
    AuditTrailMiddleware,
    InputSanitizerMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    Middleware,
    MiddlewareContext,
    MiddlewareStack,
    OutputTruncationMiddleware,
    PerToolRateLimitMiddleware,
    create_default_stack,
)


# ═══════════════════════════════════════════════════════════════════
# MiddlewareContext
# ═══════════════════════════════════════════════════════════════════


class TestMiddlewareContext:
    def test_defaults(self):
        ctx = MiddlewareContext()
        assert ctx.tool == ""
        assert ctx.args == {}
        assert ctx.origin == "unknown"
        assert len(ctx.trace_id) == 12
        assert ctx.status == "pending"
        assert ctx.skip_execution is False

    def test_custom_fields(self):
        ctx = MiddlewareContext(tool="read_file", args={"path": "/tmp"}, origin="cli")
        assert ctx.tool == "read_file"
        assert ctx.args["path"] == "/tmp"
        assert ctx.origin == "cli"

    def test_metadata_bag(self):
        ctx = MiddlewareContext()
        ctx.metadata["custom_key"] = "custom_value"
        assert ctx.metadata["custom_key"] == "custom_value"

    def test_unique_trace_ids(self):
        ids = {MiddlewareContext().trace_id for _ in range(100)}
        assert len(ids) == 100  # All unique


# ═══════════════════════════════════════════════════════════════════
# MiddlewareStack
# ═══════════════════════════════════════════════════════════════════


class TestMiddlewareStack:
    def test_empty_stack(self):
        stack = MiddlewareStack()
        assert len(stack) == 0

    def test_add_returns_self(self):
        stack = MiddlewareStack()
        result = stack.add(LoggingMiddleware())
        assert result is stack  # Chainable

    def test_ordering_by_order(self):
        stack = MiddlewareStack()

        class First(Middleware):
            @property
            def order(self): return 1

        class Last(Middleware):
            @property
            def order(self): return 999

        stack.add(Last())
        stack.add(First())
        assert stack.middlewares[0].order == 1
        assert stack.middlewares[1].order == 999

    def test_remove_by_name(self):
        stack = MiddlewareStack()
        stack.add(LoggingMiddleware())
        assert len(stack) == 1
        removed = stack.remove("LoggingMiddleware")
        assert removed is True
        assert len(stack) == 0

    def test_remove_nonexistent(self):
        stack = MiddlewareStack()
        removed = stack.remove("NonExistent")
        assert removed is False

    @pytest.mark.asyncio
    async def test_before_order(self):
        """Before hooks run low→high order."""
        execution_order = []

        class A(Middleware):
            @property
            def order(self): return 10
            async def before(self, ctx):
                execution_order.append("A")

        class B(Middleware):
            @property
            def order(self): return 20
            async def before(self, ctx):
                execution_order.append("B")

        stack = MiddlewareStack()
        stack.add(B()).add(A())
        await stack.run_before(MiddlewareContext())
        assert execution_order == ["A", "B"]

    @pytest.mark.asyncio
    async def test_after_reverse_order(self):
        """After hooks run high→low order (reverse onion)."""
        execution_order = []

        class A(Middleware):
            @property
            def order(self): return 10
            async def after(self, ctx):
                execution_order.append("A")

        class B(Middleware):
            @property
            def order(self): return 20
            async def after(self, ctx):
                execution_order.append("B")

        stack = MiddlewareStack()
        stack.add(A()).add(B())
        await stack.run_after(MiddlewareContext())
        assert execution_order == ["B", "A"]

    @pytest.mark.asyncio
    async def test_before_stops_on_skip(self):
        """Before hooks stop if skip_execution is set."""
        execution_order = []

        class Skipper(Middleware):
            @property
            def order(self): return 10
            async def before(self, ctx):
                execution_order.append("skipper")
                ctx.skip_execution = True
                ctx.skip_reason = "rate limited"

        class Never(Middleware):
            @property
            def order(self): return 20
            async def before(self, ctx):
                execution_order.append("never")

        stack = MiddlewareStack()
        stack.add(Skipper()).add(Never())
        ctx = MiddlewareContext()
        await stack.run_before(ctx)
        assert execution_order == ["skipper"]
        assert ctx.skip_execution is True

    @pytest.mark.asyncio
    async def test_error_isolation_before(self):
        """Failing middleware doesn't crash the pipeline."""
        execution_order = []

        class Broken(Middleware):
            @property
            def order(self): return 10
            async def before(self, ctx):
                raise RuntimeError("middleware crash")

        class Healthy(Middleware):
            @property
            def order(self): return 20
            async def before(self, ctx):
                execution_order.append("healthy")

        stack = MiddlewareStack()
        stack.add(Broken()).add(Healthy())
        await stack.run_before(MiddlewareContext())
        assert execution_order == ["healthy"]

    @pytest.mark.asyncio
    async def test_error_isolation_after(self):
        """Failing after hook doesn't crash other hooks."""
        execution_order = []

        class BrokenAfter(Middleware):
            @property
            def order(self): return 20
            async def after(self, ctx):
                raise RuntimeError("after crash")

        class HealthyAfter(Middleware):
            @property
            def order(self): return 10
            async def after(self, ctx):
                execution_order.append("healthy")

        stack = MiddlewareStack()
        stack.add(HealthyAfter()).add(BrokenAfter())
        await stack.run_after(MiddlewareContext())
        # BrokenAfter runs first (reverse order), crashes, HealthyAfter still runs
        assert execution_order == ["healthy"]


# ═══════════════════════════════════════════════════════════════════
# MetricsMiddleware
# ═══════════════════════════════════════════════════════════════════


class TestMetricsMiddleware:
    @pytest.mark.asyncio
    async def test_records_calls(self):
        mw = MetricsMiddleware()
        ctx = MiddlewareContext(tool="read_file", status="ok", duration_ms=50)
        await mw.after(ctx)
        snap = mw.snapshot()
        assert snap["total_calls"] == 1
        assert snap["tools"]["read_file"]["calls"] == 1

    @pytest.mark.asyncio
    async def test_records_errors(self):
        mw = MetricsMiddleware()
        ctx = MiddlewareContext(tool="web_search", status="error", duration_ms=100)
        await mw.after(ctx)
        snap = mw.snapshot()
        assert snap["total_errors"] == 1
        assert snap["tools"]["web_search"]["errors"] == 1

    @pytest.mark.asyncio
    async def test_latency_percentiles(self):
        mw = MetricsMiddleware()
        for i in range(100):
            ctx = MiddlewareContext(tool="fast_tool", status="ok", duration_ms=float(i))
            await mw.after(ctx)
        snap = mw.snapshot()
        latency = snap["tools"]["fast_tool"]["latency_ms"]
        assert latency["p50"] >= 40  # ~50th percentile
        assert latency["p95"] >= 90  # ~95th percentile

    @pytest.mark.asyncio
    async def test_status_distribution(self):
        mw = MetricsMiddleware()
        for status in ["ok", "ok", "ok", "error", "cached"]:
            ctx = MiddlewareContext(tool="mixed", status=status, duration_ms=10)
            await mw.after(ctx)
        snap = mw.snapshot()
        dist = snap["tools"]["mixed"]["status_distribution"]
        assert dist["ok"] == 3
        assert dist["error"] == 1
        assert dist["cached"] == 1

    @pytest.mark.asyncio
    async def test_reset(self):
        mw = MetricsMiddleware()
        ctx = MiddlewareContext(tool="t", status="ok", duration_ms=1)
        await mw.after(ctx)
        assert mw.snapshot()["total_calls"] == 1
        mw.reset()
        assert mw.snapshot()["total_calls"] == 0

    @pytest.mark.asyncio
    async def test_rolling_window(self):
        """Latency window caps at 1000 samples per tool."""
        mw = MetricsMiddleware()
        for i in range(1200):
            ctx = MiddlewareContext(tool="heavy", status="ok", duration_ms=float(i))
            await mw.after(ctx)
        # Should be capped at ~500 after eviction
        assert len(mw._durations["heavy"]) <= 1000


# ═══════════════════════════════════════════════════════════════════
# AuditTrailMiddleware
# ═══════════════════════════════════════════════════════════════════


class TestAuditTrailMiddleware:
    @pytest.mark.asyncio
    async def test_records_entry(self):
        mw = AuditTrailMiddleware()
        ctx = MiddlewareContext(tool="write_file", origin="user", status="ok", duration_ms=25)
        await mw.after(ctx)
        assert mw.size == 1
        entry = mw.recent(1)[0]
        assert entry["tool"] == "write_file"
        assert entry["origin"] == "user"

    @pytest.mark.asyncio
    async def test_max_entries_eviction(self):
        mw = AuditTrailMiddleware(max_entries=5)
        for i in range(10):
            ctx = MiddlewareContext(tool=f"tool_{i}", status="ok")
            await mw.after(ctx)
        assert mw.size == 5
        # Should have the last 5
        tools = [e["tool"] for e in mw.recent(10)]
        assert tools == ["tool_5", "tool_6", "tool_7", "tool_8", "tool_9"]

    @pytest.mark.asyncio
    async def test_search_by_tool(self):
        mw = AuditTrailMiddleware()
        for tool in ["read_file", "write_file", "read_file"]:
            ctx = MiddlewareContext(tool=tool, status="ok")
            await mw.after(ctx)
        results = mw.search(tool="read_file")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_by_status(self):
        mw = AuditTrailMiddleware()
        for status in ["ok", "error", "ok", "error", "ok"]:
            ctx = MiddlewareContext(tool="t", status=status)
            await mw.after(ctx)
        results = mw.search(status="error")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_error_recorded(self):
        mw = AuditTrailMiddleware()
        ctx = MiddlewareContext(tool="t", status="error")
        ctx.error = ValueError("test error")
        await mw.after(ctx)
        entry = mw.recent(1)[0]
        assert entry["error"] == "test error"


# ═══════════════════════════════════════════════════════════════════
# InputSanitizerMiddleware
# ═══════════════════════════════════════════════════════════════════


class TestInputSanitizerMiddleware:
    @pytest.mark.asyncio
    async def test_strips_whitespace(self):
        mw = InputSanitizerMiddleware()
        ctx = MiddlewareContext(args={"path": "  /tmp/test.txt  "})
        await mw.before(ctx)
        assert ctx.args["path"] == "/tmp/test.txt"

    @pytest.mark.asyncio
    async def test_removes_null_bytes(self):
        mw = InputSanitizerMiddleware()
        ctx = MiddlewareContext(args={"text": "hello\x00world"})
        await mw.before(ctx)
        assert ctx.args["text"] == "helloworld"

    @pytest.mark.asyncio
    async def test_truncates_oversized(self):
        mw = InputSanitizerMiddleware(max_arg_length=10)
        ctx = MiddlewareContext(args={"data": "a" * 100})
        await mw.before(ctx)
        assert len(ctx.args["data"]) < 100
        assert "truncated" in ctx.args["data"]

    @pytest.mark.asyncio
    async def test_non_string_args_unchanged(self):
        mw = InputSanitizerMiddleware()
        ctx = MiddlewareContext(args={"count": 42, "flag": True})
        await mw.before(ctx)
        assert ctx.args["count"] == 42
        assert ctx.args["flag"] is True


# ═══════════════════════════════════════════════════════════════════
# OutputTruncationMiddleware
# ═══════════════════════════════════════════════════════════════════


class TestOutputTruncationMiddleware:
    @pytest.mark.asyncio
    async def test_small_output_unchanged(self):
        mw = OutputTruncationMiddleware(max_length=100)
        ctx = MiddlewareContext(result="short output")
        await mw.after(ctx)
        assert ctx.result == "short output"
        assert "truncated" not in ctx.metadata

    @pytest.mark.asyncio
    async def test_large_output_truncated(self):
        mw = OutputTruncationMiddleware(max_length=100, keep_head=50, keep_tail=20)
        ctx = MiddlewareContext(result="x" * 200)
        await mw.after(ctx)
        assert len(ctx.result) < 200
        assert "truncated" in ctx.result
        assert ctx.metadata["truncated"] is True
        assert ctx.metadata["original_length"] == 200

    @pytest.mark.asyncio
    async def test_empty_result_ok(self):
        mw = OutputTruncationMiddleware()
        ctx = MiddlewareContext(result="")
        await mw.after(ctx)
        assert ctx.result == ""


# ═══════════════════════════════════════════════════════════════════
# PerToolRateLimitMiddleware
# ═══════════════════════════════════════════════════════════════════


class TestPerToolRateLimitMiddleware:
    @pytest.mark.asyncio
    async def test_under_limit_passes(self):
        mw = PerToolRateLimitMiddleware(limits={"web_search": 5})
        ctx = MiddlewareContext(tool="web_search")
        await mw.before(ctx)
        assert ctx.skip_execution is False

    @pytest.mark.asyncio
    async def test_over_limit_blocks(self):
        mw = PerToolRateLimitMiddleware(limits={"web_search": 2}, window_seconds=60)
        for _ in range(2):
            ctx = MiddlewareContext(tool="web_search")
            await mw.before(ctx)
        # 3rd call should be blocked
        ctx = MiddlewareContext(tool="web_search")
        await mw.before(ctx)
        assert ctx.skip_execution is True
        assert "Rate limited" in ctx.result

    @pytest.mark.asyncio
    async def test_no_limit_passes(self):
        mw = PerToolRateLimitMiddleware(limits={"web_search": 2})
        ctx = MiddlewareContext(tool="read_file")  # No limit for read_file
        await mw.before(ctx)
        assert ctx.skip_execution is False

    @pytest.mark.asyncio
    async def test_set_limit(self):
        mw = PerToolRateLimitMiddleware()
        mw.set_limit("custom_tool", 1)
        ctx = MiddlewareContext(tool="custom_tool")
        await mw.before(ctx)
        assert ctx.skip_execution is False
        ctx2 = MiddlewareContext(tool="custom_tool")
        await mw.before(ctx2)
        assert ctx2.skip_execution is True

    @pytest.mark.asyncio
    async def test_remove_limit(self):
        mw = PerToolRateLimitMiddleware(limits={"t": 1})
        mw.remove_limit("t")
        for _ in range(10):
            ctx = MiddlewareContext(tool="t")
            await mw.before(ctx)
            assert ctx.skip_execution is False


# ═══════════════════════════════════════════════════════════════════
# Default Stack Factory
# ═══════════════════════════════════════════════════════════════════


class TestDefaultStack:
    def test_creates_5_middlewares(self):
        stack = create_default_stack()
        assert len(stack) == 5

    def test_ordered_correctly(self):
        stack = create_default_stack()
        orders = [mw.order for mw in stack.middlewares]
        assert orders == sorted(orders)

    def test_has_expected_types(self):
        stack = create_default_stack()
        names = {mw.name for mw in stack.middlewares}
        assert "InputSanitizerMiddleware" in names
        assert "LoggingMiddleware" in names
        assert "MetricsMiddleware" in names
        assert "AuditTrailMiddleware" in names
        assert "OutputTruncationMiddleware" in names


# ═══════════════════════════════════════════════════════════════════
# Integration: Full Stack Run
# ═══════════════════════════════════════════════════════════════════


class TestFullStackIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Run a complete before→after cycle through the default stack."""
        stack = create_default_stack()
        ctx = MiddlewareContext(
            tool="read_file",
            args={"path": "  /tmp/test.txt\x00  "},
            origin="test",
        )
        # Before hooks
        await stack.run_before(ctx)
        # Input should be sanitized
        assert ctx.args["path"] == "/tmp/test.txt"

        # Simulate handler execution
        ctx.result = "file contents here"
        ctx.status = "ok"
        ctx.duration_ms = 42.5

        # After hooks
        await stack.run_after(ctx)

        # Metrics should have recorded
        metrics_mw = next(
            mw for mw in stack.middlewares if isinstance(mw, MetricsMiddleware)
        )
        snap = metrics_mw.snapshot()
        assert snap["total_calls"] == 1
        assert snap["tools"]["read_file"]["calls"] == 1

        # Audit trail should have recorded
        audit_mw = next(
            mw for mw in stack.middlewares if isinstance(mw, AuditTrailMiddleware)
        )
        assert audit_mw.size == 1
        assert audit_mw.recent(1)[0]["tool"] == "read_file"
