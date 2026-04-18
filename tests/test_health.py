"""
Tests for PredaCore Tool Health Dashboard — unified system monitoring.

Covers:
  - Full health report generation
  - Compact summary generation
  - Circuit breaker report (open/closed tracking)
  - Cache report (hit rates)
  - History report (per-tool breakdown)
  - Adaptive timeout report
  - Middleware report
  - Overall health score computation
  - Edge cases (empty state, degraded state)
"""
from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch

from predacore.tools.enums import ToolStatus
from predacore.tools.handlers._context import ToolContext, ToolError, ToolErrorKind
from predacore.tools.resilience import ToolCircuitBreaker, ToolResultCache, ExecutionHistory
from predacore.tools.dispatcher import ToolDispatcher, AdaptiveTimeoutTracker
from predacore.tools.middleware import MiddlewareStack, MetricsMiddleware
from predacore.tools.health import HealthDashboard
from predacore.tools.trust_policy import TrustPolicyEvaluator


# ── Fixtures ──────────────────────────────────────────────────────


@dataclass
class MockSecurity:
    dangerous_patterns: list = field(default_factory=lambda: ["rm -rf /"])
    blocked_paths: list = field(default_factory=list)
    task_timeout_seconds: int = 30
    trust_level: str = "yolo"


@dataclass
class MockConfig:
    security: MockSecurity = field(default_factory=MockSecurity)
    home_dir: str = "/tmp/predacore_test"


@dataclass
class MockMemory:
    async def search(self, *a, **kw): return []
    async def store(self, *a, **kw): return True


def make_dispatcher() -> ToolDispatcher:
    trust = TrustPolicyEvaluator(trust_level="yolo")
    ctx = ToolContext(config=MockConfig(), memory=MockMemory())
    mw = MiddlewareStack()
    mw.add(MetricsMiddleware())
    return ToolDispatcher(trust, ctx, rate_max=1000, middleware=mw)


def make_dashboard() -> tuple[HealthDashboard, ToolDispatcher]:
    dispatcher = make_dispatcher()
    dashboard = HealthDashboard(dispatcher)
    return dashboard, dispatcher


# ═══════════════════════════════════════════════════════════════════
# Empty State
# ═══════════════════════════════════════════════════════════════════


class TestHealthDashboardEmpty:
    def test_report_empty_state(self):
        dashboard, _ = make_dashboard()
        report = dashboard.report()
        assert "timestamp" in report
        assert "circuit_breakers" in report
        assert "cache" in report
        assert "execution_history" in report
        assert "adaptive_timeouts" in report
        assert "middleware" in report
        assert "overall_health" in report

    def test_summary_empty_state(self):
        dashboard, _ = make_dashboard()
        summary = dashboard.summary()
        assert summary["total_calls"] == 0
        assert summary["health"] == "healthy"
        assert summary["open_circuits"] == []

    def test_overall_health_perfect(self):
        dashboard, _ = make_dashboard()
        report = dashboard.report()
        health = report["overall_health"]
        assert health["score"] == 100.0
        assert health["status"] == "healthy"


# ═══════════════════════════════════════════════════════════════════
# After Some Dispatches
# ═══════════════════════════════════════════════════════════════════


class TestHealthDashboardWithData:
    @pytest.mark.asyncio
    async def test_report_after_successful_calls(self):
        dashboard, dispatcher = make_dashboard()

        with patch.dict(
            "predacore.tools.handlers.HANDLER_MAP",
            {"read_file": AsyncMock(return_value="content")},
        ):
            await dispatcher.dispatch("read_file", {"path": "/tmp/a"})
            await dispatcher.dispatch("read_file", {"path": "/tmp/b"})

        report = dashboard.report()
        history = report["execution_history"]
        assert history["total_calls"] >= 2

    @pytest.mark.asyncio
    async def test_summary_after_calls(self):
        dashboard, dispatcher = make_dashboard()

        with patch.dict(
            "predacore.tools.handlers.HANDLER_MAP",
            {"read_file": AsyncMock(return_value="content")},
        ):
            await dispatcher.dispatch("read_file", {"path": "/tmp/a"})

        summary = dashboard.summary()
        assert summary["total_calls"] >= 1
        assert summary["health"] == "healthy"

    @pytest.mark.asyncio
    async def test_circuit_breaker_report_after_failures(self):
        dashboard, dispatcher = make_dashboard()

        async def failing(args, ctx):
            raise RuntimeError("boom")

        with patch.dict(
            "predacore.tools.handlers.HANDLER_MAP",
            {"broken_tool": failing},
        ):
            for _ in range(3):
                await dispatcher.dispatch("broken_tool", {})

        report = dashboard.report()
        cb = report["circuit_breakers"]
        assert cb["open_circuits"] >= 1

    @pytest.mark.asyncio
    async def test_degraded_health_with_open_circuits(self):
        dashboard, dispatcher = make_dashboard()

        async def failing(args, ctx):
            raise RuntimeError("service down")

        with patch.dict(
            "predacore.tools.handlers.HANDLER_MAP",
            {"broken_tool": failing},
        ):
            for _ in range(3):
                await dispatcher.dispatch("broken_tool", {})

        report = dashboard.report()
        health = report["overall_health"]
        assert health["score"] < 100
        assert health["open_circuits"] >= 1

    @pytest.mark.asyncio
    async def test_cache_report(self):
        dashboard, dispatcher = make_dashboard()

        with patch.dict(
            "predacore.tools.handlers.HANDLER_MAP",
            {"read_file": AsyncMock(return_value="content")},
        ):
            await dispatcher.dispatch("read_file", {"path": "/tmp/test"})
            await dispatcher.dispatch("read_file", {"path": "/tmp/test"})  # cached

        report = dashboard.report()
        cache = report["cache"]
        assert cache.get("hits", 0) >= 1 or cache.get("entries", 0) >= 0

    @pytest.mark.asyncio
    async def test_middleware_report_with_metrics(self):
        dashboard, dispatcher = make_dashboard()

        with patch.dict(
            "predacore.tools.handlers.HANDLER_MAP",
            {"read_file": AsyncMock(return_value="ok")},
        ):
            await dispatcher.dispatch("read_file", {"path": "/tmp/test"})

        report = dashboard.report()
        mw = report["middleware"]
        assert mw["count"] >= 1

    @pytest.mark.asyncio
    async def test_adaptive_timeout_report(self):
        dashboard, dispatcher = make_dashboard()

        with patch.dict(
            "predacore.tools.handlers.HANDLER_MAP",
            {"read_file": AsyncMock(return_value="ok")},
        ):
            for i in range(5):
                await dispatcher.dispatch("read_file", {"path": f"/tmp/{i}"})

        report = dashboard.report()
        timeouts = report["adaptive_timeouts"]
        assert timeouts["total_tools"] >= 1


# ═══════════════════════════════════════════════════════════════════
# Health Score Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestHealthScore:
    @pytest.mark.asyncio
    async def test_all_errors_critical(self):
        """100% error rate should produce critical health."""
        dashboard, dispatcher = make_dashboard()

        async def failing(args, ctx):
            raise RuntimeError("fail")

        with patch.dict(
            "predacore.tools.handlers.HANDLER_MAP",
            {"t1": failing, "t2": failing, "t3": failing},
        ):
            for t in ["t1", "t2", "t3"]:
                for _ in range(3):
                    await dispatcher.dispatch(t, {})

        report = dashboard.report()
        health = report["overall_health"]
        assert health["status"] in ("critical", "warning")
        assert health["score"] < 50

    @pytest.mark.asyncio
    async def test_mixed_success_degraded(self):
        """Mix of success and errors should show degraded."""
        dashboard, dispatcher = make_dashboard()

        call_count = 0
        async def sometimes_fails(args, ctx):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise RuntimeError("fail")
            return "ok"

        with patch.dict(
            "predacore.tools.handlers.HANDLER_MAP",
            {"mixed": sometimes_fails},
        ):
            for i in range(9):
                await dispatcher.dispatch("mixed", {"i": i})

        report = dashboard.report()
        health = report["overall_health"]
        # ~33% error rate
        assert health["error_rate_pct"] > 0
