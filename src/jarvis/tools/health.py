"""
JARVIS Tool Health Dashboard — unified system health monitoring.

Aggregates metrics from all tool subsystems into a single JSON dashboard:
  - Circuit breaker states (open/closed/half-open per tool)
  - Cache hit rates and memory usage
  - Execution history stats (per-tool call counts, error rates)
  - Adaptive timeout states (current effective timeouts)
  - Middleware stack status
  - Latency percentiles (P50/P95/P99)

Usage:
    dashboard = HealthDashboard(dispatcher)
    report = dashboard.report()
    print(json.dumps(report, indent=2))
"""
from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .dispatcher import ToolDispatcher
    from .middleware import MetricsMiddleware


class HealthDashboard:
    """Unified health monitoring for the JARVIS tool system.

    Reads from dispatcher subsystems (circuit breaker, cache, history,
    adaptive timeouts, middleware) to produce a comprehensive health report.
    """

    def __init__(self, dispatcher: "ToolDispatcher") -> None:
        self._dispatcher = dispatcher

    def report(self) -> dict[str, Any]:
        """Generate a comprehensive health report.

        Returns a JSON-serializable dict with all subsystem states.
        """
        return {
            "timestamp": time.time(),
            "circuit_breakers": self._circuit_breaker_report(),
            "cache": self._cache_report(),
            "execution_history": self._history_report(),
            "adaptive_timeouts": self._timeout_report(),
            "middleware": self._middleware_report(),
            "overall_health": self._compute_overall_health(),
        }

    def summary(self) -> dict[str, Any]:
        """Generate a compact health summary (for CLI display).

        Returns key metrics without per-tool breakdowns.
        """
        stats = self._dispatcher.execution_history.stats()
        cb = self._dispatcher.circuit_breaker
        open_circuits = [
            tool for tool in self._known_tools()
            if cb.is_open(tool)
        ]
        cache = self._dispatcher.result_cache

        return {
            "total_calls": stats.get("total_calls", 0),
            "total_errors": stats.get("total_errors", 0),
            "error_rate_pct": round(
                stats.get("total_errors", 0) / max(stats.get("total_calls", 1), 1) * 100, 1
            ),
            "open_circuits": open_circuits,
            "cache_entries": len(cache._cache) if hasattr(cache, "_cache") else 0,
            "middleware_count": len(self._dispatcher.middleware),
            "health": "degraded" if open_circuits else "healthy",
        }

    # -- Subsystem reports ------------------------------------------------

    def _circuit_breaker_report(self) -> dict[str, Any]:
        """Report on all circuit breaker states."""
        cb = self._dispatcher.circuit_breaker
        tools = self._known_tools()
        states: dict[str, dict[str, Any]] = {}

        open_count = 0
        for tool in tools:
            state = cb.state(tool)
            is_open = cb.is_open(tool)
            if is_open:
                open_count += 1
            states[tool] = {
                "state": state.value,
                "is_open": is_open,
            }

        return {
            "total_tools": len(tools),
            "open_circuits": open_count,
            "closed_circuits": len(tools) - open_count,
            "cooldown_seconds": cb.cooldown_seconds,
            "failure_threshold": cb.failure_threshold,
            "per_tool": states,
        }

    def _cache_report(self) -> dict[str, Any]:
        """Report on result cache state."""
        cache = self._dispatcher.result_cache
        stats = cache.stats() if hasattr(cache, "stats") else {}

        return {
            "entries": stats.get("entries", 0),
            "max_entries": stats.get("max_entries", 0),
            "hits": stats.get("hits", 0),
            "misses": stats.get("misses", 0),
            "hit_rate_pct": round(
                stats.get("hits", 0) / max(stats.get("hits", 0) + stats.get("misses", 0), 1) * 100, 1
            ),
            "evictions": stats.get("evictions", 0),
        }

    def _history_report(self) -> dict[str, Any]:
        """Report on execution history stats."""
        history = self._dispatcher.execution_history
        stats = history.stats()

        # Per-tool breakdown from recent entries
        recent = history.recent(100)
        tool_stats: dict[str, dict[str, int]] = {}
        for entry in recent:
            tool = entry.get("tool", "unknown")
            status = entry.get("status", "unknown")
            if tool not in tool_stats:
                tool_stats[tool] = {"calls": 0, "errors": 0, "cached": 0}
            tool_stats[tool]["calls"] += 1
            if status in ("error", "timeout"):
                tool_stats[tool]["errors"] += 1
            elif status == "cached":
                tool_stats[tool]["cached"] += 1

        return {
            "total_calls": stats.get("total_calls", 0),
            "total_errors": stats.get("total_errors", 0),
            "error_rate_pct": round(
                stats.get("total_errors", 0) / max(stats.get("total_calls", 1), 1) * 100, 1
            ),
            "recent_100_breakdown": tool_stats,
        }

    def _timeout_report(self) -> dict[str, Any]:
        """Report on adaptive timeout states."""
        adaptive = self._dispatcher._adaptive
        tools = self._known_tools()
        timeouts: dict[str, dict[str, float]] = {}

        for tool in tools:
            static = float(self._dispatcher._tool_timeouts.get(tool, self._dispatcher._timeout))
            effective = adaptive.get_timeout(tool, static)
            samples = adaptive._latencies.get(tool)
            timeouts[tool] = {
                "static_ceiling_s": static,
                "effective_s": round(effective, 1),
                "samples": len(samples) if samples else 0,
                "adapted": effective < static,
            }

        adapted_count = sum(1 for t in timeouts.values() if t["adapted"])
        return {
            "total_tools": len(tools),
            "adapted_tools": adapted_count,
            "per_tool": timeouts,
        }

    def _middleware_report(self) -> dict[str, Any]:
        """Report on middleware stack."""
        mw_stack = self._dispatcher.middleware
        middlewares = [
            {"name": mw.name, "order": mw.order}
            for mw in mw_stack.middlewares
        ]

        # Try to get metrics from MetricsMiddleware if present
        metrics_snapshot = None
        for mw in mw_stack.middlewares:
            if hasattr(mw, "snapshot"):
                metrics_snapshot = mw.snapshot()
                break

        return {
            "count": len(middlewares),
            "stack": middlewares,
            "metrics": metrics_snapshot,
        }

    def _compute_overall_health(self) -> dict[str, Any]:
        """Compute overall system health score."""
        stats = self._dispatcher.execution_history.stats()
        cb = self._dispatcher.circuit_breaker
        open_circuits = sum(
            1 for tool in self._known_tools() if cb.is_open(tool)
        )

        total = max(stats.get("total_calls", 0), 1)
        errors = stats.get("total_errors", 0)
        error_rate = errors / total

        # Health score: 100 = perfect, 0 = everything broken
        score = 100.0
        score -= error_rate * 200  # -20 per 10% error rate
        score -= open_circuits * 10  # -10 per open circuit
        score = max(0.0, min(100.0, score))

        if score >= 90:
            status = "healthy"
        elif score >= 70:
            status = "degraded"
        elif score >= 50:
            status = "warning"
        else:
            status = "critical"

        return {
            "score": round(score, 1),
            "status": status,
            "open_circuits": open_circuits,
            "error_rate_pct": round(error_rate * 100, 1),
        }

    # -- Helpers ----------------------------------------------------------

    def _known_tools(self) -> list[str]:
        """Get list of tools that have been dispatched at least once."""
        recent = self._dispatcher.execution_history.recent(500)
        return sorted(set(entry.get("tool", "") for entry in recent if entry.get("tool")))
