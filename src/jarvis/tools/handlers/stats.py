"""Stats handler: tool_stats — debug dashboard for circuit breaker, cache, history, health, and middleware."""
from __future__ import annotations

import json
from typing import Any

from ._context import (
    ToolContext,
    ToolError,
    ToolErrorKind,
    invalid_param,
)


async def handle_tool_stats(args: dict[str, Any], ctx: ToolContext) -> str:
    """Return a diagnostic dashboard of the JARVIS tool system.

    Shows circuit breaker states, cache hit rates, execution history,
    per-tool latency stats, operator telemetry, middleware status,
    and overall system health score.

    Args:
        section: Which section(s) to return. Options:
            "all" (default), "circuit_breaker", "cache", "history",
            "operator_telemetry", "middleware", "health", "summary"
        history_count: Number of recent history entries to return (default 20).
    """
    section = str(args.get("section") or "all").strip().lower()
    history_count = max(1, min(int(args.get("history_count") or 20), 100))

    valid_sections = {
        "all", "circuit_breaker", "cache", "history",
        "operator_telemetry", "middleware", "health", "summary",
    }
    if section not in valid_sections:
        raise invalid_param(
            "section",
            f"must be one of: {', '.join(sorted(valid_sections))}",
            tool="tool_stats",
        )

    _ref = getattr(ctx, "_dispatcher_ref", None)
    dispatcher = _ref() if callable(_ref) else _ref
    result: dict[str, Any] = {}

    if dispatcher is not None:
        cb = getattr(dispatcher, "circuit_breaker", None)
        cache = getattr(dispatcher, "result_cache", None)
        history = getattr(dispatcher, "execution_history", None)
        mw_stack = getattr(dispatcher, "middleware", None)

        # ── Health dashboard (full report or summary) ──
        if section in ("all", "health"):
            try:
                from ..health import HealthDashboard
                dashboard = HealthDashboard(dispatcher)
                result["health"] = dashboard.report()
            except (ImportError, OSError, RuntimeError) as exc:
                result["health"] = {"error": str(exc)}

        if section == "summary":
            try:
                from ..health import HealthDashboard
                dashboard = HealthDashboard(dispatcher)
                result["health_summary"] = dashboard.summary()
            except (ImportError, OSError, RuntimeError):
                pass

        # ── Circuit breaker ──
        if section in ("all", "circuit_breaker", "summary"):
            if cb:
                cb_status = cb.status()
                result["circuit_breaker"] = {
                    "tracked_tools": len(cb_status),
                    "open_circuits": [
                        name for name, info in cb_status.items()
                        if info["state"] == "open"
                    ],
                    "half_open_circuits": [
                        name for name, info in cb_status.items()
                        if info["state"] == "half_open"
                    ],
                    "details": cb_status if section != "summary" else None,
                }

        # ── Cache ──
        if section in ("all", "cache", "summary"):
            if cache:
                result["cache"] = cache.stats()

        # ── History ──
        if section in ("all", "history"):
            if history:
                result["history"] = {
                    "stats": history.stats(),
                    "recent": history.recent(history_count),
                }
        elif section == "summary" and history:
            result["history_summary"] = history.stats()

        # ── Middleware ──
        if section in ("all", "middleware"):
            if mw_stack:
                mw_list = [
                    {"name": mw.name, "order": mw.order}
                    for mw in mw_stack.middlewares
                ]
                # Get metrics snapshot if MetricsMiddleware is present
                metrics = None
                for mw in mw_stack.middlewares:
                    if hasattr(mw, "snapshot"):
                        metrics = mw.snapshot()
                        break
                result["middleware"] = {
                    "count": len(mw_list),
                    "stack": mw_list,
                    "metrics": metrics,
                }

    # ── Operator telemetry ──
    if section in ("all", "operator_telemetry", "summary"):
        op = ctx.desktop_operator
        if op and hasattr(op, "telemetry"):
            result["desktop_telemetry"] = op.telemetry()

    if not result:
        result["message"] = (
            "No stats available — dispatcher or subsystems not initialized. "
            "Stats populate as tools are executed."
        )

    return json.dumps(result, indent=2, default=str)
