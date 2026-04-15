"""
JARVIS Tool Middleware — pluggable pre/post hooks for the dispatch pipeline.

Middleware runs around every tool invocation, enabling cross-cutting concerns
without modifying handler code:

  - **Logging**      — structured log of every tool call with timing
  - **Metrics**      — prometheus-style counters, histograms per tool
  - **Audit trail**  — append-only log for compliance/debugging
  - **Rate limiting** — per-tool throttling beyond the global rate limit
  - **Transforms**   — input sanitization, output truncation
  - **Custom hooks** — anything you want to run before/after dispatch

Architecture:
    ┌─────────────────────────────────────────────────┐
    │  Dispatcher.dispatch(tool, args)                │
    │    ├─ middleware[0].before(ctx)                  │
    │    ├─ middleware[1].before(ctx)                  │
    │    ├─ ... handler execution ...                  │
    │    ├─ middleware[1].after(ctx)                   │
    │    ├─ middleware[0].after(ctx)                   │
    │    └─ return result                             │
    └─────────────────────────────────────────────────┘

Middleware is ordered: before hooks run first→last, after hooks run last→first
(like an onion / Express.js middleware stack).

Usage:
    from jarvis.tools.middleware import MiddlewareStack, LoggingMiddleware

    stack = MiddlewareStack()
    stack.add(LoggingMiddleware())
    stack.add(MetricsMiddleware())

    # In dispatcher:
    ctx = MiddlewareContext(tool="read_file", args={...})
    await stack.run_before(ctx)
    result = await handler(args, tool_ctx)
    ctx.result = result
    await stack.run_after(ctx)
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Middleware Context — passed through the hook chain
# ---------------------------------------------------------------------------


@dataclass
class MiddlewareContext:
    """Context object passed through the middleware chain.

    Populated by the dispatcher before calling hooks. Middleware can
    read/modify fields (e.g., sanitize args, annotate metadata).
    """

    # ── Set by dispatcher before hooks ──
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    origin: str = "unknown"
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    # ── Set after handler execution ──
    result: str = ""
    error: Exception | None = None
    duration_ms: float = 0.0
    status: str = "pending"  # ok, error, cached, circuit_open, blocked

    # ── Middleware-writable metadata bag ──
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Control flags ──
    skip_execution: bool = False  # If True, dispatcher skips the handler
    skip_reason: str = ""


# ---------------------------------------------------------------------------
# Abstract Middleware
# ---------------------------------------------------------------------------


class Middleware(ABC):
    """Base class for dispatch middleware.

    Subclass and override ``before`` and/or ``after`` to hook into
    the dispatch pipeline. Both are async and receive a MiddlewareContext.
    """

    @property
    def name(self) -> str:
        """Human-readable middleware name."""
        return self.__class__.__name__

    @property
    def order(self) -> int:
        """Execution order (lower = runs first in before, last in after)."""
        return 100

    async def before(self, ctx: MiddlewareContext) -> None:
        """Called before handler execution. Override to add pre-processing."""
        pass

    async def after(self, ctx: MiddlewareContext) -> None:
        """Called after handler execution. Override to add post-processing."""
        pass


# ---------------------------------------------------------------------------
# Middleware Stack
# ---------------------------------------------------------------------------


class MiddlewareStack:
    """Ordered collection of middleware that runs around every dispatch.

    Maintains insertion order, sorted by ``middleware.order``.
    Before hooks run in order (low→high), after hooks in reverse (high→low).
    """

    def __init__(self) -> None:
        self._middlewares: list[Middleware] = []

    def add(self, mw: Middleware) -> "MiddlewareStack":
        """Add a middleware to the stack. Returns self for chaining."""
        self._middlewares.append(mw)
        self._middlewares.sort(key=lambda m: m.order)
        return self

    def remove(self, name: str) -> bool:
        """Remove a middleware by name. Returns True if found."""
        before = len(self._middlewares)
        self._middlewares = [m for m in self._middlewares if m.name != name]
        return len(self._middlewares) < before

    @property
    def middlewares(self) -> list[Middleware]:
        """Return the ordered middleware list (read-only copy)."""
        return list(self._middlewares)

    def __len__(self) -> int:
        return len(self._middlewares)

    async def run_before(self, ctx: MiddlewareContext) -> None:
        """Run all before hooks in order. Stops if skip_execution is set."""
        for mw in self._middlewares:
            try:
                await mw.before(ctx)
                if ctx.skip_execution:
                    logger.debug(
                        "Middleware %s skipped execution: %s",
                        mw.name, ctx.skip_reason,
                    )
                    return
            except Exception as exc:
                logger.warning("Middleware %s.before failed: %s", mw.name, exc)

    async def run_after(self, ctx: MiddlewareContext) -> None:
        """Run all after hooks in reverse order."""
        for mw in reversed(self._middlewares):
            try:
                await mw.after(ctx)
            except Exception as exc:
                logger.warning("Middleware %s.after failed: %s", mw.name, exc)


# ═══════════════════════════════════════════════════════════════════════════
# Built-in Middleware Implementations
# ═══════════════════════════════════════════════════════════════════════════


class LoggingMiddleware(Middleware):
    """Structured logging for every tool invocation.

    Logs tool name, args summary, duration, status, and trace ID.
    """

    @property
    def order(self) -> int:
        return 10  # Runs first (outermost layer)

    async def before(self, ctx: MiddlewareContext) -> None:
        """Log the tool invocation start."""
        args_summary = {k: _truncate(str(v), 100) for k, v in ctx.args.items()}
        logger.info(
            "[%s] ▶ %s args=%s origin=%s",
            ctx.trace_id, ctx.tool, args_summary, ctx.origin,
        )

    async def after(self, ctx: MiddlewareContext) -> None:
        """Log the tool invocation result."""
        result_preview = _truncate(ctx.result, 200) if ctx.result else ""
        logger.info(
            "[%s] ◀ %s status=%s duration=%.0fms result=%s",
            ctx.trace_id, ctx.tool, ctx.status,
            ctx.duration_ms, result_preview,
        )


class MetricsMiddleware(Middleware):
    """In-memory metrics collector for tool invocations.

    Tracks per-tool: call count, error count, total duration,
    min/max/avg latency, and status distribution.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[str, int] = defaultdict(int)
        self._errors: dict[str, int] = defaultdict(int)
        self._durations: dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._status_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._total_calls: int = 0
        self._total_errors: int = 0

    @property
    def order(self) -> int:
        return 20

    async def after(self, ctx: MiddlewareContext) -> None:
        """Record metrics after execution (thread-safe)."""
        with self._lock:
            self._total_calls += 1
            self._calls[ctx.tool] += 1
            self._status_counts[ctx.tool][ctx.status] += 1

            if ctx.status in ("error", "timeout"):
                self._total_errors += 1
                self._errors[ctx.tool] += 1

            self._durations[ctx.tool].append(ctx.duration_ms)

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time metrics snapshot."""
        tools: dict[str, Any] = {}
        for tool in sorted(self._calls.keys()):
            durations = self._durations.get(tool, [])
            sorted_d = sorted(durations) if durations else [0]
            tools[tool] = {
                "calls": self._calls[tool],
                "errors": self._errors.get(tool, 0),
                "status_distribution": dict(self._status_counts.get(tool, {})),
                "latency_ms": {
                    "min": round(sorted_d[0], 1),
                    "max": round(sorted_d[-1], 1),
                    "avg": round(sum(sorted_d) / len(sorted_d), 1),
                    "p50": round(_percentile(sorted_d, 50), 1),
                    "p95": round(_percentile(sorted_d, 95), 1),
                    "p99": round(_percentile(sorted_d, 99), 1),
                },
            }
        return {
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "error_rate": round(
                self._total_errors / max(self._total_calls, 1) * 100, 2
            ),
            "tools": tools,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._calls.clear()
        self._errors.clear()
        self._durations.clear()
        self._status_counts.clear()
        self._total_calls = 0
        self._total_errors = 0


class AuditTrailMiddleware(Middleware):
    """Append-only audit trail for compliance and debugging.

    Records every tool invocation with timestamp, trace_id, tool,
    args hash, status, and duration. Stored in memory with a configurable
    max size (oldest entries are evicted).
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        self._entries: deque[dict[str, Any]] = deque(maxlen=max_entries)
        self._max_entries = max_entries

    @property
    def order(self) -> int:
        return 30

    async def after(self, ctx: MiddlewareContext) -> None:
        """Record an audit entry after execution."""
        entry = {
            "trace_id": ctx.trace_id,
            "timestamp": ctx.timestamp,
            "tool": ctx.tool,
            "origin": ctx.origin,
            "status": ctx.status,
            "duration_ms": round(ctx.duration_ms, 1),
            "error": str(ctx.error) if ctx.error else None,
            "metadata": dict(ctx.metadata) if ctx.metadata else None,
        }
        self._entries.append(entry)
        # Deque handles maxlen automatically — no O(n) list copy needed

    def recent(self, n: int = 50) -> list[dict[str, Any]]:
        """Return the N most recent audit entries."""
        return list(self._entries)[-n:]

    def search(self, tool: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        """Search audit entries by tool name and/or status."""
        results = self._entries
        if tool:
            results = [e for e in results if e["tool"] == tool]
        if status:
            results = [e for e in results if e["status"] == status]
        return results

    @property
    def size(self) -> int:
        """Return the number of audit entries."""
        return len(self._entries)


class InputSanitizerMiddleware(Middleware):
    """Sanitize tool arguments before execution.

    Strips leading/trailing whitespace from string args, removes
    null bytes, and truncates oversized arguments.
    """

    def __init__(self, max_arg_length: int = 100_000) -> None:
        self._max_arg_length = max_arg_length

    @property
    def order(self) -> int:
        return 5  # Runs before logging

    async def before(self, ctx: MiddlewareContext) -> None:
        """Sanitize input arguments."""
        sanitized = {}
        for key, value in ctx.args.items():
            if isinstance(value, str):
                # Remove null bytes
                value = value.replace("\x00", "")
                # Strip whitespace
                value = value.strip()
                # Truncate oversized args
                if len(value) > self._max_arg_length:
                    value = value[:self._max_arg_length] + "...[truncated]"
            sanitized[key] = value
        ctx.args = sanitized


class OutputTruncationMiddleware(Middleware):
    """Truncate oversized tool outputs to prevent context window bloat.

    Keeps the first and last portions of large outputs with a
    truncation marker in the middle.
    """

    def __init__(self, max_length: int = 50_000, keep_head: int = 25_000, keep_tail: int = 5_000) -> None:
        self._max_length = max_length
        self._keep_head = keep_head
        self._keep_tail = keep_tail

    @property
    def order(self) -> int:
        return 90  # Runs last in after hooks

    async def after(self, ctx: MiddlewareContext) -> None:
        """Truncate oversized results."""
        if ctx.result and len(ctx.result) > self._max_length:
            original_len = len(ctx.result)
            ctx.result = (
                ctx.result[:self._keep_head]
                + f"\n\n...[truncated {original_len - self._keep_head - self._keep_tail:,} chars]...\n\n"
                + ctx.result[-self._keep_tail:]
            )
            ctx.metadata["truncated"] = True
            ctx.metadata["original_length"] = original_len


class PerToolRateLimitMiddleware(Middleware):
    """Per-tool rate limiting with configurable limits.

    Different from the global rate limiter in the dispatcher —
    this allows per-tool throttling (e.g., web_search: 10/min,
    write_file: 30/min).
    """

    def __init__(self, limits: dict[str, int] | None = None, window_seconds: float = 60.0) -> None:
        self._limits = limits or {}
        self._window = window_seconds
        self._timestamps: dict[str, list[float]] = defaultdict(list)

    @property
    def order(self) -> int:
        return 15  # After logging, before execution

    async def before(self, ctx: MiddlewareContext) -> None:
        """Check per-tool rate limit."""
        limit = self._limits.get(ctx.tool)
        if limit is None:
            return

        now = time.time()
        cutoff = now - self._window
        # Prune old timestamps
        self._timestamps[ctx.tool] = [
            t for t in self._timestamps[ctx.tool] if t > cutoff
        ]
        if len(self._timestamps[ctx.tool]) >= limit:
            ctx.skip_execution = True
            ctx.skip_reason = f"Rate limited: {ctx.tool} ({limit}/{self._window}s)"
            ctx.result = f"⚠️ Rate limited: {ctx.tool} exceeded {limit} calls per {self._window}s"
            ctx.status = "rate_limited"
            return

        self._timestamps[ctx.tool].append(now)

    def set_limit(self, tool: str, max_calls: int) -> None:
        """Set or update the rate limit for a specific tool."""
        self._limits[tool] = max_calls

    def remove_limit(self, tool: str) -> None:
        """Remove rate limit for a specific tool."""
        self._limits.pop(tool, None)
        self._timestamps.pop(tool, None)

    def cleanup_stale(self) -> int:
        """Remove timestamp entries for tools with no recent activity.

        Call periodically (e.g., every 5 minutes) in long-running daemons
        to prevent unbounded memory growth from idle tools.
        Returns the number of tools cleaned up.
        """
        now = time.time()
        cutoff = now - self._window * 2  # 2x window = definitely stale
        stale_tools = [
            tool for tool, timestamps in self._timestamps.items()
            if not timestamps or max(timestamps) < cutoff
        ]
        for tool in stale_tools:
            del self._timestamps[tool]
        return len(stale_tools)


# ---------------------------------------------------------------------------
# Default middleware stack factory
# ---------------------------------------------------------------------------


def create_default_stack() -> MiddlewareStack:
    """Create the default middleware stack with standard hooks.

    Returns a stack with: InputSanitizer → Logging → Metrics → AuditTrail → OutputTruncation
    """
    stack = MiddlewareStack()
    stack.add(InputSanitizerMiddleware())
    stack.add(LoggingMiddleware())
    stack.add(MetricsMiddleware())
    stack.add(AuditTrailMiddleware())
    stack.add(OutputTruncationMiddleware())
    return stack


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len characters with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Calculate percentile from pre-sorted data."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])
