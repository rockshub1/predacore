"""
PredaCore Tool Dispatcher — finds a handler and executes it with timeout/metrics.

Extracted from ToolExecutor.execute() (Phase 6.1). This module owns the
dispatch loop: rate-limit check → middleware.before → confirmation →
handler lookup → asyncio.wait_for → middleware.after → metrics →
output sanitization.

Phase 6.4: Middleware integration — pluggable pre/post hooks for logging,
metrics, audit trail, input sanitization, output truncation.

Includes adaptive timeout system: static registry values act as ceilings,
while observed P95 latencies dynamically tighten timeouts for faster
failure detection.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import weakref
from collections import deque
from collections.abc import Callable
from typing import Any

from predacore.services.rate_limiter import InMemoryBackend as _RateLimitBackend
from predacore.auth.security import redact_secrets as _redact
from predacore.auth.security import sanitize_tool_output as _sanitize_output
from .enums import ToolStatus, WRITE_TOOLS
from .handlers import HANDLER_MAP as _HANDLER_MAP
from .handlers import ToolContext as _ToolContext
from .handlers._context import ToolError as _ToolError
from .middleware import MiddlewareContext, MiddlewareStack
from .registry import build_full_registry as _build_registry
from .resilience import ExecutionHistory, ToolCircuitBreaker, ToolResultCache
from .trust_policy import TrustPolicyEvaluator

logger = logging.getLogger(__name__)

# Tools whose latency distribution is bimodal (fast probes AND long-running work)
# and that accept an explicit `timeout_seconds` argument. For these tools we
# skip the adaptive tracker entirely: P95 over a mixed `ls`/`npm install`
# workload is meaningless, and the shell handler enforces the real timeout
# from the `timeout_seconds` arg. The dispatcher just provides an outer safety net.
_VARIABLE_LATENCY_TOOLS: frozenset[str] = frozenset({
    "run_command",
    "python_exec",
    "execute_code",
})

# ── Ethical compliance (lightweight keyword guard) ────────────────────

_FORBIDDEN_KEYWORDS: frozenset[str] = frozenset({
    "delete_user_data",
    "disable_safety",
    "bypass_auth",
    "drop_table",
    "truncate_table",
    "format_disk",
})


def _check_ethical_compliance(
    tool_name: str, args: dict[str, Any], trust_level: str,
) -> str | None:
    """Pre-execution ethical keyword check.

    Returns an error message string if the call should be blocked,
    or *None* to allow execution.
    """
    if trust_level == "yolo":
        return None
    args_repr = str(args).lower()
    matched = [kw for kw in _FORBIDDEN_KEYWORDS if kw in args_repr]
    if not matched:
        return None
    if trust_level == "paranoid":
        return (
            f"[Blocked by ethical compliance] Tool '{tool_name}' "
            f"contains forbidden keywords: {', '.join(sorted(matched))}"
        )
    # trust_level == "normal" or anything else — warn only
    logger.warning(
        "Ethical compliance warning for tool '%s': matched keywords %s",
        tool_name, matched,
    )
    return None


# ── Constants ─────────────────────────────────────────────────────────
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3    # Failures before circuit opens
CIRCUIT_BREAKER_COOLDOWN_SECONDS = 60.0  # Seconds before circuit half-opens
RESULT_CACHE_MAX_ENTRIES = 200           # LRU cache size for tool results
EXECUTION_HISTORY_MAX_ENTRIES = 500      # Ring buffer size for execution history
LOG_ARGS_TRUNCATE_LENGTH = 200           # Max chars of tool args in log messages

# ── Gemini CLI → PredaCore tool name mapping ─────────────────────────────
_TOOL_ALIAS_MAP: dict[str, str] = {
    "list_directory": "list_directory",
    "grep_search": "semantic_search",
    "read_file": "read_file",
    "run_in_terminal": "run_command",
    "edit_file": "write_file",
    "file_search": "semantic_search",
    "run_shell_command": "run_command",
}

# Optional Prometheus metrics
try:
    from prometheus_client import Counter, Histogram

    try:
        _TOOL_REQ = Counter(
            "predacore_tool_requests_total", "Tool call count", ["tool_id", "status"]
        )
    except ValueError:
        _TOOL_REQ = None
    try:
        _TOOL_LAT = Histogram(
            "predacore_tool_latency_seconds", "Tool call latency", ["tool_id"]
        )
    except ValueError:
        _TOOL_LAT = None
except ImportError:
    _TOOL_REQ = None
    _TOOL_LAT = None


# ---------------------------------------------------------------------------
# Adaptive Timeout Tracker
# ---------------------------------------------------------------------------


class AdaptiveTimeoutTracker:
    """Tracks per-tool latencies and computes adaptive timeouts.

    Formula:
        effective = min(static_ceiling, max(min_floor, P95 x multiplier))

    Until ``min_samples`` observations are recorded for a tool, the static
    ceiling is used as-is.
    """

    def __init__(
        self,
        *,
        window_size: int = 20,
        multiplier: float = 2.0,
        min_floor: float = 5.0,
        min_samples: int = 3,
    ) -> None:
        self._window_size = window_size
        self._multiplier = multiplier
        self._min_floor = min_floor
        self._min_samples = min_samples
        self._latencies: dict[str, deque[float]] = {}

    def record(self, tool_name: str, latency: float) -> None:
        """Record an observed latency for a tool."""
        if tool_name not in self._latencies:
            self._latencies[tool_name] = deque(maxlen=self._window_size)
        self._latencies[tool_name].append(latency)

    def get_timeout(self, tool_name: str, static_ceiling: float) -> float:
        """Compute adaptive timeout, capped by the static ceiling."""
        samples = self._latencies.get(tool_name)
        if not samples or len(samples) < self._min_samples:
            return static_ceiling

        sorted_latencies = sorted(samples)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p95 = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]

        adaptive = max(self._min_floor, p95 * self._multiplier)
        effective = min(static_ceiling, adaptive)

        logger.debug(
            "Adaptive timeout for '%s': P95=%.1fs → adaptive=%.1fs → "
            "effective=%.1fs (ceiling=%ds)",
            tool_name, p95, adaptive, effective, int(static_ceiling),
        )
        return effective


# ---------------------------------------------------------------------------
# ToolDispatcher
# ---------------------------------------------------------------------------


class ToolDispatcher:
    """Takes a tool name + args, finds the handler, executes with timeout.

    Integrates middleware stack for cross-cutting concerns (logging, metrics,
    audit trail, input/output transforms).
    """

    def __init__(
        self,
        trust_evaluator: TrustPolicyEvaluator,
        tool_ctx: _ToolContext,
        *,
        rate_max: int | None = None,
        tool_timeout: int | None = None,
        middleware: MiddlewareStack | None = None,
    ) -> None:
        self._trust = trust_evaluator
        self._tool_ctx = tool_ctx
        self._tool_ctx._dispatcher_ref = weakref.ref(self)  # type: ignore[attr-defined]
        self._rate_backend = _RateLimitBackend()
        self._rate_max = rate_max or int(
            os.getenv("PREDACORE_TOOL_RATE_LIMIT", "120")
        )
        self._timeout = tool_timeout or int(
            os.getenv("PREDACORE_TOOL_TIMEOUT_SECONDS", "600")
        )

        # Per-tool timeout from registry (static ceilings)
        self._tool_timeouts: dict[str, int] = {}
        try:
            _registry = _build_registry(
                include_openclaw=True, include_marketplace=True
            )
            for tool_def in _registry.list_all():
                if tool_def.timeout_default:
                    self._tool_timeouts[tool_def.name] = tool_def.timeout_default
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("Tool registry not available for timeout config: %s", exc)

        # Adaptive timeout tracker
        self._adaptive = AdaptiveTimeoutTracker()

        # Circuit breaker
        self._circuit_breaker = ToolCircuitBreaker(
            failure_threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            cooldown_seconds=CIRCUIT_BREAKER_COOLDOWN_SECONDS,
        )

        # Result cache (LRU + TTL)
        self._result_cache = ToolResultCache(max_entries=RESULT_CACHE_MAX_ENTRIES)

        # Execution history (ring buffer)
        self._execution_history = ExecutionHistory(max_entries=EXECUTION_HISTORY_MAX_ENTRIES)

        # Middleware stack (optional — pluggable pre/post hooks)
        self._middleware = middleware or MiddlewareStack()

    # -- Public properties -------------------------------------------------

    @property
    def tool_ctx(self) -> _ToolContext:
        """Return the shared ToolContext instance."""
        return self._tool_ctx

    @property
    def circuit_breaker(self) -> ToolCircuitBreaker:
        """Return the circuit breaker instance."""
        return self._circuit_breaker

    @property
    def result_cache(self) -> ToolResultCache:
        """Return the result cache instance."""
        return self._result_cache

    @property
    def execution_history(self) -> ExecutionHistory:
        """Return the execution history instance."""
        return self._execution_history

    @property
    def middleware(self) -> MiddlewareStack:
        """Return the middleware stack."""
        return self._middleware

    # -- Dispatch pipeline -------------------------------------------------

    async def dispatch(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        confirm_fn: Callable | None = None,
        blocked_tools: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        origin: str = "user",
    ) -> str:
        """Execute a tool call and return the result as a string.

        Full pipeline:
        1. Alias normalization
        2. Rate-limit check
        3. Circuit breaker check
        4. Cache check
        5. Middleware before hooks
        6. Confirmation check (trust policy)
        7. Blocked / allowed list enforcement
        8. Handler lookup + adaptive timeout
        9. Middleware after hooks
        10. ToolError handling
        11. Latency recording
        12. Metrics recording
        13. Output sanitization
        """
        # 1. Alias normalization
        if tool_name in _TOOL_ALIAS_MAP:
            original = tool_name
            tool_name = _TOOL_ALIAS_MAP[tool_name]
            logger.debug("Alias resolved: %s → %s (origin=%s)", original, tool_name, origin)

        logger.info(
            "Executing tool: %s(%s) [origin=%s]",
            tool_name,
            json.dumps(arguments, default=str)[:LOG_ARGS_TRUNCATE_LENGTH],
            origin,
        )

        # ── Create middleware context ─────────────────────────────────
        mw_ctx = MiddlewareContext(
            tool=tool_name,
            args=dict(arguments),  # copy so middleware can mutate safely
            origin=origin,
        )

        # 2. Rate limit
        rl = self._rate_backend.sliding_window_check(
            "tool_calls", self._rate_max, 60,
        )
        if not rl.allowed:
            logger.warning("Tool rate limit hit (%d/%d per min)", self._rate_max, self._rate_max)
            mw_ctx.status = "rate_limited"
            mw_ctx.result = f"[Rate limited — too many tool calls. Retry in {int(rl.retry_after)}s]"
            await self._middleware.run_after(mw_ctx)
            return mw_ctx.result

        # 3. Circuit breaker check
        if self._circuit_breaker.is_open(tool_name):
            cb_state = self._circuit_breaker.state(tool_name)
            mw_ctx.status = ToolStatus.CIRCUIT_OPEN
            mw_ctx.result = (
                f"[Tool '{tool_name}' temporarily unavailable — "
                f"circuit breaker {cb_state.value} after repeated failures. "
                f"Will auto-retry in {int(self._circuit_breaker.cooldown_seconds)}s]"
            )
            self._execution_history.record(
                tool_name, arguments, mw_ctx.result,
                ToolStatus.CIRCUIT_OPEN, 0, origin,
            )
            await self._middleware.run_after(mw_ctx)
            return mw_ctx.result

        # 4. Cache check
        cached_result = self._result_cache.get(tool_name, arguments)
        if cached_result is not None:
            mw_ctx.status = ToolStatus.CACHED
            mw_ctx.result = cached_result
            self._execution_history.record(
                tool_name, arguments, cached_result,
                ToolStatus.CACHED, 0, origin,
            )
            await self._middleware.run_after(mw_ctx)
            return cached_result

        # 5. Middleware before hooks
        await self._middleware.run_before(mw_ctx)
        if mw_ctx.skip_execution:
            mw_ctx.duration_ms = (time.time() - mw_ctx.timestamp) * 1000
            await self._middleware.run_after(mw_ctx)
            return mw_ctx.result or mw_ctx.skip_reason

        # Use potentially sanitized args from middleware
        arguments = mw_ctx.args

        # 6. Confirmation
        if self._trust.requires_confirmation(tool_name):
            prev = self._trust.check_previous_approval(tool_name, arguments)
            if prev is True:
                logger.debug("Tool '%s' auto-approved from history", tool_name)
            elif prev is False:
                logger.info("Tool '%s' auto-denied from history", tool_name)
                return f"[Tool '{tool_name}' was previously denied]"
            elif confirm_fn:
                ctx = self._trust.assess_risk(tool_name, arguments)
                try:
                    approved = await confirm_fn(tool_name, arguments, ctx)
                except TypeError:
                    approved = await confirm_fn(tool_name, arguments)
                self._trust.record_approval(tool_name, arguments, approved)
                if not approved:
                    return f"[Tool '{tool_name}' was blocked by user]"

        # 7. Blocked / allowed enforcement
        block_reason = self._trust.is_blocked(
            tool_name,
            blocked_tools=blocked_tools,
            allowed_tools=allowed_tools,
        )
        if block_reason:
            mw_ctx.status = "blocked"
            mw_ctx.result = block_reason
            await self._middleware.run_after(mw_ctx)
            return block_reason

        # 7b. Ethical compliance check (keyword guard)
        trust_level = getattr(self._trust, "trust_level", "normal")
        ethical_block = _check_ethical_compliance(tool_name, arguments, trust_level)
        if ethical_block is not None:
            mw_ctx.status = "blocked"
            mw_ctx.result = ethical_block
            await self._middleware.run_after(mw_ctx)
            return ethical_block

        # 8. Handler lookup + execution
        _outer_timeout = float(
            os.getenv("PREDACORE_TOOL_TIMEOUT_SECONDS", str(self._timeout))
        )
        t0 = time.time()
        try:
            handler_fn = _HANDLER_MAP.get(tool_name)
            if handler_fn is not None:
                static_ceiling = float(
                    self._tool_timeouts.get(tool_name, self._timeout)
                )
                # Variable-latency tools (run_command, python_exec, execute_code):
                # skip the adaptive tracker and honor the explicit `timeout_seconds`
                # argument. P95 over mixed shell workloads is meaningless, and the
                # handler already enforces the real deadline internally.
                if tool_name in _VARIABLE_LATENCY_TOOLS:
                    explicit = arguments.get("timeout_seconds")
                    try:
                        explicit_f = float(explicit) if explicit is not None else 0.0
                    except (TypeError, ValueError):
                        explicit_f = 0.0
                    if explicit_f > 0:
                        effective_timeout = min(explicit_f + 5.0, 86400.0)
                    else:
                        effective_timeout = max(static_ceiling, _outer_timeout)
                else:
                    effective_timeout = self._adaptive.get_timeout(
                        tool_name, static_ceiling
                    )
                    effective_timeout = min(
                        effective_timeout, max(1.0, _outer_timeout - 1.0)
                    )
                try:
                    result = await asyncio.wait_for(
                        handler_fn(arguments, self._tool_ctx),
                        timeout=effective_timeout,
                    )
                except asyncio.TimeoutError:
                    elapsed = time.time() - t0
                    # Do NOT record timeout elapsed as a latency sample — it would
                    # create a self-reinforcing feedback loop that keeps strangling
                    # slow operations. Only record successes and non-timeout errors.
                    if tool_name not in _VARIABLE_LATENCY_TOOLS:
                        pass  # adaptive tracker intentionally skipped on timeout
                    self._circuit_breaker.record_failure(tool_name)
                    logger.warning(
                        "Tool '%s' timed out after %.1fs (adaptive=%.1fs, ceiling=%ds)",
                        tool_name, elapsed, effective_timeout, int(static_ceiling),
                    )
                    if _TOOL_REQ:
                        _TOOL_REQ.labels(tool_id=tool_name, status=ToolStatus.TIMEOUT).inc()
                    if _TOOL_LAT:
                        _TOOL_LAT.labels(tool_id=tool_name).observe(elapsed)
                    timeout_result = f"[Tool '{tool_name}' timed out after {effective_timeout:.0f}s]"
                    self._execution_history.record(
                        tool_name, arguments, timeout_result,
                        ToolStatus.TIMEOUT, elapsed * 1000, origin,
                    )
                    # Run after hooks for timeout
                    mw_ctx.status = ToolStatus.TIMEOUT
                    mw_ctx.duration_ms = elapsed * 1000
                    mw_ctx.result = timeout_result
                    await self._middleware.run_after(mw_ctx)
                    return timeout_result
            else:
                if _TOOL_REQ:
                    _TOOL_REQ.labels(tool_id=tool_name, status=ToolStatus.UNKNOWN_TOOL).inc()
                mw_ctx.status = ToolStatus.UNKNOWN_TOOL
                mw_ctx.result = f"[Unknown tool: {tool_name}]"
                await self._middleware.run_after(mw_ctx)
                return mw_ctx.result

            # ── Success path ─────────────────────────────────────────
            elapsed = time.time() - t0
            if tool_name not in _VARIABLE_LATENCY_TOOLS:
                self._adaptive.record(tool_name, elapsed)
            self._circuit_breaker.record_success(tool_name)

            # Invalidate cache on writes BEFORE caching (don't cache write results)
            if tool_name in WRITE_TOOLS:
                self._result_cache.invalidate_on_write(tool_name, arguments)
            elif isinstance(result, str):
                # Only cache read-only tool results
                self._result_cache.put(tool_name, arguments, result)

            # Prometheus metrics
            if _TOOL_REQ:
                _TOOL_REQ.labels(tool_id=tool_name, status=ToolStatus.OK).inc()
            if _TOOL_LAT:
                _TOOL_LAT.labels(tool_id=tool_name).observe(elapsed)

            # Execution history
            elapsed_ms = elapsed * 1000
            self._execution_history.record(
                tool_name, arguments,
                result if isinstance(result, str) else str(result),
                ToolStatus.OK, elapsed_ms, origin,
            )

            # Sanitize
            if isinstance(result, str):
                result = _sanitize_output(result)
                result = _redact(result)

            # 9. Middleware after hooks (success)
            mw_ctx.status = ToolStatus.OK
            mw_ctx.duration_ms = elapsed_ms
            mw_ctx.result = result if isinstance(result, str) else str(result)
            await self._middleware.run_after(mw_ctx)

            # Return potentially transformed result from middleware
            return mw_ctx.result

        # ── ToolError — structured handler errors ────────────────────
        except _ToolError as te:
            elapsed = time.time() - t0
            if tool_name not in _VARIABLE_LATENCY_TOOLS:
                self._adaptive.record(tool_name, elapsed)

            from .handlers._context import ToolErrorKind
            _user_errors = {
                ToolErrorKind.MISSING_PARAM,
                ToolErrorKind.INVALID_PARAM,
                ToolErrorKind.NOT_FOUND,
                ToolErrorKind.BLOCKED,
                ToolErrorKind.PERMISSION,
            }
            if te.kind not in _user_errors:
                self._circuit_breaker.record_failure(tool_name)
            else:
                self._circuit_breaker.record_success(tool_name)

            formatted = te.format()
            logger.info("Tool %s raised ToolError (%s): %s", tool_name, te.kind.value, te)
            if _TOOL_REQ:
                _TOOL_REQ.labels(tool_id=tool_name, status=ToolStatus.ERROR).inc()
            if _TOOL_LAT:
                _TOOL_LAT.labels(tool_id=tool_name).observe(elapsed)
            self._execution_history.record(
                tool_name, arguments, formatted,
                te.kind.value, elapsed * 1000, origin,
            )
            # Middleware after hooks (error)
            mw_ctx.status = ToolStatus.ERROR
            mw_ctx.error = te
            mw_ctx.duration_ms = elapsed * 1000
            mw_ctx.result = formatted
            await self._middleware.run_after(mw_ctx)
            return mw_ctx.result

        # ── Generic exceptions ───────────────────────────────────────
        except Exception as e:
            elapsed = time.time() - t0
            if tool_name not in _VARIABLE_LATENCY_TOOLS:
                self._adaptive.record(tool_name, elapsed)
            self._circuit_breaker.record_failure(tool_name)
            logger.error("Tool %s failed: %s", tool_name, e, exc_info=True)
            if _TOOL_REQ:
                _TOOL_REQ.labels(tool_id=tool_name, status=ToolStatus.ERROR).inc()
            if _TOOL_LAT:
                _TOOL_LAT.labels(tool_id=tool_name).observe(elapsed)
            error_result = f"[Tool error: {e}]"
            self._execution_history.record(
                tool_name, arguments, error_result,
                ToolStatus.ERROR, elapsed * 1000, origin,
            )
            # Middleware after hooks (error)
            mw_ctx.status = ToolStatus.ERROR
            mw_ctx.error = e
            mw_ctx.duration_ms = elapsed * 1000
            mw_ctx.result = error_result
            await self._middleware.run_after(mw_ctx)
            return mw_ctx.result
