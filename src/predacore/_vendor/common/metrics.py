"""
Prometheus Observability — Metrics, Tracing, and Health.

Provides standardized metrics for every Prometheus service using
prometheus-client. All services expose /metrics on their health port.

Metrics naming convention:
    prometheus_{service}_{metric}_{unit}

Usage:
    from src.common.metrics import TOOL_CALLS, LLM_LATENCY, track_latency

    TOOL_CALLS.labels(tool="shell_exec", status="success").inc()

    with track_latency(LLM_LATENCY, provider="gemini"):
        response = await llm.generate(messages)
"""
from __future__ import annotations

import contextlib
import logging
import time
import uuid
from collections.abc import Generator
from contextvars import ContextVar

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

logger = logging.getLogger(__name__)

# ── Global registry (use default for now, switchable for testing) ────

REGISTRY = CollectorRegistry(auto_describe=True)

# We also register with the default registry so /metrics just works
# with the default prometheus_client WSGI/ASGI handler.


# ── Trace ID propagation ────────────────────────────────────────────

_trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def new_trace_id() -> str:
    """Generate and set a new trace ID for the current async context."""
    tid = uuid.uuid4().hex[:16]
    _trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    """Get the current trace ID."""
    return _trace_id.get() or new_trace_id()


def set_trace_id(tid: str) -> None:
    """Set a specific trace ID (e.g., from an incoming request header)."""
    _trace_id.set(tid)


# ── System Info ──────────────────────────────────────────────────────

SYSTEM_INFO = Info(
    "prometheus_system",
    "Prometheus agent system information",
)

# ── Request / Message Counters ───────────────────────────────────────

REQUESTS_TOTAL = Counter(
    "prometheus_requests_total",
    "Total requests processed",
    ["channel", "status"],  # channel=telegram|discord|cli|api, status=success|error
)

MESSAGES_RECEIVED = Counter(
    "prometheus_messages_received_total",
    "Total messages received from users",
    ["channel"],
)

MESSAGES_SENT = Counter(
    "prometheus_messages_sent_total",
    "Total messages sent to users",
    ["channel"],
)

# ── Tool Execution Metrics ───────────────────────────────────────────

TOOL_CALLS = Counter(
    "prometheus_tool_calls_total",
    "Total tool executions",
    ["tool", "status"],  # tool=shell_exec|read_file|..., status=success|error|timeout
)

TOOL_LATENCY = Histogram(
    "prometheus_tool_latency_seconds",
    "Tool execution latency in seconds",
    ["tool"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

TOOL_CONFIRMATIONS = Counter(
    "prometheus_tool_confirmations_total",
    "Tool calls that required user confirmation",
    ["tool", "decision"],  # decision=approved|rejected
)

# ── LLM Provider Metrics ────────────────────────────────────────────

LLM_REQUESTS = Counter(
    "prometheus_llm_requests_total",
    "Total LLM API calls",
    [
        "provider",
        "status",
    ],  # provider=gemini|openai|openrouter|ollama, status=success|error|rate_limited
)

LLM_LATENCY = Histogram(
    "prometheus_llm_latency_seconds",
    "LLM response latency in seconds",
    ["provider"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0),
)

LLM_TOKENS = Counter(
    "prometheus_llm_tokens_total",
    "Total LLM tokens consumed",
    ["provider", "direction"],  # direction=input|output
)

LLM_FAILOVERS = Counter(
    "prometheus_llm_failovers_total",
    "Times primary LLM failed and switched to backup",
    ["from_provider", "to_provider"],
)

# ── Memory Metrics ───────────────────────────────────────────────────

MEMORY_COUNT = Gauge(
    "prometheus_memories_total",
    "Total memories stored",
    ["user", "type"],
)

MEMORY_OPS = Counter(
    "prometheus_memory_operations_total",
    "Memory operations (store, recall, delete)",
    ["operation", "status"],
)

# ── Session Metrics ──────────────────────────────────────────────────

ACTIVE_SESSIONS = Gauge(
    "prometheus_active_sessions",
    "Currently active conversation sessions",
)

SESSION_DURATION = Histogram(
    "prometheus_session_duration_seconds",
    "Session duration from first to last message",
    buckets=(60, 300, 900, 1800, 3600, 7200, 14400),
)

# ── Channel Health ───────────────────────────────────────────────────

CHANNEL_STATUS = Gauge(
    "prometheus_channel_status",
    "Channel connection status (1=connected, 0=disconnected)",
    ["channel"],
)

CHANNEL_ERRORS = Counter(
    "prometheus_channel_errors_total",
    "Channel-specific errors",
    ["channel", "error_type"],
)

# ── System Resources ────────────────────────────────────────────────

DAEMON_UPTIME = Gauge(
    "prometheus_daemon_uptime_seconds",
    "Seconds since daemon started",
)

CRON_JOBS_EXECUTED = Counter(
    "prometheus_cron_jobs_executed_total",
    "Cron jobs executed",
    ["job_name", "status"],
)

# ── EGM (Ethical Governance) ─────────────────────────────────────────

EGM_EVALUATIONS = Counter(
    "prometheus_egm_evaluations_total",
    "EGM rule evaluations",
    ["decision"],  # decision=allow|block|flag
)


# ── Helpers ──────────────────────────────────────────────────────────


@contextlib.contextmanager
def track_latency(
    histogram: Histogram,
    labels: dict | None = None,
    **label_kwargs: str,
) -> Generator[None, None, None]:
    """
    Context manager to track operation latency.

    Usage:
        with track_latency(LLM_LATENCY, provider="gemini"):
            response = await llm.generate(messages)
    """
    all_labels = {**(labels or {}), **label_kwargs}
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if all_labels:
            histogram.labels(**all_labels).observe(elapsed)
        else:
            histogram.observe(elapsed)


def get_metrics_text() -> str:
    """Generate Prometheus metrics text for /metrics endpoint."""
    return generate_latest().decode("utf-8")


def get_metrics_content_type() -> str:
    """Get the Content-Type header for metrics responses."""
    return CONTENT_TYPE_LATEST
