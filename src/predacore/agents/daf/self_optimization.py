"""
DAF Self-Optimization Module

Implements performance monitoring, adaptive configuration policy evaluation, and
agent lifecycle optimization actions for the Dynamic Agent Fabric.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Optional

from prometheus_client import Counter, Gauge, Histogram

try:
    import psutil
except ImportError:
    psutil = None  # mocked in tests

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_network_load(value: float) -> float:
    """
    Normalize network utilization to a pseudo-percentage.
    We retain backward compatibility with existing tests by leaving <=100 values
    untouched and scaling larger byte/sec values.
    """
    if value <= 0:
        return 0.0
    if value <= 100:
        return float(value)
    # Assume 100 MB/s ~= 100% utilization for a conservative heuristic.
    return min(100.0, (value / (100.0 * 1024.0 * 1024.0)) * 100.0)


@dataclass
class PerformanceMetrics:
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_latency: float = 0.0
    request_rate: float = 0.0
    error_rate: float = 0.0
    captured_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "network_latency": self.network_latency,
            "request_rate": self.request_rate,
            "error_rate": self.error_rate,
            "captured_at": self.captured_at,
        }


@dataclass
class OptimizationPolicy:
    resource_weights: dict[str, float] = field(
        default_factory=lambda: {"cpu": 0.4, "memory": 0.3, "network": 0.3}
    )
    optimization_window_seconds: int = 15 * 60
    target_error_rate: float = 0.05
    high_load_score: float = 0.75
    rollout_step: float = 0.25


class ResourceMonitor:
    """Monitors system resource usage (CPU, memory, network)."""

    def __init__(
        self,
        cpu_threshold: float = 80,
        memory_threshold: float = 85,
        network_threshold: float = 90,
    ):
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.network_threshold = network_threshold
        self._last_net_io = None
        self._last_net_time = None

    async def get_cpu_usage(self) -> float:
        if psutil is None:
            return 0.0
        return await asyncio.to_thread(psutil.cpu_percent, interval=0.1)

    async def get_memory_usage(self) -> float:
        if psutil is None:
            return 0.0
        mem = await asyncio.to_thread(psutil.virtual_memory)
        return mem.percent

    async def get_network_usage(self) -> float:
        """Network throughput in bytes/sec."""
        if psutil is None:
            return 0.0
        counters = await asyncio.to_thread(psutil.net_io_counters)
        current_total = counters.bytes_sent + counters.bytes_recv
        current_time = time.monotonic()

        if self._last_net_io is None:
            self._last_net_io = current_total
            self._last_net_time = current_time
            return 0.0

        elapsed = current_time - self._last_net_time
        if elapsed <= 0:
            return 0.0

        rate = (current_total - self._last_net_io) / elapsed
        self._last_net_io = current_total
        self._last_net_time = current_time
        return rate

    async def get_resource_usage(self) -> dict[str, float]:
        return {
            "cpu": await self.get_cpu_usage(),
            "memory": await self.get_memory_usage(),
            "network": await self.get_network_usage(),
        }


class PerformanceMonitor:
    """
    Collects runtime metrics and keeps a bounded history buffer for trend checks.
    """

    def __init__(
        self,
        resource_monitor: ResourceMonitor,
        request_rate_provider: Callable[[], float | Awaitable[float]] | None = None,
        error_rate_provider: Callable[[], float | Awaitable[float]] | None = None,
        history_size: int = 180,
    ):
        self._resource_monitor = resource_monitor
        self._request_rate_provider = request_rate_provider
        self._error_rate_provider = error_rate_provider
        self._history: deque[PerformanceMetrics] = deque(maxlen=max(10, history_size))

    async def _maybe_call(
        self, provider: Callable[[], float | Awaitable[float]]
    ) -> float:
        value = provider()
        if inspect.isawaitable(value):
            value = await value
        return float(value or 0.0)

    async def collect(
        self, resource_usage: dict[str, float] | None = None
    ) -> PerformanceMetrics:
        usage = resource_usage or await self._resource_monitor.get_resource_usage()
        request_rate = (
            await self._maybe_call(self._request_rate_provider)
            if self._request_rate_provider
            else 0.0
        )
        error_rate = (
            await self._maybe_call(self._error_rate_provider)
            if self._error_rate_provider
            else 0.0
        )
        metrics = PerformanceMetrics(
            cpu_usage=float(usage.get("cpu", 0.0)),
            memory_usage=float(usage.get("memory", 0.0)),
            network_latency=float(usage.get("network", 0.0)),
            request_rate=request_rate,
            error_rate=error_rate,
        )
        self._history.append(metrics)
        return metrics

    def detect_anomalies(
        self,
        metrics: PerformanceMetrics,
        *,
        cpu_threshold: float,
        memory_threshold: float,
        network_threshold: float,
    ) -> list[str]:
        findings: list[str] = []

        def _fmt(value: float) -> str:
            return f"{float(value):g}"

        if metrics.cpu_usage > cpu_threshold:
            findings.append(
                f"cpu={_fmt(metrics.cpu_usage)}% (threshold:{_fmt(cpu_threshold)}%)"
            )
        if metrics.memory_usage > memory_threshold:
            findings.append(
                f"memory={_fmt(metrics.memory_usage)}% (threshold:{_fmt(memory_threshold)}%)"
            )
        network_pct = _normalize_network_load(metrics.network_latency)
        if network_pct > network_threshold:
            findings.append(
                f"network={_fmt(network_pct)}% (threshold:{_fmt(network_threshold)}%)"
            )
        if metrics.error_rate > 0.15:
            findings.append(f"error_rate={metrics.error_rate:.2%} (threshold:15.00%)")
        return findings

    @property
    def history(self) -> list[PerformanceMetrics]:
        return list(self._history)


class AdaptiveConfiguration:
    """Evaluates optimization policies and proposes actions."""

    def __init__(self, policy: OptimizationPolicy | None = None):
        self.policy = policy or OptimizationPolicy()
        self._rollout = 0.0

    def evaluate(self, metrics: PerformanceMetrics, score: float) -> dict[str, Any]:
        actions: list[str] = []
        reasons: list[str] = []

        if score >= self.policy.high_load_score:
            actions.append("scale_agents")
            reasons.append(
                f"score={score:.2f} above high_load={self.policy.high_load_score:.2f}"
            )

        if metrics.error_rate > self.policy.target_error_rate:
            actions.extend(["throttle_noncritical", "increase_retries"])
            reasons.append(
                f"error_rate={metrics.error_rate:.2%} above slo={self.policy.target_error_rate:.2%}"
            )

        if (
            metrics.request_rate > 100.0
            and metrics.error_rate <= self.policy.target_error_rate
        ):
            actions.append("warm_agent_pool")
            reasons.append("request_rate high with healthy error budget")

        if (
            metrics.cpu_usage < 35.0
            and metrics.memory_usage < 40.0
            and metrics.request_rate < 5.0
        ):
            actions.append("scale_down_agents")
            reasons.append("sustained low usage suggests over-provisioning")

        if actions:
            self._rollout = min(
                1.0,
                max(self.policy.rollout_step, self._rollout + self.policy.rollout_step),
            )
        else:
            self._rollout = max(0.0, self._rollout - self.policy.rollout_step)

        deduped_actions = list(dict.fromkeys(actions))
        return {
            "actions": deduped_actions,
            "reasons": reasons,
            "rollout_fraction": self._rollout,
        }


class AgentLifecycleManager:
    """Executes optimization actions against an orchestrator if available."""

    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator

    async def apply_actions(self, actions: list[str]) -> dict[str, str]:
        results: dict[str, str] = {}
        if self.orchestrator is None:
            return {action: "skipped:no_orchestrator" for action in actions}

        for action in actions:
            action_method = getattr(self.orchestrator, action, None)
            if action_method is None:
                results[action] = "skipped:not_supported"
                continue
            try:
                outcome = action_method()
                if inspect.isawaitable(outcome):
                    await outcome
                results[action] = "applied"
            except Exception as exc:
                logger.warning("Optimization action '%s' failed: %s", action, exc)
                results[action] = f"error:{exc}"
        return results


class OptimizationStrategy:
    """Calculates optimization score and baseline action hints from resources."""

    def __init__(
        self,
        optimization_interval: int = 60,
        resource_weights: dict[str, float] | None = None,
    ):
        self.optimization_interval = optimization_interval
        self.resource_weights = resource_weights or {
            "cpu": 0.4,
            "memory": 0.3,
            "network": 0.3,
        }

    async def calculate_optimization_score(
        self, resource_usage: dict[str, float]
    ) -> float:
        weighted_sum = 0.0
        for resource, usage in resource_usage.items():
            weight = self.resource_weights.get(resource, 0.0)
            normalized = (
                _normalize_network_load(float(usage))
                if resource == "network"
                else float(usage)
            )
            weighted_sum += (normalized / 100.0) * weight
        return min(1.0, max(0.0, weighted_sum))

    async def get_optimization_actions(self, score: float) -> list[str]:
        actions: list[str] = []
        if score > 0.8:
            actions.extend(["scale_agents", "optimize_network"])
        elif score > 0.6:
            actions.append("scale_agents")
        if score > 0.9:
            actions.append("throttle_noncritical")
        return actions


class SelfOptimizer:
    """Main optimizer coordinating monitoring, policy, and lifecycle actions."""

    def __init__(
        self,
        optimization_interval: int = 60,
        cpu_threshold: float = 80,
        memory_threshold: float = 85,
        network_threshold: float = 90,
    ):
        self.optimization_interval = optimization_interval
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.network_threshold = network_threshold
        self.resource_monitor = ResourceMonitor(
            cpu_threshold=cpu_threshold,
            memory_threshold=memory_threshold,
            network_threshold=network_threshold,
        )
        self.optimization_strategy = OptimizationStrategy(
            optimization_interval=optimization_interval
        )
        self.performance_monitor = PerformanceMonitor(self.resource_monitor)
        self.adaptive_configuration = AdaptiveConfiguration()
        self.lifecycle_manager = AgentLifecycleManager()
        self.orchestrator = None
        self.alert_system = None
        self._running = False

        from .metrics_util import get_or_create_metric as _get_or_create_metric

        self._m_cycles = _get_or_create_metric(
            Counter,
            "daf_optimization_cycles_total",
            "Total DAF self-optimization cycles",
        )
        self._m_resource = _get_or_create_metric(
            Gauge,
            "daf_resource_usage",
            "DAF resource usage",
            ["resource"],
        )
        self._m_latency = _get_or_create_metric(
            Histogram,
            "daf_optimization_latency_seconds",
            "DAF self-optimization cycle latency",
        )
        self._g_agent_count = _get_or_create_metric(
            Gauge,
            "daf_agent_count",
            "Current active DAF agent count",
        )

    async def run_optimization_cycle(self) -> dict[str, Any]:
        cycle_start = time.monotonic()
        usage = await self.resource_monitor.get_resource_usage()
        score = await self.optimization_strategy.calculate_optimization_score(usage)
        baseline_actions = await self.optimization_strategy.get_optimization_actions(
            score
        )

        perf = await self.performance_monitor.collect(resource_usage=usage)
        policy_eval = self.adaptive_configuration.evaluate(perf, score)
        actions = list(dict.fromkeys(baseline_actions + policy_eval.get("actions", [])))
        alerts = self.performance_monitor.detect_anomalies(
            perf,
            cpu_threshold=self.cpu_threshold,
            memory_threshold=self.memory_threshold,
            network_threshold=self.network_threshold,
        )

        if alerts and self.alert_system:
            alert_msg = f"Resource usage above thresholds: {', '.join(alerts)}"
            await self.alert_system.send_alert("high_resource_usage", alert_msg)

        lifecycle_results: dict[str, str] = {}
        if self.orchestrator is not None and actions:
            self.lifecycle_manager.orchestrator = self.orchestrator
            lifecycle_results = await self.lifecycle_manager.apply_actions(actions)

        self._m_cycles.inc()
        self._m_latency.observe(max(0.0, time.monotonic() - cycle_start))
        self._m_resource.labels(resource="cpu").set(float(usage.get("cpu", 0.0)))
        self._m_resource.labels(resource="memory").set(float(usage.get("memory", 0.0)))
        self._m_resource.labels(resource="network").set(
            _normalize_network_load(float(usage.get("network", 0.0)))
        )

        if self.orchestrator is not None:
            count_method = getattr(self.orchestrator, "count_active_agents", None)
            if count_method is not None:
                try:
                    count_value = count_method()
                    if inspect.isawaitable(count_value):
                        count_value = await count_value
                    self._g_agent_count.set(float(count_value))
                except (TypeError, ValueError, AttributeError):
                    pass

        return {
            "score": score,
            "actions": actions,
            "usage": usage,
            "performance_metrics": perf.to_dict(),
            "alerts": alerts,
            "policy": policy_eval,
            "lifecycle_results": lifecycle_results,
        }

    async def start_monitoring(self) -> None:
        self._running = True
        while self._running:
            await self.run_optimization_cycle()
            await asyncio.sleep(self.optimization_interval)

    async def stop_monitoring(self) -> None:
        self._running = False
