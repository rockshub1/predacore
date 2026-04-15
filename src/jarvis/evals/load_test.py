"""
Load Tester — Stress-test JARVIS endpoints and measure performance.

Provides a simple, zero-dependency load testing tool that:
  - Simulates concurrent users via asyncio
  - Measures latency percentiles (p50, p95, p99)
  - Reports throughput and error rates
  - Generates summary reports
"""
from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("jarvis.evals.load_test")


@dataclass
class RequestResult:
    """Result of a single simulated request."""

    status: int = 200
    latency_ms: float = 0.0
    error: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def is_success(self) -> bool:
        return 200 <= self.status < 400 and not self.error


@dataclass
class LoadTestConfig:
    """Configuration for a load test run."""

    name: str = "default"
    concurrent_users: int = 10
    total_requests: int = 100
    ramp_up_seconds: float = 5.0
    think_time_ms: float = 100.0  # Pause between requests per user
    timeout_seconds: float = 30.0


@dataclass
class LoadTestReport:
    """Summary report of a load test run."""

    config: LoadTestConfig
    results: list[RequestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def total_requests(self) -> int:
        return len(self.results)

    @property
    def successful(self) -> int:
        return sum(1 for r in self.results if r.is_success)

    @property
    def failed(self) -> int:
        return self.total_requests - self.successful

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.successful / self.total_requests * 100

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def requests_per_second(self) -> float:
        if self.duration_seconds <= 0:
            return 0.0
        return self.total_requests / self.duration_seconds

    def latency_percentile(self, p: float) -> float:
        """Get latency at given percentile (0-100)."""
        latencies = sorted(r.latency_ms for r in self.results if r.is_success)
        if not latencies:
            return 0.0
        idx = int(len(latencies) * p / 100)
        return latencies[min(idx, len(latencies) - 1)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "total_requests": self.total_requests,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": round(self.success_rate, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "requests_per_second": round(self.requests_per_second, 2),
            "latency_ms": {
                "min": round(min((r.latency_ms for r in self.results), default=0), 2),
                "max": round(max((r.latency_ms for r in self.results), default=0), 2),
                "avg": round(
                    statistics.mean(r.latency_ms for r in self.results)
                    if self.results
                    else 0,
                    2,
                ),
                "p50": round(self.latency_percentile(50), 2),
                "p95": round(self.latency_percentile(95), 2),
                "p99": round(self.latency_percentile(99), 2),
            },
            "errors": [
                {"error": r.error, "status": r.status}
                for r in self.results
                if not r.is_success
            ][:20],  # Cap error samples
        }

    def print_summary(self) -> str:
        """Generate a human-readable summary."""
        d = self.to_dict()
        lines = [
            "═══════════════════════════════════════════════════",
            f"  Load Test Report: {d['name']}",
            "═══════════════════════════════════════════════════",
            f"  Total Requests:    {d['total_requests']}",
            f"  Successful:        {d['successful']}",
            f"  Failed:            {d['failed']}",
            f"  Success Rate:      {d['success_rate']}%",
            f"  Duration:          {d['duration_seconds']}s",
            f"  Throughput:        {d['requests_per_second']} req/s",
            "───────────────────────────────────────────────────",
            "  Latency (ms):",
            f"    min:  {d['latency_ms']['min']}",
            f"    avg:  {d['latency_ms']['avg']}",
            f"    p50:  {d['latency_ms']['p50']}",
            f"    p95:  {d['latency_ms']['p95']}",
            f"    p99:  {d['latency_ms']['p99']}",
            f"    max:  {d['latency_ms']['max']}",
            "═══════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


# ── Load Test Runner ─────────────────────────────────────────────────


class LoadTestRunner:
    """
    Async load test runner.

    Usage:
        async def my_request(user_id: int) -> RequestResult:
            start = time.time()
            # ... make your HTTP call ...
            return RequestResult(status=200, latency_ms=(time.time()-start)*1000)

        runner = LoadTestRunner()
        report = await runner.run(my_request, LoadTestConfig(
            concurrent_users=20,
            total_requests=500,
        ))
        print(report.print_summary())
    """

    async def run(
        self,
        request_fn: Callable[[int], Any],
        config: LoadTestConfig,
    ) -> LoadTestReport:
        """Run a load test."""
        report = LoadTestReport(config=config)
        report.start_time = time.time()

        # Distribute requests across users
        requests_per_user = config.total_requests // config.concurrent_users
        extra = config.total_requests % config.concurrent_users

        # Create user tasks
        tasks = []
        for user_id in range(config.concurrent_users):
            n = requests_per_user + (1 if user_id < extra else 0)
            # Ramp up delay
            delay = (user_id / config.concurrent_users) * config.ramp_up_seconds
            tasks.append(self._user_worker(user_id, n, request_fn, config, delay))

        # Run all users concurrently
        all_results = await asyncio.gather(*tasks)
        for user_results in all_results:
            report.results.extend(user_results)

        report.end_time = time.time()
        return report

    async def _user_worker(
        self,
        user_id: int,
        num_requests: int,
        request_fn: Callable,
        config: LoadTestConfig,
        start_delay: float,
    ) -> list[RequestResult]:
        """Simulate a single user making requests."""
        results = []
        await asyncio.sleep(start_delay)

        for _i in range(num_requests):
            try:
                start = time.time()
                if asyncio.iscoroutinefunction(request_fn):
                    result = await asyncio.wait_for(
                        request_fn(user_id),
                        timeout=config.timeout_seconds,
                    )
                else:
                    result = request_fn(user_id)

                if isinstance(result, RequestResult):
                    results.append(result)
                else:
                    elapsed = (time.time() - start) * 1000
                    results.append(RequestResult(latency_ms=elapsed))

            except asyncio.TimeoutError:
                results.append(RequestResult(status=408, error="timeout"))
            except Exception as e:
                results.append(RequestResult(status=500, error=str(e)))

            # Think time between requests
            if config.think_time_ms > 0:
                await asyncio.sleep(config.think_time_ms / 1000)

        return results
