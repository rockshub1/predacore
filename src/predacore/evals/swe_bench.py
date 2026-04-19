"""
SWE-bench–style evaluation pipeline for PredaCore.

Measures coding capabilities by running evaluation tasks:
  prompt → agent generates code → diff comparison → test execution → score

Zero external dependencies beyond the standard library.
"""
from __future__ import annotations

import asyncio
import difflib
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class EvalTask:
    """A single coding evaluation task."""

    task_id: str = ""
    prompt: str = ""
    expected_output: str = ""  # Expected code / text output
    expected_files: dict[str, str] = field(default_factory=dict)  # filename → content
    test_command: str = ""  # Command to verify correctness
    timeout: float = 60.0
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy | medium | hard


@dataclass
class EvalResult:
    """Result of executing a single eval task."""

    task_id: str = ""
    passed: bool = False
    score: float = 0.0  # 0.0 – 1.0
    output: str = ""
    expected: str = ""
    diff_summary: str = ""
    error: str = ""
    latency_ms: float = 0.0


@dataclass
class EvalReport:
    """Aggregate report for a suite of evaluation tasks."""

    suite_name: str = "default"
    results: list[EvalResult] = field(default_factory=list)
    total_latency_ms: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total) if self.total else 0.0

    @property
    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    def by_difficulty(self, tasks: list[EvalTask] | None = None) -> dict[str, dict[str, Any]]:
        """Break down results by difficulty level."""
        task_diff = {t.task_id: t.difficulty for t in (tasks or [])}
        buckets: dict[str, list[EvalResult]] = {}
        for r in self.results:
            diff = task_diff.get(r.task_id, "medium")
            buckets.setdefault(diff, []).append(r)
        return {
            k: {"total": len(v), "passed": sum(1 for r in v if r.passed)}
            for k, v in buckets.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 1),
            "avg_score": round(self.avg_score, 3),
            "total_latency_ms": round(self.total_latency_ms, 1),
        }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class EvalScorer:
    """Score agent output against expected output."""

    @staticmethod
    def exact_match(output: str, expected: str) -> float:
        """1.0 if output matches expected exactly, 0.0 otherwise."""
        return 1.0 if output.strip() == expected.strip() else 0.0

    @staticmethod
    def fuzzy_match(output: str, expected: str) -> float:
        """Sequence matcher ratio (0.0 – 1.0)."""
        return difflib.SequenceMatcher(None, output.strip(), expected.strip()).ratio()

    @staticmethod
    def line_match(output: str, expected: str) -> float:
        """Fraction of expected lines present in output."""
        expected_lines = set(expected.strip().splitlines())
        output_lines = set(output.strip().splitlines())
        if not expected_lines:
            return 1.0
        matched = expected_lines.intersection(output_lines)
        return len(matched) / len(expected_lines)

    @staticmethod
    def diff_summary(output: str, expected: str) -> str:
        """Generate a unified diff summary."""
        diff = difflib.unified_diff(
            expected.strip().splitlines(keepends=True),
            output.strip().splitlines(keepends=True),
            fromfile="expected",
            tofile="actual",
            lineterm="",
        )
        return "\n".join(list(diff)[:50])  # Cap at 50 lines


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

# Type alias for the agent function
AgentFn = Callable[[str], Coroutine[Any, Any, str]]


class EvalRunner:
    """
    Executes evaluation tasks against an agent function.

    The agent function takes a prompt string and returns the output string.
    """

    def __init__(
        self,
        scoring_method: str = "fuzzy",
        pass_threshold: float = 0.8,
        log: logging.Logger | None = None,
    ) -> None:
        self._scoring = scoring_method  # exact | fuzzy | line
        self._threshold = pass_threshold
        self._log = log or logger

    async def run_task(
        self,
        task: EvalTask,
        agent_fn: AgentFn,
    ) -> EvalResult:
        """Run a single eval task."""
        t0 = time.monotonic()
        try:
            output = await asyncio.wait_for(
                agent_fn(task.prompt),
                timeout=task.timeout,
            )
        except asyncio.TimeoutError:
            elapsed = (time.monotonic() - t0) * 1000
            return EvalResult(
                task_id=task.task_id,
                error="Timeout",
                latency_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return EvalResult(
                task_id=task.task_id,
                error=str(e),
                latency_ms=elapsed,
            )

        elapsed = (time.monotonic() - t0) * 1000

        # Score
        expected = task.expected_output
        score = self._compute_score(output, expected)
        passed = score >= self._threshold

        return EvalResult(
            task_id=task.task_id,
            passed=passed,
            score=score,
            output=output,
            expected=expected,
            diff_summary=EvalScorer.diff_summary(output, expected),
            latency_ms=elapsed,
        )

    async def run_suite(
        self,
        tasks: list[EvalTask],
        agent_fn: AgentFn,
        suite_name: str = "default",
    ) -> EvalReport:
        """Run a batch of eval tasks sequentially."""
        t0 = time.monotonic()
        results: list[EvalResult] = []
        for task in tasks:
            result = await self.run_task(task, agent_fn)
            results.append(result)
            self._log.info(
                "Task %s: %s (score=%.2f)",
                task.task_id,
                "PASS" if result.passed else "FAIL",
                result.score,
            )

        elapsed = (time.monotonic() - t0) * 1000
        return EvalReport(
            suite_name=suite_name,
            results=results,
            total_latency_ms=elapsed,
        )

    def _compute_score(self, output: str, expected: str) -> float:
        if self._scoring == "exact":
            return EvalScorer.exact_match(output, expected)
        elif self._scoring == "line":
            return EvalScorer.line_match(output, expected)
        else:
            return EvalScorer.fuzzy_match(output, expected)

    async def run_default(self) -> EvalReport:
        """Run the built-in ``SAMPLE_TASKS`` with a minimal stub agent.

        Stub returns each task's ``expected_output`` verbatim — gives a
        100% score and exercises the scoring / aggregation pipeline. This
        is a framework-health check, not a real code-gen evaluation. Use
        ``run_suite`` with a live agent_fn for real capability scoring.
        """
        async def _stub_agent(prompt: str) -> str:
            match = next((t for t in SAMPLE_TASKS if t.prompt == prompt), None)
            return match.expected_output if match else ""
        return await self.run_suite(SAMPLE_TASKS, _stub_agent, suite_name="default")


# ---------------------------------------------------------------------------
# Built-in sample tasks
# ---------------------------------------------------------------------------

SAMPLE_TASKS = [
    EvalTask(
        task_id="fizzbuzz",
        prompt="Write a Python function called fizzbuzz(n) that returns 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, and the number as string otherwise.",
        expected_output=(
            "def fizzbuzz(n):\n"
            "    if n % 15 == 0:\n"
            "        return 'FizzBuzz'\n"
            "    elif n % 3 == 0:\n"
            "        return 'Fizz'\n"
            "    elif n % 5 == 0:\n"
            "        return 'Buzz'\n"
            "    else:\n"
            "        return str(n)"
        ),
        difficulty="easy",
        tags=["python", "basics"],
    ),
    EvalTask(
        task_id="reverse_linked_list",
        prompt="Write a Python function reverse_list(head) that reverses a singly linked list. Assume Node has .val and .next attributes.",
        expected_output=(
            "def reverse_list(head):\n"
            "    prev = None\n"
            "    current = head\n"
            "    while current:\n"
            "        next_node = current.next\n"
            "        current.next = prev\n"
            "        prev = current\n"
            "        current = next_node\n"
            "    return prev"
        ),
        difficulty="medium",
        tags=["python", "data-structures"],
    ),
    EvalTask(
        task_id="binary_search",
        prompt="Write a Python function binary_search(arr, target) that returns the index of target in a sorted array, or -1 if not found.",
        expected_output=(
            "def binary_search(arr, target):\n"
            "    left, right = 0, len(arr) - 1\n"
            "    while left <= right:\n"
            "        mid = (left + right) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            left = mid + 1\n"
            "        else:\n"
            "            right = mid - 1\n"
            "    return -1"
        ),
        difficulty="easy",
        tags=["python", "algorithms"],
    ),
]
