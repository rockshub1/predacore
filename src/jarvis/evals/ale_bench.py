"""
ALE-bench: Agent-Level Evaluation harness for JARVIS.

Evaluates agent-level behaviors beyond coding:
- Tool selection accuracy
- Plan quality
- Error recovery
- Efficiency (steps to solution)
"""
from __future__ import annotations

import asyncio
import enum
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class CapabilityCategory(enum.Enum):
    TOOL_USE = "tool_use"
    PLANNING = "planning"
    ERROR_RECOVERY = "error_recovery"
    REASONING = "reasoning"
    MULTI_STEP = "multi_step"


@dataclass
class AgentScenario:
    """Defines a test scenario for agent evaluation."""

    scenario_id: str = ""
    name: str = ""
    description: str = ""
    category: CapabilityCategory = CapabilityCategory.TOOL_USE
    prompt: str = ""
    expected_tools: list[str] = field(default_factory=list)
    expected_outcome: str = ""
    max_steps: int = 10
    timeout: float = 60.0
    difficulty: str = "medium"


@dataclass
class AgentAction:
    """A single action taken by the agent during scenario execution."""

    step: int = 0
    action_type: str = ""  # tool_call | reasoning | output
    tool_name: str = ""
    input_data: str = ""
    output_data: str = ""
    latency_ms: float = 0.0
    error: str = ""


@dataclass
class ScenarioResult:
    """Result of running a single scenario."""

    scenario_id: str = ""
    category: CapabilityCategory = CapabilityCategory.TOOL_USE
    actions: list[AgentAction] = field(default_factory=list)
    final_output: str = ""
    expected_outcome: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    passed: bool = False
    total_latency_ms: float = 0.0
    error: str = ""

    @property
    def num_steps(self) -> int:
        return len(self.actions)

    @property
    def tools_used(self) -> list[str]:
        return [a.tool_name for a in self.actions if a.tool_name]


@dataclass
class AgentEvalReport:
    """Aggregate report for all scenarios."""

    results: list[ScenarioResult] = field(default_factory=list)
    total_latency_ms: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total) if self.total else 0.0

    def by_category(self) -> dict[str, dict[str, Any]]:
        """Break down results by capability category."""
        cats: dict[str, list[ScenarioResult]] = {}
        for r in self.results:
            cats.setdefault(r.category.value, []).append(r)
        return {
            cat: {
                "total": len(results),
                "passed": sum(1 for r in results if r.passed),
                "avg_score": (
                    sum(r.scores.get("overall", 0) for r in results) / len(results)
                    if results
                    else 0
                ),
            }
            for cat, results in cats.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 1),
            "by_category": self.by_category(),
            "total_latency_ms": round(self.total_latency_ms, 1),
        }


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class AgentScorer:
    """Score agent behavior across multiple dimensions."""

    @staticmethod
    def tool_accuracy(
        used: list[str],
        expected: list[str],
    ) -> float:
        """Fraction of expected tools that were actually used."""
        if not expected:
            return 1.0
        matched = set(used).intersection(set(expected))
        return len(matched) / len(expected)

    @staticmethod
    def plan_quality(actions: list[AgentAction], max_steps: int) -> float:
        """
        Score based on efficiency — fewer steps is better.
        1.0 if ≤ half max_steps, degrades linearly to 0.0 at max_steps.
        """
        n = len(actions)
        ideal = max(1, max_steps // 2)
        if n <= ideal:
            return 1.0
        if n >= max_steps:
            return 0.0
        return 1.0 - (n - ideal) / (max_steps - ideal)

    @staticmethod
    def error_recovery(actions: list[AgentAction]) -> float:
        """
        Score based on how well the agent recovered from errors.
        1.0 if no errors or all errors followed by successful retry.
        """
        errors = [i for i, a in enumerate(actions) if a.error]
        if not errors:
            return 1.0

        recoveries = 0
        for err_idx in errors:
            # Check if the next action was successful
            if err_idx + 1 < len(actions) and not actions[err_idx + 1].error:
                recoveries += 1

        return recoveries / len(errors) if errors else 1.0

    @staticmethod
    def outcome_match(output: str, expected: str) -> float:
        """Simple keyword overlap score for outcome verification."""
        if not expected:
            return 1.0
        expected_words = set(expected.lower().split())
        output_words = set(output.lower().split())
        if not expected_words:
            return 1.0
        overlap = expected_words.intersection(output_words)
        return len(overlap) / len(expected_words)

    @classmethod
    def score_scenario(
        cls,
        result: ScenarioResult,
        scenario: AgentScenario,
    ) -> dict[str, float]:
        """Compute all score dimensions for a scenario result."""
        scores = {
            "tool_accuracy": cls.tool_accuracy(
                result.tools_used, scenario.expected_tools
            ),
            "plan_quality": cls.plan_quality(result.actions, scenario.max_steps),
            "error_recovery": cls.error_recovery(result.actions),
            "outcome_match": cls.outcome_match(
                result.final_output, scenario.expected_outcome
            ),
        }
        scores["overall"] = sum(scores.values()) / len(scores)
        return scores


# ---------------------------------------------------------------------------
# Agent simulator (for evaluation)
# ---------------------------------------------------------------------------

# The agent function receives a prompt and returns (actions, final_output)
AgentFn = Callable[[str], Coroutine[Any, Any, tuple[list[AgentAction], str]]]


class AgentEvalRunner:
    """
    Runs agent evaluation scenarios.

    The agent function should accept a prompt string and return
    a tuple of (List[AgentAction], final_output_string).
    """

    def __init__(
        self,
        pass_threshold: float = 0.6,
        log: logging.Logger | None = None,
    ) -> None:
        self._threshold = pass_threshold
        self._log = log or logger

    async def run_scenario(
        self,
        scenario: AgentScenario,
        agent_fn: AgentFn,
    ) -> ScenarioResult:
        """Execute a single scenario against the agent."""
        t0 = time.monotonic()
        try:
            actions, final_output = await asyncio.wait_for(
                agent_fn(scenario.prompt),
                timeout=scenario.timeout,
            )
        except asyncio.TimeoutError:
            elapsed = (time.monotonic() - t0) * 1000
            return ScenarioResult(
                scenario_id=scenario.scenario_id,
                category=scenario.category,
                error="Timeout",
                total_latency_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return ScenarioResult(
                scenario_id=scenario.scenario_id,
                category=scenario.category,
                error=str(e),
                total_latency_ms=elapsed,
            )

        elapsed = (time.monotonic() - t0) * 1000

        # Truncate to max_steps
        actions = actions[: scenario.max_steps]

        result = ScenarioResult(
            scenario_id=scenario.scenario_id,
            category=scenario.category,
            actions=actions,
            final_output=final_output,
            expected_outcome=scenario.expected_outcome,
            total_latency_ms=elapsed,
        )

        # Score
        result.scores = AgentScorer.score_scenario(result, scenario)
        result.passed = result.scores.get("overall", 0) >= self._threshold

        return result

    async def run_suite(
        self,
        scenarios: list[AgentScenario],
        agent_fn: AgentFn,
    ) -> AgentEvalReport:
        """Run a full suite of scenarios."""
        t0 = time.monotonic()
        results: list[ScenarioResult] = []
        for scenario in scenarios:
            result = await self.run_scenario(scenario, agent_fn)
            results.append(result)
            self._log.info(
                "Scenario %s: %s (overall=%.2f)",
                scenario.scenario_id,
                "PASS" if result.passed else "FAIL",
                result.scores.get("overall", 0),
            )

        elapsed = (time.monotonic() - t0) * 1000
        return AgentEvalReport(results=results, total_latency_ms=elapsed)


# ---------------------------------------------------------------------------
# Built-in scenarios
# ---------------------------------------------------------------------------

SAMPLE_SCENARIOS = [
    AgentScenario(
        scenario_id="file_manipulation",
        name="File Creation and Editing",
        description="Agent must create a file, write content, and verify it exists.",
        category=CapabilityCategory.TOOL_USE,
        prompt="Create a file called test.txt with the content 'hello world', then verify it exists.",
        expected_tools=["create_file", "read_file"],
        expected_outcome="file created successfully",
        max_steps=5,
        difficulty="easy",
    ),
    AgentScenario(
        scenario_id="web_search",
        name="Web Search and Synthesis",
        description="Agent must search the web and summarize findings.",
        category=CapabilityCategory.REASONING,
        prompt="Search for the capital of France and provide a brief summary.",
        expected_tools=["web_search"],
        expected_outcome="Paris capital France",
        max_steps=5,
        difficulty="easy",
    ),
    AgentScenario(
        scenario_id="code_debugging",
        name="Code Debugging",
        description="Agent must find and fix a bug in provided code.",
        category=CapabilityCategory.ERROR_RECOVERY,
        prompt="Debug this Python code: def add(a, b): return a - b. The function should add two numbers.",
        expected_tools=["execute_code"],
        expected_outcome="return a + b",
        max_steps=5,
        difficulty="easy",
    ),
    AgentScenario(
        scenario_id="multi_step_planning",
        name="Multi-Step Task Planning",
        description="Agent must break down a complex task into steps and execute them.",
        category=CapabilityCategory.MULTI_STEP,
        prompt="Create a Python project with: 1) a main.py file, 2) a utils.py with a helper function, 3) a test file",
        expected_tools=["create_file", "execute_code"],
        expected_outcome="project files created",
        max_steps=10,
        difficulty="medium",
    ),
]
