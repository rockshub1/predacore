"""Pattern ABC — strategy interface for orchestration patterns.

Six canonical patterns from Anthropic's "Building Effective Agents":
  - autonomous            single agent loop with tools
  - prompt_chain          sequential N agents
  - routing               classify → one specialist
  - parallelize           known-N parallel agents
  - orchestrator_workers  lead decomposes → dynamic subagents → synthesize
  - evaluator_optimizer   generator + critic loop

Plus one we add for predacore (per Self-MoA paper, Feb 2025):
  - self_moa              same model, N samples, judge synthesizes

All patterns produce a PatternResult with output + per-subagent breakdown
+ token/cost accounting.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..runners.base import AgentResult, RunContext, Runner


@dataclass
class PatternResult:
    """Output of one pattern execution."""

    pattern: str                                  # name of the pattern that ran
    output: str                                   # final user-facing answer
    success: bool = True
    error: str = ""
    subagent_results: list[AgentResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Pattern(ABC):
    """Strategy that takes a task + RunContext and returns a PatternResult."""

    name: str = "abstract"

    @abstractmethod
    async def execute(self, task: str, ctx: RunContext, runner: Runner) -> PatternResult:
        """Run the pattern. Must respect ctx.budget and ctx.cancellation."""
