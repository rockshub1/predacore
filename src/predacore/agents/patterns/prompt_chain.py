"""PromptChainPattern — sequential N agents, each consuming the previous
agent's output.

Per Anthropic ("Building Effective Agents"): decompose tasks into fixed
sequential steps with programmatic gates. Trades latency for accuracy.

Examples:
  - draft → translate → polish
  - extract entities → look up canonical names → fill report
  - parse log → classify error → propose fix

When to use: known fixed pipeline of stages. If steps are independent,
use ParallelizePattern instead. If decomposition is dynamic, use
OrchestratorWorkersPattern.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..exceptions import BudgetExceededError
from ..runners.base import AgentResult, RunContext, Runner
from ..spec import AgentSpec
from .base import Pattern, PatternResult


@dataclass(frozen=True)
class ChainStep:
    """One stage in a prompt chain."""

    base_type: str
    specialization: str
    objective_template: str   # may reference {input} (the prior step's output)
    output_format: str
    allowed_tools: tuple[str, ...] = ("memory_recall",)
    max_steps: int = 6
    success_criteria: tuple[str, ...] = ()


class PromptChainPattern(Pattern):
    name = "prompt_chain"

    def __init__(self, steps: Sequence[ChainStep]) -> None:
        if not steps:
            raise ValueError("PromptChainPattern requires at least one step")
        self._steps = list(steps)

    async def execute(self, task: str, ctx: RunContext, runner: Runner) -> PatternResult:
        results: list[AgentResult] = []
        last_output = task

        for i, step in enumerate(self._steps):
            try:
                ctx.budget.record_subagent_spawn()
            except BudgetExceededError as exc:
                return PatternResult(
                    pattern=self.name,
                    output=last_output,
                    success=False,
                    error=f"budget exhausted at step {i+1}: {exc}",
                    subagent_results=results,
                )

            objective = step.objective_template.format(input=last_output)
            spec = AgentSpec.create(
                base_type=step.base_type,
                specialization=f"{step.specialization} (step {i+1}/{len(self._steps)})",
                objective=objective,
                output_format=step.output_format,
                success_criteria=step.success_criteria
                or (f"step {i+1} output is non-empty",),
                allowed_tools=step.allowed_tools,
                max_steps=step.max_steps,
                max_tokens=ctx.budget.max_total_tokens // max(1, len(self._steps)),
                parent_run_id=ctx.run_id,
                trace_id=f"{ctx.trace_id}-step{i}",
            )
            result = await runner.run_spec(spec, ctx)
            results.append(result)
            if not result.success or not result.output.strip():
                return PatternResult(
                    pattern=self.name,
                    output=last_output,
                    success=False,
                    error=f"step {i+1} failed: {result.error}",
                    subagent_results=results,
                )
            last_output = result.output

        return PatternResult(
            pattern=self.name,
            output=last_output,
            success=True,
            subagent_results=results,
            metadata={"n_steps": len(self._steps)},
        )
