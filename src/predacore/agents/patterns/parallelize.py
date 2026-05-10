"""ParallelizePattern — known-N parallel agents, fan-out, gather, synthesize.

When to use: independent subtasks that share a common synthesis step.
Examples: "compare X and Y" → 2 parallel agents (one per item) + lead
synthesis; "summarize 10 papers" → 10 parallel summarizers + 1 merger.

Distinct from orchestrator_workers: here the lead pre-decides the N
subtasks. With orchestrator_workers, the lead decides dynamically based
on intermediate findings.

Uses asyncio.gather → fully concurrent for I/O-bound (LLM HTTP) work.
Wall time = max(individual subagent latencies), not sum.
"""
from __future__ import annotations

import asyncio
from typing import Sequence

from ..runners.base import RunContext, Runner
from ..spec import AgentSpec
from .base import Pattern, PatternResult


class ParallelizePattern(Pattern):
    name = "parallelize"

    def __init__(self, specs: Sequence[AgentSpec] | None = None) -> None:
        """`specs`: pre-built specs (caller decided the fan-out shape).

        If None, falls back to a 2-way generalist split when execute()
        is called — useful for tests or when orchestrator hasn't
        decomposed yet.
        """
        self._specs = list(specs or ())

    async def execute(self, task: str, ctx: RunContext, runner: Runner) -> PatternResult:
        specs = self._specs or self._default_specs(task, ctx)
        # Spawn budget check (raises BudgetExceeded if over max_subagents)
        for _ in specs:
            ctx.budget.record_subagent_spawn()

        async def _wrap(s: AgentSpec) -> "AgentResult":  # noqa: F821
            return await runner.run_spec(s, ctx)

        results = await asyncio.gather(*[_wrap(s) for s in specs], return_exceptions=False)
        ok_results = [r for r in results if r.success]
        merged = self._merge_outputs(ok_results)
        return PatternResult(
            pattern=self.name,
            output=merged,
            success=bool(ok_results),
            subagent_results=list(results),
            metadata={"n_succeeded": len(ok_results), "n_total": len(results)},
        )

    def _default_specs(self, task: str, ctx: RunContext) -> list[AgentSpec]:
        """Trivial default — used only when no specs were pre-supplied."""
        return [
            AgentSpec.create(
                base_type="analyst",
                specialization=f"perspective {i+1}",
                objective=f"Analyze the following from perspective {i+1}: {task}",
                output_format="markdown bullet list of findings with citations",
                success_criteria=(f"perspective {i+1} addressed",),
                allowed_tools=("web_search", "web_scrape", "memory_recall"),
                max_steps=10,
                max_tokens=ctx.budget.max_total_tokens // 4,
                parent_run_id=ctx.run_id,
                trace_id=ctx.trace_id,
            )
            for i in range(2)
        ]

    @staticmethod
    def _merge_outputs(results: list["AgentResult"]) -> str:  # noqa: F821
        """Trivial concatenation. Pattern users typically pass merged
        output to a synthesizer subagent themselves."""
        sections: list[str] = []
        for i, r in enumerate(results, 1):
            sections.append(f"### Subagent {i} ({r.spec.specialization})\n{r.output}")
        return "\n\n".join(sections)
