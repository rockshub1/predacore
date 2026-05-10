"""EvaluatorOptimizerPattern — generator + critic loop.

Per Anthropic ("Building Effective Agents"): one LLM generates, another
provides iterative feedback. Effective when there are clear evaluation
criteria and iterative refinement provides measurable value.

When to use:
  - Code generation with tests (run tests → critique → revise)
  - Literary translation (style critique → revise)
  - Long-form writing where structure matters
  - Plan refinement before execution

Loop:
  1. Generator produces draft
  2. Critic scores against criteria; if PASS → done; else → REVISE
  3. Generator revises with critic's notes
  4. Repeat up to max_rounds
"""
from __future__ import annotations

import logging
from typing import Sequence

from ..exceptions import BudgetExceededError
from ..runners.base import RunContext, Runner
from ..spec import AgentSpec
from .base import Pattern, PatternResult

logger = logging.getLogger(__name__)


class EvaluatorOptimizerPattern(Pattern):
    name = "evaluator_optimizer"

    def __init__(
        self,
        *,
        criteria: Sequence[str] = (),
        max_rounds: int = 3,
        critic_model: str | None = None,
    ) -> None:
        self._criteria = tuple(criteria) or (
            "Output addresses the task fully",
            "All factual claims are sourced",
            "No internal inconsistencies",
        )
        self._max_rounds = max(1, min(5, max_rounds))
        self._critic_model = critic_model

    async def execute(self, task: str, ctx: RunContext, runner: Runner) -> PatternResult:
        gen_history: list[str] = []
        critic_history: list[str] = []
        last_output = ""

        for round_idx in range(self._max_rounds):
            # ── Generator ──────────────────────────────────────────────
            try:
                ctx.budget.record_subagent_spawn()
            except BudgetExceededError as exc:
                logger.info("EvaluatorOptimizer budget hit at round %d: %s", round_idx, exc)
                break

            gen_spec = AgentSpec.create(
                base_type="generalist",
                specialization=f"generator round {round_idx+1}",
                objective=self._build_generator_objective(task, critic_history),
                output_format="full final answer (be complete; this will be evaluated)",
                success_criteria=tuple(self._criteria),
                allowed_tools=("read_file", "write_file", "web_search",
                               "memory_recall", "run_command"),
                max_steps=12,
                max_tokens=ctx.budget.max_total_tokens // (self._max_rounds * 2),
                parent_run_id=ctx.run_id,
                trace_id=f"{ctx.trace_id}-gen{round_idx}",
            )
            gen_result = await runner.run_spec(gen_spec, ctx)
            if not gen_result.success or not gen_result.output.strip():
                if round_idx == 0:
                    return PatternResult(
                        pattern=self.name, output="", success=False,
                        error=f"generator failed: {gen_result.error}",
                        subagent_results=[gen_result],
                    )
                # Use last good output
                break
            last_output = gen_result.output
            gen_history.append(last_output)

            # ── Critic ─────────────────────────────────────────────────
            try:
                ctx.budget.record_subagent_spawn()
            except BudgetExceededError:
                break

            critic_spec = AgentSpec.create(
                base_type="critic",
                specialization=f"critic round {round_idx+1}",
                objective=self._build_critic_objective(task, last_output),
                output_format=(
                    "JSON: {\"verdict\": \"PASS\"|\"REVISE\", "
                    "\"score\": <0-10>, \"issues\": [<string>...], "
                    "\"suggested_revisions\": <string>}"
                ),
                success_criteria=("scored each criterion", "verdict is PASS or REVISE"),
                allowed_tools=("memory_recall",),
                max_steps=4,
                max_tokens=2000,
                parent_run_id=ctx.run_id,
                trace_id=f"{ctx.trace_id}-crit{round_idx}",
            )
            crit_result = await runner.run_spec(critic_spec, ctx)
            critic_history.append(crit_result.output)

            if self._critic_says_pass(crit_result.output):
                return PatternResult(
                    pattern=self.name,
                    output=last_output,
                    success=True,
                    subagent_results=[gen_result, crit_result],
                    metadata={"rounds": round_idx + 1, "verdict": "PASS"},
                )
        # Loop exhausted — return the last generator output
        return PatternResult(
            pattern=self.name,
            output=last_output,
            success=bool(last_output),
            subagent_results=[],  # large; orchestrator can re-run with logging
            metadata={"rounds": self._max_rounds, "verdict": "MAX_ROUNDS"},
        )

    def _build_generator_objective(self, task: str, critic_notes: list[str]) -> str:
        if not critic_notes:
            return f"Produce a complete answer to: {task}"
        return (
            f"Revise the draft answer to address the critic's concerns. "
            f"Original task: {task}\nLatest critic notes: {critic_notes[-1][:1500]}"
        )

    def _build_critic_objective(self, task: str, draft: str) -> str:
        return (
            f"Evaluate the draft answer against these criteria: "
            f"{'; '.join(self._criteria)}. "
            f"Original task: {task}\n"
            f"Draft to evaluate:\n{draft[:3000]}"
        )

    @staticmethod
    def _critic_says_pass(critic_output: str) -> bool:
        s = (critic_output or "").lower()
        # Liberal parse — any "PASS" verdict accepts.
        if '"verdict"' in s and '"pass"' in s:
            return True
        if "verdict: pass" in s or "verdict:pass" in s:
            return True
        return False
