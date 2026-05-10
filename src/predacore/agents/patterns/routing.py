"""RoutingPattern — classify input → dispatch to one specialist.

Per Anthropic ("Building Effective Agents"): for tasks with distinct
categories that are better handled separately. Examples: route customer
support queries by type; route simple questions to a smaller cheaper
model.

This pattern doesn't fan out. It picks the ONE best specialist and
delegates. Cheap and fast. Use when:
  - Task class is clear from the input
  - Categories have specialized handlers
  - You want cheap-model routing for trivial queries
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from ..exceptions import BudgetExceededError
from ..runners.base import RunContext, Runner
from ..spec import AgentSpec
from .base import Pattern, PatternResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Route:
    """One route in a routing table."""

    name: str                 # "code_question", "factual_lookup", ...
    description: str          # for the classifier prompt
    base_type: str            # specialist agent template
    specialization: str
    output_format: str
    allowed_tools: tuple[str, ...]
    max_steps: int = 8
    objective_template: str = "Answer the user's task: {task}"


_CLASSIFIER_TEMPLATE = """Classify the user's task into ONE of these routes:

{routes}

User's task:
<<<TASK
{task}
END_TASK>>>

Output ONLY the route name (one word, exact match from the list above)."""


class RoutingPattern(Pattern):
    name = "routing"

    def __init__(self, routes: Sequence[Route], *, fallback: Route | None = None) -> None:
        if not routes:
            raise ValueError("RoutingPattern requires at least one route")
        self._routes = list(routes)
        self._fallback = fallback or routes[0]
        self._table = {r.name: r for r in routes}

    async def execute(self, task: str, ctx: RunContext, runner: Runner) -> PatternResult:
        # ── 1. Classify ────────────────────────────────────────────────
        route = await self._classify(task, ctx)

        # ── 2. Dispatch to specialist ──────────────────────────────────
        try:
            ctx.budget.record_subagent_spawn()
        except BudgetExceededError as exc:
            return PatternResult(
                pattern=self.name, output="", success=False, error=str(exc)
            )

        spec = AgentSpec.create(
            base_type=route.base_type,
            specialization=route.specialization,
            objective=route.objective_template.format(task=task),
            output_format=route.output_format,
            success_criteria=("user's task is answered",),
            allowed_tools=route.allowed_tools,
            max_steps=route.max_steps,
            max_tokens=ctx.budget.max_total_tokens // 2,
            parent_run_id=ctx.run_id,
            trace_id=ctx.trace_id,
        )
        result = await runner.run_spec(spec, ctx)
        return PatternResult(
            pattern=self.name,
            output=result.output,
            success=result.success,
            error=result.error,
            subagent_results=[result],
            metadata={"route": route.name},
        )

    async def _classify(self, task: str, ctx: RunContext) -> Route:
        if ctx.llm is None:
            return self._fallback
        routes_text = "\n".join(f"- {r.name}: {r.description}" for r in self._routes)
        prompt = _CLASSIFIER_TEMPLATE.format(routes=routes_text, task=task)
        try:
            response = await ctx.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20,
            )
        except Exception as exc:  # noqa: BLE001 — classifier failure → fallback
            logger.debug("classifier failed: %s — using fallback", exc)
            return self._fallback

        usage = (response or {}).get("usage") or {}
        try:
            ctx.budget.record_llm_call(
                model=str((response or {}).get("model") or ""),
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                label="route_classify",
            )
        except BudgetExceededError:
            return self._fallback
        guess = str((response or {}).get("content") or "").strip().split()[:1]
        if guess and guess[0] in self._table:
            return self._table[guess[0]]
        return self._fallback
