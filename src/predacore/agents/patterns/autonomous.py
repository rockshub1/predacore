"""AutonomousPattern — single agent loop. The simplest pattern.

When to use: trivial single-step or short tool-use tasks. Per Anthropic
("Building Effective Agents"), most queries should land here. Multi-
agent only when needed.

This is essentially the existing _agent_loop wrapped in a Pattern shape.
"""
from __future__ import annotations

from ..runners.base import RunContext, Runner
from ..spec import AgentSpec
from .base import Pattern, PatternResult


class AutonomousPattern(Pattern):
    name = "autonomous"

    async def execute(self, task: str, ctx: RunContext, runner: Runner) -> PatternResult:
        spec = AgentSpec.create(
            base_type="generalist",
            specialization="single autonomous agent",
            objective=f"Answer the user's question completely: {task}",
            output_format="natural language answer with citations to tool outputs",
            success_criteria=("user's question is answered", "all factual claims are sourced"),
            allowed_tools=("read_file", "write_file", "list_directory",
                           "web_search", "web_scrape", "memory_recall", "memory_store",
                           "run_command", "python_exec"),
            max_steps=15,
            max_tokens=ctx.budget.max_total_tokens,
            parent_run_id=ctx.run_id,
            trace_id=ctx.trace_id,
        )
        try:
            ctx.budget.record_subagent_spawn()
            result = await runner.run_spec(spec, ctx)
        except Exception as exc:  # noqa: BLE001 — pattern boundary
            return PatternResult(
                pattern=self.name, output="", success=False, error=str(exc)
            )
        return PatternResult(
            pattern=self.name,
            output=result.output,
            success=result.success,
            error=result.error,
            subagent_results=[result],
        )
