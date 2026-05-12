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


# Default tool allowlist for delegated sub-agents.
#
# Includes most capability tools (filesystem / shell / code exec / web /
# memory read+store / git / desktop / voice / creative / pipeline) so a
# delegated sub-agent can actually complete real work.
#
# EXCLUDES tools that only the main agent should call:
#   - multi_agent / strategic_plan / openclaw_delegate (no nested orchestration
#     from sub-agents — depth is also bounded by _MAX_DELEGATION_DEPTH)
#   - identity_read / identity_update / journal_append (identity is the main
#     agent's domain; sub-agents are spec-driven workers without a persona)
#   - channel_configure / channel_install (config mgmt is main-agent only)
#   - secret_set / secret_list (credentials live in the main process)
#   - mcp_add / mcp_remove / mcp_restart / api_add / api_remove (registry
#     management is main-agent only; api_call + mcp_list pass through fine)
#   - memory_delete / memory_bulk_abort (destructive memory ops stay
#     gated to the main agent)
#   - marketplace_install_skill / collective_intelligence_sync / skill_evolve
#     / skill_scan / skill_endorse (skill/marketplace mgmt is main-agent only)
DEFAULT_SUBAGENT_TOOLS: tuple[str, ...] = (
    # File / shell / code
    "read_file", "write_file", "list_directory", "run_command",
    "python_exec", "execute_code",
    # Web / search
    "web_search", "web_scrape", "deep_search", "semantic_search",
    "browser_control",
    # Memory — read + non-destructive write
    "memory_recall", "memory_get", "memory_stats", "memory_explain",
    "memory_store", "memory_bulk_index", "memory_index_status",
    "memory_scan_directory",
    # Voice
    "speak", "voice_note",
    # Desktop / mobile
    "desktop_control", "screen_vision", "android_control",
    # Git
    "git_context", "git_diff_summary", "git_commit_suggest",
    "git_find_files", "git_semantic_search",
    # Creative
    "image_gen", "pdf_reader", "diagram",
    # Cron / pipeline / stats
    "cron_task", "tool_pipeline", "tool_stats",
    # Registry pass-throughs (read + invoke, not register/remove)
    "api_call", "mcp_list", "marketplace_list_skills", "marketplace_invoke_skill",
    "collective_intelligence_status",
)


class AutonomousPattern(Pattern):
    name = "autonomous"

    async def execute(self, task: str, ctx: RunContext, runner: Runner) -> PatternResult:
        spec = AgentSpec.create(
            base_type="generalist",
            specialization="single autonomous agent",
            objective=f"Answer the user's question completely: {task}",
            output_format="natural language answer with citations to tool outputs",
            success_criteria=("user's question is answered", "all factual claims are sourced"),
            allowed_tools=DEFAULT_SUBAGENT_TOOLS,
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
