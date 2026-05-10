"""InProcessRunner — runs subagents in the same Python process via asyncio.

For LLM-bound workloads (95% of agent work) this gives the same
parallelism as DAF without the 1-2 s spawn cost or per-worker HNSW
duplication. asyncio.gather over `await llm.chat(...)` calls is
fully concurrent — N agents finish in max(individual latencies),
not sum.

Delegates the actual agent loop to predacore.agents.engine.AgentEngine
which already implements ReAct-style tool-use. The orchestrator layer
above provides the higher-level patterns.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from ..exceptions import CancellationError
from ..spec import AgentSpec
from .base import AgentResult, RunContext, Runner

logger = logging.getLogger(__name__)


class InProcessRunner(Runner):
    """Default runner — runs subagents inline using asyncio.

    Reuses the existing AgentEngine.run_task implementation, so this is
    a thin orchestration adapter rather than a duplicate agent loop.
    """

    name = "inprocess"

    def __init__(self, agent_engine: Any | None = None) -> None:
        """`agent_engine`: optional override (else built from RunContext)."""
        self._engine = agent_engine

    async def run_spec(self, spec: AgentSpec, ctx: RunContext) -> AgentResult:
        ctx.cancellation.raise_if_cancelled()
        started = time.monotonic()

        # Resolve the AgentEngine: use injected one, else build from ctx
        engine = self._engine or self._build_engine(ctx)
        if engine is None:
            return AgentResult(
                spec_id=spec.id,
                spec=spec,
                output="",
                success=False,
                error="No AgentEngine available in RunContext",
                runner=self.name,
            )

        # Build the prompt: spec system block + preloaded memory + objective
        system_prompt = self._build_system_prompt(spec, ctx)
        user_prompt = self._build_user_prompt(spec, ctx)

        try:
            # AgentEngine.run_task is the existing single-agent loop —
            # we don't reimplement it, we adapt the result.
            result = await engine.run_task(
                prompt=user_prompt,
                agent_type=spec.base_type,
                dynamic_spec=None,  # we already compiled the prompt
                system_prompt_override=system_prompt,
                prompt_override=user_prompt,
                max_steps=spec.max_steps,
                # AgentEngine.run_task doesn't accept user_id/session_id —
                # those flow through ToolContext + LLMInterface state instead.
            )
        except CancellationError:
            return AgentResult(
                spec_id=spec.id,
                spec=spec,
                output="",
                success=False,
                error="cancelled",
                latency_ms=(time.monotonic() - started) * 1000,
                runner=self.name,
            )
        except Exception as exc:  # noqa: BLE001 — runner boundary
            logger.warning(
                "InProcessRunner.run_spec(%s) failed: %s", spec.id, exc, exc_info=True
            )
            return AgentResult(
                spec_id=spec.id,
                spec=spec,
                output="",
                success=False,
                error=str(exc),
                latency_ms=(time.monotonic() - started) * 1000,
                runner=self.name,
            )

        output = str((result or {}).get("output") or "").strip()
        success = bool((result or {}).get("success", True)) and bool(output)
        usage = (result or {}).get("usage") or {}

        # Account for token usage on the budget
        try:
            ctx.budget.record_llm_call(
                model=str((result or {}).get("model") or ""),
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                label=f"{spec.base_type}/{spec.specialization}",
            )
        except Exception:  # noqa: BLE001 — budget overshoot caught here doesn't
            # mean we lose this subagent's output. Surface via error field.
            success = False  # next subagent will hit the same ceiling

        return AgentResult(
            spec_id=spec.id,
            spec=spec,
            output=output,
            success=success,
            error=str((result or {}).get("error") or ""),
            tool_calls=list((result or {}).get("tool_calls") or []),
            latency_ms=(time.monotonic() - started) * 1000,
            tokens_in=int(usage.get("input_tokens") or 0),
            tokens_out=int(usage.get("output_tokens") or 0),
            runner=self.name,
        )

    def _build_engine(self, ctx: RunContext) -> Any | None:
        """Lazy-construct an AgentEngine from RunContext if not injected."""
        if not (ctx.llm and ctx.handler_map):
            return None
        try:
            from ..engine import AgentEngine

            # tool_ctx duck-typed; AgentEngine pulls what it needs from it
            return AgentEngine(
                llm=ctx.llm,
                tool_ctx=ctx.tool_executor,
                handler_map=ctx.handler_map,
            )
        except (ImportError, TypeError, AttributeError) as exc:
            logger.debug("InProcessRunner could not build AgentEngine: %s", exc)
            return None

    def _build_system_prompt(self, spec: AgentSpec, ctx: RunContext) -> str:
        """Spec block + preloaded memory + remaining-budget hint."""
        parts: list[str] = [spec.system_prompt_block()]
        if spec.preloaded_memory:
            parts.append("")
            parts.append("## Preloaded memory context")
            for i, mem in enumerate(spec.preloaded_memory[:8], 1):
                content = str(mem.get("content") or "").strip()
                if content:
                    parts.append(f"{i}. [{mem.get('memory_type', 'memory')}] {content[:300]}")
        rem = ctx.budget.remaining()
        parts.append("")
        parts.append(
            f"## Budget remaining\n"
            f"- tokens: {rem['tokens']} / dollars: ${rem['dollars']:.2f} / "
            f"wall: {rem['wall_seconds']}s / subagents: {rem['subagents']}"
        )
        return "\n".join(parts)

    def _build_user_prompt(self, spec: AgentSpec, ctx: RunContext) -> str:
        """The actual task framing for the subagent."""
        return spec.objective
