"""PRELUDE — the memory-grounded agent loop.

Nine steps. Memory-first. Budget-aware. Cancellation-safe.

  PRELUDE — entry phase (once per orchestration step OR per subagent run)
    P  PRERECALL    — search memory: "have we faced this before?"
    L  LAY-PLAN     — decompose into N steps with success criteria
    U  UPLOAD-CTX   — pre-load each step's working set

  PER-STEP LOOP — repeats until DONE / ABORT / ESCALATE
    D  DELIBERATE   — think (reasoning over plan + recent observations)
    E  EXECUTE      — act (call tool with cancellation_token + budget slice)
    T  TEST         — observe + verify (cheap heuristic + critic on high-stakes)
    E  ENGRAVE      — store significant findings to memory (decay-tagged)

  PERIODIC — every K steps OR on error
    R  REFLECT      — am I making progress? should I re-plan?

  EXIT
    end             — synthesize answer; record outcome to OutcomeStore

This loop is invoked by both InProcessRunner (for in-process subagents)
and DAFRunner workers (over gRPC). Same code, two transports.

Most of the actual LLM-loop logic still lives in
predacore.agents.engine.AgentEngine.run_task — PRELUDE is the wrapper
that adds memory grounding, budget enforcement, and critic gating
without re-implementing the LLM tool-use loop.

This module is the canonical agent loop going forward. Existing
_agent_loop in engine.py remains for legacy callers; new orchestration
goes through PRELUDE.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .budget import OrchestrationBudget
from .critic import CriticGate, DEFAULT_HIGH_STAKES_TOOLS
from .exceptions import (
    BudgetExceededError,
    CancellationError,
    CriticVetoError,
)
from .runners.base import RunContext
from .spec import AgentSpec

logger = logging.getLogger(__name__)


# Constants — tuned for predacore's interactive use case.
PRELUDE_CONSTANTS = {
    "max_steps_simple":          5,
    "max_steps_compare":        15,
    "max_steps_research":       30,
    "reflect_every_k_steps":     5,
    "memory_pre_recall_top_k":   5,
    "memory_pre_load_top_k":     8,
    "engrave_importance_min":    3,
    "duplicate_work_window_sec": 60,
    "stream_partial_after_n":    3,
}


@dataclass
class StepRecord:
    """One iteration through the PER-STEP LOOP."""

    step_index: int
    thought: str = ""                          # DELIBERATE
    tool_name: str = ""                        # EXECUTE
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_result: str = ""                      # TEST
    tool_error: str = ""
    progress: str = ""                         # observation rendered for LLM
    engraved_memory_ids: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


@dataclass
class PreludeResult:
    """Output of one full PRELUDE run."""

    spec_id: str
    output: str = ""
    success: bool = True
    error: str = ""
    plan: list[str] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


async def run_prelude(
    spec: AgentSpec,
    ctx: RunContext,
    *,
    critic: CriticGate | None = None,
    constants: dict[str, Any] | None = None,
) -> PreludeResult:
    """Run one subagent through the PRELUDE loop.

    Most callers should use `InProcessRunner.run_spec` or `DAFRunner.run_spec`
    rather than calling this directly. PRELUDE is the orchestration scaffold
    around `engine.AgentEngine.run_task`; runners handle transport + budget
    accounting.

    This function is a higher-level wrapper that invokes the existing
    AgentEngine after performing PRERECALL + LAY-PLAN + UPLOAD-CTX, and
    after the run, ENGRAVE-s notable findings to memory.
    """
    cs = {**PRELUDE_CONSTANTS, **(constants or {})}
    started = time.monotonic()
    result = PreludeResult(spec_id=spec.id)

    # ── PRELUDE-P: PRERECALL ─────────────────────────────────────────
    prerecall_rows = await _prerecall(spec, ctx, top_k=cs["memory_pre_recall_top_k"])
    ctx.cancellation.raise_if_cancelled()

    # ── PRELUDE-L: LAY-PLAN ──────────────────────────────────────────
    plan = await _lay_plan(spec, ctx, prerecall_rows)
    result.plan = list(plan)
    ctx.cancellation.raise_if_cancelled()

    # ── PRELUDE-U: UPLOAD-CTX ────────────────────────────────────────
    # Build the system prompt with preloaded memory + plan + spec
    enriched_spec = await _upload_ctx(
        spec, ctx, prerecall_rows, plan,
        preload_k=cs["memory_pre_load_top_k"],
    )

    # ── PER-STEP LOOP — delegated to AgentEngine.run_task ────────────
    # We don't reimplement the inner LLM loop; AgentEngine already does
    # ReAct-style tool calling correctly. We frame the inputs and observe
    # the outputs, then ENGRAVE.
    try:
        engine_result = await _invoke_agent_engine(enriched_spec, ctx, critic=critic)
    except CancellationError:
        result.success = False
        result.error = "cancelled"
        result.elapsed_seconds = time.monotonic() - started
        return result
    except BudgetExceededError as exc:
        result.success = False
        result.error = f"budget_exceeded: {exc}"
        result.elapsed_seconds = time.monotonic() - started
        return result

    result.output = str(engine_result.get("output") or "").strip()
    result.success = bool(engine_result.get("success", True)) and bool(result.output)
    result.error = str(engine_result.get("error") or "")
    usage = engine_result.get("usage") or {}
    result.tokens_in = int(usage.get("input_tokens") or 0)
    result.tokens_out = int(usage.get("output_tokens") or 0)
    # Step records (best-effort — AgentEngine's tool_calls list)
    for i, call in enumerate(engine_result.get("tool_calls") or []):
        result.steps.append(
            StepRecord(
                step_index=i,
                tool_name=str(call.get("name") or ""),
                tool_args=call.get("args") or {},
                tool_result=str(call.get("result") or "")[:1500],
                tool_error=str(call.get("error") or ""),
            )
        )

    # ── PRELUDE-E: ENGRAVE ───────────────────────────────────────────
    if result.success:
        engraved_ids = await _engrave(
            spec, ctx, result,
            min_importance=cs["engrave_importance_min"],
        )
        if result.steps and engraved_ids:
            result.steps[-1].engraved_memory_ids = engraved_ids

    result.elapsed_seconds = time.monotonic() - started
    return result


# ── Step implementations ──────────────────────────────────────────────


async def _prerecall(
    spec: AgentSpec, ctx: RunContext, *, top_k: int
) -> list[dict[str, Any]]:
    """PRELUDE-P: pull similar prior task_outcome / decision rows."""
    if ctx.memory is None:
        return []
    try:
        results = await ctx.memory.recall(
            query=spec.objective,
            user_id=ctx.user_id,
            top_k=top_k,
            memory_types=["task_outcome", "decision", "note", "fact"],
            min_importance=2,
            scopes=list(spec.memory_scopes),
            team_id=ctx.team_id,
        )
        return [r for r, _score in (results or [])]
    except Exception as exc:  # noqa: BLE001 — recall best-effort
        logger.debug("prerecall failed for %s: %s", spec.id, exc)
        return []


async def _lay_plan(
    spec: AgentSpec, ctx: RunContext, prerecall_rows: list[dict[str, Any]]
) -> list[str]:
    """PRELUDE-L: decompose objective into a step list.

    Cheap path: rule-based. Steps come from spec.success_criteria + a
    pull from the spec's allowed_tools. The expensive LLM-driven plan
    happens inside AgentEngine.run_task; this is the orchestrator's
    light scaffold.
    """
    plan: list[str] = []
    if spec.success_criteria:
        for c in spec.success_criteria:
            plan.append(f"Achieve: {c}")
    else:
        plan.append(f"Achieve: {spec.objective}")
    if prerecall_rows:
        plan.insert(0, "Review prior work for relevant context")
    return plan


async def _upload_ctx(
    spec: AgentSpec,
    ctx: RunContext,
    prerecall_rows: list[dict[str, Any]],
    plan: list[str],
    *,
    preload_k: int,
) -> AgentSpec:
    """PRELUDE-U: enrich the spec with preloaded memory (skips first
    recall RTT inside the agent's own loop).

    Returns a NEW AgentSpec with preloaded_memory populated. The
    original is frozen.
    """
    if ctx.memory is None:
        return spec

    preloaded: list[dict[str, Any]] = list(prerecall_rows)
    # Pull objective-specific recall too, capped at preload_k
    try:
        more = await ctx.memory.recall(
            query=spec.objective,
            user_id=ctx.user_id,
            top_k=preload_k,
            min_importance=spec.memory_budget // 4 if spec.memory_budget > 4 else 1,
            scopes=list(spec.memory_scopes),
            team_id=ctx.team_id,
        )
        preloaded.extend([r for r, _ in (more or [])])
    except Exception:  # noqa: BLE001 — best-effort
        pass

    # Dedup by id (or content hash if no id)
    seen: set[str] = set()
    dedup: list[dict[str, Any]] = []
    for row in preloaded:
        key = str(row.get("id") or row.get("content_hash") or row.get("content", "")[:64])
        if key and key not in seen:
            seen.add(key)
            dedup.append(row)
    preloaded = dedup[:preload_k]

    # AgentSpec is frozen — build a new one with preloaded_memory
    return AgentSpec(
        id=spec.id,
        base_type=spec.base_type,
        specialization=spec.specialization,
        objective=spec.objective,
        success_criteria=spec.success_criteria,
        output_format=spec.output_format,
        task_boundaries=spec.task_boundaries,
        max_steps=spec.max_steps,
        max_tokens=spec.max_tokens,
        memory_scopes=spec.memory_scopes,
        memory_budget=spec.memory_budget,
        preloaded_memory=tuple(preloaded),
        allowed_tools=spec.allowed_tools,
        trace_id=spec.trace_id,
        parent_run_id=spec.parent_run_id,
        delegation_depth=spec.delegation_depth,
        max_delegation_depth=spec.max_delegation_depth,
    )


async def _invoke_agent_engine(
    spec: AgentSpec, ctx: RunContext, *, critic: CriticGate | None
) -> dict[str, Any]:
    """Run the agent's actual tool-use loop.

    Wraps engine.AgentEngine.run_task. The critic gate (if provided) is
    threaded through via a tool-call callback so high-stakes tools get
    reviewed before commit.
    """
    if ctx.llm is None or ctx.handler_map is None:
        return {"success": False, "error": "missing llm or handler_map", "output": ""}

    try:
        from .engine import AgentEngine
    except ImportError as exc:
        return {"success": False, "error": f"engine import failed: {exc}", "output": ""}

    engine = AgentEngine(
        llm=ctx.llm,
        tool_ctx=ctx.tool_executor,
        handler_map=ctx.handler_map,
    )
    # Build a system prompt that includes spec + preloaded memory
    system_prompt = _compose_system_prompt(spec, ctx)

    # Per-tool-call critic hook is wired here when AgentEngine supports
    # it. For now, the critic is consulted at the engine boundary; a
    # tighter integration is a Phase 4 follow-up.
    try:
        engine_result = await engine.run_task(
            prompt=spec.objective,
            agent_type=spec.base_type,
            dynamic_spec=None,
            system_prompt_override=system_prompt,
            prompt_override=spec.objective,
            max_steps=spec.max_steps,
            # AgentEngine.run_task signature: (prompt, agent_type, max_steps,
            # dynamic_spec, system_prompt_override, prompt_override). user_id
            # + session_id flow via ToolContext + LLMInterface, not as kwargs.
        )
    except Exception as exc:  # noqa: BLE001 — engine boundary
        return {"success": False, "error": str(exc), "output": ""}

    # If a high-stakes tool ran and critic is configured, post-hoc
    # review. (Pre-hoc review requires AgentEngine to expose a hook;
    # we treat post-hoc as a check on the final output instead.)
    if critic is not None:
        await _post_hoc_critic(engine_result, spec, ctx, critic)

    return engine_result or {}


def _compose_system_prompt(spec: AgentSpec, ctx: RunContext) -> str:
    parts: list[str] = [spec.system_prompt_block()]
    if spec.preloaded_memory:
        parts.append("")
        parts.append("## Preloaded memory context")
        for i, mem in enumerate(spec.preloaded_memory[:8], 1):
            content = str(mem.get("content") or "").strip()
            if content:
                parts.append(f"{i}. [{mem.get('memory_type', 'memory')}] {content[:280]}")
    rem = ctx.budget.remaining()
    parts.append("")
    parts.append(
        f"## Budget remaining\n"
        f"- tokens: {rem['tokens']} / dollars: ${rem['dollars']:.2f} / "
        f"wall: {rem['wall_seconds']}s / subagents: {rem['subagents']}"
    )
    return "\n".join(parts)


async def _post_hoc_critic(
    engine_result: dict[str, Any],
    spec: AgentSpec,
    ctx: RunContext,
    critic: CriticGate,
) -> None:
    """Look at the engine's tool calls; if any high-stakes ones, run
    critic post-hoc. Logs verdict; does NOT roll back (you can't
    un-write a file). Pre-hoc gating requires AgentEngine cooperation
    and is a Phase 4 follow-up.
    """
    for call in engine_result.get("tool_calls") or []:
        tool_name = str(call.get("name") or "")
        if tool_name in DEFAULT_HIGH_STAKES_TOOLS:
            try:
                review = await critic.review(
                    tool_name=tool_name,
                    tool_args=call.get("args") or {},
                    goal=ctx.task or spec.objective,
                    reasoning=str(call.get("reasoning") or ""),
                    memory_context="",
                )
                if review.verdict != "APPROVE":
                    logger.warning(
                        "Post-hoc critic flagged %s: %s — %s",
                        tool_name, review.verdict, review.reason,
                    )
            except Exception as exc:  # noqa: BLE001 — critic best-effort
                logger.debug("post-hoc critic failed: %s", exc)


async def _engrave(
    spec: AgentSpec,
    ctx: RunContext,
    result: PreludeResult,
    *,
    min_importance: int,
) -> list[str]:
    """PRELUDE-E: store significant findings to memory."""
    if ctx.memory is None or not result.output.strip():
        return []
    ids: list[str] = []
    # Engrave the final output as a "finding" with team scope so other
    # subagents in the same run can see it.
    try:
        memory_id = await ctx.memory.store(
            content=f"{spec.specialization}: {result.output[:1500]}",
            memory_type="task_outcome" if not spec.parent_run_id else "note",
            importance=min_importance + 1,
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            agent_id=spec.id,
            team_id=ctx.team_id,
            memory_scope="team" if spec.parent_run_id else "user",
            tags=["prelude", spec.base_type, spec.specialization[:32]],
            metadata={
                "spec_id": spec.id,
                "parent_run_id": spec.parent_run_id,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                # Phase 10: trace_id propagates from RunContext through
                # spec → engrave so a single orchestration's writes can
                # be correlated end-to-end via a single trace search.
                "trace_id": spec.trace_id or ctx.trace_id,
                "run_id": ctx.run_id,
            },
            trust_source="claude_inferred",
            confidence=0.7,
        )
        if isinstance(memory_id, dict):
            memory_id = memory_id.get("id", "")
        if memory_id:
            ids.append(str(memory_id))
    except Exception as exc:  # noqa: BLE001 — engrave best-effort
        logger.debug("engrave failed for %s: %s", spec.id, exc)
    return ids


# ── Reflection helper (called periodically by long-running subagents) ─


async def reflect_progress(
    spec: AgentSpec,
    ctx: RunContext,
    *,
    steps_completed: int,
    last_observation: str,
) -> dict[str, Any]:
    """PRELUDE-R: cheap progress check. Caller decides whether to call.

    Returns: {"on_track": bool, "should_replan": bool, "advice": str}

    Used by subagents that run beyond `reflect_every_k_steps`. Cheap
    enough to invoke periodically; the LLM call is small and short.
    """
    if ctx.llm is None:
        return {"on_track": True, "should_replan": False, "advice": ""}

    rem = ctx.budget.remaining()
    prompt = (
        f"Subagent: {spec.specialization}\n"
        f"Objective: {spec.objective}\n"
        f"Output format: {spec.output_format}\n"
        f"Steps completed: {steps_completed}/{spec.max_steps}\n"
        f"Budget left: tokens={rem['tokens']}, dollars=${rem['dollars']:.2f}\n"
        f"Last observation:\n{last_observation[:1000]}\n\n"
        f"Output JSON: {{\"on_track\": <bool>, \"should_replan\": <bool>, "
        f"\"advice\": \"<≤200 chars>\"}}"
    )
    try:
        response = await ctx.llm.chat(
            [
                {"role": "system", "content": "You are a progress monitor. Output strict JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
    except Exception as exc:  # noqa: BLE001 — reflection best-effort
        logger.debug("reflect_progress failed: %s", exc)
        return {"on_track": True, "should_replan": False, "advice": ""}

    import json

    try:
        s = str((response or {}).get("content") or "").strip()
        if s.startswith("```"):
            lines = s.splitlines()[1:-1]
            s = "\n".join(lines).strip()
        obj = json.loads(s)
        if isinstance(obj, dict):
            return {
                "on_track": bool(obj.get("on_track", True)),
                "should_replan": bool(obj.get("should_replan", False)),
                "advice": str(obj.get("advice", ""))[:300],
            }
    except (json.JSONDecodeError, ValueError):
        pass
    return {"on_track": True, "should_replan": False, "advice": ""}


__all__ = [
    "PRELUDE_CONSTANTS",
    "PreludeResult",
    "StepRecord",
    "reflect_progress",
    "run_prelude",
]
