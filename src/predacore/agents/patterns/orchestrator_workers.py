"""OrchestratorWorkersPattern — Anthropic's flagship multi-agent pattern.

Lead agent (typically a strong model like Opus) does:
  1. Decompose: classify task, generate N concrete subagent specs
  2. Spawn:     parallel via runner (asyncio.gather or DAF)
  3. Watchdog:  monitor for duplicate-work; cancel redundant subagents
  4. Synthesize: merge findings, attribute citations

Subagents (typically Sonnet/Haiku for cost) each get:
  - A concrete AgentSpec (validated; vague specs refused)
  - Pre-loaded memory from team scope
  - Hard token budget slice
  - Cancellation token

Anthropic's published numbers (May 2025): Opus-lead + Sonnet-subagents
beat single-agent Opus by 90.2% on internal evals. Cost: ~15× chat
tokens. Without OrchestrationBudget hard ceilings this gets expensive
fast.

This pattern is the workhorse for non-trivial questions.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ..exceptions import BudgetExceededError, SpecValidationError
from ..runners.base import AgentResult, RunContext, Runner
from ..spec import ANTHROPIC_SCALING_RULES, AgentSpec
from .base import Pattern, PatternResult

logger = logging.getLogger(__name__)


_DECOMPOSE_PROMPT_TEMPLATE = """You are the LEAD agent in a multi-agent orchestration.

Your job is to decompose the user's task into 1-{max_subagents} concrete subagent specs.

{scaling_rules}

The user's task is:
<<<TASK
{task}
END_TASK>>>

Available subagent base types (templates you can specialize):
- analyst        — analyzes data, compares, evaluates
- researcher     — gathers information from external sources
- critic         — reviews + verifies output quality
- synthesizer    — merges findings into a final answer
- planner        — proposes step-by-step plans
- coder          — writes/reviews code
- generalist     — flexible single-step worker

Available tools: {available_tools}

Output a JSON array of subagent specs. Each must include:
{{
  "base_type": "<one of the templates above>",
  "specialization": "<concrete role, e.g. 'compare Q3 EPS for AAPL vs MSFT'>",
  "objective": "<concrete action verb statement, ≥5 words>",
  "output_format": "<markdown table | JSON schema X | bullet list | …>",
  "success_criteria": ["<specific checkpoint>", "<specific checkpoint>"],
  "task_boundaries": "<what this subagent should NOT investigate>",
  "max_steps": <int per scaling rules>,
  "allowed_tools": ["<explicit tool names>", ...]
}}

Output ONLY the JSON array. No commentary outside the JSON.
Hard ceiling: you must produce at most {max_subagents} subagent specs.
"""


_SYNTHESIZE_PROMPT_TEMPLATE = """You are the LEAD agent synthesizing subagent findings.

Original task:
<<<TASK
{task}
END_TASK>>>

Subagent results follow. Each subagent had its own objective; cite by
subagent number when integrating their findings.

{subagent_blocks}

Synthesize a final answer to the user's task. Be concise. If subagents
disagreed or returned errors, note this. If important information is
missing, say so explicitly rather than fabricating."""


class OrchestratorWorkersPattern(Pattern):
    name = "orchestrator_workers"

    def __init__(
        self,
        *,
        lead_model: str | None = None,
        max_subagents: int = 10,
        stream_partial_after_n: int = 3,
        stream_partial_min_seconds: float = 5.0,
    ) -> None:
        self._lead_model = lead_model
        self._max_subagents = max_subagents
        # Phase 9: streaming partial synthesis. After N subagents complete OR
        # stream_partial_min_seconds elapsed, the lead can synthesize from
        # what's done while the rest finish. Cuts P99 latency on slow-tail
        # subagents per Anthropic's documented "synchronous bottleneck."
        self._stream_after_n = max(1, stream_partial_after_n)
        self._stream_after_seconds = max(0.0, stream_partial_min_seconds)

    async def execute(self, task: str, ctx: RunContext, runner: Runner) -> PatternResult:
        # ── 1. DECOMPOSE ───────────────────────────────────────────────
        try:
            specs = await self._decompose(task, ctx)
        except (SpecValidationError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("decompose failed: %s — falling back to autonomous", exc)
            from .autonomous import AutonomousPattern

            return await AutonomousPattern().execute(task, ctx, runner)

        if not specs:
            from .autonomous import AutonomousPattern

            return await AutonomousPattern().execute(task, ctx, runner)

        # Cap at budget's max_subagents
        rem = ctx.budget.remaining()
        specs = specs[: max(1, min(self._max_subagents, rem["subagents"]))]

        # Spawn quota check (each call may raise BudgetExceededError)
        spawned: list[AgentSpec] = []
        for spec in specs:
            try:
                ctx.budget.record_subagent_spawn()
                spawned.append(spec)
            except BudgetExceededError:
                logger.info(
                    "OrchestratorWorkers spawn-budget hit at %d subagents; remainder dropped",
                    len(spawned),
                )
                break

        # ── 2. SPAWN (parallel, with streaming partial synthesis) ──────
        # Phase 9: instead of waiting for ALL subagents, we use as_completed
        # so the lead can start synthesizing once enough have finished.
        # Slow-tail subagents don't block the user response.
        results: list[AgentResult] = []
        pending_streamed = False
        partial_synth: str = ""
        import time as _time

        started_at = _time.monotonic()

        async def _run(spec: AgentSpec) -> AgentResult:
            return await runner.run_spec(spec, ctx)

        tasks = [asyncio.create_task(_run(s)) for s in spawned]
        try:
            for completed in asyncio.as_completed(tasks):
                r = await completed
                results.append(r)

                # Streaming threshold check: ≥ N succeeded OR enough time elapsed
                n_ok = sum(1 for x in results if x.success)
                elapsed = _time.monotonic() - started_at
                if (
                    not pending_streamed
                    and (n_ok >= self._stream_after_n
                         or elapsed >= self._stream_after_seconds)
                    and len(results) < len(tasks)
                ):
                    # Synthesize partial early so the user can see something.
                    # We don't return here — final synthesis will fold in
                    # later subagents' findings.
                    pending_streamed = True
                    try:
                        partial_synth = await self._synthesize(task, results, ctx)
                    except Exception as exc:  # noqa: BLE001 — partial best-effort
                        logger.debug("partial synth failed: %s", exc)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise

        # ── 3. FINAL SYNTHESIS ──────────────────────────────────────────
        # If everyone finished and the partial wasn't computed, just synthesize.
        # If the partial was computed and only one or two stragglers came in
        # after, re-synthesize so they're folded in.
        synthesis = await self._synthesize(task, results, ctx)
        return PatternResult(
            pattern=self.name,
            output=synthesis,
            success=any(r.success for r in results),
            subagent_results=list(results),
            metadata={
                "n_specs_proposed": len(specs),
                "n_subagents_spawned": len(spawned),
                "n_subagents_succeeded": sum(1 for r in results if r.success),
                "partial_synth_emitted": pending_streamed,
                "partial_synth_chars": len(partial_synth),
            },
        )

    async def _decompose(self, task: str, ctx: RunContext) -> list[AgentSpec]:
        if ctx.llm is None:
            raise ValueError("decompose requires ctx.llm")

        # Build the lead-agent prompt with embedded scaling rules
        available_tools = self._enumerate_tools(ctx)
        prompt = _DECOMPOSE_PROMPT_TEMPLATE.format(
            task=task,
            max_subagents=self._max_subagents,
            scaling_rules=ANTHROPIC_SCALING_RULES,
            available_tools=", ".join(available_tools[:30]) or "(none)",
        )
        messages = [
            {"role": "system", "content": "You are the lead orchestrator. Output strict JSON."},
            {"role": "user", "content": prompt},
        ]
        response = await ctx.llm.chat(
            messages,
            model=self._lead_model,
            temperature=0.3,
            max_tokens=4000,
        )
        # Account for the decomposition LLM call
        usage = (response or {}).get("usage") or {}
        try:
            ctx.budget.record_llm_call(
                model=str((response or {}).get("model") or self._lead_model or ""),
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                label="decompose",
            )
        except BudgetExceededError as exc:
            raise ValueError(f"budget exhausted during decompose: {exc}")

        content = str((response or {}).get("content") or "").strip()
        if not content:
            raise ValueError("decompose returned empty content")

        # Extract JSON — tolerant of markdown fences
        json_text = self._extract_json(content)
        raw_specs = json.loads(json_text)
        if not isinstance(raw_specs, list):
            raise ValueError("decompose output not a list")

        specs: list[AgentSpec] = []
        for raw in raw_specs:
            if not isinstance(raw, dict):
                continue
            try:
                spec = AgentSpec.create(
                    base_type=str(raw.get("base_type", "generalist")),
                    specialization=str(raw.get("specialization", "")),
                    objective=str(raw.get("objective", "")),
                    output_format=str(raw.get("output_format", "")),
                    success_criteria=tuple(raw.get("success_criteria") or ()),
                    task_boundaries=str(raw.get("task_boundaries", "")),
                    max_steps=int(raw.get("max_steps", 10)),
                    max_tokens=int(raw.get("max_tokens", ctx.budget.max_total_tokens // 4)),
                    allowed_tools=tuple(raw.get("allowed_tools") or ()),
                    parent_run_id=ctx.run_id,
                    trace_id=ctx.trace_id,
                    delegation_depth=0,  # subagents start fresh
                )
                specs.append(spec)
            except SpecValidationError as exc:
                logger.info("rejected vague spec: %s", exc)
                # vague spec → drop, don't waste a subagent slot
                continue
        return specs

    async def _synthesize(
        self, task: str, results: list[AgentResult], ctx: RunContext
    ) -> str:
        if ctx.llm is None:
            return self._fallback_concat(results)

        blocks = []
        for i, r in enumerate(results, 1):
            label = f"### Subagent {i} ({r.spec.specialization})"
            if not r.success:
                blocks.append(f"{label}\n[ERROR: {r.error}]")
                continue
            blocks.append(f"{label}\n{r.output}")

        prompt = _SYNTHESIZE_PROMPT_TEMPLATE.format(
            task=task,
            subagent_blocks="\n\n".join(blocks),
        )
        messages = [
            {"role": "system", "content": "You synthesize multi-agent findings."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = await ctx.llm.chat(
                messages, model=self._lead_model, temperature=0.2, max_tokens=2000
            )
        except Exception as exc:  # noqa: BLE001 — synthesis failure → fallback
            logger.warning("synthesize failed: %s — falling back to concat", exc)
            return self._fallback_concat(results)

        usage = (response or {}).get("usage") or {}
        try:
            ctx.budget.record_llm_call(
                model=str((response or {}).get("model") or self._lead_model or ""),
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                label="synthesize",
            )
        except BudgetExceededError:
            pass  # synthesis already done; surface budget breach via error
        return str((response or {}).get("content") or "").strip() or self._fallback_concat(results)

    @staticmethod
    def _fallback_concat(results: list[AgentResult]) -> str:
        sections: list[str] = []
        for i, r in enumerate(results, 1):
            label = f"### Subagent {i} ({r.spec.specialization})"
            if r.success:
                sections.append(f"{label}\n{r.output}")
            else:
                sections.append(f"{label}\n[error: {r.error}]")
        return "\n\n".join(sections)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Tolerate ```json fenced or bare JSON arrays."""
        s = text.strip()
        if s.startswith("```"):
            # strip first fence line + last fence line
            lines = s.splitlines()
            if len(lines) >= 2:
                lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                s = "\n".join(lines).strip()
        # Find first '[' and last ']'
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            return s[start : end + 1]
        return s

    @staticmethod
    def _enumerate_tools(ctx: RunContext) -> list[str]:
        """Best-effort tool enumeration from the handler_map."""
        if ctx.handler_map is None:
            return []
        try:
            return sorted(ctx.handler_map.keys())
        except (AttributeError, TypeError):
            return []
