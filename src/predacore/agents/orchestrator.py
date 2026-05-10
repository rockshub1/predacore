"""Orchestrator — single entry point that selects a Pattern + Runner.

This is the layer that replaces the 3 duplicated orchestration paths
(`engine._fan_out`, `collaboration.AgentTeam`, `daf_bridge.dispatch_multi_agent`).

Responsibilities:
  1. Classify the task → pick a Pattern
  2. Choose a Runner (in-process default, DAF for escalation)
  3. Construct the RunContext + budget + cancellation
  4. Hand off to the Pattern strategy
  5. Persist outcome to OutcomeStore (post-run)

Routing rules are workload-class-based, NOT agent-count-based:

  - Pattern.AUTONOMOUS               trivial, single-step, can be done by 1 agent
  - Pattern.ROUTING                  task class is clear from input
  - Pattern.PROMPT_CHAIN             user explicitly supplied a fixed pipeline
  - Pattern.PARALLELIZE              N independent subtasks pre-decided
  - Pattern.ORCHESTRATOR_WORKERS     non-trivial, lead must decompose
  - Pattern.EVALUATOR_OPTIMIZER      iterative refinement against criteria
  - Pattern.SELF_MOA                 high-stakes consensus output

Default pattern when classifier is uncertain: ORCHESTRATOR_WORKERS for
non-trivial tasks (≥30 words), AUTONOMOUS for trivial ones.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import uuid4

from .budget import CancellationToken, OrchestrationBudget
from .classifier import Classification, LLMClassifier, rule_based_classify
from .exceptions import OrchestrationError
from .patterns import (
    AutonomousPattern,
    EvaluatorOptimizerPattern,
    OrchestratorWorkersPattern,
    ParallelizePattern,
    Pattern,
    PatternResult,
    PromptChainPattern,
    RoutingPattern,
    SelfMoAPattern,
)
from .runners import DAFRunner, InProcessRunner, RunContext, Runner

logger = logging.getLogger(__name__)


class PatternName(str, Enum):
    AUTONOMOUS = "autonomous"
    ROUTING = "routing"
    PROMPT_CHAIN = "prompt_chain"
    PARALLELIZE = "parallelize"
    ORCHESTRATOR_WORKERS = "orchestrator_workers"
    EVALUATOR_OPTIMIZER = "evaluator_optimizer"
    SELF_MOA = "self_moa"


@dataclass
class OrchestratorConfig:
    """Knobs for routing decisions."""

    # If task is shorter than this in chars, prefer AUTONOMOUS
    autonomous_threshold_chars: int = 80

    # If task includes any of these phrases, hint orchestrator-workers
    decompose_hints: tuple[str, ...] = (
        "compare", "list all", "research", "find every",
        "analyze across", "summarize the", "draft a report",
    )

    # If task includes any of these, hint self-moa (high-stakes)
    consensus_hints: tuple[str, ...] = (
        "important email", "legal", "contract",
        "production deploy", "prod fix",
    )

    # If task includes any of these, hint evaluator-optimizer
    iterative_hints: tuple[str, ...] = (
        "write code that", "draft + review",
        "polish", "refine the",
    )

    # Default model used for lead (decomposition + synthesis)
    lead_model: str | None = None

    # Default routes for RoutingPattern (callers can override)
    default_routes: tuple = ()

    # Top-k for prior-run lookup that informs the classifier.
    # NOTE: this is metadata-only (not injected into LLM context). The
    # main user-facing recall (top_k=20, max_chars=80000) is configured
    # in core.py + retriever.py and is independent of this knob.
    pre_recall_top_k: int = 5

    # Top-k for per-subagent preloaded_memory during UPLOAD-CTX. Each
    # subagent gets this many rows pre-loaded into its system prompt to
    # skip the first recall RTT.
    preload_top_k: int = 8

    # LLM-autonomous routing — fall back to LLM classifier when rule-based
    # confidence is below this threshold. 0.7 = ambiguous tasks get an LLM
    # opinion; 1.0 = always use LLM (expensive); 0.0 = never use LLM (rules only).
    llm_classifier_threshold: float = 0.7

    # Cheap classifier model (~$0.001 per call). None = use the default
    # provider's smallest model.
    classifier_model: str | None = None


@dataclass
class OrchestrationResult:
    """Final result returned to the caller."""

    run_id: str
    pattern: str
    runner: str
    output: str
    success: bool
    error: str = ""
    elapsed_seconds: float = 0.0
    budget_snapshot: dict[str, Any] | None = None
    pattern_metadata: dict[str, Any] | None = None


class Orchestrator:
    """The single entry point for non-trivial agent work.

    Usage:
        orch = Orchestrator(llm=..., memory=..., handler_map=..., tool_executor=...)
        result = await orch.run(task="compare Anthropic vs OpenAI on safety")

    The orchestrator picks the pattern + runner + builds context. Callers
    can override pattern / runner via kwargs to bypass classification.
    """

    def __init__(
        self,
        *,
        llm: Any = None,
        memory: Any = None,
        handler_map: Any = None,
        tool_executor: Any = None,
        outcome_store: Any = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.handler_map = handler_map
        self.tool_executor = tool_executor
        self.outcome_store = outcome_store
        self.config = config or OrchestratorConfig()

    # ── Public entry point ────────────────────────────────────────────

    async def run(
        self,
        task: str,
        *,
        user_id: str = "default",
        session_id: str = "",
        budget: OrchestrationBudget | None = None,
        cancellation: CancellationToken | None = None,
        # Manual overrides (skip classification)
        pattern: PatternName | str | None = None,
        runner: Runner | str | None = None,
        # Optional pre-built pattern instance for advanced callers
        pattern_instance: Pattern | None = None,
    ) -> OrchestrationResult:
        run_id = f"orch_{uuid4().hex[:12]}"
        trace_id = f"trace_{uuid4().hex[:8]}"
        ctx = RunContext(
            run_id=run_id,
            team_id=run_id,  # team scope = run scope by default
            trace_id=trace_id,
            user_id=user_id,
            session_id=session_id,
            llm=self.llm,
            memory=self.memory,
            tool_executor=self.tool_executor,
            handler_map=self.handler_map,
            budget=budget or OrchestrationBudget(),
            cancellation=cancellation or CancellationToken(),
            task=task,
        )

        # Pre-recall: hint the lead with similar prior runs (memory-grounded)
        prior_runs = await self._pre_recall(task, ctx)

        # ── Autonomous routing (rule fast-path → LLM fallback) ────────
        classification = await self._classify(task, prior_runs, override=pattern)

        # Choose pattern (override > classifier > rule fallback)
        chosen_pattern = pattern_instance or self._select_pattern(
            task, prior_runs, classification=classification, override=pattern
        )
        ctx.pattern_name = chosen_pattern.name

        # Choose runner (override > classifier > rule fallback)
        chosen_runner = self._select_runner(
            task, classification=classification, override=runner
        )
        ctx.runner_name = chosen_runner.name

        logger.info(
            "Orchestrator run=%s pattern=%s runner=%s task=%r",
            run_id, ctx.pattern_name, ctx.runner_name,
            task[:80] + ("…" if len(task) > 80 else ""),
        )

        # Execute
        started = time.monotonic()
        try:
            pattern_result = await chosen_pattern.execute(task, ctx, chosen_runner)
        except OrchestrationError as exc:
            elapsed = time.monotonic() - started
            await self._record_outcome(
                run_id, task, ctx, success=False, error=str(exc), elapsed=elapsed
            )
            return OrchestrationResult(
                run_id=run_id,
                pattern=ctx.pattern_name,
                runner=ctx.runner_name,
                output="",
                success=False,
                error=str(exc),
                elapsed_seconds=elapsed,
                budget_snapshot=ctx.budget.snapshot(),
            )
        except Exception as exc:  # noqa: BLE001 — orchestrator boundary
            elapsed = time.monotonic() - started
            logger.exception("Orchestrator unexpected failure: %s", exc)
            await self._record_outcome(
                run_id, task, ctx, success=False, error=str(exc), elapsed=elapsed
            )
            return OrchestrationResult(
                run_id=run_id,
                pattern=ctx.pattern_name,
                runner=ctx.runner_name,
                output="",
                success=False,
                error=str(exc),
                elapsed_seconds=elapsed,
                budget_snapshot=ctx.budget.snapshot(),
            )

        elapsed = time.monotonic() - started
        await self._record_outcome(
            run_id, task, ctx, success=pattern_result.success,
            error=pattern_result.error, elapsed=elapsed,
            output=pattern_result.output,
            metadata=pattern_result.metadata,
        )
        return OrchestrationResult(
            run_id=run_id,
            pattern=ctx.pattern_name,
            runner=ctx.runner_name,
            output=pattern_result.output,
            success=pattern_result.success,
            error=pattern_result.error,
            elapsed_seconds=elapsed,
            budget_snapshot=ctx.budget.snapshot(),
            pattern_metadata=dict(pattern_result.metadata or {}),
        )

    # ── Autonomous routing ────────────────────────────────────────────

    async def _classify(
        self,
        task: str,
        prior_runs: list[dict[str, Any]],
        *,
        override: PatternName | str | None,
    ) -> Classification:
        """Hybrid classifier: rule fast-path → LLM fallback for ambiguous.

        Skipped entirely when the caller passed an explicit pattern
        override.
        """
        if override is not None:
            return Classification(reason="explicit_override", confidence=1.0)

        # 1. Rule-based fast path
        rule_verdict = rule_based_classify(
            task,
            autonomous_threshold_chars=self.config.autonomous_threshold_chars,
        )

        # 2. If rules confident enough, accept and return
        if rule_verdict.confidence >= self.config.llm_classifier_threshold:
            return rule_verdict

        # 3. Ambiguous — call the LLM classifier
        if self.llm is None:
            return rule_verdict

        classifier = LLMClassifier(llm=self.llm, model=self.config.classifier_model)
        llm_verdict = await classifier.classify(task, prior_runs=prior_runs)

        # 4. If LLM confidence is meaningful, prefer it
        if llm_verdict.confidence >= 0.5:
            logger.debug(
                "classifier: LLM picked pattern=%s runner=%s (rule was %s, conf %.2f)",
                llm_verdict.pattern.value, llm_verdict.runner,
                rule_verdict.pattern.value, rule_verdict.confidence,
            )
            return llm_verdict

        # 5. LLM also unsure — fall back to rules
        return rule_verdict

    # ── Pattern selection ─────────────────────────────────────────────

    def _select_pattern(
        self,
        task: str,
        prior_runs: list[dict[str, Any]],
        *,
        classification: Classification | None = None,
        override: PatternName | str | None = None,
    ) -> Pattern:
        """Pick a pattern. Override > classifier > prior-run hint > rule-based."""
        if override is not None:
            name = override.value if isinstance(override, PatternName) else str(override)
            try:
                return self._build_pattern(PatternName(name))
            except ValueError:
                raise OrchestrationError(f"unknown pattern: {name!r}")

        # Classifier (LLM or rule) verdict — prefer when confident
        if classification is not None and classification.confidence >= 0.5:
            return self._build_pattern(classification.pattern)

        # Prior-run hint: if we've solved a very similar task before with
        # a specific pattern, prefer it.
        if prior_runs:
            hinted = self._extract_pattern_hint(prior_runs)
            if hinted is not None:
                return self._build_pattern(hinted)

        # Rule-based fall-through
        t = task.lower().strip()
        if len(task) < self.config.autonomous_threshold_chars:
            return self._build_pattern(PatternName.AUTONOMOUS)
        if any(h in t for h in self.config.consensus_hints):
            return self._build_pattern(PatternName.SELF_MOA)
        if any(h in t for h in self.config.iterative_hints):
            return self._build_pattern(PatternName.EVALUATOR_OPTIMIZER)
        if any(h in t for h in self.config.decompose_hints):
            return self._build_pattern(PatternName.ORCHESTRATOR_WORKERS)
        # Default: orchestrator-workers for non-trivial questions
        return self._build_pattern(PatternName.ORCHESTRATOR_WORKERS)

    def _build_pattern(self, name: PatternName) -> Pattern:
        if name == PatternName.AUTONOMOUS:
            return AutonomousPattern()
        if name == PatternName.ORCHESTRATOR_WORKERS:
            return OrchestratorWorkersPattern(lead_model=self.config.lead_model)
        if name == PatternName.SELF_MOA:
            return SelfMoAPattern()
        if name == PatternName.EVALUATOR_OPTIMIZER:
            return EvaluatorOptimizerPattern()
        if name == PatternName.PARALLELIZE:
            return ParallelizePattern()  # default 2-way; caller may inject
        if name == PatternName.ROUTING:
            if not self.config.default_routes:
                # Fall back to autonomous when no routes configured
                return AutonomousPattern()
            return RoutingPattern(self.config.default_routes)
        if name == PatternName.PROMPT_CHAIN:
            # PromptChain requires explicit step list; fall back if none
            return AutonomousPattern()
        return AutonomousPattern()

    def _extract_pattern_hint(self, prior_runs: list[dict[str, Any]]) -> PatternName | None:
        """If prior similar runs converged on a pattern, suggest it.

        Looks at the metadata of pre-recalled task_outcome memories.
        Conservative: requires ≥2 prior runs of the same pattern + success.
        """
        counts: dict[str, int] = {}
        for run in prior_runs[:5]:
            meta = run.get("metadata") or {}
            if isinstance(meta, dict) and meta.get("success"):
                p = meta.get("pattern")
                if isinstance(p, str):
                    counts[p] = counts.get(p, 0) + 1
        if not counts:
            return None
        winner, n = max(counts.items(), key=lambda kv: kv[1])
        if n >= 2:
            try:
                return PatternName(winner)
            except ValueError:
                return None
        return None

    # ── Runner selection ──────────────────────────────────────────────

    def _select_runner(
        self,
        task: str,
        *,
        classification: Classification | None = None,
        override: Runner | str | None = None,
    ) -> Runner:
        """Choose a runner. Override > classifier > rule-based.

        DAF when:
          - explicit override
          - classifier picked daf with confidence ≥ 0.5
            (untrusted skill, long-running, deploy, crash-prone)
        Otherwise InProcessRunner — parallelism for I/O without spawn cost.
        """
        if isinstance(override, Runner):
            return override
        if isinstance(override, str):
            if override == "daf":
                return DAFRunner()
            return InProcessRunner()
        if classification is not None and classification.confidence >= 0.5:
            return DAFRunner() if classification.runner == "daf" else InProcessRunner()
        return InProcessRunner()

    # ── Memory-grounded extras ────────────────────────────────────────

    async def _pre_recall(self, task: str, ctx: RunContext) -> list[dict[str, Any]]:
        """Search memory for similar prior task_outcomes — informs the
        pattern classifier and pre-loads context for subagents.

        Best-effort: if memory unavailable or recall fails, return [].
        """
        if ctx.memory is None:
            return []
        try:
            results = await ctx.memory.recall(
                query=task,
                user_id=ctx.user_id,
                top_k=self.config.pre_recall_top_k,
                memory_types=["task_outcome", "decision", "fact"],
                min_importance=2,
            )
        except Exception as exc:  # noqa: BLE001 — recall best-effort
            logger.debug("pre_recall failed: %s", exc)
            return []
        return [r for r, _score in (results or [])]

    async def _record_outcome(
        self,
        run_id: str,
        task: str,
        ctx: RunContext,
        *,
        success: bool,
        error: str = "",
        elapsed: float = 0.0,
        output: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist outcome to memory + OutcomeStore.

        The next orchestration's _pre_recall will see this — orchestration
        teaches itself which patterns work for which task classes.
        """
        if ctx.memory is None:
            return
        try:
            content = (
                f"Task: {task[:300]}\n"
                f"Pattern: {ctx.pattern_name} via {ctx.runner_name}\n"
                f"Outcome: {'success' if success else f'failure: {error}'}\n"
                f"Elapsed: {elapsed:.1f}s, "
                f"tokens={ctx.budget.total_tokens}, "
                f"dollars=${ctx.budget.used_dollars:.3f}, "
                f"subagents={ctx.budget.spawned_subagents}"
            )
            await ctx.memory.store(
                content=content,
                memory_type="task_outcome",
                importance=4 if success else 3,
                user_id=ctx.user_id,
                session_id=ctx.session_id,
                memory_scope="user",
                tags=["orchestrator", ctx.pattern_name, ctx.runner_name],
                metadata={
                    "run_id": run_id,
                    "trace_id": ctx.trace_id,
                    "pattern": ctx.pattern_name,
                    "runner": ctx.runner_name,
                    "success": success,
                    "elapsed": elapsed,
                    "budget": ctx.budget.snapshot(),
                    **(metadata or {}),
                },
                trust_source="claude_inferred",
                confidence=0.8,
            )
        except Exception as exc:  # noqa: BLE001 — outcome record best-effort
            logger.debug("outcome record failed: %s", exc)

        # OutcomeStore is structured + queryable; complementary to memory
        if self.outcome_store is not None:
            try:
                self.outcome_store.record(  # type: ignore[attr-defined]
                    user_id=ctx.user_id,
                    session_id=ctx.session_id,
                    user_message=task,
                    response_summary=output[:2000] if output else error[:2000],
                    success=success,
                    metadata={
                        "run_id": run_id,
                        "pattern": ctx.pattern_name,
                        "runner": ctx.runner_name,
                        "elapsed_seconds": elapsed,
                        "budget": ctx.budget.snapshot(),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("OutcomeStore record failed: %s", exc)
