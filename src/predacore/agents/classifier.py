"""LLMClassifier — autonomous pattern + runner selection via LLM.

The Orchestrator's rule-based router (`_select_pattern`, `_select_runner`)
handles obvious cases via substring match on the task. For ambiguous
tasks ("help me with my taxes", "what should I do today") the rules
guess wrong.

This classifier is the LLM-autonomous fallback:
  - rule-based fast path runs first (free, instant)
  - if rule confidence is low OR task length is moderate, classifier runs
  - returns (pattern, runner, reason) — orchestrator follows
  - decision cached in OutcomeStore so future similar tasks short-circuit

Cost: one cheap-model LLM call (~500ms, ~$0.001 with Haiku). Skipped
when caller passes an explicit override.

The classifier sees:
  - The task (truncated)
  - Available patterns + their use cases
  - Available runners + escalation criteria
  - Anthropic's scaling rules (so it knows N-subagent budget)
  - Recent similar prior runs (memory-grounded)

It outputs strict JSON:
  {"pattern": "<name>", "runner": "in_process|daf",
   "reason": "<≤200 chars>",
   "estimated_subagents": <int>, "estimated_seconds": <int>}
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .orchestrator import PatternName  # noqa: F401

logger = logging.getLogger(__name__)


def _default_pattern() -> "PatternName":
    """Lazy-import PatternName to break circular import with orchestrator."""
    from .orchestrator import PatternName

    return PatternName.AUTONOMOUS


@dataclass
class Classification:
    """One classifier verdict."""

    pattern: "PatternName" = field(default_factory=_default_pattern)
    runner: str = "in_process"            # "in_process" | "daf"
    reason: str = ""
    estimated_subagents: int = 1
    estimated_seconds: int = 30
    confidence: float = 0.5               # 0..1
    raw: str = ""                         # raw LLM response for debug


_CLASSIFIER_SYSTEM = """You are an autonomous orchestration router.
Given a user task, decide:
  1. Which pattern best handles it (one of the listed patterns)
  2. Which runner (in_process default; daf only when needed)
  3. Estimated subagent count (per Anthropic's scaling rules)
  4. Estimated wall-time in seconds

Output strict JSON only — no commentary outside the JSON."""


_CLASSIFIER_USER_TEMPLATE = """## Task
{task}

## Patterns
- autonomous              one agent loop with tools (default for trivial)
- routing                 classify input → pick one specialist (clear categories)
- prompt_chain            sequential N-step pipeline (caller supplies steps)
- parallelize             fan out N pre-decided subtasks, gather results
- orchestrator_workers    lead decomposes dynamically → subagents → synthesize
- evaluator_optimizer     generator + critic loop (iterative refinement)
- self_moa                same model, N samples, judge synthesizes (consensus)

## Runners
- in_process     async parallelism, fast, default. Use unless you need…
- daf            multi-process isolation. Use ONLY when:
                   - untrusted code execution (Flame skill from peer)
                   - long-running (≥60s) background work
                   - crash-prone tools (browser/Android automation that hangs)
                   - high-stakes deploy (kubectl/terraform/prod migrations)
                   - CPU-bound parallel work needing multi-core

## Scaling rules (per Anthropic, May 2025)
- Simple fact-finding:    1 agent,  3-10 tool calls
- Direct comparison:      2-4 subagents, 10-15 calls each
- Complex research:       10+ subagents

## Prior similar runs (if any)
{prior_runs}

## Output
Strict JSON:
{{
  "pattern": "<one of the 7 above>",
  "runner": "in_process" or "daf",
  "estimated_subagents": <int 1-15>,
  "estimated_seconds": <int>,
  "confidence": <0.0-1.0>,
  "reason": "<≤200 chars>"
}}"""


class LLMClassifier:
    """Hybrid pattern + runner classifier.

    Defaults to autonomous + in_process if classification fails or LLM
    unavailable — fail-safe fallback never blocks the orchestration.
    """

    def __init__(
        self,
        *,
        llm: Any,
        model: str | None = None,
        timeout_seconds: float = 6.0,
        rule_confidence_threshold: float = 0.7,
    ) -> None:
        self._llm = llm
        self._model = model
        self._timeout = timeout_seconds
        self._rule_threshold = rule_confidence_threshold

    async def classify(
        self,
        task: str,
        *,
        prior_runs: list[dict[str, Any]] | None = None,
    ) -> Classification:
        """Classify task → (pattern, runner, ...).

        Returns a Classification with confidence; caller decides whether
        to use it. Confidence < 0.6 typically means "fall back to rules."
        """
        if self._llm is None:
            return Classification(reason="no_llm", confidence=0.0)

        prompt = _CLASSIFIER_USER_TEMPLATE.format(
            task=task[:1500],
            prior_runs=self._format_prior(prior_runs or []),
        )
        try:
            import asyncio

            response = await asyncio.wait_for(
                self._llm.chat(
                    [
                        {"role": "system", "content": _CLASSIFIER_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    model=self._model,
                    temperature=0.0,
                    max_tokens=300,
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            logger.debug("classifier timeout after %.1fs", self._timeout)
            return Classification(reason="classifier_timeout", confidence=0.0)
        except Exception as exc:  # noqa: BLE001 — classifier failure → fall back
            logger.debug("classifier failed: %s", exc)
            return Classification(reason=f"classifier_error: {exc}", confidence=0.0)

        content = str((response or {}).get("content") or "").strip()
        return self._parse(content)

    @staticmethod
    def _format_prior(prior_runs: list[dict[str, Any]]) -> str:
        if not prior_runs:
            return "(none)"
        lines = []
        for r in prior_runs[:3]:
            meta = r.get("metadata") or {}
            pattern = meta.get("pattern", "?")
            success = meta.get("success", False)
            elapsed = meta.get("elapsed", 0)
            lines.append(
                f"- pattern={pattern}, success={success}, elapsed={elapsed:.0f}s"
            )
        return "\n".join(lines)

    def _parse(self, content: str) -> Classification:
        """Parse classifier JSON. Tolerant of fences."""
        from .orchestrator import PatternName  # lazy import to break cycle

        s = content.strip()
        if s.startswith("```"):
            lines = s.splitlines()[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return Classification(
                reason="classifier_malformed_json", confidence=0.0, raw=content
            )
        if not isinstance(obj, dict):
            return Classification(reason="classifier_not_dict", confidence=0.0, raw=content)

        # Pattern
        pattern_name = str(obj.get("pattern") or "autonomous").strip().lower()
        try:
            pattern = PatternName(pattern_name)
        except ValueError:
            pattern = PatternName.AUTONOMOUS

        # Runner
        runner = str(obj.get("runner") or "in_process").strip().lower()
        if runner not in {"in_process", "daf"}:
            runner = "in_process"

        return Classification(
            pattern=pattern,
            runner=runner,
            reason=str(obj.get("reason") or "")[:300],
            estimated_subagents=max(1, min(15, int(obj.get("estimated_subagents") or 1))),
            estimated_seconds=max(1, int(obj.get("estimated_seconds") or 30)),
            confidence=max(0.0, min(1.0, float(obj.get("confidence") or 0.5))),
            raw=content,
        )


# ── Rule-based fast path (free, instant) ──────────────────────────────


def rule_based_classify(task: str, *, autonomous_threshold_chars: int = 80) -> Classification:
    """Cheap pre-filter — returns high confidence on clear cases,
    low confidence (so caller falls back to LLM) on ambiguous ones.

    This mirrors `Orchestrator._select_pattern` heuristics so the
    orchestrator can skip the LLM call when the rules are confident.
    """
    from .orchestrator import PatternName  # lazy import to break cycle

    t = task.lower().strip()

    # Trivially short → autonomous, high confidence
    if len(task) < autonomous_threshold_chars:
        return Classification(
            pattern=PatternName.AUTONOMOUS,
            runner="in_process",
            reason="task too short for multi-agent",
            estimated_subagents=1,
            estimated_seconds=10,
            confidence=0.9,
        )

    # Strong DAF triggers
    daf_runner = "in_process"
    if any(k in t for k in ("kubectl apply", "terraform apply", "prod deploy")):
        daf_runner = "daf"
    if any(k in t for k in ("watch", "monitor", "every hour", "every minute")):
        daf_runner = "daf"
    if any(k in t for k in ("untrusted", "peer skill", "sandbox")):
        daf_runner = "daf"

    # Pattern triggers
    if any(k in t for k in ("important email", "legal", "contract", "production fix")):
        return Classification(
            pattern=PatternName.SELF_MOA,
            runner=daf_runner,
            reason="high-stakes consensus",
            estimated_subagents=5, estimated_seconds=60, confidence=0.85,
        )
    if any(k in t for k in ("write code that", "draft + review", "polish", "refine the")):
        return Classification(
            pattern=PatternName.EVALUATOR_OPTIMIZER,
            runner=daf_runner,
            reason="iterative refinement",
            estimated_subagents=4, estimated_seconds=45, confidence=0.85,
        )
    if any(k in t for k in ("compare", "list all", "research", "find every",
                             "analyze across", "summarize the", "draft a report")):
        return Classification(
            pattern=PatternName.ORCHESTRATOR_WORKERS,
            runner=daf_runner,
            reason="needs decomposition",
            estimated_subagents=4, estimated_seconds=60, confidence=0.85,
        )

    # Ambiguous — caller should fall back to LLM
    return Classification(
        pattern=PatternName.AUTONOMOUS,
        runner=daf_runner,
        reason="rule-based fallback",
        estimated_subagents=1, estimated_seconds=30, confidence=0.4,
    )


__all__ = [
    "Classification",
    "LLMClassifier",
    "rule_based_classify",
]
