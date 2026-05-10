"""SelfMoAPattern — Self-MoA: same model, multiple samples, judge synthesizes.

Per Li et al., "Rethinking Mixture-of-Agents" (Feb 2025, ICLR):
intra-model diversity (multiple samples from one strong model) beats
inter-model diversity (mixing different LLMs) by ~6.6% on AlpacaEval 2.0.

When to use: high-stakes outputs where consensus matters.
  - Hard reasoning that benefits from sampling
  - Critical writing (SOPs, contracts, key emails)
  - Verification-heavy tasks

Cost: N samples + 1 judge call against the SAME provider. Cheaper +
better than mixing N providers. The judge can be a smaller cheap model
without quality loss.

KNOWN FAILURE MODE: Self-MoA AMPLIFIES systematic errors. When the model
has a sycophancy bias (RLHF-induced agreement-with-user), all N samples
share the bias and the judge consensus surfaces the wrong-answer-with-
HIGH-confidence. The sycophancy guard below reads recent SycophancyAxis
probe results; when sycophancy_rate > threshold, the judge prompt is
rewritten to apply extra skepticism instead of trusting agreement.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from ..exceptions import BudgetExceededError
from ..runners.base import AgentResult, RunContext, Runner
from ..spec import AgentSpec
from .base import Pattern, PatternResult

logger = logging.getLogger(__name__)


# Sycophancy guard threshold. Above this, the judge prompt is rewritten
# to apply skepticism. 0.30 = "agent agreed with planted falsehood on
# >30% of leading-question canaries in recent probes" — meaningful bias.
_SYCOPHANCY_GUARD_THRESHOLD = float(
    os.environ.get("PREDACORE_SELF_MOA_SYCOPHANCY_THRESHOLD", "0.30")
)
# Maximum age of a sycophancy probe result before we treat it as stale
# (ignore — fall back to the unguarded judge prompt). 7 days is generous;
# probes default to 2-hour cadence so a fresh result is always available
# in steady-state.
_SYCOPHANCY_RESULT_MAX_AGE_SECONDS = 7 * 24 * 3600


_JUDGE_PROMPT_TEMPLATE = """You are a judge selecting + synthesizing the best
answer from multiple independent attempts to the same task.

The original task:
<<<TASK
{task}
END_TASK>>>

{n} independent attempts follow. They came from the same model with
different sampling, so any one could be best.

{samples}

Synthesize a final answer. You may:
- pick one attempt verbatim if it is clearly best
- merge insights from multiple attempts
- correct factual errors any attempt makes (cite which one was wrong)
- note disagreement where attempts differ on a verifiable claim

Output the final answer only. No meta-commentary about the judging."""


_JUDGE_PROMPT_SKEPTICAL_TEMPLATE = """You are a SKEPTICAL judge selecting +
synthesizing an answer from multiple attempts to the same task.

**ACTIVE SYCOPHANCY ALERT.** Recent out-of-band probes measured the
agent (which generated all attempts below) as sycophancy-prone in this
period — meaning all N samples likely share the SAME biased reasoning.
Self-MoA's normal "trust the consensus" assumption FAILS here because
the consensus is a high-confidence mistake.

Apply these rules:
- **Disagreement is signal, not noise.** If 1 of N attempts disagrees
  with the rest, examine its reasoning carefully — it may be correct
  while the agreeing majority shares a baked-in bias.
- **Verify each factual claim independently.** Don't accept a claim
  just because multiple attempts assert it.
- **Reject flattery and echo.** If attempts uncritically agree with
  the user's framing in the original task, downgrade them.
- **Prefer concrete over hedged.** Sycophantic outputs hedge to please;
  real answers commit.
- **When in doubt, output the SHORTEST high-confidence answer** rather
  than synthesizing a longer pleasing one. Length is sycophancy's
  favorite hiding place.

The original task:
<<<TASK
{task}
END_TASK>>>

{n} attempts follow.

{samples}

Output the final answer only. No meta-commentary about judging or
sycophancy."""


def _load_recent_sycophancy_rate(audit_dir: Path) -> float | None:
    """Read the most recent SycophancyAxis probe result.

    Returns the sycophancy_rate (0.0 to 1.0) from the latest entry in
    ``<audit_dir>/sycophancy_probe.jsonl``, or None if no probe has run
    yet OR the most recent result is older than the staleness window.

    None means "no signal — use the default judge prompt." This is the
    cold-start policy: trust consensus until probes have run.
    """
    path = audit_dir / "sycophancy_probe.jsonl"
    if not path.exists():
        return None
    try:
        # JSONL — read the file, take the last non-empty line
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if not isinstance(obj, dict):
                continue
            # Staleness check
            ts = obj.get("timestamp")
            if isinstance(ts, (int, float)):
                import time as _time
                if _time.time() - float(ts) > _SYCOPHANCY_RESULT_MAX_AGE_SECONDS:
                    return None
            rate = obj.get("sycophancy_rate")
            if isinstance(rate, (int, float)):
                return float(rate)
            return None
        return None
    except OSError:
        return None


class SelfMoAPattern(Pattern):
    name = "self_moa"

    def __init__(
        self,
        *,
        n_samples: int = 5,
        sample_temperature: float = 0.7,
        sample_model: str | None = None,
        judge_model: str | None = None,
    ) -> None:
        self._n_samples = max(2, min(7, n_samples))
        self._temp = sample_temperature
        self._sample_model = sample_model
        self._judge_model = judge_model or sample_model

    async def execute(self, task: str, ctx: RunContext, runner: Runner) -> PatternResult:
        if ctx.llm is None:
            return PatternResult(
                pattern=self.name, output="", success=False,
                error="self_moa requires ctx.llm",
            )

        # ── 1. Generate N samples in parallel ──────────────────────────
        async def _sample(i: int) -> dict[str, Any]:
            spec = AgentSpec.create(
                base_type="generalist",
                specialization=f"sample {i+1} of {self._n_samples}",
                objective=f"Answer the task with care: {task}",
                output_format="natural language answer with reasoning",
                success_criteria=("answer is on-topic", "reasoning is included"),
                allowed_tools=("memory_recall",),
                max_steps=8,
                max_tokens=ctx.budget.max_total_tokens // (self._n_samples + 2),
                parent_run_id=ctx.run_id,
                trace_id=f"{ctx.trace_id}-s{i}",
            )
            return {"spec": spec, "result": await runner.run_spec(spec, ctx)}

        for _ in range(self._n_samples):
            try:
                ctx.budget.record_subagent_spawn()
            except BudgetExceededError:
                break
        bundle = await asyncio.gather(
            *[_sample(i) for i in range(self._n_samples)], return_exceptions=False
        )
        results: list[AgentResult] = [b["result"] for b in bundle]
        ok = [r for r in results if r.success and r.output.strip()]
        if not ok:
            return PatternResult(
                pattern=self.name, output="", success=False,
                error="all self-MoA samples failed",
                subagent_results=results,
            )

        # ── 2. Judge ───────────────────────────────────────────────────
        sample_blocks = []
        for i, r in enumerate(ok, 1):
            sample_blocks.append(f"### Attempt {i}\n{r.output}")

        # Sycophancy guard: when recent probes show elevated sycophancy,
        # swap to the skeptical judge prompt so consensus alone doesn't
        # win — judge must verify claims and prefer disagreeing samples.
        # Cold-start (no probe history): use default prompt.
        audit_dir = Path(
            os.environ.get(
                "PREDACORE_SAFETY_PROBE_AUDIT_DIR",
                str(Path.home() / ".predacore" / "audit" / "safety_probes"),
            )
        )
        recent_sycophancy = _load_recent_sycophancy_rate(audit_dir)
        sycophancy_guard_active = (
            recent_sycophancy is not None
            and recent_sycophancy > _SYCOPHANCY_GUARD_THRESHOLD
        )
        if sycophancy_guard_active:
            judge_template = _JUDGE_PROMPT_SKEPTICAL_TEMPLATE
            logger.info(
                "self_moa: sycophancy guard active (rate=%.2f > %.2f) — "
                "skeptical judge prompt enabled",
                recent_sycophancy, _SYCOPHANCY_GUARD_THRESHOLD,
            )
        else:
            judge_template = _JUDGE_PROMPT_TEMPLATE

        judge_prompt = judge_template.format(
            task=task,
            n=len(ok),
            samples="\n\n".join(sample_blocks),
        )
        try:
            response = await ctx.llm.chat(
                [
                    {"role": "system", "content": "You are an impartial judge of LLM outputs."},
                    {"role": "user", "content": judge_prompt},
                ],
                model=self._judge_model,
                temperature=0.0,
                max_tokens=2000,
            )
        except Exception as exc:  # noqa: BLE001 — judge failure → fall back
            logger.warning("self_moa judge failed: %s — falling back to first sample", exc)
            return PatternResult(
                pattern=self.name,
                output=ok[0].output,
                success=True,
                subagent_results=results,
                metadata={"judged": False},
            )

        usage = (response or {}).get("usage") or {}
        try:
            ctx.budget.record_llm_call(
                model=str((response or {}).get("model") or self._judge_model or ""),
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                label="self_moa_judge",
            )
        except BudgetExceededError:
            pass

        output = str((response or {}).get("content") or "").strip() or ok[0].output
        return PatternResult(
            pattern=self.name,
            output=output,
            success=True,
            subagent_results=results,
            metadata={
                "judged": True,
                "n_samples": len(ok),
                "sycophancy_guard_active": sycophancy_guard_active,
                "sycophancy_rate": recent_sycophancy,
            },
        )
