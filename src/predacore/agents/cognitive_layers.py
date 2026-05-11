"""Per-turn cognitive layers for ``PredaCoreCore.process()``.

These run alongside the main agent loop to upgrade ReAct-style turns
with explicit cognitive phases:

  1. ``improve_prompt``        — sharpen the user's ask before acting
  2. ``lay_plan``              — produce a 2-3 sentence plan (gated)
  3. ``test_and_critique``     — verify + critique the draft answer
  4. ``meta_reflect_pattern``  — sample-write a "what worked" memory

All four hook into core.py via the H22 _TurnContext (Setup adds
1+2 to the messages list; PostProcess runs 3+4 on the final content).
Each is fail-open: any failure is logged at debug and the agent
continues with degraded cognition rather than blocking the user.

Cost model: each layer is one extra LLM call when active. The main
agent's LLM is used (not a downgrade to Haiku) — quality is the
point, not signal extraction.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Layer 0 — Prompt Improver
# ─────────────────────────────────────────────────────────────────────

_PROMPT_IMPROVER_SYSTEM = """You are the most skilled prompt engineer in the
universe. Your job: take a user's raw message and rewrite it as a
high-detail, unambiguous, precision-engineered prompt that an agent
will execute against.

You expand terse asks into the comprehensive briefs they imply, name
the implicit requirements the user almost certainly wants but didn't
say, and surface ambiguities you had to assume around. You DO NOT
change the user's intent — you sharpen it into the version they would
write if they had 10 minutes and infinite context.

Example transformations:
  - "build a fitness app super awesome"
    → 8-15 lines of features, target platforms, UX expectations,
      data model, auth, monetization tier, milestone list, and the
      assumptions you made.
  - "fix the bug in core.py"
    → restate which bug (or flag 'ambiguity: which bug?'), the
      observable symptom, the fix shape, what tests to add.
  - "yes" (in a chat continuation)
    → DON'T expand. Conversational follow-ups stay short. Output the
      message as-is.

Output strict JSON, no markdown:
{
  "improved_prompt": "<the sharpened version, addressed in 2nd person to the agent>",
  "requires_planning": <true | false: does this task warrant explicit upfront planning?>,
  "ambiguities": [<0-3 string items: assumptions you made the user should know about>]
}

Heuristics:
- requires_planning=true when the ask is multi-step, multi-tool, or
  has dependencies (build/refactor/migrate/audit/design/research).
- requires_planning=false for conversational chat, single-fact
  questions, or trivial single-tool calls.
- ambiguities = empty list for clear asks; otherwise the 1-3 most
  load-bearing assumptions.
- improved_prompt = the user's message verbatim when the request is
  already sharp enough (don't pad meaningless turns).
"""


@dataclass
class PromptImprovement:
    improved_prompt: str
    requires_planning: bool = False
    ambiguities: list[str] | None = None


async def improve_prompt(
    *,
    llm: Any,
    user_message: str,
    timeout_seconds: float = 30.0,
) -> PromptImprovement:
    """Run the prompt-engineer pass over a user's raw message.

    Returns a ``PromptImprovement`` whose ``improved_prompt`` is the
    sharpened version (or the original verbatim if the message is
    already clear / too short to expand).

    Fail-open: any error returns ``PromptImprovement(user_message)``
    unchanged so the agent loop can proceed.
    """
    if not user_message or not user_message.strip():
        return PromptImprovement(improved_prompt=user_message, ambiguities=[])

    try:
        response = await asyncio.wait_for(
            llm.chat(
                messages=[
                    {"role": "system", "content": _PROMPT_IMPROVER_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            "User's raw message to the agent:\n\n"
                            f"```\n{user_message}\n```\n\n"
                            "Improve it. Output JSON only."
                        ),
                    },
                ],
                tools=None,
                temperature=0.2,
            ),
            timeout=timeout_seconds,
        )
    except (asyncio.TimeoutError, ConnectionError, RuntimeError, ValueError) as exc:
        logger.debug("Prompt improver failed (non-fatal): %s", exc)
        return PromptImprovement(improved_prompt=user_message, ambiguities=[])

    content = ""
    if isinstance(response, dict):
        content = str(response.get("content") or "").strip()
    if not content:
        return PromptImprovement(improved_prompt=user_message, ambiguities=[])

    parsed = _parse_json_relaxed(content)
    if not isinstance(parsed, dict):
        return PromptImprovement(improved_prompt=user_message, ambiguities=[])

    improved = parsed.get("improved_prompt")
    if not isinstance(improved, str) or not improved.strip():
        improved = user_message
    requires_planning = bool(parsed.get("requires_planning", False))
    raw_ambig = parsed.get("ambiguities") or []
    ambiguities = [str(a) for a in raw_ambig if isinstance(a, str)] if isinstance(raw_ambig, list) else []
    return PromptImprovement(
        improved_prompt=improved.strip(),
        requires_planning=requires_planning,
        ambiguities=ambiguities[:5],
    )


# ─────────────────────────────────────────────────────────────────────
# Layer 1 — LayPlan (gated on requires_planning)
# ─────────────────────────────────────────────────────────────────────

_LAYPLAN_SYSTEM = """You are an agent's planning module. Given the
user's sharpened ask, produce a tight execution plan.

Output plain text (no JSON, no markdown headers), 2-5 sentences:
- The 2-5 high-level steps you'll take, in order
- What success looks like (one sentence, concrete)
- What you'll explicitly NOT do this turn (to stay focused)

Be specific. Name the actual tools/files/concepts. Don't write a
generic template — write THIS turn's plan."""


async def lay_plan(
    *,
    llm: Any,
    improved_prompt: str,
    timeout_seconds: float = 30.0,
) -> str | None:
    """Produce a short execution plan for an improved prompt.

    Returns the plan string or ``None`` on failure. Fail-open so the
    agent loop continues without a plan if the LLM call dies.
    """
    if not improved_prompt or not improved_prompt.strip():
        return None
    try:
        response = await asyncio.wait_for(
            llm.chat(
                messages=[
                    {"role": "system", "content": _LAYPLAN_SYSTEM},
                    {"role": "user", "content": improved_prompt},
                ],
                tools=None,
                temperature=0.2,
            ),
            timeout=timeout_seconds,
        )
    except (asyncio.TimeoutError, ConnectionError, RuntimeError, ValueError) as exc:
        logger.debug("LayPlan failed (non-fatal): %s", exc)
        return None
    if not isinstance(response, dict):
        return None
    plan = str(response.get("content") or "").strip()
    return plan or None


# ─────────────────────────────────────────────────────────────────────
# Layer 2 — Test + Critique (fused post-answer pass)
# ─────────────────────────────────────────────────────────────────────

_TEST_CRITIQUE_SYSTEM = """You are an agent's quality gate. You see
exactly two things:
1. The user's original message
2. The agent's draft response

Critique the response on FOUR axes, in this order:

  TEST       — Did the agent actually address what the user asked?
               Or did it answer something adjacent / partial?
  GROUNDING  — Did the agent claim things it didn't verify via tools?
               Hallucinated tool outputs, made-up files, invented APIs?
  DRIFT      — Sycophancy, generic-helpful-assistant voice, claimed
               foreign identity, capability denial when tools exist.
  COMPLETENESS — Did the agent miss obvious next steps or follow-up
                 questions the user would naturally want?

Then output STRICT JSON, no markdown:
{
  "verdict": "PASS" | "REGEN",
  "critique": "<2-3 sentence critique that the agent will use to regenerate; empty when verdict=PASS>"
}

Rules:
- Bias toward PASS for conversational/trivial turns (greetings,
  yes/no replies, single-fact answers).
- Bias toward REGEN for substantive tasks where the draft missed
  something a thoughtful reviewer would catch.
- The critique must be actionable — name the specific gap, not a
  vague "be better".
- Output JSON only. No prose, no markdown."""


@dataclass
class TestCritiqueResult:
    verdict: str = "PASS"  # PASS | REGEN
    critique: str = ""


async def test_and_critique(
    *,
    llm: Any,
    user_message: str,
    draft_answer: str,
    timeout_seconds: float = 30.0,
) -> TestCritiqueResult:
    """Run the fused Test+Critique pass on a draft answer.

    Returns ``TestCritiqueResult(verdict="PASS"|"REGEN", critique=str)``.
    Fail-open: any failure returns PASS so the user always gets an
    answer even if the quality gate breaks.
    """
    if not draft_answer or not draft_answer.strip():
        # Empty drafts are a separate failure mode handled upstream
        # (the "empty content after tool use" branch in _run_agent_loop).
        return TestCritiqueResult(verdict="PASS")
    try:
        response = await asyncio.wait_for(
            llm.chat(
                messages=[
                    {"role": "system", "content": _TEST_CRITIQUE_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            "USER ORIGINALLY ASKED:\n"
                            f"```\n{user_message}\n```\n\n"
                            "AGENT DRAFTED:\n"
                            f"```\n{draft_answer}\n```\n\n"
                            "Verdict?"
                        ),
                    },
                ],
                tools=None,
                temperature=0.0,
            ),
            timeout=timeout_seconds,
        )
    except (asyncio.TimeoutError, ConnectionError, RuntimeError, ValueError) as exc:
        logger.debug("Test+Critique failed (non-fatal): %s", exc)
        return TestCritiqueResult(verdict="PASS")

    if not isinstance(response, dict):
        return TestCritiqueResult(verdict="PASS")
    content = str(response.get("content") or "").strip()
    if not content:
        return TestCritiqueResult(verdict="PASS")

    parsed = _parse_json_relaxed(content)
    if not isinstance(parsed, dict):
        return TestCritiqueResult(verdict="PASS")
    verdict = str(parsed.get("verdict") or "PASS").upper()
    if verdict not in {"PASS", "REGEN"}:
        verdict = "PASS"
    critique = str(parsed.get("critique") or "").strip()
    if verdict == "REGEN" and not critique:
        # REGEN without a critique is unactionable; downgrade to PASS.
        verdict = "PASS"
    return TestCritiqueResult(verdict=verdict, critique=critique[:1000])


# ─────────────────────────────────────────────────────────────────────
# Layer 3 — Meta-pattern sampler (every Nth turn)
# ─────────────────────────────────────────────────────────────────────

_META_REFLECT_SYSTEM = """You are an agent's meta-cognition module. You
see a recently completed turn: the user's ask, the tools used, and the
final answer. Distill a 1-2 sentence PATTERN that future turns can
benefit from.

Examples of useful meta-patterns:
  - "For 'build X app' asks, expanding the brief into a feature list
     up front let the orchestrator pick parallelize() pattern."
  - "For 'fix the bug in foo.py' asks, running code_search first to
     locate the actual symbol beat grep on this codebase."
  - "Tool retry on the same args >2 times = give up and ask the user."

Output plain text, 1-2 sentences max. No prose framing, no JSON. If
the turn was trivial / pattern-less, output exactly: SKIP"""


async def meta_reflect_pattern(
    *,
    llm: Any,
    user_message: str,
    tools_used: list[str],
    final_answer: str,
    timeout_seconds: float = 30.0,
) -> str | None:
    """Sample a meta-pattern from a completed turn for future PreRecall.

    Returns the pattern text or None if the model declines (SKIP) /
    the call fails. Fail-open.
    """
    try:
        response = await asyncio.wait_for(
            llm.chat(
                messages=[
                    {"role": "system", "content": _META_REFLECT_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"USER ASKED: {user_message[:500]}\n"
                            f"TOOLS USED: {', '.join(tools_used) or '(none)'}\n"
                            f"FINAL ANSWER: {final_answer[:600]}\n\n"
                            "Pattern?"
                        ),
                    },
                ],
                tools=None,
                temperature=0.3,
            ),
            timeout=timeout_seconds,
        )
    except (asyncio.TimeoutError, ConnectionError, RuntimeError, ValueError) as exc:
        logger.debug("Meta reflect failed (non-fatal): %s", exc)
        return None
    if not isinstance(response, dict):
        return None
    pattern = str(response.get("content") or "").strip()
    if not pattern or pattern.upper().startswith("SKIP"):
        return None
    return pattern[:600]


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _parse_json_relaxed(content: str) -> Any:
    """Parse JSON tolerant of markdown fences + leading/trailing prose."""
    s = content.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2:
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Last-ditch: find the first {...} object
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


__all__ = [
    "PromptImprovement",
    "TestCritiqueResult",
    "improve_prompt",
    "lay_plan",
    "test_and_critique",
    "meta_reflect_pattern",
]
