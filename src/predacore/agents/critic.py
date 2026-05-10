"""CriticGate — separate critic LLM reviews high-stakes actions before commit.

The critic is independent of the agent that proposed the action. It
sees the planned tool call, the current orchestration goal, and the
relevant memory context. It returns APPROVE / REVISE / ABORT.

When to gate:
  - destructive tools: write_file, run_command, execute_code, memory_delete
  - cross-domain bridges: openclaw_delegate (proxy to remote LLM)
  - browser actions on sensitive sites (banking, email, billing)
  - channel sends (telegram/email — once a message is out, it's out)

Cost: ~500 ms + ~$0.001 per protected action with a small/cheap model.
Cheap insurance against the LLM going off-script on dangerous tools.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .exceptions import CriticVetoError

logger = logging.getLogger(__name__)


# Tools that always go through the critic gate. Override per-call if needed.
DEFAULT_HIGH_STAKES_TOOLS: frozenset[str] = frozenset(
    {
        "write_file",
        "run_command",
        "execute_code",
        "python_exec",
        "openclaw_delegate",
        "memory_delete",
        "channel_send",
        "twilio_send",
        "slack_send",
        "discord_send",
        "email_send",
    }
)


@dataclass
class CriticReview:
    """One review verdict from the critic."""

    verdict: str = "APPROVE"          # APPROVE | REVISE | ABORT
    score: int = 0                     # 0-10
    reason: str = ""                   # why
    revision_hint: str = ""            # if REVISE, what to change
    raw_response: str = field(default="", repr=False)


_CRITIC_SYSTEM_PROMPT = """You are an independent safety critic for an
agent system. Your job is to review one tool call BEFORE it executes
and decide whether it's safe to proceed.

You see:
  - The orchestration goal (the user's original task)
  - The tool name and full arguments
  - The agent's reasoning (if available)
  - Relevant memory context

Output strict JSON:
  {"verdict": "APPROVE" | "REVISE" | "ABORT",
   "score": <0-10>,
   "reason": "<≤200 chars>",
   "revision_hint": "<≤300 chars when verdict=REVISE>"}

Heuristics:
  - APPROVE if the action plainly serves the goal and is reversible
  - REVISE if the action mostly serves the goal but has scope creep,
    overly broad path, missing dry-run, or hostile-input pattern
  - ABORT if the action could cause data loss, exfiltrate secrets,
    write outside the project root, or contradicts the user's task

When in doubt, REVISE rather than ABORT — let the agent try again."""


_REVIEW_PROMPT_TEMPLATE = """Goal: {goal}

Tool to be called: {tool_name}
Arguments: {tool_args}

Agent's reasoning (if any):
{reasoning}

Relevant memory:
{memory}

Output your verdict as JSON only."""


class CriticGate:
    """Stateless gate — call review() before each high-stakes tool call."""

    def __init__(
        self,
        *,
        llm: Any,
        model: str | None = None,
        high_stakes_tools: frozenset[str] = DEFAULT_HIGH_STAKES_TOOLS,
        timeout_seconds: float = 8.0,
    ) -> None:
        self._llm = llm
        self._model = model  # let LLMInterface pick if None
        self._high_stakes = high_stakes_tools
        self._timeout = timeout_seconds

    def is_high_stakes(self, tool_name: str) -> bool:
        return tool_name in self._high_stakes

    async def review(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        goal: str,
        reasoning: str = "",
        memory_context: str = "",
    ) -> CriticReview:
        """Review one tool call. Returns a CriticReview verdict.

        Caller is responsible for branching on verdict:
          APPROVE  → execute
          REVISE   → raise CriticVetoError(revision_hint=...) to bounce
                      back to DELIBERATE with the hint
          ABORT    → raise CriticVetoError to abort the orchestration
        """
        if self._llm is None:
            # No critic available — fail open with a warning
            logger.warning("CriticGate has no llm; defaulting to APPROVE for %s", tool_name)
            return CriticReview(verdict="APPROVE", score=5, reason="critic-unavailable")

        prompt = _REVIEW_PROMPT_TEMPLATE.format(
            goal=goal[:1000] or "(no goal)",
            tool_name=tool_name,
            tool_args=json.dumps(tool_args, ensure_ascii=False)[:1500],
            reasoning=reasoning[:1500] or "(none)",
            memory=memory_context[:1500] or "(none)",
        )
        try:
            import asyncio

            response = await asyncio.wait_for(
                self._llm.chat(
                    [
                        {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    model=self._model,
                    temperature=0.0,
                    max_tokens=300,
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("CriticGate timed out reviewing %s — defaulting to APPROVE", tool_name)
            return CriticReview(verdict="APPROVE", score=5, reason="critic-timeout")
        except Exception as exc:  # noqa: BLE001 — critic failure → fail open
            logger.warning("CriticGate review failed: %s — defaulting to APPROVE", exc)
            return CriticReview(verdict="APPROVE", score=5, reason="critic-error")

        content = str((response or {}).get("content") or "").strip()
        review = self._parse(content)
        review.raw_response = content
        return review

    def enforce(self, review: CriticReview, *, tool_name: str) -> None:
        """Raise CriticVetoError if not APPROVE."""
        if review.verdict == "APPROVE":
            return
        if review.verdict == "REVISE":
            raise CriticVetoError(
                f"critic asked to REVISE {tool_name}: {review.reason}",
                revision_hint=review.revision_hint or review.reason,
            )
        # ABORT
        raise CriticVetoError(
            f"critic ABORTED {tool_name}: {review.reason}",
            revision_hint=None,
        )

    @staticmethod
    def _parse(content: str) -> CriticReview:
        """Parse JSON verdict; tolerant of markdown fences."""
        s = content.strip()
        if s.startswith("```"):
            lines = s.splitlines()
            if len(lines) >= 2:
                lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                s = "\n".join(lines).strip()
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            # Malformed — treat as APPROVE (fail open) so we don't block
            # legitimate work on a misbehaving critic
            return CriticReview(verdict="APPROVE", score=5, reason="critic-malformed")
        if not isinstance(obj, dict):
            return CriticReview(verdict="APPROVE", score=5, reason="critic-malformed")
        verdict = str(obj.get("verdict", "APPROVE")).upper()
        if verdict not in {"APPROVE", "REVISE", "ABORT"}:
            verdict = "APPROVE"
        return CriticReview(
            verdict=verdict,
            score=int(obj.get("score") or 5),
            reason=str(obj.get("reason") or "")[:300],
            revision_hint=str(obj.get("revision_hint") or "")[:500],
        )
