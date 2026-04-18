"""
BELIEFS.md crystallization state machine.

This is the legible record of what the agent currently believes and how
hardened each belief is. The state ladder is:

    observation → working_theory → tested → committed

Plus a demoted state for beliefs that got falsified.

Promotion is driven by evidence accumulation:
- 2+ consistent observations → working_theory
- 5+ consistent cases → tested
- 8+ cases AND explicit confirmation → committed
- Any contradiction → demoted (with reason logged)

Storage is a JSON sidecar (beliefs.json) for machine state, plus a
rendered BELIEFS.md that the agent sees in its system prompt and that
a human can read directly. Both files live in the agent workspace.

Usage:
    store = BeliefStore(workspace)
    bid = store.add_observation(
        text="Tier-by-severity audits work well for multi-issue findings",
        session_id="abc",
        falsification="Would update if a flat list produced better outcomes",
    )
    store.record_evidence(bid, session_id="def")  # auto-promotes if threshold hit
    store.demote(bid, reason="...", session_id="ghi")  # explicit falsification
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Promotion thresholds (evidence count required to reach each state)
_PROMOTE_TO_WORKING_THEORY = 2
_PROMOTE_TO_TESTED = 5
_PROMOTE_TO_COMMITTED = 8


class BeliefState(str, Enum):
    OBSERVATION = "observation"
    WORKING_THEORY = "working_theory"
    TESTED = "tested"
    COMMITTED = "committed"
    DEMOTED = "demoted"

    @property
    def level(self) -> int:
        return {
            "observation": 0,
            "working_theory": 1,
            "tested": 2,
            "committed": 3,
            "demoted": -1,
        }[self.value]


@dataclass
class Belief:
    id: str
    text: str
    state: str  # stored as string for JSON round-trip
    created_at: str
    last_updated: str
    evidence_count: int = 1
    triggering_sessions: list[str] = field(default_factory=list)
    falsification: str = ""
    notes: list[str] = field(default_factory=list)  # timestamped notes (e.g., demotion reasons)

    def short_title(self) -> str:
        first_line = self.text.strip().split("\n")[0]
        return first_line[:80] + ("..." if len(first_line) > 80 else "")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Belief:
        # Defensive: keep only known fields
        known = {k: data.get(k) for k in cls.__dataclass_fields__}
        # Fix up defaults for missing collections
        known["triggering_sessions"] = known.get("triggering_sessions") or []
        known["notes"] = known.get("notes") or []
        known["evidence_count"] = int(known.get("evidence_count") or 1)
        return cls(**known)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id(prefix: str = "B") -> str:
    """Generate a short sortable belief ID like B-20260413-a1b2."""
    from uuid import uuid4
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = uuid4().hex[:4]
    return f"{prefix}-{stamp}-{suffix}"


class BeliefStore:
    """
    Crystallization state machine for agent beliefs.

    JSON is canonical; BELIEFS.md is a rendered view. Every mutation
    updates both atomically.
    """

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.json_path = self.workspace / "beliefs.json"
        self.md_path = self.workspace / "BELIEFS.md"
        self._beliefs: dict[str, Belief] = {}
        self._load()

    # ── Load / save ───────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.json_path.exists():
            return
        try:
            raw = json.loads(self.json_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "beliefs" in raw:
                for item in raw["beliefs"]:
                    try:
                        b = Belief.from_dict(item)
                        self._beliefs[b.id] = b
                    except (TypeError, ValueError, KeyError) as exc:
                        logger.debug("Skipping malformed belief entry: %s", exc)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("BeliefStore load failed: %s", exc)

    def _save(self) -> None:
        """Atomic save: write temp file, then rename. Re-renders BELIEFS.md.

        Diffs the rendered markdown against the prior version and logs
        any change to EVOLUTION.md so belief-ladder transitions show up
        in the agent's audit trail.
        """
        data = {
            "version": 1,
            "updated_at": _now_iso(),
            "beliefs": [b.to_dict() for b in self._beliefs.values()],
        }
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.workspace),
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, str(self.json_path))
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as exc:
            logger.error("BeliefStore save failed: %s", exc)
            return

        # Capture prior rendered markdown for the evolution diff, then
        # re-render. If the content changed, append the diff to EVOLUTION.md.
        old_md = ""
        if self.md_path.exists():
            try:
                old_md = self.md_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                old_md = ""

        new_md = self.render_markdown()
        try:
            from .engine import _atomic_write_text, _identity_file_cache
            _atomic_write_text(self.md_path, new_md)
            _identity_file_cache.pop(str(self.md_path), None)
        except OSError as exc:
            logger.warning("BELIEFS.md render failed: %s", exc)
            return

        if old_md != new_md:
            try:
                from .engine import _log_evolution_to_file
                _log_evolution_to_file(
                    self.workspace,
                    "BELIEFS.md",
                    old_md,
                    new_md,
                    reason="belief ladder update",
                )
            except (ImportError, OSError) as exc:
                logger.debug("BELIEFS evolution log skipped: %s", exc)

    # ── Mutations ─────────────────────────────────────────────────────

    def add_observation(
        self,
        text: str,
        *,
        session_id: str = "",
        falsification: str = "",
    ) -> str:
        """Add a new belief at observation state. Returns the new belief ID."""
        text = text.strip()
        if not text:
            raise ValueError("Belief text cannot be empty")

        now = _now_iso()
        bid = _short_id()
        belief = Belief(
            id=bid,
            text=text,
            state=BeliefState.OBSERVATION.value,
            created_at=now,
            last_updated=now,
            evidence_count=1,
            triggering_sessions=[session_id] if session_id else [],
            falsification=falsification.strip(),
        )
        self._beliefs[bid] = belief
        self._save()
        logger.info("Belief observed: %s (%s)", bid, belief.short_title())
        return bid

    def record_evidence(
        self,
        belief_id: str,
        *,
        session_id: str = "",
    ) -> str:
        """
        Increment evidence count on a belief. Auto-promotes if threshold crossed.
        Returns the belief's state after processing.
        """
        belief = self._beliefs.get(belief_id)
        if not belief:
            raise KeyError(f"No belief with id {belief_id}")
        if belief.state == BeliefState.DEMOTED.value:
            return belief.state  # demoted beliefs do not auto-re-promote

        belief.evidence_count += 1
        belief.last_updated = _now_iso()
        if session_id and session_id not in belief.triggering_sessions:
            belief.triggering_sessions.append(session_id)
            # Cap to last 10 sessions to keep the file bounded
            belief.triggering_sessions = belief.triggering_sessions[-10:]

        # Check auto-promotion
        old_state = belief.state
        new_state = self._compute_autopromotion(belief)
        if new_state != old_state:
            belief.state = new_state
            belief.notes.append(
                f"{_now_iso()}: auto-promoted {old_state} → {new_state} "
                f"at evidence count {belief.evidence_count}"
            )
            logger.info(
                "Belief %s promoted: %s → %s (evidence=%d)",
                belief_id, old_state, new_state, belief.evidence_count,
            )

        self._save()
        return belief.state

    def _compute_autopromotion(self, belief: Belief) -> str:
        """Decide what state a belief should be in given its evidence count."""
        count = belief.evidence_count
        if count >= _PROMOTE_TO_COMMITTED:
            return BeliefState.COMMITTED.value
        if count >= _PROMOTE_TO_TESTED:
            return BeliefState.TESTED.value
        if count >= _PROMOTE_TO_WORKING_THEORY:
            return BeliefState.WORKING_THEORY.value
        return BeliefState.OBSERVATION.value

    # Ladder in ascending crystallization order — shared by promote().
    _LADDER: tuple[BeliefState, ...] = (
        BeliefState.OBSERVATION,
        BeliefState.WORKING_THEORY,
        BeliefState.TESTED,
        BeliefState.COMMITTED,
    )

    def promote(self, belief_id: str) -> str:
        """Manually bump a belief one step up the ladder."""
        belief = self._beliefs.get(belief_id)
        if not belief:
            raise KeyError(f"No belief with id {belief_id}")
        if belief.state == BeliefState.DEMOTED.value:
            raise ValueError(
                "Cannot promote a demoted belief directly — re-add as observation"
            )

        current_level = BeliefState(belief.state).level
        next_rung = next(
            (s for s in self._LADDER if s.level == current_level + 1),
            None,
        )
        if next_rung is None:
            return belief.state  # already at top (committed)

        belief.notes.append(
            f"{_now_iso()}: manually promoted {belief.state} → {next_rung.value}"
        )
        belief.state = next_rung.value
        belief.last_updated = _now_iso()
        self._save()
        return belief.state

    def demote(
        self,
        belief_id: str,
        *,
        reason: str,
        session_id: str = "",
    ) -> str:
        """Mark a belief as falsified. Records the reason."""
        belief = self._beliefs.get(belief_id)
        if not belief:
            raise KeyError(f"No belief with id {belief_id}")

        old_state = belief.state
        belief.state = BeliefState.DEMOTED.value
        belief.last_updated = _now_iso()
        note = f"{_now_iso()}: demoted from {old_state} — {reason.strip()}"
        if session_id:
            note += f" (session: {session_id})"
        belief.notes.append(note)
        self._save()
        logger.info(
            "Belief %s demoted from %s: %s", belief_id, old_state, reason.strip()[:120]
        )
        return belief.state

    def update_text(self, belief_id: str, new_text: str) -> None:
        """Update the wording of a belief without changing its state."""
        belief = self._beliefs.get(belief_id)
        if not belief:
            raise KeyError(f"No belief with id {belief_id}")
        belief.text = new_text.strip()
        belief.last_updated = _now_iso()
        self._save()

    # ── Queries ───────────────────────────────────────────────────────

    def get(self, belief_id: str) -> Belief | None:
        return self._beliefs.get(belief_id)

    def get_by_state(self, state: BeliefState) -> list[Belief]:
        return [b for b in self._beliefs.values() if b.state == state.value]

    def get_all(self) -> list[Belief]:
        return list(self._beliefs.values())

    def stats(self) -> dict[str, int]:
        counts = {s.value: 0 for s in BeliefState}
        for b in self._beliefs.values():
            counts[b.state] = counts.get(b.state, 0) + 1
        counts["total"] = len(self._beliefs)
        return counts

    # ── Rendering ─────────────────────────────────────────────────────

    def render_markdown(self) -> str:
        """Render BELIEFS.md from current state. Groups by state, ordered by evidence."""
        sections: list[str] = [
            "# Beliefs",
            "",
            "_What I currently hold as true, and how hardened each belief is._",
            "",
            "_State ladder: observation → working_theory → tested → committed. "
            "Beliefs that get falsified are demoted with the reason logged._",
            "",
        ]

        state_order = [
            (BeliefState.COMMITTED, "Committed", "Beliefs I would defend under contrary pressure."),
            (BeliefState.TESTED, "Tested", "Working theories that have held up across multiple independent cases."),
            (BeliefState.WORKING_THEORY, "Working Theories", "Patterns I'm starting to trust but still testing."),
            (BeliefState.OBSERVATION, "Observations", "One-off noticings, not yet patterns."),
            (BeliefState.DEMOTED, "Demoted", "Beliefs I used to hold but had to update."),
        ]

        for state, title, tagline in state_order:
            beliefs = self.get_by_state(state)
            if not beliefs:
                continue
            # Sort by evidence count desc, then last_updated desc
            beliefs.sort(
                key=lambda b: (-b.evidence_count, b.last_updated),
                reverse=False,
            )
            sections.append(f"## {title}")
            sections.append(f"_{tagline}_")
            sections.append("")
            for b in beliefs:
                sections.append(self._render_belief(b))
                sections.append("")

        stats = self.stats()
        sections.append("---")
        sections.append(
            f"_Total: {stats.get('total', 0)} beliefs. "
            f"Committed: {stats.get('committed', 0)}. "
            f"Tested: {stats.get('tested', 0)}. "
            f"Working theory: {stats.get('working_theory', 0)}. "
            f"Observation: {stats.get('observation', 0)}. "
            f"Demoted: {stats.get('demoted', 0)}._"
        )
        sections.append("")
        return "\n".join(sections)

    def _render_belief(self, belief: Belief) -> str:
        lines = [f"### {belief.id}: {belief.short_title()}"]
        lines.append("")
        lines.append(belief.text)
        lines.append("")
        meta: list[str] = []
        meta.append(f"**Evidence count:** {belief.evidence_count}")
        meta.append(f"**Created:** {belief.created_at[:10]}")
        meta.append(f"**Updated:** {belief.last_updated[:10]}")
        if belief.triggering_sessions:
            shown = ", ".join(belief.triggering_sessions[-5:])
            meta.append(f"**Sessions:** {shown}")
        if belief.falsification:
            meta.append(f"**Would update if:** {belief.falsification}")
        lines.append(" · ".join(meta))
        if belief.notes:
            lines.append("")
            lines.append(f"_Recent notes: {belief.notes[-1]}_")
        return "\n".join(lines)
