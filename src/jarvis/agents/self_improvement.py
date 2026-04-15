"""
JARVIS Self-Improvement Engine — Failure analysis and evolution.

Analyzes recorded outcomes to identify recurring failure patterns, proposes
improvements (prompt tuning, tool config, code fixes), and can auto-apply
low-risk changes.

Safety constraints:
  - Protected files cannot be modified
  - Max N modifications per day
  - All changes logged to audit trail
  - High-risk changes require human approval

Usage:
    engine = SelfImprovementEngine(config, outcome_store, llm)
    proposals = await engine.analyze_failures()
    for p in proposals:
        if p.auto_applicable:
            await engine.apply_proposal(p)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ProposalCategory(str, Enum):
    PROMPT_TUNING = "prompt_tuning"
    TOOL_CONFIG = "tool_config"
    NEW_TOOL = "new_tool"
    CODE_FIX = "code_fix"
    MEMORY_CLEANUP = "memory_cleanup"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ImprovementProposal:
    """A proposed improvement derived from failure analysis."""

    proposal_id: str = ""
    category: ProposalCategory = ProposalCategory.PROMPT_TUNING
    description: str = ""
    target_file: str = ""
    proposed_change: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    auto_applicable: bool = False
    evidence: list[str] = field(default_factory=list)
    created_at: float = 0.0
    applied: bool = False
    applied_at: float | None = None

    def __post_init__(self):
        if not self.proposal_id:
            import uuid

            self.proposal_id = uuid.uuid4().hex[:10]
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "category": self.category.value,
            "description": self.description,
            "target_file": self.target_file,
            "proposed_change": self.proposed_change,
            "risk_level": self.risk_level.value,
            "auto_applicable": self.auto_applicable,
            "evidence": self.evidence,
            "created_at": self.created_at,
            "applied": self.applied,
            "applied_at": self.applied_at,
        }


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class AuditLog:
    """Append-only JSON-lines audit log for self-improvement actions."""

    def __init__(self, log_path: Path):
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, action: str, details: dict[str, Any]) -> None:
        entry = {
            "timestamp": time.time(),
            "action": action,
            **details,
        }
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:
            logger.warning("Audit log write failed: %s", exc)

    async def alog(self, action: str, details: dict[str, Any]) -> None:
        """Async-safe audit log — won't block the event loop."""
        await asyncio.to_thread(self.log, action, details)

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        try:
            lines = self._path.read_text(encoding="utf-8").strip().splitlines()
            entries = []
            for line in lines[-limit:]:
                try:
                    entries.append(json.loads(line))
                except (ValueError, json.JSONDecodeError):
                    continue
            return entries
        except OSError:
            return []

    def count_today(self) -> int:
        """Count modifications applied today (UTC to match time.time() entries)."""
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        entries = self.get_recent(limit=200)
        return sum(
            1
            for e in entries
            if e.get("action") in ("apply_proposal", "apply_proposed")
            and e.get("timestamp", 0) >= today_start
        )

    async def acount_today(self) -> int:
        """Async-safe count — won't block the event loop."""
        return await asyncio.to_thread(self.count_today)


# ---------------------------------------------------------------------------
# Protected files
# ---------------------------------------------------------------------------

DEFAULT_PROTECTED_FILES = frozenset(
    {
        "security.py",
        "auth.py",
        "auth_middleware.py",
        "config.py",
        "__init__.py",
        "SOUL_SEED.md",
        "BOOTSTRAP.md",
        "engine.py",
    }
)


# ---------------------------------------------------------------------------
# Self-Improvement Engine
# ---------------------------------------------------------------------------


class SelfImprovementEngine:
    """
    Analyzes failures and proposes improvements.

    The engine works in three stages:
    1. Analyze: Read failure patterns from OutcomeStore
    2. Propose: Generate ImprovementProposals
    3. Apply: Execute low-risk proposals (with audit trail)
    """

    def __init__(
        self,
        home_dir: str | Path,
        outcome_store: Any,
        max_daily_modifications: int = 3,
        protected_files: frozenset | None = None,
        enable_auto_apply: bool = True,
    ):
        self._home = Path(home_dir)
        self._outcome_store = outcome_store
        self._max_daily_mods = max_daily_modifications
        self._protected_files = protected_files or DEFAULT_PROTECTED_FILES
        self._enable_auto_apply = enable_auto_apply
        self._audit = AuditLog(self._home / "self_improvement_audit.jsonl")
        self._proposals: list[ImprovementProposal] = []

    # ── Analysis ──────────────────────────────────────────────────────

    async def analyze_failures(
        self, window_hours: float = 24
    ) -> list[ImprovementProposal]:
        """
        Analyze recent failures and generate improvement proposals.

        Strategies:
        1. Tool consistently failing → suggest disabling or fixing
        2. High drift scores → suggest prompt improvement
        3. Repeated tool errors → suggest tool config change
        4. Low feedback scores → identify problem patterns
        """
        proposals: list[ImprovementProposal] = []

        # 1. Tool failure patterns
        failure_patterns = await self._outcome_store.get_failure_patterns(
            window_hours=window_hours
        )
        for pattern in failure_patterns:
            if pattern["failure_count"] >= 3:
                proposals.append(
                    ImprovementProposal(
                        category=ProposalCategory.TOOL_CONFIG,
                        description=(
                            f"Tool '{pattern['tool_name']}' failed {pattern['failure_count']} times "
                            f"in the last {window_hours}h. Consider disabling or fixing."
                        ),
                        target_file="tool_registry.py",
                        proposed_change=f"Disable or add retry logic for '{pattern['tool_name']}'",
                        risk_level=RiskLevel.MEDIUM,
                        auto_applicable=False,
                        evidence=pattern.get("error_samples", [])[:3],
                    )
                )

        # 2. Provider reliability
        provider_stats = await self._outcome_store.get_provider_stats(
            window_hours=window_hours
        )
        for stat in provider_stats:
            if stat["total_calls"] >= 5 and stat["success_rate"] < 70:
                proposals.append(
                    ImprovementProposal(
                        category=ProposalCategory.TOOL_CONFIG,
                        description=(
                            f"Provider '{stat['provider']}' model '{stat['model']}' has "
                            f"{stat['success_rate']:.0f}% success rate over {stat['total_calls']} calls. "
                            "Consider switching default model."
                        ),
                        target_file="config.py",
                        proposed_change=f"Consider switching from {stat['model']} to a more reliable model",
                        risk_level=RiskLevel.MEDIUM,
                        auto_applicable=False,
                        evidence=[
                            f"Success rate: {stat['success_rate']:.1f}%",
                            f"Avg latency: {stat['avg_latency_ms']:.0f}ms",
                        ],
                    )
                )

        # 3. Feedback-driven proposals
        feedback = await self._outcome_store.get_feedback_summary(
            window_hours=window_hours
        )
        total_fb = feedback.get("good", 0) + feedback.get("bad", 0)
        if total_fb >= 5:
            bad_rate = feedback.get("bad", 0) / total_fb * 100
            if bad_rate >= 30:
                proposals.append(
                    ImprovementProposal(
                        category=ProposalCategory.PROMPT_TUNING,
                        description=(
                            f"User feedback is {bad_rate:.0f}% negative ({feedback['bad']} bad "
                            f"out of {total_fb} total). System prompt may need tuning."
                        ),
                        target_file="agents/jarvis/SOUL_SEED.md",
                        proposed_change="Review and improve system prompt based on negative feedback patterns",
                        risk_level=RiskLevel.LOW,
                        auto_applicable=False,
                        evidence=[
                            f"Good feedback: {feedback.get('good', 0)}",
                            f"Bad feedback: {feedback.get('bad', 0)}",
                        ],
                    )
                )

        # 4. High drift score outcomes
        recent = await self._outcome_store.get_recent(limit=50)
        high_drift_count = sum(1 for r in recent if r.get("drift_score", 0) > 0.4)
        if high_drift_count >= 3:
            proposals.append(
                ImprovementProposal(
                    category=ProposalCategory.PROMPT_TUNING,
                    description=(
                        f"{high_drift_count} recent responses had high persona drift scores. "
                        "System prompt identity section may need strengthening."
                    ),
                    target_file="agents/jarvis/SOUL_SEED.md",
                    proposed_change="Strengthen persona identity anchors in SOUL.md",
                    risk_level=RiskLevel.LOW,
                    auto_applicable=False,
                    evidence=[f"High drift outcomes: {high_drift_count}/50 recent"],
                )
            )

        self._proposals = proposals
        await self._audit.alog(
            "analyze_failures",
            {
                "window_hours": window_hours,
                "proposals_generated": len(proposals),
                "categories": [p.category.value for p in proposals],
            },
        )

        logger.info(
            "Self-improvement analysis: %d proposals from %dh window",
            len(proposals),
            window_hours,
        )
        return proposals

    # ── Application ───────────────────────────────────────────────────

    async def apply_proposal(self, proposal: ImprovementProposal) -> bool:
        """
        Apply a proposal if it passes safety checks.

        Returns True if applied, False if blocked.
        """
        # Safety check: daily limit
        today_count = await self._audit.acount_today()
        if today_count >= self._max_daily_mods:
            logger.warning(
                "Daily modification limit reached (%d/%d). Proposal %s blocked.",
                today_count,
                self._max_daily_mods,
                proposal.proposal_id,
            )
            await self._audit.alog(
                "apply_blocked",
                {
                    "proposal_id": proposal.proposal_id,
                    "reason": "daily_limit_reached",
                },
            )
            return False

        # Safety check: protected files (resolve fully to prevent path traversal)
        try:
            resolved_target = Path(proposal.target_file).resolve()
        except (ValueError, OSError):
            logger.warning("Proposal %s has invalid target path.", proposal.proposal_id)
            return False
        target_basename = resolved_target.name
        if target_basename in self._protected_files:
            logger.warning(
                "Proposal %s targets protected file '%s'. Blocked.",
                proposal.proposal_id,
                proposal.target_file,
            )
            await self._audit.alog(
                "apply_blocked",
                {
                    "proposal_id": proposal.proposal_id,
                    "reason": "protected_file",
                    "file": proposal.target_file,
                },
            )
            return False

        # Safety check: auto-apply only for low risk
        if not self._enable_auto_apply:
            logger.info(
                "Auto-apply disabled. Proposal %s logged but not applied.",
                proposal.proposal_id,
            )
            await self._audit.alog(
                "apply_skipped",
                {
                    "proposal_id": proposal.proposal_id,
                    "reason": "auto_apply_disabled",
                },
            )
            return False

        if proposal.risk_level != RiskLevel.LOW:
            logger.info(
                "Proposal %s has risk_level=%s (not LOW). Requires manual approval.",
                proposal.proposal_id,
                proposal.risk_level.value,
            )
            await self._audit.alog(
                "apply_deferred",
                {
                    "proposal_id": proposal.proposal_id,
                    "reason": "risk_level_too_high",
                    "risk": proposal.risk_level.value,
                },
            )
            return False

        # Apply the proposal — write the change to the target file
        target_path = resolved_target
        try:
            if proposal.category == ProposalCategory.TOOL_CONFIG and target_path.suffix in (".yaml", ".yml", ".json"):
                # For config files: read, apply proposed_change as patch description, write
                if not target_path.exists():
                    logger.warning("Target config file does not exist: %s", target_path)
                    return False
                # Config changes are logged but require manual application
                proposal.applied = False
                await self._audit.alog(
                    "apply_proposed",
                    {
                        "proposal_id": proposal.proposal_id,
                        "category": proposal.category.value,
                        "description": proposal.description[:200],
                        "target_file": str(target_path),
                        "proposed_change": proposal.proposed_change[:500],
                        "action_required": "manual_review",
                    },
                )
                logger.info(
                    "Proposal %s logged for manual review: %s",
                    proposal.proposal_id,
                    proposal.description[:100],
                )
                return True
            else:
                # All other categories: log the proposal for human review
                proposal.applied = False
                await self._audit.alog(
                    "apply_proposed",
                    {
                        "proposal_id": proposal.proposal_id,
                        "category": proposal.category.value,
                        "description": proposal.description[:200],
                        "target_file": str(target_path),
                        "proposed_change": proposal.proposed_change[:500],
                        "action_required": "manual_review",
                    },
                )
                logger.info(
                    "Proposal %s logged for manual review: %s",
                    proposal.proposal_id,
                    proposal.description[:100],
                )
                return True
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error("Failed to process proposal %s: %s", proposal.proposal_id, exc)
            await self._audit.alog(
                "apply_error",
                {"proposal_id": proposal.proposal_id, "error": str(exc)},
            )
            return False

    # ── Accessors ─────────────────────────────────────────────────────

    @property
    def proposals(self) -> list[ImprovementProposal]:
        return list(self._proposals)

    @property
    def audit_log(self) -> AuditLog:
        return self._audit

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the improvement engine state."""
        return {
            "proposals_pending": sum(1 for p in self._proposals if not p.applied),
            "proposals_applied": sum(1 for p in self._proposals if p.applied),
            "daily_modifications": self._audit.count_today(),
            "max_daily_modifications": self._max_daily_mods,
            "auto_apply_enabled": self._enable_auto_apply,
            "protected_files": sorted(self._protected_files),
        }
