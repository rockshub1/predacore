"""
Tests for Self-Improvement Engine — Phase 5.

Tests cover:
  - ImprovementProposal dataclass
  - AuditLog (write, read, count_today)
  - SelfImprovementEngine.analyze_failures (tool failures, provider, feedback, drift)
  - Safety guards (daily limit, protected files, risk level)
  - apply_proposal flow
"""
import time

import pytest

from src.predacore.services.outcome_store import OutcomeStore, TaskOutcome
from src.predacore.agents.self_improvement import (
    DEFAULT_PROTECTED_FILES,
    AuditLog,
    ImprovementProposal,
    ProposalCategory,
    RiskLevel,
    SelfImprovementEngine,
)

# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def outcome_store(tmp_path):
    return OutcomeStore(tmp_path / "test.db")


@pytest.fixture
def engine(tmp_path, outcome_store):
    return SelfImprovementEngine(
        home_dir=tmp_path,
        outcome_store=outcome_store,
        max_daily_modifications=3,
        enable_auto_apply=True,
    )


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "audit.jsonl")


# ── ImprovementProposal ──────────────────────────────────────────────

class TestImprovementProposal:
    def test_auto_id(self):
        p = ImprovementProposal()
        assert p.proposal_id
        assert len(p.proposal_id) == 10

    def test_auto_timestamp(self):
        before = time.time()
        p = ImprovementProposal()
        after = time.time()
        assert before <= p.created_at <= after

    def test_to_dict(self):
        p = ImprovementProposal(
            category=ProposalCategory.TOOL_CONFIG,
            description="Fix run_command",
            risk_level=RiskLevel.MEDIUM,
        )
        d = p.to_dict()
        assert d["category"] == "tool_config"
        assert d["risk_level"] == "medium"
        assert d["description"] == "Fix run_command"
        assert d["applied"] is False

    def test_explicit_values(self):
        p = ImprovementProposal(
            proposal_id="abc",
            category=ProposalCategory.CODE_FIX,
            risk_level=RiskLevel.HIGH,
            auto_applicable=True,
        )
        assert p.proposal_id == "abc"
        assert p.category == ProposalCategory.CODE_FIX


# ── AuditLog ──────────────────────────────────────────────────────────

class TestAuditLog:
    def test_log_and_read(self, audit):
        audit.log("test_action", {"key": "value"})
        entries = audit.get_recent()
        assert len(entries) == 1
        assert entries[0]["action"] == "test_action"
        assert entries[0]["key"] == "value"
        assert "timestamp" in entries[0]

    def test_multiple_entries(self, audit):
        for i in range(5):
            audit.log(f"action_{i}", {"index": i})
        entries = audit.get_recent()
        assert len(entries) == 5

    def test_recent_limit(self, audit):
        for i in range(10):
            audit.log(f"action_{i}", {})
        entries = audit.get_recent(limit=3)
        assert len(entries) == 3

    def test_count_today(self, audit):
        audit.log("apply_proposal", {"proposal_id": "a"})
        audit.log("apply_proposal", {"proposal_id": "b"})
        audit.log("other_action", {"proposal_id": "c"})
        assert audit.count_today() == 2

    def test_empty_log(self, audit):
        entries = audit.get_recent()
        assert entries == []
        assert audit.count_today() == 0


# ── SelfImprovementEngine: analyze_failures ───────────────────────────

class TestAnalyzeFailures:
    @pytest.mark.asyncio
    async def test_no_failures_no_proposals(self, engine, outcome_store):
        await outcome_store.record(TaskOutcome(
            task_id="t1", user_id="u1", success=True,
            provider_used="gemini", model_used="flash",
        ))
        proposals = await engine.analyze_failures()
        assert len(proposals) == 0

    @pytest.mark.asyncio
    async def test_tool_failure_proposal(self, engine, outcome_store):
        for i in range(4):
            await outcome_store.record(TaskOutcome(
                task_id=f"f{i}", user_id="u1", success=False,
                tool_errors=["run_command: Permission denied"],
                provider_used="gemini", model_used="flash",
            ))
        proposals = await engine.analyze_failures()
        tool_proposals = [p for p in proposals if p.category == ProposalCategory.TOOL_CONFIG]
        assert len(tool_proposals) >= 1
        assert "run_command" in tool_proposals[0].description

    @pytest.mark.asyncio
    async def test_provider_reliability_proposal(self, engine, outcome_store):
        for i in range(6):
            await outcome_store.record(TaskOutcome(
                task_id=f"p{i}", user_id="u1",
                success=(i < 2),  # 2 success, 4 failures = 33% success
                provider_used="openai", model_used="gpt-4o",
            ))
        proposals = await engine.analyze_failures()
        provider_proposals = [
            p for p in proposals
            if "provider" in p.description.lower() or "success rate" in p.description.lower()
        ]
        assert len(provider_proposals) >= 1

    @pytest.mark.asyncio
    async def test_feedback_proposal(self, engine, outcome_store):
        for i in range(6):
            fb = "bad" if i < 3 else "good"
            await outcome_store.record(TaskOutcome(
                task_id=f"fb{i}", user_id="u1", success=True,
                user_feedback=fb,
                provider_used="gemini", model_used="flash",
            ))
        proposals = await engine.analyze_failures()
        prompt_proposals = [p for p in proposals if p.category == ProposalCategory.PROMPT_TUNING]
        assert len(prompt_proposals) >= 1
        assert "negative" in prompt_proposals[0].description.lower()

    @pytest.mark.asyncio
    async def test_audit_logged_on_analysis(self, engine, outcome_store):
        await engine.analyze_failures()
        entries = engine.audit_log.get_recent()
        assert any(e["action"] == "analyze_failures" for e in entries)


# ── SelfImprovementEngine: apply_proposal ─────────────────────────────

class TestApplyProposal:
    @pytest.mark.asyncio
    async def test_apply_low_risk(self, engine):
        proposal = ImprovementProposal(
            category=ProposalCategory.PROMPT_TUNING,
            description="Strengthen identity anchors",
            target_file="prompts.py",
            risk_level=RiskLevel.LOW,
        )
        result = await engine.apply_proposal(proposal)
        assert result is True
        # Proposals are now logged for manual review, not auto-applied
        entries = engine.audit_log.get_recent()
        assert any(e["action"] == "apply_proposed" for e in entries)

    @pytest.mark.asyncio
    async def test_block_high_risk(self, engine):
        proposal = ImprovementProposal(
            category=ProposalCategory.CODE_FIX,
            description="Modify core logic",
            target_file="tool_handlers.py",
            risk_level=RiskLevel.HIGH,
        )
        result = await engine.apply_proposal(proposal)
        assert result is False
        assert proposal.applied is False

    @pytest.mark.asyncio
    async def test_block_medium_risk(self, engine):
        proposal = ImprovementProposal(
            risk_level=RiskLevel.MEDIUM,
            target_file="tool_handlers.py",
        )
        result = await engine.apply_proposal(proposal)
        assert result is False

    @pytest.mark.asyncio
    async def test_block_protected_file(self, engine):
        proposal = ImprovementProposal(
            category=ProposalCategory.TOOL_CONFIG,
            description="Change config",
            target_file="config.py",
            risk_level=RiskLevel.LOW,
        )
        result = await engine.apply_proposal(proposal)
        assert result is False
        entries = engine.audit_log.get_recent()
        assert any(e.get("reason") == "protected_file" for e in entries)

    @pytest.mark.asyncio
    async def test_block_daily_limit(self, engine):
        # Exhaust the daily limit
        for i in range(3):
            p = ImprovementProposal(
                target_file="identity/SOUL.md",
                risk_level=RiskLevel.LOW,
            )
            await engine.apply_proposal(p)

        # 4th should be blocked
        p4 = ImprovementProposal(
            target_file="identity/SOUL.md",
            risk_level=RiskLevel.LOW,
        )
        result = await engine.apply_proposal(p4)
        assert result is False

    @pytest.mark.asyncio
    async def test_block_auto_apply_disabled(self, tmp_path, outcome_store):
        engine = SelfImprovementEngine(
            home_dir=tmp_path,
            outcome_store=outcome_store,
            enable_auto_apply=False,
        )
        proposal = ImprovementProposal(
            target_file="identity/SOUL.md",
            risk_level=RiskLevel.LOW,
        )
        result = await engine.apply_proposal(proposal)
        assert result is False


# ── SelfImprovementEngine: get_summary ────────────────────────────────

class TestGetSummary:
    @pytest.mark.asyncio
    async def test_summary_initial(self, engine):
        summary = engine.get_summary()
        assert summary["proposals_pending"] == 0
        assert summary["proposals_applied"] == 0
        assert summary["max_daily_modifications"] == 3
        assert summary["auto_apply_enabled"] is True
        assert "config.py" in summary["protected_files"]

    @pytest.mark.asyncio
    async def test_summary_after_analysis(self, engine, outcome_store):
        for i in range(4):
            await outcome_store.record(TaskOutcome(
                task_id=f"s{i}", user_id="u1", success=False,
                tool_errors=["bad_tool: error"],
                provider_used="gemini", model_used="flash",
            ))
        await engine.analyze_failures()
        summary = engine.get_summary()
        assert summary["proposals_pending"] >= 1


# ── Default protected files ───────────────────────────────────────────

class TestProtectedFiles:
    def test_security_protected(self):
        assert "security.py" in DEFAULT_PROTECTED_FILES

    def test_auth_protected(self):
        assert "auth.py" in DEFAULT_PROTECTED_FILES

    def test_config_protected(self):
        assert "config.py" in DEFAULT_PROTECTED_FILES

    def test_init_protected(self):
        assert "__init__.py" in DEFAULT_PROTECTED_FILES
