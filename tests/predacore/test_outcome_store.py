"""
Tests for OutcomeStore — Phase 2 outcome tracking.

Tests cover:
  - TaskOutcome dataclass defaults
  - SQLite schema creation
  - Record + retrieve outcomes
  - Feedback detection (positive / negative / explicit command)
  - Feedback update on most recent outcome
  - Failure pattern analysis
  - Tool stats aggregation
  - Provider stats aggregation
"""
import time

import pytest

from src.predacore.services.outcome_store import (
    OutcomeStore,
    TaskOutcome,
    detect_feedback,
)

# ── Helpers ───────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    """Create a fresh OutcomeStore with a temp database."""
    return OutcomeStore(tmp_path / "test_outcomes.db")


def _make_outcome(**overrides) -> TaskOutcome:
    defaults = dict(
        user_id="user1",
        user_message="hello",
        response_summary="hi there",
        tools_used=["run_command"],
        provider_used="gemini",
        model_used="flash",
        latency_ms=150.0,
        token_count_prompt=100,
        token_count_completion=50,
        iterations=2,
        success=True,
    )
    defaults.update(overrides)
    return TaskOutcome(**defaults)


# ── TaskOutcome dataclass ─────────────────────────────────────────────

class TestTaskOutcome:
    def test_auto_task_id(self):
        o = TaskOutcome()
        assert o.task_id
        assert len(o.task_id) == 12

    def test_auto_timestamp(self):
        before = time.time()
        o = TaskOutcome()
        after = time.time()
        assert before <= o.timestamp <= after

    def test_explicit_values(self):
        o = TaskOutcome(
            task_id="abc123",
            user_id="u1",
            tools_used=["read_file", "write_file"],
            success=False,
            error="timeout",
        )
        assert o.task_id == "abc123"
        assert o.tools_used == ["read_file", "write_file"]
        assert not o.success
        assert o.error == "timeout"


# ── Feedback detection ────────────────────────────────────────────────

class TestFeedbackDetection:
    def test_positive_thanks(self):
        assert detect_feedback("thanks!") == "good"

    def test_positive_great(self):
        assert detect_feedback("Great, that worked!") == "good"

    def test_negative_wrong(self):
        assert detect_feedback("That's wrong.") == "bad"

    def test_negative_useless(self):
        assert detect_feedback("useless answer") == "bad"

    def test_explicit_command_good(self):
        assert detect_feedback("/feedback good") == "good"

    def test_explicit_command_bad(self):
        assert detect_feedback("/feedback bad") == "bad"

    def test_explicit_command_invalid(self):
        assert detect_feedback("/feedback maybe") is None

    def test_no_feedback_normal_message(self):
        assert detect_feedback("What's the weather?") is None

    def test_no_feedback_empty(self):
        assert detect_feedback("") is None

    def test_no_feedback_long_message(self):
        long_msg = "thanks " + "x" * 300
        assert detect_feedback(long_msg) is None

    def test_case_insensitive(self):
        assert detect_feedback("THANK YOU") == "good"
        assert detect_feedback("WRONG!") == "bad"


# ── OutcomeStore core operations ──────────────────────────────────────

class TestOutcomeStoreRecord:
    @pytest.mark.asyncio
    async def test_record_and_retrieve(self, store):
        o = _make_outcome(task_id="t1")
        await store.record(o)
        recent = await store.get_recent(limit=5)
        assert len(recent) == 1
        assert recent[0]["task_id"] == "t1"
        assert recent[0]["user_id"] == "user1"
        assert recent[0]["success"] == 1

    @pytest.mark.asyncio
    async def test_record_multiple(self, store):
        for i in range(5):
            await store.record(_make_outcome(task_id=f"t{i}"))
        recent = await store.get_recent(limit=10)
        assert len(recent) == 5

    @pytest.mark.asyncio
    async def test_retrieve_by_user(self, store):
        await store.record(_make_outcome(task_id="t1", user_id="alice"))
        await store.record(_make_outcome(task_id="t2", user_id="bob"))
        await store.record(_make_outcome(task_id="t3", user_id="alice"))
        alice = await store.get_recent(user_id="alice")
        assert len(alice) == 2
        assert all(r["user_id"] == "alice" for r in alice)

    @pytest.mark.asyncio
    async def test_upsert_replaces(self, store):
        await store.record(_make_outcome(task_id="t1", latency_ms=100))
        await store.record(_make_outcome(task_id="t1", latency_ms=200))
        recent = await store.get_recent()
        assert len(recent) == 1
        assert recent[0]["latency_ms"] == 200.0

    @pytest.mark.asyncio
    async def test_db_file_created(self, store):
        assert store.db_path.exists()


# ── Feedback update ───────────────────────────────────────────────────

class TestFeedbackUpdate:
    @pytest.mark.asyncio
    async def test_update_most_recent(self, store):
        await store.record(_make_outcome(task_id="t1", user_id="u1"))
        await store.record(_make_outcome(task_id="t2", user_id="u1"))
        updated = await store.update_feedback("u1", "bad")
        assert updated is True
        recent = await store.get_recent(user_id="u1")
        # Most recent (t2) should have feedback, t1 should not
        by_id = {r["task_id"]: r for r in recent}
        assert by_id["t2"]["user_feedback"] == "bad"
        assert by_id["t1"]["user_feedback"] is None

    @pytest.mark.asyncio
    async def test_update_nonexistent_user(self, store):
        updated = await store.update_feedback("nobody", "good")
        assert updated is False


# ── Analytics: failure patterns ───────────────────────────────────────

class TestFailurePatterns:
    @pytest.mark.asyncio
    async def test_no_failures(self, store):
        await store.record(_make_outcome(success=True))
        patterns = await store.get_failure_patterns()
        assert patterns == []

    @pytest.mark.asyncio
    async def test_failures_grouped_by_tool(self, store):
        await store.record(_make_outcome(
            task_id="f1", success=False,
            tool_errors=["run_command: Permission denied", "run_command: Timeout"],
        ))
        await store.record(_make_outcome(
            task_id="f2", success=False,
            tool_errors=["read_file: Not found"],
        ))
        patterns = await store.get_failure_patterns()
        assert len(patterns) == 2
        # run_command should be first (2 failures)
        assert patterns[0]["tool_name"] == "run_command"
        assert patterns[0]["failure_count"] == 2
        assert patterns[1]["tool_name"] == "read_file"
        assert patterns[1]["failure_count"] == 1

    @pytest.mark.asyncio
    async def test_window_filter(self, store):
        old = _make_outcome(task_id="old", success=False,
                            tool_errors=["x: old error"])
        old.timestamp = time.time() - 48 * 3600  # 2 days ago
        await store.record(old)
        recent = _make_outcome(task_id="new", success=False,
                               tool_errors=["y: new error"])
        await store.record(recent)
        patterns = await store.get_failure_patterns(window_hours=24)
        assert len(patterns) == 1
        assert patterns[0]["tool_name"] == "y"


# ── Analytics: tool stats ─────────────────────────────────────────────

class TestToolStats:
    @pytest.mark.asyncio
    async def test_tool_usage_stats(self, store):
        await store.record(_make_outcome(
            task_id="t1", tools_used=["run_command"], success=True, latency_ms=100,
        ))
        await store.record(_make_outcome(
            task_id="t2", tools_used=["run_command"], success=False, latency_ms=200,
        ))
        await store.record(_make_outcome(
            task_id="t3", tools_used=["read_file"], success=True, latency_ms=50,
        ))
        stats = await store.get_tool_stats("run_command")
        assert stats["total_uses"] == 2
        assert stats["success_rate"] == 50.0
        assert stats["avg_latency_ms"] == 150.0

    @pytest.mark.asyncio
    async def test_unknown_tool_stats(self, store):
        stats = await store.get_tool_stats("nonexistent")
        assert stats["total_uses"] == 0
        assert stats["success_rate"] == 0


# ── Analytics: provider stats ─────────────────────────────────────────

class TestProviderStats:
    @pytest.mark.asyncio
    async def test_provider_aggregation(self, store):
        await store.record(_make_outcome(
            task_id="t1", provider_used="gemini", model_used="flash",
            latency_ms=100, token_count_prompt=50, token_count_completion=30,
        ))
        await store.record(_make_outcome(
            task_id="t2", provider_used="openai", model_used="gpt-4o",
            latency_ms=300, token_count_prompt=100, token_count_completion=80,
        ))
        await store.record(_make_outcome(
            task_id="t3", provider_used="gemini", model_used="flash",
            latency_ms=200, token_count_prompt=60, token_count_completion=40,
        ))
        stats = await store.get_provider_stats()
        assert len(stats) == 2
        gemini = next(s for s in stats if s["provider"] == "gemini")
        assert gemini["total_calls"] == 2
        assert gemini["total_prompt_tokens"] == 110


# ── Feedback summary ──────────────────────────────────────────────────

class TestFeedbackSummary:
    @pytest.mark.asyncio
    async def test_feedback_counts(self, store):
        await store.record(_make_outcome(task_id="t1", user_feedback="good"))
        await store.record(_make_outcome(task_id="t2", user_feedback="good"))
        await store.record(_make_outcome(task_id="t3", user_feedback="bad"))
        await store.record(_make_outcome(task_id="t4"))  # no feedback
        summary = await store.get_feedback_summary()
        assert summary["good"] == 2
        assert summary["bad"] == 1
