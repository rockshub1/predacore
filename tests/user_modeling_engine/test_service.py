import json

from jarvis._vendor.user_modeling_engine.service import UserModelingEngineService, UserProfile


def test_profile_round_trip_and_deep_merge(tmp_path):
    svc = UserModelingEngineService(data_path=str(tmp_path))
    original = UserProfile(
        user_id="alice",
        preferences={"mode": "beast", "nested": {"a": 1}},
        goals=["launch"],
        cognitive_style="direct",
    )
    svc.save_profile(original)

    loaded = svc.get_profile("alice")
    assert loaded.user_id == "alice"
    assert loaded.preferences["mode"] == "beast"
    assert loaded.preferences["nested"]["a"] == 1

    updated = svc.update_profile(
        "alice",
        {
            "preferences": {"nested": {"b": 2}},
            "goals": ["launch", "scale"],
            "notes": "focus reliability",
        },
    )
    assert updated.preferences["nested"]["a"] == 1
    assert updated.preferences["nested"]["b"] == 2
    assert updated.goals == ["launch", "scale"]
    assert updated.notes == "focus reliability"


def test_record_interaction_writes_event_and_updates_profile(tmp_path):
    svc = UserModelingEngineService(data_path=str(tmp_path))
    svc.record_interaction(
        user_id="u/1",
        message="hello world",
        metadata={"trace_id": "trace-123"},
    )

    events_path = tmp_path / "events" / "u_1.jsonl"
    assert events_path.exists()
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["user_id"] == "u/1"
    assert event["message"] == "hello world"
    assert event["metadata"]["trace_id"] == "trace-123"

    profile = svc.get_profile("u/1")
    assert profile.last_interaction_at


def test_build_planning_context_contains_core_fields(tmp_path):
    svc = UserModelingEngineService(data_path=str(tmp_path))
    svc.update_profile(
        "bob",
        {
            "preferences": {"tone": "concise"},
            "goals": ["ship public beast"],
            "knowledge_areas": {"python": "advanced"},
            "cognitive_style": "systems",
        },
    )
    ctx = svc.build_planning_context("bob")
    assert ctx["user_id"] == "bob"
    assert ctx["preferences"]["tone"] == "concise"
    assert ctx["goals"] == ["ship public beast"]
    assert ctx["knowledge_areas"]["python"] == "advanced"
    assert ctx["cognitive_style"] == "systems"
    assert "updated_at" in ctx
