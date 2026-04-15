"""Tests for src/common/memory_service.py — SQLite-backed memory persistence."""
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from jarvis._vendor.common.memory_service import (
    ImportanceLevel,
    Memory,
    MemoryService,
    MemoryType,
    _cosine_similarity,
    _decode_embedding,
    _encode_embedding,
    remember_fact,
    remember_preference,
)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory for tests."""
    return str(tmp_path / "memory_test")


@pytest.fixture
def service(tmp_data_dir):
    """Create a fresh MemoryService."""
    svc = MemoryService(data_path=tmp_data_dir)
    yield svc
    svc.close()


class TestMemoryModel:
    """Tests for the Memory dataclass."""

    def test_create_default_memory(self):
        m = Memory(content="test", user_id="u1")
        assert m.content == "test"
        assert m.memory_type == MemoryType.FACT
        assert m.importance == ImportanceLevel.MEDIUM
        assert m.access_count == 0
        assert m.embedding is None

    def test_to_dict_and_back(self):
        m = Memory(
            content="user likes dark mode",
            user_id="shubham",
            memory_type=MemoryType.PREFERENCE,
            importance=ImportanceLevel.HIGH,
            tags=["ui", "preference"],
            metadata={"source": "chat"},
        )
        d = m.to_dict()
        m2 = Memory.from_dict(d)
        assert m2.content == m.content
        assert m2.memory_type == m.memory_type
        assert m2.importance == m.importance
        assert m2.tags == m.tags
        assert str(m2.id) == str(m.id)

    def test_expired_memory_detection(self):
        m = Memory(
            content="temp",
            user_id="u1",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert m.expires_at < datetime.now(timezone.utc)


class TestMemoryServiceInit:
    """Tests for service initialization."""

    def test_creates_data_directory(self, tmp_data_dir):
        svc = MemoryService(data_path=tmp_data_dir)
        assert Path(tmp_data_dir).exists()
        svc.close()

    def test_creates_sqlite_database(self, tmp_data_dir):
        svc = MemoryService(data_path=tmp_data_dir)
        db_path = Path(tmp_data_dir) / "memory.db"
        assert db_path.exists()
        svc.close()

    def test_database_has_correct_schema(self, tmp_data_dir):
        svc = MemoryService(data_path=tmp_data_dir)
        conn = sqlite3.connect(str(Path(tmp_data_dir) / "memory.db"))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "memories" in tables
        assert "schema_version" in tables
        conn.close()
        svc.close()


class TestMemoryStore:
    """Tests for storing memories."""

    @pytest.mark.asyncio
    async def test_store_basic_memory(self, service):
        m = await service.store("test fact", user_id="u1")
        assert m.content == "test fact"
        assert m.user_id == "u1"
        assert m.memory_type == MemoryType.FACT

    @pytest.mark.asyncio
    async def test_store_with_all_params(self, service):
        expires = datetime.now(timezone.utc) + timedelta(days=7)
        m = await service.store(
            content="important note",
            user_id="u1",
            memory_type=MemoryType.TASK,
            importance=ImportanceLevel.CRITICAL,
            tags=["urgent", "deadline"],
            metadata={"project": "prometheus"},
            source="chat",
            expires_at=expires,
        )
        assert m.memory_type == MemoryType.TASK
        assert m.importance == ImportanceLevel.CRITICAL
        assert m.tags == ["urgent", "deadline"]
        assert m.metadata["project"] == "prometheus"

    @pytest.mark.asyncio
    async def test_stored_memory_persists_across_restart(self, tmp_data_dir):
        svc1 = MemoryService(data_path=tmp_data_dir)
        m = await svc1.store("persistent fact", user_id="u1")
        svc1.close()

        # New service instance should find the memory
        svc2 = MemoryService(data_path=tmp_data_dir)
        found = svc2.get(m.id, "u1")
        assert found is not None
        assert found.content == "persistent fact"
        svc2.close()

    @pytest.mark.asyncio
    async def test_store_multiple_users(self, service):
        await service.store("fact for alice", user_id="alice")
        await service.store("fact for bob", user_id="bob")
        assert service.get_stats("alice")["total"] == 1
        assert service.get_stats("bob")["total"] == 1


class TestMemoryRecall:
    """Tests for recall (keyword search)."""

    @pytest.mark.asyncio
    async def test_recall_by_keyword(self, service):
        await service.store("user prefers dark mode", user_id="u1")
        await service.store("user likes python", user_id="u1")

        results = await service.recall("dark mode", user_id="u1")
        assert len(results) >= 1
        assert any("dark mode" in m.content for m, _ in results)

    @pytest.mark.asyncio
    async def test_recall_empty_for_unknown_user(self, service):
        results = await service.recall("anything", user_id="nonexistent")
        assert results == []

    @pytest.mark.asyncio
    async def test_recall_filters_by_type(self, service):
        await service.store("fact1", user_id="u1", memory_type=MemoryType.FACT)
        await service.store("pref1", user_id="u1", memory_type=MemoryType.PREFERENCE)

        results = await service.recall(
            "pref1", user_id="u1",
            memory_types=[MemoryType.PREFERENCE],
        )
        assert all(m.memory_type == MemoryType.PREFERENCE for m, _ in results)

    @pytest.mark.asyncio
    async def test_recall_filters_by_importance(self, service):
        await service.store("low", user_id="u1", importance=ImportanceLevel.LOW)
        await service.store("critical thing", user_id="u1", importance=ImportanceLevel.CRITICAL)

        results = await service.recall(
            "thing", user_id="u1",
            min_importance=ImportanceLevel.HIGH,
        )
        for m, _ in results:
            assert m.importance.value >= ImportanceLevel.HIGH.value

    @pytest.mark.asyncio
    async def test_recall_updates_access_stats(self, service):
        m = await service.store("test access tracking", user_id="u1")
        assert m.access_count == 0

        results = await service.recall("access tracking", user_id="u1")
        assert len(results) >= 1
        assert results[0][0].access_count == 1

    @pytest.mark.asyncio
    async def test_recall_respects_top_k(self, service):
        for i in range(10):
            await service.store(f"fact number {i}", user_id="u1")

        results = await service.recall("fact", user_id="u1", top_k=3)
        assert len(results) <= 3


class TestMemoryDelete:
    """Tests for deleting memories."""

    @pytest.mark.asyncio
    async def test_delete_existing_memory(self, service):
        m = await service.store("to delete", user_id="u1")
        assert await service.delete(m.id, "u1") is True
        assert service.get(m.id, "u1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_memory(self, service):
        assert await service.delete(uuid4(), "u1") is False

    @pytest.mark.asyncio
    async def test_delete_persists(self, tmp_data_dir):
        svc1 = MemoryService(data_path=tmp_data_dir)
        m = await svc1.store("to delete", user_id="u1")
        await svc1.delete(m.id, "u1")
        svc1.close()

        svc2 = MemoryService(data_path=tmp_data_dir)
        assert svc2.get(m.id, "u1") is None
        svc2.close()


class TestMemoryListBy:
    """Tests for list_by_type and list_by_tag."""

    @pytest.mark.asyncio
    async def test_list_by_type(self, service):
        await service.store("fact1", user_id="u1", memory_type=MemoryType.FACT)
        await service.store("fact2", user_id="u1", memory_type=MemoryType.FACT)
        await service.store("pref1", user_id="u1", memory_type=MemoryType.PREFERENCE)

        facts = service.list_by_type("u1", MemoryType.FACT)
        assert len(facts) == 2
        assert all(m.memory_type == MemoryType.FACT for m in facts)

    @pytest.mark.asyncio
    async def test_list_by_tag(self, service):
        await service.store("tagged1", user_id="u1", tags=["python", "coding"])
        await service.store("tagged2", user_id="u1", tags=["python", "ai"])
        await service.store("untagged", user_id="u1", tags=["rust"])

        python_mems = service.list_by_tag("u1", "python")
        assert len(python_mems) == 2


class TestMemoryExpirationAndMigration:
    """Tests for expiration and JSON migration."""

    @pytest.mark.asyncio
    async def test_expired_memories_excluded_from_recall(self, service):
        await service.store(
            "expired",
            user_id="u1",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        results = await service.recall("expired", user_id="u1")
        assert len(results) == 0

    def test_json_migration(self, tmp_path):
        """Test that legacy JSON memories are migrated to SQLite."""
        data_dir = tmp_path / "migrate_test"
        user_dir = data_dir / "test_user"
        user_dir.mkdir(parents=True)

        # Create a legacy JSON file
        memories = [
            {
                "id": str(uuid4()),
                "content": "migrated fact",
                "memory_type": "fact",
                "importance": 2,
                "tags": ["legacy"],
                "metadata": {},
                "user_id": "test_user",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_accessed": datetime.now(timezone.utc).isoformat(),
                "access_count": 0,
                "source": "",
                "expires_at": None,
            }
        ]
        with open(user_dir / "memories.json", "w") as f:
            json.dump(memories, f)

        svc = MemoryService(data_path=str(data_dir))
        assert svc.get_stats("test_user")["total"] == 1

        # JSON file should be renamed
        assert not (user_dir / "memories.json").exists()
        assert (user_dir / "memories.json.migrated").exists()
        svc.close()


class TestEmbeddingHelpers:
    """Tests for embedding encode/decode and cosine similarity."""

    def test_encode_decode_roundtrip(self):
        original = [0.1, 0.2, 0.3, -0.5, 1.0]
        encoded = _encode_embedding(original)
        decoded = _decode_embedding(encoded)
        for a, b in zip(original, decoded, strict=False):
            assert abs(a - b) < 1e-6

    def test_cosine_similarity_identical(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_cosine_similarity_opposite(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestConvenienceFunctions:
    """Tests for remember_fact and remember_preference helpers."""

    @pytest.mark.asyncio
    async def test_remember_fact(self, service):
        m = await remember_fact(service, "u1", "Python is great", tags=["lang"])
        assert m.memory_type == MemoryType.FACT
        assert m.tags == ["lang"]

    @pytest.mark.asyncio
    async def test_remember_preference(self, service):
        m = await remember_preference(service, "u1", "dark mode")
        assert m.memory_type == MemoryType.PREFERENCE
        assert m.importance == ImportanceLevel.HIGH
        assert "preference" in m.tags


class TestMemoryStats:
    """Tests for get_stats."""

    @pytest.mark.asyncio
    async def test_stats_empty_user(self, service):
        stats = service.get_stats("nobody")
        assert stats["total"] == 0

    @pytest.mark.asyncio
    async def test_stats_with_data(self, service):
        await service.store("f1", user_id="u1", memory_type=MemoryType.FACT)
        await service.store("f2", user_id="u1", memory_type=MemoryType.FACT)
        await service.store("p1", user_id="u1", memory_type=MemoryType.PREFERENCE,
                            importance=ImportanceLevel.HIGH)

        stats = service.get_stats("u1")
        assert stats["total"] == 3
        assert stats["by_type"]["fact"] == 2
        assert stats["by_type"]["preference"] == 1
        assert "HIGH" in stats["by_importance"]


class TestSummarizeContext:
    """Tests for summarize_context."""

    @pytest.mark.asyncio
    async def test_summarize_empty(self, service):
        result = await service.summarize_context("u1", "anything")
        assert result == ""

    @pytest.mark.asyncio
    async def test_summarize_with_memories(self, service):
        await service.store("user codes in python", user_id="u1",
                            memory_type=MemoryType.FACT)
        await service.store("user prefers vim", user_id="u1",
                            memory_type=MemoryType.PREFERENCE)

        result = await service.summarize_context("u1", "python")
        assert len(result) > 0
        assert "[" in result  # Should have type prefix
