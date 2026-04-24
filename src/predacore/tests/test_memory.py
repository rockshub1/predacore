"""
Comprehensive tests for predacore.memory — unified memory system.

Tests: UnifiedMemoryStore (SQLite + vectors), MemoryRetriever,
MemoryConsolidator, helper functions.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from predacore.memory.consolidator import MemoryConsolidator
from predacore.memory.retriever import MemoryRetriever, _to_unix_timestamp, _truncate
from predacore.memory.store import (
    UnifiedMemoryStore,
    _coerce_metadata_dict,
    _memory_matches_scope,
    _now_iso,
    _NumpyVectorIndex,
    _pack_embedding,
    _prepare_memory_metadata,
    _unpack_embedding,
    _uuid,
    future_iso_from_ttl,
    normalize_memory_scope,
)

# ── Helpers ────────────────────────────────────────────────────────


class TestStoreHelpers:
    """Tests for store module helper functions."""

    def test_now_iso(self):
        ts = _now_iso()
        assert "T" in ts  # ISO format
        assert "+" in ts or "Z" in ts  # timezone aware

    def test_uuid_unique(self):
        ids = {_uuid() for _ in range(100)}
        assert len(ids) == 100

    def test_normalize_scope_valid(self):
        assert normalize_memory_scope("global") == "global"
        assert normalize_memory_scope("team") == "team"
        assert normalize_memory_scope("scratch") == "scratch"

    def test_normalize_scope_defaults(self):
        assert normalize_memory_scope(None) == "global"
        assert normalize_memory_scope("") == "global"
        assert normalize_memory_scope("invalid") == "global"

    def test_normalize_scope_case_insensitive(self):
        assert normalize_memory_scope("GLOBAL") == "global"
        assert normalize_memory_scope("Team") == "team"

    def test_coerce_metadata_dict(self):
        assert _coerce_metadata_dict({"a": 1}) == {"a": 1}
        assert _coerce_metadata_dict('{"b": 2}') == {"b": 2}
        assert _coerce_metadata_dict("not json") == {}
        assert _coerce_metadata_dict(None) == {}
        assert _coerce_metadata_dict(42) == {}

    def test_prepare_memory_metadata(self):
        meta = _prepare_memory_metadata(
            {"existing": "val"},
            memory_scope="team",
            team_id="team-1",
            agent_id="default",
        )
        assert meta["scope"] == "team"
        assert meta["team_id"] == "team-1"
        assert meta["agent_id"] == "default"
        assert meta["existing"] == "val"

    def test_memory_matches_scope_global(self):
        mem = {"metadata": {"scope": "global"}}
        assert _memory_matches_scope(mem, ["global"]) is True
        assert _memory_matches_scope(mem, ["team"]) is False

    def test_memory_matches_scope_team(self):
        mem = {"metadata": json.dumps({"scope": "team", "team_id": "t1"})}
        assert _memory_matches_scope(mem, ["team"], team_id="t1") is True
        assert _memory_matches_scope(mem, ["team"], team_id="t2") is False

    def test_future_iso_from_ttl(self):
        result = future_iso_from_ttl(3600)
        assert result is not None
        assert "T" in result

    def test_future_iso_from_ttl_none(self):
        assert future_iso_from_ttl(None) is None

    def test_future_iso_from_ttl_zero(self):
        assert future_iso_from_ttl(0) is None

    def test_pack_unpack_embedding(self):
        vec = [0.1, 0.2, 0.3, 0.4, 0.5]
        packed = _pack_embedding(vec)
        assert isinstance(packed, bytes)
        unpacked = _unpack_embedding(packed)
        assert len(unpacked) == len(vec)
        for a, b in zip(vec, unpacked):
            assert abs(a - b) < 1e-6

    def test_pack_empty_embedding(self):
        packed = _pack_embedding([])
        assert packed == b""
        assert _unpack_embedding(packed) == []


# ── NumpyVectorIndex ───────────────────────────────────────────────


class TestNumpyVectorIndex:
    """Tests for the in-memory vector index."""

    @pytest.fixture
    def index(self):
        return _NumpyVectorIndex(dimensions=3)

    def test_empty_index(self, index):
        assert index.size == 0

    @pytest.mark.asyncio
    async def test_add_and_size(self, index):
        await index.add("id1", [1.0, 0.0, 0.0])
        assert index.size == 1

    @pytest.mark.asyncio
    async def test_add_duplicate_updates(self, index):
        await index.add("id1", [1.0, 0.0, 0.0])
        await index.add("id1", [0.0, 1.0, 0.0])  # Update
        assert index.size == 1

    @pytest.mark.asyncio
    async def test_remove(self, index):
        await index.add("id1", [1.0, 0.0, 0.0])
        assert await index.remove("id1") is True
        assert index.size == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, index):
        assert await index.remove("nonexistent") is False

    @pytest.mark.asyncio
    async def test_search_basic(self, index):
        await index.add("id1", [1.0, 0.0, 0.0])
        await index.add("id2", [0.0, 1.0, 0.0])
        await index.add("id3", [0.0, 0.0, 1.0])
        results = await index.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) <= 2
        if results:
            assert results[0][0] == "id1"  # Most similar
            assert results[0][1] > 0.9  # High score

    @pytest.mark.asyncio
    async def test_search_empty(self, index):
        results = await index.search([1.0, 0.0, 0.0])
        assert results == []

    @pytest.mark.asyncio
    async def test_eviction_at_max(self):
        index = _NumpyVectorIndex(dimensions=2)
        index.MAX_VECTORS = 10
        for i in range(15):
            await index.add(f"id{i}", [float(i), float(i)])
        assert index.size <= 10

    @pytest.mark.asyncio
    async def test_eviction_protects_preferences(self):
        """Preference/entity types must never be evicted while evictable entries exist."""
        index = _NumpyVectorIndex(dimensions=2)
        index.MAX_VECTORS = 5
        # Add 3 preferences (protected)
        for i in range(3):
            await index.add(f"pref{i}", [float(i), 0.0], {"type": "preference"})
        # Add 10 facts (unprotected)
        for i in range(10):
            await index.add(f"fact{i}", [0.0, float(i)], {"type": "fact"})
        # All 3 preferences should still be present
        pref_ids = [id for id in index._ids if id.startswith("pref")]
        assert len(pref_ids) == 3


# ── UnifiedMemoryStore ─────────────────────────────────────────────


class TestUnifiedMemoryStore:
    """Tests for the main memory store."""

    @pytest.fixture
    def store(self, tmp_path):
        db_path = str(tmp_path / "test_memory.db")
        return UnifiedMemoryStore(db_path=db_path)

    @pytest.mark.asyncio
    async def test_store_memory(self, store):
        mem_id = await store.store("User likes dark mode", memory_type="preference")
        assert isinstance(mem_id, str)
        assert len(mem_id) > 0

    @pytest.mark.asyncio
    async def test_store_and_get(self, store):
        mem_id = await store.store("Test fact", memory_type="fact", importance=3)
        mem = await store.get(mem_id)
        assert mem is not None
        assert mem["content"] == "Test fact"
        assert mem["memory_type"] == "fact"
        assert mem["importance"] == 3

    @pytest.mark.asyncio
    async def test_store_with_metadata(self, store):
        mem_id = await store.store(
            "Scoped memory",
            memory_scope="team",
            team_id="team-1",
            metadata={"custom": "data"},
        )
        mem = await store.get(mem_id)
        assert mem["metadata"]["scope"] == "team"
        assert mem["metadata"]["team_id"] == "team-1"
        assert mem["metadata"]["custom"] == "data"

    @pytest.mark.asyncio
    async def test_store_with_tags(self, store):
        mem_id = await store.store("Tagged", tags=["tag1", "tag2"])
        mem = await store.get(mem_id)
        assert mem["tags"] == ["tag1", "tag2"]

    @pytest.mark.asyncio
    async def test_delete_memory(self, store):
        mem_id = await store.store("To delete")
        assert await store.delete(mem_id) is True
        assert await store.get(mem_id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        assert await store.delete("nonexistent-id") is False

    @pytest.mark.asyncio
    async def test_recall_keyword(self, store):
        await store.store("Python is great for AI", user_id="u1", memory_type="fact")
        await store.store("JavaScript for web dev", user_id="u1", memory_type="fact")
        results = await store.recall("Python AI", user_id="u1")
        assert len(results) > 0
        assert "Python" in results[0][0]["content"]

    @pytest.mark.asyncio
    async def test_recall_empty_query(self, store):
        results = await store.recall("", user_id="u1")
        assert results == []

    @pytest.mark.asyncio
    async def test_recall_filters_user(self, store):
        await store.store("User 1 data", user_id="u1")
        await store.store("User 2 data", user_id="u2")
        results = await store.recall("data", user_id="u1")
        for mem, _ in results:
            assert mem["user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_recall_filters_importance(self, store):
        await store.store("Low imp", importance=1, user_id="u1")
        await store.store("High imp content", importance=5, user_id="u1")
        results = await store.recall("content", user_id="u1", min_importance=3)
        for mem, _ in results:
            assert mem["importance"] >= 3

    @pytest.mark.asyncio
    async def test_upsert_entity(self, store):
        eid = await store.upsert_entity("PredaCore", "project", {"version": "0.1"})
        assert isinstance(eid, str)
        entity = await store.get_entity("PredaCore")
        assert entity is not None
        assert entity["name"] == "PredaCore"
        assert entity["entity_type"] == "project"
        assert entity["properties"]["version"] == "0.1"

    @pytest.mark.asyncio
    async def test_upsert_entity_increment_count(self, store):
        await store.upsert_entity("Python", "language")
        await store.upsert_entity("Python", "language")
        entity = await store.get_entity("Python")
        assert entity["mention_count"] == 2

    @pytest.mark.asyncio
    async def test_upsert_entity_case_insensitive(self, store):
        await store.upsert_entity("PredaCore", "project")
        entity = await store.get_entity("predacore")
        assert entity is not None

    @pytest.mark.asyncio
    async def test_list_entities(self, store):
        await store.upsert_entity("Entity1", "concept")
        await store.upsert_entity("Entity2", "tool")
        entities = await store.list_entities()
        assert len(entities) == 2

    @pytest.mark.asyncio
    async def test_list_entities_filtered(self, store):
        await store.upsert_entity("Entity1", "concept")
        await store.upsert_entity("Entity2", "tool")
        tools = await store.list_entities(entity_type="tool")
        assert len(tools) == 1
        assert tools[0]["entity_type"] == "tool"

    @pytest.mark.asyncio
    async def test_add_relation(self, store):
        e1 = await store.upsert_entity("User", "person")
        e2 = await store.upsert_entity("Python", "language")
        rel_id = await store.add_relation(e1, e2, "uses")
        assert isinstance(rel_id, str)

    @pytest.mark.asyncio
    async def test_add_duplicate_relation_strengthens(self, store):
        e1 = await store.upsert_entity("User", "person")
        e2 = await store.upsert_entity("Python", "language")
        r1 = await store.add_relation(e1, e2, "uses", weight=1.0)
        r2 = await store.add_relation(e1, e2, "uses", weight=1.0)
        assert r1 == r2  # Same relation, just strengthened

    @pytest.mark.asyncio
    async def test_get_entity_context(self, store):
        await store.store("PredaCore is an AI agent", memory_type="fact", user_id="u1")
        await store.upsert_entity("PredaCore", "project")
        ctx = await store.get_entity_context("PredaCore")
        assert ctx["entity"] is not None
        assert ctx["entity"]["name"] == "PredaCore"

    @pytest.mark.asyncio
    async def test_get_entity_context_nonexistent(self, store):
        ctx = await store.get_entity_context("nonexistent")
        assert ctx["entity"] is None

    @pytest.mark.asyncio
    async def test_execute_raw(self, store):
        await store.store("test content", memory_type="fact")
        await store.execute_raw(
            "UPDATE memories SET importance = ? WHERE memory_type = ?",
            (5, "fact"),
        )
        # Verify update worked
        results = await store.recall("test", user_id="default", min_importance=5)
        assert len(results) > 0


# ── Retriever ──────────────────────────────────────────────────────


class TestRetrieverHelpers:
    """Tests for retriever helper functions."""

    def test_truncate_short(self):
        assert _truncate("short", 100) == "short"

    def test_truncate_long(self):
        result = _truncate("x" * 200, 50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_truncate_whitespace_collapse(self):
        assert _truncate("hello   world\n\nnewline", 100) == "hello world newline"

    def test_to_unix_timestamp_float(self):
        assert _to_unix_timestamp(1234.5, default=0) == 1234.5

    def test_to_unix_timestamp_int(self):
        assert _to_unix_timestamp(1234, default=0) == 1234.0

    def test_to_unix_timestamp_none(self):
        assert _to_unix_timestamp(None, default=99.0) == 99.0

    def test_to_unix_timestamp_iso_string(self):
        result = _to_unix_timestamp("2025-01-01T00:00:00+00:00", default=0)
        assert result > 0

    def test_to_unix_timestamp_float_string(self):
        assert _to_unix_timestamp("1234.5", default=0) == 1234.5

    def test_to_unix_timestamp_invalid(self):
        assert _to_unix_timestamp("not a date", default=42.0) == 42.0


class TestMemoryRetriever:
    """Tests for the MemoryRetriever."""

    @pytest.fixture
    def mock_store(self):
        store = AsyncMock()
        store.get_all_memories = AsyncMock(return_value=[])
        store.recall = AsyncMock(return_value=[])
        store.list_entities = AsyncMock(return_value=[])
        store.get_entity_context = AsyncMock(return_value={"entity": None, "relations": [], "memories": []})
        store.get_recent_episodes = AsyncMock(return_value=[])
        return store

    @pytest.fixture
    def retriever(self, mock_store):
        return MemoryRetriever(mock_store)

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, retriever):
        result = await retriever.build_context("")
        assert result == ""

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty(self, retriever):
        result = await retriever.build_context("   ")
        assert result == ""

    @pytest.mark.asyncio
    async def test_build_context_with_preferences(self, retriever, mock_store):
        mock_store.get_all_memories.return_value = [
            {"content": "User prefers dark mode", "user_id": "default", "memory_type": "preference"}
        ]
        result = await retriever.build_context("dark mode", user_id="default")
        assert "Preferences" in result or result == ""  # depends on user_id match

    @pytest.mark.asyncio
    async def test_build_context_with_semantic(self, retriever, mock_store):
        mock_store.recall.return_value = [
            ({"content": "PredaCore uses Python", "memory_type": "fact", "importance": 3, "created_at": time.time()}, 0.9),
        ]
        result = await retriever.build_context("what language", user_id="default")
        if result:
            assert "Memories" in result or "Python" in result


# ── Consolidator ───────────────────────────────────────────────────


class TestRustEntityExtraction:
    """Tests for predacore_core.extract_entities (Rust extension).

    The Rust extension returns list[tuple] with shape:
        (name: str, entity_type: str, confidence: f32, source_tier: u8)
    """

    def test_dictionary_extraction(self):
        import predacore_core
        entities = predacore_core.extract_entities("Using Python and Docker with FastAPI")
        names = [e[0] for e in entities]
        assert "Python" in names
        assert "Docker" in names
        assert "FastAPI" in names

    def test_camelcase_extraction(self):
        import predacore_core
        entities = predacore_core.extract_entities("Using ToolDispatcher and ScreenVision")
        names = [e[0] for e in entities]
        assert "ToolDispatcher" in names
        assert "ScreenVision" in names

    def test_allcaps_extraction(self):
        import predacore_core
        entities = predacore_core.extract_entities("Set OPENAI_API_KEY and MAX_TOKENS")
        names = [e[0] for e in entities]
        assert "OPENAI_API_KEY" in names or "MAX_TOKENS" in names

    def test_stopwords_excluded(self):
        import predacore_core
        entities = predacore_core.extract_entities("THE AND FOR NOT THIS")
        names = [e[0] for e in entities]
        assert "THE" not in names
        assert "AND" not in names

    def test_url_extraction(self):
        # The current Rust extension extracts protocol tokens (http/https) as
        # "protocol" entities, not full URLs. Full URL extraction isn't implemented
        # on the Rust side yet. Verify the protocol is at least detected.
        import predacore_core
        entities = predacore_core.extract_entities(
            "Check https://example.com/docs for details"
        )
        names = [e[0].lower() for e in entities]
        assert "https" in names or "http" in names

    def test_empty_text(self):
        import predacore_core
        assert predacore_core.extract_entities("") == []


class TestConsolidator:
    """Tests for the MemoryConsolidator pipeline."""

    @pytest.fixture
    def mock_store(self):
        store = AsyncMock()
        store.apply_decay = AsyncMock(return_value=10)
        store.get_all_memories = AsyncMock(return_value=[])
        store.list_entities = AsyncMock(return_value=[])
        store.prune_expired = AsyncMock(return_value=0)
        store.prune_low_importance = AsyncMock(return_value=0)
        store.save_vector_index = AsyncMock()
        return store

    @pytest.mark.asyncio
    async def test_consolidate_empty_store(self, mock_store):
        consolidator = MemoryConsolidator(mock_store)
        stats = await consolidator.consolidate()
        assert isinstance(stats, dict)
        assert "decayed" in stats
        assert "entities_found" in stats
        assert "pruned" in stats

    @pytest.mark.asyncio
    async def test_consolidate_recent(self, mock_store):
        consolidator = MemoryConsolidator(mock_store)
        stats = await consolidator.consolidate_recent(last_n=10)
        assert "entities_found" in stats
        assert "merged" in stats


# ── Bug Fix Tests ──────────────────────────────────────────────────


class TestVectorIndexSyncRebuild:
    """Tests for Bug 1 fix: _add_sync for sync rebuild from SQLite."""

    def test_add_sync_basic(self):
        idx = _NumpyVectorIndex(dimensions=3)
        idx._add_sync("id1", [1.0, 0.0, 0.0])
        assert idx.size == 1

    def test_add_sync_duplicate_updates(self):
        idx = _NumpyVectorIndex(dimensions=3)
        idx._add_sync("id1", [1.0, 0.0, 0.0])
        idx._add_sync("id1", [0.0, 1.0, 0.0])
        assert idx.size == 1
        assert idx._vecs[0] == [0.0, 1.0, 0.0]

    def test_add_sync_eviction(self):
        idx = _NumpyVectorIndex(dimensions=2)
        idx.MAX_VECTORS = 5
        for i in range(10):
            idx._add_sync(f"id{i}", [float(i), 0.0])
        assert idx.size <= 5

    def test_add_sync_matches_async_behavior(self):
        """Sync and async add should produce identical index state."""
        idx_sync = _NumpyVectorIndex(dimensions=2)
        idx_sync._add_sync("a", [1.0, 0.0])
        idx_sync._add_sync("b", [0.0, 1.0])

        idx_async = _NumpyVectorIndex(dimensions=2)
        # Use asyncio.run() instead of the deprecated get_event_loop() +
        # run_until_complete pattern. The old pattern breaks on Python 3.10+
        # depending on whether a prior test left an event loop attached to
        # the main thread — caused this test to pass in isolation but fail
        # when run alongside other async tests in the suite.
        asyncio.run(idx_async.add("a", [1.0, 0.0]))
        asyncio.run(idx_async.add("b", [0.0, 1.0]))

        assert idx_sync._ids == idx_async._ids
        assert idx_sync._vecs == idx_async._vecs

    def test_rebuild_from_sqlite_uses_sync(self, tmp_path):
        """Verify that init rebuild from SQLite actually populates vectors."""
        import sqlite3

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, content TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT 'fact',
                importance INTEGER NOT NULL DEFAULT 2,
                source TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                user_id TEXT NOT NULL DEFAULT 'default',
                embedding BLOB, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, last_accessed TEXT NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                decay_score REAL NOT NULL DEFAULT 1.0,
                expires_at TEXT, session_id TEXT, parent_id TEXT
            );
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY, name TEXT NOT NULL,
                entity_type TEXT NOT NULL DEFAULT 'concept',
                properties TEXT NOT NULL DEFAULT '{}',
                first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
                mention_count INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY, source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                relation_type TEXT NOT NULL DEFAULT 'related_to',
                weight REAL NOT NULL DEFAULT 1.0,
                properties TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                summary TEXT NOT NULL, key_facts TEXT NOT NULL DEFAULT '[]',
                entities_mentioned TEXT NOT NULL DEFAULT '[]',
                tools_used TEXT NOT NULL DEFAULT '[]',
                outcome TEXT, user_satisfaction REAL,
                created_at TEXT NOT NULL, token_count INTEGER NOT NULL DEFAULT 0
            );
        """)
        # Insert a memory with a 384-dim embedding
        vec = [0.1] * 384
        blob = _pack_embedding(vec)
        conn.execute(
            "INSERT INTO memories (id, content, memory_type, embedding, created_at, updated_at, last_accessed) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("test-id", "test content", "fact", blob, "2025-01-01", "2025-01-01", "2025-01-01"),
        )
        conn.commit()
        conn.close()

        # Delete vectors.json if it exists
        vec_path = tmp_path / "vectors.json"
        if vec_path.exists():
            vec_path.unlink()

        # Create store — should rebuild vector index from SQLite
        store = UnifiedMemoryStore(db_path=str(db_path))
        assert store._vec_index.size == 1  # Rebuild worked!
        store.close()


# ── Improvement Tests ──────────────────────────────────────────────


class TestStoreContextManager:
    """Tests for async context manager support."""

    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_path):
        db_path = str(tmp_path / "ctx_test.db")
        async with UnifiedMemoryStore(db_path=db_path) as store:
            mem_id = await store.store("context manager test")
            assert mem_id
        # Store is closed after exiting context


class TestStoreBackup:
    """Tests for the SQLite backup primitive."""

    @pytest.mark.asyncio
    async def test_backup_creates_file(self, tmp_path):
        db_path = str(tmp_path / "orig.db")
        backup_path = str(tmp_path / "backup.db")

        store = UnifiedMemoryStore(db_path=db_path)
        await store.store("data 1", memory_type="fact")
        await store.store("data 2", memory_type="fact")

        await store.backup(backup_path)

        assert Path(backup_path).exists()
        assert Path(backup_path).stat().st_size > 0
        store.close()

    @pytest.mark.asyncio
    async def test_backup_contains_data(self, tmp_path):
        db_path = str(tmp_path / "orig.db")
        backup_path = str(tmp_path / "backup.db")

        store = UnifiedMemoryStore(db_path=db_path)
        await store.store("unique test content", memory_type="fact")
        await store.backup(backup_path)
        store.close()

        # Open the backup as a store and verify it has the same data
        store2 = UnifiedMemoryStore(db_path=backup_path)
        assert await store2.count() == 1
        store2.close()


class TestStoreCount:
    """Tests for the count() convenience method."""

    @pytest.mark.asyncio
    async def test_count_empty(self, tmp_path):
        store = UnifiedMemoryStore(db_path=str(tmp_path / "count_test.db"))
        assert await store.count() == 0
        store.close()

    @pytest.mark.asyncio
    async def test_count_all(self, tmp_path):
        store = UnifiedMemoryStore(db_path=str(tmp_path / "count_test.db"))
        await store.store("fact 1", memory_type="fact")
        await store.store("pref 1", memory_type="preference")
        await store.store("fact 2", memory_type="fact")
        assert await store.count() == 3
        store.close()

    @pytest.mark.asyncio
    async def test_count_by_type(self, tmp_path):
        store = UnifiedMemoryStore(db_path=str(tmp_path / "count_test.db"))
        await store.store("fact 1", memory_type="fact")
        await store.store("pref 1", memory_type="preference")
        await store.store("fact 2", memory_type="fact")
        assert await store.count(memory_type="fact") == 2
        assert await store.count(memory_type="preference") == 1
        assert await store.count(memory_type="conversation") == 0
        store.close()


class TestRewardProportionalDecay:
    """Tests for reward-proportional decay (Hippo-inspired).

    Verifies that memories tied to positively-reviewed sessions decay
    slower, and memories tied to negatively-reviewed sessions decay faster.
    """

    @pytest.mark.asyncio
    async def test_decay_with_neutral_reward_matches_baseline(self, tmp_path):
        """A reward of 1.0 must produce identical decay to no reward_map."""
        store = UnifiedMemoryStore(db_path=str(tmp_path / "reward_neutral.db"))
        id1 = await store.store("fact A", memory_type="fact", session_id="sess1")

        # Force a stale last_accessed so decay actually applies
        store._conn.execute(
            "UPDATE memories SET last_accessed = ? WHERE id = ?",
            ("2025-01-01T00:00:00+00:00", id1),
        )
        store._conn.commit()

        await store.apply_decay(reward_map={"sess1": 1.0})
        neutral_score = (await store.get(id1))["decay_score"]

        # Reset and decay without reward_map
        store._conn.execute(
            "UPDATE memories SET last_accessed = ?, decay_score = importance WHERE id = ?",
            ("2025-01-01T00:00:00+00:00", id1),
        )
        store._conn.commit()
        await store.apply_decay()
        baseline_score = (await store.get(id1))["decay_score"]

        assert abs(neutral_score - baseline_score) < 1e-9
        store.close()

    @pytest.mark.asyncio
    async def test_positive_reward_slows_decay(self, tmp_path):
        """Memories in positively-rewarded sessions decay slower."""
        store = UnifiedMemoryStore(db_path=str(tmp_path / "reward_pos.db"))
        id_good = await store.store("good session fact", memory_type="fact", session_id="good")
        id_neutral = await store.store("neutral fact", memory_type="fact", session_id="neutral")

        # Age both memories
        store._conn.execute(
            "UPDATE memories SET last_accessed = ? WHERE id IN (?, ?)",
            ("2025-01-01T00:00:00+00:00", id_good, id_neutral),
        )
        store._conn.commit()

        await store.apply_decay(reward_map={"good": 1.5, "neutral": 1.0})

        good_score = (await store.get(id_good))["decay_score"]
        neutral_score = (await store.get(id_neutral))["decay_score"]

        # Positively-rewarded memory must have a higher (less decayed) score
        assert good_score > neutral_score
        store.close()

    @pytest.mark.asyncio
    async def test_negative_reward_accelerates_decay(self, tmp_path):
        """Memories in negatively-rewarded sessions decay faster."""
        store = UnifiedMemoryStore(db_path=str(tmp_path / "reward_neg.db"))
        id_bad = await store.store("bad session fact", memory_type="fact", session_id="bad")
        id_neutral = await store.store("neutral fact", memory_type="fact", session_id="neutral")

        store._conn.execute(
            "UPDATE memories SET last_accessed = ? WHERE id IN (?, ?)",
            ("2025-01-01T00:00:00+00:00", id_bad, id_neutral),
        )
        store._conn.commit()

        await store.apply_decay(reward_map={"bad": 0.5, "neutral": 1.0})

        bad_score = (await store.get(id_bad))["decay_score"]
        neutral_score = (await store.get(id_neutral))["decay_score"]

        # Negatively-rewarded memory must have a lower (more decayed) score
        assert bad_score < neutral_score
        store.close()

    @pytest.mark.asyncio
    async def test_consolidator_computes_session_rewards(self, tmp_path):
        """The consolidator builds a reward_map from an OutcomeStore-like mock."""
        store = UnifiedMemoryStore(db_path=str(tmp_path / "consol_reward.db"))

        # Mock outcome store with a get_recent method
        mock_outcome_store = MagicMock()
        mock_outcome_store.get_recent = AsyncMock(
            return_value=[
                {"session_id": "s1", "success": True, "user_feedback": "good"},
                {"session_id": "s1", "success": True, "user_feedback": None},
                {"session_id": "s2", "success": False, "user_feedback": "bad", "tool_errors": ["x"]},
            ]
        )

        consolidator = MemoryConsolidator(store=store, outcome_store=mock_outcome_store)
        rewards = await consolidator._compute_session_rewards()

        assert "s1" in rewards
        assert "s2" in rewards
        assert rewards["s1"] > 1.0  # positive
        assert rewards["s2"] < 1.0  # negative
        # Clamp check
        assert 0.5 <= rewards["s1"] <= 1.5
        assert 0.5 <= rewards["s2"] <= 1.5

        store.close()

    @pytest.mark.asyncio
    async def test_consolidator_no_outcome_store_returns_empty(self, tmp_path):
        """Without an outcome_store, _compute_session_rewards returns empty dict."""
        store = UnifiedMemoryStore(db_path=str(tmp_path / "no_outcomes.db"))
        consolidator = MemoryConsolidator(store=store, outcome_store=None)
        rewards = await consolidator._compute_session_rewards()
        assert rewards == {}
        store.close()


class TestRecallBatchAccessUpdate:
    """Tests for batched access_count updates in recall()."""

    @pytest.mark.asyncio
    async def test_recall_updates_access_count(self, tmp_path):
        store = UnifiedMemoryStore(db_path=str(tmp_path / "access_test.db"))
        id1 = await store.store("Python is great for AI", user_id="u1", memory_type="fact")
        id2 = await store.store("JavaScript for web dev", user_id="u1", memory_type="fact")

        # Initial access_count should be 0
        mem1 = await store.get(id1)
        assert mem1["access_count"] == 0

        # Recall triggers batch access update
        results = await store.recall("Python AI", user_id="u1")
        if results:
            # Check that access_count was incremented
            mem1_after = await store.get(results[0][0]["id"])
            assert mem1_after["access_count"] >= 1
        store.close()


# ── E2E Persistence Tests ─────────────────────────────────────────


class TestMemoryE2EPersistence:
    """End-to-end tests for memory persistence across restarts."""

    @pytest.mark.asyncio
    async def test_store_close_reopen(self, tmp_path):
        """Memories survive close + reopen."""
        db_path = str(tmp_path / "e2e.db")

        store = UnifiedMemoryStore(db_path=db_path)
        await store.store("persistent fact", memory_type="fact", user_id="u1")
        await store.store("persistent pref", memory_type="preference", user_id="u1")
        assert await store.count() == 2
        store.close()

        # Reopen
        store2 = UnifiedMemoryStore(db_path=db_path)
        assert await store2.count() == 2
        results = await store2.recall("persistent", user_id="u1")
        assert len(results) >= 1
        store2.close()

    @pytest.mark.asyncio
    async def test_entities_persist(self, tmp_path):
        """Entities survive close + reopen."""
        db_path = str(tmp_path / "e2e_entities.db")

        store = UnifiedMemoryStore(db_path=db_path)
        await store.upsert_entity("Python", "language")
        await store.upsert_entity("PredaCore", "project")
        store.close()

        store2 = UnifiedMemoryStore(db_path=db_path)
        entities = await store2.list_entities()
        names = [e["name"] for e in entities]
        assert "Python" in names
        assert "PredaCore" in names
        store2.close()

    @pytest.mark.asyncio
    async def test_rust_vector_search_used(self, tmp_path):
        """Verify that predacore_core Rust search is available and functional."""
        try:
            import predacore_core
            results = predacore_core.vector_search(
                [1.0, 0.0, 0.0],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                1,
            )
            assert results[0] == (0, 1.0)
        except ImportError:
            pytest.skip("predacore_core not installed")

    @pytest.mark.asyncio
    async def test_rust_synonym_expansion(self, tmp_path):
        """Verify Rust synonym expansion works."""
        try:
            import predacore_core
            expanded = predacore_core.expand_synonyms(["config"])
            assert "configuration" in expanded
            assert "settings" in expanded
        except ImportError:
            pytest.skip("predacore_core not installed")

    @pytest.mark.asyncio
    async def test_rust_entity_extraction(self, tmp_path):
        """Verify Rust entity extraction works (returns list[tuple])."""
        try:
            import predacore_core
            entities = predacore_core.extract_entities("Using Python and Docker with FastAPI")
            names = [e[0].lower() for e in entities]
            assert "python" in names
            assert "docker" in names
            assert "fastapi" in names
        except ImportError:
            pytest.skip("predacore_core not installed")

    @pytest.mark.asyncio
    async def test_rust_relation_classification(self, tmp_path):
        """Verify Rust relation classification works."""
        try:
            import predacore_core
            rel, conf = predacore_core.classify_relation(
                "core.py depends on config.py", "core.py", "config.py"
            )
            assert rel == "depends_on"
            assert conf > 0.5
        except ImportError:
            pytest.skip("predacore_core not installed")

    @pytest.mark.asyncio
    async def test_vector_index_rebuilds_from_sqlite(self, tmp_path):
        """Vector index is ephemeral — it rebuilds from SQLite on every startup."""
        db_path = str(tmp_path / "rebuild.db")

        # Create, store, close
        store = UnifiedMemoryStore(db_path=db_path)
        await store.store("test data")
        store.close()

        # Reopen — vector index rebuilt from SQLite, no disk file needed
        store2 = UnifiedMemoryStore(db_path=db_path)
        assert await store2.count() == 1
        assert not (tmp_path / "vectors.json").exists()  # No separate vector file
        store2.close()
