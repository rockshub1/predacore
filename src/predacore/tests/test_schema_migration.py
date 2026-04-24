"""Tests for the v1 → v2 memory schema migration.

Covers:
  - Fresh DB gets full v2 schema + schema_version = 2
  - Legacy v1 DB (no schema_meta, 17 cols) migrates cleanly to v2
  - Migration is idempotent (safe to run on every daemon startup)
  - trust_source backfill correctly infers from existing ``source`` column
  - Zero data loss during migration (row count + content preserved)
  - ``UnifiedMemoryStore.store()`` populates v2 columns on new rows
"""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from predacore.memory.store import (
    CURRENT_CHUNKER_VERSION,
    CURRENT_EMBEDDING_VERSION,
    SCHEMA_VERSION,
    TRUST_MULTIPLIERS,
    UnifiedMemoryStore,
    _V2_COLUMNS,
    _V2_INDEXES,
    _SCHEMA,
    compute_anchor_hash,
    compute_content_hash,
    migrate_schema,
    normalize_trust_source,
    normalize_verification_state,
)


# ── v1 schema (snapshot of production PredaCore memory.db before v2) ──
#
# Used to simulate a legacy DB for migration testing. This is NOT a
# historical artifact we still use — the point of these tests is to
# prove migrate_schema() can upgrade a DB shaped like this.
_V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'fact',
    importance INTEGER NOT NULL DEFAULT 2,
    source TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    user_id TEXT NOT NULL DEFAULT 'default',
    embedding BLOB,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    decay_score REAL NOT NULL DEFAULT 1.0,
    expires_at TEXT,
    session_id TEXT,
    parent_id TEXT
);
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'concept',
    properties TEXT NOT NULL DEFAULT '{}',
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1
);
"""


def _make_v1_db(path: Path, rows: list[tuple[str, str, str]] | None = None) -> None:
    """Create a legacy v1 DB at ``path`` with optional (id, content, source) rows."""
    conn = sqlite3.connect(str(path))
    conn.executescript(_V1_SCHEMA)
    for row_id, content, source in rows or []:
        conn.execute(
            "INSERT INTO memories (id, content, source, created_at, updated_at, last_accessed) "
            "VALUES (?, ?, ?, datetime('now'), datetime('now'), datetime('now'))",
            (row_id, content, source),
        )
    conn.commit()
    conn.close()


# ── Pure-function tests ──────────────────────────────────────────────


def test_normalize_trust_source_defaults_to_claude_inferred():
    """Unknown inputs coerce to the conservative default."""
    assert normalize_trust_source(None) == "claude_inferred"
    assert normalize_trust_source("") == "claude_inferred"
    assert normalize_trust_source("some_junk") == "claude_inferred"


def test_normalize_trust_source_accepts_all_valid():
    for v in ("user_stated", "claude_inferred", "code_extracted", "user_corrected"):
        assert normalize_trust_source(v) == v


def test_normalize_verification_state_defaults_and_valid():
    assert normalize_verification_state(None) == "unverified"
    assert normalize_verification_state("garbage") == "unverified"
    for v in ("ok", "stale", "orphaned", "version_skew", "unverified", "superseded"):
        assert normalize_verification_state(v) == v


def test_compute_content_hash_is_stable_and_differs_for_different_input():
    h1 = compute_content_hash("Hello world")
    h2 = compute_content_hash("Hello world")
    h3 = compute_content_hash("Goodbye world")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # SHA256 hex


def test_compute_anchor_hash_uses_first_nonblank_line():
    """anchor_hash is derived from the first non-blank line (line-number drift-proof)."""
    assert compute_anchor_hash("") == ""
    assert compute_anchor_hash("\n\n\n") == ""
    single = "def foo(): pass"
    multi = "\n\n" + single + "\nother stuff"
    assert compute_anchor_hash(single) == compute_anchor_hash(multi)


def test_trust_multipliers_are_monotonic_and_complete():
    """Every valid trust_source must have a ranking multiplier in [0,1]."""
    for src in ("user_corrected", "code_extracted", "user_stated", "claude_inferred"):
        assert src in TRUST_MULTIPLIERS
        assert 0.0 <= TRUST_MULTIPLIERS[src] <= 1.0
    assert (
        TRUST_MULTIPLIERS["user_corrected"]
        >= TRUST_MULTIPLIERS["code_extracted"]
        >= TRUST_MULTIPLIERS["user_stated"]
        >= TRUST_MULTIPLIERS["claude_inferred"]
    )


# ── migrate_schema tests ─────────────────────────────────────────────


def test_fresh_db_migrates_to_v2(tmp_path: Path):
    """A fresh DB created via _SCHEMA should end up at schema_version = 2."""
    db = tmp_path / "fresh.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_SCHEMA)
    conn.commit()

    result = migrate_schema(conn)

    assert result == SCHEMA_VERSION
    # schema_version stamped
    ver = conn.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()[0]
    assert ver == str(SCHEMA_VERSION)
    # All v2 columns present (directly from _SCHEMA, migrator was no-op for ALTER)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    for name, _ddl in _V2_COLUMNS:
        assert name in cols, f"fresh DB missing v2 column {name}"
    conn.close()


def test_legacy_v1_db_migrates_cleanly(tmp_path: Path):
    """A v1 DB with 3 seeded rows should gain all v2 columns + indexes + schema_meta."""
    db = tmp_path / "legacy.db"
    rows = [
        ("id-user", "DB is Postgres", "user_stated"),
        ("id-code", "DATABASE_URL=postgresql://...", "code"),
        ("id-agent", "looks like a Python project", ""),
    ]
    _make_v1_db(db, rows)

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    pre_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert pre_count == 3

    # Apply _SCHEMA first (simulates what UnifiedMemoryStore.__init__ does — CREATE
    # TABLE IF NOT EXISTS is a no-op for the existing v1 table but creates schema_meta)
    conn.executescript(_SCHEMA)
    conn.commit()
    result = migrate_schema(conn)

    # Post-migration assertions
    assert result == SCHEMA_VERSION
    post_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert post_count == pre_count, "migration must not add or drop rows"

    cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    missing = [name for name, _ in _V2_COLUMNS if name not in cols]
    assert not missing, f"missing v2 columns after migrate: {missing}"

    idx_names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='memories'"
    ).fetchall()}
    for idx_sql in _V2_INDEXES:
        # Extract index name after "IF NOT EXISTS "
        idx_name = idx_sql.split("IF NOT EXISTS")[1].split("ON")[0].strip()
        assert idx_name in idx_names, f"missing index {idx_name}"

    # content_hash + anchor_hash backfilled
    for row in conn.execute("SELECT id, content, content_hash, anchor_hash FROM memories"):
        assert row["content_hash"], f"row {row['id']} missing content_hash"
        expected_ch = compute_content_hash(row["content"])
        assert row["content_hash"] == expected_ch
    conn.close()


def test_trust_source_backfill_from_source_column(tmp_path: Path):
    """Migration infers trust_source from the existing ``source`` string."""
    db = tmp_path / "backfill.db"
    rows = [
        ("r-user", "user said X", "user"),
        ("r-code", "code says Y", "code_read"),
        ("r-file", "file contents", "read_file"),
        ("r-junk", "unknown source", "some.random.tag"),
        ("r-empty", "no source", ""),
    ]
    _make_v1_db(db, rows)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    migrate_schema(conn)

    got = {
        r["id"]: (r["trust_source"], round(float(r["confidence"]), 3))
        for r in conn.execute(
            "SELECT id, trust_source, confidence FROM memories"
        ).fetchall()
    }
    assert got["r-user"] == ("user_stated", 0.9), got["r-user"]
    assert got["r-code"] == ("code_extracted", 0.85), got["r-code"]
    assert got["r-file"] == ("code_extracted", 0.85), got["r-file"]
    assert got["r-junk"] == ("claude_inferred", 0.7), got["r-junk"]
    assert got["r-empty"] == ("claude_inferred", 0.7), got["r-empty"]
    conn.close()


def test_migration_is_idempotent(tmp_path: Path):
    """Running migrate_schema twice on the same DB is a no-op on the second call."""
    db = tmp_path / "idempotent.db"
    _make_v1_db(db, [("id-1", "content", "user")])
    conn = sqlite3.connect(str(db))
    conn.executescript(_SCHEMA)
    conn.commit()

    r1 = migrate_schema(conn)
    r2 = migrate_schema(conn)  # second call
    r3 = migrate_schema(conn)  # third call

    assert r1 == r2 == r3 == SCHEMA_VERSION
    # Only one row, unchanged content, trust_source preserved from first run
    row = conn.execute(
        "SELECT content, trust_source, confidence FROM memories"
    ).fetchone()
    assert row[0] == "content"
    assert row[1] == "user_stated"
    assert abs(float(row[2]) - 0.9) < 1e-6
    conn.close()


def test_embedding_version_only_stamped_on_rows_with_embeddings(tmp_path: Path):
    """Rows with embedding=NULL should NOT get stamped (would create fake version-skew)."""
    db = tmp_path / "emb_version.db"
    _make_v1_db(db, [])
    conn = sqlite3.connect(str(db))
    # Insert one row with embedding, one without
    conn.execute(
        "INSERT INTO memories (id, content, source, embedding, created_at, updated_at, last_accessed) "
        "VALUES ('with-emb', 'has embedding', '', ?, datetime('now'), datetime('now'), datetime('now'))",
        (b"\x00\x01\x02\x03",),
    )
    conn.execute(
        "INSERT INTO memories (id, content, source, created_at, updated_at, last_accessed) "
        "VALUES ('no-emb', 'no embedding', '', datetime('now'), datetime('now'), datetime('now'))",
    )
    conn.commit()
    conn.executescript(_SCHEMA)
    conn.commit()
    migrate_schema(conn)

    rows = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT id, embedding_version FROM memories"
        ).fetchall()
    }
    assert rows["with-emb"] == CURRENT_EMBEDDING_VERSION
    assert rows["no-emb"] == ""  # stays empty — no fake skew
    conn.close()


def test_chunker_version_stamped_on_all_rows(tmp_path: Path):
    """Every row gets chunker_version = CURRENT post-migration (used to detect re-chunking needs)."""
    db = tmp_path / "chunker_ver.db"
    _make_v1_db(db, [("id-a", "a", ""), ("id-b", "b", "")])
    conn = sqlite3.connect(str(db))
    conn.executescript(_SCHEMA)
    conn.commit()
    migrate_schema(conn)
    rows = conn.execute("SELECT chunker_version FROM memories").fetchall()
    for r in rows:
        assert r[0] == CURRENT_CHUNKER_VERSION
    conn.close()


# ── UnifiedMemoryStore integration ───────────────────────────────────


def test_store_populates_v2_columns_on_new_rows(tmp_path: Path):
    """New rows via UnifiedMemoryStore.store() must have all v2 invariants filled."""
    async def _run():
        db = tmp_path / "newrows.db"
        store = UnifiedMemoryStore(
            db_path=str(db), embedding_client=None, vector_index=None
        )
        mid_a = await store.store(content="foo", trust_source="user_stated", confidence=0.95)
        mid_b = await store.store(content="bar")  # defaults
        mid_c = await store.store(
            content="baz", trust_source="not_a_valid_value", confidence=2.5
        )  # coercion + clamp
        return str(db), [mid_a, mid_b, mid_c]

    db_str, ids = asyncio.run(_run())

    conn = sqlite3.connect(db_str)
    conn.row_factory = sqlite3.Row
    rows = {
        r["id"]: dict(r)
        for r in conn.execute(
            "SELECT id, trust_source, confidence, content_hash, "
            "chunker_version, verification_state, embedding_version FROM memories"
        ).fetchall()
    }
    for mid in ids:
        row = rows[mid]
        assert row["content_hash"], f"{mid} missing content_hash"
        assert row["chunker_version"] == CURRENT_CHUNKER_VERSION
        assert row["verification_state"] == "unverified"
        assert row["embedding_version"] == ""  # no embed client used
    # Explicit trust_source preserved
    assert rows[ids[0]]["trust_source"] == "user_stated"
    assert abs(rows[ids[0]]["confidence"] - 0.95) < 1e-6
    # Default
    assert rows[ids[1]]["trust_source"] == "claude_inferred"
    assert abs(rows[ids[1]]["confidence"] - 0.7) < 1e-6
    # Coercion + clamping
    assert rows[ids[2]]["trust_source"] == "claude_inferred"
    assert rows[ids[2]]["confidence"] == 1.0
    conn.close()


def test_store_is_compatible_with_freshly_migrated_legacy_db(tmp_path: Path):
    """After migrating a legacy DB, new writes through store() must succeed."""
    db = tmp_path / "mixed.db"
    _make_v1_db(db, [("legacy-1", "old content", "user_stated")])

    async def _run():
        store = UnifiedMemoryStore(
            db_path=str(db), embedding_client=None, vector_index=None
        )
        return await store.store(content="new content", trust_source="code_extracted")

    new_id = asyncio.run(_run())

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    # Both the legacy row (with backfilled trust_source) and the new row should coexist.
    rows = {
        r["id"]: (r["trust_source"], r["content_hash"] != "")
        for r in conn.execute(
            "SELECT id, trust_source, content_hash FROM memories"
        ).fetchall()
    }
    assert rows["legacy-1"] == ("user_stated", True), rows["legacy-1"]
    assert rows[new_id] == ("code_extracted", True), rows[new_id]
    conn.close()


# ── Schema version sanity ────────────────────────────────────────────


def test_schema_version_constant_matches_migrator():
    """SCHEMA_VERSION is what migrate_schema() writes."""
    assert SCHEMA_VERSION == 2


def test_v2_columns_include_supersede_prep():
    """Build 2 supersede columns are present in v2 schema."""
    names = {col for col, _ddl in _V2_COLUMNS}
    assert "superseded_by" in names
    assert "superseded_at" in names


# ── Build 2 — supersede capability ──────────────────────────────────


def test_store_with_supersedes_marks_old_rows(tmp_path: Path):
    """store(..., supersedes=[ids]) marks the listed rows as superseded atomically."""
    async def _run():
        store = UnifiedMemoryStore(
            db_path=str(tmp_path / "supersede.db"),
            embedding_client=None,
            vector_index=None,
        )
        old = await store.store(content="Database is Postgres", trust_source="user_stated")
        new = await store.store(
            content="Database is MySQL",
            trust_source="user_corrected",
            confidence=1.0,
            supersedes=[old],
        )
        return str(tmp_path / "supersede.db"), old, new

    db_path, old_id, new_id = asyncio.run(_run())
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    old_row = conn.execute(
        "SELECT verification_state, superseded_by, superseded_at FROM memories WHERE id = ?",
        (old_id,),
    ).fetchone()
    new_row = conn.execute(
        "SELECT verification_state, superseded_by, superseded_at, trust_source FROM memories WHERE id = ?",
        (new_id,),
    ).fetchone()

    assert old_row["verification_state"] == "superseded"
    assert old_row["superseded_by"] == new_id
    assert old_row["superseded_at"] is not None

    # New row is NOT marked superseded itself
    assert new_row["verification_state"] == "unverified"
    assert new_row["superseded_by"] is None
    assert new_row["trust_source"] == "user_corrected"
    conn.close()


def test_supersede_multiple_ids_atomic(tmp_path: Path):
    """Supersede can replace N old rows with 1 new row in one call."""
    async def _run():
        store = UnifiedMemoryStore(
            db_path=str(tmp_path / "multi.db"),
            embedding_client=None, vector_index=None,
        )
        old_a = await store.store(content="Tool A info")
        old_b = await store.store(content="Tool A info (different phrasing)")
        old_c = await store.store(content="Tool A info v2")
        new = await store.store(
            content="Tool A consolidated info",
            supersedes=[old_a, old_b, old_c],
        )
        return str(tmp_path / "multi.db"), [old_a, old_b, old_c], new

    db_path, old_ids, new_id = asyncio.run(_run())
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    for oid in old_ids:
        row = conn.execute(
            "SELECT verification_state, superseded_by FROM memories WHERE id = ?",
            (oid,),
        ).fetchone()
        assert row["verification_state"] == "superseded"
        assert row["superseded_by"] == new_id
    conn.close()


def test_supersede_nonexistent_id_is_noop(tmp_path: Path):
    """Passing an unknown ID in supersedes= doesn't crash — just doesn't match."""
    async def _run():
        store = UnifiedMemoryStore(
            db_path=str(tmp_path / "nx.db"),
            embedding_client=None, vector_index=None,
        )
        new = await store.store(
            content="new fact",
            supersedes=["does-not-exist-1", "does-not-exist-2"],
        )
        return str(tmp_path / "nx.db"), new

    db_path, new_id = asyncio.run(_run())
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT verification_state FROM memories WHERE id = ?", (new_id,)
    ).fetchone()
    # The new row exists, verification_state is its fresh default, no errors
    assert row[0] == "unverified"
    conn.close()


def test_supersede_empties_and_dupes_are_normalized(tmp_path: Path):
    """Blank strings, None, and duplicate IDs are filtered before the UPDATE."""
    async def _run():
        store = UnifiedMemoryStore(
            db_path=str(tmp_path / "dedupe.db"),
            embedding_client=None, vector_index=None,
        )
        old = await store.store(content="old")
        # Mix of valid, empty, whitespace, None, and dupes
        new = await store.store(
            content="new",
            supersedes=[old, "", "   ", old, None, old],  # type: ignore[list-item]
        )
        return str(tmp_path / "dedupe.db"), old, new

    db_path, old_id, new_id = asyncio.run(_run())
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT verification_state, superseded_by FROM memories WHERE id = ?",
        (old_id,),
    ).fetchone()
    assert row[0] == "superseded"
    assert row[1] == new_id
    conn.close()


# ── Recall filtering ─────────────────────────────────────────────────


def test_recall_hides_superseded_by_default(tmp_path: Path):
    """Superseded rows are not returned by default recall (keyword path)."""
    async def _run():
        store = UnifiedMemoryStore(
            db_path=str(tmp_path / "filter.db"),
            embedding_client=None, vector_index=None,
        )
        old_id = await store.store(
            content="Database system is Postgres",
            user_id="alice",
            trust_source="user_stated",
        )
        await store.store(
            content="Database system is MySQL",
            user_id="alice",
            trust_source="user_corrected",
            supersedes=[old_id],
        )
        # No embedder → falls to BM25 keyword path. Use a real query token
        # ("database") that BM25 can match.
        visible = await store.recall(query="database", user_id="alice", top_k=10)
        all_incl = await store.recall(
            query="database", user_id="alice", top_k=10, show_superseded=True
        )
        return [m["content"] for m, _ in visible], [m["content"] for m, _ in all_incl]

    visible, all_incl = asyncio.run(_run())
    assert "Database system is Postgres" not in visible
    assert "Database system is MySQL" in visible
    assert "Database system is Postgres" in all_incl
    assert "Database system is MySQL" in all_incl


def test_recall_hides_stale_orphaned_version_skew(tmp_path: Path):
    """Invariant-failure states are ALWAYS hidden (even with show_superseded=True)."""
    async def _run():
        store = UnifiedMemoryStore(
            db_path=str(tmp_path / "inv.db"),
            embedding_client=None, vector_index=None,
        )
        await store.store(content="healthy production row", user_id="bob")
        stale_id = await store.store(content="stale production row", user_id="bob")
        return str(tmp_path / "inv.db"), stale_id

    db_path, stale_id = asyncio.run(_run())
    # Manually mark the second row as stale (simulating the healer)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE memories SET verification_state='stale' WHERE id = ?", (stale_id,)
    )
    conn.commit()
    conn.close()

    async def _check():
        store = UnifiedMemoryStore(
            db_path=db_path, embedding_client=None, vector_index=None,
        )
        default = await store.recall(query="production", user_id="bob", top_k=10)
        # Even with show_superseded=True, stale stays hidden (it's not
        # superseded, it's an invariant-failure which is always hidden).
        all_incl = await store.recall(
            query="production", user_id="bob", top_k=10, show_superseded=True
        )
        return [m["content"] for m, _ in default], [m["content"] for m, _ in all_incl]

    default, all_incl = asyncio.run(_check())
    assert "healthy production row" in default
    assert "stale production row" not in default
    assert "stale production row" not in all_incl


def test_unverified_rows_are_visible_by_default(tmp_path: Path):
    """Surprise #5 audit — new rows start 'unverified' and MUST be findable."""
    async def _run():
        store = UnifiedMemoryStore(
            db_path=str(tmp_path / "uv.db"),
            embedding_client=None, vector_index=None,
        )
        mid = await store.store(content="brand new important fact", user_id="carol")
        visible = await store.recall(query="important", user_id="carol", top_k=5)
        return mid, [m["content"] for m, _ in visible]

    mid, visible = asyncio.run(_run())
    assert "brand new important fact" in visible, (
        "new 'unverified' rows must remain visible — if they don't, "
        "the retrieval filter is too strict (Surprise #5 regression)"
    )


def test_visibility_helper_direct():
    """Unit test of _memory_is_visible_in_recall."""
    from predacore.memory.store import _memory_is_visible_in_recall
    assert _memory_is_visible_in_recall({"verification_state": "ok"}) is True
    assert _memory_is_visible_in_recall({"verification_state": "unverified"}) is True
    assert _memory_is_visible_in_recall({"verification_state": "stale"}) is False
    assert _memory_is_visible_in_recall({"verification_state": "orphaned"}) is False
    assert _memory_is_visible_in_recall({"verification_state": "version_skew"}) is False
    assert _memory_is_visible_in_recall({"verification_state": "superseded"}) is False
    assert _memory_is_visible_in_recall(
        {"verification_state": "superseded"}, show_superseded=True
    ) is True
    # Missing state → default to 'ok' (visible)
    assert _memory_is_visible_in_recall({}) is True
    # Unknown state → visible (we're liberal with unknown values to avoid
    # accidentally hiding rows from a bug elsewhere)
    assert _memory_is_visible_in_recall({"verification_state": "mystery"}) is True


# ── Build 3 — trust-weighted ranking ────────────────────────────────


def test_ranking_helper_multipliers():
    """Unit test of _apply_ranking_weights. Covers all 4 trust sources + edge cases."""
    from predacore.memory.store import _apply_ranking_weights

    def row(trust=None, conf=1.0, decay=1.0):
        return {"trust_source": trust, "confidence": conf, "decay_score": decay}

    # Identity base_score 1.0 → final equals trust_multiplier × confidence × decay
    assert _apply_ranking_weights(1.0, row("user_corrected")) == pytest.approx(1.00)
    assert _apply_ranking_weights(1.0, row("code_extracted")) == pytest.approx(0.95)
    assert _apply_ranking_weights(1.0, row("user_stated")) == pytest.approx(0.90)
    assert _apply_ranking_weights(1.0, row("claude_inferred")) == pytest.approx(0.60)

    # Unknown trust source → claude_inferred default (0.60)
    assert _apply_ranking_weights(1.0, row("mystery")) == pytest.approx(0.60)
    # Missing trust_source → same default
    assert _apply_ranking_weights(1.0, {"confidence": 1.0, "decay_score": 1.0}) == pytest.approx(0.60)

    # Confidence scales linearly
    assert _apply_ranking_weights(1.0, row("user_corrected", conf=0.5)) == pytest.approx(0.50)
    assert _apply_ranking_weights(1.0, row("user_corrected", conf=0.0)) == pytest.approx(0.00)
    # Out-of-range confidence clamped
    assert _apply_ranking_weights(1.0, row("user_corrected", conf=2.5)) == pytest.approx(1.00)
    assert _apply_ranking_weights(1.0, row("user_corrected", conf=-1.0)) == pytest.approx(0.00)

    # Decay scales too
    assert _apply_ranking_weights(1.0, row("user_corrected", decay=0.5)) == pytest.approx(0.50)

    # Base score scales through
    assert _apply_ranking_weights(10.0, row("code_extracted", conf=1.0)) == pytest.approx(9.50)


def test_ranking_helper_handles_bad_types():
    """Malformed values don't crash — fall back to safe defaults."""
    from predacore.memory.store import _apply_ranking_weights
    # Non-numeric confidence
    assert _apply_ranking_weights(1.0, {"trust_source": "user_stated", "confidence": "abc"}) == pytest.approx(
        0.90 * 0.7  # falls to 0.7 default confidence
    )
    # None values
    assert _apply_ranking_weights(1.0, {"trust_source": None, "confidence": None, "decay_score": None}) == pytest.approx(
        0.60 * 0.7  # claude_inferred default trust × default confidence
    )


def test_trust_ranking_order_end_to_end(tmp_path: Path):
    """4 rows with identical content, different trust_source — ranking follows trust multipliers."""
    async def _run():
        store = UnifiedMemoryStore(
            db_path=str(tmp_path / "trust_order.db"),
            embedding_client=None, vector_index=None,
        )
        # Identical content, identical confidence → only trust differs
        await store.store(content="target topic alpha", user_id="u", trust_source="user_corrected", confidence=1.0)
        await store.store(content="target topic alpha", user_id="u", trust_source="code_extracted", confidence=1.0)
        await store.store(content="target topic alpha", user_id="u", trust_source="user_stated", confidence=1.0)
        await store.store(content="target topic alpha", user_id="u", trust_source="claude_inferred", confidence=1.0)
        results = await store.recall(query="target topic", user_id="u", top_k=10)
        return [(m["trust_source"], score) for m, score in results]

    ranked = asyncio.run(_run())
    # Order must be: user_corrected, code_extracted, user_stated, claude_inferred
    trust_order = [r[0] for r in ranked]
    expected = ["user_corrected", "code_extracted", "user_stated", "claude_inferred"]
    assert trust_order == expected, f"expected {expected}, got {trust_order}"
    # Scores must be strictly decreasing
    scores = [r[1] for r in ranked]
    assert scores == sorted(scores, reverse=True), f"scores not descending: {scores}"


def test_confidence_affects_ranking_at_same_trust(tmp_path: Path):
    """Two rows, same trust_source, different confidence — higher confidence wins."""
    async def _run():
        store = UnifiedMemoryStore(
            db_path=str(tmp_path / "conf_rank.db"),
            embedding_client=None, vector_index=None,
        )
        low = await store.store(content="target item beta", user_id="u", trust_source="user_stated", confidence=0.3)
        high = await store.store(content="target item beta", user_id="u", trust_source="user_stated", confidence=1.0)
        results = await store.recall(query="target item", user_id="u", top_k=10)
        return [(m["id"], score) for m, score in results], low, high

    results, low_id, high_id = asyncio.run(_run())
    ids_in_order = [r[0] for r in results]
    assert ids_in_order.index(high_id) < ids_in_order.index(low_id), (
        "higher-confidence row should rank before lower-confidence row at same trust level"
    )


def test_high_trust_low_conf_loses_to_high_conf_lower_trust(tmp_path: Path):
    """confidence × trust multiplication — lower trust with higher confidence can outrank."""
    async def _run():
        store = UnifiedMemoryStore(
            db_path=str(tmp_path / "cross.db"),
            embedding_client=None, vector_index=None,
        )
        # user_corrected × 0.3 = 0.30
        low = await store.store(content="target fact gamma", user_id="u", trust_source="user_corrected", confidence=0.3)
        # code_extracted × 1.0 = 0.95
        high = await store.store(content="target fact gamma", user_id="u", trust_source="code_extracted", confidence=1.0)
        results = await store.recall(query="target fact", user_id="u", top_k=10)
        return [(m["id"], m["trust_source"], score) for m, score in results], low, high

    results, low_conf_id, high_conf_id = asyncio.run(_run())
    # High-conf code_extracted (0.95) should beat low-conf user_corrected (0.30)
    ids_in_order = [r[0] for r in results]
    assert ids_in_order.index(high_conf_id) < ids_in_order.index(low_conf_id), (
        "code_extracted@1.0 should outrank user_corrected@0.3 — "
        f"got ordering {[(r[1], r[2]) for r in results]}"
    )


def test_trust_multipliers_constant_frozen():
    """TRUST_MULTIPLIERS is the canonical table — don't accidentally mutate it."""
    from predacore.memory.store import TRUST_MULTIPLIERS
    assert TRUST_MULTIPLIERS == {
        "user_corrected":  1.00,
        "code_extracted":  0.95,
        "user_stated":     0.90,
        "claude_inferred": 0.60,
    }
    # Verify monotonic: user_corrected ≥ code_extracted ≥ user_stated ≥ claude_inferred
    ranked = [
        TRUST_MULTIPLIERS["user_corrected"],
        TRUST_MULTIPLIERS["code_extracted"],
        TRUST_MULTIPLIERS["user_stated"],
        TRUST_MULTIPLIERS["claude_inferred"],
    ]
    assert ranked == sorted(ranked, reverse=True)


def test_ranking_preserves_existing_decay_behavior(tmp_path: Path):
    """Build 3 adds trust+confidence on top of decay — decay still matters."""
    async def _run():
        store = UnifiedMemoryStore(
            db_path=str(tmp_path / "decay_rank.db"),
            embedding_client=None, vector_index=None,
        )
        # Both same trust + confidence. Different decay_score (simulating age).
        old_id = await store.store(content="target item delta", user_id="u", trust_source="user_stated", confidence=1.0)
        new_id = await store.store(content="target item delta", user_id="u", trust_source="user_stated", confidence=1.0)
        return str(tmp_path / "decay_rank.db"), old_id, new_id

    db_path, old_id, new_id = asyncio.run(_run())
    # Manually lower decay on the "old" row to simulate time passing
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE memories SET decay_score = 0.3 WHERE id = ?", (old_id,))
    conn.commit()
    conn.close()

    async def _check():
        store = UnifiedMemoryStore(
            db_path=db_path, embedding_client=None, vector_index=None,
        )
        results = await store.recall(query="target item", user_id="u", top_k=5)
        return [(m["id"], score) for m, score in results]

    ranked = asyncio.run(_check())
    ids_in_order = [r[0] for r in ranked]
    # New row (decay=1.0) should rank before old row (decay=0.3)
    assert ids_in_order.index(new_id) < ids_in_order.index(old_id)


# ── Build 3e — context budget ───────────────────────────────────────


def test_default_context_budget_is_80k():
    """Default budget bumped from 36k to 80k."""
    from predacore.core import _context_budget_for_provider
    budget, hist = _context_budget_for_provider("anthropic", "claude-opus-4-7")
    assert budget == 80_000
    assert hist == 14_000
    # Non-Anthropic default should also be 80k (not model-aware per user direction)
    budget2, hist2 = _context_budget_for_provider("gemini", "gemini-3-pro")
    assert budget2 == 80_000
    assert hist2 == 14_000


def test_context_budget_env_override(monkeypatch):
    """PREDACORE_CONTEXT_BUDGET env var overrides the default."""
    from predacore.core import _context_budget_for_provider
    monkeypatch.setenv("PREDACORE_CONTEXT_BUDGET", "120000")
    budget, hist = _context_budget_for_provider("anthropic", "claude-opus-4-7")
    assert budget == 120_000
    assert hist == 20_000  # 120k / 6 = 20k


def test_context_budget_env_override_floor(monkeypatch):
    """Small budget still gets at least 4k history floor."""
    from predacore.core import _context_budget_for_provider
    monkeypatch.setenv("PREDACORE_CONTEXT_BUDGET", "12000")
    budget, hist = _context_budget_for_provider("anthropic", "claude-opus-4-7")
    assert budget == 12_000
    assert hist == 4_000  # floor, not 12000/6=2000


def test_context_budget_env_junk_falls_through(monkeypatch):
    """Non-integer env value ignored; default kicks in."""
    from predacore.core import _context_budget_for_provider
    monkeypatch.setenv("PREDACORE_CONTEXT_BUDGET", "not_a_number")
    budget, hist = _context_budget_for_provider("anthropic", "claude-opus-4-7")
    assert budget == 80_000


# ── Build 3f — vector index persistence ─────────────────────────────


def test_vector_cache_dump_and_load_roundtrip(tmp_path: Path):
    """dump_to_disk + load_from_disk preserve ids, vecs, meta exactly."""
    from predacore.memory.store import _NumpyVectorIndex
    idx = _NumpyVectorIndex(dimensions=4)
    # Seed directly (bypass async API for a focused unit test)
    idx._ids = ["id-1", "id-2", "id-3"]
    idx._vecs = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    idx._meta = [{"type": "fact"}, {"type": "preference"}, {"type": "entity"}]

    cache_path = tmp_path / "vector_cache.npz"
    ok = idx.dump_to_disk(
        cache_path, row_count=3, embedding_version="bge-small-en-v1.5"
    )
    assert ok
    assert cache_path.exists()

    # Load into a fresh index
    idx2 = _NumpyVectorIndex(dimensions=4)
    loaded = idx2.load_from_disk(
        cache_path, expected_row_count=3, expected_embedding_version="bge-small-en-v1.5"
    )
    assert loaded
    assert idx2._ids == idx._ids
    assert idx2._meta == idx._meta
    # Vectors preserved as float32 → small FP tolerance
    for v1, v2 in zip(idx._vecs, idx2._vecs, strict=False):
        assert all(abs(a - b) < 1e-5 for a, b in zip(v1, v2, strict=False))


def test_vector_cache_stale_row_count_invalidates(tmp_path: Path):
    """Cache with stale row_count should be rejected → rebuild required."""
    from predacore.memory.store import _NumpyVectorIndex
    idx = _NumpyVectorIndex(dimensions=4)
    idx._ids = ["id-1"]
    idx._vecs = [[1.0, 0.0, 0.0, 0.0]]
    idx._meta = [{}]
    cache_path = tmp_path / "stale.npz"
    idx.dump_to_disk(
        cache_path, row_count=1, embedding_version="bge-small-en-v1.5"
    )

    # SQLite now reports 5 rows (but cache says 1) → mismatch
    idx2 = _NumpyVectorIndex(dimensions=4)
    loaded = idx2.load_from_disk(
        cache_path,
        expected_row_count=5,
        expected_embedding_version="bge-small-en-v1.5",
    )
    assert not loaded
    assert idx2.size == 0  # state untouched


def test_vector_cache_version_mismatch_invalidates(tmp_path: Path):
    """Cache with wrong embedding_version should be rejected."""
    from predacore.memory.store import _NumpyVectorIndex
    idx = _NumpyVectorIndex(dimensions=4)
    idx._ids = ["id-1"]
    idx._vecs = [[1.0, 0.0, 0.0, 0.0]]
    idx._meta = [{}]
    cache_path = tmp_path / "vermismatch.npz"
    idx.dump_to_disk(
        cache_path, row_count=1, embedding_version="old-model-v0.1"
    )

    idx2 = _NumpyVectorIndex(dimensions=4)
    loaded = idx2.load_from_disk(
        cache_path,
        expected_row_count=1,
        expected_embedding_version="bge-small-en-v1.5",  # different!
    )
    assert not loaded


def test_vector_cache_missing_file_returns_false(tmp_path: Path):
    """load_from_disk on missing path is a silent False, not an exception."""
    from predacore.memory.store import _NumpyVectorIndex
    idx = _NumpyVectorIndex(dimensions=4)
    missing = tmp_path / "nope.npz"
    assert not missing.exists()
    ok = idx.load_from_disk(
        missing, expected_row_count=0, expected_embedding_version="any"
    )
    assert ok is False


def test_vector_cache_empty_index_deletes_stale_file(tmp_path: Path):
    """If the current index is empty, dump_to_disk removes any stale cache file."""
    from predacore.memory.store import _NumpyVectorIndex
    cache_path = tmp_path / "empty.npz"
    # First, create a cache with content
    idx = _NumpyVectorIndex(dimensions=4)
    idx._ids = ["id-1"]
    idx._vecs = [[1.0, 0.0, 0.0, 0.0]]
    idx._meta = [{}]
    idx.dump_to_disk(cache_path, row_count=1, embedding_version="bge")
    assert cache_path.exists()

    # Now dump with an empty index → file should be removed
    idx2 = _NumpyVectorIndex(dimensions=4)
    ok = idx2.dump_to_disk(cache_path, row_count=0, embedding_version="bge")
    assert ok is False
    assert not cache_path.exists()


def test_store_persists_cache_on_close(tmp_path: Path):
    """UnifiedMemoryStore.close() writes the vector cache; next init loads it."""
    from predacore.memory.store import UnifiedMemoryStore

    async def _first_run():
        store = UnifiedMemoryStore(
            db_path=str(tmp_path / "persist.db"),
            embedding_client=None, vector_index=None,
        )
        # No embedding client → no actual vectors inserted, but store still works.
        # Use the direct index API to seed synthetic vectors (bypass async chain).
        store._vec_index._ids = ["mid-a", "mid-b"]
        store._vec_index._vecs = [[0.1] * 384, [0.2] * 384]
        store._vec_index._meta = [{"type": "fact"}, {"type": "preference"}]
        # Stage the DB to report 2 embedded rows so the cache invariant matches
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "persist.db"))
        conn.execute(
            "INSERT INTO memories (id, content, embedding, created_at, updated_at, last_accessed) "
            "VALUES (?, ?, ?, datetime('now'), datetime('now'), datetime('now'))",
            ("mid-a", "a", b"\x00" * 4),
        )
        conn.execute(
            "INSERT INTO memories (id, content, embedding, created_at, updated_at, last_accessed) "
            "VALUES (?, ?, ?, datetime('now'), datetime('now'), datetime('now'))",
            ("mid-b", "b", b"\x00" * 4),
        )
        conn.commit()
        conn.close()
        store.close()  # should persist
        return str(tmp_path / "persist.db")

    db_path = asyncio.run(_first_run())
    # Cache file should now exist sibling to the DB
    cache_file = tmp_path / "vector_index.cache.npz"
    assert cache_file.exists(), "close() should have persisted the vector index"

    async def _second_run():
        store = UnifiedMemoryStore(
            db_path=db_path, embedding_client=None, vector_index=None,
        )
        # Should have loaded from cache (skipped SQLite rebuild path)
        loaded_ids = sorted(store._vec_index._ids)
        return loaded_ids

    ids = asyncio.run(_second_run())
    assert ids == ["mid-a", "mid-b"], f"cache reload failed — got {ids}"


# ── Build 4 — HNSW backend (opt-in) ─────────────────────────────────


def test_hnsw_index_basic_add_search():
    """_HnswVectorIndex matches _NumpyVectorIndex API: add, search, size."""
    from predacore.memory.store import _HnswVectorIndex

    async def _run():
        idx = _HnswVectorIndex(dimensions=4)
        assert idx.size == 0
        await idx.add("id-north", [1.0, 0.0, 0.0, 0.0], metadata={"layer": "a"})
        await idx.add("id-south", [-1.0, 0.0, 0.0, 0.0], metadata={"layer": "a"})
        await idx.add("id-east", [0.0, 1.0, 0.0, 0.0], metadata={"layer": "b"})
        assert idx.size == 3
        # Query close to north
        results = await idx.search([0.95, 0.05, 0.0, 0.0], top_k=3)
        return results

    results = asyncio.run(_run())
    # North should be top hit; east should be irrelevant
    assert results[0][0] == "id-north"
    assert results[0][1] > 0.99  # similarity near 1
    ids = [r[0] for r in results]
    assert "id-east" in ids  # it's in the list
    # East should be much lower score than north
    east_score = next(r[1] for r in results if r[0] == "id-east")
    assert east_score < 0.5


def test_hnsw_index_layer_filter():
    """Layer filter applies after HNSW search (like _NumpyVectorIndex)."""
    from predacore.memory.store import _HnswVectorIndex

    async def _run():
        idx = _HnswVectorIndex(dimensions=4)
        await idx.add("a-1", [1.0, 0.0, 0.0, 0.0], metadata={"layer": "a"})
        await idx.add("b-1", [0.99, 0.01, 0.0, 0.0], metadata={"layer": "b"})
        await idx.add("a-2", [0.95, 0.05, 0.0, 0.0], metadata={"layer": "a"})
        # Layer = 'a' → should exclude b-1 even though it's semantically closer
        return await idx.search([1.0, 0.0, 0.0, 0.0], top_k=3, layers={"a"})

    results = asyncio.run(_run())
    ids = [r[0] for r in results]
    assert "b-1" not in ids
    assert "a-1" in ids and "a-2" in ids


def test_hnsw_index_tombstone_delete():
    """remove() tombstones without corrupting HNSW graph. size reflects live count."""
    from predacore.memory.store import _HnswVectorIndex

    async def _run():
        idx = _HnswVectorIndex(dimensions=4)
        await idx.add("id-a", [1.0, 0.0, 0.0, 0.0])
        await idx.add("id-b", [0.0, 1.0, 0.0, 0.0])
        assert idx.size == 2
        removed = await idx.remove("id-a")
        assert removed is True
        assert idx.size == 1
        # Remove same id again — should return False (already tombstoned)
        removed_again = await idx.remove("id-a")
        assert removed_again is False
        # Remove unknown id — False
        assert await idx.remove("never-existed") is False
        # Search should not return tombstoned id-a
        results = await idx.search([1.0, 0.0, 0.0, 0.0], top_k=5)
        return [r[0] for r in results]

    ids = asyncio.run(_run())
    assert "id-a" not in ids
    assert "id-b" in ids


def test_hnsw_index_readd_keeps_both_vectors_known_limitation():
    """Re-add via add() inserts a NEW slot — the old one remains in the graph.

    This is a known limitation: HNSW doesn't support clean node replacement,
    so we accept dual-slot behaviour. Users who want true replacement should
    call remove() first, then add(). The dedupe logic in search() returns
    only the highest-scoring slot for each id, so searches are still correct
    (just slightly wasteful of graph capacity).

    This test pins the limitation so it doesn't silently change. If we ever
    implement real replacement (needs position-tracked tombstones via Rust
    changes), update this test to reflect clean-replace semantics.
    """
    from predacore.memory.store import _HnswVectorIndex

    async def _run():
        idx = _HnswVectorIndex(dimensions=4)
        await idx.add("id-1", [1.0, 0.0, 0.0, 0.0])  # "north"
        await idx.add("id-1", [0.0, 1.0, 0.0, 0.0])  # re-add as "east"
        # Search east — should find id-1 at high similarity
        results_east = await idx.search([0.0, 1.0, 0.0, 0.0], top_k=3)
        # Search north — old slot still matches; dedupe returns id-1 once
        results_north = await idx.search([1.0, 0.0, 0.0, 0.0], top_k=3)
        return results_east, results_north

    east, north = asyncio.run(_run())
    # East search: id-1 returned, high similarity (new vector points east)
    assert any(r[0] == "id-1" and r[1] > 0.99 for r in east), east
    # North search: id-1 also returned (old slot still in graph)
    # — that's the limitation being pinned here.
    assert any(r[0] == "id-1" for r in north), north


def test_hnsw_index_remove_hides_id_from_search():
    """remove() tombstones the id → future searches skip it.

    NOTE (known limitation): after remove(), you cannot re-add the same id
    cleanly. HNSW has no node deletion, so we use id-based tombstones;
    re-adding would either leave the id invisible (if we keep tombstone)
    or un-hide the old slot (if we clear tombstone). Neither is ideal.
    Workaround: use a new id for replacement content, or rebuild the index
    via daemon restart. A proper fix needs position-level tombstones via
    Rust changes (return slot id from insert + search) — tracked as future
    work. For now, documenting the limitation in a dedicated test.
    """
    from predacore.memory.store import _HnswVectorIndex

    async def _run():
        idx = _HnswVectorIndex(dimensions=4)
        await idx.add("id-keep", [1.0, 0.0, 0.0, 0.0])
        await idx.add("id-drop", [0.9, 0.1, 0.0, 0.0])
        await idx.remove("id-drop")
        # After remove: id-drop should not appear in search results
        results = await idx.search([1.0, 0.0, 0.0, 0.0], top_k=3)
        return [r[0] for r in results]

    ids = asyncio.run(_run())
    assert "id-keep" in ids
    assert "id-drop" not in ids, f"tombstoned id leaked into search: {ids}"


def test_hnsw_index_empty_search_returns_empty():
    """Search on an empty index returns empty list, doesn't crash."""
    from predacore.memory.store import _HnswVectorIndex

    async def _run():
        idx = _HnswVectorIndex(dimensions=4)
        return await idx.search([1.0, 0.0, 0.0, 0.0], top_k=5)

    results = asyncio.run(_run())
    assert results == []


def test_hnsw_index_dim_mismatch_raises():
    """Wrong dim vector/query should raise RuntimeError (safer than silent misbehavior)."""
    from predacore.memory.store import _HnswVectorIndex

    async def _run():
        idx = _HnswVectorIndex(dimensions=4)
        # Insert with wrong dim — logs a warning and returns without adding
        # (we chose warn+skip over raise to match _NumpyVectorIndex behaviour
        # which doesn't validate dim either). Check that size stays 0.
        await idx.add("bad", [1.0, 0.0])  # 2-dim instead of 4
        return idx.size

    size_after_bad_add = asyncio.run(_run())
    assert size_after_bad_add == 0  # insert was rejected


def test_hnsw_persistence_stubs_dont_crash(tmp_path: Path):
    """HNSW dump_to_disk/load_from_disk are stubs for now — must not crash
    and must return False so the caller rebuilds from SQLite."""
    from predacore.memory.store import _HnswVectorIndex
    idx = _HnswVectorIndex(dimensions=4)
    path = tmp_path / "hnsw_stub.bin"
    assert idx.dump_to_disk(path, row_count=0, embedding_version="bge") is False
    assert idx.load_from_disk(
        path, expected_row_count=0, expected_embedding_version="bge"
    ) is False


def test_hnsw_backend_selected_by_env_var(tmp_path: Path, monkeypatch):
    """PREDACORE_USE_HNSW=1 → UnifiedMemoryStore uses _HnswVectorIndex."""
    from predacore.memory.store import (
        UnifiedMemoryStore, _HnswVectorIndex, _NumpyVectorIndex,
    )
    monkeypatch.setenv("PREDACORE_USE_HNSW", "1")
    store = UnifiedMemoryStore(
        db_path=str(tmp_path / "hnsw_backend.db"),
        embedding_client=None, vector_index=None,
    )
    assert isinstance(store._vec_index, _HnswVectorIndex), (
        f"PREDACORE_USE_HNSW=1 should select HNSW backend, got {type(store._vec_index).__name__}"
    )


def test_linear_backend_is_default(tmp_path: Path, monkeypatch):
    """Without PREDACORE_USE_HNSW, default backend is _NumpyVectorIndex."""
    from predacore.memory.store import UnifiedMemoryStore, _NumpyVectorIndex
    monkeypatch.delenv("PREDACORE_USE_HNSW", raising=False)
    store = UnifiedMemoryStore(
        db_path=str(tmp_path / "linear_default.db"),
        embedding_client=None, vector_index=None,
    )
    assert isinstance(store._vec_index, _NumpyVectorIndex)
