"""T2 — Method-level tests for UnifiedMemoryStore augmentations
added in this session's memory upgrade:

  • B1  — scan_for_secrets ingress
  • B2a — store() +6 new kwargs (project_id, branch, source_path, etc.)
  • B2b — reindex_file()
  • B3  — sync_git_changes()
  • B4  — warmup_embedder()
  • B5  — sync_git_changes(prior_head=...) closes committed-changes gap
  • B6  — recall_explain() (sophisticated per-stage trace, L4)
  • L4  — _invariant_skips counter + reset_invariant_skips()
  • L5  — project_id filter on recall() + project_mismatch counter

Uses real BGE via the session-scoped bge_embedder fixture. Most tests
use the per-test memory_store fixture for isolation.
"""
from __future__ import annotations

import sqlite3
import subprocess
import time
from pathlib import Path

import pytest


# ═════════════════════════════════════════════════════════════════════
# B1 — INGRESS SECRET SCAN
# ═════════════════════════════════════════════════════════════════════


class TestIngressSecretScan:
    """Content with secrets is REFUSED at store() — no row inserted."""

    async def test_secret_content_returns_empty_id(self, memory_store):
        """A row containing a recognizable secret returns "" instead of a UUID."""
        secret = "github_token=ghp_AAAAaaaa1111BBBBbbbb2222CCCCcccc3333DD"
        rid = await memory_store.store(content=secret, memory_type="note")
        assert rid == ""

    async def test_secret_content_increments_safety_counter(self, memory_store):
        """The block is recorded in self._safety_stats by_kind."""
        await memory_store.store(
            content="aws AKIAIOSFODNN7EXAMPLE leaked",
            memory_type="note",
        )
        stats = memory_store._safety_stats.as_dict()
        assert stats["secrets_blocked"] >= 1
        assert "aws_access_key_id" in stats["by_kind"]

    async def test_innocent_content_stored_normally(self, memory_store):
        """Non-secret content stores fine and returns a real id."""
        rid = await memory_store.store(
            content="user prefers terse responses",
            memory_type="preference",
        )
        assert rid  # non-empty UUID

    async def test_secret_block_does_not_corrupt_subsequent_stores(self, memory_store):
        """After a refusal, the next store() still works."""
        await memory_store.store(content="sk-proj-1234567890abcdefghij", memory_type="note")
        rid = await memory_store.store(content="totally fine", memory_type="note")
        assert rid


# ═════════════════════════════════════════════════════════════════════
# B2a — STORE() +6 NEW KWARGS
# ═════════════════════════════════════════════════════════════════════


class TestStoreExtendedKwargs:
    """The 6 new kwargs (project_id, branch, source_path, source_blob_sha,
    source_mtime, chunk_ordinal) round-trip into SQLite correctly."""

    async def test_all_six_kwargs_persist(self, memory_store, tmp_path):
        rid = await memory_store.store(
            content="chunk content for round-trip test",
            memory_type="note",
            project_id="proj-x",
            branch="feature/abc",
            source_path="/path/to/foo.py",
            source_blob_sha="deadbeef",
            source_mtime=1700000000,
            chunk_ordinal=3,
        )
        # Read raw row directly from SQLite to verify
        conn = sqlite3.connect(memory_store._db_path)
        row = conn.execute(
            "SELECT project_id, branch, source_path, source_blob_sha, "
            "source_mtime, chunk_ordinal FROM memories WHERE id = ?",
            (rid,),
        ).fetchone()
        conn.close()
        assert row == ("proj-x", "feature/abc", "/path/to/foo.py", "deadbeef",
                       1700000000, 3)

    async def test_defaults_when_kwargs_omitted(self, memory_store):
        """Backward compat: stores without the new kwargs use defaults."""
        rid = await memory_store.store(content="legacy-style call", memory_type="fact")
        conn = sqlite3.connect(memory_store._db_path)
        row = conn.execute(
            "SELECT project_id, branch, source_path, chunk_ordinal "
            "FROM memories WHERE id = ?",
            (rid,),
        ).fetchone()
        conn.close()
        assert row == ("default", None, None, 0)

    async def test_chunk_ordinal_not_null_constraint_holds(self, memory_store):
        """chunk_ordinal INTEGER NOT NULL DEFAULT 0 — never null."""
        rid = await memory_store.store(content="foo", memory_type="fact")
        conn = sqlite3.connect(memory_store._db_path)
        ord_val = conn.execute(
            "SELECT chunk_ordinal FROM memories WHERE id = ?", (rid,)
        ).fetchone()[0]
        conn.close()
        assert ord_val == 0  # never NULL


# ═════════════════════════════════════════════════════════════════════
# B2b — REINDEX_FILE()
# ═════════════════════════════════════════════════════════════════════


class TestReindexFile:
    """File → semantic chunks → fresh code_extracted rows."""

    async def test_basic_python_file_chunks(self, memory_store, tmp_path):
        f = tmp_path / "sample.py"
        f.write_text("def foo(): return 1\n\nclass Bar:\n    pass\n")
        result = await memory_store.reindex_file(str(f), project_id="test")
        assert result["chunk_count"] == 2
        assert result["stale_rows_removed"] == 0
        assert len(result["new_ids"]) == 2

    async def test_idempotent_re_run_purges_stale(self, memory_store, tmp_path):
        """Re-running reindex_file on the same file purges old chunks first."""
        f = tmp_path / "sample.py"
        f.write_text("def foo(): return 1\n\nclass Bar:\n    pass\n")

        await memory_store.reindex_file(str(f))
        result2 = await memory_store.reindex_file(str(f))
        assert result2["stale_rows_removed"] == 2  # the 2 from first run
        assert result2["chunk_count"] == 2

        # DB should still have only 2 rows for this file
        resolved = str(f.resolve())
        conn = sqlite3.connect(memory_store._db_path)
        n = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE source_path = ?", (resolved,)
        ).fetchone()[0]
        conn.close()
        assert n == 2

    async def test_file_modification_reflects_in_chunks(self, memory_store, tmp_path):
        """After file edit, reindex shows the new structure."""
        f = tmp_path / "sample.py"
        f.write_text("def foo(): return 1\n")
        r1 = await memory_store.reindex_file(str(f))
        assert r1["chunk_count"] == 1

        # Add a class
        f.write_text("def foo(): return 1\n\nclass Bar:\n    pass\n")
        r2 = await memory_store.reindex_file(str(f))
        assert r2["chunk_count"] == 2
        assert r2["stale_rows_removed"] == 1

    async def test_missing_file_returns_error(self, memory_store):
        result = await memory_store.reindex_file("/tmp/__nonexistent_file_42__.py")
        assert result["error"] == "file_not_found"

    async def test_directory_returns_not_a_file_error(self, memory_store, tmp_path):
        result = await memory_store.reindex_file(str(tmp_path))
        assert result["error"] == "not_a_file"

    async def test_binary_file_returns_read_skipped(self, memory_store, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00\x01\x02hello\x03binary")
        result = await memory_store.reindex_file(str(f))
        assert result["error"] == "read_skipped"

    async def test_chunks_tagged_code_extracted_with_line_metadata(
        self, memory_store, tmp_path
    ):
        f = tmp_path / "sample.py"
        f.write_text("def foo():\n    return 1\n")
        result = await memory_store.reindex_file(str(f), project_id="proj-x")

        # Pull the row back and verify metadata
        rid = result["new_ids"][0]["id"]
        mem = await memory_store.get(rid)
        assert mem["trust_source"] == "code_extracted"
        assert mem["project_id"] == "proj-x"
        assert "code_extracted" in mem["tags"]


# ═════════════════════════════════════════════════════════════════════
# B3 + B5 — SYNC_GIT_CHANGES() (working-tree + with prior_head)
# ═════════════════════════════════════════════════════════════════════


class TestSyncGitChanges:
    """Git mutation → memory sync. B3 covers working-tree; B5 covers
    prior_head for committed changes."""

    async def test_non_git_path_returns_error(self, memory_store, tmp_path):
        not_a_repo = tmp_path / "no-git-here"
        not_a_repo.mkdir()
        result = await memory_store.sync_git_changes(str(not_a_repo))
        assert result["error"] == "not_a_git_repo"

    async def test_clean_repo_no_changes(self, memory_store, git_repo):
        result = await memory_store.sync_git_changes(str(git_repo.path))
        assert result["modified"] == 0
        assert result["deleted"] == 0
        assert result["committed_diff_used"] is False

    async def test_uncommitted_modification_detected(self, memory_store, git_repo):
        """Working-tree modifications get caught by `git status` walk."""
        (git_repo / "a.py").write_text("def alpha(): return 1\n")
        git_repo._git("add", ".")
        git_repo._git("commit", "-q", "-m", "init")

        # Modify the file (don't commit)
        (git_repo / "a.py").write_text("def alpha(): return 1\n\ndef beta(): return 2\n")

        result = await memory_store.sync_git_changes(str(git_repo.path), project_id="proj-x")
        assert result["modified"] == 1
        assert result["chunks_added"] >= 1

    async def test_uncommitted_deletion_purges(self, memory_store, git_repo):
        """A deleted file with prior code_extracted rows gets purged."""
        (git_repo / "doomed.py").write_text("def doomed(): return 1\n")
        git_repo._git("add", ".")
        git_repo._git("commit", "-q", "-m", "init")

        # Pre-populate memory for this file
        await memory_store.reindex_file(str((git_repo / "doomed.py").resolve()))

        # Delete (uncommitted)
        (git_repo / "doomed.py").unlink()
        git_repo._git("add", "-A")  # stage the deletion

        result = await memory_store.sync_git_changes(str(git_repo.path))
        assert result["deleted"] == 1
        assert result["rows_purged"] >= 1

    async def test_prior_head_catches_committed_changes(self, memory_store, git_repo):
        """B5: with prior_head, sync catches changes that landed in commits
        (which `git status` alone wouldn't see in a clean working tree)."""
        # Initial commit
        (git_repo / "a.py").write_text("def alpha(): return 1\n")
        git_repo._git("add", ".")
        git_repo._git("commit", "-q", "-m", "init")
        prior_head = git_repo._git("rev-parse", "HEAD")

        # Make + commit changes
        (git_repo / "a.py").write_text("def alpha(): return 1\n\nclass Aleph:\n    pass\n")
        (git_repo / "b.py").write_text("def bravo(): return 2\n")
        git_repo._git("add", "-A")
        git_repo._git("commit", "-q", "-m", "feature")

        # Working tree is now CLEAN — without prior_head, sync would miss everything
        r_no_prior = await memory_store.sync_git_changes(str(git_repo.path))
        assert r_no_prior["modified"] == 0
        assert r_no_prior["committed_diff_used"] is False

        # WITH prior_head, sync catches the committed changes
        r_with_prior = await memory_store.sync_git_changes(
            str(git_repo.path), prior_head=prior_head, project_id="proj-x"
        )
        assert r_with_prior["committed_diff_used"] is True
        assert r_with_prior["modified"] == 2  # a.py + b.py both changed


# ═════════════════════════════════════════════════════════════════════
# B4 — WARMUP_EMBEDDER()
# ═════════════════════════════════════════════════════════════════════


class TestWarmupEmbedder:
    """Pre-load the embedding model. Idempotent."""

    async def test_warmup_with_real_embedder(self, memory_store):
        result = await memory_store.warmup_embedder()
        assert result["warmed"] is True
        assert "embedder" in result

    async def test_warmup_already_loaded_is_fast(self, memory_store):
        """Second call should hit `already_loaded` and be quick."""
        await memory_store.warmup_embedder()  # ensure loaded
        t0 = time.perf_counter()
        r = await memory_store.warmup_embedder()
        elapsed = time.perf_counter() - t0
        assert r["warmed"] is True
        assert r.get("already_loaded") is True
        assert elapsed < 0.5  # should be near-instant

    async def test_warmup_no_embedder_returns_no_embedder_reason(
        self, tmp_path
    ):
        """Store with embedding_client=None reports the reason cleanly."""
        from predacore.memory.store import UnifiedMemoryStore
        store = UnifiedMemoryStore(
            db_path=str(tmp_path / "noembed.db"),
            embedding_client=None,
        )
        result = await store.warmup_embedder()
        assert result["warmed"] is False
        assert result["reason"] == "no_embedder"
        store.close()


# ═════════════════════════════════════════════════════════════════════
# B6 + L4 — RECALL_EXPLAIN() (sophisticated per-stage trace)
# ═════════════════════════════════════════════════════════════════════


class TestRecallExplain:
    """Sophisticated per-stage retrieval trace + invariant_skips integration."""

    async def test_basic_trace_structure(self, memory_store):
        await memory_store.store(content="hello world", memory_type="note")
        trace = await memory_store.recall_explain(query="hello", top_k=5)

        # Must have stages dict
        assert "stages" in trace
        for k in ("vector.raw", "vector.kept", "keyword.kept", "filtered_out", "final"):
            assert k in trace["stages"]

        # Must have backward-compat top-level fields
        for k in ("results_count", "results", "verification_state_counts",
                  "filtered_by_invariants", "embedding_version", "schema_version"):
            assert k in trace

    async def test_filtered_out_counts_per_query_delta(
        self, memory_store
    ):
        """filtered_out shows the COUNT of rows rejected during THIS query
        (delta of _invariant_skips before vs after)."""
        # Store 2 healthy memories
        await memory_store.store(content="hello world", memory_type="note")
        await memory_store.store(content="hello universe", memory_type="note")

        # Manually flag one as stale
        rid_stale = await memory_store.store(content="hello stale", memory_type="note")
        memory_store._conn.execute(
            "UPDATE memories SET verification_state = ? WHERE id = ?",
            ("stale", rid_stale),
        )
        memory_store._conn.commit()

        memory_store.reset_invariant_skips()
        trace = await memory_store.recall_explain(query="hello", top_k=10)
        fo = trace["stages"]["filtered_out"]
        assert isinstance(fo, dict)
        assert fo.get("stale_verification", 0) >= 1

    async def test_results_in_final_have_expected_shape(self, memory_store):
        await memory_store.store(content="findable", memory_type="fact")
        trace = await memory_store.recall_explain(query="findable", top_k=5)
        if trace["results"]:
            r = trace["results"][0]
            for key in ("id", "score", "memory_type", "preview"):
                assert key in r

    async def test_verification_state_counts_all_states_present(self, memory_store):
        """The whole-DB breakdown includes all 6 known states."""
        trace = await memory_store.recall_explain(query="anything", top_k=5)
        states = trace["verification_state_counts"]
        for s in ("ok", "unverified", "stale", "orphaned", "version_skew", "superseded"):
            assert s in states


# ═════════════════════════════════════════════════════════════════════
# L4 — _invariant_skips counter + reset_invariant_skips()
# ═════════════════════════════════════════════════════════════════════


class TestInvariantSkips:
    """The counter infrastructure that powers recall_explain's per-stage delta."""

    def test_initial_counter_state(self, memory_store):
        c = memory_store._invariant_skips
        for key in ("stale_verification", "orphaned", "version_skew",
                    "project_mismatch"):
            assert c[key] == 0

    async def test_stale_row_increments_counter_during_recall(self, memory_store):
        """When recall filters out a stale row, the counter goes up."""
        rid = await memory_store.store(content="will be stale", memory_type="note")
        memory_store._conn.execute(
            "UPDATE memories SET verification_state = ? WHERE id = ?", ("stale", rid),
        )
        memory_store._conn.commit()

        memory_store.reset_invariant_skips()
        await memory_store.recall(query="stale", top_k=10)
        assert memory_store._invariant_skips["stale_verification"] >= 1

    def test_reset_returns_snapshot_and_zeros(self, memory_store):
        memory_store._invariant_skips["stale_verification"] = 7
        memory_store._invariant_skips["orphaned"] = 3
        snap = memory_store.reset_invariant_skips()
        # Whitelist the keys we care about (rather than full dict-equality)
        # so adding new counter keys (e.g. verification_failed) doesn't
        # break this test going forward.
        assert snap["stale_verification"] == 7
        assert snap["orphaned"] == 3
        assert snap["version_skew"] == 0
        assert snap["project_mismatch"] == 0
        # All zero after reset
        assert all(v == 0 for v in memory_store._invariant_skips.values())


# ═════════════════════════════════════════════════════════════════════
# L5 — PROJECT_ID FILTER ON RECALL()
# ═════════════════════════════════════════════════════════════════════


class TestProjectIdFilter:
    """Recall's project_id filter + project_mismatch counter."""

    async def test_no_filter_returns_all(self, memory_store):
        await memory_store.store(content="alpha note", memory_type="note", project_id="proj-A")
        await memory_store.store(content="beta note", memory_type="note", project_id="proj-B")
        results = await memory_store.recall(query="note", top_k=10)
        assert len(results) == 2

    async def test_string_project_id_filters(self, memory_store):
        await memory_store.store(content="alpha note", memory_type="note", project_id="proj-A")
        await memory_store.store(content="beta note", memory_type="note", project_id="proj-B")
        results = await memory_store.recall(query="note", top_k=10, project_id="proj-A")
        assert len(results) == 1
        assert results[0][0]["project_id"] == "proj-A"

    async def test_list_project_id_filters_to_any(self, memory_store):
        await memory_store.store(content="alpha note", memory_type="note", project_id="proj-A")
        await memory_store.store(content="beta note", memory_type="note", project_id="proj-B")
        await memory_store.store(content="gamma note", memory_type="note", project_id="proj-C")
        results = await memory_store.recall(
            query="note", top_k=10, project_id=["proj-A", "proj-B"]
        )
        assert len(results) == 2

    async def test_all_sentinel_disables_filter(self, memory_store):
        await memory_store.store(content="alpha note", memory_type="note", project_id="proj-A")
        await memory_store.store(content="beta note", memory_type="note", project_id="proj-B")
        results = await memory_store.recall(query="note", top_k=10, project_id="all")
        assert len(results) == 2

    async def test_project_mismatch_counter_increments(self, memory_store):
        await memory_store.store(content="alpha note", memory_type="note", project_id="proj-A")
        await memory_store.store(content="beta note", memory_type="note", project_id="proj-B")

        memory_store.reset_invariant_skips()
        await memory_store.recall(query="note", top_k=10, project_id="proj-A")
        assert memory_store._invariant_skips["project_mismatch"] >= 1
