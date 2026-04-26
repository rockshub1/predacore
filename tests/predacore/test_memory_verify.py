"""T5+ — Tests for the verify-with-code layer.

Covers:
  • _verify_chunk_against_source helper (3 outcomes: True/False/None)
  • recall(verify=True) annotates each result with _verified
  • recall(verify=True, verify_drop=True) excludes drifted code chunks
  • Synthesis memories (no source_path) are unverifiable but kept
  • verification_failed counter increments on drift
  • The path to TRUE 100% accuracy on code-backed memories
"""
from __future__ import annotations

from pathlib import Path

import pytest

from predacore.memory.store import _verify_chunk_against_source


# ═════════════════════════════════════════════════════════════════════
# Helper: _verify_chunk_against_source — the 3-state verdict
# ═════════════════════════════════════════════════════════════════════


class TestVerifyHelper:
    """Module-level helper. 3 outcomes: True / False / None."""

    def test_no_source_path_returns_none(self):
        """Synthesis memory (no source_path) → None (unverifiable)."""
        mem = {"content": "user prefers terse responses", "source_path": None}
        assert _verify_chunk_against_source(mem) is None

    def test_missing_source_path_field_returns_none(self):
        mem = {"content": "some content"}  # no source_path key at all
        assert _verify_chunk_against_source(mem) is None

    def test_empty_content_returns_false(self, tmp_path):
        f = tmp_path / "exists.py"
        f.write_text("def alpha(): pass\n")
        mem = {"content": "", "source_path": str(f)}
        assert _verify_chunk_against_source(mem) is False

    def test_file_missing_returns_false(self, tmp_path):
        mem = {"content": "anything",
               "source_path": str(tmp_path / "no_such_file.py")}
        assert _verify_chunk_against_source(mem) is False

    def test_content_present_returns_true(self, tmp_path):
        f = tmp_path / "src.py"
        f.write_text("def alpha(): return 1\n\nclass Bar:\n    pass\n")
        mem = {"content": "def alpha(): return 1\n", "source_path": str(f)}
        assert _verify_chunk_against_source(mem) is True

    def test_anchor_match_when_body_drifted(self, tmp_path):
        """Tier-2 verification: first non-blank line of chunk still in file
        even if the body has changed."""
        f = tmp_path / "src.py"
        f.write_text("def alpha():\n    return 999  # body changed\n")
        # Original chunk had def alpha + return 1
        mem = {"content": "def alpha():\n    return 1\n", "source_path": str(f)}
        # Anchor "def alpha():" is still in the file → verified=True
        assert _verify_chunk_against_source(mem) is True

    def test_content_completely_gone_returns_false(self, tmp_path):
        f = tmp_path / "src.py"
        f.write_text("# completely different file now\n")
        mem = {"content": "def alpha(): return 1\n", "source_path": str(f)}
        assert _verify_chunk_against_source(mem) is False


# ═════════════════════════════════════════════════════════════════════
# recall(verify=True) — annotation
# ═════════════════════════════════════════════════════════════════════


class TestRecallWithVerify:
    """recall(verify=True) annotates results with _verified."""

    async def test_synthesis_memory_marked_none(self, memory_store):
        await memory_store.store(
            content="user likes concise answers", memory_type="preference",
        )
        results = await memory_store.recall(query="concise", top_k=5, verify=True)
        assert len(results) == 1
        mem, _ = results[0]
        assert mem["_verified"] is None

    async def test_code_chunk_verified_true(self, memory_store, tmp_path):
        """A chunk pointing to an existing file with matching content → True."""
        f = tmp_path / "fresh.py"
        f.write_text("def alpha(): return 1\n")
        await memory_store.reindex_file(str(f))

        results = await memory_store.recall(query="alpha", top_k=5, verify=True)
        verified = [m for m, _ in results if m.get("_verified") is True]
        assert len(verified) >= 1

    async def test_code_chunk_drifts_to_false(self, memory_store, tmp_path):
        """If the source file is deleted after indexing, verify returns False."""
        f = tmp_path / "doomed.py"
        f.write_text("def alpha(): return 1\n")
        await memory_store.reindex_file(str(f))
        f.unlink()  # delete the file

        results = await memory_store.recall(query="alpha", top_k=5, verify=True)
        # The chunk is still in memory (we deleted the file, not the row)
        # — but verification should mark it False
        for mem, _ in results:
            if mem.get("source_path") and "doomed" in mem["source_path"]:
                assert mem["_verified"] is False

    async def test_verify_failed_counter_increments(self, memory_store, tmp_path):
        f = tmp_path / "vanish.py"
        f.write_text("def beta(): pass\n")
        await memory_store.reindex_file(str(f))
        f.unlink()

        memory_store.reset_invariant_skips()
        await memory_store.recall(query="beta", top_k=5, verify=True)
        assert memory_store._invariant_skips["verification_failed"] >= 1


# ═════════════════════════════════════════════════════════════════════
# recall(verify=True, verify_drop=True) — true 100% on code
# ═════════════════════════════════════════════════════════════════════


class TestVerifyDrop:
    """verify_drop=True excludes drifted code chunks while keeping
    synthesis (None-verified) rows."""

    async def test_drifted_code_dropped(self, memory_store, tmp_path):
        """Chunk pointing to a deleted file is excluded when verify_drop=True."""
        # Index then delete one file (will drift)
        bad = tmp_path / "deleted.py"
        bad.write_text("def gamma(): return 3\n")
        await memory_store.reindex_file(str(bad))
        bad.unlink()

        # Index another that stays (verified)
        good = tmp_path / "kept.py"
        good.write_text("def gamma(): return 3\n")
        await memory_store.reindex_file(str(good))

        # Default recall returns BOTH chunks
        all_results = await memory_store.recall(query="gamma", top_k=10)
        assert len(all_results) == 2

        # verify=True + verify_drop=True returns ONLY the kept one
        verified = await memory_store.recall(
            query="gamma", top_k=10, verify=True, verify_drop=True,
        )
        for mem, _ in verified:
            assert mem.get("_verified") is True
            assert "kept" in mem["source_path"]
        assert len(verified) == 1

    async def test_synthesis_kept_under_verify_drop(self, memory_store):
        """Memories without source_path (synthesis) are kept under verify_drop
        because we can't disprove them."""
        await memory_store.store(
            content="user always wants TypeScript strict mode",
            memory_type="preference",
        )
        results = await memory_store.recall(
            query="typescript strict", top_k=5, verify=True, verify_drop=True,
        )
        assert len(results) == 1
        mem, _ = results[0]
        assert mem["_verified"] is None  # unverifiable
        # but kept anyway because verify_drop only drops False, not None

    async def test_full_100_percent_path_for_code_queries(
        self, memory_store, tmp_path,
    ):
        """The complete T5+ value prop:
          1. Index 3 files — all verified
          2. Modify 1 file (anchor still matches) — drifted but tier-2 verifies
          3. Delete 1 file — drifted, verification fails
          4. recall(verify=True, verify_drop=True) returns the 2 still-grounded
        """
        f1 = tmp_path / "stable.py"
        f1.write_text("def stable_func(): return 1\n")
        f2 = tmp_path / "modified.py"
        f2.write_text("def modified_func():\n    return 1\n")
        f3 = tmp_path / "deleted.py"
        f3.write_text("def deleted_func(): return 3\n")

        for f in (f1, f2, f3):
            await memory_store.reindex_file(str(f))

        # Mutate f2's body but keep "def modified_func" line (anchor)
        f2.write_text("def modified_func():\n    return 999  # changed\n")
        # Delete f3
        f3.unlink()

        verified = await memory_store.recall(
            query="func", top_k=10, verify=True, verify_drop=True,
        )

        # f3 (deleted) should be dropped; f1 + f2 (anchor still matches) kept
        sources = {m.get("source_path") for m, _ in verified}
        assert any("stable" in s for s in sources if s)
        assert any("modified" in s for s in sources if s)
        assert not any("deleted" in s for s in sources if s)


# ═════════════════════════════════════════════════════════════════════
# Backward compatibility — verify defaults to False (no behavior change)
# ═════════════════════════════════════════════════════════════════════


class TestVerifyBackwardCompat:
    async def test_default_no_verify_no_annotation(self, memory_store, tmp_path):
        """recall() without verify=True doesn't add the _verified field."""
        f = tmp_path / "x.py"
        f.write_text("def x(): pass\n")
        await memory_store.reindex_file(str(f))

        results = await memory_store.recall(query="x", top_k=5)
        for mem, _ in results:
            assert "_verified" not in mem

    async def test_default_doesnt_increment_verification_counter(
        self, memory_store, tmp_path,
    ):
        f = tmp_path / "y.py"
        f.write_text("def y(): pass\n")
        await memory_store.reindex_file(str(f))
        f.unlink()  # would have been verified=False if verify=True was set

        memory_store.reset_invariant_skips()
        await memory_store.recall(query="y", top_k=5)  # no verify
        assert memory_store._invariant_skips["verification_failed"] == 0
