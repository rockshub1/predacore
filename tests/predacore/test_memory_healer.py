"""
Tests for ``predacore.memory.healer.Healer`` repair logic.

Before this file the Healer had only thread-lifecycle coverage in
``test_memory_subsystem.py``. The actual repair code paths — orphan
sweep, audit invariants, re-embed on version skew, snapshot, integrity
— were untested. Those paths can DELETE rows from the store, so silent
behavior change is a real production risk.

Each test instantiates a fresh Healer attached to a real ``UnifiedMemoryStore``
in a tmp directory. The Healer thread is NOT started — these tests call
the per-check methods directly (``audit_invariants()``, ``sweep_orphans()``,
``reembed_skewed()``, ``take_snapshot()``) so we can observe their effect
deterministically without waiting on the loop cadence.

The 2-phase orphan sweep (audit-marks-then-sweep-deletes) is the
specific contract verified — fixed in BACKLOG.md B2 to prevent a single
``os.path.exists`` false (transient FUSE / network-mount issue) from
destroying up to 500 code-extracted rows per cycle.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from predacore.memory.healer import (
    AUDIT_SAMPLE_SIZE,
    MAX_AUTO_REPAIRS_PER_HOUR,
    ORPHAN_SWEEP_MAX,
    REEMBED_BATCH_MAX,
    SNAPSHOT_RETENTION_DAYS,
    Healer,
)


@pytest.fixture
def healer(memory_store, tmp_path) -> Healer:
    """Bare Healer attached to a fresh store. Thread NOT started — tests
    invoke per-check methods directly for deterministic observation.
    """
    snapshots_dir = tmp_path / "_test_snapshots"
    return Healer(
        store=memory_store,
        user="test-user",
        snapshots_dir=snapshots_dir,
        enabled=True,
    )


class TestHealerConstants:
    """The cadence / batch / cap constants are load-bearing for safety.
    Pin them so a silent change shows up in review."""

    def test_audit_sample_size_is_reasonable(self) -> None:
        """Per-pass audit window — too small misses drift, too big stalls."""
        assert 100 <= AUDIT_SAMPLE_SIZE <= 10_000

    def test_orphan_sweep_max_is_capped(self) -> None:
        """Hard limit on how many rows a single sweep can delete."""
        assert 100 <= ORPHAN_SWEEP_MAX <= 50_000

    def test_reembed_batch_max_protects_throughput(self) -> None:
        """Re-embed pass shouldn't saturate BGE for hours on a version bump."""
        assert 10 <= REEMBED_BATCH_MAX <= 10_000

    def test_repair_cap_is_high_enough_for_personal_use(self) -> None:
        """Per BACKLOG note: bumped to 100k to avoid false-pauses on
        personal-use BGE upgrades that may flag many rows at once."""
        assert MAX_AUTO_REPAIRS_PER_HOUR >= 1000

    def test_snapshot_retention_is_substantial(self) -> None:
        """Snapshots are the recovery path for `_restore_from_latest_snapshot`."""
        assert SNAPSHOT_RETENTION_DAYS >= 7


class TestLifecycle:
    """Healer thread start/stop must be safe."""

    def test_start_when_enabled_spawns_thread(self, healer) -> None:
        healer.start()
        try:
            assert healer._thread is not None
            assert healer._thread.is_alive()
        finally:
            healer.stop()

    def test_start_when_disabled_is_noop(self, memory_store, tmp_path) -> None:
        disabled = Healer(
            store=memory_store,
            user="test-user",
            snapshots_dir=tmp_path / "snap",
            enabled=False,
        )
        disabled.start()
        assert disabled._thread is None  # never spawned

    def test_stop_is_idempotent(self, healer) -> None:
        healer.start()
        healer.stop()
        # Calling stop again should not raise
        healer.stop()


class TestAuditInvariants:
    """``audit_invariants`` labels rows with verification_state.
    Critical that:
      - It only samples from {unverified, ok} — won't re-process stale/orphaned.
      - It marks orphaned for missing source_path files.
      - It marks version_skew for old embedding_version.
    """

    @pytest.mark.asyncio
    async def test_audit_returns_stats_dict_shape(self, healer, memory_store) -> None:
        await memory_store.store(content="audit me", memory_type="fact")
        result = healer.audit_invariants()
        # Shape contract: "audited" + "updates" keys
        assert "audited" in result
        assert "updates" in result
        assert isinstance(result["audited"], int)
        assert isinstance(result["updates"], int)

    @pytest.mark.asyncio
    async def test_audit_marks_orphaned_for_missing_source(
        self, healer, memory_store, tmp_path
    ) -> None:
        """A row with source_path pointing at a deleted file → orphaned."""
        # Create a real file, store it, then delete the file
        fake_path = tmp_path / "will_be_deleted.py"
        fake_path.write_text("print('hi')")
        await memory_store.store(
            content="content from file",
            memory_type="fact",
            source_path=str(fake_path),
        )
        fake_path.unlink()  # remove the file

        healer.audit_invariants()
        # Verify the row's verification_state was flipped
        rows = memory_store._conn.execute(
            "SELECT verification_state FROM memories WHERE source_path = ?",
            (str(fake_path),),
        ).fetchall()
        assert any(r[0] == "orphaned" for r in rows), (
            f"expected at least one row flagged 'orphaned', got: {[r[0] for r in rows]}"
        )

    @pytest.mark.asyncio
    async def test_audit_skips_already_flagged_rows(self, healer, memory_store) -> None:
        """Rows already in stale/orphaned/version_skew aren't re-audited.
        The audit query has `WHERE verification_state IN ('unverified', 'ok')`."""
        await memory_store.store(content="x", memory_type="fact")
        # Manually mark a row as already-orphaned via SQL
        memory_store._conn.execute(
            "UPDATE memories SET verification_state = 'orphaned'"
        )
        memory_store._conn.commit()
        result = healer.audit_invariants()
        assert result["audited"] == 0, "should not sample already-flagged rows"


class TestSweepOrphansTwoPhase:
    """The 2-phase contract (BACKLOG.md B2): audit marks orphaned, then
    sweep deletes ONLY rows already marked. A transient stat failure
    cannot destroy data on its own."""

    @pytest.mark.asyncio
    async def test_sweep_only_deletes_orphaned_flagged_rows(
        self, healer, memory_store
    ) -> None:
        # Three rows. Two flagged orphaned manually; one stays ok.
        await memory_store.store(content="ok row", memory_type="fact")
        await memory_store.store(content="bad 1", memory_type="fact")
        await memory_store.store(content="bad 2", memory_type="fact")
        memory_store._conn.execute(
            "UPDATE memories SET verification_state = 'orphaned' "
            "WHERE content IN ('bad 1', 'bad 2')"
        )
        memory_store._conn.commit()

        deleted = healer.sweep_orphans()
        assert deleted == 2

        survivors = memory_store._conn.execute(
            "SELECT content FROM memories"
        ).fetchall()
        survivor_contents = {r[0] for r in survivors}
        assert "ok row" in survivor_contents
        assert "bad 1" not in survivor_contents
        assert "bad 2" not in survivor_contents

    @pytest.mark.asyncio
    async def test_sweep_skips_orphaned_row_if_file_came_back(
        self, healer, memory_store, tmp_path
    ) -> None:
        """If a previously-orphaned row's file now exists again (mount
        recovered), the sweep skips it so the next audit pass can flip
        it back to 'ok'."""
        fake_path = tmp_path / "back_from_dead.py"
        fake_path.write_text("alive again")
        await memory_store.store(
            content="x",
            memory_type="fact",
            source_path=str(fake_path),
        )
        memory_store._conn.execute(
            "UPDATE memories SET verification_state = 'orphaned'"
        )
        memory_store._conn.commit()

        # File exists at sweep time
        deleted = healer.sweep_orphans()
        assert deleted == 0, "row with existing source_path must NOT be deleted"

        # Row is still there
        rows = memory_store._conn.execute("SELECT id FROM memories").fetchall()
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_sweep_with_no_orphaned_rows_is_noop(self, healer, memory_store) -> None:
        await memory_store.store(content="ok row", memory_type="fact")
        deleted = healer.sweep_orphans()
        assert deleted == 0


class TestReembedSkewed:
    """``reembed_skewed`` re-embeds rows whose embedding_version is stale."""

    @pytest.mark.asyncio
    async def test_no_skewed_rows_returns_zero(self, healer, memory_store) -> None:
        """Fresh store, all current version → reembed is a no-op."""
        await memory_store.store(content="recent", memory_type="fact")
        n = healer.reembed_skewed()
        assert n == 0

    def test_reembeds_rows_with_old_version(self, healer, memory_store) -> None:
        """Manually stamp a row with an old embedding_version → reembed
        bumps it back to CURRENT and clears version_skew.

        Sync test (no @pytest.mark.asyncio) — ``healer.reembed_skewed`` is
        designed to be invoked from the healer's OWN thread+loop, and
        running it from inside pytest-asyncio's event loop raises
        ``Cannot run the event loop while another loop is running``. We
        seed the row via direct SQL (bypasses the store's async insert
        path), then call the sync repair method, which runs on its own
        private loop and completes cleanly.
        """
        from predacore.memory.store import CURRENT_EMBEDDING_VERSION

        # Direct INSERT — minimal columns needed for the reembed query to find it
        memory_store._conn.execute(
            """INSERT INTO memories (
                   id, content, content_hash, anchor_hash, memory_type,
                   user_id, embedding, embedding_version,
                   verification_state, trust_source, confidence,
                   created_at, updated_at, last_accessed, decay_score
               ) VALUES (
                   'test-reembed-1', 'reembed me', 'h', 'h', 'fact',
                   'default', X'00010203', 'bge-OLD',
                   'version_skew', 'user_stated', 1.0,
                   '2026-05-11T00:00:00', '2026-05-11T00:00:00',
                   '2026-05-11T00:00:00', 1.0
               )"""
        )
        memory_store._conn.commit()

        n = healer.reembed_skewed()
        assert n >= 1

        # Verify the row's version was bumped and state cleared
        row = memory_store._conn.execute(
            "SELECT embedding_version, verification_state FROM memories WHERE id = 'test-reembed-1'"
        ).fetchone()
        assert row[0] == CURRENT_EMBEDDING_VERSION
        assert row[1] == "ok"


class TestSnapshot:
    """``take_snapshot`` writes today's `.db` to snapshots_dir."""

    @pytest.mark.asyncio
    async def test_snapshot_creates_today_db_file(
        self, healer, memory_store, tmp_path
    ) -> None:
        from datetime import datetime, timezone

        await memory_store.store(content="snap me", memory_type="fact")
        result = healer.take_snapshot()
        # Today's snapshot should now exist in the configured dir
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        expected = tmp_path / "_test_snapshots" / f"{today}.db"
        assert expected.exists(), f"snapshot file missing: {expected}"
        # Result is a stats dict (no 'skipped' key on success path)
        assert "skipped" not in result

    @pytest.mark.asyncio
    async def test_snapshot_skipped_when_no_dir_configured(
        self, memory_store
    ) -> None:
        # Construct a Healer with no snapshots_dir
        bare = Healer(
            store=memory_store,
            user="test-user",
            snapshots_dir=None,
            enabled=True,
        )
        # Force-clear the auto-derived dir
        bare._snapshots_dir = None
        result = bare.take_snapshot()
        assert "skipped" in result


class TestRepairRateCap:
    """``_record_repair`` + ``_too_many_repairs`` enforce the per-hour cap
    so a runaway healer can't destroy a corpus in one cycle."""

    def test_no_repairs_means_not_throttled(self, healer) -> None:
        assert healer._too_many_repairs() is False

    def test_record_repair_accumulates(self, healer) -> None:
        healer._record_repair(count=5)
        # Internal state: list of (timestamp, count) tuples
        total = sum(c for _, c in healer._repair_events)
        assert total == 5

    def test_excessive_repairs_trip_the_cap(self, healer) -> None:
        """Push well past the per-hour cap, verify _too_many_repairs flips True."""
        # Use the actual current cap for the configured store size — for an
        # empty store this falls to MAX_AUTO_REPAIRS_PER_HOUR.
        cap = healer._current_repair_cap()
        healer._record_repair(count=cap + 1)
        assert healer._too_many_repairs() is True
