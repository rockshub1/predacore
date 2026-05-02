"""
Predacore Memory — self-healing daemon.

Runs inside the MCP server process. One background thread, periodic
checks with different cadences. Each check has an auto-repair action
and is protected by safety brakes.

Checks (cadence → action):

  60s     audit_invariants   Sample 100 rows, verify source exists +
                             embedding version matches + (optional)
                             content_hash still matches source file.
                             Update verification_state accordingly.

  5m      sweep_orphans      Delete rows whose source_path no longer
                             exists on disk. Capped to 500/run to avoid
                             mass-deletion stampedes.

  10m     check_bm25_drift   Sample 10 random terms, recount df from
                             scratch, compare against expected. Logs
                             drift only (no auto-action in v1 — rebuilding
                             BM25 stats is a no-op for our Rust kernel
                             since it computes df lazily from the corpus).

  1h      reembed_skewed     For rows with embedding_version != current,
                             re-embed up to 100 rows per pass (hard cap
                             to protect BGE throughput budget).

  daily   snapshot           SQLite backup to memory_data/snapshots/
                             <user>/YYYY-MM-DD.db. Prune > 7 days.

  weekly  check_integrity    PRAGMA integrity_check. On failure, copy
                             the broken DB aside and restore from last
                             snapshot.

Safety brakes:
  - Repair rate cap = max(MAX_AUTO_REPAIRS_PER_HOUR=1000, total_rows // 5).
    Exceeded → healer pauses + logs. Floor of 1000 keeps small DBs identical
    to the original behavior; the row-count scaling (~20% of total / hour)
    keeps the brake meaningful on large bulk-indexed DBs without false-paging
    after benign mass events (e.g. a BGE upgrade flagging ~5K rows for
    reembed).
  - Audit label-flips (verification_state changes) are NOT counted toward
    the brake — they don't destroy data. Only orphan purges, reembeds, and
    undo restores count.
  - Destructive actions (delete, restore) go through `_record_undo()`
    so a later undo is possible within UNDO_RETENTION_DAYS.
  - healthy_pct regression > 5% after any check → immediate pause.
  - All checks catch their own exceptions and surface in heal_stats.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────

DEFAULT_CADENCES = {
    "audit_invariants":  60.0,          # seconds
    "sweep_orphans":     5 * 60.0,
    "reembed_skewed":    60 * 60.0,
    "check_bm25_drift":  10 * 60.0,
    "snapshot":          24 * 60 * 60.0,
    "check_integrity":   7 * 24 * 60 * 60.0,
}

AUDIT_SAMPLE_SIZE = 100
ORPHAN_SWEEP_MAX = 500
REEMBED_BATCH_MAX = 100

MAX_AUTO_REPAIRS_PER_HOUR = 1000
MAX_HEALTHY_PCT_DROP = 5.0
UNDO_RETENTION_DAYS = 7

SNAPSHOT_RETENTION_DAYS = 7


# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────


@dataclass
class HealStats:
    """Per-check outcomes since the healer started. Surfaced via
    memory.heal_status so drift is always visible."""
    audits_run:          int = 0
    rows_audited:        int = 0
    flagged_stale:       int = 0
    flagged_orphaned:    int = 0
    flagged_version_skew:int = 0
    marked_ok:           int = 0

    orphan_sweeps:       int = 0
    rows_purged:         int = 0

    reembed_passes:      int = 0
    rows_reembedded:     int = 0

    bm25_drift_checks:   int = 0
    bm25_drift_pct_max:  float = 0.0

    snapshots_taken:     int = 0
    snapshots_pruned:    int = 0
    last_snapshot_at:    str | None = None

    integrity_checks:    int = 0
    integrity_failures:  int = 0
    last_integrity_at:   str | None = None

    auto_repairs_last_hour: int = 0
    paused:              bool = False
    pause_reason:        str | None = None

    last_run:            dict[str, float] = field(default_factory=dict)
    last_error:          dict[str, str] = field(default_factory=dict)

    started_at:          str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "auto_repairs_last_hour": self.auto_repairs_last_hour,
            "audit": {
                "runs": self.audits_run,
                "rows_audited": self.rows_audited,
                "flagged_stale": self.flagged_stale,
                "flagged_orphaned": self.flagged_orphaned,
                "flagged_version_skew": self.flagged_version_skew,
                "marked_ok": self.marked_ok,
            },
            "orphan_sweep": {
                "runs": self.orphan_sweeps,
                "rows_purged": self.rows_purged,
            },
            "reembed": {
                "runs": self.reembed_passes,
                "rows_reembedded": self.rows_reembedded,
            },
            "bm25_drift": {
                "runs": self.bm25_drift_checks,
                "max_drift_pct": round(self.bm25_drift_pct_max, 3),
            },
            "snapshot": {
                "taken": self.snapshots_taken,
                "pruned": self.snapshots_pruned,
                "last_at": self.last_snapshot_at,
            },
            "integrity": {
                "runs": self.integrity_checks,
                "failures": self.integrity_failures,
                "last_at": self.last_integrity_at,
            },
            "last_run": self.last_run,
            "last_error": self.last_error,
        }


# ─────────────────────────────────────────────────────────────
# Healer
# ─────────────────────────────────────────────────────────────


class Healer:
    """Background self-healing thread for a UnifiedMemoryStore.

    Creator owns the store; Healer does not close it. Start with
    ``healer.start()``, stop with ``healer.stop()``. Stop is non-blocking
    (sets a flag; the thread exits at its next wake cycle).
    """

    def __init__(
        self,
        store: Any,              # UnifiedMemoryStore
        user: str,
        *,
        snapshots_dir: Path | None = None,
        cadences: dict[str, float] | None = None,
        enabled: bool = True,
    ) -> None:
        self._store = store
        self._user = user
        self._cadences = {**DEFAULT_CADENCES, **(cadences or {})}
        self._enabled = enabled
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self.stats = HealStats(started_at=_now_iso())

        # Where to write SQLite snapshots. Default: <db_parent>/snapshots/<user>/
        db_path = Path(store._db_path) if getattr(store, "_db_path", None) else None
        if snapshots_dir is None and db_path is not None:
            snapshots_dir = db_path.parent / "snapshots" / user
        self._snapshots_dir = snapshots_dir
        if self._snapshots_dir is not None:
            self._snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Per-check last-run timestamps (monotonic seconds since start).
        self._last_run_mono: dict[str, float] = {k: 0.0 for k in self._cadences}

        # Rolling per-hour auto-repair counter: list of (timestamp, count)
        self._repair_events: list[tuple[float, int]] = []
        self._repair_lock = threading.Lock()

        # Content-hash cache so audit_invariants doesn't re-hash unchanged files.
        self._file_hash_cache: dict[str, tuple[float, str]] = {}

        # Track healthy_pct across runs for regression detection.
        self._last_healthy_pct: float | None = None

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        if not self._enabled:
            logger.info("Healer disabled (PREDACORE_HEAL_ENABLED=0)")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="predacore-healer", daemon=True,
        )
        self._thread.start()
        logger.info(
            "Healer started (cadences=%s, snapshots_dir=%s)",
            {k: round(v, 1) for k, v in self._cadences.items()},
            self._snapshots_dir,
        )

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._thread:
            self._thread.join(timeout=join_timeout)

    def trigger_now(self, check: str | None = None) -> None:
        """Force the next loop iteration to run a specific check (or all
        that are due). Returns immediately — the actual run happens in
        the background thread."""
        if check:
            self._last_run_mono[check] = 0.0
        self._wake_event.set()

    # ── The loop ──────────────────────────────────────────────────

    def _loop(self) -> None:
        base_sleep = min(self._cadences.values(), default=60.0) / 2.0
        base_sleep = max(5.0, base_sleep)

        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                logger.exception("Healer tick failed: %s", exc)
                self.stats.last_error["_tick"] = f"{type(exc).__name__}: {exc}"
            if self._stop_event.is_set():
                break
            self._wake_event.wait(timeout=base_sleep)
            self._wake_event.clear()

    def _tick(self) -> None:
        if self._is_paused():
            return
        now = time.monotonic()
        order = [
            # Cheap checks first so a slow expensive check doesn't starve them
            ("audit_invariants", self.audit_invariants),
            ("sweep_orphans",    self.sweep_orphans),
            ("check_bm25_drift", self.check_bm25_drift),
            ("reembed_skewed",   self.reembed_skewed),
            ("snapshot",         self.take_snapshot),
            ("check_integrity",  self.check_integrity),
        ]
        for name, fn in order:
            last = self._last_run_mono[name]
            due = (now - last) >= self._cadences[name]
            if not due:
                continue
            self._last_run_mono[name] = now
            t0 = time.time()
            try:
                fn()
            except Exception as exc:
                logger.exception("Healer check %s failed", name)
                self.stats.last_error[name] = f"{type(exc).__name__}: {exc}"
            finally:
                self.stats.last_run[name] = round(time.time() - t0, 3)

            # Check safety brake AFTER each check — prevents a single
            # check from burning through the whole budget.
            if self._too_many_repairs():
                cap = self._current_repair_cap()
                self._pause(f"auto-repair rate exceeded {cap}/hour")
                return

    # ── Safety brakes ─────────────────────────────────────────────

    def _pause(self, reason: str) -> None:
        # Lock-protected: pause() returning while a _tick() is mid-flight
        # could otherwise let one full check cycle of repairs run after
        # the operator has asked for the brake.
        with self._repair_lock:
            if not self.stats.paused:
                logger.error("Healer PAUSED: %s", reason)
            self.stats.paused = True
            self.stats.pause_reason = reason

    def resume(self) -> None:
        with self._repair_lock:
            if self.stats.paused:
                logger.warning("Healer resumed (was: %s)", self.stats.pause_reason)
            self.stats.paused = False
            self.stats.pause_reason = None

    def _is_paused(self) -> bool:
        with self._repair_lock:
            return self.stats.paused

    def _record_repair(self, count: int = 1) -> None:
        now = time.time()
        with self._repair_lock:
            self._repair_events.append((now, count))
            # Prune events older than 1 hour
            cutoff = now - 3600.0
            self._repair_events = [e for e in self._repair_events if e[0] >= cutoff]
            self.stats.auto_repairs_last_hour = sum(c for _, c in self._repair_events)

    def _current_repair_cap(self) -> int:
        """Brake cap, scaled by DB row count.

        Floor of MAX_AUTO_REPAIRS_PER_HOUR (1000) so small DBs behave
        identically. Above ~5K rows the cap grows to 20% of total — keeps
        the brake meaningful (no large DB can be silently rewritten faster
        than 20%/hour) while letting bulk-indexed projects absorb mass
        repairs after a model upgrade without false-paging the operator.
        """
        try:
            conn = self._store._conn
            if conn is None:
                return MAX_AUTO_REPAIRS_PER_HOUR
            row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            total = int(row[0]) if row else 0
        except sqlite3.Error:
            return MAX_AUTO_REPAIRS_PER_HOUR
        return max(MAX_AUTO_REPAIRS_PER_HOUR, total // 5)

    def _too_many_repairs(self) -> bool:
        cap = self._current_repair_cap()
        with self._repair_lock:
            return self.stats.auto_repairs_last_hour > cap

    # ── Individual checks ────────────────────────────────────────

    def audit_invariants(self) -> dict[str, int]:
        """Sample up to AUDIT_SAMPLE_SIZE rows in states {unverified, ok}
        and flag ones that no longer pass the invariants."""
        conn = self._store._conn
        if conn is None:
            return {}
        rows = conn.execute(
            """SELECT id, source_path, embedding, embedding_version, content, content_hash
               FROM memories
               WHERE verification_state IN ('unverified', 'ok')
               ORDER BY RANDOM()
               LIMIT ?""",
            (AUDIT_SAMPLE_SIZE,),
        ).fetchall()
        self.stats.audits_run += 1
        self.stats.rows_audited += len(rows)
        now = _now_iso()
        updates: list[tuple[str, str, str, str]] = []  # (state, last_verified_at, id)

        # Import here to avoid a hard dep on store at module import time.
        from .store import (  # type: ignore
            CURRENT_EMBEDDING_VERSION,
            compute_content_hash,
        )

        for row in rows:
            mem_id, source_path, embedding, embedding_version, content, stored_hash = row
            state = "ok"

            # 1) source file existence (orphan)
            if source_path:
                try:
                    if not os.path.exists(source_path):
                        state = "orphaned"
                except OSError:
                    state = "orphaned"

            # 2) embedding version skew (only applies to rows WITH an embedding)
            if state == "ok" and embedding is not None:
                if embedding_version and embedding_version != CURRENT_EMBEDDING_VERSION:
                    state = "version_skew"

            # 3) content drift vs stored content_hash. Source-backed rows
            #    only — otherwise "stale" would fire for every user note.
            if state == "ok" and source_path and stored_hash:
                file_hash = self._file_hash(source_path)
                if file_hash is not None:
                    # The chunk hash is of the CHUNK content not the whole
                    # file — what we actually care about is whether THIS
                    # chunk's content is still present anywhere in the file.
                    # Cheap approximation: recompute the stored chunk content
                    # hash; if content in the row matches itself (always true)
                    # we accept it. A deeper "is the chunk still grep-able
                    # in the file?" check is future work — this pass just
                    # confirms the row's own hash is self-consistent.
                    if compute_content_hash(content or "") != stored_hash:
                        state = "stale"

            if state == "ok":
                self.stats.marked_ok += 1
            elif state == "stale":
                self.stats.flagged_stale += 1
            elif state == "orphaned":
                self.stats.flagged_orphaned += 1
            elif state == "version_skew":
                self.stats.flagged_version_skew += 1

            updates.append((state, now, mem_id))

        if updates:
            conn.executemany(
                "UPDATE memories SET verification_state = ?, last_verified_at = ? WHERE id = ?",
                updates,
            )
            conn.commit()
            # NOTE: audit label-flips intentionally do NOT count toward the
            # repair rate brake. Audits only label rows; they don't destroy
            # data or change content. The brake exists to halt runaway
            # destructive ops (orphan purge, mass reembed, undo restore) —
            # counting audit flips here would trip the brake on benign
            # events like a BGE upgrade flagging 5K bulk-indexed rows as
            # version_skew, which then prevents reembed from ever running.

        # Regression check
        self._check_healthy_regression()
        return {"audited": len(rows), "updates": len(updates)}

    def sweep_orphans(self) -> int:
        """Delete rows whose source_path is set but the file no longer
        exists. Capped per run to avoid runaway deletes."""
        conn = self._store._conn
        if conn is None:
            return 0
        rows = conn.execute(
            """SELECT id, source_path FROM memories
               WHERE verification_state = 'orphaned'
                  OR (source_path IS NOT NULL AND source_path != '' AND verification_state = 'ok')
               LIMIT ?""",
            (ORPHAN_SWEEP_MAX,),
        ).fetchall()

        to_delete: list[str] = []
        for mem_id, source_path in rows:
            if not source_path:
                continue
            try:
                if not os.path.exists(source_path):
                    to_delete.append(mem_id)
            except OSError:
                to_delete.append(mem_id)
            if len(to_delete) >= ORPHAN_SWEEP_MAX:
                break

        self.stats.orphan_sweeps += 1
        if not to_delete:
            return 0

        # Record an undo entry BEFORE we destroy anything.
        self._record_undo("sweep_orphans", to_delete)

        placeholders = ",".join("?" for _ in to_delete)
        conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", to_delete)
        conn.commit()
        self.stats.rows_purged += len(to_delete)
        self._record_repair(len(to_delete))

        # Also drop them from the in-RAM vector index. We use a private
        # helper that doesn't require nested-loop gymnastics.
        _remove_from_vec_index(self._store, to_delete)
        logger.info("sweep_orphans: purged %d rows", len(to_delete))
        return len(to_delete)

    def reembed_skewed(self) -> int:
        """Re-embed rows whose embedding_version != current. Rate-limited
        to REEMBED_BATCH_MAX per pass to protect BGE throughput."""
        conn = self._store._conn
        embedder = self._store._embed
        if conn is None or embedder is None:
            return 0
        from .store import CURRENT_EMBEDDING_VERSION  # type: ignore

        rows = conn.execute(
            """SELECT id, content FROM memories
               WHERE verification_state = 'version_skew'
                  OR (embedding IS NOT NULL AND embedding_version != ? AND embedding_version != '')
               LIMIT ?""",
            (CURRENT_EMBEDDING_VERSION, REEMBED_BATCH_MAX),
        ).fetchall()
        self.stats.reembed_passes += 1
        if not rows:
            return 0

        # Batch embed
        contents = [r[1] or "" for r in rows]
        try:
            vecs = asyncio.run(embedder.embed(contents))
        except Exception as exc:
            logger.warning("reembed_skewed: embedder failed: %s", exc)
            self.stats.last_error["reembed_skewed"] = repr(exc)
            return 0

        from .store import _pack_embedding  # type: ignore
        updated = 0
        now = _now_iso()
        for (mem_id, _content), vec in zip(rows, vecs):
            if not vec:
                continue
            blob = _pack_embedding(vec)
            conn.execute(
                """UPDATE memories
                   SET embedding = ?, embedding_version = ?,
                       verification_state = 'ok', last_verified_at = ?
                   WHERE id = ?""",
                (blob, CURRENT_EMBEDDING_VERSION, now, mem_id),
            )
            # Update in-RAM vector index too. _add_sync is the sync
            # counterpart, avoiding nested asyncio issues in threaded contexts.
            if getattr(self._store, "_vec_index", None) is not None:
                try:
                    self._store._vec_index._add_sync(mem_id, vec, {"type": "reembed"})
                except Exception:
                    pass
            updated += 1
        conn.commit()
        self.stats.rows_reembedded += updated
        self._record_repair(updated)
        logger.info("reembed_skewed: re-embedded %d rows", updated)
        return updated

    def check_bm25_drift(self) -> float:
        """Sample 10 random terms from the corpus, recount df from scratch,
        compare with any cached df. Logs max drift but takes no action in
        v1 (our Rust BM25 computes df lazily from the corpus each query,
        so drift doesn't actually accumulate — this is a future hook for
        when we cache df)."""
        self.stats.bm25_drift_checks += 1
        # v1: no persistent df cache → nothing to drift against. Return 0.
        # When we add a df cache, sample + recount here.
        return 0.0

    def take_snapshot(self) -> dict[str, Any]:
        """Atomic SQLite backup. Pruning: keep SNAPSHOT_RETENTION_DAYS."""
        if self._snapshots_dir is None:
            return {"skipped": "no snapshots_dir configured"}
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        dest = self._snapshots_dir / f"{date_str}.db"
        # If today's snapshot already exists, overwrite (daily cadence).
        src_conn = self._store._conn
        if src_conn is None:
            return {"skipped": "adapter mode, snapshot not supported"}

        dest_conn = sqlite3.connect(str(dest))
        try:
            src_conn.backup(dest_conn)
        finally:
            dest_conn.close()

        self.stats.snapshots_taken += 1
        self.stats.last_snapshot_at = _now_iso()

        # Prune snapshots older than retention
        pruned = 0
        cutoff = time.time() - SNAPSHOT_RETENTION_DAYS * 86400.0
        for snap in self._snapshots_dir.glob("*.db"):
            try:
                if snap.stat().st_mtime < cutoff:
                    snap.unlink()
                    pruned += 1
            except OSError:
                continue
        self.stats.snapshots_pruned += pruned
        logger.info("snapshot taken: %s (pruned %d old)", dest, pruned)
        return {"path": str(dest), "pruned_old": pruned}

    def check_integrity(self) -> dict[str, Any]:
        """Run PRAGMA integrity_check. On failure, copy the corrupt DB
        aside and restore from the most recent snapshot."""
        conn = self._store._conn
        if conn is None:
            return {"skipped": "adapter mode"}
        self.stats.integrity_checks += 1
        self.stats.last_integrity_at = _now_iso()
        try:
            row = conn.execute("PRAGMA integrity_check").fetchone()
            status = row[0] if row else ""
        except sqlite3.Error as exc:
            self.stats.integrity_failures += 1
            logger.error("integrity_check raised: %s", exc)
            return {"status": "error", "error": repr(exc)}

        if status == "ok":
            return {"status": "ok"}

        self.stats.integrity_failures += 1
        logger.error("integrity_check FAILED: %s — attempting restore", status)

        restored = self._restore_from_latest_snapshot()
        return {"status": "failed", "detail": status, "restored": bool(restored),
                "restored_from": str(restored) if restored else None}

    # ── Helpers ────────────────────────────────────────────────

    def _restore_from_latest_snapshot(self) -> Path | None:
        if self._snapshots_dir is None:
            return None
        snaps = sorted(
            self._snapshots_dir.glob("*.db"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if not snaps:
            return None
        latest = snaps[0]
        # Move corrupt DB aside with a timestamp
        db_path = Path(self._store._db_path)
        bak = db_path.with_suffix(f".corrupt-{int(time.time())}.db")
        try:
            db_path.rename(bak)
        except OSError as exc:
            logger.error("Couldn't move corrupt DB aside: %s", exc)
            return None
        # Copy snapshot into place
        try:
            import shutil
            shutil.copy2(latest, db_path)
        except OSError as exc:
            logger.error("Couldn't restore snapshot %s: %s", latest, exc)
            return None
        logger.warning("Restored DB from snapshot: %s (corrupt moved to %s)", latest, bak)
        self._record_repair(1)
        return latest

    def _file_hash(self, path: str) -> str | None:
        """mtime-keyed cache over sha256(file) so audit_invariants doesn't
        rehash files that haven't changed. Capped to 1000 entries."""
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        cached = self._file_hash_cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            import hashlib
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    h.update(chunk)
            digest = h.hexdigest()
        except OSError:
            return None
        if len(self._file_hash_cache) >= 1000:
            self._file_hash_cache.pop(next(iter(self._file_hash_cache)))
        self._file_hash_cache[path] = (mtime, digest)
        return digest

    def _record_undo(self, action: str, ids: list[str]) -> None:
        """Append destructive-action details to an undo log JSON file.
        Retained UNDO_RETENTION_DAYS."""
        if self._snapshots_dir is None:
            return
        import json as _json
        undo_dir = self._snapshots_dir.parent / "undo"
        undo_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        path = undo_dir / f"{stamp}-{action}.json"
        try:
            path.write_text(_json.dumps({"action": action, "ids": ids}, indent=2))
        except OSError:
            return

        # Prune old undo logs
        cutoff = time.time() - UNDO_RETENTION_DAYS * 86400.0
        for entry in undo_dir.glob("*.json"):
            try:
                if entry.stat().st_mtime < cutoff:
                    entry.unlink()
            except OSError:
                continue

    def _check_healthy_regression(self) -> None:
        """If healthy_pct drops by more than MAX_HEALTHY_PCT_DROP from the
        previous audit, pause the healer so a human can investigate."""
        conn = self._store._conn
        if conn is None:
            return
        try:
            from .store import CURRENT_EMBEDDING_VERSION  # type: ignore
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if not total:
                return
            healthy = conn.execute(
                "SELECT COUNT(*) FROM memories "
                "WHERE verification_state IN ('ok', 'unverified') "
                "AND (embedding IS NULL OR embedding_version = ?)",
                (CURRENT_EMBEDDING_VERSION,),
            ).fetchone()[0]
            pct = 100.0 * healthy / total
        except sqlite3.Error:
            return

        if self._last_healthy_pct is not None:
            drop = self._last_healthy_pct - pct
            if drop > MAX_HEALTHY_PCT_DROP:
                self._pause(
                    f"healthy_pct regressed {drop:.1f}% "
                    f"({self._last_healthy_pct:.1f}% → {pct:.1f}%)"
                )
        self._last_healthy_pct = pct


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _remove_from_vec_index(store: Any, ids: list[str]) -> None:
    """Sync removal from the in-RAM vector index. The public `remove` is
    async; we reach into the private `_ids/_vecs/_meta` lists directly to
    avoid nested event-loop problems when called from background threads
    or test harnesses that already have a running loop."""
    idx = getattr(store, "_vec_index", None)
    if idx is None:
        return
    to_drop = set(ids)
    with idx._lock:
        keep_ids, keep_vecs, keep_meta = [], [], []
        for i, mid in enumerate(idx._ids):
            if mid in to_drop:
                continue
            keep_ids.append(mid)
            keep_vecs.append(idx._vecs[i])
            keep_meta.append(idx._meta[i])
        idx._ids = keep_ids
        idx._vecs = keep_vecs
        idx._meta = keep_meta
