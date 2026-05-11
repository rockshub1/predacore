"""
Tests for predacore.identity.beliefs — BeliefStore state machine,
bloat caps (decay + prune + MD render filtering), and EVOLUTION.md
append/compaction (in predacore.identity.engine).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from predacore.identity.beliefs import (
    Belief,
    BeliefState,
    BeliefStore,
    _DEMOTED_PRUNE_DAYS,
    _MD_RENDER_OBSERVATION_CAP,
    _STALE_OBSERVATION_DAYS,
)
from predacore.identity.engine import (
    _MAX_EVOLUTION_ENTRIES,
    _compact_evolution_log,
    _log_evolution_to_file,
)


def _seed_belief(
    store: BeliefStore,
    *,
    id_: str,
    state: str,
    days_old: int,
    evidence: int = 1,
    text: str = "x",
) -> None:
    ts = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    store._beliefs[id_] = Belief(
        id=id_,
        text=text,
        state=state,
        created_at=ts,
        last_updated=ts,
        evidence_count=evidence,
    )


class TestBeliefStateMachine:
    def test_add_observation_creates_observation_state(self, tmp_path: Path) -> None:
        store = BeliefStore(tmp_path)
        bid = store.add_observation("the sky is sometimes blue")
        belief = store.get(bid)
        assert belief is not None
        assert belief.state == BeliefState.OBSERVATION.value
        assert belief.evidence_count == 1

    def test_evidence_auto_promotes_through_ladder(self, tmp_path: Path) -> None:
        store = BeliefStore(tmp_path)
        bid = store.add_observation("repeated pattern X")
        # 1 → 2 evidence: promotes to working_theory
        store.record_evidence(bid)
        assert store.get(bid).state == BeliefState.WORKING_THEORY.value
        # 2 → 5: tested
        for _ in range(3):
            store.record_evidence(bid)
        assert store.get(bid).state == BeliefState.TESTED.value
        # 5 → 8: committed
        for _ in range(3):
            store.record_evidence(bid)
        assert store.get(bid).state == BeliefState.COMMITTED.value

    def test_demote_records_reason(self, tmp_path: Path) -> None:
        store = BeliefStore(tmp_path)
        bid = store.add_observation("turned out to be wrong")
        store.demote(bid, reason="observed counterexample in session abc")
        belief = store.get(bid)
        assert belief.state == BeliefState.DEMOTED.value
        assert any("counterexample" in n for n in belief.notes)


class TestBloatCaps:
    """Wave 12 bloat caps: decay stale obs, prune old demoted, MD filtering."""

    def test_decay_demotes_stale_single_evidence_observations(self, tmp_path: Path) -> None:
        store = BeliefStore(tmp_path)
        # Stale (35d) single-evidence observation → should decay
        _seed_belief(store, id_="B-stale", state="observation",
                     days_old=_STALE_OBSERVATION_DAYS + 5, evidence=1)
        # Stale but multi-evidence → should NOT decay (caller can record_evidence later)
        _seed_belief(store, id_="B-stale-multi", state="observation",
                     days_old=_STALE_OBSERVATION_DAYS + 5, evidence=2)
        # Fresh single-evidence → should NOT decay
        _seed_belief(store, id_="B-fresh", state="observation",
                     days_old=1, evidence=1)

        result = store.compact()
        assert result["decayed"] == 1
        assert store.get("B-stale").state == BeliefState.DEMOTED.value
        assert store.get("B-stale-multi").state == BeliefState.OBSERVATION.value
        assert store.get("B-fresh").state == BeliefState.OBSERVATION.value

    def test_prune_removes_long_demoted_beliefs(self, tmp_path: Path) -> None:
        store = BeliefStore(tmp_path)
        # Old demoted (100d) → prune
        _seed_belief(store, id_="B-old-demo", state="demoted",
                     days_old=_DEMOTED_PRUNE_DAYS + 10, evidence=5)
        # Recently demoted (10d) → keep
        _seed_belief(store, id_="B-fresh-demo", state="demoted",
                     days_old=10, evidence=5)

        result = store.compact()
        assert result["pruned"] == 1
        assert store.get("B-old-demo") is None
        assert store.get("B-fresh-demo") is not None

    def test_md_render_drops_demoted_section(self, tmp_path: Path) -> None:
        store = BeliefStore(tmp_path)
        _seed_belief(store, id_="B-demo", state="demoted", days_old=1, evidence=3)
        _seed_belief(store, id_="B-active", state="working_theory", days_old=1, evidence=3)
        md = store.render_markdown()
        assert "## Demoted" not in md
        assert "## Working Theories" in md

    def test_md_render_caps_observations(self, tmp_path: Path) -> None:
        store = BeliefStore(tmp_path)
        # Seed cap+5 fresh observations so cap kicks in but decay does not
        for i in range(_MD_RENDER_OBSERVATION_CAP + 5):
            _seed_belief(store, id_=f"B-obs-{i:02d}", state="observation",
                         days_old=0, evidence=1, text=f"obs {i}")
        md = store.render_markdown()
        obs_lines = [line for line in md.splitlines() if line.startswith("### B-obs-")]
        assert len(obs_lines) == _MD_RENDER_OBSERVATION_CAP

    def test_compact_is_idempotent(self, tmp_path: Path) -> None:
        store = BeliefStore(tmp_path)
        _seed_belief(store, id_="B-stale", state="observation",
                     days_old=_STALE_OBSERVATION_DAYS + 5, evidence=1)
        first = store.compact()
        second = store.compact()
        assert first == {"decayed": 1, "pruned": 0}
        assert second == {"decayed": 0, "pruned": 0}

    def test_compact_runs_at_load(self, tmp_path: Path) -> None:
        """A fresh BeliefStore should run compact() in __init__."""
        store = BeliefStore(tmp_path)
        _seed_belief(store, id_="B-stale", state="observation",
                     days_old=_STALE_OBSERVATION_DAYS + 5, evidence=1)
        store._save()  # persist to JSON
        # Now reload — compact should run at __init__, decaying the stale obs
        store2 = BeliefStore(tmp_path)
        assert store2.get("B-stale").state == BeliefState.DEMOTED.value


class TestEvolutionLog:
    """EVOLUTION.md must append in O(1) and compact only at size threshold."""

    def test_append_is_o1_no_rewrite(self, tmp_path: Path) -> None:
        # First call creates the file with header; subsequent calls append.
        _log_evolution_to_file(tmp_path, "BELIEFS.md", "a", "b", reason="r1")
        evo = tmp_path / "EVOLUTION.md"
        size1 = evo.stat().st_size
        _log_evolution_to_file(tmp_path, "BELIEFS.md", "b", "c", reason="r2")
        size2 = evo.stat().st_size
        # Second call must extend the file, not rewrite from scratch
        assert size2 > size1
        content = evo.read_text()
        # Header appears exactly once; both entries present
        assert content.count("# Evolution Log") == 1
        assert content.count("\n## ") == 2

    def test_compact_trims_to_cap(self, tmp_path: Path) -> None:
        evo = tmp_path / "EVOLUTION.md"
        # Stuff cap+50 fake entries directly
        header = "# Evolution Log\n\n_Legible record._\n"
        entries = "".join(
            f"\n\n## 2026-01-{(i % 28) + 1:02d} 00:00 UTC — BELIEFS.md\n\n"
            f"```diff\n+entry {i}\n```\n"
            for i in range(_MAX_EVOLUTION_ENTRIES + 50)
        )
        evo.write_text(header + entries)
        _compact_evolution_log(evo, header)
        content = evo.read_text()
        assert content.count("\n## ") == _MAX_EVOLUTION_ENTRIES
        # Most recent entries are kept (last one survived)
        assert f"+entry {_MAX_EVOLUTION_ENTRIES + 49}" in content
        # Oldest entries pruned
        assert "+entry 0\n" not in content

    def test_compact_noop_when_under_cap(self, tmp_path: Path) -> None:
        evo = tmp_path / "EVOLUTION.md"
        header = "# Evolution Log\n\n_Legible record._\n"
        content_before = header + "\n\n## 2026-01-01 00:00 UTC — test\n\n```diff\n+x\n```\n"
        evo.write_text(content_before)
        _compact_evolution_log(evo, header)
        # Below cap → file unchanged
        assert evo.read_text() == content_before
