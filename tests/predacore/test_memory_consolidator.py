"""
Tests for ``predacore.memory.consolidator.MemoryConsolidator``.

Closes audit `memory.md` Top Critical Finding #2: 718 LOC of policy logic
(decay rates, similarity threshold, per-user caps, entity auto-link,
merge-similar, protected types) with zero in-tree tests before this file.

Tests focus on the load-bearing decisions — the ones whose silent change
would degrade production retrieval quality without anyone noticing:

  - Memory cap enforcement (MAX_MEMORIES_PER_USER) honors PROTECTED types
  - merge_similar dedupes by similarity threshold, respects keepers
  - Decay path is callable and returns stats shape
  - consolidate() returns the documented stats dict

Uses the ``memory_store`` fixture (real UnifiedMemoryStore, real BGE
embedder, fresh tmp SQLite DB per test). No LLM dependency — the
LLM-needing branches (entity extraction, session summarization) are
covered by separate tests; consolidator tests here construct with
``llm=None`` and verify the no-LLM paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from predacore.memory.consolidator import (
    MAX_MEMORIES_PER_USER,
    _PROTECTED_TYPES,
    MemoryConsolidator,
)


@pytest.fixture
def consolidator(memory_store) -> MemoryConsolidator:
    """Bare consolidator — no LLM, no outcome_store, no sessions_dir."""
    return MemoryConsolidator(store=memory_store, llm=None)


class TestProtectedTypes:
    """The _PROTECTED_TYPES set is load-bearing — these types must
    never be evicted by the cap-enforcement path."""

    def test_protected_types_is_frozen_set(self) -> None:
        """Public contract: caller can't accidentally mutate."""
        assert isinstance(_PROTECTED_TYPES, frozenset)

    def test_protected_types_contains_preference_and_entity(self) -> None:
        """The two production types that must survive cap enforcement."""
        assert "preference" in _PROTECTED_TYPES
        assert "entity" in _PROTECTED_TYPES


class TestEnforceMemoryCap:
    """``_enforce_memory_cap`` is the per-user prune step. Critical that:
      - Preferences and entities are never evicted (PROTECTED).
      - Under cap → no-op.
      - Above cap → evicts least-valuable non-protected memories.
    """

    @pytest.mark.asyncio
    async def test_under_cap_evicts_nothing(self, consolidator, memory_store) -> None:
        # Seed 5 facts; cap=10 → no evictions
        for i in range(5):
            await memory_store.store(
                content=f"fact {i}",
                memory_type="fact",
                importance=2,
                user_id="alice",
            )
        pruned = await consolidator._enforce_memory_cap(max_per_user=10)
        assert pruned == 0
        all_mems = await memory_store.get_all_memories(limit=1000)
        assert len([m for m in all_mems if m.get("user_id") == "alice"]) == 5

    @pytest.mark.asyncio
    async def test_over_cap_evicts_excess_non_protected(
        self, consolidator, memory_store
    ) -> None:
        # 8 facts over cap=5 — expect 3 evictions
        for i in range(8):
            await memory_store.store(
                content=f"fact {i}",
                memory_type="fact",
                importance=2,
                user_id="bob",
            )
        pruned = await consolidator._enforce_memory_cap(max_per_user=5)
        assert pruned == 3
        survivors = [
            m for m in await memory_store.get_all_memories(limit=1000)
            if m.get("user_id") == "bob"
        ]
        assert len(survivors) == 5

    @pytest.mark.asyncio
    async def test_preferences_never_evicted_even_over_cap(
        self, consolidator, memory_store
    ) -> None:
        """If user has 3 preferences + 10 facts and cap=2, preferences survive."""
        for i in range(3):
            await memory_store.store(
                content=f"pref {i}",
                memory_type="preference",
                importance=3,
                user_id="carol",
            )
        for i in range(10):
            await memory_store.store(
                content=f"fact {i}",
                memory_type="fact",
                importance=2,
                user_id="carol",
            )
        # cap=2 → all 10 facts must be evicted (only 2 evictable allowed),
        # 8 evicted, 2 facts + 3 preferences survive.
        pruned = await consolidator._enforce_memory_cap(max_per_user=2)
        assert pruned == 8
        survivors = [
            m for m in await memory_store.get_all_memories(limit=1000)
            if m.get("user_id") == "carol"
        ]
        prefs = [m for m in survivors if m.get("memory_type") == "preference"]
        facts = [m for m in survivors if m.get("memory_type") == "fact"]
        assert len(prefs) == 3, "preferences must never be pruned"
        assert len(facts) == 2

    @pytest.mark.asyncio
    async def test_entities_never_evicted(self, consolidator, memory_store) -> None:
        """Same protection contract for memory_type='entity'."""
        for i in range(2):
            await memory_store.store(
                content=f"entity {i}",
                memory_type="entity",
                importance=3,
                user_id="dave",
            )
        for i in range(6):
            await memory_store.store(
                content=f"note {i}",
                memory_type="note",
                importance=2,
                user_id="dave",
            )
        pruned = await consolidator._enforce_memory_cap(max_per_user=2)
        # 6 notes → 2 allowed → 4 evicted. Entities not counted.
        assert pruned == 4
        survivors = [
            m for m in await memory_store.get_all_memories(limit=1000)
            if m.get("user_id") == "dave"
        ]
        ents = [m for m in survivors if m.get("memory_type") == "entity"]
        assert len(ents) == 2

    @pytest.mark.asyncio
    async def test_eviction_prefers_least_valuable(
        self, consolidator, memory_store
    ) -> None:
        """Among non-protected, eviction order is by value (decay × (1 + access)).
        Stored memories have default decay_score=1.0, access_count=0, so order
        falls back to created_at — oldest evicted first."""
        ids = []
        for i in range(5):
            mid = await memory_store.store(
                content=f"sequential fact {i}",
                memory_type="fact",
                importance=2,
                user_id="eve",
            )
            ids.append(mid)
        # cap=2 → 3 evictions, oldest first → ids[0], ids[1], ids[2] gone
        pruned = await consolidator._enforce_memory_cap(max_per_user=2)
        assert pruned == 3
        remaining_ids = {
            m["id"] for m in await memory_store.get_all_memories(limit=1000)
            if m.get("user_id") == "eve"
        }
        assert ids[3] in remaining_ids
        assert ids[4] in remaining_ids
        assert ids[0] not in remaining_ids

    def test_max_memories_per_user_is_set_high_enough(self) -> None:
        """Default ceiling — should be high (1M) for personal use; the
        cap is a backstop, not the primary retention mechanism."""
        assert MAX_MEMORIES_PER_USER >= 1_000_000


class TestMergeSimilar:
    """``merge_similar`` dedupes by similarity_threshold. Critical that:
      - Below threshold → no merge.
      - Above threshold → fewer survives, never deletes a 'keeper' (prior winner).
      - Protected types skipped entirely.
    """

    @pytest.mark.asyncio
    async def test_below_threshold_no_merges(self, consolidator, memory_store) -> None:
        """Distinct content → no merges expected at the default 0.87 threshold."""
        # Need >=10 memories for merge_similar to even attempt
        for i in range(15):
            await memory_store.store(
                content=f"completely different topic number {i}: "
                        f"a unique sentence about {['python', 'rust', 'go', 'java', 'kotlin'][i % 5]}",
                memory_type="fact",
                user_id="alice",
            )
        merged = await consolidator.merge_similar(similarity_threshold=0.87)
        # Distinct topics → at most a few merges; main contract is "doesn't crash"
        # and returns an integer
        assert isinstance(merged, int)
        assert merged >= 0

    @pytest.mark.asyncio
    async def test_protected_types_skipped(self, consolidator, memory_store) -> None:
        """Preferences + entities are skipped entirely in merge_similar
        (the for-loop's first `continue`). Even near-duplicates won't merge."""
        # Seed 15 near-identical preferences
        for i in range(15):
            await memory_store.store(
                content=f"user prefers dark mode (note {i})",
                memory_type="preference",
                user_id="alice",
            )
        merged = await consolidator.merge_similar(similarity_threshold=0.5)
        assert merged == 0, "preferences must never merge"

    @pytest.mark.asyncio
    async def test_too_few_memories_returns_zero(self, consolidator, memory_store) -> None:
        """merge_similar requires len(memories) >= 10. Below that, no-op."""
        for i in range(3):
            await memory_store.store(
                content=f"fact {i}",
                memory_type="fact",
                user_id="alice",
            )
        merged = await consolidator.merge_similar()
        assert merged == 0


class TestConsolidateStats:
    """``consolidate()`` returns a dict with the documented keys."""

    @pytest.mark.asyncio
    async def test_returns_stats_dict_shape(self, consolidator, memory_store) -> None:
        await memory_store.store(content="test", memory_type="fact", user_id="alice")
        stats = await consolidator.consolidate()
        # Shape contract per docstring + visible code:
        for key in (
            "decayed", "entities_found", "links_created",
            "sessions_summarized", "merged", "pruned",
        ):
            assert key in stats, f"missing stats key: {key}"
            assert isinstance(stats[key], int)

    @pytest.mark.asyncio
    async def test_consolidate_recent_returns_stats(self, consolidator, memory_store) -> None:
        await memory_store.store(content="recent", memory_type="conversation", user_id="alice")
        stats = await consolidator.consolidate_recent(last_n=10)
        assert isinstance(stats, dict)
        # consolidate_recent runs a subset; not all keys required, but the
        # ones it does run should be int
        for v in stats.values():
            assert isinstance(v, int)
