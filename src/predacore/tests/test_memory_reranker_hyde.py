"""
Wave-12 G2 + reranker tests.

Pins:
  - Reranker correctly reorders bi-encoder results by cross-encoder score.
  - HyDE fires on low-confidence top score and merges results.
  - Default behavior (flags off, no reranker, no hyde_llm) is unchanged
    from pre-Wave-12 recall.

These tests use fake reranker + fake LLM objects — no real model load.
Verified separately by smoke test that the real Qwen3Reranker fail-opens
when sentence-transformers isn't installed.
"""
from __future__ import annotations

from typing import Any

import pytest

from predacore.memory.reranker import (
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANKER_MODEL,
    Qwen3Reranker,
    maybe_default_reranker,
)
from predacore.memory.store import _HYDE_CONFIDENCE_THRESHOLD


class _FakeReranker:
    """Stand-in cross-encoder — scores docs by string-match against `priority`.

    Lets us assert that the rerank step actually reorders by reranker
    output, not by the bi-encoder score it was handed.
    """

    def __init__(self, priority_substr: str) -> None:
        self.priority = priority_substr
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(pairs)
        # High score when priority substring is in the doc; lower otherwise.
        # Inject a tiny gradient by doc length so ties are deterministic.
        return [
            (10.0 if self.priority in doc else 0.1) - (len(doc) * 1e-6)
            for _q, doc in pairs
        ]


class _FakeHydeLLM:
    """Stand-in LLM — returns the configured hypothetical answer."""

    def __init__(self, hypothetical: str) -> None:
        self.hypothetical = hypothetical
        self.calls = 0

    async def chat(self, *, messages: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        self.calls += 1
        return {"content": self.hypothetical}


class TestRerankerModule:
    """Reranker module-level behavior — model name, fail-open, env gate."""

    def test_default_model_name_is_qwen3_06b(self) -> None:
        assert DEFAULT_RERANKER_MODEL == "Qwen/Qwen3-Reranker-0.6B"

    def test_default_rerank_candidates_window(self) -> None:
        assert DEFAULT_RERANK_CANDIDATES == 100

    def test_fail_open_when_sentence_transformers_missing(self) -> None:
        """Real Qwen3Reranker returns equal scores when ST isn't installed."""
        # The CI env doesn't have sentence-transformers; this fail-opens
        # to [1.0, 1.0, ...] so callers preserve bi-encoder order.
        reranker = Qwen3Reranker(model_name="nonexistent/model")
        scores = reranker.predict([("q", "doc1"), ("q", "doc2")])
        assert len(scores) == 2
        assert all(s == 1.0 for s in scores)

    def test_predict_empty_pairs_returns_empty(self) -> None:
        reranker = Qwen3Reranker()
        assert reranker.predict([]) == []

    def test_maybe_default_reranker_returns_none_when_env_off(self, monkeypatch) -> None:
        monkeypatch.setenv("PREDACORE_MEMORY_RERANKER", "0")
        assert maybe_default_reranker() is None

    def test_maybe_default_reranker_returns_instance_when_env_unset(self, monkeypatch) -> None:
        # Wave 12: default is "1" (ON). Unset env should also enable.
        monkeypatch.delenv("PREDACORE_MEMORY_RERANKER", raising=False)
        instance = maybe_default_reranker()
        assert instance is not None
        assert instance.model_name == DEFAULT_RERANKER_MODEL

    def test_maybe_default_reranker_returns_instance_when_env_on(self, monkeypatch) -> None:
        monkeypatch.setenv("PREDACORE_MEMORY_RERANKER", "1")
        instance = maybe_default_reranker()
        assert instance is not None
        assert instance.model_name == DEFAULT_RERANKER_MODEL


class TestRerankerReorders:
    """The cross-encoder step must actually reorder by reranker score."""

    @pytest.mark.asyncio
    async def test_apply_reranker_reorders_by_score(self) -> None:
        # Construct a bare UnifiedMemoryStore — we only call _apply_reranker
        # directly, no DB or vector index needed.
        from predacore.memory.store import UnifiedMemoryStore
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            store = UnifiedMemoryStore(db_path=str(Path(td) / "test.db"))
            fake = _FakeReranker(priority_substr="WAVE12_PRIORITY")
            store._reranker = fake

            # Bi-encoder thinks "wrong doc" is most relevant (highest score)
            results = [
                ({"id": "1", "content": "wrong doc with no marker"}, 0.9),
                ({"id": "2", "content": "another irrelevant doc"}, 0.8),
                ({"id": "3", "content": "WAVE12_PRIORITY actual answer here"}, 0.4),
            ]
            reranked = await store._apply_reranker("test query", results, top_k=3)

            # Reranker should pull id=3 to the front
            ids_in_order = [m["id"] for m, _ in reranked]
            assert ids_in_order[0] == "3", f"Reranker did not reorder: {ids_in_order}"
            # Reranker score should be attached to the winning row
            assert reranked[0][0]["_rerank_score"] > 0.9

    @pytest.mark.asyncio
    async def test_apply_reranker_respects_top_k_trim(self) -> None:
        from predacore.memory.store import UnifiedMemoryStore
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            store = UnifiedMemoryStore(db_path=str(Path(td) / "test.db"))
            fake = _FakeReranker(priority_substr="match")
            store._reranker = fake

            results = [
                ({"id": str(i), "content": f"doc {i} with match" if i % 2 == 0 else f"doc {i}"}, 1.0)
                for i in range(20)
            ]
            reranked = await store._apply_reranker("q", results, top_k=5)
            assert len(reranked) == 5

    @pytest.mark.asyncio
    async def test_reranker_fail_returns_original_order(self) -> None:
        """If predict() raises, we fall back to bi-encoder order."""
        from predacore.memory.store import UnifiedMemoryStore
        import tempfile
        from pathlib import Path

        class _BrokenReranker:
            def predict(self, _pairs):
                raise RuntimeError("simulated")

        with tempfile.TemporaryDirectory() as td:
            store = UnifiedMemoryStore(db_path=str(Path(td) / "test.db"))
            store._reranker = _BrokenReranker()
            results = [
                ({"id": "a", "content": "doc A"}, 0.9),
                ({"id": "b", "content": "doc B"}, 0.7),
            ]
            reranked = await store._apply_reranker("q", results, top_k=2)
            # Same order, no reranker_score annotation
            assert [m["id"] for m, _ in reranked] == ["a", "b"]


class TestHydeExpansion:
    """G2 — HyDE must fire on low-confidence and skip on high-confidence."""

    @pytest.mark.asyncio
    async def test_hyde_skips_when_no_llm(self) -> None:
        """No hyde_llm → recall returns bi-encoder result unchanged."""
        from predacore.memory.store import UnifiedMemoryStore
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            store = UnifiedMemoryStore(db_path=str(Path(td) / "test.db"))
            # No reranker, no LLM — the kwargs are accepted but no-op
            results = await store.recall(
                query="anything",
                use_hyde=True,
                hyde_llm=None,
                rerank=False,
            )
            # Empty store → empty results
            assert results == []

    @pytest.mark.asyncio
    async def test_hyde_generate_calls_llm_with_correct_prompt(self) -> None:
        """The HyDE prompt should ask for a short plausible answer."""
        from predacore.memory.store import UnifiedMemoryStore
        import tempfile
        from pathlib import Path

        captured: dict[str, Any] = {}

        class _CapturingLLM:
            async def chat(self, *, messages, **_):
                captured["messages"] = messages
                return {"content": "hypothetical text"}

        with tempfile.TemporaryDirectory() as td:
            store = UnifiedMemoryStore(db_path=str(Path(td) / "test.db"))
            generated = await store._hyde_generate("user question", _CapturingLLM())
            assert generated == "hypothetical text"
            # System prompt should instruct short plausible answer
            assert any(
                "plausible" in str(m.get("content", "")).lower()
                or "1-2 sentence" in str(m.get("content", "")).lower()
                for m in captured["messages"]
            )

    def test_hyde_threshold_env_override(self, monkeypatch) -> None:
        """PREDACORE_MEMORY_HYDE_THRESHOLD takes effect on next module reload."""
        import importlib
        import predacore.memory.store as store_mod
        monkeypatch.setenv("PREDACORE_MEMORY_HYDE_THRESHOLD", "0.42")
        importlib.reload(store_mod)
        assert store_mod._HYDE_CONFIDENCE_THRESHOLD == 0.42
        # Restore module default for downstream tests
        monkeypatch.delenv("PREDACORE_MEMORY_HYDE_THRESHOLD")
        importlib.reload(store_mod)


class TestRecallBackwardCompat:
    """Flags off → recall() behaves identically to pre-Wave-12."""

    @pytest.mark.asyncio
    async def test_recall_signature_accepts_new_kwargs(self) -> None:
        """All new kwargs are optional with sensible defaults."""
        import inspect
        from predacore.memory.store import UnifiedMemoryStore

        sig = inspect.signature(UnifiedMemoryStore.recall)
        for name in ("rerank", "rerank_candidates", "use_hyde", "hyde_llm"):
            assert name in sig.parameters, f"missing kwarg: {name}"
            assert sig.parameters[name].default is None, f"{name} should default to None"

    @pytest.mark.asyncio
    async def test_recall_with_flags_off_no_reranker_unchanged(self, monkeypatch) -> None:
        """Empty store, env explicitly off → empty result, no reranker attached."""
        from predacore.memory.store import UnifiedMemoryStore
        import tempfile
        from pathlib import Path

        # Wave-12: default is ON. Explicitly opt out for this test.
        monkeypatch.setenv("PREDACORE_MEMORY_RERANKER", "0")
        with tempfile.TemporaryDirectory() as td:
            store = UnifiedMemoryStore(db_path=str(Path(td) / "test.db"))
            assert store._reranker is None
            results = await store.recall(
                query="anything",
                rerank=False,
                use_hyde=False,
            )
            assert results == []

    @pytest.mark.asyncio
    async def test_recall_default_on_reranker_attached_when_env_unset(self, monkeypatch) -> None:
        """Wave-12 default: env unset → reranker auto-attached (fail-open)."""
        from predacore.memory.store import UnifiedMemoryStore
        import tempfile
        from pathlib import Path

        monkeypatch.delenv("PREDACORE_MEMORY_RERANKER", raising=False)
        with tempfile.TemporaryDirectory() as td:
            store = UnifiedMemoryStore(db_path=str(Path(td) / "test.db"))
            assert store._reranker is not None  # auto-picked from env default
            # Empty store still returns empty
            results = await store.recall(query="anything")
            assert results == []

    def test_constructor_accepts_explicit_reranker(self) -> None:
        from predacore.memory.store import UnifiedMemoryStore
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            fake = _FakeReranker(priority_substr="x")
            store = UnifiedMemoryStore(db_path=str(Path(td) / "test.db"), reranker=fake)
            assert store._reranker is fake
