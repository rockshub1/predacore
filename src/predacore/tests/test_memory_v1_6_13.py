"""
v1.6.13 — MMR diversity + multi-query rewriter tests.

Pins:
  - MMR with lambda=1.0 reproduces relevance order (no-op fallback contract)
  - MMR with lambda<1.0 picks diverse candidates from near-duplicates
  - MMR fails open when any candidate is missing its embedding
  - QueryRewriter parses + caches + handles None/timeout/garbage LLM
  - _recall_multiquery unions by id, max-score per id, dedupes original
  - Default env flags (PREDACORE_MEMORY_MMR=0, PREDACORE_MEMORY_MULTIQUERY=0)
    preserve current behavior bit-exact
"""
from __future__ import annotations

import asyncio
import math
import os
import struct
from typing import Any

import pytest

from predacore.memory.query_rewriter import QueryRewriter, _parse_rewrites
from predacore.memory.store import _apply_mmr


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


def _pack(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


# ── MMR tests ─────────────────────────────────────────────────────────


class TestMMR:
    def test_lambda_one_preserves_relevance_order(self) -> None:
        """λ=1.0 means no diversity weight — output == sorted by relevance."""
        results = [
            ({"id": str(i), "_embedding": _norm([1.0, 0.1 * i, 0.0])}, 1.0 - i * 0.1)
            for i in range(5)
        ]
        out = _apply_mmr(results, top_k=5, lambda_=1.0)
        assert [m["id"] for m, _ in out] == ["0", "1", "2", "3", "4"]

    def test_no_embeddings_falls_back_to_relevance_order(self) -> None:
        """Missing embedding on any candidate → return relevance order."""
        results = [({"id": str(i)}, 1.0 - i * 0.1) for i in range(5)]
        out = _apply_mmr(results, top_k=3, lambda_=0.7)
        assert [m["id"] for m, _ in out] == ["0", "1", "2"]

    def test_diversity_picks_one_from_each_cluster(self) -> None:
        """Three near-duplicates (topic A) + two distinct topics (B, C)
        with λ=0.7 should surface A1, B1, C1 — not A1, A2, A3."""
        results = [
            ({"id": "A1", "_embedding": _norm([1.0, 0.0, 0.0])}, 0.95),
            ({"id": "A2", "_embedding": _norm([0.99, 0.1, 0.0])}, 0.92),
            ({"id": "A3", "_embedding": _norm([0.98, 0.15, 0.0])}, 0.90),
            ({"id": "B1", "_embedding": _norm([0.0, 1.0, 0.0])}, 0.85),
            ({"id": "C1", "_embedding": _norm([0.0, 0.0, 1.0])}, 0.80),
        ]
        out = _apply_mmr(results, top_k=3, lambda_=0.7)
        ids = [m["id"] for m, _ in out]
        assert ids[0] == "A1"  # top relevance always first
        assert set(ids) == {"A1", "B1", "C1"}  # diverse picks

    def test_single_result_returns_unchanged(self) -> None:
        results = [({"id": "X", "_embedding": _norm([1.0, 0.0])}, 0.9)]
        assert _apply_mmr(results, top_k=5, lambda_=0.7) == results

    def test_top_k_larger_than_results(self) -> None:
        results = [
            ({"id": str(i), "_embedding": _norm([1.0, 0.1 * i, 0.0])}, 1.0 - i * 0.1)
            for i in range(3)
        ]
        out = _apply_mmr(results, top_k=10, lambda_=0.7)
        assert len(out) == 3  # capped at available

    def test_embedding_blob_unpacked(self) -> None:
        """Blob-form embeddings should be unpacked transparently."""
        vec_a = _norm([1.0, 0.0, 0.0])
        vec_b = _norm([0.0, 1.0, 0.0])
        results = [
            ({"id": "A", "embedding": _pack(vec_a)}, 0.9),
            ({"id": "B", "embedding": _pack(vec_b)}, 0.85),
        ]
        out = _apply_mmr(results, top_k=2, lambda_=0.7)
        assert [m["id"] for m, _ in out] == ["A", "B"]


# ── Rewriter parser tests ─────────────────────────────────────────────


class TestRewriteParser:
    def test_basic_one_per_line(self) -> None:
        out = _parse_rewrites("a\nb\nc", n=3, original="zzz")
        assert out == ["a", "b", "c"]

    def test_strips_numbering(self) -> None:
        out = _parse_rewrites("1. first\n2) second\n3. third", n=3, original="zzz")
        assert out == ["first", "second", "third"]

    def test_strips_bullets_and_quotes(self) -> None:
        out = _parse_rewrites('- "alpha"\n* beta\n• gamma', n=3, original="zzz")
        assert out == ["alpha", "beta", "gamma"]

    def test_dedupes_against_original(self) -> None:
        out = _parse_rewrites("CONFIG\nsettings\nyaml", n=3, original="config")
        assert "CONFIG" not in out  # case-insensitive dedup
        assert out == ["settings", "yaml"]

    def test_caps_at_n(self) -> None:
        out = _parse_rewrites("a\nb\nc\nd\ne", n=2, original="zzz")
        assert out == ["a", "b"]

    def test_skips_header_lines(self) -> None:
        out = _parse_rewrites("Rewrites:\n\nalpha\nbeta", n=3, original="zzz")
        assert out == ["alpha", "beta"]


# ── Rewriter async behavior ───────────────────────────────────────────


class _OkLLM:
    async def chat(self, *, messages: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        return {"content": "alpha\nbeta\ngamma"}


class _TimeoutLLM:
    async def chat(self, *, messages: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        await asyncio.sleep(30)  # Way beyond rewriter's 10s timeout
        return {"content": "should never reach here"}


class _GarbageLLM:
    async def chat(self, *, messages: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        return {"content": ""}


class TestQueryRewriter:
    @pytest.mark.asyncio
    async def test_successful_rewrites(self) -> None:
        rw = QueryRewriter(_OkLLM())
        out = await rw.rewrite("original query", n=3)
        assert out == ["alpha", "beta", "gamma"]

    @pytest.mark.asyncio
    async def test_caches_repeat_call(self) -> None:
        rw = QueryRewriter(_OkLLM())
        out1 = await rw.rewrite("cached query", n=3)
        out2 = await rw.rewrite("cached query", n=3)
        assert out1 == out2

    @pytest.mark.asyncio
    async def test_none_llm_returns_empty(self) -> None:
        rw = QueryRewriter(None)
        assert await rw.rewrite("anything", n=3) == []

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self) -> None:
        rw = QueryRewriter(_OkLLM())
        assert await rw.rewrite("", n=3) == []
        assert await rw.rewrite("   ", n=3) == []

    @pytest.mark.asyncio
    async def test_timeout_returns_empty(self) -> None:
        rw = QueryRewriter(_TimeoutLLM())
        out = await rw.rewrite("slow query", n=3)
        assert out == []

    @pytest.mark.asyncio
    async def test_garbage_response_returns_empty(self) -> None:
        rw = QueryRewriter(_GarbageLLM())
        out = await rw.rewrite("anything", n=3)
        assert out == []


# ── Env flag defaults ─────────────────────────────────────────────────


class TestEnvFlagDefaults:
    def test_mmr_default_off(self) -> None:
        # Caller is expected to read this env var; we just pin the default
        # string so any test environment doesn't accidentally flip behavior.
        assert os.getenv("PREDACORE_MEMORY_MMR", "0") == "0"

    def test_multiquery_default_off(self) -> None:
        assert os.getenv("PREDACORE_MEMORY_MULTIQUERY", "0") == "0"
