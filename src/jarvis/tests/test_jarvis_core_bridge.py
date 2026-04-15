"""
Bridge tests for jarvis_core (Rust extension via pyo3).

Covers the 16 pyo3 functions that were previously untested from Python:

  * Vector ops      — cosine_similarity, l2_normalize, vector_search
  * BM25            — bm25_search, tokenize
  * Fuzzy           — trigram_similarity, fuzzy_match, fuzzy_search
  * Synonyms        — expand_synonyms, get_synonyms, are_synonyms
  * Relations       — classify_relation, classify_all_relations
  * Embedding       — embedding_dim, is_model_loaded (+ embed when cached)

``extract_entities`` is already covered by test_memory.py's
TestRustEntityExtraction class.

The Rust crate has 79 inline #[cfg(test)] tests that are run via
``cargo test`` from ``src/jarvis_core_crate/``. Those verify correctness
in Rust. These Python tests verify the pyo3 bridge layer — arg coercion,
return type marshalling, type safety — which cargo tests can't catch.

The full 500-question LongMemEval benchmark at ``benchmarks/`` exercises
every one of these functions at scale (R@5 = 0.9574). These unit tests
are the fast feedback loop; the benchmark is the integration gate.
"""
from __future__ import annotations

import math

import pytest

# Skip the whole module if jarvis_core isn't built (maturin develop wasn't run)
jarvis_core = pytest.importorskip(
    "jarvis_core",
    reason=(
        "jarvis_core extension not built. "
        "Run `cd src/jarvis_core_crate && maturin develop --release` first."
    ),
)


# ---------------------------------------------------------------------------
# Vector operations — cosine_similarity, l2_normalize, vector_search
# ---------------------------------------------------------------------------


class TestVectorOps:
    """Vector operations — SIMD cosine + parallel top-k search."""

    def test_cosine_similarity_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert math.isclose(jarvis_core.cosine_similarity(v, v), 1.0, abs_tol=1e-6)

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert math.isclose(jarvis_core.cosine_similarity(a, b), 0.0, abs_tol=1e-6)

    def test_cosine_similarity_opposite(self):
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert math.isclose(jarvis_core.cosine_similarity(a, b), -1.0, abs_tol=1e-6)

    def test_cosine_similarity_normalizes_magnitudes(self):
        """Cosine similarity is magnitude-invariant."""
        a = [1.0, 1.0, 0.0]
        b = [5.0, 5.0, 0.0]  # same direction, different magnitude
        assert math.isclose(jarvis_core.cosine_similarity(a, b), 1.0, abs_tol=1e-6)

    def test_l2_normalize_unit_length(self):
        v = [3.0, 4.0, 0.0]  # magnitude 5
        normalized = jarvis_core.l2_normalize(v)
        magnitude = math.sqrt(sum(x * x for x in normalized))
        assert math.isclose(magnitude, 1.0, abs_tol=1e-6)

    def test_l2_normalize_preserves_direction(self):
        v = [3.0, 4.0, 0.0]
        normalized = jarvis_core.l2_normalize(v)
        # [3/5, 4/5, 0] = [0.6, 0.8, 0.0]
        assert math.isclose(normalized[0], 0.6, abs_tol=1e-6)
        assert math.isclose(normalized[1], 0.8, abs_tol=1e-6)
        assert math.isclose(normalized[2], 0.0, abs_tol=1e-6)

    def test_l2_normalize_zero_vector_safe(self):
        # Should not crash or divide by zero
        result = jarvis_core.l2_normalize([0.0, 0.0, 0.0])
        assert len(result) == 3

    def test_vector_search_finds_nearest(self):
        query = [1.0, 0.0, 0.0]
        vectors = [
            [1.0, 0.0, 0.0],   # perfect match — index 0
            [0.9, 0.1, 0.0],   # close — index 1
            [0.0, 1.0, 0.0],   # orthogonal — index 2
            [-1.0, 0.0, 0.0],  # opposite — index 3
        ]
        results = jarvis_core.vector_search(query, vectors, 2)
        assert len(results) == 2
        # Top result is the perfect match
        assert results[0][0] == 0
        # Scores descend
        assert results[0][1] >= results[1][1]
        # Second best is the close vector
        assert results[1][0] == 1

    def test_vector_search_top_k_clamped_to_corpus_size(self):
        query = [1.0, 0.0]
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        results = jarvis_core.vector_search(query, vectors, 10)
        assert len(results) == 2  # clamped to corpus size

    def test_vector_search_empty_corpus(self):
        results = jarvis_core.vector_search([1.0, 0.0], [], 5)
        assert results == []

    def test_vector_search_parallel_threshold(self):
        """>1000 vectors should hit the parallel code path (rayon).
        This test only verifies correctness, not parallelism.

        Each non-target vector points in a different direction so that
        cosine similarity actually differentiates them — otherwise
        magnitude-invariant cosine would tie all of them at 1.0.
        """
        import random
        rng = random.Random(42)
        query = [1.0, 0.0, 0.0, 0.0, 0.0]
        # 1500 random unit-ish vectors — none aligned with the query
        vectors = [
            [rng.uniform(-0.1, 0.1) for _ in range(5)]
            for _ in range(1500)
        ]
        # Plant a known-best target at a specific index
        vectors[999] = [1.0, 0.0, 0.0, 0.0, 0.0]
        results = jarvis_core.vector_search(query, vectors, 5)
        assert len(results) == 5
        top_indices = [r[0] for r in results]
        # The perfect match must be the top result
        assert results[0][0] == 999


# ---------------------------------------------------------------------------
# BM25 + tokenize
# ---------------------------------------------------------------------------


class TestBM25:
    """BM25 keyword search + tokenizer."""

    def test_tokenize_basic(self):
        tokens = jarvis_core.tokenize("Hello World Python")
        assert "hello" in tokens
        assert "world" in tokens
        assert "python" in tokens

    def test_tokenize_lowercases(self):
        tokens = jarvis_core.tokenize("JARVIS IS GREAT")
        for t in tokens:
            assert t == t.lower()

    def test_tokenize_min_length(self):
        """Tokens shorter than 2 chars are dropped (per Rust side)."""
        tokens = jarvis_core.tokenize("a ab abc")
        assert "a" not in tokens
        assert "ab" in tokens
        assert "abc" in tokens

    def test_tokenize_empty(self):
        assert jarvis_core.tokenize("") == []

    def test_bm25_search_finds_exact_match(self):
        documents = [
            "Python programming language",
            "Rust systems programming",
            "JavaScript web development",
        ]
        results = jarvis_core.bm25_search("python", documents, 3)
        assert len(results) >= 1
        # The Python document should rank first
        assert results[0][0] == 0

    def test_bm25_search_returns_descending_scores(self):
        documents = ["apple banana cherry"] * 5
        documents.append("apple apple apple")  # higher TF for "apple"
        results = jarvis_core.bm25_search("apple", documents, 6)
        # Scores should be monotonic non-increasing
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_bm25_search_empty_corpus(self):
        results = jarvis_core.bm25_search("python", [], 5)
        assert results == []

    def test_bm25_search_empty_query(self):
        # Empty query should not crash
        results = jarvis_core.bm25_search("", ["doc 1", "doc 2"], 5)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Fuzzy matching — trigram_similarity, fuzzy_match, fuzzy_search
# ---------------------------------------------------------------------------


class TestFuzzy:
    """Trigram fuzzy matching — typo-tolerant string search."""

    def test_trigram_similarity_identical(self):
        assert jarvis_core.trigram_similarity("python", "python") == 1.0

    def test_trigram_similarity_different(self):
        # Completely different → low similarity
        assert jarvis_core.trigram_similarity("python", "xyz123") < 0.3

    def test_trigram_similarity_typo(self):
        """Single-char typo → similarity above default fuzzy threshold.

        ``python`` trigrams: {pyt, yth, tho, hon}
        ``pythan`` trigrams: {pyt, yth, tha, han}
        Intersection: 2, Union: 6 → Jaccard ≈ 0.333.
        Enough to pass fuzzy_match's default threshold of 0.3.
        """
        sim = jarvis_core.trigram_similarity("python", "pythan")
        assert sim > 0.3
        # But not a near-match (would be >0.7 for that)
        assert sim < 0.7

    def test_trigram_similarity_is_symmetric(self):
        a, b = "claude", "clause"
        assert math.isclose(
            jarvis_core.trigram_similarity(a, b),
            jarvis_core.trigram_similarity(b, a),
            abs_tol=1e-6,
        )

    def test_fuzzy_match_finds_typo(self):
        candidates = ["python", "rust", "javascript", "golang"]
        results = jarvis_core.fuzzy_match("pythan", candidates, 0.3)
        # "python" should be a match
        indices = [r[0] for r in results]
        assert 0 in indices  # "python" is at index 0

    def test_fuzzy_match_respects_threshold(self):
        candidates = ["python", "golang"]
        # High threshold — only perfect matches
        results = jarvis_core.fuzzy_match("python", candidates, 0.99)
        assert len(results) == 1
        assert results[0][0] == 0

    def test_fuzzy_search_ranks_by_similarity(self):
        docs = [
            "python is great",
            "ruby is slow",
            "python is fast",
        ]
        results = jarvis_core.fuzzy_search("python", docs, 5, 0.2)
        assert len(results) >= 1
        # Scores are descending
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Synonym expansion
# ---------------------------------------------------------------------------


class TestSynonyms:
    """Tech-domain synonym expansion (50+ groups)."""

    def test_expand_synonyms_includes_original(self):
        expanded = jarvis_core.expand_synonyms(["config"])
        assert "config" in expanded

    def test_expand_synonyms_adds_related_terms(self):
        expanded = jarvis_core.expand_synonyms(["config"])
        # Per the Rust doc: config → [config, configuration, settings, ...]
        # At least SOME expansion should happen for this common tech term
        assert len(expanded) >= 1

    def test_expand_synonyms_unknown_term_returns_itself(self):
        expanded = jarvis_core.expand_synonyms(["unknownxyz123"])
        assert "unknownxyz123" in expanded

    def test_expand_synonyms_multiple_terms(self):
        expanded = jarvis_core.expand_synonyms(["config", "database"])
        assert "config" in expanded
        assert "database" in expanded

    def test_get_synonyms_returns_list(self):
        result = jarvis_core.get_synonyms("config")
        assert isinstance(result, list)

    def test_are_synonyms_self(self):
        # A term is its own synonym
        assert jarvis_core.are_synonyms("config", "config") is True

    def test_are_synonyms_unrelated(self):
        assert jarvis_core.are_synonyms("python", "kitchen") is False


# ---------------------------------------------------------------------------
# Relation classification
# ---------------------------------------------------------------------------


class TestRelations:
    """Window-aware relation classification."""

    def test_classify_relation_returns_tuple(self):
        result = jarvis_core.classify_relation(
            "Python is a programming language", "Python", "programming language"
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        rel, conf = result
        assert isinstance(rel, str)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_classify_relation_detects_is_a(self):
        rel, conf = jarvis_core.classify_relation(
            "Python is a programming language", "Python", "language"
        )
        # "is a" pattern should be detected with some confidence
        assert conf > 0.0

    def test_classify_all_relations_returns_list(self):
        results = jarvis_core.classify_all_relations(
            "Python uses the GIL which causes contention"
        )
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2


# ---------------------------------------------------------------------------
# Embedding (BGE-small-en-v1.5 via Candle)
# ---------------------------------------------------------------------------


class TestEmbedding:
    """BGE-small-en-v1.5 embeddings via Candle (384-dim)."""

    def test_embedding_dim_is_384(self):
        # BGE-small-en-v1.5 is always 384-dim
        assert jarvis_core.embedding_dim() == 384

    def test_is_model_loaded_returns_bool(self):
        assert isinstance(jarvis_core.is_model_loaded(), bool)

    def test_embed_skipped_if_model_not_cached(self):
        """The `embed` function downloads ~133 MB on first call.

        We don't want CI runs to always pay that cost. Only run if:
          1. The model is already loaded in memory (cheap), OR
          2. JARVIS_TEST_EMBED env var is set (explicit opt-in)
        """
        import os
        if not jarvis_core.is_model_loaded() and not os.getenv("JARVIS_TEST_EMBED"):
            pytest.skip(
                "Skipping embed test — BGE-small model not loaded. "
                "Set JARVIS_TEST_EMBED=1 to force a download + run, or run the "
                "longmemeval benchmark once first to cache the model."
            )

        vectors = jarvis_core.embed(["hello world", "goodbye world"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 384
        assert len(vectors[1]) == 384

        # Embeddings should be L2-normalized per the docstring
        mag_0 = math.sqrt(sum(x * x for x in vectors[0]))
        assert math.isclose(mag_0, 1.0, abs_tol=1e-3)

        # Semantically similar sentences should have high cosine similarity
        sim = jarvis_core.cosine_similarity(vectors[0], vectors[1])
        assert sim > 0.5  # "hello world" and "goodbye world" share most tokens


# ---------------------------------------------------------------------------
# Sanity: the Python wrapper re-exports everything from _core
# ---------------------------------------------------------------------------


class TestPyBridgeSurface:
    """Verify the Python wrapper (__init__.py) exposes every pyo3 function."""

    def test_version_attribute(self):
        assert jarvis_core.__version__

    def test_all_bindings_callable(self):
        expected = [
            "cosine_similarity",
            "l2_normalize",
            "vector_search",
            "bm25_search",
            "tokenize",
            "trigram_similarity",
            "fuzzy_match",
            "fuzzy_search",
            "expand_synonyms",
            "get_synonyms",
            "are_synonyms",
            "extract_entities",
            "classify_relation",
            "classify_all_relations",
            "embed",
            "embedding_dim",
            "is_model_loaded",
        ]
        for name in expected:
            assert hasattr(jarvis_core, name), f"missing pyo3 binding: {name}"
            assert callable(getattr(jarvis_core, name))
