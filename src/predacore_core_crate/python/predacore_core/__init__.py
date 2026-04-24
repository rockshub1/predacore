"""predacore_core — Rust-powered intelligence layer for PredaCore."""

from predacore_core._core import (
    # Vector operations
    cosine_similarity,
    l2_normalize,
    vector_search,
    # BM25 search
    bm25_search,
    tokenize,
    # Fuzzy matching
    trigram_similarity,
    fuzzy_match,
    fuzzy_search,
    # Synonym expansion
    expand_synonyms,
    get_synonyms,
    are_synonyms,
    # Entity extraction
    extract_entities,
    # Relation classification
    classify_relation,
    classify_all_relations,
    # Embedding (GTE-small via Candle)
    embed,
    embedding_dim,
    is_model_loaded,
    # HNSW index class (O(log n) ANN vector search at scale)
    PyHnswIndex,
)

__version__ = "1.1.1"

__all__ = [
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
    "PyHnswIndex",
]
