//! predacore_core — Rust-powered intelligence layer for PredaCore.
//!
//! Hard dependency (no Python fallbacks). Provides:
//! - SIMD cosine similarity + top-k vector search (parallel for >1000 vectors)
//! - BM25 keyword search with IDF smoothing
//! - Trigram fuzzy matching (typo-tolerant)
//! - Tech-domain synonym expansion
//! - 3-tier entity extraction (dictionary + regex + stopwords)
//! - Window-aware relation classification
//! - BGE-small-en-v1.5 embeddings via Candle (384-dim, ~133 MB, MTEB 62.2)

mod bm25;
mod embedding;
mod entity;
mod fuzzy;
mod relations;
mod synonyms;
mod vector;

use pyo3::prelude::*;

/// Compute cosine similarity between two vectors.
#[pyfunction]
fn cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> f32 {
    vector::cosine_similarity(&a, &b)
}

/// L2-normalize a vector. Returns the normalized copy.
#[pyfunction]
fn l2_normalize(mut vec: Vec<f32>) -> Vec<f32> {
    vector::l2_normalize(&mut vec);
    vec
}

/// Search for the top-k most similar vectors to the query.
/// Returns list of (index, score) sorted by score descending.
#[pyfunction]
#[pyo3(signature = (query, vectors, top_k = 10))]
fn vector_search(query: Vec<f32>, vectors: Vec<Vec<f32>>, top_k: usize) -> Vec<(usize, f32)> {
    if vectors.len() > 1000 {
        vector::vector_search_parallel(&query, &vectors, top_k)
    } else {
        vector::vector_search(&query, &vectors, top_k)
    }
}

/// BM25 keyword search over a corpus of documents.
/// Returns list of (doc_index, score) sorted by score descending.
#[pyfunction]
#[pyo3(signature = (query, documents, top_k = 10))]
fn bm25_search(query: &str, documents: Vec<String>, top_k: usize) -> Vec<(usize, f32)> {
    bm25::bm25_search(query, &documents, top_k)
}

/// Tokenize text into lowercase terms (min 2 chars).
#[pyfunction]
fn tokenize(text: &str) -> Vec<String> {
    bm25::tokenize(text)
}

/// Compute trigram similarity between two strings (0.0 to 1.0).
#[pyfunction]
fn trigram_similarity(a: &str, b: &str) -> f32 {
    fuzzy::trigram_similarity(a, b)
}

/// Find fuzzy matches for a query in a list of candidates.
/// Returns list of (index, similarity) above threshold.
#[pyfunction]
#[pyo3(signature = (query, candidates, threshold = 0.3))]
fn fuzzy_match(query: &str, candidates: Vec<String>, threshold: f32) -> Vec<(usize, f32)> {
    fuzzy::fuzzy_match(query, &candidates, threshold)
}

/// Fuzzy search across document contents.
/// Returns list of (doc_index, score) sorted by score descending.
#[pyfunction]
#[pyo3(signature = (query, documents, top_k = 10, threshold = 0.3))]
fn fuzzy_search(query: &str, documents: Vec<String>, top_k: usize, threshold: f32) -> Vec<(usize, f32)> {
    fuzzy::fuzzy_search(query, &documents, top_k, threshold)
}

/// Expand query terms with tech domain synonyms.
/// "config" → ["config", "configuration", "settings", "preferences", "options"]
#[pyfunction]
fn expand_synonyms(terms: Vec<String>) -> Vec<String> {
    synonyms::expand_synonyms(&terms)
}

/// Get synonyms for a single term.
#[pyfunction]
fn get_synonyms(term: &str) -> Vec<String> {
    synonyms::get_synonyms(term)
}

/// Check if two terms are synonyms.
#[pyfunction]
fn are_synonyms(a: &str, b: &str) -> bool {
    synonyms::are_synonyms(a, b)
}

/// Extract entities from text using 3-tier strategy (dictionary + regex + stopwords).
/// Returns list of (name, entity_type, confidence, source_tier) tuples.
/// source_tier: 1=dictionary, 2=tfidf, 3=regex
#[pyfunction]
fn extract_entities(text: &str) -> Vec<(String, String, f32, u8)> {
    entity::extract_entities(text)
        .into_iter()
        .map(|e| (e.name, e.entity_type, e.confidence, e.source_tier))
        .collect()
}

/// Classify relation type between two entities from sentence context.
/// Returns (relation_type, confidence).
#[pyfunction]
fn classify_relation(sentence: &str, entity_a: &str, entity_b: &str) -> (String, f32) {
    let (rel, conf) = relations::classify_relation(sentence, entity_a, entity_b);
    (rel.as_str().to_string(), conf)
}

/// Detect all relation types present in a sentence.
/// Returns list of (relation_type, confidence).
#[pyfunction]
fn classify_all_relations(sentence: &str) -> Vec<(String, f32)> {
    relations::classify_all_relations(sentence)
        .into_iter()
        .map(|(r, c)| (r.as_str().to_string(), c))
        .collect()
}

/// Embed texts using GTE-small (384-dim). Downloads model on first call (~67MB).
/// Returns list of 384-dim float vectors, L2-normalized.
#[pyfunction]
fn embed(texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
    embedding::embed(&texts).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

/// Get the embedding dimension (384 for GTE-small).
#[pyfunction]
fn embedding_dim() -> usize {
    embedding::embedding_dim()
}

/// Check if the embedding model is already loaded in memory.
#[pyfunction]
fn is_model_loaded() -> bool {
    embedding::is_model_loaded()
}

/// predacore_core Python module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Vector operations
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(l2_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(vector_search, m)?)?;

    // BM25 search
    m.add_function(wrap_pyfunction!(bm25_search, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;

    // Fuzzy matching
    m.add_function(wrap_pyfunction!(trigram_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(fuzzy_match, m)?)?;
    m.add_function(wrap_pyfunction!(fuzzy_search, m)?)?;

    // Synonym expansion
    m.add_function(wrap_pyfunction!(expand_synonyms, m)?)?;
    m.add_function(wrap_pyfunction!(get_synonyms, m)?)?;
    m.add_function(wrap_pyfunction!(are_synonyms, m)?)?;

    // Entity extraction
    m.add_function(wrap_pyfunction!(extract_entities, m)?)?;

    // Relation classification
    m.add_function(wrap_pyfunction!(classify_relation, m)?)?;
    m.add_function(wrap_pyfunction!(classify_all_relations, m)?)?;

    // Embedding (BGE-small-en-v1.5 via Candle)
    m.add_function(wrap_pyfunction!(embed, m)?)?;
    m.add_function(wrap_pyfunction!(embedding_dim, m)?)?;
    m.add_function(wrap_pyfunction!(is_model_loaded, m)?)?;

    // Module metadata
    m.add("__version__", "0.1.0")?;

    Ok(())
}
