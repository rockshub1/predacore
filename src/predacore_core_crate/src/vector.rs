/// SIMD-accelerated vector search — cosine similarity + top-k selection.
///
/// Replaces numpy-based vector search in PredaCore memory store.
/// ~10-50x faster for large vector sets.

/// Compute cosine similarity between two vectors.
/// Both vectors are assumed to be L2-normalized (dot product = cosine sim).
/// Falls back to safe computation if not normalized.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    let len = a.len().min(b.len());

    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    // Process in chunks of 8 for auto-vectorization (SIMD)
    let chunks = len / 8;
    let remainder = len % 8;

    for i in 0..chunks {
        let base = i * 8;
        for j in 0..8 {
            let ai = unsafe { *a.get_unchecked(base + j) };
            let bi = unsafe { *b.get_unchecked(base + j) };
            dot += ai * bi;
            norm_a += ai * ai;
            norm_b += bi * bi;
        }
    }

    // Handle remainder
    let base = chunks * 8;
    for j in 0..remainder {
        let ai = unsafe { *a.get_unchecked(base + j) };
        let bi = unsafe { *b.get_unchecked(base + j) };
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    dot / denom
}

/// L2-normalize a vector in-place.
pub fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vec.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Search for top-k most similar vectors.
/// Returns Vec<(index, score)> sorted by score descending.
pub fn vector_search(query: &[f32], vectors: &[Vec<f32>], top_k: usize) -> Vec<(usize, f32)> {
    if vectors.is_empty() || query.is_empty() {
        return Vec::new();
    }

    let mut scores: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_similarity(query, v)))
        .collect();

    // Partial sort: only need top-k, no need to sort everything
    if top_k < scores.len() {
        scores.select_nth_unstable_by(top_k, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(top_k);
    }

    // Sort the top-k by score descending
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores
}

/// Parallel vector search using rayon for large vector sets.
/// Uses rayon for parallel cosine similarity computation.
pub fn vector_search_parallel(query: &[f32], vectors: &[Vec<f32>], top_k: usize) -> Vec<(usize, f32)> {
    use rayon::prelude::*;

    if vectors.is_empty() || query.is_empty() {
        return Vec::new();
    }

    let mut scores: Vec<(usize, f32)> = vectors
        .par_iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_similarity(query, v)))
        .collect();

    if top_k < scores.len() {
        scores.select_nth_unstable_by(top_k, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(top_k);
    }

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similar() {
        let a = vec![1.0, 1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim > 0.5 && sim < 1.0);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero() {
        let mut v = vec![0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }

    #[test]
    fn test_vector_search_basic() {
        let query = vec![1.0, 0.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0, 0.0],  // identical
            vec![0.0, 1.0, 0.0],  // orthogonal
            vec![0.9, 0.1, 0.0],  // similar
        ];
        let results = vector_search(&query, &vectors, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // identical should be first
    }

    #[test]
    fn test_vector_search_empty() {
        let query = vec![1.0, 0.0];
        let vectors: Vec<Vec<f32>> = Vec::new();
        assert!(vector_search(&query, &vectors, 5).is_empty());
    }

    #[test]
    fn test_vector_search_top_k_larger_than_corpus() {
        let query = vec![1.0, 0.0];
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let results = vector_search(&query, &vectors, 100);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_vector_search_384_dim() {
        // Simulate GTE-small dimensionality
        let mut query = vec![0.0_f32; 384];
        query[0] = 1.0;
        let mut v1 = vec![0.0_f32; 384];
        v1[0] = 0.9;
        v1[1] = 0.1;
        let mut v2 = vec![0.0_f32; 384];
        v2[100] = 1.0;
        let results = vector_search(&query, &[v1, v2], 1);
        assert_eq!(results[0].0, 0); // v1 more similar
    }
}
