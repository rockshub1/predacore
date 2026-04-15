/// BM25 search engine — keyword-based document ranking.
///
/// Replaces Python BM25 implementation in JARVIS memory store.
/// Standard BM25 with k1=1.5, b=0.75.

use std::collections::HashMap;

const K1: f32 = 1.5;
const B: f32 = 0.75;

/// Tokenize text into lowercase terms (min 2 chars).
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| w.len() >= 2)
        .map(|w| w.to_string())
        .collect()
}

/// Compute BM25 scores for a query against a corpus of documents.
/// Returns Vec<(doc_index, score)> sorted by score descending, top-k only.
pub fn bm25_search(query: &str, documents: &[String], top_k: usize) -> Vec<(usize, f32)> {
    if documents.is_empty() || query.is_empty() {
        return Vec::new();
    }

    let query_terms = tokenize(query);
    if query_terms.is_empty() {
        return Vec::new();
    }

    // Pre-compute document lengths and average
    let doc_tokens: Vec<Vec<String>> = documents.iter().map(|d| tokenize(d)).collect();
    let doc_lengths: Vec<f32> = doc_tokens.iter().map(|t| t.len() as f32).collect();
    let avg_dl = doc_lengths.iter().sum::<f32>() / doc_lengths.len().max(1) as f32;
    let n_docs = documents.len() as f32;

    // Compute document frequency for each query term
    let mut df: HashMap<&str, f32> = HashMap::new();
    for term in &query_terms {
        let count = doc_tokens
            .iter()
            .filter(|tokens| tokens.iter().any(|t| t == term))
            .count();
        df.insert(term.as_str(), count as f32);
    }

    // Score each document
    let mut scores: Vec<(usize, f32)> = doc_tokens
        .iter()
        .enumerate()
        .filter_map(|(i, tokens)| {
            let doc_len = doc_lengths[i];
            let mut score = 0.0_f32;

            for term in &query_terms {
                let tf = tokens.iter().filter(|t| *t == term).count() as f32;
                if tf > 0.0 {
                    let doc_freq = df.get(term.as_str()).copied().unwrap_or(0.0);
                    // IDF with smoothing
                    let idf = ((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln();
                    let numerator = tf * (K1 + 1.0);
                    let denominator = tf + K1 * (1.0 - B + B * doc_len / avg_dl.max(1.0));
                    score += idf * numerator / denominator;
                }
            }

            if score > 0.0 {
                Some((i, score))
            } else {
                None
            }
        })
        .collect();

    // Normalize to [0, 1]
    if let Some(max_score) = scores.iter().map(|(_, s)| *s).reduce(f32::max) {
        if max_score > 0.0 {
            scores.iter_mut().for_each(|(_, s)| *s /= max_score);
        }
    }

    // Sort by score descending and truncate to top-k
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(top_k);
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello World, this is a test!");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Single char "a" should be filtered
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_tokenize_underscore() {
        let tokens = tokenize("web_search tool_name");
        assert!(tokens.contains(&"web_search".to_string()));
        assert!(tokens.contains(&"tool_name".to_string()));
    }

    #[test]
    fn test_tokenize_empty() {
        assert!(tokenize("").is_empty());
    }

    #[test]
    fn test_bm25_basic() {
        let docs = vec![
            "Python is great for AI and machine learning".to_string(),
            "JavaScript is used for web development".to_string(),
            "Rust is fast and safe for systems programming".to_string(),
        ];
        let results = bm25_search("Python AI", &docs, 3);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0); // Python doc should rank first
    }

    #[test]
    fn test_bm25_no_match() {
        let docs = vec!["hello world".to_string()];
        let results = bm25_search("xyzzy", &docs, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_empty_corpus() {
        let results = bm25_search("test", &[], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_empty_query() {
        let docs = vec!["hello".to_string()];
        let results = bm25_search("", &docs, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_scores_normalized() {
        let docs = vec![
            "Python Python Python".to_string(),
            "Python once".to_string(),
        ];
        let results = bm25_search("Python", &docs, 2);
        // Highest score should be 1.0 (normalized)
        assert!((results[0].1 - 1.0).abs() < 1e-6);
        // Second score should be less than 1.0
        assert!(results[1].1 < 1.0);
    }

    #[test]
    fn test_bm25_multiple_terms() {
        let docs = vec![
            "Python machine learning AI".to_string(),
            "Python web framework".to_string(),
            "JavaScript React frontend".to_string(),
        ];
        let results = bm25_search("Python machine learning", &docs, 3);
        assert_eq!(results[0].0, 0); // Best match has all 3 terms
    }

    #[test]
    fn test_bm25_top_k_limits() {
        let docs: Vec<String> = (0..100).map(|i| format!("document {i} with test content")).collect();
        let results = bm25_search("test", &docs, 5);
        assert!(results.len() <= 5);
    }
}
