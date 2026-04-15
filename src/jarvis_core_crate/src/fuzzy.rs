/// Trigram-based fuzzy matching — typo-tolerant search.
///
/// "congif" → finds "config" (trigram overlap)
/// "authetication" → finds "authentication"

use std::collections::HashSet;

/// Extract trigrams (3-character substrings) from text.
pub fn trigrams(text: &str) -> HashSet<String> {
    let lower = text.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    if chars.len() < 3 {
        let mut set = HashSet::new();
        if !lower.is_empty() {
            set.insert(lower);
        }
        return set;
    }
    chars
        .windows(3)
        .map(|w| w.iter().collect::<String>())
        .collect()
}

/// Compute Jaccard similarity between trigram sets of two strings.
/// Returns a value in [0.0, 1.0] where 1.0 is identical.
pub fn trigram_similarity(a: &str, b: &str) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let tri_a = trigrams(a);
    let tri_b = trigrams(b);

    let intersection = tri_a.intersection(&tri_b).count() as f32;
    let union = tri_a.union(&tri_b).count() as f32;

    if union == 0.0 {
        return 0.0;
    }
    intersection / union
}

/// Find fuzzy matches for a query against a list of candidates.
/// Returns Vec<(index, similarity)> for candidates above threshold,
/// sorted by similarity descending.
pub fn fuzzy_match(query: &str, candidates: &[String], threshold: f32) -> Vec<(usize, f32)> {
    if query.is_empty() || candidates.is_empty() {
        return Vec::new();
    }

    let mut results: Vec<(usize, f32)> = candidates
        .iter()
        .enumerate()
        .filter_map(|(i, candidate)| {
            let sim = trigram_similarity(query, candidate);
            if sim >= threshold {
                Some((i, sim))
            } else {
                None
            }
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Fuzzy search across document contents.
/// Splits query into terms, finds fuzzy matches for each term in each document,
/// returns aggregate scores per document.
pub fn fuzzy_search(query: &str, documents: &[String], top_k: usize, threshold: f32) -> Vec<(usize, f32)> {
    if query.is_empty() || documents.is_empty() {
        return Vec::new();
    }

    let query_terms: Vec<&str> = query.split_whitespace().filter(|w| w.len() >= 2).collect();
    if query_terms.is_empty() {
        return Vec::new();
    }

    let mut scores: Vec<(usize, f32)> = documents
        .iter()
        .enumerate()
        .filter_map(|(doc_idx, doc)| {
            let doc_words: Vec<String> = doc
                .to_lowercase()
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .filter(|w| w.len() >= 2)
                .map(|w| w.to_string())
                .collect();

            let mut total_score = 0.0_f32;
            let mut matches = 0;

            for term in &query_terms {
                let best_match = doc_words
                    .iter()
                    .map(|w| trigram_similarity(term, w))
                    .fold(0.0_f32, f32::max);

                if best_match >= threshold {
                    total_score += best_match;
                    matches += 1;
                }
            }

            if matches > 0 {
                // Average score weighted by match coverage
                let coverage = matches as f32 / query_terms.len() as f32;
                let avg_score = total_score / matches as f32;
                Some((doc_idx, avg_score * coverage))
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

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(top_k);
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigrams_basic() {
        let tris = trigrams("config");
        assert!(tris.contains("con"));
        assert!(tris.contains("onf"));
        assert!(tris.contains("nfi"));
        assert!(tris.contains("fig"));
    }

    #[test]
    fn test_trigrams_short() {
        let tris = trigrams("hi");
        assert_eq!(tris.len(), 1);
        assert!(tris.contains("hi"));
    }

    #[test]
    fn test_trigrams_empty() {
        assert!(trigrams("").is_empty());
    }

    #[test]
    fn test_trigrams_case_insensitive() {
        let a = trigrams("Config");
        let b = trigrams("config");
        assert_eq!(a, b);
    }

    #[test]
    fn test_similarity_identical() {
        assert!((trigram_similarity("config", "config") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_typo() {
        let sim = trigram_similarity("config", "congif");
        // Short words have fewer trigrams, so overlap is lower
        assert!(sim > 0.1, "typo should have some similarity: {sim}");
        // Longer words with typos have much better trigram overlap
        let sim2 = trigram_similarity("authentication", "authetication");
        assert!(sim2 > 0.5, "longer typo should have high similarity: {sim2}");
    }

    #[test]
    fn test_similarity_similar_words() {
        let sim = trigram_similarity("authentication", "authetication");
        assert!(sim > 0.5, "near-typo should have high similarity: {sim}");
    }

    #[test]
    fn test_similarity_unrelated() {
        let sim = trigram_similarity("config", "banana");
        assert!(sim < 0.2, "unrelated words should have low similarity: {sim}");
    }

    #[test]
    fn test_similarity_empty() {
        assert_eq!(trigram_similarity("", ""), 1.0);
        assert_eq!(trigram_similarity("hello", ""), 0.0);
        assert_eq!(trigram_similarity("", "world"), 0.0);
    }

    #[test]
    fn test_fuzzy_match_basic() {
        let candidates = vec![
            "configuration".to_string(),
            "banana".to_string(),
            "config".to_string(),
        ];
        let results = fuzzy_match("congif", &candidates, 0.1);
        assert!(!results.is_empty());
        // "config" or "configuration" should be matches
        assert!(results.iter().any(|(i, _)| *i == 0 || *i == 2));
    }

    #[test]
    fn test_fuzzy_match_empty() {
        assert!(fuzzy_match("", &["test".to_string()], 0.3).is_empty());
        assert!(fuzzy_match("test", &[], 0.3).is_empty());
    }

    #[test]
    fn test_fuzzy_match_threshold() {
        let candidates = vec!["config".to_string(), "zzzzzzz".to_string()];
        let results = fuzzy_match("congif", &candidates, 0.8);
        // "zzzzzzz" should not match at high threshold
        assert!(results.iter().all(|(i, _)| *i != 1));
    }

    #[test]
    fn test_fuzzy_search_documents() {
        let docs = vec![
            "How to configure the authentication system".to_string(),
            "Banana smoothie recipe for breakfast".to_string(),
            "Config file for the auth module settings".to_string(),
        ];
        let results = fuzzy_search("congif authetication", &docs, 3, 0.2);
        assert!(!results.is_empty());
        // Doc 0 and 2 should rank higher than doc 1
        if results.len() >= 2 {
            assert!(results.iter().all(|(i, _)| *i != 1));
        }
    }
}
