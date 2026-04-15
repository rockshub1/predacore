/// Relation type classification from sentence patterns.
///
/// Detects 8 relation types between entities using regex patterns.
/// No LLM needed.

use once_cell::sync::Lazy;
use regex::Regex;

/// Supported relation types between entities.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RelationType {
    Uses,
    DependsOn,
    Imports,
    ReplacedBy,
    PartOf,
    FailedWith,
    ConfiguredWith,
    SimilarTo,
    RelatedTo, // fallback
}

impl RelationType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Uses => "uses",
            Self::DependsOn => "depends_on",
            Self::Imports => "imports",
            Self::ReplacedBy => "replaced_by",
            Self::PartOf => "part_of",
            Self::FailedWith => "failed_with",
            Self::ConfiguredWith => "configured_with",
            Self::SimilarTo => "similar_to",
            Self::RelatedTo => "related_to",
        }
    }
}

// ── Pattern matchers ────────────────────────────────────────────────

struct RelationPattern {
    regex: Regex,
    relation: RelationType,
}

static PATTERNS: Lazy<Vec<RelationPattern>> = Lazy::new(|| {
    vec![
        // Uses: "X uses Y", "X using Y", "X with Y"
        RelationPattern {
            regex: Regex::new(r"(?i)\buses?\b|\busing\b|\butilize[sd]?\b|\bemploy[sed]*\b").unwrap(),
            relation: RelationType::Uses,
        },
        // DependsOn: "X depends on Y", "X requires Y", "X needs Y"
        RelationPattern {
            regex: Regex::new(r"(?i)\bdepends?\s+on\b|\brequires?\b|\bneeds?\b|\brelies?\s+on\b").unwrap(),
            relation: RelationType::DependsOn,
        },
        // Imports: "X imports Y", "import X from Y"
        RelationPattern {
            regex: Regex::new(r"(?i)\bimports?\b|\bfrom\s+\S+\s+import\b|\brequire[sd]?\b").unwrap(),
            relation: RelationType::Imports,
        },
        // ReplacedBy: "replaced X with Y", "migrated from X to Y", "switched from X to Y"
        RelationPattern {
            regex: Regex::new(r"(?i)\breplaced?\b|\bmigrat(?:ed?|ing)\b|\bswitched?\b|\bupgraded?\b").unwrap(),
            relation: RelationType::ReplacedBy,
        },
        // PartOf: "X is part of Y", "X inside Y", "X within Y", "X belongs to Y"
        RelationPattern {
            regex: Regex::new(r"(?i)\bpart\s+of\b|\binside\b|\bwithin\b|\bbelongs?\s+to\b|\bcontained?\s+in\b").unwrap(),
            relation: RelationType::PartOf,
        },
        // FailedWith: "X failed with Y", "X error Y", "X crashed", "X broke"
        RelationPattern {
            regex: Regex::new(r"(?i)\bfailed?\b|\berror\b|\bcrashed?\b|\bbroke[n]?\b|\bbug\b|\btimeout\b").unwrap(),
            relation: RelationType::FailedWith,
        },
        // ConfiguredWith: "X configured with Y", "X set to Y", "X enabled Y"
        RelationPattern {
            regex: Regex::new(r"(?i)\bconfigured?\b|\bset\s+(?:to|up)\b|\benabled?\b|\bsettings?\b").unwrap(),
            relation: RelationType::ConfiguredWith,
        },
        // SimilarTo: "X similar to Y", "X like Y", "X alternative to Y"
        RelationPattern {
            regex: Regex::new(r"(?i)\bsimilar\s+to\b|\blike\b|\balternative\b|\bcomparable\b|\bequivalent\b").unwrap(),
            relation: RelationType::SimilarTo,
        },
    ]
});

fn confidence_for(rel: &RelationType) -> f32 {
    match rel {
        RelationType::ReplacedBy => 0.85,
        RelationType::DependsOn => 0.80,
        RelationType::Imports => 0.80,
        RelationType::PartOf => 0.75,
        RelationType::FailedWith => 0.75,
        RelationType::Uses => 0.70,
        RelationType::ConfiguredWith => 0.70,
        RelationType::SimilarTo => 0.65,
        _ => 0.50,
    }
}

/// Extract the window of text between two entities (plus 30 chars on each side)
/// for more precise verb-phrase matching. Safe with multi-byte UTF-8.
fn extract_entity_window(sentence: &str, entity_a: &str, entity_b: &str) -> Option<String> {
    let lower = sentence.to_lowercase();
    let pos_a = lower.find(&entity_a.to_lowercase())?;
    let pos_b = lower.find(&entity_b.to_lowercase())?;

    let (first_pos, first_len, _, second_len) = if pos_a < pos_b {
        (pos_a, entity_a.len(), pos_b, entity_b.len())
    } else {
        (pos_b, entity_b.len(), pos_a, entity_a.len())
    };

    let window_start = first_pos.saturating_sub(30);
    let window_end = (pos_a.max(pos_b) + second_len.max(first_len) + 30).min(sentence.len());

    // Snap to char boundaries to avoid splitting multi-byte UTF-8
    let mut safe_start = window_start;
    while safe_start > 0 && !sentence.is_char_boundary(safe_start) {
        safe_start -= 1;
    }
    let mut safe_end = window_end;
    while safe_end < sentence.len() && !sentence.is_char_boundary(safe_end) {
        safe_end += 1;
    }

    Some(sentence[safe_start..safe_end].to_string())
}

/// Classify the relation type between two entities based on the sentence context.
/// When both entities are found in the sentence, extracts a ~60-char window around
/// them for more precise verb-phrase matching. Returns (relation, confidence).
pub fn classify_relation(sentence: &str, entity_a: &str, entity_b: &str) -> (RelationType, f32) {
    // Extract window around entities if possible, else use full sentence
    let context = if !entity_a.is_empty() && !entity_b.is_empty() {
        extract_entity_window(sentence, entity_a, entity_b).unwrap_or_else(|| sentence.to_string())
    } else {
        sentence.to_string()
    };

    let mut best_relation = RelationType::RelatedTo;
    let mut best_confidence = 0.0_f32;

    for pattern in PATTERNS.iter() {
        if pattern.regex.is_match(&context) {
            let confidence = confidence_for(&pattern.relation);
            if confidence > best_confidence {
                best_confidence = confidence;
                best_relation = pattern.relation.clone();
            }
        }
    }

    if best_confidence == 0.0 {
        best_confidence = 0.30;
    }

    (best_relation, best_confidence)
}

/// Classify multiple potential relations from a text containing multiple entities.
/// Returns all detected relation types.
pub fn classify_all_relations(sentence: &str) -> Vec<(RelationType, f32)> {
    PATTERNS
        .iter()
        .filter(|p| p.regex.is_match(sentence))
        .map(|p| (p.relation.clone(), confidence_for(&p.relation)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uses() {
        let (rel, _) = classify_relation("JARVIS uses Python for scripting", "JARVIS", "Python");
        assert_eq!(rel, RelationType::Uses);
    }

    #[test]
    fn test_depends_on() {
        let (rel, _) = classify_relation("core.py depends on config.py", "core.py", "config.py");
        assert_eq!(rel, RelationType::DependsOn);
    }

    #[test]
    fn test_imports() {
        let (rel, _) = classify_relation("from jarvis.config import load_config", "jarvis", "config");
        assert_eq!(rel, RelationType::Imports);
    }

    #[test]
    fn test_replaced_by() {
        let (rel, _) = classify_relation("We replaced Flask with FastAPI", "Flask", "FastAPI");
        assert_eq!(rel, RelationType::ReplacedBy);
    }

    #[test]
    fn test_part_of() {
        let (rel, _) = classify_relation("OAuth is part of the auth module", "OAuth", "auth");
        assert_eq!(rel, RelationType::PartOf);
    }

    #[test]
    fn test_failed_with() {
        let (rel, _) = classify_relation("web_search failed with timeout", "web_search", "timeout");
        assert_eq!(rel, RelationType::FailedWith);
    }

    #[test]
    fn test_configured_with() {
        let (rel, _) = classify_relation("Redis configured with max memory 256MB", "Redis", "memory");
        assert_eq!(rel, RelationType::ConfiguredWith);
    }

    #[test]
    fn test_similar_to() {
        let (rel, _) = classify_relation("Rust is similar to C++ in performance", "Rust", "C++");
        assert_eq!(rel, RelationType::SimilarTo);
    }

    #[test]
    fn test_fallback_related_to() {
        let (rel, conf) = classify_relation("Python and Docker are great", "Python", "Docker");
        assert_eq!(rel, RelationType::RelatedTo);
        assert!(conf < 0.5);
    }

    #[test]
    fn test_case_insensitive() {
        let (rel, _) = classify_relation("jarvis USES python", "jarvis", "python");
        assert_eq!(rel, RelationType::Uses);
    }

    #[test]
    fn test_classify_all() {
        let rels = classify_all_relations("We replaced the old config and set up the new one");
        let types: Vec<&str> = rels.iter().map(|(r, _)| r.as_str()).collect();
        assert!(types.contains(&"replaced_by"));
        assert!(types.contains(&"configured_with"));
    }

    #[test]
    fn test_relation_as_str() {
        assert_eq!(RelationType::Uses.as_str(), "uses");
        assert_eq!(RelationType::DependsOn.as_str(), "depends_on");
        assert_eq!(RelationType::ReplacedBy.as_str(), "replaced_by");
    }
}
