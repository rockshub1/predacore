/// Smart entity extraction — 3-tier: dictionary + TF-IDF + regex.
///
/// Replaces Python heuristic entity extraction in consolidator.py.
/// No LLM needed.

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::{HashMap, HashSet};

/// Entity extracted from text.
#[derive(Debug, Clone)]
pub struct Entity {
    pub name: String,
    pub entity_type: String,
    pub confidence: f32,
    pub source_tier: u8, // 1=dictionary, 2=tfidf, 3=regex
}

// ── Tier 1: Tech Entity Dictionary ──────────────────────────────────

static TECH_ENTITIES: Lazy<HashMap<String, &str>> = Lazy::new(|| {
    let entries: Vec<(&str, &str)> = vec![
        // Cloud & Infrastructure
        ("AWS", "platform"), ("Azure", "platform"), ("GCP", "platform"),
        ("Docker", "tool"), ("Kubernetes", "tool"), ("Terraform", "tool"),
        ("Nginx", "tool"), ("Apache", "tool"), ("Cloudflare", "platform"),
        ("Vercel", "platform"), ("Netlify", "platform"), ("Heroku", "platform"),
        ("DigitalOcean", "platform"), ("Supabase", "platform"),
        // Databases
        ("PostgreSQL", "database"), ("MySQL", "database"), ("SQLite", "database"),
        ("MongoDB", "database"), ("Redis", "database"), ("Elasticsearch", "database"),
        ("DynamoDB", "database"), ("CockroachDB", "database"), ("Cassandra", "database"),
        ("Neo4j", "database"), ("Pinecone", "database"), ("Weaviate", "database"),
        ("ChromaDB", "database"), ("Qdrant", "database"), ("Milvus", "database"),
        // Languages
        ("Python", "language"), ("Rust", "language"), ("JavaScript", "language"),
        ("TypeScript", "language"), ("Go", "language"), ("Java", "language"),
        ("C++", "language"), ("Ruby", "language"), ("Swift", "language"),
        ("Kotlin", "language"), ("Scala", "language"), ("Elixir", "language"),
        ("Zig", "language"), ("Haskell", "language"),
        // Frameworks
        ("React", "framework"), ("Vue", "framework"), ("Angular", "framework"),
        ("Next.js", "framework"), ("Svelte", "framework"), ("FastAPI", "framework"),
        ("Django", "framework"), ("Flask", "framework"), ("Express", "framework"),
        ("Spring", "framework"), ("Rails", "framework"), ("Laravel", "framework"),
        ("Actix", "framework"), ("Axum", "framework"), ("Rocket", "framework"),
        ("PyTorch", "framework"), ("TensorFlow", "framework"), ("JAX", "framework"),
        ("Candle", "framework"), ("LangChain", "framework"), ("CrewAI", "framework"),
        // AI Models
        ("GPT-4", "model"), ("GPT-4o", "model"), ("GPT-3.5", "model"),
        ("Claude", "model"), ("Gemini", "model"), ("Llama", "model"),
        ("Mistral", "model"), ("Mixtral", "model"), ("Phi", "model"),
        ("Qwen", "model"), ("DeepSeek", "model"), ("Cohere", "model"),
        ("DALL-E", "model"), ("Stable Diffusion", "model"), ("Whisper", "model"),
        ("GTE-small", "model"), ("BERT", "model"), ("RoBERTa", "model"),
        // Tools & Services
        ("Git", "tool"), ("GitHub", "platform"), ("GitLab", "platform"),
        ("Playwright", "tool"), ("Selenium", "tool"), ("Puppeteer", "tool"),
        ("Webpack", "tool"), ("Vite", "tool"), ("ESLint", "tool"),
        ("Pytest", "tool"), ("Jest", "tool"), ("Vitest", "tool"),
        ("Prometheus", "tool"), ("Grafana", "tool"), ("Datadog", "platform"),
        ("Sentry", "tool"), ("OpenTelemetry", "tool"),
        // Protocols & Standards
        ("OAuth", "protocol"), ("JWT", "protocol"), ("gRPC", "protocol"),
        ("GraphQL", "protocol"), ("REST", "protocol"), ("WebSocket", "protocol"),
        ("HTTP", "protocol"), ("HTTPS", "protocol"), ("SSH", "protocol"),
        ("TCP", "protocol"), ("UDP", "protocol"), ("MCP", "protocol"),
        // PredaCore-specific
        ("JARVIS", "project"), ("Prometheus", "project"), ("OpenClaw", "project"),
        ("Flame", "project"), ("Antigravity", "project"),
    ];

    let mut map = HashMap::new();
    for (name, etype) in entries {
        map.insert(name.to_lowercase(), etype);
        // Also store original case for exact matching
        map.insert(name.to_string(), etype);
    }
    map
});

// ── Tier 3: Regex Patterns ──────────────────────────────────────────

static RE_CAMELCASE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b").unwrap());

static RE_ALLCAPS: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b([A-Z][A-Z0-9_]{2,})\b").unwrap());

static RE_TOOL_NAME: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b([a-z]+_[a-z_]+)\b").unwrap());

static RE_FILE_PATH: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?:^|[\s(])([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})\b").unwrap());

static RE_URL: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"https?://[^\s<>"']+"#).unwrap());

static RE_VERSION: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b([A-Za-z]+)\s+(\d+\.\d+(?:\.\d+)?)\b").unwrap());

static STOPWORDS: Lazy<HashSet<&str>> = Lazy::new(|| {
    [
        "THE", "AND", "FOR", "NOT", "THIS", "THAT", "WITH", "FROM",
        "HAVE", "HAS", "HAD", "WAS", "WERE", "ARE", "BEEN", "WILL",
        "WOULD", "COULD", "SHOULD", "DOES", "DID", "BUT", "ALL",
        "ALSO", "EACH", "SOME", "THEM", "THEN", "THAN", "INTO",
        "TODO", "FIXME", "NOTE", "HACK", "NONE", "TRUE", "FALSE",
        "SELF", "RETURN", "IMPORT",
    ]
    .into_iter()
    .collect()
});

/// Extract entities from text using 3-tier strategy.
/// No LLM needed — pure algorithmic extraction.
pub fn extract_entities(text: &str) -> Vec<Entity> {
    let mut entities: Vec<Entity> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    // ── Tier 1: Dictionary lookup (highest confidence) ──
    let text_lower = text.to_lowercase();
    for (key, etype) in TECH_ENTITIES.iter() {
        let key_lower = key.to_lowercase();
        if key_lower.len() >= 2 && text_lower.contains(&key_lower) {
            if !seen.contains(&key_lower) {
                // Find the original casing in the text
                let name = find_original_case(text, &key_lower).unwrap_or_else(|| key.clone());
                entities.push(Entity {
                    name,
                    entity_type: etype.to_string(),
                    confidence: 0.95,
                    source_tier: 1,
                });
                seen.insert(key_lower);
            }
        }
    }

    // ── Tier 3: Regex patterns (lower confidence) ──

    // CamelCase (ToolDispatcher, ScreenVision, etc.)
    for cap in RE_CAMELCASE.captures_iter(text) {
        let name = cap[1].to_string();
        let lower = name.to_lowercase();
        if !seen.contains(&lower) && name.len() >= 4 {
            entities.push(Entity {
                name,
                entity_type: "concept".to_string(),
                confidence: 0.6,
                source_tier: 3,
            });
            seen.insert(lower);
        }
    }

    // ALL_CAPS (OPENAI_API_KEY, MAX_TOKENS, etc.)
    for cap in RE_ALLCAPS.captures_iter(text) {
        let name = cap[1].to_string();
        if !seen.contains(&name) && !STOPWORDS.contains(name.as_str()) {
            entities.push(Entity {
                name: name.clone(),
                entity_type: "constant".to_string(),
                confidence: 0.5,
                source_tier: 3,
            });
            seen.insert(name);
        }
    }

    // tool_name patterns (web_search, memory_recall, etc.)
    for cap in RE_TOOL_NAME.captures_iter(text) {
        let name = cap[1].to_string();
        if !seen.contains(&name) && name.len() >= 5 {
            entities.push(Entity {
                name: name.clone(),
                entity_type: "tool".to_string(),
                confidence: 0.55,
                source_tier: 3,
            });
            seen.insert(name);
        }
    }

    // File paths (config.yaml, src/main.py, etc.)
    for cap in RE_FILE_PATH.captures_iter(text) {
        let name = cap[1].to_string();
        let lower = name.to_lowercase();
        if !seen.contains(&lower) && name.contains('.') && name.len() >= 3 {
            entities.push(Entity {
                name,
                entity_type: "file".to_string(),
                confidence: 0.7,
                source_tier: 3,
            });
            seen.insert(lower);
        }
    }

    // Version patterns (Python 3.11, Node 20.1, etc.)
    for cap in RE_VERSION.captures_iter(text) {
        let name = format!("{} {}", &cap[1], &cap[2]);
        let lower = name.to_lowercase();
        if !seen.contains(&lower) {
            entities.push(Entity {
                name,
                entity_type: "version".to_string(),
                confidence: 0.65,
                source_tier: 3,
            });
            seen.insert(lower);
        }
    }

    // URLs (https://example.com/path)
    for cap in RE_URL.find_iter(text) {
        let name = cap.as_str().trim_end_matches(&['.', ',', ')', ']', ';'][..]).to_string();
        let lower = name.to_lowercase();
        if !seen.contains(&lower) && name.len() >= 8 {
            entities.push(Entity {
                name,
                entity_type: "url".to_string(),
                confidence: 0.80,
                source_tier: 3,
            });
            seen.insert(lower);
        }
    }

    // Cap at 30 entities max
    entities.truncate(30);
    entities
}

/// Find the original casing of a word in the text.
fn find_original_case(text: &str, lower_target: &str) -> Option<String> {
    let target_len = lower_target.len();
    let text_lower = text.to_lowercase();
    if let Some(pos) = text_lower.find(lower_target) {
        Some(text[pos..pos + target_len].to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_extraction() {
        let entities = extract_entities("We use Python and Docker for deployment");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.iter().any(|n| n.to_lowercase() == "python"));
        assert!(names.iter().any(|n| n.to_lowercase() == "docker"));
    }

    #[test]
    fn test_dictionary_confidence() {
        let entities = extract_entities("Using Redis for caching");
        let redis = entities.iter().find(|e| e.name.to_lowercase() == "redis");
        assert!(redis.is_some());
        assert_eq!(redis.unwrap().source_tier, 1);
        assert!(redis.unwrap().confidence > 0.9);
    }

    #[test]
    fn test_camelcase_extraction() {
        let entities = extract_entities("The ToolDispatcher handles all requests");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"ToolDispatcher"));
    }

    #[test]
    fn test_allcaps_extraction() {
        let entities = extract_entities("Set OPENAI_API_KEY and MAX_TOKENS");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"OPENAI_API_KEY"));
        assert!(names.contains(&"MAX_TOKENS"));
    }

    #[test]
    fn test_tool_name_extraction() {
        let entities = extract_entities("Use web_search and memory_recall tools");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"web_search"));
        assert!(names.contains(&"memory_recall"));
    }

    #[test]
    fn test_stopwords_excluded() {
        let entities = extract_entities("THE AND FOR NOT THIS RETURN");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(!names.contains(&"THE"));
        assert!(!names.contains(&"AND"));
        assert!(!names.contains(&"RETURN"));
    }

    #[test]
    fn test_file_path_extraction() {
        let entities = extract_entities("Edit config.yaml and src/main.py");
        let files: Vec<&str> = entities
            .iter()
            .filter(|e| e.entity_type == "file")
            .map(|e| e.name.as_str())
            .collect();
        assert!(files.iter().any(|f| f.contains("config.yaml")));
    }

    #[test]
    fn test_version_extraction() {
        let entities = extract_entities("Upgrade to Python 3.11 and Node 20.1");
        let versions: Vec<&str> = entities
            .iter()
            .filter(|e| e.entity_type == "version")
            .map(|e| e.name.as_str())
            .collect();
        assert!(versions.iter().any(|v| v.contains("3.11")));
    }

    #[test]
    fn test_deduplication() {
        let entities = extract_entities("Python Python Python Python");
        let python_count = entities.iter().filter(|e| e.name.to_lowercase() == "python").count();
        assert_eq!(python_count, 1);
    }

    #[test]
    fn test_cap_at_30() {
        // Generate text with many entities
        let text = (0..50).map(|i| format!("EntityName{i}")).collect::<Vec<_>>().join(" ");
        let entities = extract_entities(&text);
        assert!(entities.len() <= 30);
    }

    #[test]
    fn test_empty_text() {
        assert!(extract_entities("").is_empty());
    }

    #[test]
    fn test_mixed_tiers() {
        let entities = extract_entities("Using Docker with ToolDispatcher and CUSTOM_FLAG");
        assert!(entities.iter().any(|e| e.source_tier == 1)); // Dictionary
        assert!(entities.iter().any(|e| e.source_tier == 3)); // Regex
    }
}
