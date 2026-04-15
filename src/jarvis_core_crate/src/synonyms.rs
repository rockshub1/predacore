/// Tech domain synonym expansion for enhanced BM25 search.
///
/// Expands query terms with related words so "vehicle" also matches "car",
/// and "config" also matches "settings", "yaml", etc.

use once_cell::sync::Lazy;
use std::collections::HashMap;

/// Built-in tech domain synonym groups.
/// Each group contains terms that should be treated as equivalent in search.
static SYNONYM_MAP: Lazy<HashMap<String, Vec<String>>> = Lazy::new(|| {
    let groups: Vec<Vec<&str>> = vec![
        // Authentication & Security
        vec!["auth", "authentication", "authorization", "login", "signin", "oauth", "sso"],
        vec!["security", "secure", "protection", "safety", "firewall"],
        vec!["password", "credential", "secret", "token", "apikey"],
        vec!["permission", "access", "role", "privilege", "acl"],
        vec!["encrypt", "encryption", "cipher", "cryptography", "fernet"],
        // Configuration
        vec!["config", "configuration", "settings", "preferences", "options"],
        vec!["yaml", "yml", "toml", "json", "dotenv", "env"],
        vec!["parameter", "param", "argument", "arg", "flag", "option"],
        // Errors & Debugging
        vec!["error", "exception", "failure", "fault", "crash", "panic", "bug"],
        vec!["debug", "debugging", "troubleshoot", "diagnose", "inspect"],
        vec!["log", "logging", "logger", "trace", "tracing"],
        vec!["warning", "warn", "caution", "alert"],
        vec!["fix", "patch", "repair", "resolve", "hotfix"],
        // Database & Storage
        vec!["database", "db", "sqlite", "postgres", "mysql", "sql", "datastore"],
        vec!["store", "storage", "persist", "persistence", "save", "write"],
        vec!["query", "select", "fetch", "retrieve", "lookup", "find"],
        vec!["cache", "caching", "cached", "memoize", "ttl"],
        vec!["index", "indexing", "indexed"],
        // Network & API
        vec!["api", "endpoint", "route", "handler", "controller"],
        vec!["request", "req", "http", "https", "fetch", "call"],
        vec!["response", "res", "reply", "result", "output"],
        vec!["server", "service", "daemon", "backend", "microservice"],
        vec!["client", "consumer", "frontend", "caller"],
        vec!["websocket", "ws", "socket", "realtime"],
        vec!["url", "uri", "link", "href", "path"],
        // Code Structure
        vec!["function", "func", "fn", "method", "def", "procedure"],
        vec!["class", "struct", "type", "model", "schema", "interface"],
        vec!["variable", "var", "field", "attribute", "property", "prop"],
        vec!["module", "package", "crate", "library", "lib", "dependency"],
        vec!["import", "require", "include", "use", "dependency"],
        vec!["test", "testing", "spec", "unittest", "pytest", "jest"],
        vec!["async", "asynchronous", "await", "concurrent", "parallel"],
        // Files & Paths
        vec!["file", "document", "doc", "artifact"],
        vec!["directory", "dir", "folder", "path"],
        vec!["create", "new", "generate", "init", "initialize", "setup"],
        vec!["delete", "remove", "drop", "destroy", "cleanup"],
        vec!["update", "modify", "change", "edit", "patch", "alter"],
        // Tools & Infra
        vec!["docker", "container", "containerize", "image"],
        vec!["git", "version", "commit", "branch", "merge", "repo"],
        vec!["deploy", "deployment", "release", "ship", "publish"],
        vec!["ci", "cd", "pipeline", "workflow", "github_actions"],
        vec!["monitor", "monitoring", "observability", "metrics", "prometheus"],
        // AI & ML
        vec!["embedding", "vector", "encode", "representation"],
        vec!["model", "llm", "neural", "ai", "ml"],
        vec!["prompt", "template", "instruction", "system_prompt"],
        vec!["memory", "context", "history", "recall"],
        vec!["agent", "assistant", "bot", "chatbot"],
        // Common Actions
        vec!["start", "begin", "launch", "run", "execute"],
        vec!["stop", "halt", "shutdown", "kill", "terminate"],
        vec!["send", "emit", "dispatch", "publish", "broadcast"],
        vec!["receive", "listen", "subscribe", "consume", "handle"],
    ];

    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    for group in &groups {
        for term in group {
            let synonyms: Vec<String> = group
                .iter()
                .filter(|t| *t != term)
                .map(|t| t.to_string())
                .collect();
            map.insert(term.to_string(), synonyms);
        }
    }
    map
});

/// Expand a list of query terms with their synonyms.
/// Returns the original terms plus all synonyms (deduplicated).
pub fn expand_synonyms(terms: &[String]) -> Vec<String> {
    let mut expanded: Vec<String> = terms.to_vec();
    for term in terms {
        let lower = term.to_lowercase();
        if let Some(synonyms) = SYNONYM_MAP.get(&lower) {
            for syn in synonyms {
                if !expanded.contains(syn) {
                    expanded.push(syn.clone());
                }
            }
        }
    }
    expanded
}

/// Get synonyms for a single term.
pub fn get_synonyms(term: &str) -> Vec<String> {
    SYNONYM_MAP
        .get(&term.to_lowercase())
        .cloned()
        .unwrap_or_default()
}

/// Check if two terms are synonyms of each other.
pub fn are_synonyms(a: &str, b: &str) -> bool {
    let lower_a = a.to_lowercase();
    let lower_b = b.to_lowercase();
    if lower_a == lower_b {
        return true;
    }
    SYNONYM_MAP
        .get(&lower_a)
        .map(|syns| syns.contains(&lower_b))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_basic() {
        let terms = vec!["config".to_string()];
        let expanded = expand_synonyms(&terms);
        assert!(expanded.contains(&"config".to_string()));
        assert!(expanded.contains(&"configuration".to_string()));
        assert!(expanded.contains(&"settings".to_string()));
    }

    #[test]
    fn test_expand_preserves_original() {
        let terms = vec!["unknownterm123".to_string()];
        let expanded = expand_synonyms(&terms);
        assert_eq!(expanded, terms); // Unknown term, no expansion
    }

    #[test]
    fn test_expand_multiple_terms() {
        let terms = vec!["auth".to_string(), "error".to_string()];
        let expanded = expand_synonyms(&terms);
        assert!(expanded.contains(&"authentication".to_string()));
        assert!(expanded.contains(&"exception".to_string()));
    }

    #[test]
    fn test_get_synonyms() {
        let syns = get_synonyms("database");
        assert!(syns.contains(&"db".to_string()));
        assert!(syns.contains(&"sqlite".to_string()));
    }

    #[test]
    fn test_get_synonyms_unknown() {
        assert!(get_synonyms("xyzzy12345").is_empty());
    }

    #[test]
    fn test_are_synonyms() {
        assert!(are_synonyms("auth", "authentication"));
        assert!(are_synonyms("config", "settings"));
        assert!(are_synonyms("error", "bug"));
        assert!(!are_synonyms("auth", "banana"));
    }

    #[test]
    fn test_are_synonyms_case_insensitive() {
        assert!(are_synonyms("Auth", "AUTHENTICATION"));
    }

    #[test]
    fn test_are_synonyms_same_word() {
        assert!(are_synonyms("config", "config"));
    }

    #[test]
    fn test_synonym_groups_bidirectional() {
        // If A is synonym of B, then B is synonym of A
        assert!(are_synonyms("docker", "container"));
        assert!(are_synonyms("container", "docker"));
    }
}
