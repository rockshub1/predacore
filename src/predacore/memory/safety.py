"""
Ingress safety for Predacore Memory.

Two guards applied BEFORE any content reaches the vector index:

1. Secret scanner — blocks content that contains AWS / GitHub / OpenAI /
   Anthropic / private-key / SSH private key / JWT patterns. A high-
   entropy generic scan covers the long tail.

2. `.memoryignore` — per-project file listing glob patterns that must
   never be indexed. Same syntax as .gitignore (with anchored and
   unanchored matching). A `.gitignore` file in the same directory
   is ALSO honored if present, so generated/build artifacts stay out
   of memory by default.

Design rules:
  * Never raise to the caller for "this content has a secret" — we
    return a decision object. The caller increments a counter and
    silently skips.
  * Regex passes are cheap — all patterns compiled once at import.
  * `.memoryignore` is walked once per path and cached for 30s; tearing
    the cache down is free since it lives on the Store instance.
"""
from __future__ import annotations

import fnmatch
import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Secret patterns
# ─────────────────────────────────────────────────────────────

# Each pattern is (name, compiled_regex). Names double as stat keys.
# Rules kept conservative — false positives waste nothing except one
# row we refused to store; a false negative lets a secret into memory.
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # AWS access key ID
    ("aws_access_key_id", re.compile(r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}(?![A-Za-z0-9])")),
    # AWS secret access key — looks like 40 base64 chars, only used alongside AK
    # (we match the "aws_secret_access_key = XXX" form to reduce noise)
    ("aws_secret_access_key", re.compile(
        r"(?i)aws[_\-\s]?secret[_\-\s]?access[_\-\s]?key[\"'\s]*[:=][\"'\s]*[A-Za-z0-9/+=]{40}"
    )),
    # GitHub personal access tokens + app tokens
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b")),
    # OpenAI keys
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    # Anthropic keys
    ("anthropic_key", re.compile(r"\bsk-ant-(?:api|adm)[0-9]{2}-[A-Za-z0-9_-]{80,}\b")),
    # Google API keys (e.g. AIzaSy...)
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    # Slack tokens
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    # Stripe keys
    ("stripe_key", re.compile(r"\b(?:sk|pk|rk)_(?:test|live)_[A-Za-z0-9]{16,}\b")),
    # Generic private key block (PEM)
    ("pem_private_key", re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED |PGP )?PRIVATE KEY-----"
    )),
    # SSH private key header (covers openssh format explicitly)
    ("ssh_private_key", re.compile(r"-----BEGIN OPENSSH PRIVATE KEY-----")),
    # JWT — three base64 segments separated by dots. High false-positive
    # rate, so require the "eyJ" header pattern that typical JWTs have.
    ("jwt_token", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    # Generic 'password' / 'secret' assignment (best-effort — noisy by nature)
    ("generic_secret_assignment", re.compile(
        r"(?i)(?:password|passwd|secret|api[_\-]?key|auth[_\-]?token)"
        r"[\s]*[:=][\s]*[\"']?[A-Za-z0-9_\-!@#$%^&*+/=]{16,}[\"']?"
    )),
]


# ─────────────────────────────────────────────────────────────
# Entropy heuristic (catches long, high-entropy tokens pattern-
# scanners miss). Opt-in via scan_for_secrets(include_entropy=True).
# ─────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[A-Za-z0-9+/=_\-]{30,}")


def _shannon_entropy(s: str) -> float:
    """Shannon entropy per char (bits). 5+ on a 30+ char string is
    suspicious; random base64 keys typically score 4.5-5.5."""
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────


@dataclass
class SecretMatch:
    name: str           # pattern name (e.g. "github_token")
    position: int       # character offset in the input
    length: int         # match length
    preview: str        # redacted preview (first/last 3 chars + mask)

    def as_dict(self) -> dict:
        return {
            "kind": self.name,
            "position": self.position,
            "length": self.length,
            "preview": self.preview,
        }


def _redact(s: str) -> str:
    if len(s) <= 6:
        return "*" * len(s)
    return f"{s[:3]}{'*' * (len(s) - 6)}{s[-3:]}"


def scan_for_secrets(text: str, *, include_entropy: bool = False) -> list[SecretMatch]:
    """Return a list of SecretMatch for every secret-looking thing in `text`.
    Empty list = safe to store.

    `include_entropy=True` adds a shannon-entropy pass that catches
    long high-entropy tokens the pattern scanner misses. Useful for
    paranoia mode; default off because it's the biggest source of
    false positives.
    """
    if not text:
        return []
    matches: list[SecretMatch] = []
    for name, pattern in _SECRET_PATTERNS:
        for m in pattern.finditer(text):
            matches.append(SecretMatch(
                name=name,
                position=m.start(),
                length=len(m.group(0)),
                preview=_redact(m.group(0)),
            ))
    if include_entropy:
        for m in _WORD_RE.finditer(text):
            token = m.group(0)
            if _shannon_entropy(token) >= 4.5:
                matches.append(SecretMatch(
                    name="high_entropy_token",
                    position=m.start(),
                    length=len(token),
                    preview=_redact(token),
                ))
    return matches


def is_sensitive_path(path: str | Path) -> bool:
    """Quick heuristic path check — refuses .env*, *.pem, *.key, ssh-ish
    names even if the content looks innocuous. Paranoia wins here."""
    p = Path(path)
    name = p.name.lower()
    if name.startswith(".env") or name == ".npmrc":
        return True
    suffixes = {".pem", ".key", ".p12", ".pfx", ".asc", ".gpg"}
    if p.suffix.lower() in suffixes:
        return True
    if "id_rsa" in name or "id_ed25519" in name or "id_ecdsa" in name:
        return True
    return False


# ─────────────────────────────────────────────────────────────
# .memoryignore
# ─────────────────────────────────────────────────────────────


class MemoryIgnore:
    """
    .gitignore-style matcher for memory indexing.

    Matcher is relative-path based — feed it paths resolved against
    a repo root. Lines beginning with `#` or blank are ignored. A
    pattern starting with `/` is anchored to the root; otherwise it
    matches anywhere in the tree. A trailing `/` restricts to directories.
    Patterns starting with `!` are negations (un-ignore).

    Reads BOTH `.memoryignore` (primary) and `.gitignore` (fallback) so
    users don't have to maintain two files for basic build-output exclusion.
    """

    def __init__(self, patterns: Iterable[str] | None = None) -> None:
        self.includes: list[str] = []    # positive patterns
        self.excludes: list[str] = []    # negations (!pattern)
        self.dir_only: set[str] = set()  # patterns with trailing /
        if patterns:
            for raw in patterns:
                self._add(raw)

    def _add(self, raw: str) -> None:
        line = raw.rstrip()
        if not line or line.startswith("#"):
            return
        negate = line.startswith("!")
        if negate:
            line = line[1:]
        # We keep patterns with their trailing "/" intact; the matcher
        # decides what directory-only semantics mean.
        if negate:
            self.excludes.append(line)
        else:
            self.includes.append(line)

    @classmethod
    def for_root(cls, root: str | Path) -> "MemoryIgnore":
        """Load patterns from `<root>/.memoryignore` and `<root>/.gitignore`.
        Missing files are tolerated — a fresh repo just has no ignore rules."""
        root_path = Path(root)
        patterns: list[str] = []
        for name in (".memoryignore", ".gitignore"):
            f = root_path / name
            if not f.exists():
                continue
            try:
                patterns.extend(f.read_text(errors="replace").splitlines())
            except OSError:
                continue
        return cls(patterns)

    def matches(self, rel_path: str) -> bool:
        """True if this path should be excluded from memory indexing.

        `rel_path` is expected to be relative to the root the matcher
        was loaded for, with POSIX-style separators.
        """
        rel_path = rel_path.lstrip("/")
        hit = False
        for pat in self.includes:
            if _pattern_matches(pat, rel_path):
                hit = True
                break
        if not hit:
            return False
        for pat in self.excludes:
            if _pattern_matches(pat, rel_path):
                return False
        return True


def _pattern_matches(pattern: str, path: str) -> bool:
    """Evaluate one gitignore-style pattern against a relative path.

    Rules (simplified gitignore):
      - Leading "/" anchors the pattern to the root.
      - Trailing "/" means directory-only — any path WHERE pattern is
        a directory prefix also matches. `build/` matches `build/foo`.
      - No "/" in pattern → match against any single segment (so
        `*.log` matches `deploy.log` AND `deep/logs/x.log`).
      - Multi-segment unanchored pattern (e.g. `src/foo`) matches as
        a subtree anywhere.
    """
    anchored = pattern.startswith("/")
    if anchored:
        pattern = pattern.lstrip("/")
    dir_only = pattern.endswith("/")
    if dir_only:
        pattern = pattern.rstrip("/")
    if not pattern:
        return False

    segments = path.split("/")

    if anchored:
        if dir_only:
            # "target/" anchored → match path if it's inside target/
            return path == pattern or path.startswith(pattern + "/")
        # Anchored exact glob — also match as directory prefix for convenience
        if fnmatch.fnmatch(path, pattern):
            return True
        return path.startswith(pattern + "/")

    # Unanchored
    if "/" not in pattern:
        if dir_only:
            # "build/" unanchored → any segment matches `build`; path must have
            # something inside that directory.
            for i, seg in enumerate(segments[:-1]):
                if fnmatch.fnmatch(seg, pattern):
                    return True
            return False
        # Simple glob like `*.log` — match any segment
        return any(fnmatch.fnmatch(seg, pattern) for seg in segments)

    # Multi-segment unanchored like `build/output` — match as subtree anywhere
    for i in range(len(segments)):
        tail = "/".join(segments[i:])
        if fnmatch.fnmatch(tail, pattern):
            return True
        if tail.startswith(pattern + "/"):
            return True
    return False


# ─────────────────────────────────────────────────────────────
# SafetyStats — counters the store carries around
# ─────────────────────────────────────────────────────────────


@dataclass
class SafetyStats:
    secrets_blocked:      int = 0            # total store() calls refused
    by_kind:              Counter = field(default_factory=Counter)
    ignored_paths:        int = 0            # files skipped by memoryignore
    sensitive_paths_skipped: int = 0         # files skipped by path heuristic

    def record_block(self, matches: list[SecretMatch]) -> None:
        self.secrets_blocked += 1
        for m in matches:
            self.by_kind[m.name] += 1

    def as_dict(self) -> dict:
        return {
            "secrets_blocked": self.secrets_blocked,
            "by_kind": dict(self.by_kind),
            "ignored_paths": self.ignored_paths,
            "sensitive_paths_skipped": self.sensitive_paths_skipped,
        }
