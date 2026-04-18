"""
Comprehensive tests for predacore.prompts — persona drift detection,
prompt assembly, identity file loading, and regex patterns.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from predacore.prompts import (
    DIRECT_CAPABILITY_DENIAL_RE,
    DIRECT_TOOL_PREFIX_RE,
    PERSONA_DRIFT_PATTERNS,
    PERSONA_IDENTITY_QUERY_RE,
    TIMEOUT_HINT_RE,
    UNVERIFIED_ACTION_CLAIM_RE,
    UNVERIFIED_MODEL_SWITCH_RE,
    VERIFICATION_REQUEST_RE,
    PersonaDriftAssessment,
)
from predacore.identity.engine import _read_cached as _read_file_cached


# ── PersonaDriftAssessment ─────────────────────────────────────────


class TestPersonaDriftAssessment:
    """Tests for the drift assessment dataclass."""

    def test_no_regeneration_below_threshold(self):
        a = PersonaDriftAssessment(score=0.2, threshold=0.42)
        assert a.needs_regeneration is False

    def test_regeneration_at_threshold(self):
        a = PersonaDriftAssessment(score=0.42, threshold=0.42)
        assert a.needs_regeneration is True

    def test_regeneration_above_threshold(self):
        a = PersonaDriftAssessment(score=0.8, threshold=0.42)
        assert a.needs_regeneration is True

    def test_reasons_stored(self):
        a = PersonaDriftAssessment(
            score=0.5, threshold=0.42, reasons=["generic_model_identity"]
        )
        assert "generic_model_identity" in a.reasons

    def test_empty_reasons(self):
        a = PersonaDriftAssessment(score=0.0, threshold=0.42)
        assert a.reasons == []


# ── Persona Drift Patterns ─────────────────────────────────────────


class TestPersonaDriftPatterns:
    """Tests for all persona drift detection regex patterns."""

    def test_generic_model_identity(self):
        pattern, weight, name = PERSONA_DRIFT_PATTERNS[0]
        # Patterns are case-sensitive (match lowercase LLM output)
        assert pattern.search("as an ai model, i cannot")
        assert pattern.search("as an ai language model")
        assert not pattern.search("PredaCore here, ready to help")

    def test_foreign_identity_openai(self):
        pattern, weight, name = PERSONA_DRIFT_PATTERNS[1]
        assert pattern.search("i am chatgpt and i can help")
        assert pattern.search("i'm gpt-4")
        assert not pattern.search("i am predacore")

    def test_foreign_identity_anthropic(self):
        pattern, weight, name = PERSONA_DRIFT_PATTERNS[2]
        assert pattern.search("i am claude, made by anthropic")
        assert pattern.search("i'm anthropic")
        assert not pattern.search("i use claude as my backend")

    def test_foreign_identity_gemini(self):
        pattern, weight, name = PERSONA_DRIFT_PATTERNS[3]
        assert pattern.search("i am gemini")
        assert pattern.search("i'm google ai")
        assert not pattern.search("i use gemini for inference")

    def test_capability_denial(self):
        pattern, weight, name = PERSONA_DRIFT_PATTERNS[4]
        assert pattern.search("i cannot access the filesystem")
        assert pattern.search("i can't run commands")
        assert not pattern.search("i ran the command successfully")

    def test_weights_are_positive(self):
        for _, weight, _ in PERSONA_DRIFT_PATTERNS:
            assert weight > 0


# ── Verification & Action Patterns ─────────────────────────────────


class TestVerificationPatterns:
    """Tests for verification and action claim patterns."""

    def test_verification_request(self):
        assert VERIFICATION_REQUEST_RE.search("check the status")
        assert VERIFICATION_REQUEST_RE.search("what model are you using")
        assert VERIFICATION_REQUEST_RE.search("run a command to verify")

    def test_unverified_action_claim(self):
        # Case-sensitive — matches lowercase "i"
        assert UNVERIFIED_ACTION_CLAIM_RE.search("i have checked the file")
        assert UNVERIFIED_ACTION_CLAIM_RE.search("i verified the output")
        assert UNVERIFIED_ACTION_CLAIM_RE.search("i ran the command")

    def test_unverified_model_switch(self):
        assert UNVERIFIED_MODEL_SWITCH_RE.search("i have switched the model")
        assert UNVERIFIED_MODEL_SWITCH_RE.search("i updated the provider")

    def test_direct_capability_denial(self):
        assert DIRECT_CAPABILITY_DENIAL_RE.search("i cannot access the shell")
        assert DIRECT_CAPABILITY_DENIAL_RE.search("i don't have direct shell access")
        assert not DIRECT_CAPABILITY_DENIAL_RE.search("i ran the shell command")

    def test_persona_identity_query(self):
        assert PERSONA_IDENTITY_QUERY_RE.search("who are you?")
        assert PERSONA_IDENTITY_QUERY_RE.search("what are you")
        assert PERSONA_IDENTITY_QUERY_RE.search("are you predacore")
        assert PERSONA_IDENTITY_QUERY_RE.search("do you remember me")


# ── Tool Prefix & Timeout Patterns ─────────────────────────────────


class TestToolPatterns:
    """Tests for direct tool prefix and timeout hint patterns."""

    def test_direct_tool_prefix(self):
        m = DIRECT_TOOL_PREFIX_RE.match("run web_search query=test")
        assert m
        assert m.group(1) == "web_search"

    def test_direct_tool_prefix_execute(self):
        m = DIRECT_TOOL_PREFIX_RE.match("execute memory_recall")
        assert m
        assert m.group(1) == "memory_recall"

    def test_direct_tool_prefix_case_insensitive(self):
        m = DIRECT_TOOL_PREFIX_RE.match("Run Web_Search")
        assert m

    def test_direct_tool_no_match(self):
        assert not DIRECT_TOOL_PREFIX_RE.match("please search for something")

    def test_timeout_hint(self):
        m = TIMEOUT_HINT_RE.search("timeout=30")
        assert m
        assert m.group(1) == "30"

    def test_timeout_hint_colon(self):
        m = TIMEOUT_HINT_RE.search("timeout: 60")
        assert m
        assert m.group(1) == "60"

    def test_timeout_hint_float(self):
        m = TIMEOUT_HINT_RE.search("timeout=5.5")
        assert m
        assert m.group(1) == "5.5"

    def test_timeout_seconds_variant(self):
        m = TIMEOUT_HINT_RE.search("timeout_seconds=120")
        assert m
        assert m.group(1) == "120"


# ── File Caching ───────────────────────────────────────────────────


class TestFileCache:
    """Tests for mtime-based file content caching."""

    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("hello world")
        content = _read_file_cached(f)
        assert content == "hello world"

    def test_read_nonexistent(self, tmp_path):
        f = tmp_path / "missing.md"
        assert _read_file_cached(f) is None

    def test_caching_returns_same(self, tmp_path):
        f = tmp_path / "cached.md"
        f.write_text("content")
        c1 = _read_file_cached(f)
        c2 = _read_file_cached(f)
        assert c1 == c2 == "content"

    def test_strips_whitespace(self, tmp_path):
        f = tmp_path / "ws.md"
        f.write_text("  content with spaces  \n\n")
        content = _read_file_cached(f)
        assert content == "content with spaces"
