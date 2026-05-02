"""Tests for the browser SelectorCache (T4b).

Eight behavioral guarantees:
  1. domain extraction normalizes urls to bare hosts
  2. intent_hash is stable across calls + role-aware
  3. record() is idempotent (UPSERT, not INSERT-or-fail)
  4. record() bumps success_count on conflict
  5. lookup() returns None for unknown (domain, intent)
  6. invalidate() removes a row
  7. cross-domain isolation (same intent, different domains, no collision)
  8. stats() returns sane counters
"""
from __future__ import annotations

from pathlib import Path

import pytest

from predacore.operators.selector_cache import (
    SelectorCache, SelectorEntry, _normalize_domain, intent_hash,
)


@pytest.fixture
def cache(tmp_path: Path) -> SelectorCache:
    return SelectorCache(str(tmp_path / "browser_cache.db"))


def test_normalize_domain_strips_scheme_and_port():
    assert _normalize_domain("https://twitter.com/home") == "twitter.com"
    assert _normalize_domain("http://Localhost:9222/") == "localhost"
    assert _normalize_domain("twitter.com") == "twitter.com"
    assert _normalize_domain("") == ""


def test_intent_hash_stable_and_role_aware():
    h1 = intent_hash("Login button")
    h2 = intent_hash("login button")  # case
    h3 = intent_hash("  login button  ")  # whitespace
    h4 = intent_hash("login button", role="button")  # role-prefixed
    assert h1 == h2 == h3
    assert h1 != h4
    # Stable across calls
    assert intent_hash("Login button") == h1


def test_lookup_miss_returns_none(cache: SelectorCache):
    assert cache.lookup("example.com", "Login") is None


def test_record_then_lookup_roundtrips(cache: SelectorCache):
    cache.record(
        domain="https://twitter.com/home",
        text="Login", xpath="#login_btn",
        role="button", label="Login",
    )
    entry = cache.lookup("twitter.com", "Login", role="button")
    assert entry is not None
    assert isinstance(entry, SelectorEntry)
    assert entry.xpath == "#login_btn"
    assert entry.role == "button"
    assert entry.label == "Login"
    assert entry.success_count == 1


def test_record_is_idempotent_and_bumps_count(cache: SelectorCache):
    for _ in range(3):
        cache.record(
            domain="twitter.com", text="Login",
            xpath="#login_btn", role="button",
        )
    entry = cache.lookup("twitter.com", "Login", role="button")
    assert entry is not None
    assert entry.success_count == 3


def test_record_replaces_xpath_on_conflict(cache: SelectorCache):
    cache.record(domain="x.com", text="Search", xpath="#search_v1")
    cache.record(domain="x.com", text="Search", xpath="#search_v2")
    entry = cache.lookup("x.com", "Search")
    assert entry is not None
    assert entry.xpath == "#search_v2"


def test_invalidate_removes_row(cache: SelectorCache):
    cache.record(domain="x.com", text="Login", xpath="#a")
    assert cache.invalidate(domain="x.com", text="Login") is True
    assert cache.lookup("x.com", "Login") is None
    # Re-invalidate is a no-op (idempotent)
    assert cache.invalidate(domain="x.com", text="Login") is False


def test_cross_domain_isolation(cache: SelectorCache):
    cache.record(domain="twitter.com", text="Login", xpath="#tw_login")
    cache.record(domain="github.com", text="Login", xpath="#gh_login")
    tw = cache.lookup("twitter.com", "Login")
    gh = cache.lookup("github.com", "Login")
    assert tw is not None and tw.xpath == "#tw_login"
    assert gh is not None and gh.xpath == "#gh_login"


def test_bump_use_increments_only_existing_row(cache: SelectorCache):
    # No-op when row doesn't exist
    cache.bump_use(domain="x.com", text="Login")
    assert cache.lookup("x.com", "Login") is None
    # Increments when row exists
    cache.record(domain="x.com", text="Login", xpath="#a")
    cache.bump_use(domain="x.com", text="Login")
    cache.bump_use(domain="x.com", text="Login")
    entry = cache.lookup("x.com", "Login")
    assert entry is not None
    assert entry.success_count == 3  # 1 from record + 2 bumps


def test_stats_reports_counters(cache: SelectorCache):
    cache.record(domain="a.com", text="Login", xpath="#a", label="Login")
    cache.record(domain="a.com", text="Search", xpath="#s", label="Search")
    cache.record(domain="b.com", text="Login", xpath="#b", label="Login")
    s = cache.stats()
    assert s["total_rows"] == 3
    assert s["distinct_domains"] == 2
    assert len(s["top_intents"]) == 3


def test_blank_inputs_return_none_or_noop(cache: SelectorCache):
    # Empty domain rejected at lookup
    assert cache.lookup("", "Login") is None
    # Empty xpath rejected at record
    cache.record(domain="x.com", text="Login", xpath="")
    assert cache.lookup("x.com", "Login") is None
    # Empty domain rejected at record
    cache.record(domain="", text="Login", xpath="#a")
    assert cache.lookup("", "Login") is None
