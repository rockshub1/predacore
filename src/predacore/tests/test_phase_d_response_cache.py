"""
Phase D — Local LLM response idempotency cache tests.

Cache is opt-in via ``PREDACORE_IDEMPOTENT=1``, SQLite-backed, and keyed by
``sha256(provider, model, messages, tools, temperature)``. Works for any
provider because it sits at the router layer (``LLMInterface.chat``) — the
provider adapters don't know or care.
"""
from __future__ import annotations

import os
import time

import pytest

from predacore.llm_providers.response_cache import (
    ResponseCache,
    _compute_key,
    is_enabled,
)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Isolated cache per test — no cross-test pollution."""
    monkeypatch.setenv("PREDACORE_IDEMPOTENT", "1")
    db = tmp_path / "responses.db"
    return ResponseCache(db_path=db, ttl_seconds=3600)


# ---------------------------------------------------------------------------
# is_enabled gate
# ---------------------------------------------------------------------------


class TestEnabledGate:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("PREDACORE_IDEMPOTENT", raising=False)
        assert is_enabled() is False

    def test_enabled_via_env(self, monkeypatch):
        monkeypatch.setenv("PREDACORE_IDEMPOTENT", "1")
        assert is_enabled() is True

    def test_various_truthy_values(self, monkeypatch):
        for v in ["1", "true", "yes", "on", "TRUE"]:
            monkeypatch.setenv("PREDACORE_IDEMPOTENT", v)
            assert is_enabled() is True, f"value {v!r} should enable"

    def test_falsy_values_disable(self, monkeypatch):
        for v in ["0", "false", "no", "", "off"]:
            monkeypatch.setenv("PREDACORE_IDEMPOTENT", v)
            assert is_enabled() is False, f"value {v!r} should disable"


# ---------------------------------------------------------------------------
# Key computation
# ---------------------------------------------------------------------------


class TestKeyStability:
    def test_same_inputs_same_key(self):
        k1 = _compute_key("openai", "gpt-4o", [{"role": "u", "content": "hi"}], None, 0.0)
        k2 = _compute_key("openai", "gpt-4o", [{"role": "u", "content": "hi"}], None, 0.0)
        assert k1 == k2 and len(k1) == 64

    def test_different_provider_different_key(self):
        k1 = _compute_key("openai", "gpt-4o", [{"role": "u", "content": "hi"}], None, 0.0)
        k2 = _compute_key("anthropic", "gpt-4o", [{"role": "u", "content": "hi"}], None, 0.0)
        assert k1 != k2

    def test_different_model_different_key(self):
        k1 = _compute_key("openai", "gpt-4o", [{"role": "u", "content": "hi"}], None, 0.0)
        k2 = _compute_key("openai", "gpt-5", [{"role": "u", "content": "hi"}], None, 0.0)
        assert k1 != k2

    def test_dict_key_order_insensitive(self):
        """json.dumps(sort_keys=True) so message dict key order doesn't break hits."""
        k1 = _compute_key("openai", "gpt-4o", [{"role": "u", "content": "hi"}], None, 0.0)
        k2 = _compute_key("openai", "gpt-4o", [{"content": "hi", "role": "u"}], None, 0.0)
        assert k1 == k2

    def test_non_serializable_returns_empty(self):
        """Non-JSONable content returns empty string — caller bails out."""
        class Obj:
            pass
        # Obj is serialized via default=str, so it's technically serializable.
        # Use something harder — a recursive structure.
        a: list = []
        a.append(a)
        k = _compute_key("openai", "gpt-4o", a, None, 0.0)
        # Recursive structure blows json.dumps. default=str doesn't save it.
        assert k == ""


# ---------------------------------------------------------------------------
# Core cache behaviour
# ---------------------------------------------------------------------------


class TestCacheHitMiss:
    def test_miss_returns_none(self, cache):
        r = cache.get("openai", "gpt-4o", [{"role": "user", "content": "hi"}], None, 0.0)
        assert r is None

    def test_set_then_get_hits(self, cache):
        resp = {"content": "hello there", "tool_calls": [], "usage": {}}
        cache.set("openai", "gpt-4o", [{"role": "user", "content": "hi"}], None, 0.0, resp)
        got = cache.get("openai", "gpt-4o", [{"role": "user", "content": "hi"}], None, 0.0)
        assert got is not None
        assert got["content"] == "hello there"

    def test_different_prompt_is_miss(self, cache):
        resp = {"content": "hi"}
        cache.set("openai", "gpt-4o", [{"role": "user", "content": "A"}], None, 0.0, resp)
        got = cache.get("openai", "gpt-4o", [{"role": "user", "content": "B"}], None, 0.0)
        assert got is None

    def test_different_tools_is_miss(self, cache):
        resp = {"content": "hi"}
        cache.set("openai", "gpt-4o", [{"role": "user", "content": "A"}], [], 0.0, resp)
        got = cache.get(
            "openai", "gpt-4o",
            [{"role": "user", "content": "A"}],
            [{"name": "grep"}],  # different tools
            0.0,
        )
        assert got is None

    def test_works_for_all_providers(self, cache):
        """Core claim: cache is provider-agnostic. Every provider benefits."""
        for provider in ["openai", "anthropic", "gemini", "gemini-cli", "ollama", "groq"]:
            msg = [{"role": "user", "content": f"hi from {provider}"}]
            cache.set(provider, "test-model", msg, None, 0.0, {"content": f"r-{provider}"})
            got = cache.get(provider, "test-model", msg, None, 0.0)
            assert got is not None
            assert got["content"] == f"r-{provider}"

    def test_hit_count_increments(self, cache):
        cache.set("openai", "gpt-4o", [{"role": "u", "content": "a"}], None, 0.0, {"c": 1})
        cache.get("openai", "gpt-4o", [{"role": "u", "content": "a"}], None, 0.0)
        cache.get("openai", "gpt-4o", [{"role": "u", "content": "a"}], None, 0.0)
        cache.get("openai", "gpt-4o", [{"role": "u", "content": "a"}], None, 0.0)
        s = cache.stats()
        assert s["total_hits"] == 3

    def test_replace_updates(self, cache):
        cache.set("openai", "gpt-4o", [{"role": "u", "content": "a"}], None, 0.0, {"c": "v1"})
        cache.set("openai", "gpt-4o", [{"role": "u", "content": "a"}], None, 0.0, {"c": "v2"})
        got = cache.get("openai", "gpt-4o", [{"role": "u", "content": "a"}], None, 0.0)
        assert got["c"] == "v2"


# ---------------------------------------------------------------------------
# Temperature / non-determinism guard
# ---------------------------------------------------------------------------


class TestTemperatureGuard:
    def test_nonzero_temperature_skips_set(self, cache):
        cache.set("openai", "gpt-4o", [{"role": "u", "content": "a"}], None, 0.7, {"c": 1})
        # Even get() with temp=0 returns None — nothing was stored
        got = cache.get("openai", "gpt-4o", [{"role": "u", "content": "a"}], None, 0.0)
        assert got is None

    def test_nonzero_temperature_skips_get(self, cache):
        cache.set("openai", "gpt-4o", [{"role": "u", "content": "a"}], None, 0.0, {"c": 1})
        # Reading at non-zero temp returns None even if entry exists
        got = cache.get("openai", "gpt-4o", [{"role": "u", "content": "a"}], None, 0.5)
        assert got is None


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------


class TestTTL:
    def test_fresh_entry_returned(self, cache):
        cache.set("openai", "gpt-4o", [{"role": "u", "content": "a"}], None, 0.0, {"c": 1})
        assert cache.get("openai", "gpt-4o", [{"role": "u", "content": "a"}], None, 0.0) is not None

    def test_expired_entry_not_returned(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PREDACORE_IDEMPOTENT", "1")
        # TTL 1 second
        c = ResponseCache(db_path=tmp_path / "c.db", ttl_seconds=1)
        c.set("openai", "gpt-4o", [{"role": "u", "content": "a"}], None, 0.0, {"c": 1})
        time.sleep(1.1)
        assert c.get("openai", "gpt-4o", [{"role": "u", "content": "a"}], None, 0.0) is None

    def test_expired_rows_pruned_on_write(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PREDACORE_IDEMPOTENT", "1")
        c = ResponseCache(db_path=tmp_path / "c.db", ttl_seconds=1)
        c.set("openai", "gpt-4o", [{"role": "u", "content": "old"}], None, 0.0, {"c": 1})
        time.sleep(1.1)
        c.set("openai", "gpt-4o", [{"role": "u", "content": "new"}], None, 0.0, {"c": 2})
        s = c.stats()
        # Old row pruned; only the new row remains
        assert s["total_rows"] == 1


# ---------------------------------------------------------------------------
# Disabled-cache guard
# ---------------------------------------------------------------------------


class TestDisabledGuard:
    def test_disabled_cache_get_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PREDACORE_IDEMPOTENT", raising=False)
        c = ResponseCache(db_path=tmp_path / "c.db")
        assert c.get("openai", "gpt-4o", [{"role": "u", "content": "a"}], None, 0.0) is None

    def test_disabled_cache_set_no_op(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PREDACORE_IDEMPOTENT", raising=False)
        c = ResponseCache(db_path=tmp_path / "c.db")
        c.set("openai", "gpt-4o", [{"role": "u", "content": "a"}], None, 0.0, {"c": 1})
        s = c.stats()
        assert s["total_rows"] == 0


# ---------------------------------------------------------------------------
# Stats + clear
# ---------------------------------------------------------------------------


class TestStatsAndClear:
    def test_stats_before_use(self, cache):
        s = cache.stats()
        assert s["enabled"] is True
        assert s["total_rows"] == 0
        assert s["total_hits"] == 0

    def test_stats_after_writes(self, cache):
        cache.set("openai", "gpt-4o", [{"role": "u", "content": "a"}], None, 0.0, {"c": 1})
        cache.set("anthropic", "claude", [{"role": "u", "content": "b"}], None, 0.0, {"c": 2})
        s = cache.stats()
        assert s["total_rows"] == 2
        assert s["by_provider"] == {"openai": 1, "anthropic": 1}

    def test_clear_removes_all(self, cache):
        for i in range(5):
            cache.set("openai", "gpt-4o", [{"role": "u", "content": f"q{i}"}], None, 0.0, {"c": i})
        assert cache.stats()["total_rows"] == 5
        deleted = cache.clear()
        assert deleted == 5
        assert cache.stats()["total_rows"] == 0
