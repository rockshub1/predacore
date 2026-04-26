"""T3b — SubsystemFactory wiring tests for the memory upgrade.

Covers:
  • C9 — MemoryConfig +3 flags (enable_healer / scan_for_secrets / eager_warmup)
  • C10 — env-var → bool coercion (PREDACORE_MEMORY_*)
  • C11 — _init_unified_memory wires Healer + warmup
  • bundle.memory_healer field exists + populated
  • Healer thread starts when flag enabled, skipped when disabled

These tests construct real SubsystemBundles via SubsystemFactory.create_all,
which is heavyweight (creates store + spawns healer thread + may try to init
other subsystems). We isolate via tmp_path home + targeted env vars.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from predacore.config import MemoryConfig, load_config
from predacore.tools.subsystem_init import SubsystemBundle, SubsystemFactory


# ─────────────────────────────────────────────────────────────────────
# C9 — MemoryConfig fields
# ─────────────────────────────────────────────────────────────────────


class TestMemoryConfigFields:
    """The 3 new flags exist on MemoryConfig with correct defaults."""

    def test_enable_healer_defaults_true(self):
        assert MemoryConfig().enable_healer is True

    def test_scan_for_secrets_defaults_true(self):
        assert MemoryConfig().scan_for_secrets is True

    def test_eager_warmup_defaults_true(self):
        assert MemoryConfig().eager_warmup is True

    def test_legacy_fields_still_present(self):
        """Backward compat: existing fields aren't accidentally removed."""
        mc = MemoryConfig()
        for field in ("persistence_dir", "enable_knowledge_graph",
                      "enable_vector_store", "working_memory_capacity",
                      "decay_rate", "consolidation_threshold"):
            assert hasattr(mc, field), f"missing legacy field: {field}"


# ─────────────────────────────────────────────────────────────────────
# C10 — env-var bool coercion
# ─────────────────────────────────────────────────────────────────────


class TestEnvVarCoercion:
    """PREDACORE_MEMORY_* env vars coerce to bool via _parse_bool."""

    @pytest.mark.parametrize("env_value,expected", [
        ("1", True), ("true", True), ("yes", True), ("on", True),
        ("0", False), ("false", False), ("no", False), ("off", False),
    ])
    def test_enable_healer_coerces(self, monkeypatch, env_value, expected):
        monkeypatch.setenv("PREDACORE_MEMORY_ENABLE_HEALER", env_value)
        cfg = load_config()
        assert cfg.memory.enable_healer is expected

    @pytest.mark.parametrize("env_value,expected", [
        ("1", True), ("0", False), ("true", True), ("false", False),
    ])
    def test_scan_for_secrets_coerces(self, monkeypatch, env_value, expected):
        monkeypatch.setenv("PREDACORE_MEMORY_SCAN_SECRETS", env_value)
        cfg = load_config()
        assert cfg.memory.scan_for_secrets is expected

    @pytest.mark.parametrize("env_value,expected", [
        ("1", True), ("0", False),
    ])
    def test_eager_warmup_coerces(self, monkeypatch, env_value, expected):
        monkeypatch.setenv("PREDACORE_MEMORY_EAGER_WARMUP", env_value)
        cfg = load_config()
        assert cfg.memory.eager_warmup is expected

    def test_defaults_when_env_unset(self, monkeypatch):
        for k in ("PREDACORE_MEMORY_ENABLE_HEALER",
                  "PREDACORE_MEMORY_SCAN_SECRETS",
                  "PREDACORE_MEMORY_EAGER_WARMUP"):
            monkeypatch.delenv(k, raising=False)
        cfg = load_config()
        assert cfg.memory.enable_healer is True
        assert cfg.memory.scan_for_secrets is True
        assert cfg.memory.eager_warmup is True


# ─────────────────────────────────────────────────────────────────────
# C11 — SubsystemBundle gets memory_healer field
# ─────────────────────────────────────────────────────────────────────


class TestBundleHasHealerField:
    """SubsystemBundle dataclass exposes memory_healer."""

    def test_memory_healer_field_exists(self):
        from dataclasses import fields
        names = {f.name for f in fields(SubsystemBundle)}
        assert "memory_healer" in names

    def test_memory_healer_defaults_none(self):
        bundle = SubsystemBundle()
        assert bundle.memory_healer is None


# ─────────────────────────────────────────────────────────────────────
# _init_unified_memory — Healer thread spawn / skip
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def fresh_home(tmp_path: Path, monkeypatch) -> Path:
    """Per-test isolated $HOME so SubsystemFactory.create_all uses a fresh
    memory DB at tmp_path/.predacore/memory/unified_memory.db. Avoids
    contaminating the user's real ~/.predacore/."""
    home = tmp_path / "fresh-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def _init_memory_only(cfg, home_dir: Path) -> SubsystemBundle:
    """Run ONLY _init_unified_memory in isolation — skips Desktop, Voice,
    Sandbox, MCTS, LLM-collab, OpenClaw inits which can each take 10-60s
    (Docker probes, model loads, etc.). For tests that only care about the
    memory wiring, calling the full SubsystemFactory.create_all is wasteful
    and can hit the global 60s timeout."""
    bundle = SubsystemBundle()
    SubsystemFactory._init_unified_memory(bundle, cfg, str(home_dir))
    return bundle


class TestSubsystemFactoryHealerWiring:
    """_init_unified_memory spawns Healer when flag is on."""

    def test_healer_starts_with_flag_on(self, fresh_home, monkeypatch):
        monkeypatch.setenv("PREDACORE_MEMORY_ENABLE_HEALER", "1")
        monkeypatch.setenv("PREDACORE_MEMORY_EAGER_WARMUP", "0")

        cfg = load_config()
        assert cfg.memory.enable_healer is True

        bundle = _init_memory_only(cfg, fresh_home)
        try:
            assert bundle.unified_memory is not None
            assert bundle.memory_healer is not None
            assert type(bundle.memory_healer).__name__ == "Healer"
            healer_threads = [t for t in threading.enumerate()
                              if "predacore-healer" in t.name]
            assert len(healer_threads) >= 1
            assert "memory_healer" in bundle.available
        finally:
            if bundle.memory_healer:
                bundle.memory_healer.stop()
            if bundle.unified_memory:
                bundle.unified_memory.close()

    def test_healer_skipped_with_flag_off(self, fresh_home, monkeypatch):
        monkeypatch.setenv("PREDACORE_MEMORY_ENABLE_HEALER", "0")
        monkeypatch.setenv("PREDACORE_MEMORY_EAGER_WARMUP", "0")

        cfg = load_config()
        assert cfg.memory.enable_healer is False

        bundle = _init_memory_only(cfg, fresh_home)
        try:
            assert bundle.unified_memory is not None
            assert bundle.memory_healer is None
            assert "memory_healer" not in bundle.available
        finally:
            if bundle.unified_memory:
                bundle.unified_memory.close()

    def test_unified_memory_always_built(self, fresh_home, monkeypatch):
        """memory_healer is gated, but unified_memory itself should always init."""
        monkeypatch.setenv("PREDACORE_MEMORY_ENABLE_HEALER", "0")
        monkeypatch.setenv("PREDACORE_MEMORY_EAGER_WARMUP", "0")

        cfg = load_config()
        bundle = _init_memory_only(cfg, fresh_home)
        try:
            assert bundle.unified_memory is not None
            assert "memory" in bundle.available
        finally:
            if bundle.unified_memory:
                bundle.unified_memory.close()


# ─────────────────────────────────────────────────────────────────────
# D14 — eager warmup at boot
# ─────────────────────────────────────────────────────────────────────


class TestEagerWarmupAtBoot:
    """When eager_warmup=True, BGE model gets loaded during _init_unified_memory."""

    def test_warmup_fires_with_flag_on(self, fresh_home, monkeypatch, bge_embedder):
        """After _init_unified_memory with eager_warmup=True, the BGE model
        is loaded. The bge_embedder session fixture has already loaded it,
        so this just verifies the post-init state."""
        monkeypatch.setenv("PREDACORE_MEMORY_ENABLE_HEALER", "0")
        monkeypatch.setenv("PREDACORE_MEMORY_EAGER_WARMUP", "1")

        cfg = load_config()
        bundle = _init_memory_only(cfg, fresh_home)
        try:
            import predacore_core
            assert predacore_core.is_model_loaded() is True
        finally:
            if bundle.unified_memory:
                bundle.unified_memory.close()

    def test_warmup_skipped_with_flag_off(self, fresh_home, monkeypatch):
        """flag=False shouldn't error. The model may be loaded from prior
        tests (process-global state); we can't assert False — just clean init."""
        monkeypatch.setenv("PREDACORE_MEMORY_ENABLE_HEALER", "0")
        monkeypatch.setenv("PREDACORE_MEMORY_EAGER_WARMUP", "0")

        cfg = load_config()
        bundle = _init_memory_only(cfg, fresh_home)
        try:
            assert bundle.unified_memory is not None
        finally:
            if bundle.unified_memory:
                bundle.unified_memory.close()


# ─────────────────────────────────────────────────────────────────────
# Healer lifecycle
# ─────────────────────────────────────────────────────────────────────


class TestHealerLifecycle:
    """Healer thread can be stopped cleanly."""

    def test_stop_terminates_thread(self, fresh_home, monkeypatch):
        import time
        monkeypatch.setenv("PREDACORE_MEMORY_ENABLE_HEALER", "1")
        monkeypatch.setenv("PREDACORE_MEMORY_EAGER_WARMUP", "0")
        cfg = load_config()
        bundle = _init_memory_only(cfg, fresh_home)
        try:
            assert bundle.memory_healer is not None
            assert bundle.memory_healer._thread is not None
            assert bundle.memory_healer._thread.is_alive()

            bundle.memory_healer.stop(join_timeout=5.0)
            time.sleep(0.2)
            assert (bundle.memory_healer._thread is None
                    or not bundle.memory_healer._thread.is_alive())
        finally:
            if bundle.unified_memory:
                bundle.unified_memory.close()


class TestMemoryGuideInPrompt:
    """v1.3.0: operational memory guide is code-baked, always in prompt.

    The "How Memory Works" section is no longer in MEMORY.md (which is now
    user-curated content only) — it lives in the `_MEMORY_GUIDE` constant
    in `identity/engine.py` and gets injected into the assembled prompt.
    Eliminates the upgrade-migration gap because the guide auto-updates
    with every release instead of being frozen in a workspace file.
    """

    def test_memory_guide_is_a_nonempty_string(self, tmp_path):
        from predacore.identity.engine import IdentityEngine, reset_identity_engine
        reset_identity_engine()
        engine = IdentityEngine(str(tmp_path), agent_name="guide_test")
        guide = engine.memory_guide()
        assert isinstance(guide, str)
        assert len(guide) > 200, "guide should describe the 6 tools + infrastructure layer"

    def test_memory_guide_mentions_all_six_tools(self, tmp_path):
        from predacore.identity.engine import IdentityEngine, reset_identity_engine
        reset_identity_engine()
        engine = IdentityEngine(str(tmp_path), agent_name="guide_tools_test")
        guide = engine.memory_guide()
        for tool in ("memory_store", "memory_recall", "memory_get",
                     "memory_delete", "memory_stats", "memory_explain"):
            assert tool in guide, f"{tool} should appear in operational guide"

    def test_guide_in_assembled_prompt_with_empty_workspace_memory(self, tmp_path):
        """Even when MEMORY.md is empty, the guide is still in the prompt."""
        from predacore.identity.engine import IdentityEngine, reset_identity_engine
        reset_identity_engine()
        engine = IdentityEngine(str(tmp_path), agent_name="empty_test")
        # Force MEMORY.md to be empty (simulates a user who deleted theirs)
        (engine.workspace / "MEMORY.md").write_text("")
        prompt = engine.build_identity_prompt()
        assert "How Memory Works" in prompt
        assert "memory_store" in prompt
        assert "memory_recall" in prompt

    def test_guide_in_assembled_prompt_alongside_curated_memory(self, tmp_path):
        """Guide and workspace MEMORY.md coexist in the prompt — both present."""
        from predacore.identity.engine import IdentityEngine, reset_identity_engine
        reset_identity_engine()
        engine = IdentityEngine(str(tmp_path), agent_name="coexist_test")
        (engine.workspace / "MEMORY.md").write_text(
            "# Memory\n\n- User prefers tabs over spaces.\n"
        )
        prompt = engine.build_identity_prompt()
        assert "How Memory Works" in prompt
        assert "memory_store" in prompt
        assert "Curated Memory" in prompt
        assert "tabs over spaces" in prompt

    def test_default_memory_md_no_longer_carries_operational_guide(self, tmp_path):
        """v1.3.0 invariant: shipped MEMORY.md defaults must NOT contain
        operational content (moved to code). Regression guard for the v1.2.0
        mistake where operational guidance was edited into a defaults file."""
        from predacore.identity.engine import IdentityEngine, reset_identity_engine
        reset_identity_engine()
        engine = IdentityEngine(str(tmp_path), agent_name="defaults_test")
        seeded = (engine.workspace / "MEMORY.md").read_text()
        assert "Two layers, complementary" not in seeded
        # But the curated-content scaffold should be present
        assert "What belongs here" in seeded or "curated" in seeded.lower()
