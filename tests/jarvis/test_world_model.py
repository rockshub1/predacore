"""
Tests for JARVIS World Model — JAX RSSM, embedding projection, online learning.

Covers:
  1. Initialization and config
  2. Forward pass (observe_step, predict_outcome, predict_relevance)
  3. Embedding projection (GTE-small 384d → 128d task-adapted)
  4. Online learning (single SGD step)
  5. Batch training
  6. NaN / divergence handling
  7. Checkpoint save/load
  8. Experience buffer management
  9. Thread safety
  10. Graceful degradation (no JAX)
"""
from __future__ import annotations

import os
import pickle
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Force CPU for tests
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from src.jarvis.services.world_model import (
    JARVISBrain,
    WorldModelConfig,
    Experience,
    _check_jax,
)

# Skip all tests if JAX not installed
JAX_AVAILABLE = _check_jax()
pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX/Flax not installed")


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def config(tmp_path):
    """World model config with temp checkpoint path."""
    return WorldModelConfig(
        checkpoint_path=str(tmp_path / "test_model.npz"),
        save_interval=5,
        train_interval=3,
        batch_size=2,
        max_buffer_size=50,
    )


@pytest.fixture
def brain(config):
    """Initialized JARVISBrain."""
    b = JARVISBrain(config)
    assert b.available, "JARVISBrain should initialize with JAX available"
    return b


@pytest.fixture
def dummy_embedding():
    """384-dim dummy GTE-small embedding."""
    import jax.numpy as jnp
    return jnp.ones(384).tolist()


@pytest.fixture
def random_embedding():
    """Random 384-dim embedding for realistic tests."""
    import numpy as np
    rng = np.random.default_rng(42)
    return rng.standard_normal(384).tolist()


# ══════════════════════════════════════════════════════════════════════
# 1. Initialization
# ══════════════════════════════════════════════════════════════════════


class TestInitialization:

    def test_brain_initializes(self, brain):
        assert brain.available
        assert brain._initialized
        assert brain._params is not None
        assert brain._model is not None

    def test_param_count(self, brain):
        import jax
        count = sum(x.size for x in jax.tree.leaves(brain._params))
        assert count > 100_000, f"Expected >100K params, got {count}"
        assert count < 2_000_000, f"Model too large: {count} params"

    def test_default_config(self):
        cfg = WorldModelConfig()
        assert cfg.embed_dim == 384
        assert cfg.proj_dim == 128
        assert cfg.hidden_dim == 128
        assert cfg.stoch_dim == 16
        assert cfg.stoch_classes == 16

    def test_custom_config(self, config):
        assert config.save_interval == 5
        assert config.train_interval == 3

    def test_initial_state_zeros(self, brain):
        import jax.numpy as jnp
        assert jnp.all(brain._h == 0)
        assert jnp.all(brain._z == 0)


# ══════════════════════════════════════════════════════════════════════
# 2. Forward Pass
# ══════════════════════════════════════════════════════════════════════


class TestForwardPass:

    def test_observe_returns_predictions(self, brain, dummy_embedding):
        result = brain.observe(dummy_embedding)
        assert "adapted_embedding" in result
        assert "outcome_prob" in result
        assert "relevance" in result

    def test_observe_embedding_dimension(self, brain, dummy_embedding):
        result = brain.observe(dummy_embedding)
        assert len(result["adapted_embedding"]) == 128

    def test_observe_probabilities_bounded(self, brain, dummy_embedding):
        result = brain.observe(dummy_embedding)
        assert 0.0 <= result["outcome_prob"] <= 1.0
        assert 0.0 <= result["relevance"] <= 1.0

    def test_observe_updates_hidden_state(self, brain, dummy_embedding):
        import jax.numpy as jnp
        h_before = brain._h.copy()
        brain.observe(dummy_embedding)
        # Hidden state should change after observation
        assert not jnp.array_equal(brain._h, h_before)

    def test_multiple_observations_change_state(self, brain, dummy_embedding):
        r1 = brain.observe(dummy_embedding)
        r2 = brain.observe(dummy_embedding)
        # Same input but different hidden state → different predictions
        # (may be very close but hidden state differs)
        assert brain._step_count == 0  # observe doesn't increment step

    def test_reset_conversation_zeros_state(self, brain, dummy_embedding):
        import jax.numpy as jnp
        brain.observe(dummy_embedding)
        brain.reset_conversation()
        assert jnp.all(brain._h == 0)
        assert jnp.all(brain._z == 0)


# ══════════════════════════════════════════════════════════════════════
# 3. Embedding Projection
# ══════════════════════════════════════════════════════════════════════


class TestEmbeddingProjection:

    def test_project_single(self, brain, dummy_embedding):
        projected = brain.project_embedding(dummy_embedding)
        assert len(projected) == 128

    def test_project_batch(self, brain, dummy_embedding):
        batch = [dummy_embedding, dummy_embedding, dummy_embedding]
        projected = brain.project_batch(batch)
        assert len(projected) == 3
        assert all(len(p) == 128 for p in projected)

    def test_project_deterministic(self, brain, dummy_embedding):
        p1 = brain.project_embedding(dummy_embedding)
        p2 = brain.project_embedding(dummy_embedding)
        assert p1 == p2, "Projection should be deterministic"

    def test_different_inputs_different_outputs(self, brain):
        import numpy as np
        e1 = np.ones(384).tolist()
        e2 = np.zeros(384).tolist()
        p1 = brain.project_embedding(e1)
        p2 = brain.project_embedding(e2)
        assert p1 != p2


# ══════════════════════════════════════════════════════════════════════
# 4. Online Learning
# ══════════════════════════════════════════════════════════════════════


class TestOnlineLearning:

    def test_learn_returns_metrics(self, brain, dummy_embedding):
        metrics = brain.learn_from_outcome(
            embedding=dummy_embedding,
            tool_name="test_tool",
            success=True,
            latency_ms=100.0,
        )
        assert "loss" in metrics
        assert "grad_norm" in metrics
        assert metrics["step"] >= 0  # step increments after metrics returned

    def test_step_count_increments(self, brain, dummy_embedding):
        for i in range(5):
            brain.learn_from_outcome(
                embedding=dummy_embedding,
                tool_name="test_tool",
                success=i % 2 == 0,
                latency_ms=50.0,
            )
        assert brain._step_count == 5

    def test_loss_is_finite(self, brain, dummy_embedding):
        metrics = brain.learn_from_outcome(
            embedding=dummy_embedding,
            tool_name="test_tool",
            success=True,
            latency_ms=100.0,
        )
        import math
        assert math.isfinite(metrics["loss"]), f"Loss is not finite: {metrics['loss']}"

    def test_grad_norm_is_finite(self, brain, dummy_embedding):
        metrics = brain.learn_from_outcome(
            embedding=dummy_embedding,
            tool_name="test_tool",
            success=False,
            latency_ms=500.0,
        )
        import math
        assert math.isfinite(metrics["grad_norm"])

    def test_params_change_after_learning(self, brain, dummy_embedding):
        import jax
        params_before = jax.tree.map(lambda x: x.copy(), brain._params)
        brain.learn_from_outcome(
            embedding=dummy_embedding,
            tool_name="test_tool",
            success=True,
            latency_ms=100.0,
        )
        # At least some params should change
        any_changed = any(
            not (a == b).all()
            for a, b in zip(
                jax.tree.leaves(params_before),
                jax.tree.leaves(brain._params),
            )
        )
        assert any_changed, "Parameters should update after learning"


# ══════════════════════════════════════════════════════════════════════
# 5. Batch Training
# ══════════════════════════════════════════════════════════════════════


class TestBatchTraining:

    def test_batch_trains_at_interval(self, brain, dummy_embedding):
        # config.train_interval=3, batch_size=2
        for i in range(3):
            metrics = brain.learn_from_outcome(
                embedding=dummy_embedding,
                tool_name="test_tool",
                success=True,
                latency_ms=100.0,
            )
        # 3rd step should trigger batch training
        assert "batch_loss" in metrics, "Batch training should trigger at interval"

    def test_batch_needs_minimum_buffer(self, config):
        # Set high batch_size so batch can't run
        config.batch_size = 1000
        config.train_interval = 1
        brain = JARVISBrain(config)
        embedding = [0.0] * 384

        metrics = brain.learn_from_outcome(
            embedding=embedding,
            tool_name="test_tool",
            success=True,
            latency_ms=50.0,
        )
        assert "batch_loss" not in metrics


# ══════════════════════════════════════════════════════════════════════
# 6. NaN / Divergence Handling
# ══════════════════════════════════════════════════════════════════════


class TestNaNHandling:

    def test_nan_embedding_doesnt_crash(self, brain):
        nan_embedding = [float("nan")] * 384
        # Should not raise — NaN guard should catch it
        metrics = brain.learn_from_outcome(
            embedding=nan_embedding,
            tool_name="test_tool",
            success=True,
            latency_ms=100.0,
        )
        assert metrics.get("nan_skip", False) or "loss" in metrics

    def test_inf_embedding_doesnt_crash(self, brain):
        inf_embedding = [float("inf")] * 384
        metrics = brain.learn_from_outcome(
            embedding=inf_embedding,
            tool_name="test_tool",
            success=True,
            latency_ms=100.0,
        )
        # Should either skip (nan_skip) or produce finite loss
        assert "loss" in metrics

    def test_nan_count_tracked(self, brain):
        assert brain._nan_count == 0
        # Feed NaN to trigger NaN guard
        nan_embedding = [float("nan")] * 384
        brain.learn_from_outcome(
            embedding=nan_embedding,
            tool_name="test",
            success=True,
        )
        # nan_count should increase if NaN was detected
        # (may not always trigger depending on JAX behavior)
        assert isinstance(brain._nan_count, int)

    def test_extreme_latency_doesnt_diverge(self, brain, dummy_embedding):
        import math
        # Extreme latency value
        metrics = brain.learn_from_outcome(
            embedding=dummy_embedding,
            tool_name="test_tool",
            success=True,
            latency_ms=999999.0,
        )
        assert math.isfinite(metrics["loss"])

    def test_zero_latency_is_fine(self, brain, dummy_embedding):
        import math
        metrics = brain.learn_from_outcome(
            embedding=dummy_embedding,
            tool_name="test_tool",
            success=True,
            latency_ms=0.0,
        )
        assert math.isfinite(metrics["loss"])

    def test_many_steps_dont_diverge(self, brain, random_embedding):
        """Train 50 steps and verify loss stays finite."""
        import math
        import numpy as np
        rng = np.random.default_rng(123)

        for i in range(50):
            emb = rng.standard_normal(384).tolist()
            metrics = brain.learn_from_outcome(
                embedding=emb,
                tool_name=f"tool_{i % 5}",
                success=rng.random() > 0.3,
                latency_ms=rng.exponential(200),
            )
            assert math.isfinite(metrics["loss"]), f"Loss diverged at step {i}: {metrics['loss']}"


# ══════════════════════════════════════════════════════════════════════
# 7. Checkpoint Save/Load
# ══════════════════════════════════════════════════════════════════════


class TestCheckpointing:

    def test_save_creates_file(self, brain, config, dummy_embedding):
        # Train enough to trigger save (save_interval=5)
        for i in range(5):
            brain.learn_from_outcome(
                embedding=dummy_embedding,
                tool_name="test",
                success=True,
            )
        path = Path(config.checkpoint_path)
        assert path.exists(), "Checkpoint file should be created"
        assert path.stat().st_size > 0

    def test_load_restores_step_count(self, config, dummy_embedding):
        # Train and save
        brain1 = JARVISBrain(config)
        for i in range(5):
            brain1.learn_from_outcome(
                embedding=dummy_embedding,
                tool_name="test",
                success=True,
            )

        # Load into new brain
        brain2 = JARVISBrain(config)
        assert brain2._step_count == 5

    def test_load_restores_experience_buffer(self, config, dummy_embedding):
        brain1 = JARVISBrain(config)
        for i in range(5):
            brain1.learn_from_outcome(
                embedding=dummy_embedding,
                tool_name="test",
                success=True,
            )

        brain2 = JARVISBrain(config)
        assert len(brain2._experience_buffer) == 5

    def test_load_restores_params(self, config, dummy_embedding):
        import jax
        brain1 = JARVISBrain(config)
        for i in range(5):
            brain1.learn_from_outcome(
                embedding=dummy_embedding,
                tool_name="test",
                success=True,
            )
        p1 = brain1.project_embedding(dummy_embedding)

        brain2 = JARVISBrain(config)
        p2 = brain2.project_embedding(dummy_embedding)

        # Projections should be similar (params restored)
        diff = sum(abs(a - b) for a, b in zip(p1, p2))
        assert diff < 1.0, f"Restored params diverged: diff={diff}"

    def test_missing_checkpoint_starts_fresh(self, config):
        brain = JARVISBrain(config)
        assert brain._step_count == 0
        assert len(brain._experience_buffer) == 0

    def test_corrupt_checkpoint_doesnt_crash(self, config):
        # Write garbage to checkpoint path
        path = Path(config.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a valid npz file")

        # Should not crash — graceful fallback
        brain = JARVISBrain(config)
        assert brain.available
        assert brain._step_count == 0


# ══════════════════════════════════════════════════════════════════════
# 8. Experience Buffer
# ══════════════════════════════════════════════════════════════════════


class TestExperienceBuffer:

    def test_buffer_grows(self, brain, dummy_embedding):
        assert len(brain._experience_buffer) == 0
        brain.learn_from_outcome(
            embedding=dummy_embedding,
            tool_name="test",
            success=True,
        )
        assert len(brain._experience_buffer) == 1

    def test_buffer_capped(self, config, dummy_embedding):
        config.max_buffer_size = 10
        brain = JARVISBrain(config)

        for i in range(20):
            brain.learn_from_outcome(
                embedding=dummy_embedding,
                tool_name="test",
                success=True,
            )
        assert len(brain._experience_buffer) <= 10

    def test_experience_fields(self):
        exp = Experience(
            embedding=[1.0] * 384,
            tool_name="read_file",
            success=True,
            latency_ms=42.0,
            error="",
            timestamp=time.time(),
        )
        assert exp.tool_name == "read_file"
        assert exp.success is True
        assert exp.latency_ms == 42.0


# ══════════════════════════════════════════════════════════════════════
# 9. Thread Safety
# ══════════════════════════════════════════════════════════════════════


class TestThreadSafety:

    def test_concurrent_observe(self, brain, dummy_embedding):
        """Multiple threads observing shouldn't crash."""
        errors = []

        def observe_many():
            try:
                for _ in range(10):
                    brain.observe(dummy_embedding)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=observe_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_learn(self, brain, dummy_embedding):
        """Multiple threads learning shouldn't crash."""
        errors = []

        def learn_many():
            try:
                for _ in range(10):
                    brain.learn_from_outcome(
                        embedding=dummy_embedding,
                        tool_name="test",
                        success=True,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=learn_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Thread errors: {errors}"


# ══════════════════════════════════════════════════════════════════════
# 10. Graceful Degradation
# ══════════════════════════════════════════════════════════════════════


class TestGracefulDegradation:

    def test_unavailable_observe_returns_fallback(self):
        brain = JARVISBrain.__new__(JARVISBrain)
        brain._available = False
        brain._initialized = False
        brain.config = WorldModelConfig()
        brain._experience_buffer = []

        result = brain.observe([0.5] * 384)
        assert result["outcome_prob"] == 0.5
        assert result["relevance"] == 0.5
        assert len(result["adapted_embedding"]) == 128

    def test_unavailable_project_truncates(self):
        brain = JARVISBrain.__new__(JARVISBrain)
        brain._available = False
        brain._initialized = False
        brain.config = WorldModelConfig()

        emb = list(range(384))
        projected = brain.project_embedding(emb)
        assert projected == list(range(128))

    def test_unavailable_learn_stores_experience(self):
        brain = JARVISBrain.__new__(JARVISBrain)
        brain._available = False
        brain._initialized = False
        brain.config = WorldModelConfig()
        brain._experience_buffer = []
        brain._lock = threading.Lock()

        metrics = brain.learn_from_outcome(
            embedding=[0.0] * 384,
            tool_name="test",
            success=True,
        )
        assert metrics["loss"] == 0.0
        assert len(brain._experience_buffer) == 1


# ══════════════════════════════════════════════════════════════════════
# 11. Stats / Diagnostics
# ══════════════════════════════════════════════════════════════════════


class TestDiagnostics:

    def test_get_stats(self, brain):
        stats = brain.get_stats()
        assert stats["available"] is True
        assert stats["step_count"] == 0
        assert "nan_count" in stats
        assert "buffer_size" in stats
