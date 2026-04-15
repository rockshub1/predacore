"""
JARVIS World Model — JAX/Flax RSSM adapted from DreamerV3.

Hybrid architecture:
  GTE-small (384-dim, pretrained, static) = eyes → sees text
  World Model (JAX RSSM, online-trained) = brain → learns from experience

The world model learns a projection layer on top of GTE-small embeddings,
adapting them for JARVIS's specific use patterns. It gets better over time
as JARVIS processes more conversations and records more outcomes.

Architecture (adapted from SINGULARITY_OMEGA DreamerV3):
  1. Projection MLP: 384 → 256 → 128 (task-adapted embedding)
  2. GRU Cell: 128-dim recurrent state (conversation context)
  3. RSSM: Encoder + Prior with Gumbel-Softmax categorical states
  4. Outcome Head: Predict tool success probability
  5. Relevance Head: Predict memory retrieval relevance

Training signal:
  - OutcomeStore success/failure → outcome predictor target
  - Tool errors → negative signal
  - Latency → efficiency signal

Online learning:
  - Single SGD step after each _record_outcome() call
  - Periodic batch training from OutcomeStore history
  - Checkpoint to ~/.prometheus/world_model.npz (numpy, NOT pickle)
"""
from __future__ import annotations

import json
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Force JAX to CPU on macOS (Metal backend doesn't support default_memory_space).
# MUST be set BEFORE jax is imported anywhere — once JAX initializes a backend
# the config is locked.
if platform.system() == "Darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Checkpoint format version — bump when architecture changes
CHECKPOINT_VERSION = 2

# Max time (seconds) for a single training step before we skip it
TRAINING_TIMEOUT_SEC = float(os.getenv("JARVIS_WM_TRAIN_TIMEOUT", "0.2"))


@dataclass
class WorldModelConfig:
    """World model hyperparameters — adapted from DreamerV3 Config.

    All numeric fields can be overridden via JARVIS_WM_* env vars:
        JARVIS_WM_EMBED_DIM, JARVIS_WM_PROJ_DIM, JARVIS_WM_HIDDEN_DIM,
        JARVIS_WM_LEARNING_RATE, JARVIS_WM_BATCH_SIZE, JARVIS_WM_TRAIN_INTERVAL,
        JARVIS_WM_SAVE_INTERVAL, JARVIS_WM_MAX_BUFFER_SIZE, JARVIS_WM_CHECKPOINT_PATH
    """
    # Input dimension (GTE-small output)
    embed_dim: int = 384

    # Projection dimension (task-adapted embedding output)
    proj_dim: int = 128

    # RSSM dimensions (scaled down from DreamerV3's 256/32/32)
    hidden_dim: int = 128       # GRU hidden state
    stoch_dim: int = 16         # Categorical state dimensions
    stoch_classes: int = 16     # Classes per dimension

    # Training
    learning_rate: float = 2e-4
    lr_warmup_steps: int = 100
    lr_init: float = 1e-5
    grad_clip: float = 1.0
    kl_coef: float = 0.1

    # Online learning
    online_lr: float = 1e-4     # Lower LR for single-step updates
    batch_size: int = 32
    train_interval: int = 50    # Full batch train every N outcomes

    # Persistence — .npz (numpy) for security, NOT pickle
    checkpoint_path: str = "~/.prometheus/world_model.npz"
    save_interval: int = 100    # Save every N updates

    # Experience buffer
    max_buffer_size: int = 10000

    @classmethod
    def from_env(cls) -> "WorldModelConfig":
        """Create config with env var overrides (JARVIS_WM_*)."""
        cfg = cls()
        _env_int = lambda key, default: int(os.getenv(f"JARVIS_WM_{key}", default))
        _env_float = lambda key, default: float(os.getenv(f"JARVIS_WM_{key}", default))
        cfg.embed_dim = _env_int("EMBED_DIM", cfg.embed_dim)
        cfg.proj_dim = _env_int("PROJ_DIM", cfg.proj_dim)
        cfg.hidden_dim = _env_int("HIDDEN_DIM", cfg.hidden_dim)
        cfg.learning_rate = _env_float("LEARNING_RATE", cfg.learning_rate)
        cfg.batch_size = _env_int("BATCH_SIZE", cfg.batch_size)
        cfg.train_interval = _env_int("TRAIN_INTERVAL", cfg.train_interval)
        cfg.save_interval = _env_int("SAVE_INTERVAL", cfg.save_interval)
        cfg.max_buffer_size = _env_int("MAX_BUFFER_SIZE", cfg.max_buffer_size)
        cfg.checkpoint_path = os.getenv("JARVIS_WM_CHECKPOINT_PATH", cfg.checkpoint_path)
        return cfg


# ---------------------------------------------------------------------------
# JAX/Flax Neural Network Components
# Directly adapted from SINGULARITY_OMEGA_TRAIN_SOL.py
# ---------------------------------------------------------------------------

def _check_jax():
    """Check if JAX + Flax are available."""
    try:
        import jax  # noqa: F401
        import flax  # noqa: F401
        import optax  # noqa: F401

        logger.info("JAX: backend=%s, devices=%s", jax.default_backend(), jax.devices())
        return True
    except ImportError:
        return False


def _build_modules(config: WorldModelConfig):
    """Build all Flax modules. Returns dict of module instances."""
    import jax.numpy as jnp
    from flax import linen as nn

    class MLP(nn.Module):
        """MLP with SiLU + LayerNorm — same as DreamerV3."""
        hidden: int
        output: int

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(self.hidden)(x)
            x = nn.silu(x)
            x = nn.LayerNorm()(x)
            x = nn.Dense(self.output)(x)
            return x

    class GRUCell(nn.Module):
        """Gated Recurrent Unit — same as DreamerV3."""
        hidden_dim: int

        @nn.compact
        def __call__(self, h, x):
            combined = jnp.concatenate([h, x], axis=-1)
            z = nn.sigmoid(nn.Dense(self.hidden_dim)(combined))
            r = nn.sigmoid(nn.Dense(self.hidden_dim)(combined))
            combined_reset = jnp.concatenate([r * h, x], axis=-1)
            h_candidate = jnp.tanh(nn.Dense(self.hidden_dim)(combined_reset))
            h_new = (1 - z) * h + z * h_candidate
            return h_new

    class JARVISWorldModel(nn.Module):
        """
        RSSM-based world model for JARVIS.

        Adapted from DreamerV3 RSSM:
        - Projection: 384d GTE-small → 128d task-adapted
        - GRU: Recurrent state tracking across conversation turns
        - Encoder: [obs, h] → posterior stochastic state z
        - Prior: h → predicted z (for temporal consistency)
        - Outcome head: [h, z] → P(success)
        - Relevance head: [h, z] → relevance score for retrieval
        """
        hidden_dim: int = 128
        proj_dim: int = 128
        stoch_dim: int = 16
        stoch_classes: int = 16
        embed_dim: int = 384

        def setup(self):
            # Projection: GTE-small 384d → 128d task-adapted
            self.projector = MLP(256, self.proj_dim)

            # RSSM core (same architecture as DreamerV3)
            self.gru = GRUCell(self.hidden_dim)
            self.encoder = MLP(self.hidden_dim, self.stoch_dim * self.stoch_classes)
            self.prior = MLP(self.hidden_dim, self.stoch_dim * self.stoch_classes)

            # Prediction heads
            self.outcome_head = MLP(self.hidden_dim, 1)       # P(success)
            self.relevance_head = MLP(self.hidden_dim, 1)     # Relevance score
            self.latency_head = MLP(self.hidden_dim, 1)       # Expected latency

        def __call__(self, obs, rng):
            """Full forward pass for initialization."""
            import jax
            from jax import random

            batch_size = obs.shape[0]
            h = jnp.zeros((batch_size, self.hidden_dim))
            z = jnp.zeros((batch_size, self.stoch_dim * self.stoch_classes))

            # Project embedding
            projected = self.projector(obs)

            # GRU step
            combined = jnp.concatenate([z, projected], axis=-1)
            h = self.gru(h, combined)

            # Encoder + Prior
            post_logits = self.encoder(jnp.concatenate([projected, h], axis=-1))
            prior_logits = self.prior(h)

            # Predictions
            state = jnp.concatenate([h, z], axis=-1)
            outcome = jax.nn.sigmoid(self.outcome_head(state))
            relevance = jax.nn.sigmoid(self.relevance_head(state))
            latency = nn.softplus(self.latency_head(state))

            return h, projected, outcome

        def initial_state(self, batch_size):
            """Initialize hidden + stochastic state."""
            h = jnp.zeros((batch_size, self.hidden_dim))
            z = jnp.zeros((batch_size, self.stoch_dim * self.stoch_classes))
            return h, z

        def observe_step(self, h, z, obs, rng):
            """Process one observation, update state. Returns adapted embedding."""
            import jax
            from jax import random

            # Project GTE-small embedding → task-adapted
            projected = self.projector(obs)

            # GRU update
            combined = jnp.concatenate([z, projected], axis=-1)
            h = self.gru(h, combined)

            # Posterior (has access to observation)
            post_logits = self.encoder(jnp.concatenate([projected, h], axis=-1))
            post_logits = post_logits.reshape(-1, self.stoch_dim, self.stoch_classes)

            # Prior (prediction from dynamics alone)
            prior_logits = self.prior(h)
            prior_logits = prior_logits.reshape(-1, self.stoch_dim, self.stoch_classes)

            # Sample with Gumbel-Softmax (straight-through)
            z = self._sample_categorical(post_logits, rng)
            z_flat = z.reshape(-1, self.stoch_dim * self.stoch_classes)

            return h, z_flat, projected, post_logits, prior_logits

        def predict_outcome(self, h, z):
            """Predict success probability from current state."""
            import jax
            state = jnp.concatenate([h, z], axis=-1)
            logit = self.outcome_head(state)
            return jax.nn.sigmoid(logit)

        def predict_relevance(self, h, z):
            """Predict relevance score for memory retrieval."""
            import jax
            state = jnp.concatenate([h, z], axis=-1)
            logit = self.relevance_head(state)
            return jax.nn.sigmoid(logit)

        def get_adapted_embedding(self, obs):
            """Get the task-adapted embedding without updating state."""
            return self.projector(obs)

        def _sample_categorical(self, logits, rng):
            """Gumbel-Softmax sampling — straight-through estimator."""
            import jax
            from jax import random
            gumbel = random.gumbel(rng, logits.shape)
            indices = jnp.argmax(logits + gumbel, axis=-1)
            one_hot = jax.nn.one_hot(indices, self.stoch_classes)
            soft = jax.nn.softmax(logits, axis=-1)
            return one_hot - jax.lax.stop_gradient(soft) + soft

    return JARVISWorldModel(
        hidden_dim=config.hidden_dim,
        proj_dim=config.proj_dim,
        stoch_dim=config.stoch_dim,
        stoch_classes=config.stoch_classes,
        embed_dim=config.embed_dim,
    )


# ---------------------------------------------------------------------------
# Experience Buffer
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    """A single experience for training the world model."""
    embedding: list[float]       # 384-dim GTE-small vector
    tool_name: str               # Tool that was called
    success: bool                # Whether it succeeded
    latency_ms: float            # Response latency
    error: str = ""              # Error message if failed
    timestamp: float = 0.0       # When this happened


# ---------------------------------------------------------------------------
# World Model Manager
# ---------------------------------------------------------------------------

class JARVISBrain:
    """
    Manages the JAX world model lifecycle:
    - Initialization and parameter management
    - Online learning (single-step SGD after each outcome)
    - Batch training from experience buffer
    - Checkpoint save/load
    - Embedding projection (GTE-small → task-adapted)

    This is the "brain" that sits on top of GTE-small "eyes."
    It learns from every interaction and gets better over time.
    """

    def __init__(self, config: WorldModelConfig | None = None):
        self.config = config or WorldModelConfig.from_env()
        self._available = _check_jax()
        self._initialized = False
        self._model = None
        self._params = None
        self._opt_state = None
        self._optimizer = None
        self._rng = None
        self._step_count = 0
        self._nan_count = 0  # Track NaN gradients for diagnostics
        self._timeout_count = 0  # Track training timeouts
        self._experience_buffer: list[Experience] = []
        self._lock = threading.Lock()  # Thread safety for async callers

        # Hidden state for conversation tracking
        self._h = None
        self._z = None

        # JIT-compiled functions (set during _initialize)
        self._jit_online_loss = None
        self._jit_batch_loss = None

        # Prometheus observability
        self._setup_metrics()

        if self._available:
            self._initialize()

    def _setup_metrics(self):
        """Register Prometheus metrics for world model observability."""
        try:
            from prometheus_client import Counter, Gauge, Histogram, REGISTRY

            def _get_or_create(cls, name, *args, **kwargs):
                try:
                    return cls(name, *args, **kwargs)
                except ValueError:
                    collector = REGISTRY._names_to_collectors.get(name)
                    if collector:
                        return collector
                    # Try with suffix variants
                    for sfx in ["_total", "_created"]:
                        collector = REGISTRY._names_to_collectors.get(name + sfx)
                        if collector:
                            try:
                                REGISTRY.unregister(collector)
                            except (KeyError, AttributeError):
                                pass
                            return cls(name, *args, **kwargs)
                    return cls(name, *args, **kwargs)

            self._m_steps = _get_or_create(
                Counter, "jarvis_wm_train_steps_total",
                "Total world model training steps",
            )
            self._m_loss = _get_or_create(
                Gauge, "jarvis_wm_last_loss",
                "Most recent training loss",
            )
            self._m_nan = _get_or_create(
                Counter, "jarvis_wm_nan_skips_total",
                "Training steps skipped due to NaN gradients",
            )
            self._m_timeouts = _get_or_create(
                Counter, "jarvis_wm_timeout_skips_total",
                "Training steps skipped due to timeout",
            )
            self._m_buffer = _get_or_create(
                Gauge, "jarvis_wm_buffer_size",
                "Current experience buffer size",
            )
            self._m_train_latency = _get_or_create(
                Histogram, "jarvis_wm_train_latency_seconds",
                "Training step latency",
                buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
            )
            self._m_checkpoint = _get_or_create(
                Counter, "jarvis_wm_checkpoints_saved_total",
                "Total checkpoints saved",
            )
            self._metrics_available = True
        except ImportError:
            self._metrics_available = False

    @property
    def available(self) -> bool:
        return self._available and self._initialized

    def _initialize(self):
        """Initialize model, parameters, optimizer, and JIT-compile loss functions."""
        try:
            import jax
            import jax.numpy as jnp
            from jax import random, jit, value_and_grad
            import optax

            self._rng = random.PRNGKey(42)

            # Build model
            self._model = _build_modules(self.config)

            # Initialize parameters with dummy input
            self._rng, init_rng = random.split(self._rng)
            dummy_obs = jnp.zeros((1, self.config.embed_dim))
            self._params = self._model.init(init_rng, dummy_obs, init_rng)

            # Optimizer: Adam with warmup + gradient clipping
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=self.config.lr_init,
                peak_value=self.config.learning_rate,
                warmup_steps=self.config.lr_warmup_steps,
                decay_steps=50000,
                end_value=self.config.lr_init,
            )
            self._optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.grad_clip),
                optax.adam(schedule),
            )
            self._opt_state = self._optimizer.init(self._params)

            # JIT-compile loss + grad functions for 10-100x speedup
            model = self._model
            kl_coef = self.config.kl_coef

            def _loss_fn(params, obs, target_success, target_latency, rng):
                """Unified loss: outcome BCE + latency MSE + KL divergence."""
                batch_size = obs.shape[0]
                h, z = model.apply(params, batch_size, method=model.initial_state)
                h, z, projected, post_logits, prior_logits = model.apply(
                    params, h, z, obs, rng, method=model.observe_step,
                )

                # Outcome prediction loss (binary cross-entropy)
                pred_success = model.apply(params, h, z, method=model.predict_outcome)
                bce = -(
                    target_success * jnp.log(pred_success + 1e-8)
                    + (1 - target_success) * jnp.log(1 - pred_success + 1e-8)
                )

                # Latency prediction loss (MSE on log-scaled latency)
                state = jnp.concatenate([h, z], axis=-1)
                pred_latency = jax.nn.softplus(model.apply(
                    params, state, method=lambda self, s: self.latency_head(s),
                ))
                latency_loss = jnp.mean((pred_latency - target_latency) ** 2)

                # KL divergence between posterior and prior
                post_dist = jax.nn.softmax(post_logits, axis=-1)
                prior_dist = jax.nn.softmax(prior_logits, axis=-1)
                kl = jnp.sum(
                    post_dist * (jnp.log(post_dist + 1e-8) - jnp.log(prior_dist + 1e-8)),
                    axis=-1,
                )

                total_loss = jnp.mean(bce) + 0.1 * latency_loss + kl_coef * jnp.mean(kl)
                return total_loss

            self._jit_loss_and_grad = jit(value_and_grad(_loss_fn))

            # Try to load existing checkpoint
            self._load_checkpoint()

            # Initialize conversation state
            self._reset_state()

            self._initialized = True
            param_count = sum(
                x.size for x in jax.tree.leaves(self._params)
            )
            logger.info(
                "JARVISBrain initialized: %d parameters, JAX backend: %s (JIT compiled)",
                param_count, jax.default_backend(),
            )

        except (ImportError, OSError, ValueError, RuntimeError) as e:
            logger.warning("JARVISBrain init failed: %s", e)
            self._available = False
            self._initialized = False

    def _reset_state(self):
        """Reset conversation-level hidden state."""
        import jax.numpy as jnp
        self._h = jnp.zeros((1, self.config.hidden_dim))
        self._z = jnp.zeros((1, self.config.stoch_dim * self.config.stoch_classes))

    # ------------------------------------------------------------------
    # Core API: Embedding projection
    # ------------------------------------------------------------------

    def project_embedding(self, gte_embedding: list[float]) -> list[float]:
        """
        Project a GTE-small 384-dim embedding → 128-dim task-adapted embedding.

        This is the main "product" of the world model. The projection layer
        learns from JARVIS's experience which dimensions of the embedding
        space matter most for its specific tasks.

        Args:
            gte_embedding: 384-dim vector from GTE-small

        Returns:
            128-dim task-adapted vector
        """
        if not self.available:
            # Fallback: truncate to 128 dims (still better than nothing)
            return gte_embedding[:self.config.proj_dim]

        import jax.numpy as jnp

        obs = jnp.array([gte_embedding], dtype=jnp.float32)
        projected = self._model.apply(
            self._params, obs, method=self._model.get_adapted_embedding
        )
        return projected[0].tolist()

    def project_batch(self, embeddings: list[list[float]]) -> list[list[float]]:
        """Project a batch of GTE-small embeddings → task-adapted embeddings."""
        if not self.available:
            return [e[:self.config.proj_dim] for e in embeddings]

        import jax.numpy as jnp

        obs = jnp.array(embeddings, dtype=jnp.float32)
        projected = self._model.apply(
            self._params, obs, method=self._model.get_adapted_embedding
        )
        return projected.tolist()

    # ------------------------------------------------------------------
    # Core API: State tracking
    # ------------------------------------------------------------------

    def observe(self, gte_embedding: list[float]) -> dict[str, Any]:
        """
        Feed an observation (embedded message) into the RSSM.

        Updates the hidden state and returns predictions:
        - adapted_embedding: 128-dim projected vector
        - outcome_prob: predicted P(success) for next tool call
        - relevance: predicted relevance for memory retrieval

        Thread-safe: uses a lock to protect hidden state mutation.
        Call this for each user message / tool result in a conversation.
        """
        if not self.available:
            return {
                "adapted_embedding": gte_embedding[:self.config.proj_dim],
                "outcome_prob": 0.5,
                "relevance": 0.5,
            }

        import jax.numpy as jnp
        from jax import random

        with self._lock:
            obs = jnp.array([gte_embedding], dtype=jnp.float32)
            self._rng, step_rng = random.split(self._rng)

            h, z, projected, post_logits, prior_logits = self._model.apply(
                self._params, self._h, self._z, obs, step_rng,
                method=self._model.observe_step,
            )
            self._h = h
            self._z = z

            outcome_prob = self._model.apply(
                self._params, h, z, method=self._model.predict_outcome
            )
            relevance = self._model.apply(
                self._params, h, z, method=self._model.predict_relevance
            )

        return {
            "adapted_embedding": projected[0].tolist(),
            "outcome_prob": float(outcome_prob[0, 0]),
            "relevance": float(relevance[0, 0]),
        }

    def reset_conversation(self):
        """Reset hidden state for a new conversation."""
        if self.available:
            self._reset_state()

    def reset(self) -> dict[str, Any]:
        """Full model reset — reinitialize params, optimizer, and clear buffer.

        Use when the model has diverged or you want a fresh start.
        The old checkpoint is preserved with a .bak suffix.
        """
        if not self._available:
            return {"status": "unavailable", "reason": "JAX not installed"}

        with self._lock:
            # Backup existing checkpoint
            path = Path(self.config.checkpoint_path).expanduser()
            if path.suffix == ".pkl":
                path = path.with_suffix(".npz")
            if path.exists():
                bak = path.with_suffix(".npz.bak")
                try:
                    path.rename(bak)
                    logger.info("Old checkpoint backed up to %s", bak)
                except OSError as e:
                    logger.warning("Could not backup checkpoint: %s", e)

            old_steps = self._step_count
            old_buffer = len(self._experience_buffer)

            # Clear state
            self._step_count = 0
            self._nan_count = 0
            self._timeout_count = 0
            self._experience_buffer.clear()

            # Reinitialize model from scratch
            self._initialized = False
            self._initialize()

            logger.info(
                "JARVISBrain reset: cleared %d steps, %d experiences",
                old_steps, old_buffer,
            )
            return {
                "status": "reset",
                "old_steps": old_steps,
                "old_buffer": old_buffer,
                "backup": str(path.with_suffix(".npz.bak")),
            }

    # ------------------------------------------------------------------
    # Online Learning
    # ------------------------------------------------------------------

    def learn_from_outcome(
        self,
        embedding: list[float],
        tool_name: str,
        success: bool,
        latency_ms: float = 0.0,
        error: str = "",
    ) -> dict[str, float]:
        """
        Single SGD step from one outcome. Called after each _record_outcome().

        This is the "online learning" component — the model updates immediately
        from each interaction, getting incrementally better over time.

        Thread-safe: uses a lock since this can be called from async contexts.

        Returns training metrics (loss, grad_norm).
        """
        # Store experience (lock-free append is fine for list)
        exp = Experience(
            embedding=embedding,
            tool_name=tool_name,
            success=success,
            latency_ms=latency_ms,
            error=error,
            timestamp=time.time(),
        )
        with self._lock:
            self._experience_buffer.append(exp)
            # Trim buffer (under lock to prevent race with concurrent appends)
            if len(self._experience_buffer) > self.config.max_buffer_size:
                self._experience_buffer = self._experience_buffer[-self.config.max_buffer_size:]

        if not self.available:
            return {"loss": 0.0, "grad_norm": 0.0, "buffer_size": len(self._experience_buffer)}

        # Thread-safe gradient update
        with self._lock:
            # Online SGD step (JIT-compiled, NaN-guarded)
            metrics = self._online_step(exp)

            self._step_count += 1

            # Periodic batch training
            if (self._step_count % self.config.train_interval == 0
                    and len(self._experience_buffer) >= self.config.batch_size):
                batch_metrics = self._batch_train()
                metrics["batch_loss"] = batch_metrics.get("loss", 0.0)

            # Periodic checkpoint
            if self._step_count % self.config.save_interval == 0:
                self._save_checkpoint()
                if self._metrics_available:
                    self._m_checkpoint.inc()

            # Emit Prometheus metrics
            if self._metrics_available:
                self._m_steps.inc()
                self._m_loss.set(metrics.get("loss", 0.0))
                self._m_buffer.set(len(self._experience_buffer))
                train_ms = metrics.get("train_ms", 0)
                if train_ms:
                    self._m_train_latency.observe(train_ms / 1000.0)
                if metrics.get("timeout_skip"):
                    self._m_timeouts.inc()

        return metrics

    def _apply_gradients(self, loss, grads) -> dict[str, float]:
        """Apply gradients with NaN guard. Returns metrics."""
        import jax
        import jax.numpy as jnp

        # NaN guard — skip update if gradients are corrupted
        # (same protection as DreamerV3 NaN-skip)
        grad_norm = float(jnp.sqrt(
            sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads))
        ))
        loss_val = float(loss)

        if not (jnp.isfinite(loss) and jnp.isfinite(jnp.array(grad_norm))):
            self._nan_count += 1
            if self._metrics_available:
                self._m_nan.inc()
            logger.warning(
                "NaN detected in gradients (count=%d), skipping update",
                self._nan_count,
            )
            return {
                "loss": loss_val,
                "grad_norm": grad_norm,
                "nan_skip": True,
                "nan_count": self._nan_count,
                "step": self._step_count,
                "buffer_size": len(self._experience_buffer),
            }

        # Apply gradients
        updates, self._opt_state = self._optimizer.update(
            grads, self._opt_state, self._params
        )
        self._params = jax.tree.map(lambda p, u: p + u, self._params, updates)

        return {
            "loss": loss_val,
            "grad_norm": grad_norm,
            "nan_skip": False,
            "step": self._step_count,
            "buffer_size": len(self._experience_buffer),
        }

    def _online_step(self, exp: Experience) -> dict[str, float]:
        """Single JIT-compiled gradient step from one experience.

        Skips if the step takes longer than TRAINING_TIMEOUT_SEC to
        protect low-end hardware from blocking the event loop.
        """
        import jax.numpy as jnp
        from jax import random

        t0 = time.monotonic()

        obs = jnp.array([exp.embedding], dtype=jnp.float32)
        target_success = jnp.array([[1.0 if exp.success else 0.0]])
        # Log-scale latency for stable training (latency can be 0-10000ms)
        target_latency = jnp.array([[jnp.log1p(exp.latency_ms / 1000.0)]])

        self._rng, step_rng = random.split(self._rng)

        # JIT-compiled loss + grad (10-100x faster than Python mode)
        loss, grads = self._jit_loss_and_grad(
            self._params, obs, target_success, target_latency, step_rng
        )

        elapsed = time.monotonic() - t0
        # Skip timeout check on first few steps — JIT compilation is slow
        # but only happens once. After warmup, enforce the timeout.
        if elapsed > TRAINING_TIMEOUT_SEC and self._step_count > 3:
            self._timeout_count += 1
            logger.warning(
                "Online step took %.0fms (limit: %.0fms) — skipping gradient apply "
                "(timeouts=%d)",
                elapsed * 1000, TRAINING_TIMEOUT_SEC * 1000, self._timeout_count,
            )
            return {
                "loss": float(loss),
                "grad_norm": 0.0,
                "timeout_skip": True,
                "step": self._step_count,
                "buffer_size": len(self._experience_buffer),
            }

        metrics = self._apply_gradients(loss, grads)
        metrics["train_ms"] = round(elapsed * 1000, 1)
        return metrics

    def _batch_train(self) -> dict[str, float]:
        """Train on a random batch from the experience buffer (JIT-compiled).

        Timeout-guarded: skips if batch training exceeds 5x the online timeout.
        """
        import jax.numpy as jnp
        from jax import random
        import numpy as np

        t0 = time.monotonic()

        # Sample random batch
        indices = np.random.choice(
            len(self._experience_buffer),
            size=min(self.config.batch_size, len(self._experience_buffer)),
            replace=False,
        )
        batch = [self._experience_buffer[i] for i in indices]

        obs = jnp.array([e.embedding for e in batch], dtype=jnp.float32)
        targets = jnp.array(
            [[1.0 if e.success else 0.0] for e in batch], dtype=jnp.float32
        )
        target_latency = jnp.array(
            [[jnp.log1p(e.latency_ms / 1000.0)] for e in batch], dtype=jnp.float32
        )

        self._rng, step_rng = random.split(self._rng)

        # JIT-compiled loss + grad
        loss, grads = self._jit_loss_and_grad(
            self._params, obs, targets, target_latency, step_rng
        )

        elapsed = time.monotonic() - t0
        batch_timeout = TRAINING_TIMEOUT_SEC * 5  # batch gets more headroom
        if elapsed > batch_timeout:
            self._timeout_count += 1
            logger.warning(
                "Batch train took %.0fms (limit: %.0fms) — skipping apply",
                elapsed * 1000, batch_timeout * 1000,
            )
            return {
                "loss": float(loss), "batch_size": len(batch),
                "timeout_skip": True,
            }

        metrics = self._apply_gradients(loss, grads)
        metrics["batch_size"] = len(batch)
        metrics["batch_train_ms"] = round(elapsed * 1000, 1)
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_checkpoint(self):
        """Save parameters + experience buffer to disk (numpy .npz, NOT pickle).

        Format:
          - param__<flat_key> arrays: model parameters (one per leaf)
          - _meta.json embedded: step_count, config, version, timestamp
          - _experience.json embedded: serialized experience buffer
        """
        import jax
        import numpy as np

        path = Path(self.config.checkpoint_path).expanduser()
        # Migrate legacy .pkl path → .npz
        if path.suffix == ".pkl":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Flatten JAX params tree → dict of numpy arrays
        flat_params = {
            f"param__{'/'.join(str(k) for k in ks)}": np.asarray(v)
            for ks, v in jax.tree_util.tree_leaves_with_path(self._params)
        }

        # Metadata as JSON string → numpy byte array (no pickle needed)
        meta = {
            "version": CHECKPOINT_VERSION,
            "step_count": self._step_count,
            "nan_count": self._nan_count,
            "timestamp": time.time(),
            "config": {
                "embed_dim": self.config.embed_dim,
                "proj_dim": self.config.proj_dim,
                "hidden_dim": self.config.hidden_dim,
                "stoch_dim": self.config.stoch_dim,
                "stoch_classes": self.config.stoch_classes,
            },
        }
        flat_params["_meta"] = np.array(json.dumps(meta).encode("utf-8"))

        # Experience buffer as JSON (last 5K entries)
        buffer_data = [
            {
                "embedding": e.embedding,
                "tool_name": e.tool_name,
                "success": e.success,
                "latency_ms": e.latency_ms,
                "error": e.error,
                "timestamp": e.timestamp,
            }
            for e in self._experience_buffer[-5000:]
        ]
        flat_params["_experience"] = np.array(json.dumps(buffer_data).encode("utf-8"))

        try:
            # numpy.savez_compressed appends .npz if not present, so use a
            # temp file that already ends in .npz to avoid double-suffix
            tmp_path = path.parent / (path.stem + "_tmp.npz")
            np.savez_compressed(tmp_path, **flat_params)
            tmp_path.replace(path)  # atomic rename
            logger.info(
                "World model checkpoint saved: v%d step=%d buffer=%d path=%s",
                CHECKPOINT_VERSION, self._step_count, len(self._experience_buffer), path,
            )
        except OSError as e:
            logger.warning("Failed to save world model checkpoint: %s", e)
            # Clean up temp file on failure
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _load_checkpoint(self):
        """Load parameters + experience buffer from disk (numpy .npz).

        Validates checkpoint version — rejects incompatible versions
        rather than silently loading stale architectures.
        """
        import jax
        import jax.numpy as jnp
        import numpy as np

        path = Path(self.config.checkpoint_path).expanduser()
        if path.suffix == ".pkl":
            path = path.with_suffix(".npz")

        # Legacy migration: if .pkl exists but .npz doesn't, warn and skip
        pkl_path = path.with_suffix(".pkl")
        if not path.exists() and pkl_path.exists():
            logger.warning(
                "Legacy pickle checkpoint found at %s — ignoring for security. "
                "Delete it manually or the model will retrain from scratch.",
                pkl_path,
            )
            return

        if not path.exists():
            logger.info("No world model checkpoint found at %s — starting fresh", path)
            return

        try:
            data = np.load(path, allow_pickle=False)

            # Validate version
            meta_raw = bytes(data["_meta"])
            meta = json.loads(meta_raw.decode("utf-8"))
            version = meta.get("version", 1)
            if version != CHECKPOINT_VERSION:
                logger.warning(
                    "Checkpoint version mismatch: file=%d, expected=%d — starting fresh "
                    "(old checkpoint preserved at %s)",
                    version, CHECKPOINT_VERSION, path,
                )
                return

            # Restore params — reconstruct tree from flat keys
            param_arrays = {
                k: jnp.asarray(data[k], dtype=jnp.float32)
                for k in data.files
                if k.startswith("param__")
            }
            # Rebuild params tree by mapping leaf paths back
            flat_current = {
                "/".join(str(k) for k in ks): v
                for ks, v in jax.tree_util.tree_leaves_with_path(self._params)
            }
            restored_leaves = []
            for ks, _v in jax.tree_util.tree_leaves_with_path(self._params):
                key = "/".join(str(k) for k in ks)
                arr_key = f"param__{key}"
                if arr_key in param_arrays:
                    restored_leaves.append(param_arrays[arr_key])
                else:
                    logger.warning("Checkpoint missing param %s — keeping init value", key)
                    restored_leaves.append(jnp.asarray(_v))

            self._params = jax.tree.unflatten(
                jax.tree.structure(self._params), restored_leaves
            )

            # Re-init optimizer state for restored params
            self._opt_state = self._optimizer.init(self._params)

            self._step_count = meta.get("step_count", 0)
            self._nan_count = meta.get("nan_count", 0)

            # Restore experience buffer
            if "_experience" in data.files:
                exp_raw = bytes(data["_experience"])
                exp_list = json.loads(exp_raw.decode("utf-8"))
                self._experience_buffer = [
                    Experience(**e) for e in exp_list
                ]

            age_hours = (time.time() - meta.get("timestamp", time.time())) / 3600
            logger.info(
                "World model checkpoint loaded: v%d step=%d buffer=%d age=%.0fh",
                version, self._step_count, len(self._experience_buffer), age_hours,
            )
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as e:
            logger.warning("Failed to load world model checkpoint: %s", e)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return world model statistics."""
        stats = {
            "available": self._available,
            "initialized": self._initialized,
            "step_count": self._step_count,
            "buffer_size": len(self._experience_buffer),
            "nan_count": self._nan_count,
            "timeout_count": self._timeout_count,
            "checkpoint_version": CHECKPOINT_VERSION,
            "config": {
                "embed_dim": self.config.embed_dim,
                "proj_dim": self.config.proj_dim,
                "hidden_dim": self.config.hidden_dim,
                "learning_rate": self.config.learning_rate,
                "train_timeout_ms": TRAINING_TIMEOUT_SEC * 1000,
                "max_buffer_size": self.config.max_buffer_size,
            },
        }

        if self._available and self._initialized:
            import jax
            param_count = sum(
                x.size for x in jax.tree.leaves(self._params)
            )
            stats["param_count"] = param_count
            stats["backend"] = str(jax.default_backend())
            stats["jit_compiled"] = self._jit_loss_and_grad is not None

            # Buffer stats
            if self._experience_buffer:
                successes = sum(1 for e in self._experience_buffer if e.success)
                stats["buffer_success_rate"] = successes / len(self._experience_buffer)
                stats["buffer_oldest"] = min(e.timestamp for e in self._experience_buffer)
                stats["buffer_newest"] = max(e.timestamp for e in self._experience_buffer)

        return stats


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_brain_instance: JARVISBrain | None = None


def get_brain(config: WorldModelConfig | None = None) -> JARVISBrain:
    """Get or create the singleton JARVISBrain instance."""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = JARVISBrain(config)
    return _brain_instance


def reset_brain() -> dict[str, Any]:
    """Reset the singleton brain instance. Returns reset stats."""
    global _brain_instance
    if _brain_instance is None:
        return {"status": "no_instance"}
    result = _brain_instance.reset()
    return result
