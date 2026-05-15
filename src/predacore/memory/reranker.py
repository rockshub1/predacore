"""
Cross-encoder reranker for memory recall.

The bi-encoder (BGE-small via Rust / predacore_core) does first-stage
retrieval — fast top-K candidate selection over the full corpus. The
cross-encoder rerunner here does the slow precise re-ordering of those
candidates: for each (query, doc) pair, the model produces a relevance
score that the bi-encoder's vector-similarity score can't capture.

Industry-standard two-stage retrieval. The reranker operates on the raw
TEXT of (query, doc) pairs — it does NOT consume the bi-encoder's
embeddings. That's why BGE-small as the embedder + Qwen3-Reranker-0.6B
as the reranker is a clean combination: different model families, but
the reranker only needs the document text and produces its own scoring.

Default model: ``Qwen/Qwen3-Reranker-0.6B`` (Apache 2.0, 32k context,
100+ languages including code, MTEB-R 65.80). The model loads lazily
via ``sentence_transformers.CrossEncoder`` on first ``predict()`` call;
subsequent calls reuse the loaded weights.

Idle-unload: the loaded model holds ~0.7-0.9 GB in MPS / CUDA memory
(fp16 weights — was ~1.2-1.5 GB before the fp16 swap in v1.6.12).
After ``idle_timeout_sec`` seconds without a predict() call (default
600 = 10 min via ``PREDACORE_RERANKER_IDLE_TIMEOUT_SEC``), a background
watcher thread drops the model reference and calls ``mps.empty_cache``
to release the GPU reservation. Next predict() reloads from the HF
cache (~8-10s with fp16 weights). Set the env var to 0 to disable
(always-loaded mode).

Wired in via:
  - ``UnifiedMemoryStore(reranker=Qwen3Reranker())`` at construction, OR
  - env flag ``PREDACORE_MEMORY_RERANKER=1`` + ``recall(rerank=True)``

Failures fail-open: if the model can't load, the reranker logs the error
and returns the candidates unchanged. The bi-encoder result stays
authoritative; reranking is enrichment, not a hard requirement.

Wave 12 (v1.6.0): ``sentence-transformers`` + ``torch`` + ``transformers``
moved into core ``dependencies`` so the reranker works out of the box
on a fresh install. The ``[reranker]`` extra is kept as a back-compat
no-op alias so existing install commands don't error.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


# Default model — Qwen3-Reranker-0.6B is the current SOTA small-model
# cross-encoder (MTEB-R 65.80, 32k context, 0.6B params, Apache 2.0).
# Confirmed latest in the Qwen reranker family as of 2026-05-15 — no
# Qwen3.5 / Qwen4 reranker released yet. Can be overridden via env or
# constructor for evaluation / A-B tests.
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"

# How many candidates we ask the bi-encoder for when reranking is on.
# More candidates = higher recall ceiling, linear latency cost in the
# reranker. 100 is the sweet spot for personal-use scale (~10-25k rows):
# pushes recall@10 from BGE's ~93% to ~98% at ~200ms total per query.
DEFAULT_RERANK_CANDIDATES = 100

# Idle-unload default: free the ~1.2-1.5GB MPS weights after this many
# seconds without a predict() call. Reload takes ~14s on warm cache.
# Tuned for personal-use: most conversations don't hit memory recall on
# every turn, so the reranker sits idle most of the time. 10 minutes
# is long enough that an active session stays loaded, short enough
# that overnight + during code work the RAM comes back. Set to 0 via
# PREDACORE_RERANKER_IDLE_TIMEOUT_SEC=0 to disable (always-loaded).
DEFAULT_IDLE_TIMEOUT_SEC = 600


class Qwen3Reranker:
    """Lazy-loading cross-encoder reranker.

    Holds a ``sentence_transformers.CrossEncoder`` instance that's loaded
    on first ``predict()`` call. Subsequent calls reuse the same model
    object — no re-init cost.

    Thread-safe: ``CrossEncoder.predict`` releases the GIL during model
    forward pass; the caller (``UnifiedMemoryStore.recall``) wraps the
    call in ``asyncio.to_thread`` to keep the event loop free.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        max_length: int = 512,
        idle_timeout_sec: int | None = None,
    ) -> None:
        # ``max_length`` is the per-pair token cap inside the cross-encoder.
        # Qwen3 supports 32k but most predacore memories are chunked at
        # ~1000 tokens already (MAX_CHUNK_CHARS=4000 / ~4 chars per token).
        # 512 covers >99% of stored content and keeps per-pair inference
        # fast. Bump if you start storing long-form summaries.
        self._model_name = model_name
        self._max_length = max_length
        self._model: Any = None  # CrossEncoder, lazy
        self._load_failed = False
        # Lazy-unload bookkeeping. Reading the env var here makes per-test
        # overrides cheap; pass an explicit int via constructor to bypass.
        if idle_timeout_sec is None:
            try:
                idle_timeout_sec = int(
                    os.getenv("PREDACORE_RERANKER_IDLE_TIMEOUT_SEC",
                              str(DEFAULT_IDLE_TIMEOUT_SEC))
                )
            except ValueError:
                idle_timeout_sec = DEFAULT_IDLE_TIMEOUT_SEC
        self._idle_timeout_sec = max(0, int(idle_timeout_sec))
        self._last_used_mono = 0.0
        self._lock = threading.Lock()
        self._unload_thread: threading.Thread | None = None

    def _ensure_loaded(self) -> bool:
        """Load the model on first use. Returns False on import/load failure.

        Also refreshes ``_last_used_mono`` (so the idle-unload watcher
        knows we're active) and starts the watcher thread on first
        successful load (one watcher per instance).
        """
        if self._model is not None:
            self._last_used_mono = time.monotonic()
            return True
        if self._load_failed:
            return False
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            # Wave 12: sentence-transformers is in core deps now. This
            # path should be unreachable on a fresh `pip install predacore`.
            # Surfacing instead of crashing covers source-install edge
            # cases (e.g. broken venv, deliberate stripping of torch).
            logger.warning(
                "Reranker disabled — sentence-transformers not installed in "
                "this venv. Run `pip install -U predacore` to repair (got: %s)",
                exc,
            )
            self._load_failed = True
            return False
        try:
            # Detect HF cache hit before logging so the message is honest.
            # Old wording said "downloads ~1.2GB on first run" on EVERY load,
            # including warm-cache reloads — confused users into thinking the
            # daemon was re-downloading on each restart. Now we check the
            # HF hub cache for the model directory + snapshot before logging.
            from pathlib import Path
            try:
                from huggingface_hub import constants as _hf_constants
                _hf_cache_root = Path(_hf_constants.HF_HUB_CACHE)
            except (ImportError, AttributeError):
                _hf_cache_root = Path.home() / ".cache" / "huggingface" / "hub"
            _model_cache_dir = (
                _hf_cache_root / ("models--" + self._model_name.replace("/", "--"))
            )
            _cached = (
                _model_cache_dir.exists()
                and (_model_cache_dir / "snapshots").exists()
                and any((_model_cache_dir / "snapshots").iterdir())
            )
            if _cached:
                logger.info("Loading reranker model %s (from local cache)",
                            self._model_name)
            else:
                logger.info("Downloading reranker model %s (first install — ~1.2GB; cached at %s)",
                            self._model_name, _hf_cache_root)
            # fp16 weights: ~50% RAM (1.5GB → ~0.8GB) and ~30-50% faster
            # forwards on MPS / CUDA. Cross-encoder ranking is empirically
            # robust to half precision — published reranker benchmarks
            # (BGE, MiniLM, Qwen) show <0.5pp MTEB-R delta vs fp32.
            # Passed as a string so we don't have to import torch at
            # module scope (torch import alone costs ~1-2s of daemon
            # startup). HF AutoModel.from_pretrained accepts string dtypes.
            self._model = CrossEncoder(
                self._model_name,
                max_length=self._max_length,
                trust_remote_code=False,
                model_kwargs={"torch_dtype": "float16"},
            )
            logger.info("Reranker %s loaded", self._model_name)
            self._last_used_mono = time.monotonic()
            self._maybe_start_unload_watcher()
            return True
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Reranker %s failed to load: %s — falling back to bi-encoder only",
                self._model_name, exc,
            )
            self._load_failed = True
            return False

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score (query, doc) text pairs. Returns a list of floats aligned with `pairs`.

        Higher score = more relevant. Sigmoid-applied so values are in [0, 1].
        Returns equal scores (1.0) if the model couldn't load — caller
        should treat that as "no reordering" rather than failing the recall.
        """
        if not pairs:
            return []
        if not self._ensure_loaded():
            # Fail-open: equal scores → bi-encoder order preserved on sort
            return [1.0] * len(pairs)
        try:
            # ``predict`` returns numpy array; convert to list for JSON-safe
            # downstream. Apply sigmoid so scores are bounded [0, 1] for
            # easier interpretation in logs / debug traces.
            import numpy as _np
            # batch_size=64: 2x the ST default of 32. On MPS kernel-launch
            # overhead dominates small batches, so one 64-pair forward
            # outperforms two 32-pair forwards. 64 is conservative — 100
            # would fit memory-wise, but 64 keeps headroom for max_length
            # outliers without OOM.
            # inference_mode: disables autograd tracking (saves ~10-20%
            # over default eval mode). Lazy torch import — only paid once
            # per process via Python's import cache.
            import torch as _torch
            with _torch.inference_mode():
                raw = self._model.predict(
                    pairs,
                    convert_to_numpy=True,
                    batch_size=64,
                )
            arr = _np.asarray(raw, dtype=_np.float32).reshape(-1)
            # Sigmoid: Qwen3-Reranker outputs raw logits by default.
            scores = 1.0 / (1.0 + _np.exp(-arr))
            # Refresh idle timer — watcher uses monotonic delta from this.
            self._last_used_mono = time.monotonic()
            return scores.tolist()
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("Reranker predict failed: %s — preserving bi-encoder order", exc)
            return [1.0] * len(pairs)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _unload(self) -> bool:
        """Drop the model + free GPU memory. Returns True if anything was freed.

        Two-step free: drop the Python reference (so GC can reclaim CPU
        objects + Torch tensors) AND call ``torch.mps.empty_cache()`` /
        ``torch.cuda.empty_cache()`` (so the GPU driver releases the MPS
        unified-memory / CUDA reservation). Without the explicit cache
        flush, MPS holds onto the freed weights indefinitely on Apple
        Silicon.
        """
        with self._lock:
            if self._model is None:
                return False
            logger.info(
                "Reranker %s unloading (idle > %ss) — freeing ~1.2-1.5GB MPS",
                self._model_name, self._idle_timeout_sec,
            )
            self._model = None
        # Force Python GC to drop the underlying torch tensors right now;
        # otherwise the MPS allocator won't see the references go away
        # until the next gen-2 collection (could be minutes).
        import gc as _gc
        _gc.collect()
        # Best-effort GPU-cache flush. Failures here are silent —
        # the unload still happened; the cache release is bonus.
        try:
            import torch as _torch
            if hasattr(_torch, "mps") and _torch.backends.mps.is_available():
                _torch.mps.empty_cache()
        except (ImportError, AttributeError, RuntimeError):
            pass
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except (ImportError, AttributeError, RuntimeError):
            pass
        return True

    def _maybe_start_unload_watcher(self) -> None:
        """Start the idle-unload watcher thread once per instance.

        Skipped entirely when ``idle_timeout_sec <= 0`` (always-loaded
        mode — set ``PREDACORE_RERANKER_IDLE_TIMEOUT_SEC=0`` to opt out).
        The watcher is a daemon thread, dies with the process; no
        explicit shutdown hook needed.
        """
        if self._unload_thread is not None:
            return
        if self._idle_timeout_sec <= 0:
            return
        # Poll every min(60s, timeout/4) — frequent enough to react
        # close to the deadline, infrequent enough that the watcher
        # itself is a no-op on RAM/CPU. For the default 600s timeout
        # that's a 60s poll, which is fine.
        poll_interval = max(15, min(60, self._idle_timeout_sec // 4))

        def _watch() -> None:
            while True:
                time.sleep(poll_interval)
                # Fast path: no model loaded means nothing to do; wait
                # for next predict() to reload and restart the cycle.
                if self._model is None:
                    continue
                idle = time.monotonic() - self._last_used_mono
                if idle >= self._idle_timeout_sec:
                    try:
                        self._unload()
                    except Exception as exc:  # noqa: BLE001 — watcher must never crash
                        logger.warning(
                            "Reranker unload failed: %s — keeping model loaded", exc,
                        )

        thread = threading.Thread(
            target=_watch,
            name=f"reranker-idle-unload-{self._model_name}",
            daemon=True,
        )
        thread.start()
        self._unload_thread = thread
        logger.debug(
            "Reranker idle-unload watcher started (timeout=%ss, poll=%ss)",
            self._idle_timeout_sec, poll_interval,
        )


def maybe_default_reranker() -> Qwen3Reranker | None:
    """Construct a reranker instance when ``PREDACORE_MEMORY_RERANKER=1``.

    Wave 12 (2026-05-11): default flipped to ``"1"`` (ON). The reranker
    fail-opens when ``sentence-transformers`` isn't installed — predict()
    returns ``[1.0] * N`` so bi-encoder order is preserved. Setting the
    default to ON means users who install ``predacore[reranker]`` get
    the recall boost automatically; users who don't have the extra
    installed see no difference. Set ``PREDACORE_MEMORY_RERANKER=0`` to
    explicitly disable even when the package is installed.

    Returns None when disabled so callers can pass it as ``reranker=``
    without branching at every site.
    """
    if os.getenv("PREDACORE_MEMORY_RERANKER", "1").strip().lower() in {"1", "true", "yes", "on"}:
        model_override = os.getenv("PREDACORE_MEMORY_RERANKER_MODEL", "").strip()
        if model_override:
            return Qwen3Reranker(model_name=model_override)
        return Qwen3Reranker()
    return None


__all__ = [
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_RERANK_CANDIDATES",
    "Qwen3Reranker",
    "maybe_default_reranker",
]
