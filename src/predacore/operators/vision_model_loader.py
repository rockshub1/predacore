"""TinyClick downloader for local UI grounding (T4c).

Picks the smallest Apache/MIT-licensed GUI-grounding model that's
benchmark-competitive in 2026 and downloads it from HuggingFace into
``~/.predacore/models/tinyclick/`` so the Rust inference path can load it.

Why TinyClick specifically (May 2026 research)
----------------------------------------------
- **MIT license** — commercial-safe (same family as Apache-2.0). The
  alternatives have baggage:
    * OmniParser v2's icon_detect is AGPL — viral; would force PredaCore
      itself to relicense from Apache-2.0.
    * UGround / Gelato / ShowUI are 2B-30B — 4GB+ on disk, 4-6GB RAM.
      Real friction for users on 8GB Macs.
- **0.27B parameters** — 7× smaller than 2B alternatives. Disk: ~540MB
  fp16, ~135MB at INT4. RAM: ~1GB fp16, ~270MB INT4. Fits in our
  bootstrap auto-download flow alongside BGE (133MB) without gating.
- **73.8% on ScreenSpot** — within 1.3pts of ShowUI-2B (75.1%) at 1/7 the
  size. Beats GPT-4V, SeeClick, OmniParser v2 (39.6%).
- **Sub-second latency** — designed for on-device; ANE/CoreML on Apple
  Silicon should land near 100-200ms.
- **Florence-2-base backbone** — Microsoft's official ONNX exports
  exist for Florence-2; the Rust ``ort`` path can reuse them.

This module ONLY downloads. Inference lives in the Rust ``ort``-based
module (predacore_core_crate/src/omniparser.rs, to be repurposed) — see
T4c task notes.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Default model + download target. Overridable via env if a user wants to
# pin a different revision or swap to a different small grounding model.
DEFAULT_MODEL_REPO = os.environ.get(
    "PREDACORE_VISION_MODEL_REPO", "Samsung/TinyClick",
)

# Match the BGE cache convention so users have ONE place to manage HF
# weights. ``hf_hub`` (used by the Rust BGE path) writes to
# ``~/.cache/huggingface/hub/models--<owner>--<repo>/`` by default; the
# Python ``huggingface_hub`` lib does the same. Letting both reuse this
# directory means ``huggingface-cli scan-cache`` shows everything we
# downloaded in one place.
def _default_model_dir(repo: str = DEFAULT_MODEL_REPO) -> Path:
    """HF cache path for a repo id like ``Samsung/TinyClick``."""
    cache_root = Path(
        os.environ.get(
            "HF_HOME", str(Path.home() / ".cache" / "huggingface"),
        )
    ).expanduser() / "hub"
    return cache_root / f"models--{repo.replace('/', '--')}"


DEFAULT_MODEL_DIR = _default_model_dir()

# Approximate download size in MB (for the prompt UX / progress message).
# TinyClick fp16 = ~540MB. Small enough that we don't strictly need an
# opt-in prompt, but OptInGate stays in place so users on metered
# connections can still opt out.
APPROX_DOWNLOAD_MB = 540


def is_model_present(model_dir: Path | str | None = None) -> bool:
    """Cheap check: does the model dir have at least the config + weights?

    Works for both layouts:
      * Flat dir (when caller passes a custom ``model_dir`` that's just a
        local folder of files) — config.json + weights at the top level.
      * HF cache layout (default) — files live under ``snapshots/<sha>/``.
        We recurse one level and accept any commit's snapshot.
    """
    p = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if not p.is_dir():
        return False
    # Flat layout
    if (p / "config.json").is_file():
        if any(p.glob("*.safetensors")) or any(p.glob("*.bin")):
            return True
    # HF cache layout: snapshots/<commit_sha>/{config.json, *.safetensors}
    snap_root = p / "snapshots"
    if snap_root.is_dir():
        for snap in snap_root.iterdir():
            if not snap.is_dir():
                continue
            if (snap / "config.json").is_file() and (
                any(snap.glob("*.safetensors")) or any(snap.glob("*.bin"))
            ):
                return True
    return False


def download_model(
    *,
    model_dir: Path | str | None = None,
    repo: str = DEFAULT_MODEL_REPO,
    progress_cb: Callable[[str], None] | None = None,
) -> Path:
    """Download the UGround weights into ``model_dir``. Returns the path.

    Uses ``huggingface_hub.snapshot_download`` if available — that's the
    standard way every transformers user pulls weights, handles resume
    + caching + Git LFS automatically. Falls back to a clear error if
    ``huggingface_hub`` isn't installed (it's lightweight; we don't bundle
    it since not every user wants the local-vision path).

    Raises:
        RuntimeError if huggingface_hub is missing or the download fails.
    """
    target = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    target.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub not installed. "
            "Run: pip install huggingface_hub  (or pip install predacore[vision])"
        ) from exc

    if progress_cb:
        progress_cb(
            f"Downloading {repo} (~{APPROX_DOWNLOAD_MB} MB) to {target}..."
        )
    try:
        downloaded = snapshot_download(
            repo_id=repo,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            # Skip git artifacts and example assets that bloat the disk.
            ignore_patterns=["*.md", "*.gif", "*.png", ".gitattributes"],
        )
    except Exception as exc:  # noqa: BLE001 — hub raises a wide bag
        raise RuntimeError(f"UGround download failed: {exc}") from exc
    if progress_cb:
        progress_cb(f"Downloaded UGround to {downloaded}")
    return Path(downloaded)


def remove_model(model_dir: Path | str | None = None) -> int:
    """Delete the model directory. Returns bytes freed (best-effort).

    Lets users reclaim disk if they decide they don't want local vision
    after all. Idempotent — no-op when the directory doesn't exist.
    """
    p = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if not p.exists():
        return 0
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    import shutil
    shutil.rmtree(p, ignore_errors=True)
    return total
