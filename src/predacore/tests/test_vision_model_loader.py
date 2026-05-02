"""Tests for the UGround model downloader (T4c).

We don't actually exercise huggingface_hub.snapshot_download in these tests
— that would require network + ~4GB of disk per run. Instead we verify
the contract and side effects via mocking + filesystem.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from predacore.operators import vision_model_loader as vml


def test_is_model_present_false_for_empty_dir(tmp_path: Path):
    assert vml.is_model_present(tmp_path) is False


def test_is_model_present_true_for_flat_layout(tmp_path: Path):
    """Caller-passed dir with files at the top level (custom dirs, tests)."""
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 10)
    assert vml.is_model_present(tmp_path) is True


def test_is_model_present_false_with_only_config(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")
    assert vml.is_model_present(tmp_path) is False


def test_is_model_present_accepts_pytorch_bin(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "pytorch_model.bin").write_bytes(b"\x00" * 10)
    assert vml.is_model_present(tmp_path) is True


def test_is_model_present_true_for_hf_cache_layout(tmp_path: Path):
    """HuggingFace cache layout: snapshots/<sha>/{config,weights}."""
    snap = tmp_path / "snapshots" / "abcdef1234567890"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"\x00" * 10)
    assert vml.is_model_present(tmp_path) is True


def test_is_model_present_false_for_empty_snapshots(tmp_path: Path):
    """snapshots/ exists but no commit dir inside → not present."""
    (tmp_path / "snapshots").mkdir()
    assert vml.is_model_present(tmp_path) is False


def test_default_model_dir_matches_bge_cache_root():
    """DEFAULT_MODEL_DIR must live under HF's hub/ root for one-cache UX."""
    p = str(vml.DEFAULT_MODEL_DIR)
    assert "huggingface" in p and "hub" in p
    assert p.endswith("models--Samsung--TinyClick")


def test_remove_model_no_op_for_missing_dir(tmp_path: Path):
    target = tmp_path / "missing"
    assert vml.remove_model(target) == 0


def test_remove_model_returns_bytes_freed(tmp_path: Path):
    payload = b"\x00" * 1024
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(payload)
    freed = vml.remove_model(tmp_path)
    # Non-zero (the exact size depends on FS overhead)
    assert freed >= 1024
    assert not tmp_path.exists()


def test_download_model_raises_when_hf_hub_missing(tmp_path: Path, monkeypatch):
    """If huggingface_hub isn't installed, surface a clear actionable error."""
    # Force the import inside download_model to fail
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError) as exc_info:
        vml.download_model(model_dir=tmp_path)
    msg = str(exc_info.value)
    assert "huggingface_hub" in msg
    assert "pip install" in msg


def test_download_model_calls_snapshot_with_repo(tmp_path: Path):
    """Happy path: hub is present; we invoke snapshot_download once."""
    fake_hub_path = tmp_path / "hub_target"
    fake_hub_path.mkdir()

    captured_kwargs: dict = {}

    def fake_snapshot(**kwargs):
        captured_kwargs.update(kwargs)
        return str(fake_hub_path)

    fake_module = type("M", (), {"snapshot_download": fake_snapshot})

    import sys
    sys.modules["huggingface_hub"] = fake_module
    try:
        out = vml.download_model(model_dir=tmp_path / "uground")
    finally:
        sys.modules.pop("huggingface_hub", None)

    assert out == fake_hub_path
    assert captured_kwargs["repo_id"] == vml.DEFAULT_MODEL_REPO
    assert captured_kwargs["local_dir"] == str(tmp_path / "uground")


def test_download_model_progress_callback(tmp_path: Path):
    """progress_cb is invoked at least once with a string description."""
    fake_hub_path = tmp_path / "out"
    fake_hub_path.mkdir()
    fake_module = type("M", (), {
        "snapshot_download": lambda **k: str(fake_hub_path),
    })
    import sys
    sys.modules["huggingface_hub"] = fake_module
    messages: list[str] = []
    try:
        vml.download_model(
            model_dir=tmp_path / "uground",
            progress_cb=messages.append,
        )
    finally:
        sys.modules.pop("huggingface_hub", None)
    assert any("Downloading" in m for m in messages)
    assert any("Downloaded" in m for m in messages)
