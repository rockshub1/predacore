"""
Tests for ``predacore.bootstrap_prefetch`` — first-time-setup detection
and the game-style single progress bar.

No real HuggingFace downloads happen in these tests. We monkeypatch the
HF cache dir to a tmp path and verify the detection logic + log-only
fallback path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from predacore.bootstrap_prefetch import (
    _ESSENTIAL_REPOS,
    _ESTIMATED_TOTAL_BYTES,
    _is_repo_cached,
    _repo_cache_dir,
    needs_first_time_setup,
)


class TestRepoCacheDetection:
    """``_is_repo_cached`` follows the HF cache layout: a repo is
    "cached" if any non-empty snapshot directory exists under
    ``<root>/models--<org>--<name>/snapshots/<sha>/``.
    """

    def test_uncached_repo_returns_false(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        assert _is_repo_cached("Some/UncachedRepo") is False

    def test_empty_cache_dir_returns_false(self, tmp_path, monkeypatch) -> None:
        """Cache dir present but no snapshots → not cached."""
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        # Create the repo dir but leave it empty
        repo_dir = _repo_cache_dir("Some/Repo")
        repo_dir.mkdir(parents=True)
        assert _is_repo_cached("Some/Repo") is False

    def test_snapshots_dir_without_files_returns_false(
        self, tmp_path, monkeypatch
    ) -> None:
        """Snapshots dir exists but the snapshot inside is empty."""
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        repo_dir = _repo_cache_dir("Some/Repo")
        (repo_dir / "snapshots" / "abc123").mkdir(parents=True)
        assert _is_repo_cached("Some/Repo") is False

    def test_cached_repo_with_snapshot_file_returns_true(
        self, tmp_path, monkeypatch
    ) -> None:
        """Real cache state: snapshots/<sha>/<filename> present."""
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        snap_dir = _repo_cache_dir("Some/Repo") / "snapshots" / "abc123"
        snap_dir.mkdir(parents=True)
        (snap_dir / "config.json").write_text("{}")
        assert _is_repo_cached("Some/Repo") is True


class TestFirstTimeSetupDetection:
    """``needs_first_time_setup`` returns True when ANY essential repo
    isn't cached. The set itself is small (BGE + Qwen3-Reranker) — we
    don't include TinyClick or Chromium because those are lazy-loaded
    only when their specific code paths fire."""

    def test_essentials_includes_memory_embedder(self) -> None:
        """BGE-small must be in the essential set — every recall needs it."""
        assert "BAAI/bge-small-en-v1.5" in _ESSENTIAL_REPOS

    def test_essentials_includes_recall_reranker(self) -> None:
        """Qwen3-Reranker is in essentials — reranker defaults ON in v1.6.0."""
        assert "Qwen/Qwen3-Reranker-0.6B" in _ESSENTIAL_REPOS

    def test_essentials_excludes_tinyclick(self) -> None:
        """TinyClick is canvas-vision niche — stays lazy. Most users
        never hit a Figma/Miro page so we don't burden them with the 540MB
        download up front."""
        assert not any("TinyClick" in r for r in _ESSENTIAL_REPOS)

    def test_returns_true_when_no_cache(self, tmp_path, monkeypatch) -> None:
        """Fresh machine, no cache → needs setup."""
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        assert needs_first_time_setup() is True

    def test_returns_false_when_all_cached(self, tmp_path, monkeypatch) -> None:
        """Both essentials present → no setup needed."""
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        for repo_id in _ESSENTIAL_REPOS:
            snap_dir = _repo_cache_dir(repo_id) / "snapshots" / "fakesha"
            snap_dir.mkdir(parents=True)
            (snap_dir / "config.json").write_text("{}")
        assert needs_first_time_setup() is False

    def test_returns_true_when_one_missing(self, tmp_path, monkeypatch) -> None:
        """Even one missing essential → setup needed."""
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        first_repo = next(iter(_ESSENTIAL_REPOS))
        snap_dir = _repo_cache_dir(first_repo) / "snapshots" / "fakesha"
        snap_dir.mkdir(parents=True)
        (snap_dir / "config.json").write_text("{}")
        # The OTHER repo is still missing
        assert needs_first_time_setup() is True


class TestEstimateBytes:
    """The progress bar uses an estimated total. Verify it's in a
    sensible range — not zero (would divide by zero in progress calc),
    not absurdly large."""

    def test_estimate_is_reasonable(self) -> None:
        # Should be roughly 1-2GB given the essential set (BGE 133MB + Qwen3 1.2GB)
        assert 1_000_000_000 <= _ESTIMATED_TOTAL_BYTES <= 3_000_000_000


class TestRunFirstTimeSetup:
    """The setup function returns True quickly when no downloads are
    needed. We don't run real downloads in tests."""

    def test_no_op_when_cache_complete(self, tmp_path, monkeypatch) -> None:
        """All cached → returns True without attempting download."""
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        for repo_id in _ESSENTIAL_REPOS:
            snap_dir = _repo_cache_dir(repo_id) / "snapshots" / "fakesha"
            snap_dir.mkdir(parents=True)
            (snap_dir / "config.json").write_text("{}")
        from predacore.bootstrap_prefetch import run_first_time_setup
        result = run_first_time_setup()
        assert result is True
