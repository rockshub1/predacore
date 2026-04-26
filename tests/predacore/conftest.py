"""Shared fixtures for predacore memory subsystem tests.

Design principles for this suite (post-2026-04-26 memory upgrade):

1. **Per-test isolation**: every store gets a fresh tmp_path-backed SQLite
   DB. No shared state. No risk of one test's writes contaminating another's
   reads. Pytest's tmp_path fixture handles cleanup automatically.

2. **Embedder reuse**: BGE-small loads ~133MB into RAM and takes 1-2s on
   first call. Loading per-test would balloon the suite from ~1 min to
   ~30 min. The session-scoped `bge_embedder` fixture loads once and
   shares across ALL tests in the session.

3. **No real network in default runs**: `--real` flag (defined in root
   conftest.py) gates LLM-in-the-loop tests. Default `pytest` skips them.

4. **Healer disabled by default**: tests that don't explicitly need the
   background healer construct stores with the embedder + tmp DB only;
   no Healer thread spawned. Prevents cross-test thread interference.

5. **Global 60s timeout** (pyproject.toml): no test can hang indefinitely.
   The 10h e2e_smoke incident must not repeat.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any, Iterator

import pytest


# ─────────────────────────────────────────────────────────────────────
# Embedder — session-scoped (load once per pytest run)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def bge_embedder():
    """Real BGE-small embedder via predacore_core, loaded once per session.

    The Rust kernel's BGE model is process-global state — once loaded, it
    stays in RAM until the Python process exits. This fixture triggers
    that load on first use, then returns a thin async wrapper compatible
    with UnifiedMemoryStore's embedding_client interface.
    """
    try:
        import predacore_core
    except ImportError:
        pytest.skip("predacore_core wheel not available — install via maturin")

    # Warm the model on first use so subsequent test embed calls are hot.
    if not predacore_core.is_model_loaded():
        predacore_core.embed(["warmup"])

    class _SessionEmbedder:
        dim = 384

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return await asyncio.to_thread(predacore_core.embed, texts)

    return _SessionEmbedder()


# ─────────────────────────────────────────────────────────────────────
# UnifiedMemoryStore — function-scoped (fresh DB per test)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def memory_store(tmp_path: Path, bge_embedder):
    """Fresh UnifiedMemoryStore in a tmp dir, embedder shared across session.

    Each test gets its OWN SQLite DB so no cross-test contamination is
    possible. The embedder is shared because reloading BGE per test would
    balloon the suite.

    Healer is NOT started by this fixture (tests that need the daemon
    should use `memory_store_with_healer` instead).

    The store's `close()` is called automatically at test teardown.
    """
    from predacore.memory.store import UnifiedMemoryStore

    db_path = tmp_path / "test_memory.db"
    store = UnifiedMemoryStore(
        db_path=str(db_path),
        embedding_client=bge_embedder,
    )
    try:
        yield store
    finally:
        store.close()


@pytest.fixture
def memory_store_with_healer(tmp_path: Path, bge_embedder):
    """Like memory_store but ALSO spawns a Healer thread.

    Use only for tests that need to verify Healer behavior. Most tests
    should use the plain `memory_store` fixture to avoid the overhead +
    cross-test thread interference of running the daemon.
    """
    from predacore.memory.store import UnifiedMemoryStore
    from predacore.memory.healer import Healer

    db_path = tmp_path / "test_memory.db"
    store = UnifiedMemoryStore(
        db_path=str(db_path),
        embedding_client=bge_embedder,
    )
    healer = Healer(store, user="test-user")
    healer.start()
    try:
        yield store, healer
    finally:
        healer.stop()
        store.close()


# ─────────────────────────────────────────────────────────────────────
# Fast embedder — for tests that don't care about embedding quality
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def fast_embedder():
    """Deterministic hashing embedder for tests that don't need real BGE.

    ~1000× faster than BGE for tests that just need *some* vector to make
    the store happy (e.g. testing the schema, the recall mechanism shape,
    the healer's audit logic — none of which depend on BGE quality).

    Use `bge_embedder` when you're testing actual semantic recall ranking;
    use `fast_embedder` for everything else.
    """
    import hashlib
    import struct

    class _HashEmbedder:
        dim = 384

        async def embed(self, texts: list[str]) -> list[list[float]]:
            results = []
            for text in texts:
                # Hash → 384 deterministic floats in [-1, 1]
                vec = []
                seed = hashlib.sha256(text.encode("utf-8")).digest()
                # Tile the 32-byte hash to fill 384 floats
                for i in range(384):
                    byte = seed[i % 32]
                    vec.append((byte / 255.0) * 2.0 - 1.0)
                # Normalize so cosine similarity is well-defined
                norm = sum(v * v for v in vec) ** 0.5
                if norm > 0:
                    vec = [v / norm for v in vec]
                results.append(vec)
            return results

    return _HashEmbedder()


@pytest.fixture
def fast_memory_store(tmp_path: Path, fast_embedder):
    """Like memory_store but uses the fast hashing embedder. For tests
    where embedding QUALITY doesn't matter — only the storage / recall
    mechanism shape. ~1000× faster setup."""
    from predacore.memory.store import UnifiedMemoryStore

    db_path = tmp_path / "test_memory.db"
    store = UnifiedMemoryStore(
        db_path=str(db_path),
        embedding_client=fast_embedder,
    )
    try:
        yield store
    finally:
        store.close()


# ─────────────────────────────────────────────────────────────────────
# Git repo helper — for D13 / sync_git_changes tests
# ─────────────────────────────────────────────────────────────────────


class _GitRepo:
    """Lightweight handle for a git repo created by the `git_repo` fixture.

    `path` is the repo root; `_git(*args)` runs `git <args>` in that root
    and returns stdout. Callers can do ``(repo.path / "foo.py").write_text(...)``
    or ``repo._git("add", ".")`` etc.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=str(self.path),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def __truediv__(self, other: str) -> Path:
        """Convenience: ``repo / "file.py"`` returns the path inside the repo."""
        return self.path / other


@pytest.fixture
def git_repo(tmp_path: Path) -> _GitRepo:
    """Create an initialized empty git repo at tmp_path/repo and return a
    handle to it.

    Used by D13 / sync_git_changes tests that need a real git working
    tree to mutate. Configures user.email + user.name so commits work.

    Usage:
        def test_thing(git_repo):
            (git_repo / "foo.py").write_text("...")
            git_repo._git("add", ".")
            git_repo._git("commit", "-q", "-m", "init")
            head = git_repo._git("rev-parse", "HEAD")
    """
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    repo = _GitRepo(repo_path)
    repo._git("init", "-q", "-b", "main")
    repo._git("config", "user.email", "test@example.com")
    repo._git("config", "user.name", "Test User")
    return repo


# ─────────────────────────────────────────────────────────────────────
# Project-id detection — clear cache between tests
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_project_id_cache():
    """The project_id helper has a 60s in-process cache. Clear between
    every test so tests don't observe each other's cwd resolutions.
    Autouse = applies to ALL tests in this directory transparently."""
    try:
        from predacore.memory.project_id import clear_cache
        clear_cache()
        yield
        clear_cache()
    except ImportError:
        # Module not available (older predacore) — no-op
        yield


# ─────────────────────────────────────────────────────────────────────
# Env isolation — block PREDACORE_MEMORY_PROJECT bleed-through
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_predacore_env(monkeypatch: pytest.MonkeyPatch):
    """Strip PREDACORE_MEMORY_* env vars from the test process so a
    user's local config doesn't bleed into test outcomes. Tests that
    need a specific env value should set it explicitly via monkeypatch.
    """
    for key in list(os.environ.keys()):
        if key.startswith("PREDACORE_MEMORY_"):
            monkeypatch.delenv(key, raising=False)
    yield
