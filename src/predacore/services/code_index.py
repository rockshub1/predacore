"""
Semantic Code Search — Vector-indexed search over git-tracked files.

Indexes file paths + content signatures (imports, classes, functions, docstrings)
using local embeddings (GTE-small, 384-dim). Queries with natural language:
  "where's the rate limiting logic" → finds rate_limiter.py
  "telegram message handler" → finds channels/telegram.py

Hybrid scoring: cosine similarity (70%) + BM25 keyword (30%).

Features:
  1. Full-file chunk indexing (50-line blocks, not just first 150 lines)
  2. Git-aware incremental indexing (only re-embed changed files)
  3. Commit message indexing (semantic search over git history)
  4. Persistent index to disk (instant cold starts)
  5. AST-level extraction for Python (decorators, args, return types)
  6. Cross-file import graph (dependency tracking)
  7. Code change search (semantic search over diffs)
  8. Runtime data fusion (connect OutcomeStore errors to source files)
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import math
import os
import pickle
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# File extensions we index (source code + config)
_INDEXABLE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".kt",
    ".rb", ".sh", ".bash", ".zsh", ".yml", ".yaml", ".toml", ".json",
    ".md", ".txt", ".cfg", ".ini", ".env", ".sql", ".html", ".css",
    ".swift", ".m", ".c", ".cpp", ".h", ".hpp",
}

_MAX_FILE_BYTES = 100_000  # 100KB
_MAX_FILES = 5_000
_CHUNK_LINES = 50          # Lines per chunk for full-file indexing
_MAX_COMMITS = 500         # Max commit messages to index
_EMBED_BATCH_SIZE = 64     # Texts per batch for embedding calls
_COMMIT_FILES_LIMIT = 100  # Fetch changed-file lists for this many recent commits
_COMMIT_FILES_PER_COMMIT = 20  # Max changed files stored per commit
_HYBRID_VECTOR_WEIGHT = 0.7    # Cosine similarity weight in hybrid scoring
_HYBRID_KEYWORD_WEIGHT = 0.3   # BM25 keyword weight in hybrid scoring
_MIN_SEARCH_SCORE = 0.05       # Minimum hybrid score to include in results
_DIFF_STAT_TRUNCATE = 500      # Max chars of git diff --stat output
_SEARCH_CHANGES_TOP_K = 5      # Number of commits to fetch diffs for in change search


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class FileSignature:
    """Compact representation of a file for embedding."""
    path: str
    extension: str
    imports: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    docstring: str = ""
    first_lines: str = ""

    def to_text(self) -> str:
        parts = [f"File: {self.path}"]
        if self.imports:
            parts.append("Imports: " + ", ".join(self.imports[:20]))
        if self.classes:
            parts.append("Classes: " + ", ".join(self.classes))
        if self.functions:
            parts.append("Functions: " + ", ".join(self.functions[:30]))
        if self.decorators:
            parts.append("Decorators: " + ", ".join(self.decorators[:10]))
        if self.docstring:
            parts.append(f"Purpose: {self.docstring}")
        if self.first_lines:
            parts.append(self.first_lines)
        return "\n".join(parts)


@dataclass
class FileChunk:
    """A chunk of a file (50-line block) for granular search."""
    path: str
    chunk_index: int        # 0-based chunk number within the file
    start_line: int         # 1-based line number
    end_line: int
    content: str            # Raw text of the chunk
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        parts = [f"File: {self.path} (lines {self.start_line}-{self.end_line})"]
        if self.classes:
            parts.append("Classes: " + ", ".join(self.classes))
        if self.functions:
            parts.append("Functions: " + ", ".join(self.functions))
        parts.append(self.content[:600])
        return "\n".join(parts)


@dataclass
class CommitSignature:
    """A git commit for semantic search over history."""
    hash: str
    message: str
    author: str
    date: str
    files_changed: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        parts = [f"Commit: {self.message}"]
        if self.files_changed:
            parts.append("Files: " + ", ".join(self.files_changed[:20]))
        parts.append(f"Author: {self.author} | Date: {self.date}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# #5: AST-level extraction for Python
# ---------------------------------------------------------------------------


def _extract_ast_signature(path: str, content: str) -> FileSignature:
    """Extract rich signature from Python using AST (decorators, args, returns)."""
    ext = Path(path).suffix.lower()
    sig = FileSignature(path=path, extension=ext)

    try:
        tree = ast.parse(content, filename=path)
    except SyntaxError:
        return _extract_regex_signature(path, content)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                sig.imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                sig.imports.append(node.module)
        elif isinstance(node, ast.ClassDef):
            sig.classes.append(node.name)
            # Class docstring
            ds = ast.get_docstring(node)
            if ds and not sig.docstring:
                sig.docstring = ds[:300]
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Build rich function signature
            name = node.name
            if not name.startswith("_") or name.startswith("__"):
                # Extract decorators
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name):
                        sig.decorators.append(f"@{dec.id}")
                    elif isinstance(dec, ast.Attribute):
                        sig.decorators.append(f"@{ast.dump(dec)[:40]}")

                # Extract args
                args = []
                for arg in node.args.args:
                    arg_name = arg.arg
                    if arg.annotation:
                        try:
                            ann = ast.unparse(arg.annotation)
                            arg_name += f": {ann}"
                        except (ValueError, AttributeError):
                            pass
                    args.append(arg_name)

                # Extract return type
                ret = ""
                if node.returns:
                    try:
                        ret = f" -> {ast.unparse(node.returns)}"
                    except Exception:
                        pass

                func_sig = f"{name}({', '.join(args[:6])}){ret}"
                sig.functions.append(func_sig)

    # Module docstring
    ds = ast.get_docstring(tree)
    if ds:
        sig.docstring = ds[:300]

    # First meaningful lines
    lines = content.split("\n")[:50]
    meaningful = [s.strip() for s in lines if s.strip() and not s.strip().startswith("#")]
    sig.first_lines = "\n".join(meaningful[:5])

    return sig


def _extract_regex_signature(path: str, content: str) -> FileSignature:
    """Extract signature using regex (fallback for non-Python or syntax errors)."""
    ext = Path(path).suffix.lower()
    sig = FileSignature(path=path, extension=ext)
    lines = content.split("\n")

    for line in lines:
        stripped = line.strip()

        if ext == ".py":
            if stripped.startswith("import ") or stripped.startswith("from "):
                m = re.match(r"(?:from\s+([\w.]+)|import\s+([\w.]+))", stripped)
                if m:
                    sig.imports.append(m.group(1) or m.group(2))
            if stripped.startswith("class "):
                m = re.match(r"class\s+(\w+)", stripped)
                if m:
                    sig.classes.append(m.group(1))
            if stripped.startswith("def ") or stripped.startswith("async def "):
                m = re.match(r"(?:async\s+)?def\s+(\w+)", stripped)
                if m and not m.group(1).startswith("_"):
                    sig.functions.append(m.group(1))

        elif ext in (".js", ".ts", ".jsx", ".tsx"):
            if "import " in stripped:
                sig.imports.append(stripped[:80])
            m = re.match(r"(?:export\s+)?(?:class|function|const|let)\s+(\w+)", stripped)
            if m:
                sig.functions.append(m.group(1))

        elif ext == ".go":
            if stripped.startswith("func "):
                m = re.match(r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)", stripped)
                if m:
                    sig.functions.append(m.group(1))
            if stripped.startswith("type ") and "struct" in stripped:
                m = re.match(r"type\s+(\w+)", stripped)
                if m:
                    sig.classes.append(m.group(1))

    # Module docstring
    if ext == ".py":
        joined = "\n".join(lines[:30])
        m = re.search(r'"""(.*?)"""', joined, re.DOTALL)
        if not m:
            m = re.search(r"'''(.*?)'''", joined, re.DOTALL)
        if m:
            sig.docstring = m.group(1).strip()[:300]

    meaningful = [s.strip() for s in lines[:50] if s.strip() and not s.strip().startswith("#")]
    sig.first_lines = "\n".join(meaningful[:5])

    return sig


def _extract_signature(path: str, content: str) -> FileSignature:
    """Extract signature — AST for Python, regex for everything else."""
    if Path(path).suffix.lower() == ".py":
        return _extract_ast_signature(path, content)
    return _extract_regex_signature(path, content)


# ---------------------------------------------------------------------------
# #1: Full-file chunk extraction
# ---------------------------------------------------------------------------


def _extract_chunks(path: str, content: str) -> list[FileChunk]:
    """Split a file into 50-line chunks for granular search."""
    lines = content.split("\n")
    chunks: list[FileChunk] = []

    for i in range(0, len(lines), _CHUNK_LINES):
        chunk_lines = lines[i:i + _CHUNK_LINES]
        chunk_text = "\n".join(chunk_lines)
        if not chunk_text.strip():
            continue

        chunk = FileChunk(
            path=path,
            chunk_index=len(chunks),
            start_line=i + 1,
            end_line=min(i + _CHUNK_LINES, len(lines)),
            content=chunk_text,
        )

        # Extract functions/classes in this chunk
        ext = Path(path).suffix.lower()
        for line in chunk_lines:
            stripped = line.strip()
            if ext == ".py":
                m = re.match(r"(?:async\s+)?def\s+(\w+)", stripped)
                if m:
                    chunk.functions.append(m.group(1))
                m = re.match(r"class\s+(\w+)", stripped)
                if m:
                    chunk.classes.append(m.group(1))
            elif ext in (".js", ".ts", ".jsx", ".tsx"):
                m = re.match(r"(?:export\s+)?(?:class|function|const|let)\s+(\w+)", stripped)
                if m:
                    chunk.functions.append(m.group(1))
            elif ext == ".go":
                m = re.match(r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)", stripped)
                if m:
                    chunk.functions.append(m.group(1))

        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# #6: Cross-file import graph
# ---------------------------------------------------------------------------


class ImportGraph:
    """Tracks cross-file import relationships."""

    def __init__(self):
        # file -> list of modules it imports
        self._imports: dict[str, list[str]] = {}
        # file -> list of files that import it
        self._imported_by: dict[str, list[str]] = defaultdict(list)
        # module path -> file path mapping
        self._module_to_file: dict[str, str] = {}

    def build(self, signatures: list[FileSignature]) -> None:
        """Build import graph from file signatures."""
        self._imports.clear()
        self._imported_by.clear()
        self._module_to_file.clear()

        # Build module-to-file mapping
        for sig in signatures:
            if sig.extension == ".py":
                # Convert file path to module path
                module = sig.path.replace("/", ".").replace("\\", ".")
                if module.endswith(".py"):
                    module = module[:-3]
                if module.endswith(".__init__"):
                    module = module[:-9]
                self._module_to_file[module] = sig.path
                # Also register short name (last component)
                short = module.rsplit(".", 1)[-1]
                if short not in self._module_to_file:
                    self._module_to_file[short] = sig.path

        # Build edges
        for sig in signatures:
            self._imports[sig.path] = list(sig.imports)
            for imp in sig.imports:
                # Try to resolve import to a file in the repo
                target = self._resolve_import(imp)
                if target and target != sig.path:
                    self._imported_by[target].append(sig.path)

    def _resolve_import(self, module_path: str) -> str | None:
        """Resolve a module import to a file path."""
        # Try exact match
        if module_path in self._module_to_file:
            return self._module_to_file[module_path]
        # Try progressively shorter prefixes
        parts = module_path.split(".")
        for i in range(len(parts), 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in self._module_to_file:
                return self._module_to_file[prefix]
        return None

    def get_dependencies(self, file_path: str) -> list[str]:
        """What files does this file import? (outgoing edges)"""
        resolved = []
        for imp in self._imports.get(file_path, []):
            target = self._resolve_import(imp)
            if target:
                resolved.append(target)
        return sorted(set(resolved))

    def get_dependents(self, file_path: str) -> list[str]:
        """What files import this file? (incoming edges)"""
        return sorted(set(self._imported_by.get(file_path, [])))

    def get_blast_radius(self, file_path: str, max_depth: int = 3) -> dict[str, int]:
        """Get all files affected by changing this file, with depth.

        Returns {file_path: depth} where depth=1 means direct dependent.
        """
        visited: dict[str, int] = {}
        queue = [(file_path, 0)]
        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited[current] = depth
            if depth < max_depth:
                for dep in self.get_dependents(current):
                    if dep not in visited:
                        queue.append((dep, depth + 1))
        visited.pop(file_path, None)  # Don't include the file itself
        return visited

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_tracked": len(self._imports),
            "edges": sum(len(v) for v in self._imported_by.values()),
            "imports": dict(self._imports),
            "imported_by": dict(self._imported_by),
        }


# ---------------------------------------------------------------------------
# Rust Embedder Adapter (wraps predacore_core.embed for async interface)
# ---------------------------------------------------------------------------


class _RustEmbedder:
    """Wraps predacore_core.embed() with the same async interface as EmbeddingClient."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        import predacore_core
        return await asyncio.to_thread(predacore_core.embed, texts)


# ---------------------------------------------------------------------------
# Core: CodeIndex
# ---------------------------------------------------------------------------


class CodeIndex:
    """Semantic index over git-tracked source files.

    Features: file signatures, chunk-level search, commit indexing,
    import graph, incremental updates, disk persistence.
    """

    def __init__(self, repo_root: str | None = None, home_dir: str | None = None):
        self._repo_root = repo_root
        self._home_dir = home_dir or os.path.expanduser("~/.predacore")
        self._embedder: Any = None
        # File-level index
        self._vectors: list[list[float]] = []
        self._signatures: list[FileSignature] = []
        self._path_to_idx: dict[str, int] = {}
        # Chunk-level index (#1)
        self._chunk_vectors: list[list[float]] = []
        self._chunks: list[FileChunk] = []
        # Commit index (#3)
        self._commit_vectors: list[list[float]] = []
        self._commits: list[CommitSignature] = []
        # Import graph (#6)
        self._import_graph = ImportGraph()
        # Staleness tracking
        self._index_hash: str = ""
        self._file_hashes: dict[str, str] = {}  # path -> content hash (#2)
        self._last_index_time: float = 0
        self._commit_head: str = ""  # HEAD hash when commits were indexed
        self._lock = asyncio.Lock()
        self._ready = False

    # ── Git helpers ────────────────────────────────────────────────

    async def _get_repo_root(self) -> str:
        if self._repo_root:
            return self._repo_root
        proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "--show-toplevel",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError("Not a git repository")
        self._repo_root = out.decode().strip()
        return self._repo_root

    async def _get_tracked_files(self) -> list[str]:
        root = await self._get_repo_root()
        proc = await asyncio.create_subprocess_exec(
            "git", "ls-files",
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            return []
        return out.decode().splitlines()

    async def _get_changed_files(self) -> list[str]:
        """#2: Get files changed since last index (git diff)."""
        root = await self._get_repo_root()
        proc = await asyncio.create_subprocess_exec(
            "git", "diff", "--name-only", "HEAD",
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        changed = out.decode().splitlines() if proc.returncode == 0 else []
        # Also check unstaged/untracked
        proc2 = await asyncio.create_subprocess_exec(
            "git", "diff", "--name-only", "--cached",
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out2, _ = await proc2.communicate()
        if proc2.returncode == 0:
            changed.extend(out2.decode().splitlines())
        return list(set(changed))

    async def _get_head_hash(self) -> str:
        proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        return out.decode().strip() if proc.returncode == 0 else ""

    def _should_index(self, path: str) -> bool:
        ext = Path(path).suffix.lower()
        if ext not in _INDEXABLE_EXTENSIONS:
            return False
        skip_dirs = {
            "node_modules", ".git", "__pycache__", ".mypy_cache",
            "dist", "build", ".egg-info", "vendor", ".tox",
        }
        return not any(p in skip_dirs for p in Path(path).parts)

    async def _ensure_embedder(self):
        if self._embedder is not None:
            return
        # Single source of truth: the Rust kernel. Same BGE-small embeddings
        # as the memory store, so code-index vectors and memory vectors live
        # in the same semantic space (cross-querying is meaningful).
        # ``predacore_core`` is a hard dep of the memory subsystem already —
        # we don't carry a Python fallback here either.
        import predacore_core  # noqa: F401  — fail fast if Rust kernel missing
        self._embedder = _RustEmbedder()
        logger.info("Code index using Rust embedder (predacore_core)")

    async def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches."""
        await self._ensure_embedder()
        all_vecs: list[list[float]] = []
        batch_size = _EMBED_BATCH_SIZE
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vecs = await self._embedder.embed(batch)
            all_vecs.extend(vecs)
        return all_vecs

    # ── #4: Persistent index to disk ─────────────────────────────

    def _index_path(self) -> Path:
        return Path(self._home_dir) / "code_index.pkl"

    def save_to_disk(self) -> bool:
        """Save index to disk for instant cold starts."""
        if not self._ready:
            return False
        path = self._index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = {
                "version": 2,
                "repo_root": self._repo_root,
                "index_hash": self._index_hash,
                "file_hashes": self._file_hashes,
                "signatures": self._signatures,
                "vectors": self._vectors,
                "chunks": self._chunks,
                "chunk_vectors": self._chunk_vectors,
                "commits": self._commits,
                "commit_vectors": self._commit_vectors,
                "commit_head": self._commit_head,
                "import_graph_data": self._import_graph.to_dict(),
                "saved_at": time.time(),
            }
            data["version"] = 3  # Bump version — no more pickle
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"), default=str)
            logger.info("Code index saved to disk: %s (%.1fKB)", path, path.stat().st_size / 1024)
            return True
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to save code index: %s", exc)
            return False

    def load_from_disk(self) -> bool:
        """Load index from disk. Returns True if loaded successfully."""
        path = self._index_path()
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = json.load(f)
            if data.get("version") != 3:
                logger.info("Code index version mismatch, rebuilding")
                return False
            self._repo_root = data["repo_root"]
            self._index_hash = data["index_hash"]
            self._file_hashes = data.get("file_hashes", {})
            self._signatures = data["signatures"]
            self._vectors = data["vectors"]
            self._chunks = data.get("chunks", [])
            self._chunk_vectors = data.get("chunk_vectors", [])
            self._commits = data.get("commits", [])
            self._commit_vectors = data.get("commit_vectors", [])
            self._commit_head = data.get("commit_head", "")
            self._path_to_idx = {sig.path: idx for idx, sig in enumerate(self._signatures)}
            # Rebuild import graph from signatures
            self._import_graph = ImportGraph()
            self._import_graph.build(self._signatures)
            self._ready = True
            self._last_index_time = data.get("saved_at", time.monotonic())
            logger.info(
                "Code index loaded from disk: %d files, %d chunks, %d commits",
                len(self._signatures), len(self._chunks), len(self._commits),
            )
            return True
        except (OSError, pickle.UnpicklingError, ValueError, KeyError) as exc:
            logger.warning("Failed to load code index: %s", exc)
            return False

    # ── Build index ──────────────────────────────────────────────

    async def build_index(self, force: bool = False) -> int:
        """Index all git-tracked source files. Returns count of indexed files."""
        async with self._lock:
            # Try loading from disk first
            if not force and not self._ready and self.load_from_disk():
                return len(self._signatures)

            root = await self._get_repo_root()
            all_files = await self._get_tracked_files()
            indexable = [f for f in all_files if self._should_index(f)][:_MAX_FILES]

            file_list_hash = hashlib.md5(
                "\n".join(sorted(indexable)).encode(),
                usedforsecurity=False,
            ).hexdigest()

            if not force and file_list_hash == self._index_hash and self._ready:
                # #2: Check for content changes even if file list is the same
                changed = await self._get_changed_files()
                changed_indexable = [f for f in changed if f in self._path_to_idx]
                if not changed_indexable:
                    logger.debug("Code index up-to-date (%d files)", len(self._signatures))
                    return len(self._signatures)
                # Incremental update
                return await self._incremental_update(root, changed_indexable)

            logger.info("Building code index: %d files from %s", len(indexable), root)
            t0 = time.monotonic()

            # Extract signatures + chunks
            signatures: list[FileSignature] = []
            all_chunks: list[FileChunk] = []
            file_hashes: dict[str, str] = {}

            for rel_path in indexable:
                abs_path = os.path.join(root, rel_path)
                try:
                    size = os.path.getsize(abs_path)
                    if size > _MAX_FILE_BYTES or size == 0:
                        continue
                    with open(abs_path, errors="replace") as f:
                        content = f.read()
                    # Content hash for incremental updates (non-security)
                    file_hashes[rel_path] = hashlib.md5(
                        content.encode(), usedforsecurity=False,
                    ).hexdigest()
                    # File-level signature (#5: AST for Python)
                    sig = _extract_signature(rel_path, content)
                    signatures.append(sig)
                    # Chunk-level (#1)
                    chunks = _extract_chunks(rel_path, content)
                    all_chunks.extend(chunks)
                except (OSError, UnicodeDecodeError):
                    continue

            if not signatures:
                logger.warning("No indexable files found")
                return 0

            # Batch embed file signatures
            sig_texts = [sig.to_text() for sig in signatures]
            sig_vecs = await self._batch_embed(sig_texts)

            # Batch embed chunks
            chunk_texts = [ch.to_text() for ch in all_chunks]
            chunk_vecs = await self._batch_embed(chunk_texts) if chunk_texts else []

            # Store
            self._signatures = signatures
            self._vectors = sig_vecs
            self._chunks = all_chunks
            self._chunk_vectors = chunk_vecs
            self._path_to_idx = {sig.path: idx for idx, sig in enumerate(signatures)}
            self._file_hashes = file_hashes
            self._index_hash = file_list_hash
            self._last_index_time = time.monotonic()

            # Build import graph (#6)
            self._import_graph = ImportGraph()
            self._import_graph.build(signatures)

            self._ready = True

            elapsed = time.monotonic() - t0
            logger.info(
                "Code index built: %d files, %d chunks, %d vectors in %.1fs",
                len(signatures), len(all_chunks),
                len(sig_vecs) + len(chunk_vecs), elapsed,
            )

            # Index commits (#3)
            await self._index_commits()

            # Save to disk (#4)
            self.save_to_disk()

            return len(signatures)

    async def _incremental_update(self, root: str, changed_files: list[str]) -> int:
        """#2: Only re-embed changed files instead of full rebuild."""
        logger.info("Incremental index update: %d changed files", len(changed_files))
        t0 = time.monotonic()

        for rel_path in changed_files:
            abs_path = os.path.join(root, rel_path)
            idx = self._path_to_idx.get(rel_path)
            if idx is None:
                continue

            try:
                with open(abs_path, errors="replace") as f:
                    content = f.read()
            except (OSError, UnicodeDecodeError):
                continue

            new_hash = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()
            if self._file_hashes.get(rel_path) == new_hash:
                continue  # Content unchanged

            # Re-extract signature
            sig = _extract_signature(rel_path, content)
            self._signatures[idx] = sig
            self._file_hashes[rel_path] = new_hash

            # Re-embed
            vecs = await self._batch_embed([sig.to_text()])
            if vecs:
                self._vectors[idx] = vecs[0]

            # Re-extract and re-embed chunks for this file
            new_chunks = _extract_chunks(rel_path, content)
            new_chunk_texts = [ch.to_text() for ch in new_chunks]
            new_chunk_vecs = await self._batch_embed(new_chunk_texts) if new_chunk_texts else []

            # Filter out old chunks for this file, then append new ones.
            # This avoids index alignment bugs from popping mid-array.
            filtered_chunks = []
            filtered_vecs = []
            for i, ch in enumerate(self._chunks):
                if ch.path != rel_path:
                    filtered_chunks.append(ch)
                    if i < len(self._chunk_vectors):
                        filtered_vecs.append(self._chunk_vectors[i])
            self._chunks = filtered_chunks + list(new_chunks)
            self._chunk_vectors = filtered_vecs + list(new_chunk_vecs)

        # Rebuild import graph
        self._import_graph.build(self._signatures)

        elapsed = time.monotonic() - t0
        logger.info("Incremental update done in %.1fs", elapsed)

        # Save updated index
        self.save_to_disk()

        return len(self._signatures)

    # ── #3: Commit message indexing ──────────────────────────────

    async def _index_commits(self) -> int:
        """Index recent commit messages for semantic history search."""
        root = await self._get_repo_root()
        head = await self._get_head_hash()

        if head == self._commit_head and self._commits:
            return len(self._commits)

        proc = await asyncio.create_subprocess_exec(
            "git", "log", f"--max-count={_MAX_COMMITS}",
            "--format=%H||%s||%an||%ai",
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            return 0

        commits: list[CommitSignature] = []
        for line in out.decode().splitlines():
            parts = line.split("||", 3)
            if len(parts) < 4:
                continue
            commits.append(CommitSignature(
                hash=parts[0][:10],
                message=parts[1],
                author=parts[2],
                date=parts[3][:10],
            ))

        if not commits:
            return 0

        # Get changed files for each commit (batch for speed)
        for commit in commits[:_COMMIT_FILES_LIMIT]:
            proc = await asyncio.create_subprocess_exec(
                "git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit.hash,
                cwd=root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, _ = await proc.communicate()
            if proc.returncode == 0:
                commit.files_changed = out.decode().splitlines()[:_COMMIT_FILES_PER_COMMIT]

        # Embed
        texts = [c.to_text() for c in commits]
        vecs = await self._batch_embed(texts)

        self._commits = commits
        self._commit_vectors = vecs
        self._commit_head = head

        logger.info("Indexed %d commits", len(commits))
        return len(commits)

    # ── Search methods ───────────────────────────────────────────

    async def search(
        self,
        query: str,
        top_k: int = 10,
        file_pattern: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over indexed files (file-level)."""
        if not self._ready:
            count = await self.build_index()
            if count == 0:
                return []

        query_vec = (await self._batch_embed([query]))[0]
        scores = _cosine_scores(query_vec, self._vectors)
        bm25 = _bm25_scores(
            query.lower().split(),
            [sig.to_text().lower() for sig in self._signatures],
        )

        hybrid = []
        for idx in range(len(self._signatures)):
            vec_s = scores[idx] if idx < len(scores) else 0.0
            kw_s = bm25[idx] if idx < len(bm25) else 0.0
            hybrid.append((idx, _HYBRID_VECTOR_WEIGHT * vec_s + _HYBRID_KEYWORD_WEIGHT * kw_s, vec_s, kw_s))

        if file_pattern:
            pat = file_pattern.lower()
            hybrid = [
                (i, s, vs, ks) for i, s, vs, ks in hybrid
                if pat in self._signatures[i].path.lower()
                or Path(self._signatures[i].path).match(file_pattern)
            ]

        hybrid.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, combined, vec_s, kw_s in hybrid[:top_k]:
            if combined < _MIN_SEARCH_SCORE:
                continue
            sig = self._signatures[idx]
            results.append({
                "path": sig.path,
                "score": round(combined, 4),
                "vector_score": round(vec_s, 4),
                "keyword_score": round(kw_s, 4),
                "classes": sig.classes,
                "functions": sig.functions[:15],
                "docstring": sig.docstring[:200] if sig.docstring else "",
            })
        return results

    async def search_chunks(
        self,
        query: str,
        top_k: int = 10,
        file_pattern: str | None = None,
    ) -> list[dict[str, Any]]:
        """#1: Search at chunk level — returns exact code locations."""
        if not self._ready:
            await self.build_index()
        if not self._chunks:
            return []

        query_vec = (await self._batch_embed([query]))[0]
        scores = _cosine_scores(query_vec, self._chunk_vectors)
        bm25 = _bm25_scores(
            query.lower().split(),
            [ch.to_text().lower() for ch in self._chunks],
        )

        hybrid = []
        for idx in range(len(self._chunks)):
            vec_s = scores[idx] if idx < len(scores) else 0.0
            kw_s = bm25[idx] if idx < len(bm25) else 0.0
            hybrid.append((idx, _HYBRID_VECTOR_WEIGHT * vec_s + _HYBRID_KEYWORD_WEIGHT * kw_s, vec_s, kw_s))

        if file_pattern:
            pat = file_pattern.lower()
            hybrid = [
                (i, s, vs, ks) for i, s, vs, ks in hybrid
                if pat in self._chunks[i].path.lower()
            ]

        hybrid.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, combined, vec_s, kw_s in hybrid[:top_k]:
            if combined < _MIN_SEARCH_SCORE:
                continue
            ch = self._chunks[idx]
            results.append({
                "path": ch.path,
                "lines": f"{ch.start_line}-{ch.end_line}",
                "score": round(combined, 4),
                "functions": ch.functions,
                "classes": ch.classes,
                "preview": ch.content[:300],
            })
        return results

    async def search_commits(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """#3: Semantic search over commit messages."""
        if not self._commits:
            await self._index_commits()
        if not self._commits:
            return []

        query_vec = (await self._batch_embed([query]))[0]
        scores = _cosine_scores(query_vec, self._commit_vectors)
        bm25 = _bm25_scores(
            query.lower().split(),
            [c.to_text().lower() for c in self._commits],
        )

        hybrid = []
        for idx in range(len(self._commits)):
            vec_s = scores[idx] if idx < len(scores) else 0.0
            kw_s = bm25[idx] if idx < len(bm25) else 0.0
            hybrid.append((idx, _HYBRID_VECTOR_WEIGHT * vec_s + _HYBRID_KEYWORD_WEIGHT * kw_s, vec_s, kw_s))

        hybrid.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, combined, _, _ in hybrid[:top_k]:
            if combined < _MIN_SEARCH_SCORE:
                continue
            c = self._commits[idx]
            results.append({
                "hash": c.hash,
                "message": c.message,
                "author": c.author,
                "date": c.date,
                "files_changed": c.files_changed,
                "score": round(combined, 4),
            })
        return results

    async def search_changes(
        self,
        query: str,
        max_commits: int = 20,
    ) -> list[dict[str, Any]]:
        """#7: Search for code changes — combines commit search with diff context."""
        commits = await self.search_commits(query, top_k=max_commits)
        if not commits:
            return []

        root = await self._get_repo_root()
        results = []
        for commit in commits[:_SEARCH_CHANGES_TOP_K]:
            proc = await asyncio.create_subprocess_exec(
                "git", "diff", f"{commit['hash']}~1..{commit['hash']}",
                "--stat",
                cwd=root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, _ = await proc.communicate()
            diff_stat = out.decode()[:_DIFF_STAT_TRUNCATE] if proc.returncode == 0 else ""
            results.append({
                **commit,
                "diff_stat": diff_stat,
            })
        return results

    # ── #6: Import graph queries ─────────────────────────────────

    def get_dependencies(self, file_path: str) -> list[str]:
        """What does this file depend on?"""
        return self._import_graph.get_dependencies(file_path)

    def get_dependents(self, file_path: str) -> list[str]:
        """What files depend on this file?"""
        return self._import_graph.get_dependents(file_path)

    def get_blast_radius(self, file_path: str, max_depth: int = 3) -> dict[str, int]:
        """What's the impact of changing this file?"""
        return self._import_graph.get_blast_radius(file_path, max_depth)

    def get_import_graph_stats(self) -> dict[str, Any]:
        """Summary of the import graph."""
        graph = self._import_graph.to_dict()
        most_imported = sorted(
            graph["imported_by"].items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )[:10]
        return {
            "files_tracked": graph["files_tracked"],
            "total_edges": graph["edges"],
            "most_depended_on": [
                {"file": f, "dependents": len(deps)}
                for f, deps in most_imported
            ],
        }

    # ── #8: Runtime data fusion ──────────────────────────────────

    async def get_hot_spots(
        self,
        outcome_store: Any = None,
        window_hours: float = 72,
    ) -> list[dict[str, Any]]:
        """#8: Connect OutcomeStore errors to source files.

        Joins tool failure data with code index to find which source files
        are responsible for the most errors.
        """
        if outcome_store is None:
            return []

        try:
            failures = await outcome_store.get_failure_patterns(window_hours=window_hours)
        except (OSError, ValueError, KeyError):
            return []

        if not failures:
            return []

        # Map tool names to their handler files
        tool_to_file: dict[str, str] = {}
        for sig in self._signatures:
            for func in sig.functions:
                func_name = func.split("(")[0]  # Strip args from AST signatures
                if func_name.startswith("handle_"):
                    tool_name = func_name[7:]  # handle_desktop_control -> desktop_control
                    tool_to_file[tool_name] = sig.path

        hot_spots = []
        for pattern in failures:
            tool = pattern.get("tool_name", "")
            source_file = tool_to_file.get(tool)
            if source_file:
                dependents = self.get_dependents(source_file)
                hot_spots.append({
                    "tool": tool,
                    "source_file": source_file,
                    "failure_count": pattern.get("failure_count", 0),
                    "error_samples": pattern.get("error_samples", [])[:3],
                    "dependents": dependents[:5],
                    "blast_radius": len(self.get_blast_radius(source_file)),
                })

        hot_spots.sort(key=lambda x: x["failure_count"], reverse=True)
        return hot_spots

    # ── Properties ───────────────────────────────────────────────

    @property
    def file_count(self) -> int:
        return len(self._signatures)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def commit_count(self) -> int:
        return len(self._commits)

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def import_graph(self) -> ImportGraph:
        return self._import_graph

    def get_stats(self) -> dict[str, Any]:
        return {
            "files": len(self._signatures),
            "chunks": len(self._chunks),
            "commits": len(self._commits),
            "vectors": len(self._vectors) + len(self._chunk_vectors) + len(self._commit_vectors),
            "import_edges": self._import_graph.to_dict()["edges"],
            "ready": self._ready,
            "last_indexed": self._last_index_time,
        }


# ---------------------------------------------------------------------------
# Scoring helpers (shared)
# ---------------------------------------------------------------------------


def _cosine_scores(query_vec: list[float], vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    try:
        import numpy as np
        q = np.array(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        mat = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / (norms + 1e-9)
        return (mat @ q).tolist()
    except ImportError:
        q_norm = math.sqrt(sum(x * x for x in query_vec)) or 1e-9
        q = [x / q_norm for x in query_vec]
        results = []
        for vec in vectors:
            v_norm = math.sqrt(sum(x * x for x in vec)) or 1e-9
            dot = sum(a * b for a, b in zip(q, vec))
            results.append(dot / v_norm)
        return results


def _bm25_scores(query_terms: list[str], texts: list[str]) -> list[float]:
    if not query_terms or not texts:
        return [0.0] * len(texts)

    k1, b = 1.5, 0.75
    n_docs = len(texts)
    doc_lens = [len(t.split()) for t in texts]
    avg_dl = sum(doc_lens) / max(n_docs, 1)

    df: dict[str, int] = {}
    for term in query_terms:
        df[term] = sum(1 for t in texts if term in t)

    scores = []
    for i, text in enumerate(texts):
        score = 0.0
        dl = doc_lens[i]
        words = text.split()
        for term in query_terms:
            tf = sum(1 for w in words if term in w)
            if tf > 0:
                idf = math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
        scores.append(score)

    max_s = max(scores) if scores else 1.0
    return [s / max_s for s in scores] if max_s > 0 else scores


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_index: CodeIndex | None = None
_global_lock = asyncio.Lock()


async def get_code_index(repo_root: str | None = None) -> CodeIndex:
    """Get or create the global code index singleton."""
    global _global_index
    async with _global_lock:
        if _global_index is None:
            _global_index = CodeIndex(repo_root=repo_root)
        return _global_index


async def semantic_code_search(
    query: str,
    top_k: int = 10,
    file_pattern: str | None = None,
    rebuild: bool = False,
    search_type: str = "files",
) -> str:
    """High-level search function for the tool handler.

    search_type: "files" (default), "chunks", "commits", "changes",
                 "dependencies", "dependents", "blast_radius", "hot_spots"
    """
    index = await get_code_index()

    if rebuild or not index.is_ready:
        count = await index.build_index(force=rebuild)
        if count == 0:
            return "No indexable source files found in this git repository."

    # Route to the right search method
    if search_type == "chunks":
        results = await index.search_chunks(query, top_k=top_k, file_pattern=file_pattern)
        if not results:
            return f"No chunk results for '{query}'."
        lines = [f"Found {len(results)} code chunks for '{query}':", ""]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. **{r['path']}** lines {r['lines']} (score: {r['score']})")
            if r["functions"]:
                lines.append(f"   Functions: {', '.join(r['functions'])}")
            if r["classes"]:
                lines.append(f"   Classes: {', '.join(r['classes'])}")
            lines.append(f"   ```\n{r['preview']}\n   ```")
            lines.append("")
        return "\n".join(lines)

    elif search_type == "commits":
        results = await index.search_commits(query, top_k=top_k)
        if not results:
            return f"No commits matching '{query}'."
        lines = [f"Found {len(results)} commits for '{query}':", ""]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. `{r['hash']}` — {r['message']} (score: {r['score']})")
            lines.append(f"   By {r['author']} on {r['date']}")
            if r["files_changed"]:
                lines.append(f"   Files: {', '.join(r['files_changed'][:10])}")
            lines.append("")
        return "\n".join(lines)

    elif search_type == "changes":
        results = await index.search_changes(query)
        if not results:
            return f"No code changes matching '{query}'."
        lines = [f"Found {len(results)} relevant changes for '{query}':", ""]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. `{r['hash']}` — {r['message']}")
            if r.get("diff_stat"):
                lines.append(f"   {r['diff_stat'][:200]}")
            lines.append("")
        return "\n".join(lines)

    elif search_type == "dependencies":
        deps = index.get_dependencies(query)  # query = file path
        if not deps:
            return f"No dependencies found for '{query}'."
        lines = [f"**{query}** depends on ({len(deps)} files):", ""]
        for d in deps:
            lines.append(f"  - {d}")
        return "\n".join(lines)

    elif search_type == "dependents":
        deps = index.get_dependents(query)  # query = file path
        if not deps:
            return f"No files depend on '{query}'."
        lines = [f"**{query}** is imported by ({len(deps)} files):", ""]
        for d in deps:
            lines.append(f"  - {d}")
        return "\n".join(lines)

    elif search_type == "blast_radius":
        radius = index.get_blast_radius(query)  # query = file path
        if not radius:
            return f"No downstream impact detected for '{query}'."
        lines = [f"Blast radius for **{query}** ({len(radius)} affected files):", ""]
        for f, depth in sorted(radius.items(), key=lambda x: x[1]):
            indent = "  " * depth
            lines.append(f"{indent}{'→ ' * depth}{f} (depth {depth})")
        return "\n".join(lines)

    elif search_type == "hot_spots":
        spots = await index.get_hot_spots()
        if not spots:
            return "No hot spots detected (no recent failures in OutcomeStore)."
        lines = ["Error hot spots:", ""]
        for s in spots:
            lines.append(f"- **{s['tool']}** → `{s['source_file']}` ({s['failure_count']} failures)")
            if s["error_samples"]:
                lines.append(f"  Errors: {s['error_samples'][0][:100]}")
            if s["dependents"]:
                lines.append(f"  Dependents: {', '.join(s['dependents'])}")
            lines.append("")
        return "\n".join(lines)

    else:
        # Default: file-level search
        results = await index.search(query, top_k=top_k, file_pattern=file_pattern)
        if not results:
            return f"No results for '{query}'."
        lines = [f"Found {len(results)} results for '{query}':", ""]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. **{r['path']}** (score: {r['score']})")
            if r["classes"]:
                lines.append(f"   Classes: {', '.join(r['classes'])}")
            if r["functions"]:
                fns = r["functions"]
                if len(fns) > 8:
                    fns = fns[:8] + [f"...+{len(fns) - 8} more"]
                lines.append(f"   Functions: {', '.join(fns)}")
            if r["docstring"]:
                lines.append(f"   Purpose: {r['docstring']}")
            lines.append("")
        lines.append(f"Index: {index.file_count} files, {index.chunk_count} chunks, "
                     f"{index.commit_count} commits")
        return "\n".join(lines)
