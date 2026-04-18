from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .embedding import InMemoryVectorIndex

_logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    _logger.debug("FAISS not installed - using simple vector index")


class DiskBackedVectorIndex:
    """
    Simple JSONL-based persistence around an in-memory vector index.

    File format per namespace: data/vector_index/{namespace}.jsonl
    Each line: {"id": str, "vector": [float], "meta": {}}
    """

    def __init__(self, base_dir: str = "data/vector_index"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._indices: dict[str, InMemoryVectorIndex] = {}

    def _ns_path(self, namespace: str) -> Path:
        return self.base_dir / f"{namespace}.jsonl"

    def _ensure_ns(self, namespace: str):
        if namespace not in self._indices:
            self._indices[namespace] = InMemoryVectorIndex()
            # Load existing items from disk if present
            p = self._ns_path(namespace)
            if p.exists():
                try:
                    with p.open("r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                row = json.loads(line)
                                iid = row.get("id")
                                vec = row.get("vector")
                                meta = row.get("meta") or {}
                                if isinstance(iid, str) and isinstance(vec, list):
                                    self._indices[namespace].add(iid, vec, meta)
                            except json.JSONDecodeError as e:
                                _logger.warning(f"Skipping malformed line in {p}: {e}")
                                continue
                except OSError as e:
                    _logger.error(f"Failed to load vector index from {p}: {e}")

    def add(
        self,
        namespace: str,
        item_id: str,
        vector: list[float],
        meta: dict[str, Any] | None = None,
    ):
        self._ensure_ns(namespace)
        self._indices[namespace].add(item_id, vector, meta or {})
        # Append to disk
        p = self._ns_path(namespace)
        try:
            with p.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"id": item_id, "vector": vector, "meta": meta or {}},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except OSError as e:
            _logger.warning(
                f"Failed to persist vector to {p} (data in memory only): {e}"
            )

    def search(
        self, namespace: str, query_vec: list[float], top_k: int = 5
    ) -> list[tuple[str, float, dict[str, Any]]]:
        self._ensure_ns(namespace)
        return self._indices[namespace].search(query_vec, top_k)


class FaissVectorIndex:
    """
    Optional FAISS-backed index with JSONL metadata. Requires faiss installed.
    """

    def __init__(self, base_dir: str = "data/vector_index"):
        if not _FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available")
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # namespace -> (index, ids, metas)
        self._indices: dict[str, Any] = {}
        self._ids: dict[str, list[str]] = {}
        self._metas: dict[str, list[dict[str, Any]]] = {}

    def _ns_paths(self, namespace: str) -> tuple[Path, Path]:
        return (
            self.base_dir / f"{namespace}.faiss",
            self.base_dir / f"{namespace}.meta.jsonl",
        )

    def _ensure_ns(self, namespace: str):
        if namespace in self._indices:
            return
        idx_path, meta_path = self._ns_paths(namespace)
        self._ids[namespace] = []
        self._metas[namespace] = []
        if idx_path.exists() and meta_path.exists():
            try:
                index = faiss.read_index(str(idx_path))
                self._indices[namespace] = index
                with meta_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                            self._ids[namespace].append(row.get("id"))
                            self._metas[namespace].append(row.get("meta") or {})
                        except json.JSONDecodeError as e:
                            _logger.warning(
                                f"Malformed metadata line in {meta_path}: {e}"
                            )
                            continue
                return
            except Exception as e:
                _logger.error(f"Failed to load FAISS index from {idx_path}: {e}")
        # else init empty index with dimension inferred later
        self._indices[namespace] = None

    def add(
        self,
        namespace: str,
        item_id: str,
        vector: list[float],
        meta: dict[str, Any] | None = None,
    ):
        self._ensure_ns(namespace)
        vec = vector
        d = len(vec)
        if self._indices[namespace] is None:
            # Create index
            index = faiss.IndexFlatIP(d)
            self._indices[namespace] = index
        else:
            index = self._indices[namespace]
            # Sanity check dimension
            if index.d != d:
                _logger.warning(
                    "Skipping vector with dimension %d (index expects %d) for item %s",
                    d, index.d, item_id if 'item_id' in dir() else "unknown",
                )
                return
        import numpy as np

        x = np.array([vec], dtype="float32")
        # Normalize for cosine similarity (using inner product)
        nrm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        x = x / nrm
        index.add(x)
        self._ids[namespace].append(item_id)
        self._metas[namespace].append(meta or {})
        idx_path, meta_path = self._ns_paths(namespace)
        try:
            faiss.write_index(index, str(idx_path))
        except Exception as e:
            _logger.warning(f"Failed to persist FAISS index to {idx_path}: {e}")
        try:
            with meta_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps({"id": item_id, "meta": meta or {}}, ensure_ascii=False)
                    + "\n"
                )
        except OSError as e:
            _logger.warning(f"Failed to persist metadata to {meta_path}: {e}")

    def search(
        self, namespace: str, query_vec: list[float], top_k: int = 5
    ) -> list[tuple[str, float, dict[str, Any]]]:
        self._ensure_ns(namespace)
        index = self._indices[namespace]
        if index is None:
            return []
        import numpy as np

        q = np.array([query_vec], dtype="float32")
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        scores, idxs = index.search(q, top_k)
        out = []
        for pos, score in enumerate(scores[0]):
            i = int(idxs[0][pos])
            if i < 0 or i >= len(self._ids[namespace]):
                continue
            out.append(
                (self._ids[namespace][i], float(score), self._metas[namespace][i])
            )
        return out
