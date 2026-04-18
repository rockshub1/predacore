"""
Concrete vector index for the Knowledge Nexus.

Implements AbstractVectorIndex using numpy-based cosine similarity.
No external FAISS dependency — pure-Python with optional disk persistence.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

from .vector_index import AbstractVectorIndex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------


class EmbeddingProvider:
    """Abstract embedding provider."""

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    @property
    def dimensions(self) -> int:
        raise NotImplementedError


class HashEmbedding(EmbeddingProvider):
    """
    Zero-cost deterministic embedding using character-level hashing.

    Produces a fixed-size vector by distributing character hash values
    across bins and L2-normalizing. Good enough for structural similarity;
    not a semantic model — use an LLM embedding provider for that.
    """

    def __init__(self, dims: int = 128) -> None:
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dims
        for i, ch in enumerate(text.lower()):
            idx = (hash(ch) ^ (i * 2654435761)) % self._dims
            vec[idx] += 1.0
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


# ---------------------------------------------------------------------------
# Vector store entry
# ---------------------------------------------------------------------------


class _VectorEntry:
    __slots__ = ("item_id", "embedding", "metadata", "layers")

    def __init__(
        self,
        item_id: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> None:
        self.item_id = item_id
        self.embedding = embedding
        self.metadata = metadata or {}
        self.layers: set[str] = set(self.metadata.get("layers", []))


# ---------------------------------------------------------------------------
# FAISS-compatible vector index
# ---------------------------------------------------------------------------


class FAISSVectorIndex(AbstractVectorIndex):
    """
    In-memory vector index with cosine similarity search.

    Provides the same interface as a FAISS flat index but implemented
    in pure Python (numpy-free) for zero-dependency portability.
    Supports layer-based filtering, metadata storage, and disk persistence.
    """

    def __init__(
        self,
        dimensions: int = 128,
        log: logging.Logger | None = None,
    ) -> None:
        self._dims = dimensions
        self._store: dict[str, _VectorEntry] = {}
        self._log = log or logger

    # -- Core CRUD ----------------------------------------------------------

    async def add(
        self,
        item_id: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> bool:
        if len(embedding) != self._dims:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dims}, got {len(embedding)}"
            )
        self._store[item_id] = _VectorEntry(item_id, embedding, metadata)
        self._log.debug("VectorIndex: added %s (total=%d)", item_id, len(self._store))
        return True

    async def remove(self, item_id: str) -> bool:
        if item_id in self._store:
            del self._store[item_id]
            return True
        return False

    async def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        layers: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return (item_id, cosine_similarity) pairs, highest first."""
        if len(query_embedding) != self._dims:
            raise ValueError(
                f"Query dimension mismatch: expected {self._dims}, got {len(query_embedding)}"
            )

        results: list[tuple[str, float]] = []
        for entry in self._store.values():
            # Layer filter
            if layers and not entry.layers.intersection(layers):
                continue
            sim = self._cosine_similarity(query_embedding, entry.embedding)
            results.append((entry.item_id, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # -- Persistence --------------------------------------------------------

    async def save_to_disk(self, path: str) -> None:
        """Serialize the entire index to a JSON file."""
        data = {
            "dimensions": self._dims,
            "entries": [
                {
                    "item_id": e.item_id,
                    "embedding": e.embedding,
                    "metadata": e.metadata,
                }
                for e in self._store.values()
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        self._log.info("VectorIndex: saved %d entries to %s", len(self._store), path)

    @classmethod
    async def load_from_disk(
        cls,
        path: str,
        log: logging.Logger | None = None,
    ) -> FAISSVectorIndex:
        """Load an index from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        idx = cls(dimensions=data["dimensions"], log=log)
        for entry in data["entries"]:
            await idx.add(entry["item_id"], entry["embedding"], entry.get("metadata"))
        return idx

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (norm_a * norm_b)

    # -- Inspection ---------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def dimensions(self) -> int:
        return self._dims

    def get_metadata(self, item_id: str) -> dict | None:
        entry = self._store.get(item_id)
        return entry.metadata if entry else None

    def __repr__(self) -> str:
        return f"FAISSVectorIndex(dims={self._dims}, entries={len(self._store)})"
