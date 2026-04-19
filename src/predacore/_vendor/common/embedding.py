"""
Embedding client abstraction and a simple in-memory vector index.

Uses provider-agnostic env config; falls back to a hashing embedding when
no provider is configured.
"""
from __future__ import annotations

import logging
import math
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingClient:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://api.openai.com").rstrip("/")
        self.endpoint = f"{self.base_url}/v1/embeddings"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self.endpoint,
                json={"model": self.model, "input": texts},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]


class GeminiEmbeddingClient(EmbeddingClient):
    """Google Gemini embedding client using the Generative Language API."""

    def __init__(
        self,
        api_key: str,
        model: str = "models/gemini-embedding-001",
        base_url: str | None = None,
    ):
        self.api_key = api_key
        self.model = model if model.startswith("models/") else f"models/{model}"
        self.base_url = (
            base_url or "https://generativelanguage.googleapis.com"
        ).rstrip("/")
        self.single_endpoint = f"{self.base_url}/v1beta/{self.model}:embedContent"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        if not texts:
            return []

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        def _extract_embedding(payload: dict[str, Any]) -> list[float] | None:
            if isinstance(payload.get("values"), list):
                return [float(v) for v in payload["values"]]
            emb = payload.get("embedding")
            if isinstance(emb, dict) and isinstance(emb.get("values"), list):
                return [float(v) for v in emb["values"]]
            if isinstance(emb, list):
                return [float(v) for v in emb]
            return None

        async with httpx.AsyncClient(timeout=60) as client:
            vectors = []
            for text in texts:
                resp = await client.post(
                    self.single_endpoint,
                    headers=headers,
                    json={
                        "model": self.model,
                        "content": {"parts": [{"text": text}]},
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                vec = _extract_embedding(data if isinstance(data, dict) else {})
                if not vec:
                    raise RuntimeError("Unexpected Gemini embedding response format")
                vectors.append(vec)
            return vectors


class ZhiPuEmbeddingClient(EmbeddingClient):
    """ZhiPu BigModel embedding client (OpenAI-compatible API)."""

    def __init__(
        self,
        api_key: str,
        model: str = "embedding-3",
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/embeddings"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self.endpoint,
                json={"model": self.model, "input": texts},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]


class LocalEmbeddingClient(EmbeddingClient):
    """
    Local semantic embedding using sentence-transformers models via HuggingFace.

    Runs entirely offline — no API key needed. Uses GTE-small (384-dim)
    by default: fast, lightweight (~67MB), and produces real semantic vectors.
    GTE-small scores ~61.4 on MTEB vs MiniLM's ~56.3 — same dim, better quality.
    """

    _model_cache: dict[str, Any] = {}
    _model_lock = threading.Lock()

    def __init__(self, model_name: str = "thenlper/gte-small"):
        self.model_name = model_name
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        with LocalEmbeddingClient._model_lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return
            if self.model_name in LocalEmbeddingClient._model_cache:
                cached = LocalEmbeddingClient._model_cache[self.model_name]
                self._tokenizer = cached["tokenizer"]
                self._model = cached["model"]
                return
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            LocalEmbeddingClient._model_cache[self.model_name] = {
                "tokenizer": self._tokenizer,
                "model": self._model,
            }
            logger.info("LocalEmbeddingClient: loaded %s", self.model_name)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        import torch

        self._load()
        encoded = self._tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self._model(**encoded)
            mask = encoded["attention_mask"]
            token_emb = outputs.last_hidden_state
            mask_expanded = mask.unsqueeze(-1).expand(token_emb.size()).float()
            summed = torch.sum(token_emb * mask_expanded, 1)
            counts = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embeddings = torch.nn.functional.normalize(summed / counts, p=2, dim=1)
        return embeddings.tolist()


class HashingEmbeddingClient(EmbeddingClient):
    def __init__(self, dim: int = 256):
        self.dim = dim

    def _hash(self, s: str) -> list[float]:
        # Very crude deterministic embedding for offline use
        v = [0] * self.dim
        for i, ch in enumerate(s.encode("utf-8")):
            v[i % self.dim] = (v[i % self.dim] + ch) % 1000
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash(t) for t in texts]


class ResilientEmbeddingClient(EmbeddingClient):
    """
    Wrapper that falls back to hash embeddings if the primary provider fails.
    """

    def __init__(
        self, primary: EmbeddingClient, fallback: EmbeddingClient | None = None
    ):
        self.primary = primary
        self.fallback = fallback or HashingEmbeddingClient()

    @staticmethod
    def _sanitize_error(exc: Exception) -> str:
        msg = str(exc)
        msg = re.sub(r"(key=)[^&\s]+", r"\1***", msg)
        msg = re.sub(r"(Bearer\s+)[A-Za-z0-9._-]+", r"\1***", msg)
        return msg

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            return await self.primary.embed(texts)
        except Exception as exc:
            safe_err = self._sanitize_error(exc)
            logger.warning(
                "Primary embedding provider failed (%s); using hash fallback.",
                safe_err,
            )
            return await self.fallback.embed(texts)


def _local_embedding_available() -> bool:
    """Check if transformers + torch are installed for local embeddings."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _best_fallback() -> EmbeddingClient:
    """Return LocalEmbeddingClient if torch is available, else HashingEmbeddingClient."""
    if _local_embedding_available():
        return LocalEmbeddingClient()
    return HashingEmbeddingClient()


def get_default_embedding_client() -> EmbeddingClient:
    import os

    provider = (os.getenv("EMBED_PROVIDER") or "auto").lower()
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    zhipu_key = os.getenv("ZHIPU_API_KEY")
    llm_provider = (os.getenv("LLM_PROVIDER") or "").lower()
    prefer_gemini = llm_provider.startswith("gemini")

    fallback = _best_fallback()

    def _openai_client() -> EmbeddingClient:
        model = os.getenv("EMBED_MODEL") or "text-embedding-3-small"
        base_url = os.getenv("OPENAI_BASE_URL")
        primary = OpenAIEmbeddingClient(
            api_key=openai_key or "", model=model, base_url=base_url
        )
        return ResilientEmbeddingClient(primary=primary, fallback=fallback)

    def _gemini_client() -> EmbeddingClient:
        model = os.getenv("EMBED_MODEL") or "models/gemini-embedding-001"
        base_url = os.getenv("GEMINI_BASE_URL")
        primary = GeminiEmbeddingClient(
            api_key=gemini_key or "", model=model, base_url=base_url
        )
        return ResilientEmbeddingClient(primary=primary, fallback=fallback)

    def _zhipu_client() -> EmbeddingClient:
        model = os.getenv("EMBED_MODEL") or "embedding-3"
        primary = ZhiPuEmbeddingClient(api_key=zhipu_key or "", model=model)
        return ResilientEmbeddingClient(primary=primary, fallback=fallback)

    # When using Claude/Anthropic as LLM, prefer local embeddings (free, fast, offline)
    # Only use API embeddings if explicitly configured via EMBED_PROVIDER
    prefer_local = llm_provider in ("anthropic", "claude", "")

    if provider == "auto":
        # Prefer local model when using Claude — no need for OpenAI key
        if prefer_local and _local_embedding_available():
            logger.info("Using local embeddings (GTE-small, 384-dim)")
            return ResilientEmbeddingClient(primary=LocalEmbeddingClient(), fallback=fallback)
        # Follow the active LLM family, then fall back by availability
        if prefer_gemini and gemini_key:
            return _gemini_client()
        if openai_key:
            return _openai_client()
        if gemini_key:
            return _gemini_client()
        if zhipu_key:
            return _zhipu_client()

    if provider == "openai" and openai_key:
        return _openai_client()

    if provider in {"gemini", "google"} and gemini_key:
        return _gemini_client()

    if provider == "zhipu" and zhipu_key:
        return _zhipu_client()

    if provider == "local" or provider == "auto":
        return fallback

    # Final fallback — always prefer local over hash
    return fallback


class InMemoryVectorIndex:
    """
    In-memory vector index with numpy-optimized search.
    Uses vectorized matrix operations for fast cosine similarity.
    """

    def __init__(self):
        self._ids: list[str] = []
        self._metas: list[dict[str, Any]] = []
        self._vectors: Any | None = None  # numpy array, lazy init
        self._dim: int | None = None

    def add(
        self, item_id: str, vector: list[float], meta: dict[str, Any] | None = None
    ):
        try:
            import numpy as np

            vec = np.array(vector, dtype=np.float32)

            # Normalize the vector for cosine similarity
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec = vec / norm

            self._ids.append(item_id)
            self._metas.append(meta or {})

            if self._vectors is None:
                self._dim = len(vector)
                self._vectors = vec.reshape(1, -1)
            else:
                self._vectors = np.vstack([self._vectors, vec.reshape(1, -1)])
        except ImportError:
            # Fallback: store as list if numpy not available
            if not hasattr(self, "_items"):
                self._items = []
            self._items.append((item_id, vector, meta or {}))

    def search(
        self, query_vec: list[float], top_k: int = 5
    ) -> list[tuple[str, float, dict[str, Any]]]:
        if not self._ids:
            return []

        try:
            import numpy as np

            # Normalize query vector
            query = np.array(query_vec, dtype=np.float32)
            norm = np.linalg.norm(query)
            if norm > 1e-12:
                query = query / norm

            # Vectorized cosine similarity: matrix @ vector
            # Since vectors are normalized, dot product = cosine similarity
            scores = self._vectors @ query  # Shape: (n_items,)

            # Get top-k indices
            if len(scores) <= top_k:
                top_indices = np.argsort(scores)[::-1]
            else:
                # Use argpartition for efficiency when k << n
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            return [
                (self._ids[i], float(scores[i]), self._metas[i]) for i in top_indices
            ]

        except ImportError:
            # Fallback to pure Python if numpy not available
            return self._search_fallback(query_vec, top_k)

    def _search_fallback(
        self, query_vec: list[float], top_k: int
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Pure Python fallback for when numpy is not available."""

        def cosine(a: list[float], b: list[float]) -> float:
            s = sum(x * y for x, y in zip(a, b, strict=False))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return s / (na * nb)

        # Check both storage paths: _items (list fallback) and _ids+_metas (numpy path)
        if hasattr(self, "_items") and self._items:
            scored = [
                (iid, cosine(vec, query_vec), meta) for (iid, vec, meta) in self._items
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
        # Fallback: reconstruct from numpy storage if _items not populated
        if self._ids and self._vectors is not None:
            try:
                vecs_list = self._vectors.tolist() if hasattr(self._vectors, 'tolist') else []
                scored = [
                    (self._ids[i], cosine(vecs_list[i], query_vec), self._metas[i])
                    for i in range(len(self._ids))
                ]
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored[:top_k]
            except (IndexError, TypeError):
                pass
        return []
