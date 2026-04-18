"""
Abstract Vector Index for Knowledge Nexus semantic search.
"""
from abc import ABC, abstractmethod
from typing import Optional


class AbstractVectorIndex(ABC):
    """Abstract base class for vector index implementations."""

    @abstractmethod
    async def add(
        self, item_id: str, embedding: list[float], metadata: dict | None = None
    ) -> bool:
        """Add an item to the vector index."""
        pass

    @abstractmethod
    async def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        layers: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Search for similar items.

        Returns list of (item_id, similarity_score) tuples.
        """
        pass

    @abstractmethod
    async def remove(self, item_id: str) -> bool:
        """Remove an item from the index."""
        pass

    async def update(
        self, item_id: str, embedding: list[float], metadata: dict | None = None
    ) -> bool:
        """Update an item's embedding. Default implementation removes and re-adds."""
        await self.remove(item_id)
        return await self.add(item_id, embedding, metadata)
