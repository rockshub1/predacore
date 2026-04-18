"""
The Knowledge Nexus (KN) component is responsible for knowledge
representation, integration, synthesis, and lifelong learning.
"""

from .service import KnowledgeNexusService
from .storage import InMemoryKnowledgeGraphStore  # Export storage too

__all__ = [
    "KnowledgeNexusService",
    "InMemoryKnowledgeGraphStore",
]
