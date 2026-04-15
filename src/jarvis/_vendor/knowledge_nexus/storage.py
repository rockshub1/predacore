"""
Storage implementations for the Knowledge Nexus.
Includes in-memory and a simple disk-backed JSONL persistence layer.
"""
import json
import logging
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

# Assuming Pydantic models are used internally for storage consistency
from jarvis._vendor.common.models import KnowledgeEdge, KnowledgeNode


# Concept from service.py - can be formalized later if needed
class AbstractKnowledgeGraphStore:
    async def add_node(self, node: KnowledgeNode) -> bool:
        raise NotImplementedError

    async def get_node(self, node_id: UUID) -> KnowledgeNode | None:
        raise NotImplementedError

    async def update_node(self, node: KnowledgeNode) -> bool:
        raise NotImplementedError

    async def delete_node(self, node_id: UUID) -> bool:
        raise NotImplementedError

    async def add_edge(self, edge: KnowledgeEdge) -> bool:
        raise NotImplementedError

    async def get_edge(self, edge_id: UUID) -> KnowledgeEdge | None:
        raise NotImplementedError

    async def query_nodes(
        self,
        labels: set[str] | None = None,
        properties: dict[str, Any] | None = None,
        layers: set[str] | None = None,
    ) -> list[KnowledgeNode]:
        raise NotImplementedError

    async def query_edges(
        self,
        source_id: UUID | None = None,
        target_id: UUID | None = None,
        edge_type: str | None = None,
        layers: set[str] | None = None,
    ) -> list[KnowledgeEdge]:
        raise NotImplementedError

    async def get_neighbors(
        self,
        node_id: UUID,
        edge_type: str | None = None,
        direction: str = "outgoing",
    ) -> list[KnowledgeNode]:
        raise NotImplementedError

    # Add other methods like get_neighbors later


class InMemoryKnowledgeGraphStore(AbstractKnowledgeGraphStore):
    """
    A simple in-memory storage implementation for the Knowledge Graph.
    Suitable for initial development and testing. Not persistent.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self._nodes: dict[UUID, KnowledgeNode] = {}
        self._edges: dict[UUID, KnowledgeEdge] = {}
        # Add indexes later for faster querying (e.g., by label, by source/target node)
        self._label_index: dict[str, set[UUID]] = {}
        self._edge_type_index: dict[str, set[UUID]] = {}
        self._outgoing_edges: dict[UUID, set[UUID]] = {}  # node_id -> set(edge_id)
        self._incoming_edges: dict[UUID, set[UUID]] = {}  # node_id -> set(edge_id)
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("InMemoryKnowledgeGraphStore initialized.")

    async def add_node(self, node: KnowledgeNode) -> bool:
        if node.id in self._nodes:
            self.logger.warning(
                f"Node with ID {node.id} already exists. Use update_node instead."
            )
            return False
        # Deep copy to prevent external modifications affecting stored object
        self._nodes[node.id] = node.model_copy(deep=True)
        self._update_indexes_for_node(node)
        self.logger.debug(f"Added node {node.id} with labels {node.labels}")
        return True

    async def get_node(self, node_id: UUID) -> KnowledgeNode | None:
        node = self._nodes.get(node_id)
        if node:
            # Return a deep copy to prevent external modifications
            return node.model_copy(deep=True)
        self.logger.debug(f"Node {node_id} not found.")
        return None

    async def update_node(self, node: KnowledgeNode) -> bool:
        if node.id not in self._nodes:
            self.logger.warning(
                f"Node with ID {node.id} does not exist. Cannot update."
            )
            return False
        # Remove old index entries before updating
        old_node = self._nodes[node.id]
        self._remove_indexes_for_node(old_node)
        # Store deep copy
        self._nodes[node.id] = node.model_copy(deep=True)
        # Add new index entries
        self._update_indexes_for_node(node)
        self.logger.debug(f"Updated node {node.id}")
        return True

    async def delete_node(self, node_id: UUID) -> bool:
        if node_id not in self._nodes:
            self.logger.warning(
                f"Node with ID {node_id} does not exist. Cannot delete."
            )
            return False
        # Find and delete connected edges first
        edges_to_delete = set()
        edges_to_delete.update(self._outgoing_edges.get(node_id, set()))
        edges_to_delete.update(self._incoming_edges.get(node_id, set()))

        deleted_edge_count = 0
        for edge_id in list(edges_to_delete):  # Iterate over a copy
            if edge_id in self._edges:
                edge_to_delete = self._edges.pop(edge_id)
                self._remove_indexes_for_edge(edge_to_delete)
                deleted_edge_count += 1
            else:
                # Clean up potentially inconsistent index entries if edge not found
                if node_id in self._outgoing_edges:
                    self._outgoing_edges[node_id].discard(edge_id)
                if node_id in self._incoming_edges:
                    self._incoming_edges[node_id].discard(edge_id)
                # Also check edge type index consistency if needed

        if deleted_edge_count > 0:
            self.logger.debug(
                f"Deleted {deleted_edge_count} edges connected to node {node_id}"
            )

        # Now delete the node and its indexes
        node = self._nodes.pop(node_id)
        self._remove_indexes_for_node(node)
        self.logger.debug(f"Deleted node {node_id} and its associated edges.")
        return True

    async def add_edge(self, edge: KnowledgeEdge) -> bool:
        if edge.id in self._edges:
            self.logger.warning(f"Edge with ID {edge.id} already exists.")
            return False
        # Ensure source and target nodes exist (optional check, depends on desired strictness)
        if (
            edge.source_node_id not in self._nodes
            or edge.target_node_id not in self._nodes
        ):
            self.logger.warning(
                f"Cannot add edge {edge.id}: Source or target node does not exist."
            )
            return False  # Strict: reject edges with missing nodes
        # Store deep copy
        self._edges[edge.id] = edge.model_copy(deep=True)
        self._update_indexes_for_edge(edge)
        self.logger.debug(f"Added edge {edge.id} of type {edge.type}")
        return True

    async def get_edge(self, edge_id: UUID) -> KnowledgeEdge | None:
        edge = self._edges.get(edge_id)
        if edge:
            return edge.model_copy(deep=True)
        self.logger.debug(f"Edge {edge_id} not found.")
        return None

    # --- Index Helper Methods ---

    def _update_indexes_for_node(self, node: KnowledgeNode):
        for label in node.labels:
            if label not in self._label_index:
                self._label_index[label] = set()
            self._label_index[label].add(node.id)

    def _remove_indexes_for_node(self, node: KnowledgeNode):
        for label in node.labels:
            if label in self._label_index:
                self._label_index[label].discard(node.id)
                if not self._label_index[label]:  # Clean up empty sets
                    del self._label_index[label]

    def _update_indexes_for_edge(self, edge: KnowledgeEdge):
        # Edge type index
        if edge.type not in self._edge_type_index:
            self._edge_type_index[edge.type] = set()
        self._edge_type_index[edge.type].add(edge.id)
        # Incoming/Outgoing indexes
        if edge.source_node_id not in self._outgoing_edges:
            self._outgoing_edges[edge.source_node_id] = set()
        self._outgoing_edges[edge.source_node_id].add(edge.id)
        if edge.target_node_id not in self._incoming_edges:
            self._incoming_edges[edge.target_node_id] = set()
        self._incoming_edges[edge.target_node_id].add(edge.id)

    def _remove_indexes_for_edge(self, edge: KnowledgeEdge):
        # Edge type index
        if edge.type in self._edge_type_index:
            self._edge_type_index[edge.type].discard(edge.id)
            if not self._edge_type_index[edge.type]:
                del self._edge_type_index[edge.type]
        # Incoming/Outgoing indexes
        if edge.source_node_id in self._outgoing_edges:
            self._outgoing_edges[edge.source_node_id].discard(edge.id)
            if not self._outgoing_edges[edge.source_node_id]:
                del self._outgoing_edges[edge.source_node_id]
        if edge.target_node_id in self._incoming_edges:
            self._incoming_edges[edge.target_node_id].discard(edge.id)
            if not self._incoming_edges[edge.target_node_id]:
                del self._incoming_edges[edge.target_node_id]

    # --- Query Methods (To be implemented based on indexes) ---

    async def query_nodes(
        self,
        labels: set[str] | None = None,
        properties: dict[str, Any] | None = None,
        layers: set[str] | None = None,
    ) -> list[KnowledgeNode]:
        # Basic implementation using label index, needs property/layer filtering
        candidate_ids = set(self._nodes.keys())
        if labels:
            # Get sets of node IDs for each label
            label_match_sets = [self._label_index.get(lbl, set()) for lbl in labels]
            # If any label filter results in an empty set, the intersection will be empty
            if not all(label_match_sets):
                candidate_ids = set()
            else:
                # Intersect the sets to find nodes matching ALL labels
                candidate_ids.intersection_update(*label_match_sets)

        # If candidate_ids is empty after label filtering, return early
        if not candidate_ids:
            self.logger.debug("QueryNodes found 0 nodes after label filtering.")
            return []

        results = []
        for node_id in candidate_ids:
            node = self._nodes[node_id]  # Node should exist

            # Apply property filtering (exact match for all specified properties)
            if properties:
                match = True
                for key, value in properties.items():
                    if node.properties.get(key) != value:
                        match = False
                        break
                if not match:
                    continue  # Skip node if any property doesn't match

            # Apply layer filtering (node must be in at least one specified layer)
            if layers:
                if not node.layers.intersection(layers):
                    continue  # Skip node if it doesn't belong to any required layer

            # If all filters passed, add a copy to results
            results.append(node.model_copy(deep=True))

        self.logger.debug(f"QueryNodes found {len(results)} nodes after all filters.")
        return results

    async def query_edges(
        self,
        source_id: UUID | None = None,
        target_id: UUID | None = None,
        edge_type: str | None = None,
        layers: set[str] | None = None,
    ) -> list[KnowledgeEdge]:
        """Queries edges based on source, target, type, and layers."""
        candidate_ids: set[UUID] = set(self._edges.keys())
        intersect_count = 0

        # Filter by edge type (copy to avoid mutating internal indexes)
        if edge_type is not None:
            type_matches = self._edge_type_index.get(edge_type, set())
            if intersect_count > 0:
                candidate_ids.intersection_update(type_matches)
            else:
                candidate_ids = set(type_matches)
            intersect_count += 1
            if not candidate_ids:
                return []  # Early exit if no matches

        # Filter by source node
        if source_id is not None:
            source_matches = self._outgoing_edges.get(source_id, set())
            if intersect_count > 0:
                candidate_ids.intersection_update(source_matches)
            else:
                candidate_ids = set(source_matches)
            intersect_count += 1
            if not candidate_ids:
                return []

        # Filter by target node
        if target_id is not None:
            target_matches = self._incoming_edges.get(target_id, set())
            if intersect_count > 0:
                candidate_ids.intersection_update(target_matches)
            else:
                candidate_ids = set(target_matches)
            intersect_count += 1
            if not candidate_ids:
                return []

        # Retrieve edges and filter by layer
        results = []
        for edge_id in candidate_ids:
            edge = self._edges[edge_id]  # Should exist if ID is in candidate_ids
            # Apply layer filter (edge must be in at least one specified layer)
            if layers is not None:
                if not edge.layers.intersection(layers):
                    continue  # Skip if no overlap with required layers

            results.append(edge.model_copy(deep=True))

        self.logger.debug(f"QueryEdges found {len(results)} edges matching criteria.")
        return results

    async def get_neighbors(
        self,
        node_id: UUID,
        edge_type: str | None = None,
        direction: str = "outgoing",
    ) -> list[KnowledgeNode]:
        """Retrieves neighbor nodes connected by edges of a specific type and direction."""
        if node_id not in self._nodes:
            self.logger.warning(f"Cannot get neighbors for non-existent node {node_id}")
            return []

        neighbor_node_ids: set[UUID] = set()
        edge_ids_to_check: set[UUID] = set()

        # Determine which edges to check based on direction
        if direction in ["outgoing", "both"]:
            edge_ids_to_check.update(self._outgoing_edges.get(node_id, set()))
        if direction in ["incoming", "both"]:
            edge_ids_to_check.update(self._incoming_edges.get(node_id, set()))

        if not edge_ids_to_check:
            return []  # No relevant edges found

        # Iterate through relevant edges
        for edge_id in edge_ids_to_check:
            edge = self._edges.get(edge_id)
            if not edge:
                continue  # Should not happen if indexes are consistent

            # Filter by edge type if specified
            if edge_type is not None and edge.type != edge_type:
                continue

            # Determine neighbor ID based on direction
            is_outgoing = edge.source_node_id == node_id
            is_incoming = edge.target_node_id == node_id

            if direction == "outgoing" and is_outgoing:
                neighbor_node_ids.add(edge.target_node_id)
            elif direction == "incoming" and is_incoming:
                neighbor_node_ids.add(edge.source_node_id)
            elif direction == "both":
                if is_outgoing:
                    neighbor_node_ids.add(edge.target_node_id)
                if is_incoming:  # Can be both if it's a self-loop
                    neighbor_node_ids.add(edge.source_node_id)

        # Retrieve neighbor nodes
        results = []
        for neighbor_id in neighbor_node_ids:
            neighbor_node = await self.get_node(
                neighbor_id
            )  # Use get_node to get a deep copy
            if neighbor_node:
                results.append(neighbor_node)
            else:
                # This indicates data inconsistency (edge points to non-existent node)
                self.logger.warning(
                    f"Edge points to non-existent neighbor node {neighbor_id} from node {node_id}"
                )

        self.logger.debug(
            f"GetNeighbors for node {node_id} (type={edge_type}, dir={direction}) found {len(results)} neighbors."
        )
        return results


class DiskKnowledgeGraphStore(AbstractKnowledgeGraphStore):
    """
    Minimal disk-backed store using JSONL files for nodes and edges.
    Suitable for persistence across restarts; not optimized for scale.
    """

    def __init__(
        self, base_dir: str = "data/kn_store", logger: logging.Logger | None = None
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.nodes_path = self.base_dir / "nodes.jsonl"
        self.edges_path = self.base_dir / "edges.jsonl"
        self._nodes: dict[UUID, KnowledgeNode] = {}
        self._edges: dict[UUID, KnowledgeEdge] = {}
        self._label_index: dict[str, set[UUID]] = {}
        self._edge_type_index: dict[str, set[UUID]] = {}
        self._outgoing_edges: dict[UUID, set[UUID]] = {}
        self._incoming_edges: dict[UUID, set[UUID]] = {}
        self.logger = logger or logging.getLogger(__name__)
        self._load_from_disk()
        self.logger.info(f"DiskKnowledgeGraphStore initialized at {self.base_dir}")

    def _load_from_disk(self) -> None:
        """Load nodes and edges from disk if present."""
        if self.nodes_path.exists():
            try:
                with self.nodes_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                            node = KnowledgeNode.model_validate(row)
                            self._nodes[node.id] = node
                            self._update_indexes_for_node(node)
                        except Exception:
                            continue
            except Exception as e:
                self.logger.warning(f"Failed to load nodes: {e}")
        if self.edges_path.exists():
            try:
                with self.edges_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                            edge = KnowledgeEdge.model_validate(row)
                            self._edges[edge.id] = edge
                            self._update_indexes_for_edge(edge)
                        except Exception:
                            continue
            except Exception as e:
                self.logger.warning(f"Failed to load edges: {e}")

    def _append_jsonl(self, path: Path, obj: dict[str, Any]) -> None:
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.warning(f"Failed to append to {path}: {e}")

    def _rewrite_jsonl(self, path: Path, objs: list[dict[str, Any]]) -> None:
        try:
            with path.open("w", encoding="utf-8") as f:
                for obj in objs:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.warning(f"Failed to rewrite {path}: {e}")

    # Node operations
    async def add_node(self, node: KnowledgeNode) -> bool:
        if node.id in self._nodes:
            self.logger.warning(
                f"Node with ID {node.id} already exists. Use update_node instead."
            )
            return False
        self._nodes[node.id] = node.model_copy(deep=True)
        self._update_indexes_for_node(node)
        self._append_jsonl(self.nodes_path, node.model_dump())
        return True

    async def get_node(self, node_id: UUID) -> KnowledgeNode | None:
        node = self._nodes.get(node_id)
        return node.model_copy(deep=True) if node else None

    async def update_node(self, node: KnowledgeNode) -> bool:
        if node.id not in self._nodes:
            return False
        old = self._nodes[node.id]
        self._remove_indexes_for_node(old)
        self._nodes[node.id] = node.model_copy(deep=True)
        self._update_indexes_for_node(node)
        # Rewrite all nodes
        self._rewrite_jsonl(
            self.nodes_path, [n.model_dump() for n in self._nodes.values()]
        )
        return True

    async def delete_node(self, node_id: UUID) -> bool:
        if node_id not in self._nodes:
            return False
        # remove connected edges
        edges_to_delete = list(self._outgoing_edges.get(node_id, set())) + list(
            self._incoming_edges.get(node_id, set())
        )
        for eid in edges_to_delete:
            edge = self._edges.pop(eid, None)
            if edge:
                self._remove_indexes_for_edge(edge)
        node = self._nodes.pop(node_id)
        self._remove_indexes_for_node(node)
        self._rewrite_jsonl(
            self.nodes_path, [n.model_dump() for n in self._nodes.values()]
        )
        self._rewrite_jsonl(
            self.edges_path, [e.model_dump() for e in self._edges.values()]
        )
        return True

    # Edge operations
    async def add_edge(self, edge: KnowledgeEdge) -> bool:
        if edge.id in self._edges:
            self.logger.warning(f"Edge with ID {edge.id} already exists.")
            return False
        self._edges[edge.id] = edge.model_copy(deep=True)
        self._update_indexes_for_edge(edge)
        self._append_jsonl(self.edges_path, edge.model_dump())
        return True

    async def get_edge(self, edge_id: UUID) -> KnowledgeEdge | None:
        edge = self._edges.get(edge_id)
        return edge.model_copy(deep=True) if edge else None

    # Query methods reuse in-memory implementations
    async def query_nodes(
        self,
        labels: set[str] | None = None,
        properties: dict[str, Any] | None = None,
        layers: set[str] | None = None,
    ) -> list[KnowledgeNode]:
        return await InMemoryKnowledgeGraphStore.query_nodes(
            self, labels, properties, layers
        )  # type: ignore

    async def query_edges(
        self,
        source_id: UUID | None = None,
        target_id: UUID | None = None,
        edge_type: str | None = None,
        layers: set[str] | None = None,
    ) -> list[KnowledgeEdge]:
        return await InMemoryKnowledgeGraphStore.query_edges(
            self, source_id, target_id, edge_type, layers
        )  # type: ignore

    async def get_neighbors(
        self,
        node_id: UUID,
        edge_type: str | None = None,
        direction: str = "outgoing",
    ) -> list[KnowledgeNode]:
        return await InMemoryKnowledgeGraphStore.get_neighbors(
            self, node_id, edge_type, direction
        )  # type: ignore
