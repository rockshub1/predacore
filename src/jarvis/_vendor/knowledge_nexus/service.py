"""
gRPC Service implementation for the Knowledge Nexus (KN).
"""
import logging
import os
from concurrent import futures
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import grpc
from jarvis._vendor.common.embedding import EmbeddingClient, get_default_embedding_client
from jarvis._vendor.common.models import KnowledgeEdge, KnowledgeNode
from jarvis._vendor.common.protos import knowledge_nexus_pb2, knowledge_nexus_pb2_grpc
from google.protobuf.struct_pb2 import Struct

from .faiss_vector_index import FAISSVectorIndex
from .storage import (
    AbstractKnowledgeGraphStore,
    DiskKnowledgeGraphStore,
    InMemoryKnowledgeGraphStore,
)
from .vector_index import AbstractVectorIndex

# NOTE: AbstractKnowledgeGraphStore and AbstractVectorIndex are properly imported
# from storage.py and vector_index.py respectively.


class KnowledgeNexusService(knowledge_nexus_pb2_grpc.KnowledgeNexusServiceServicer):
    """
    Implements the gRPC methods for the Knowledge Nexus service.
    """

    def __init__(
        self,
        storage: AbstractKnowledgeGraphStore,
        vector_index: AbstractVectorIndex,
        logger: logging.Logger | None = None,
        embedding_client: EmbeddingClient | None = None,
    ):
        self.storage = storage
        self.vector_index = vector_index
        self.logger = logger or logging.getLogger(__name__)
        # Use provided embedding client or get default (falls back to hashing if no API key)
        self.embedding_client = embedding_client or get_default_embedding_client()
        self.logger.info("KnowledgeNexusService initialized with embedding support.")

    async def QueryNodes(
        self,
        request: knowledge_nexus_pb2.QueryNodesRequest,
        context: grpc.aio.ServicerContext,
    ) -> knowledge_nexus_pb2.QueryNodesResponse:
        self.logger.debug(f"Received QueryNodes request: {request}")

        # Extract and convert filters
        labels_filter = set(request.labels_filter) if request.labels_filter else None
        properties_filter = (
            self._struct_to_dict(request.properties_filter)
            if request.properties_filter.fields
            else None
        )
        layers_filter = set(request.layers_filter) if request.layers_filter else None

        # Call storage method
        try:
            found_nodes: list[KnowledgeNode] = await self.storage.query_nodes(
                labels=labels_filter, properties=properties_filter, layers=layers_filter
            )
        except Exception as e:
            self.logger.error(f"Error querying nodes in storage: {e}", exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, "Failed to query nodes")
            return knowledge_nexus_pb2.QueryNodesResponse()

        # Convert results to response messages
        response_messages = [self._node_to_message(node) for node in found_nodes]
        self.logger.debug(f"QueryNodes returning {len(response_messages)} nodes.")

        return knowledge_nexus_pb2.QueryNodesResponse(nodes=response_messages)

    async def QueryEdges(
        self,
        request: knowledge_nexus_pb2.QueryEdgesRequest,
        context: grpc.aio.ServicerContext,
    ) -> knowledge_nexus_pb2.QueryEdgesResponse:
        self.logger.debug(f"Received QueryEdges request: {request}")

        # Extract and validate filters
        source_uuid: UUID | None = None
        target_uuid: UUID | None = None
        try:
            if request.source_node_id_filter:
                source_uuid = UUID(request.source_node_id_filter)
            if request.target_node_id_filter:
                target_uuid = UUID(request.target_node_id_filter)
        except ValueError:
            self.logger.warning(
                f"Invalid UUID format in QueryEdges filter: {request.source_node_id_filter} or {request.target_node_id_filter}"
            )
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Invalid source or target node ID format (must be UUID)",
            )
            return knowledge_nexus_pb2.QueryEdgesResponse()

        edge_type_filter = (
            request.edge_type_filter if request.edge_type_filter else None
        )
        layers_filter = set(request.layers_filter) if request.layers_filter else None

        # Call storage method (needs implementation in storage.py)
        try:
            # Ensure the storage method signature matches
            found_edges: list[KnowledgeEdge] = await self.storage.query_edges(
                source_id=source_uuid,
                target_id=target_uuid,
                edge_type=edge_type_filter,
                layers=layers_filter,
            )
        except NotImplementedError:
            self.logger.error("Storage method query_edges is not implemented.")
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "Edge querying not implemented in storage layer",
            )
            return knowledge_nexus_pb2.QueryEdgesResponse()
        except Exception as e:
            self.logger.error(f"Error querying edges in storage: {e}", exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, "Failed to query edges")
            return knowledge_nexus_pb2.QueryEdgesResponse()

        # Convert results to response messages
        response_messages = [self._edge_to_message(edge) for edge in found_edges]
        self.logger.debug(f"QueryEdges returning {len(response_messages)} edges.")

        return knowledge_nexus_pb2.QueryEdgesResponse(edges=response_messages)

    async def GetNodeDetails(
        self,
        request: knowledge_nexus_pb2.GetNodeDetailsRequest,
        context: grpc.aio.ServicerContext,
    ) -> knowledge_nexus_pb2.KnowledgeNodeMessage:
        self.logger.debug(f"Received GetNodeDetails request for ID: {request.node_id}")
        try:
            node_uuid = UUID(request.node_id)
        except ValueError:
            self.logger.warning(f"Invalid UUID format received: {request.node_id}")
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Invalid node ID format (must be UUID)",
            )
            return knowledge_nexus_pb2.KnowledgeNodeMessage()

        node = await self.storage.get_node(node_uuid)

        if node is None:
            self.logger.info(f"Node {request.node_id} not found.")
            await context.abort(
                grpc.StatusCode.NOT_FOUND, f"Node {request.node_id} not found"
            )
            return knowledge_nexus_pb2.KnowledgeNodeMessage()

        self.logger.debug(f"Found node {request.node_id}, converting to message.")
        return self._node_to_message(node)

    async def SemanticSearch(
        self,
        request: knowledge_nexus_pb2.SemanticSearchRequest,
        context: grpc.aio.ServicerContext,
    ) -> knowledge_nexus_pb2.SemanticSearchResponse:
        self.logger.debug(
            f"Received SemanticSearch request: '{request.query_text[:50]}...', top_k={request.top_k}"
        )

        # 1. Generate embedding for request.query_text using embedding client
        try:
            embeddings = await self.embedding_client.embed([request.query_text])
            if not embeddings or not embeddings[0]:
                self.logger.error("Embedding client returned empty result")
                await context.abort(
                    grpc.StatusCode.INTERNAL, "Failed to generate embedding"
                )
                return knowledge_nexus_pb2.SemanticSearchResponse()
            query_embedding = embeddings[0]
            self.logger.debug(
                f"Generated embedding of dimension {len(query_embedding)}"
            )
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}", exc_info=True)
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Embedding generation failed: {e}"
            )
            return knowledge_nexus_pb2.SemanticSearchResponse()

        # 2. Call vector index search
        try:
            layers_filter = (
                set(request.layers_filter) if request.layers_filter else None
            )
            # Returns list of (node_id_str, score) tuples
            similar_results = await self.vector_index.search_similar(
                query_embedding=query_embedding,
                top_k=request.top_k,
                layers=layers_filter,
            )
        except NotImplementedError:
            self.logger.error("Vector index search_similar method not implemented.")
            await context.abort(
                grpc.StatusCode.UNIMPLEMENTED, "Vector index search not implemented"
            )
            return knowledge_nexus_pb2.SemanticSearchResponse()
        except Exception as e:
            self.logger.error(f"Error during vector index search: {e}", exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, "Vector index search failed")
            return knowledge_nexus_pb2.SemanticSearchResponse()

        # 3. Retrieve node details from storage for results
        response_results = []
        for node_id_str, score in similar_results:
            try:
                node_uuid = UUID(node_id_str)
                node = await self.storage.get_node(node_uuid)
                if node:
                    node_message = self._node_to_message(node)
                    response_results.append(
                        knowledge_nexus_pb2.SemanticSearchResponse.Result(
                            node=node_message, score=score
                        )
                    )
                else:
                    # This might indicate inconsistency between vector index and main storage
                    self.logger.warning(
                        f"Node ID {node_id_str} found in vector index but not in main storage. Skipping."
                    )
            except ValueError:
                self.logger.warning(
                    f"Invalid UUID {node_id_str} received from vector index. Skipping."
                )
            except Exception as e:
                self.logger.error(
                    f"Error retrieving node {node_id_str} from storage during semantic search: {e}",
                    exc_info=True,
                )
                # Decide whether to skip this result or abort the whole request

        # 4. Format response
        self.logger.debug(f"SemanticSearch returning {len(response_results)} results.")
        return knowledge_nexus_pb2.SemanticSearchResponse(results=response_results)

    async def AddRelation(
        self,
        request: knowledge_nexus_pb2.AddRelationRequest,
        context: grpc.aio.ServicerContext,
    ) -> knowledge_nexus_pb2.KnowledgeEdgeMessage:
        self.logger.debug(
            f"Received AddRelation request: {request.source_node_id} -> {request.target_node_id} ({request.edge_type})"
        )

        try:
            source_uuid = UUID(request.source_node_id)
            target_uuid = UUID(request.target_node_id)
        except ValueError:
            self.logger.warning(
                f"Invalid UUID format received in AddRelation: {request.source_node_id} or {request.target_node_id}"
            )
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Invalid source or target node ID format (must be UUID)",
            )
            return knowledge_nexus_pb2.KnowledgeEdgeMessage()

        # Convert properties from Struct to dict
        properties_dict = self._struct_to_dict(request.properties)

        # Create Pydantic model instance for the edge
        # A new UUID is generated by default for the edge ID
        edge_model = KnowledgeEdge(
            source_node_id=source_uuid,
            target_node_id=target_uuid,
            type=request.edge_type,
            properties=properties_dict or {},  # Ensure properties is a dict
            layers={request.layer} if request.layer else set(),
        )

        # Add edge to storage
        success = await self.storage.add_edge(edge_model)

        if not success:
            # Reason for failure might be logged in storage (e.g., already exists, nodes not found)
            self.logger.warning(f"Failed to add edge {edge_model.id} via storage.")
            # Determine appropriate error code based on storage implementation details
            # For now, use FAILED_PRECONDITION as a general indicator
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Failed to add edge. Check logs for details.",
            )
            return knowledge_nexus_pb2.KnowledgeEdgeMessage()

        self.logger.info(
            f"Successfully added edge {edge_model.id} of type {edge_model.type}"
        )
        # Convert the created edge model back to a protobuf message
        return self._edge_to_message(edge_model)

    async def IngestText(
        self,
        request: knowledge_nexus_pb2.IngestTextRequest,
        context: grpc.aio.ServicerContext,
    ) -> knowledge_nexus_pb2.IngestTextResponse:
        self.logger.debug(
            f"Received IngestText request from source: {request.source} for layer {request.layer}"
        )

        # 1. Process text (minimal for v1 - just store) & Metadata
        metadata_dict = self._struct_to_dict(request.metadata) or {}
        properties = {"text": request.text, "source": request.source}
        properties.update(metadata_dict)  # Merge metadata into properties

        # 2. Generate embedding
        generated_embedding = None
        try:
            embeds = await self.embedding_client.embed([request.text])
            if embeds and embeds[0]:
                generated_embedding = embeds[0]
        except Exception as e:
            # Non-fatal: we still store the text node and keep KN usable.
            self.logger.warning(f"Embedding generation failed during ingest: {e}")

        # 3. Create node in storage
        # A new UUID is generated by default
        text_node = KnowledgeNode(
            labels={"TextChunk"},  # Assign a label
            properties=properties,
            embedding=generated_embedding,
            layers={request.layer} if request.layer else set(),
        )

        try:
            success = await self.storage.add_node(text_node)
            if not success:
                # Storage layer should log the specific reason (e.g., duplicate ID unlikely here)
                self.logger.error(f"Failed to add text node {text_node.id} to storage.")
                await context.abort(
                    grpc.StatusCode.INTERNAL, "Failed to store ingested text node"
                )
                return knowledge_nexus_pb2.IngestTextResponse()
        except Exception as e:
            self.logger.error(
                f"Error adding text node {text_node.id} to storage: {e}", exc_info=True
            )
            await context.abort(
                grpc.StatusCode.INTERNAL, "Error storing ingested text node"
            )
            return knowledge_nexus_pb2.IngestTextResponse()

        # 4. Add embedding to vector index (if generated)
        if generated_embedding is not None:
            try:
                vec_meta = {
                    "source": request.source,
                    "layers": [request.layer] if request.layer else [],
                    "labels": ["TextChunk"],
                }
                await self.vector_index.add(
                    item_id=str(text_node.id),
                    embedding=generated_embedding,
                    metadata=vec_meta,
                )
                # Persist vector index when implementation supports it.
                save_fn = getattr(self.vector_index, "save_to_disk", None)
                vector_path = os.getenv(
                    "KN_VECTOR_PATH", "data/kn_store/vector_index.json"
                )
                if callable(save_fn):
                    await save_fn(vector_path)
            except Exception as e:
                # Non-fatal: node is already persisted in graph store.
                self.logger.warning(
                    f"Vector index update failed for node {text_node.id}: {e}"
                )

        self.logger.info(f"Successfully ingested text and created node {text_node.id}")
        return knowledge_nexus_pb2.IngestTextResponse(
            primary_entity_id=str(text_node.id)
        )

    # --- Helper Methods ---

    def _dict_to_struct(self, data: dict[str, Any] | None) -> Struct | None:
        """Converts a Python dictionary to a Protobuf Struct."""
        if data is None:
            return None
        s = Struct()
        try:
            # Note: This handles basic types (str, int, float, bool, list, dict)
            # More complex types within the dict might need specific handling.
            s.update(data)
        except Exception as e:
            self.logger.error(
                f"Error converting dict to Struct: {e}. Data: {data}", exc_info=True
            )
            # Decide on error handling: return None, empty Struct, or raise?
            return None  # Or Struct()
        return s

    def _struct_to_dict(self, struct_data: Struct | None) -> dict[str, Any] | None:
        """Converts a Protobuf Struct to a Python dictionary."""
        if struct_data is None:
            return None
        # Protobuf Struct behaves like a dict, direct conversion works for basic types
        return dict(struct_data)

    def _node_to_message(
        self, node: KnowledgeNode
    ) -> knowledge_nexus_pb2.KnowledgeNodeMessage:
        """Converts a KnowledgeNode Pydantic model to a Protobuf message."""
        # Handle embedding conversion (assuming bytes for now)
        embedding_bytes = b""
        if node.embedding is not None:
            # TODO: Define a consistent serialization format for embeddings (e.g., numpy.save to bytes)
            # Placeholder: attempt to convert directly if bytes, otherwise log warning
            if isinstance(node.embedding, bytes):
                embedding_bytes = node.embedding
            else:
                self.logger.warning(
                    f"Node {node.id} embedding type {type(node.embedding)} not directly convertible to bytes. Skipping."
                )
                # Or attempt serialization:
                # try:
                #     import io
                #     import numpy as np
                #     if isinstance(node.embedding, np.ndarray):
                #          with io.BytesIO() as buf:
                #              np.save(buf, node.embedding)
                #              embedding_bytes = buf.getvalue()
                # except Exception as e:
                #      self.logger.error(f"Failed to serialize embedding for node {node.id}: {e}")

        return knowledge_nexus_pb2.KnowledgeNodeMessage(
            id=str(node.id),
            labels=list(node.labels),
            properties=self._dict_to_struct(node.properties),
            embedding=embedding_bytes,
            layers=list(node.layers),
        )

    def _edge_to_message(
        self, edge: KnowledgeEdge
    ) -> knowledge_nexus_pb2.KnowledgeEdgeMessage:
        """Converts a KnowledgeEdge Pydantic model to a Protobuf message."""
        return knowledge_nexus_pb2.KnowledgeEdgeMessage(
            id=str(edge.id),
            source_node_id=str(edge.source_node_id),
            target_node_id=str(edge.target_node_id),
            type=edge.type,
            properties=self._dict_to_struct(edge.properties),
            layers=list(edge.layers),
        )


# Example function to start the server (for testing/standalone execution)
async def serve():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Storage backend selection
    backend = os.getenv("KN_STORE_BACKEND", "memory").lower()
    if backend == "disk":
        base_dir = os.getenv("KN_STORE_PATH", "data/kn_store")
        storage = DiskKnowledgeGraphStore(base_dir=base_dir, logger=logger)
    else:
        storage = InMemoryKnowledgeGraphStore(logger=logger)

    # Vector index selection (persistent on-disk FAISS-compatible JSON index)
    vector_path = os.getenv("KN_VECTOR_PATH", "data/kn_store/vector_index.json")
    embedding_client = get_default_embedding_client()
    vector_index: AbstractVectorIndex

    if Path(vector_path).exists():
        try:
            vector_index = await FAISSVectorIndex.load_from_disk(
                vector_path, log=logger
            )
            logger.info("Loaded vector index from %s", vector_path)
        except Exception as e:
            logger.warning("Failed loading vector index from %s: %s", vector_path, e)
            vector_index = None  # type: ignore[assignment]
    else:
        vector_index = None  # type: ignore[assignment]

    if vector_index is None:
        dim = 0
        try:
            dim = int(os.getenv("KN_VECTOR_DIM", "0"))
        except Exception:
            dim = 0
        if dim <= 0:
            try:
                probe = await embedding_client.embed(["dimension_probe"])
                if probe and probe[0]:
                    dim = len(probe[0])
            except Exception as e:
                logger.warning(
                    "Could not infer embedding dimension from provider: %s", e
                )
        if dim <= 0:
            dim = 256
        vector_index = FAISSVectorIndex(dimensions=dim, log=logger)
        logger.info("Initialized new vector index (dim=%d, path=%s)", dim, vector_path)
    # --- End Dependency Setup ---

    knowledge_nexus_pb2_grpc.add_KnowledgeNexusServiceServicer_to_server(
        KnowledgeNexusService(
            storage, vector_index, logger, embedding_client=embedding_client
        ),
        server,
    )
    listen_addr = os.getenv("KN_LISTEN_ADDR", "[::]:50051")  # Standard gRPC port
    server.add_insecure_port(listen_addr)
    logger.info(f"Starting KnowledgeNexusService on {listen_addr} (backend={backend})")
    await server.start()
    # Simple readiness/health logging
    logger.info("KnowledgeNexusService started and ready.")
    await server.wait_for_termination()


if __name__ == "__main__":
    import asyncio

    asyncio.run(serve())
