"""
Unit tests for the Knowledge Nexus storage implementations.
"""
from uuid import uuid4

import pytest
import pytest_asyncio
from predacore._vendor.common.models import KnowledgeEdge, KnowledgeNode

# Assuming the project is installed in editable mode or path is adjusted
from predacore._vendor.knowledge_nexus.storage import InMemoryKnowledgeGraphStore

# Use pytest-asyncio for async functions
pytestmark = pytest.mark.asyncio

@pytest.fixture
def store() -> InMemoryKnowledgeGraphStore:
    """Provides a fresh InMemoryKnowledgeGraphStore for each test."""
    return InMemoryKnowledgeGraphStore()

@pytest_asyncio.fixture
async def test_query_edges_setup(store: InMemoryKnowledgeGraphStore):
    """Helper fixture to set up nodes and edges for query_edges tests."""
    n1 = KnowledgeNode(labels={"N"}, properties={"name": "n1"})
    n2 = KnowledgeNode(labels={"N"}, properties={"name": "n2"})
    n3 = KnowledgeNode(labels={"N"}, properties={"name": "n3"})
    await store.add_node(n1)
    await store.add_node(n2)
    await store.add_node(n3)

    e12_t1_l1 = KnowledgeEdge(source_node_id=n1.id, target_node_id=n2.id, type="T1", layers={"L1"})
    e13_t1_l2 = KnowledgeEdge(source_node_id=n1.id, target_node_id=n3.id, type="T1", layers={"L2"})
    e23_t2_l1 = KnowledgeEdge(source_node_id=n2.id, target_node_id=n3.id, type="T2", layers={"L1"})
    e31_t2_l2 = KnowledgeEdge(source_node_id=n3.id, target_node_id=n1.id, type="T2", layers={"L2"})
    e11_t3_l1l2 = KnowledgeEdge(source_node_id=n1.id, target_node_id=n1.id, type="T3", layers={"L1", "L2"}) # Self-loop

    await store.add_edge(e12_t1_l1)
    await store.add_edge(e13_t1_l2)
    await store.add_edge(e23_t2_l1)
    await store.add_edge(e31_t2_l2)
    await store.add_edge(e11_t3_l1l2)
    return {"n1": n1, "n2": n2, "n3": n3,
            "e12": e12_t1_l1, "e13": e13_t1_l2, "e23": e23_t2_l1, "e31": e31_t2_l2, "e11": e11_t3_l1l2}

async def test_add_node_success(store: InMemoryKnowledgeGraphStore):
    """Test adding a new node successfully."""
    node = KnowledgeNode(labels={"Test", "Concept"}, properties={"name": "TestNode"})
    success = await store.add_node(node)
    assert success is True
    # Verify node is retrievable
    retrieved_node = await store.get_node(node.id)
    assert retrieved_node is not None
    assert retrieved_node.id == node.id
    assert retrieved_node.labels == {"Test", "Concept"}
    assert retrieved_node.properties == {"name": "TestNode"}
    # Verify internal state (optional, implementation detail)
    assert node.id in store._nodes
    assert node.id in store._label_index["Test"]
    assert node.id in store._label_index["Concept"]

async def test_add_node_duplicate_id(store: InMemoryKnowledgeGraphStore):
    """Test adding a node with an ID that already exists."""
    node1 = KnowledgeNode()
    await store.add_node(node1)
    # Create another node with the same ID
    node2 = KnowledgeNode(id=node1.id, labels={"Duplicate"})
    success = await store.add_node(node2)
    assert success is False
    # Verify the original node is unchanged
    retrieved_node = await store.get_node(node1.id)
    assert retrieved_node is not None
    assert retrieved_node.labels == set() # Original node had no labels

async def test_get_node_exists(store: InMemoryKnowledgeGraphStore):
    """Test retrieving an existing node."""
    node = KnowledgeNode(properties={"value": 123})
    await store.add_node(node)
    retrieved_node = await store.get_node(node.id)
    assert retrieved_node is not None
    assert retrieved_node.id == node.id
    assert retrieved_node.properties == {"value": 123}
    # Ensure a copy is returned (modify retrieved and check original)
    retrieved_node.properties["value"] = 456
    original_node = await store.get_node(node.id)
    assert original_node is not None
    assert original_node.properties == {"value": 123}


async def test_get_node_not_exists(store: InMemoryKnowledgeGraphStore):
    """Test retrieving a non-existent node."""
    non_existent_id = uuid4()
    retrieved_node = await store.get_node(non_existent_id)
    assert retrieved_node is None

async def test_update_node_success(store: InMemoryKnowledgeGraphStore):
    """Test updating an existing node's properties and labels."""
    node = KnowledgeNode(labels={"Initial"}, properties={"name": "OldName", "value": 1})
    await store.add_node(node)

    # Create updated version (must use same ID)
    updated_node_data = KnowledgeNode(
        id=node.id,
        labels={"Updated", "NewLabel"},
        properties={"name": "NewName", "status": "active"}
    )
    success = await store.update_node(updated_node_data)
    assert success is True

    # Verify updated node is retrieved correctly
    retrieved_node = await store.get_node(node.id)
    assert retrieved_node is not None
    assert retrieved_node.id == node.id
    assert retrieved_node.labels == {"Updated", "NewLabel"}
    assert retrieved_node.properties == {"name": "NewName", "status": "active"}

    # Verify indexes were updated
    assert node.id not in store._label_index.get("Initial", set())
    assert node.id in store._label_index["Updated"]
    assert node.id in store._label_index["NewLabel"]

async def test_update_node_not_exists(store: InMemoryKnowledgeGraphStore):
    """Test updating a node that doesn't exist."""
    node = KnowledgeNode() # Has a new random ID
    success = await store.update_node(node)
    assert success is False

async def test_delete_node_success(store: InMemoryKnowledgeGraphStore):
    """Test deleting an existing node."""
    node = KnowledgeNode(labels={"ToDelete"}, properties={"temp": True})
    await store.add_node(node)

    # Verify it exists first
    assert node.id in store._nodes
    assert node.id in store._label_index["ToDelete"]

    success = await store.delete_node(node.id)
    assert success is True

    # Verify it's gone
    retrieved_node = await store.get_node(node.id)
    assert retrieved_node is None
    assert node.id not in store._nodes
    assert "ToDelete" not in store._label_index # Index should be cleaned up

    # TODO: Add check for edge deletion/handling when deleting nodes

async def test_delete_node_not_exists(store: InMemoryKnowledgeGraphStore):
    """Test deleting a node that doesn't exist."""
    non_existent_id = uuid4()
    success = await store.delete_node(non_existent_id)
    assert success is False
async def test_add_edge_success(store: InMemoryKnowledgeGraphStore):
    """Test adding an edge between existing nodes."""
    node1 = KnowledgeNode()
    node2 = KnowledgeNode()
    await store.add_node(node1)
    await store.add_node(node2)

    edge = KnowledgeEdge(
        source_node_id=node1.id,
        target_node_id=node2.id,
        type="RELATES_TO",
        properties={"weight": 0.5}
    )
    success = await store.add_edge(edge)
    assert success is True

    # Verify edge is retrievable
    retrieved_edge = await store.get_edge(edge.id)
    assert retrieved_edge is not None
    assert retrieved_edge.id == edge.id
    assert retrieved_edge.source_node_id == node1.id
    assert retrieved_edge.target_node_id == node2.id
    assert retrieved_edge.type == "RELATES_TO"
    assert retrieved_edge.properties == {"weight": 0.5}

    # Verify internal state (optional)
    assert edge.id in store._edges
    assert edge.id in store._edge_type_index["RELATES_TO"]
    assert edge.id in store._outgoing_edges[node1.id]
    assert edge.id in store._incoming_edges[node2.id]

async def test_add_edge_duplicate_id(store: InMemoryKnowledgeGraphStore):
    """Test adding an edge with an ID that already exists."""
    node1 = KnowledgeNode()
    node2 = KnowledgeNode()
    await store.add_node(node1)
    await store.add_node(node2)
    edge1 = KnowledgeEdge(source_node_id=node1.id, target_node_id=node2.id, type="T1")
    await store.add_edge(edge1)

    # Create another edge with the same ID
    edge2 = KnowledgeEdge(id=edge1.id, source_node_id=node2.id, target_node_id=node1.id, type="T2")
    success = await store.add_edge(edge2)
    assert success is False

    # Verify original edge is unchanged
    retrieved_edge = await store.get_edge(edge1.id)
    assert retrieved_edge is not None
    assert retrieved_edge.type == "T1"

async def test_add_edge_missing_node(store: InMemoryKnowledgeGraphStore):
    """Test adding an edge where a source/target node doesn't exist.

    The in-memory store is strict: add_edge returns False and logs a warning
    when either endpoint is missing (see InMemoryKnowledgeGraphStore.add_edge
    in src/predacore/_vendor/knowledge_nexus/storage.py).
    """
    node1 = KnowledgeNode()
    await store.add_node(node1)
    missing_node_id = uuid4()

    edge1 = KnowledgeEdge(source_node_id=node1.id, target_node_id=missing_node_id, type="T1")
    success1 = await store.add_edge(edge1)
    assert success1 is False  # target does not exist — strict reject
    assert await store.get_edge(edge1.id) is None

    edge2 = KnowledgeEdge(source_node_id=missing_node_id, target_node_id=node1.id, type="T2")
    success2 = await store.add_edge(edge2)
    assert success2 is False  # source does not exist — strict reject
    assert await store.get_edge(edge2.id) is None

async def test_get_edge_exists(store: InMemoryKnowledgeGraphStore):
    """Test retrieving an existing edge."""
    node1 = KnowledgeNode()
    node2 = KnowledgeNode()
    await store.add_node(node1)
    await store.add_node(node2)
    edge = KnowledgeEdge(source_node_id=node1.id, target_node_id=node2.id, type="LINKS", properties={"detail": "abc"})
    await store.add_edge(edge)

    retrieved_edge = await store.get_edge(edge.id)
    assert retrieved_edge is not None
    assert retrieved_edge.id == edge.id
    assert retrieved_edge.type == "LINKS"
    assert retrieved_edge.properties == {"detail": "abc"}

    # Ensure a copy is returned
    retrieved_edge.properties["detail"] = "xyz"
    original_edge = await store.get_edge(edge.id)
    assert original_edge is not None
    assert original_edge.properties == {"detail": "abc"}

async def test_get_edge_not_exists(store: InMemoryKnowledgeGraphStore):
    """Test retrieving a non-existent edge."""
    non_existent_id = uuid4()
    retrieved_edge = await store.get_edge(non_existent_id)
    assert retrieved_edge is None
async def test_query_nodes_no_filter(store: InMemoryKnowledgeGraphStore):
    """Test querying nodes with no filters, should return all."""
    node1 = KnowledgeNode(labels={"A"}, properties={"p": 1})
    node2 = KnowledgeNode(labels={"B"}, properties={"p": 2})
    await store.add_node(node1)
    await store.add_node(node2)
    results = await store.query_nodes()
    assert len(results) == 2
    assert {n.id for n in results} == {node1.id, node2.id}

async def test_query_nodes_by_single_label(store: InMemoryKnowledgeGraphStore):
    """Test querying nodes by a single label."""
    node1 = KnowledgeNode(labels={"A", "C"}, properties={"p": 1})
    node2 = KnowledgeNode(labels={"B", "C"}, properties={"p": 2})
    node3 = KnowledgeNode(labels={"A"}, properties={"p": 3})
    await store.add_node(node1)
    await store.add_node(node2)
    await store.add_node(node3)

    results_A = await store.query_nodes(labels={"A"})
    assert len(results_A) == 2
    assert {n.id for n in results_A} == {node1.id, node3.id}

    results_C = await store.query_nodes(labels={"C"})
    assert len(results_C) == 2
    assert {n.id for n in results_C} == {node1.id, node2.id}

async def test_query_nodes_by_multiple_labels(store: InMemoryKnowledgeGraphStore):
    """Test querying nodes by multiple labels (intersection)."""
    node1 = KnowledgeNode(labels={"A", "C"}, properties={"p": 1})
    node2 = KnowledgeNode(labels={"B", "C"}, properties={"p": 2})
    node3 = KnowledgeNode(labels={"A"}, properties={"p": 3})
    await store.add_node(node1)
    await store.add_node(node2)
    await store.add_node(node3)

    results = await store.query_nodes(labels={"A", "C"})
    assert len(results) == 1
    assert results[0].id == node1.id

async def test_query_nodes_by_nonexistent_label(store: InMemoryKnowledgeGraphStore):
    """Test querying nodes by a label that doesn't exist."""
    node1 = KnowledgeNode(labels={"A"})
    await store.add_node(node1)
    results = await store.query_nodes(labels={"NonExistent"})
    assert len(results) == 0

async def test_query_nodes_by_single_property(store: InMemoryKnowledgeGraphStore):
    """Test querying nodes by a single property (exact match)."""
    node1 = KnowledgeNode(labels={"A"}, properties={"name": "N1", "value": 10})
    node2 = KnowledgeNode(labels={"B"}, properties={"name": "N2", "value": 20})
    node3 = KnowledgeNode(labels={"C"}, properties={"name": "N1", "value": 30})
    await store.add_node(node1)
    await store.add_node(node2)
    await store.add_node(node3)

    results = await store.query_nodes(properties={"name": "N1"})
    assert len(results) == 2
    assert {n.id for n in results} == {node1.id, node3.id}

    results_val = await store.query_nodes(properties={"value": 20})
    assert len(results_val) == 1
    assert results_val[0].id == node2.id

async def test_query_nodes_by_multiple_properties(store: InMemoryKnowledgeGraphStore):
    """Test querying nodes by multiple properties (exact match)."""
    node1 = KnowledgeNode(labels={"A"}, properties={"name": "N1", "value": 10, "active": True})
    node2 = KnowledgeNode(labels={"B"}, properties={"name": "N2", "value": 20, "active": True})
    node3 = KnowledgeNode(labels={"C"}, properties={"name": "N1", "value": 30, "active": False})
    await store.add_node(node1)
    await store.add_node(node2)
    await store.add_node(node3)

    results = await store.query_nodes(properties={"name": "N1", "active": True})
    assert len(results) == 1
    assert results[0].id == node1.id

async def test_query_nodes_by_nonmatching_property(store: InMemoryKnowledgeGraphStore):
    """Test querying nodes by property value that doesn't match."""
    node1 = KnowledgeNode(properties={"name": "N1"})
    await store.add_node(node1)
    results = await store.query_nodes(properties={"name": "NonExistent"})
    assert len(results) == 0

async def test_query_nodes_by_layer(store: InMemoryKnowledgeGraphStore):
    """Test querying nodes by layer."""
    node1 = KnowledgeNode(labels={"A"}, layers={"L1"})
    node2 = KnowledgeNode(labels={"B"}, layers={"L2"})
    node3 = KnowledgeNode(labels={"C"}, layers={"L1", "L2"})
    await store.add_node(node1)
    await store.add_node(node2)
    await store.add_node(node3)

    results_L1 = await store.query_nodes(layers={"L1"})
    assert len(results_L1) == 2
    assert {n.id for n in results_L1} == {node1.id, node3.id}

    results_L2 = await store.query_nodes(layers={"L2"})
    assert len(results_L2) == 2
    assert {n.id for n in results_L2} == {node2.id, node3.id}

    # Test multiple layers (OR logic - node in ANY specified layer)
    results_L1_L2 = await store.query_nodes(layers={"L1", "L2"})
    assert len(results_L1_L2) == 3
    assert {n.id for n in results_L1_L2} == {node1.id, node2.id, node3.id}

async def test_query_nodes_by_nonexistent_layer(store: InMemoryKnowledgeGraphStore):
    """Test querying nodes by a layer that doesn't exist."""
    node1 = KnowledgeNode(layers={"L1"})
    await store.add_node(node1)
    results = await store.query_nodes(layers={"NonExistent"})
    assert len(results) == 0

async def test_query_nodes_combined_filters(store: InMemoryKnowledgeGraphStore):
    """Test querying nodes using combined label, property, and layer filters."""
    node1 = KnowledgeNode(labels={"A", "Data"}, properties={"status": "active"}, layers={"L1"})
    node2 = KnowledgeNode(labels={"B", "Data"}, properties={"status": "inactive"}, layers={"L2"})
    node3 = KnowledgeNode(labels={"A", "Meta"}, properties={"status": "active"}, layers={"L1", "L2"})
    node4 = KnowledgeNode(labels={"B", "Meta"}, properties={"status": "active"}, layers={"L2"})
    await store.add_node(node1)
    await store.add_node(node2)
    await store.add_node(node3)
    await store.add_node(node4)

    # Label A AND status active
    results1 = await store.query_nodes(labels={"A"}, properties={"status": "active"})
    assert len(results1) == 2
    assert {n.id for n in results1} == {node1.id, node3.id}

    # Label Data AND layer L2
    results2 = await store.query_nodes(labels={"Data"}, layers={"L2"})
    assert len(results2) == 1
    assert results2[0].id == node2.id

    # Status active AND layer L1
    results3 = await store.query_nodes(properties={"status": "active"}, layers={"L1"})
    assert len(results3) == 2
    assert {n.id for n in results3} == {node1.id, node3.id}

    # Label A AND status active AND layer L1
    results4 = await store.query_nodes(labels={"A"}, properties={"status": "active"}, layers={"L1"})
    assert len(results4) == 2 # node1 and node3 match all
    assert {n.id for n in results4} == {node1.id, node3.id}

    # Label Meta AND status active AND layer L2
    results5 = await store.query_nodes(labels={"Meta"}, properties={"status": "active"}, layers={"L2"})
    assert len(results5) == 2 # node3 and node4 match all
    assert {n.id for n in results5} == {node3.id, node4.id}

    # Label B AND status active AND layer L1 (no match)
    results6 = await store.query_nodes(labels={"B"}, properties={"status": "active"}, layers={"L1"})
    assert len(results6) == 0

async def test_query_edges_no_filter(store: InMemoryKnowledgeGraphStore, test_query_edges_setup):
    """Test querying edges with no filters."""
    results = await store.query_edges()
    assert len(results) == 5
    assert {e.id for e in results} == {
        test_query_edges_setup["e12"].id, test_query_edges_setup["e13"].id,
        test_query_edges_setup["e23"].id, test_query_edges_setup["e31"].id,
        test_query_edges_setup["e11"].id
    }

async def test_query_edges_by_source(store: InMemoryKnowledgeGraphStore, test_query_edges_setup):
    """Test querying edges by source node."""
    n1_id = test_query_edges_setup["n1"].id
    results = await store.query_edges(source_id=n1_id)
    assert len(results) == 3
    assert {e.id for e in results} == {
        test_query_edges_setup["e12"].id, test_query_edges_setup["e13"].id, test_query_edges_setup["e11"].id
    }

async def test_query_edges_by_target(store: InMemoryKnowledgeGraphStore, test_query_edges_setup):
    """Test querying edges by target node."""
    n3_id = test_query_edges_setup["n3"].id
    results = await store.query_edges(target_id=n3_id)
    assert len(results) == 2
    assert {e.id for e in results} == {test_query_edges_setup["e13"].id, test_query_edges_setup["e23"].id}

async def test_query_edges_by_type(store: InMemoryKnowledgeGraphStore, test_query_edges_setup):
    """Test querying edges by type."""
    results_t1 = await store.query_edges(edge_type="T1")
    assert len(results_t1) == 2
    assert {e.id for e in results_t1} == {test_query_edges_setup["e12"].id, test_query_edges_setup["e13"].id}

    results_t2 = await store.query_edges(edge_type="T2")
    assert len(results_t2) == 2
    assert {e.id for e in results_t2} == {test_query_edges_setup["e23"].id, test_query_edges_setup["e31"].id}

    results_t3 = await store.query_edges(edge_type="T3")
    assert len(results_t3) == 1
    assert results_t3[0].id == test_query_edges_setup["e11"].id

async def test_query_edges_by_layer(store: InMemoryKnowledgeGraphStore, test_query_edges_setup):
    """Test querying edges by layer."""
    results_l1 = await store.query_edges(layers={"L1"})
    assert len(results_l1) == 3
    assert {e.id for e in results_l1} == {
        test_query_edges_setup["e12"].id, test_query_edges_setup["e23"].id, test_query_edges_setup["e11"].id
    }

    results_l2 = await store.query_edges(layers={"L2"})
    assert len(results_l2) == 3
    assert {e.id for e in results_l2} == {
        test_query_edges_setup["e13"].id, test_query_edges_setup["e31"].id, test_query_edges_setup["e11"].id
    }

    # Test multiple layers (OR logic)
    results_l1_l2 = await store.query_edges(layers={"L1", "L2"})
    assert len(results_l1_l2) == 5 # All edges are in L1 or L2

async def test_query_edges_combined_filters(store: InMemoryKnowledgeGraphStore, test_query_edges_setup):
    """Test querying edges with combined filters."""
    n1_id = test_query_edges_setup["n1"].id
    n3_id = test_query_edges_setup["n3"].id

    # Source n1 AND type T1
    results1 = await store.query_edges(source_id=n1_id, edge_type="T1")
    assert len(results1) == 2
    assert {e.id for e in results1} == {test_query_edges_setup["e12"].id, test_query_edges_setup["e13"].id}

    # Source n1 AND type T1 AND layer L1
    results2 = await store.query_edges(source_id=n1_id, edge_type="T1", layers={"L1"})
    assert len(results2) == 1
    assert results2[0].id == test_query_edges_setup["e12"].id

    # Target n3 AND type T2
    results3 = await store.query_edges(target_id=n3_id, edge_type="T2")
    assert len(results3) == 1
    assert results3[0].id == test_query_edges_setup["e23"].id

    # Target n3 AND type T2 AND layer L2 (no match)
    results4 = await store.query_edges(target_id=n3_id, edge_type="T2", layers={"L2"})
    assert len(results4) == 0

async def test_query_edges_nonexistent_filters(store: InMemoryKnowledgeGraphStore, test_query_edges_setup):
    """Test querying edges with filters that yield no results."""
    n1_id = test_query_edges_setup["n1"].id

    results_type = await store.query_edges(edge_type="NonExistentType")
    assert len(results_type) == 0

    results_source = await store.query_edges(source_id=uuid4())
    assert len(results_source) == 0

    results_layer = await store.query_edges(layers={"NonExistentLayer"})
    assert len(results_layer) == 0

    results_combo = await store.query_edges(source_id=n1_id, edge_type="T3", layers={"NonExistentLayer"})
    assert len(results_combo) == 0
async def test_get_neighbors_outgoing(store: InMemoryKnowledgeGraphStore, test_query_edges_setup):
    """Test getting outgoing neighbors."""
    n1_id = test_query_edges_setup["n1"].id
    n2_id = test_query_edges_setup["n2"].id
    n3_id = test_query_edges_setup["n3"].id

    neighbors = await store.get_neighbors(node_id=n1_id, direction="outgoing")
    assert len(neighbors) == 3
    assert {n.id for n in neighbors} == {n1_id, n2_id, n3_id} # n2, n3, and n1 (self-loop)

async def test_get_neighbors_incoming(store: InMemoryKnowledgeGraphStore, test_query_edges_setup):
    """Test getting incoming neighbors."""
    n1_id = test_query_edges_setup["n1"].id
    n3_id = test_query_edges_setup["n3"].id

    neighbors = await store.get_neighbors(node_id=n1_id, direction="incoming")
    assert len(neighbors) == 2
    assert {n.id for n in neighbors} == {n1_id, n3_id} # n3 and n1 (self-loop)

async def test_get_neighbors_both(store: InMemoryKnowledgeGraphStore, test_query_edges_setup):
    """Test getting neighbors in both directions."""
    n1_id = test_query_edges_setup["n1"].id
    n2_id = test_query_edges_setup["n2"].id
    n3_id = test_query_edges_setup["n3"].id

    neighbors = await store.get_neighbors(node_id=n1_id, direction="both")
    assert len(neighbors) == 3
    assert {n.id for n in neighbors} == {n1_id, n2_id, n3_id} # Outgoing: n1,n2,n3. Incoming: n1,n3. Union: n1,n2,n3

async def test_get_neighbors_with_type_filter(store: InMemoryKnowledgeGraphStore, test_query_edges_setup):
    """Test getting neighbors filtered by edge type."""
    n1_id = test_query_edges_setup["n1"].id
    n2_id = test_query_edges_setup["n2"].id
    n3_id = test_query_edges_setup["n3"].id

    # Outgoing, type T1
    neighbors_out_t1 = await store.get_neighbors(node_id=n1_id, edge_type="T1", direction="outgoing")
    assert len(neighbors_out_t1) == 2
    assert {n.id for n in neighbors_out_t1} == {n2_id, n3_id}

    # Incoming, type T2
    neighbors_in_t2 = await store.get_neighbors(node_id=n1_id, edge_type="T2", direction="incoming")
    assert len(neighbors_in_t2) == 1
    assert neighbors_in_t2[0].id == n3_id

    # Both, type T3 (self-loop)
    neighbors_both_t3 = await store.get_neighbors(node_id=n1_id, edge_type="T3", direction="both")
    assert len(neighbors_both_t3) == 1
    assert neighbors_both_t3[0].id == n1_id

    # Outgoing, non-existent type
    neighbors_out_tX = await store.get_neighbors(node_id=n1_id, edge_type="TX", direction="outgoing")
    assert len(neighbors_out_tX) == 0

async def test_get_neighbors_nonexistent_node(store: InMemoryKnowledgeGraphStore):
    """Test getting neighbors for a node that doesn't exist."""
    neighbors = await store.get_neighbors(node_id=uuid4())
    assert len(neighbors) == 0
