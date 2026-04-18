"""
Unit tests for the CSC Planner implementations.
Tests the HTN decomposition logic by mocking _parse_goal to isolate
the planner from spaCy parsing behavior.
"""
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from predacore._vendor.common.models import StatusEnum
from predacore._vendor.common.protos import knowledge_nexus_pb2, knowledge_nexus_pb2_grpc

from predacore._vendor.core_strategic_engine.planner import HierarchicalStrategicPlannerV1

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_kn_stub() -> knowledge_nexus_pb2_grpc.KnowledgeNexusServiceStub:
    stub = MagicMock(spec=knowledge_nexus_pb2_grpc.KnowledgeNexusServiceStub)
    stub.QueryNodes = AsyncMock(return_value=knowledge_nexus_pb2.QueryNodesResponse(nodes=[]))
    return stub

@pytest.fixture
def planner(mock_kn_stub) -> HierarchicalStrategicPlannerV1:
    return HierarchicalStrategicPlannerV1(kn_stub=mock_kn_stub)


async def test_htn_plan_query_knowledge_simple(planner: HierarchicalStrategicPlannerV1, mock_kn_stub):
    """Test HTN plan for a simple 'query_knowledge' intent (no ambiguity)."""
    goal_id = uuid4()
    goal_input = "Find info on spaCy"
    user_context = {}
    mock_node_id = str(uuid4())

    # Mock _parse_goal so test is independent of spaCy internals
    planner._parse_goal = AsyncMock(return_value=[
        {"raw": goal_input, "intent": "query_knowledge", "entities": ["spaCy"],
         "subject": None, "relation": None, "object": None}
    ])

    # Mock KN finding exactly one node
    mock_kn_stub.QueryNodes.return_value = knowledge_nexus_pb2.QueryNodesResponse(nodes=[
        knowledge_nexus_pb2.KnowledgeNodeMessage(id=mock_node_id, labels=["Library"])
    ])

    plan = await planner.create_plan(goal_id, goal_input, user_context)

    assert plan is not None
    assert plan.goal_id == goal_id
    # Expected primitive steps from HANDLE_QUERY method (QUERY_KN -> SUMMARIZE_DATA, DISAMBIGUATE skipped)
    assert len(plan.steps) == 2
    assert plan.status == StatusEnum.READY

    # Check Step 1: Primitive QUERY_KN_ACTION
    step1 = plan.steps[0]
    assert step1.action_type == "QUERY_KN"
    assert "Query Knowledge Nexus for 'spaCy'" in step1.description
    assert step1.parameters.get("query_text") == "spaCy"

    # Check Step 2: Primitive SUMMARIZE_DATA_ACTION
    step2 = plan.steps[1]
    assert step2.action_type == "SUMMARIZE_DATA"
    assert "Summarize information related to 'spaCy'" in step2.description
    assert step2.parameters.get("target_entity") == "spaCy"
    assert step2.parameters.get("node_ids") == [mock_node_id]

    mock_kn_stub.QueryNodes.assert_called_once()  # Verify pre-fetch call

async def test_htn_plan_query_knowledge_ambiguous(planner: HierarchicalStrategicPlannerV1, mock_kn_stub):
    """Test HTN plan for 'query_knowledge' with ambiguity."""
    goal_id = uuid4()
    goal_input = "Search for Python"
    user_context = {}
    mock_node_id1 = str(uuid4())
    mock_node_id2 = str(uuid4())

    # Mock _parse_goal
    planner._parse_goal = AsyncMock(return_value=[
        {"raw": goal_input, "intent": "query_knowledge", "entities": ["Python"],
         "subject": None, "relation": None, "object": None}
    ])

    # Mock KN finding multiple nodes
    mock_kn_stub.QueryNodes.return_value = knowledge_nexus_pb2.QueryNodesResponse(nodes=[
        knowledge_nexus_pb2.KnowledgeNodeMessage(id=mock_node_id1, labels=["Language"]),
        knowledge_nexus_pb2.KnowledgeNodeMessage(id=mock_node_id2, labels=["Animal"])
    ])

    plan = await planner.create_plan(goal_id, goal_input, user_context)

    assert plan is not None
    assert plan.status == StatusEnum.READY
    # The planner decomposes query_knowledge into primitive steps
    assert len(plan.steps) >= 1
    # Verify the decomposition produced QUERY_KN action type(s)
    action_types = [s.action_type for s in plan.steps]
    assert "QUERY_KN" in action_types, f"Expected QUERY_KN in {action_types}"


async def test_htn_plan_add_relation(planner: HierarchicalStrategicPlannerV1):
    """Test HTN plan for an 'add_relation' intent."""
    goal_id = uuid4()
    goal_input = "PredaCore uses Python language"
    user_context = {}

    # Mock _parse_goal to return properly structured add_relation intent
    planner._parse_goal = AsyncMock(return_value=[
        {"raw": goal_input, "intent": "add_relation",
         "entities": ["PredaCore", "Python language"],
         "subject": "PredaCore", "relation": "USES", "object": "Python language"}
    ])

    plan = await planner.create_plan(goal_id, goal_input, user_context)

    assert plan is not None
    assert plan.status == StatusEnum.READY
    # The planner decomposes add_relation intent into steps
    assert len(plan.steps) >= 1
    # Verify the decomposition produced ADD_RELATION_KN action type
    action_types = [s.action_type for s in plan.steps]
    assert "ADD_RELATION_KN" in action_types, f"Expected ADD_RELATION_KN in {action_types}"


async def test_htn_plan_summarize(planner: HierarchicalStrategicPlannerV1, mock_kn_stub):
    """Test HTN plan for a 'summarize' intent."""
    goal_id = uuid4()
    goal_input = "Give me an overview of the DAF component"
    user_context = {}
    mock_node_id = str(uuid4())

    # Mock _parse_goal
    planner._parse_goal = AsyncMock(return_value=[
        {"raw": goal_input, "intent": "summarize_knowledge", "entities": ["DAF component"],
         "subject": None, "relation": None, "object": None}
    ])

    # Mock KN finding the node
    mock_kn_stub.QueryNodes.return_value = knowledge_nexus_pb2.QueryNodesResponse(nodes=[
        knowledge_nexus_pb2.KnowledgeNodeMessage(id=mock_node_id, labels=["Component"])
    ])

    plan = await planner.create_plan(goal_id, goal_input, user_context)

    assert plan is not None
    # Expected: QUERY_KN -> SUMMARIZE_DATA
    assert len(plan.steps) == 2
    assert plan.status == StatusEnum.READY

    step1 = plan.steps[0]
    assert step1.action_type == "QUERY_KN"
    assert step1.parameters.get("query_text") == "DAF component"

    step2 = plan.steps[1]
    assert step2.action_type == "SUMMARIZE_DATA"
    assert step2.parameters.get("target_entity") == "DAF component"
    assert step2.parameters.get("node_ids") == [mock_node_id]

    mock_kn_stub.QueryNodes.assert_called_once()


async def test_htn_plan_generic(planner: HierarchicalStrategicPlannerV1):
    """Test HTN plan for a generic/unmatched intent."""
    goal_id = uuid4()
    goal_input = "Improve system performance"
    user_context = {}

    # Mock _parse_goal
    planner._parse_goal = AsyncMock(return_value=[
        {"raw": goal_input, "intent": "generic_process", "entities": [goal_input],
         "subject": None, "relation": None, "object": None}
    ])

    plan = await planner.create_plan(goal_id, goal_input, user_context)

    assert plan is not None
    # Expected: CLASSIFY_GOAL -> GENERIC_PROCESS
    assert len(plan.steps) == 2
    assert plan.status == StatusEnum.READY

    step1 = plan.steps[0]
    assert step1.action_type == "CLASSIFY_GOAL"
    assert "Classify or route generic goal" in step1.description

    step2 = plan.steps[1]
    assert step2.action_type == "GENERIC_PROCESS"
    assert "Attempt generic processing" in step2.description

async def test_htn_plan_no_subgoals_parsed(planner: HierarchicalStrategicPlannerV1):
    """Test HTN plan when goal parsing fails."""
    goal_id = uuid4()
    goal_input = ""  # Empty input
    user_context = {}

    planner._parse_goal = AsyncMock(return_value=[])

    plan = await planner.create_plan(goal_id, goal_input, user_context)

    # Current implementation adds a fallback step when parsing returns empty
    assert plan is not None
    assert len(plan.steps) == 1
    assert plan.steps[0].action_type == "GENERIC_PROCESS"
    assert "Fallback" in plan.steps[0].description
