"""
Integration tests for real-world, edge-case user scenarios in Project Prometheus.
Covers multi-user collaboration, error handling, fallback logic, and agent/tool robustness.
Uses mocked LLM responses for deterministic testing.
"""
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from jarvis._vendor.common.models import PlanStep, StatusEnum
from jarvis._vendor.common.protos import csc_pb2

from jarvis._vendor.core_strategic_engine.llm_planner import LLMStrategicPlanner
from jarvis._vendor.ethical_governance_module.rule_engine import BasicRuleEngine


@pytest.mark.asyncio
async def test_multi_user_collaboration():
    mock_llm_client = MagicMock()
    mock_llm_client.extract_intents_and_entities = AsyncMock(return_value=[
        {"intent": "summarize_knowledge", "entities": ["report.pdf"]},
        {"intent": "api_call", "entities": ["translate", "Spanish"]},
        {"intent": "send_email", "entities": ["alice@example.com", "bob@example.com"]},
    ])
    mock_llm_client.generate_plan_steps = AsyncMock(side_effect=lambda subgoal, ctx, feedback=None: [
        PlanStep(
            description=f"Execute {subgoal.get('intent', 'unknown')} for {subgoal.get('entities', [])}",
            action_type=subgoal.get("intent", "GENERIC_PROCESS").upper(),
            parameters=subgoal,
            status=StatusEnum.PENDING
        )
    ])

    rule_engine = BasicRuleEngine()
    planner = LLMStrategicPlanner(mock_llm_client)

    user_context = {
        "user_id": "alice",
        "team": ["alice@example.com", "bob@example.com"],
        "preferences": {"language": "en", "style": "detailed"}
    }
    goal_id = uuid4()
    goal_input = (
        "Summarize this PDF (https://example.com/report.pdf), translate the summary to Spanish, "
        "and email the result to my team."
    )

    plan = await planner.create_plan(goal_id, goal_input, user_context)
    assert plan is not None
    assert plan.status == StatusEnum.READY
    assert len(plan.steps) > 2

    # EGM compliance check for each step using proto messages
    # Some steps may use risky action types (e.g. SEND_EMAIL) which are correctly
    # flagged by the rule engine. Verify the engine processes all steps without error.
    compliant_count = 0
    flagged_count = 0
    for step in plan.steps:
        proto_step = csc_pb2.PlanStepMessage(
            description=step.description,
            action_type=step.action_type,
        )
        compliance = rule_engine.check_compliance(proto_step)
        if compliance.is_compliant:
            compliant_count += 1
        else:
            flagged_count += 1
            # Verify flagged steps have actual violations recorded
            assert len(compliance.violations) > 0
    # At least some steps should be compliant
    assert compliant_count > 0

@pytest.mark.asyncio
async def test_error_handling_and_fallback():
    mock_llm_client = MagicMock()
    mock_llm_client.extract_intents_and_entities = AsyncMock(return_value=[
        {"intent": "api_call", "entities": ["weather", "Atlantis"]},
        {"intent": "generic_process", "entities": ["scrape weather website"]},
    ])
    mock_llm_client.generate_plan_steps = AsyncMock(side_effect=lambda subgoal, ctx, feedback=None: [
        PlanStep(
            description=f"Execute {subgoal.get('intent', 'unknown')} for {subgoal.get('entities', [])}",
            action_type=subgoal.get("intent", "GENERIC_PROCESS").upper(),
            parameters=subgoal,
            status=StatusEnum.PENDING
        )
    ])

    planner = LLMStrategicPlanner(mock_llm_client)

    user_context = {"user_id": "bob", "preferences": {"language": "en"}}
    goal_id = uuid4()
    goal_input = (
        "Get the weather for Atlantis (a fictional city), and if not available, try to scrape a weather website."
    )

    plan = await planner.create_plan(goal_id, goal_input, user_context)
    assert plan is not None
    assert plan.status == StatusEnum.READY
    assert len(plan.steps) >= 1

@pytest.mark.asyncio
async def test_edge_case_empty_goal():
    mock_llm_client = MagicMock()
    mock_llm_client.extract_intents_and_entities = AsyncMock(return_value=[
        {"intent": "generic_process", "entities": [""]}
    ])
    mock_llm_client.generate_plan_steps = AsyncMock(return_value=[
        PlanStep(
            description="Attempt generic processing for empty goal",
            action_type="GENERIC_PROCESS",
            parameters={"intent": "generic_process"},
            status=StatusEnum.PENDING
        )
    ])

    planner = LLMStrategicPlanner(mock_llm_client)

    user_context = {"user_id": "charlie"}
    goal_id = uuid4()
    goal_input = ""

    plan = await planner.create_plan(goal_id, goal_input, user_context)
    assert plan is not None
    assert plan.status == StatusEnum.READY
    assert len(plan.steps) == 1
    assert "generic" in plan.steps[0].action_type.lower() or "process" in plan.steps[0].action_type.lower()

@pytest.mark.asyncio
async def test_edge_case_ambiguous_goal():
    mock_llm_client = MagicMock()
    mock_llm_client.extract_intents_and_entities = AsyncMock(return_value=[
        {"intent": "generic_process", "entities": ["Do the thing."]}
    ])
    mock_llm_client.generate_plan_steps = AsyncMock(return_value=[
        PlanStep(
            description="Generic processing for ambiguous goal: Do the thing.",
            action_type="GENERIC_PROCESS",
            parameters={"intent": "generic_process"},
            status=StatusEnum.PENDING
        )
    ])

    planner = LLMStrategicPlanner(mock_llm_client)

    user_context = {"user_id": "dana"}
    goal_id = uuid4()
    goal_input = "Do the thing."

    plan = await planner.create_plan(goal_id, goal_input, user_context)
    assert plan is not None
    assert plan.status == StatusEnum.READY
    assert len(plan.steps) == 1
    assert "generic" in plan.steps[0].action_type.lower()
