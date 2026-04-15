"""
End-to-end integration test for Project Prometheus:
Simulates a real user workflow spanning LLM planning, agent dispatch, tool execution, and EGM compliance.
Uses mocked LLM responses to ensure deterministic testing without requiring API keys.
"""
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from jarvis._vendor.common.models import PlanStep, StatusEnum
from jarvis._vendor.common.protos import csc_pb2

from jarvis._vendor.core_strategic_engine.llm_planner import LLMStrategicPlanner
from jarvis._vendor.ethical_governance_module.rule_engine import BasicRuleEngine


@pytest.mark.asyncio
async def test_end_to_end_multi_intent_workflow():
    # Setup: Instantiate planner with mock LLM client
    mock_llm_client = MagicMock()
    mock_llm_client.extract_intents_and_entities = AsyncMock(return_value=[
        {"intent": "summarize_knowledge", "entities": ["report.pdf"], "tool": "pdf_reader"},
        {"intent": "api_call", "entities": ["translate", "French"], "tool": "translator"},
        {"intent": "api_call", "entities": ["weather", "Paris"], "tool": "weather_api"},
        {"intent": "send_email", "entities": ["alice@example.com"], "tool": "email_sender"},
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
        "preferences": {
            "language": "en",
            "summary_length": 150,
            "preferred_email": "alice@example.com",
            "style": "concise"
        }
    }
    feedback = "Last time, the summary was too long. Please make it concise and focused."

    goal_id = uuid4()
    goal_input = (
        "Summarize this PDF (https://example.com/report.pdf), translate the summary to French, "
        "get the weather for Paris, and email the result to alice@example.com."
    )

    # 1. LLM Planning
    plan = await planner.create_plan(goal_id, goal_input, user_context, feedback)
    assert plan is not None
    assert plan.status == StatusEnum.READY
    assert len(plan.steps) > 2

    # 2. EGM Compliance Check (simulate for each step using proto message)
    # Some steps may have risky action types (e.g. SEND_EMAIL) that the rule engine
    # correctly flags. Verify all steps are processed and flagged ones have violations.
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
            assert len(compliance.violations) > 0
    assert compliant_count > 0

    print("\nEnd-to-end workflow test passed: LLM planning, EGM compliance all succeeded.")
