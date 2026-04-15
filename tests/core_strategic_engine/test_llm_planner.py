"""
Advanced demo/test for the LLMStrategicPlanner with user modeling, feedback, and multi-intent decomposition.
Uses a mocked LLM client to avoid requiring an API key.
"""
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from jarvis._vendor.common.models import PlanStep, StatusEnum

from jarvis._vendor.core_strategic_engine.llm_planner import (
    LLMStrategicPlanner,
    OpenRouterLLMClient,
)


@pytest.mark.asyncio
async def test_llm_planner_advanced_prompt_user_modeling():
    # Create a mock LLM client that returns structured JSON responses
    mock_llm_client = MagicMock(spec=OpenRouterLLMClient)

    # Mock extract_intents_and_entities to return subgoals
    mock_llm_client.extract_intents_and_entities = AsyncMock(return_value=[
        {"intent": "parse_pdf", "entities": ["https://example.com/report.pdf"], "subject": "report"},
        {"intent": "summarize_knowledge", "entities": ["report"], "style": "concise"},
        {"intent": "translate", "entities": ["summary"], "target_language": "French"},
        {"intent": "send_email", "entities": ["alice@example.com", "bob@example.com"], "subject": "Translated Summary"}
    ])

    # Mock generate_plan_steps to return PlanStep objects for each subgoal
    async def mock_generate_steps(subgoal, context, feedback=None):
        intent = subgoal.get("intent", "generic_process")
        step_map = {
            "parse_pdf": [PlanStep(id=uuid4(), description="Parse PDF from URL", action_type="PDF_PARSE", parameters={"url": "https://example.com/report.pdf"}, status=StatusEnum.PENDING)],
            "summarize_knowledge": [PlanStep(id=uuid4(), description="Summarize extracted content", action_type="SUMMARIZE_DATA", parameters={"style": "concise"}, status=StatusEnum.PENDING)],
            "translate": [PlanStep(id=uuid4(), description="Translate summary to French", action_type="TRANSLATE", parameters={"target_language": "French"}, status=StatusEnum.PENDING)],
            "send_email": [PlanStep(id=uuid4(), description="Email translated summary to team", action_type="SEND_EMAIL", parameters={"recipients": ["alice@example.com", "bob@example.com"]}, status=StatusEnum.PENDING)],
        }
        return step_map.get(intent, [PlanStep(id=uuid4(), description=f"Process: {intent}", action_type="GENERIC_PROCESS", parameters={}, status=StatusEnum.PENDING)])

    mock_llm_client.generate_plan_steps = AsyncMock(side_effect=mock_generate_steps)

    planner = LLMStrategicPlanner(mock_llm_client)

    goal_id = uuid4()
    user_context = {
        "user_id": "alice",
        "preferences": {
            "language": "en",
            "summary_length": 150,
            "preferred_email": "alice@example.com",
            "style": "concise"
        },
        "history": [
            {"goal": "Summarize the latest news", "feedback": "Too verbose, make it shorter next time."}
        ]
    }
    feedback = "Last time, the summary was too long and not focused on key points. Please improve."

    # Complex, multi-intent goal
    goal_input = (
        "Summarize this PDF (https://example.com/report.pdf), translate the summary to French, "
        "and email the result to my team at alice@example.com and bob@example.com."
    )

    plan = await planner.create_plan(goal_id, goal_input, user_context, feedback)

    assert plan is not None
    assert plan.status == StatusEnum.READY
    assert len(plan.steps) > 2  # Should generate multiple steps for multiple subgoals

    # Print the plan for demo purposes
    print("\nGenerated Plan Steps (Advanced LLM, User Modeling, Feedback):")
    for i, step in enumerate(plan.steps, 1):
        print(f"{i}. {step.description} [{step.action_type}]")

    # Check that the plan includes steps for PDF parsing, summarization, translation, and email
    action_types = [step.action_type for step in plan.steps]
    assert any("PDF" in step.description or "parse" in step.description.lower() for step in plan.steps)
    assert any("SUMMARIZE" in at or "SUMMARY" in step.description.upper() for at, step in zip(action_types, plan.steps, strict=False))
    assert any("TRANSLATE" in at or "translate" in step.description.lower() for at, step in zip(action_types, plan.steps, strict=False))
    assert any("EMAIL" in at or "email" in step.description.lower() for at, step in zip(action_types, plan.steps, strict=False))
