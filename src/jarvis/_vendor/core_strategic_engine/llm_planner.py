"""
LLM-based Strategic Planner for Project Prometheus.
Uses a provider-agnostic LLM client (OpenAI/OpenRouter) via src.common.llm.
"""
import logging
import os
from typing import Any, Optional
from uuid import UUID, uuid4

from jarvis._vendor.common.llm import default_params, get_default_llm_client
from jarvis._vendor.common.models import Plan, PlanStep, StatusEnum


class OpenRouterLLMClient:
    """
    Back-compat wrapper preserved for tests; delegates to provider-agnostic client.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        logger: logging.Logger | None = None,
    ):
        # NOTE: Ignoring explicit api_key/model in favor of env-based configuration via common.llm
        self.logger = logger or logging.getLogger(__name__)
        self._llm = get_default_llm_client(logger=self.logger)
        self.model = model or os.getenv("LLM_MODEL")
        self.logger.info(
            f"LLM client initialized (provider-agnostic), model={self.model or 'env default'}"
        )

    async def extract_intents_and_entities(
        self, goal_input: str, user_context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        # Advanced prompt with user modeling and feedback loop
        prompt = (
            "You are Prometheus, a world-class AI agent planner. "
            "Your job is to deeply understand the user's goal, decompose it into a sequence of subgoals, "
            "and extract all relevant intents, entities, and parameters for each subgoal. "
            "You must consider the user's context, preferences, and any prior feedback. "
            "For each subgoal, output a JSON object with keys: 'intent' (e.g., 'query_knowledge', 'add_relation', 'summarize_knowledge', 'send_email', 'api_call', 'code_execution', 'generic_process'), "
            "'entities' (list of relevant entities or parameters), and any additional fields (e.g., 'subject', 'object', 'relation', 'tool', 'target_user', etc.). "
            "If the goal is ambiguous, ask clarifying questions in the 'clarification' field. "
            "If the user has a known preference or style, adapt the plan accordingly.\n"
            f"User context: {user_context or {}}\n"
            f"User goal: {goal_input}\n"
            "Subgoals (JSON array):"
        )
        response = await self._call_llm(prompt)
        parsed = self._extract_json_from_llm(response, expect_array=True)
        if parsed is not None:
            return parsed
        self.logger.warning(
            "LLM did not return valid JSON array. Returning a single generic subgoal."
        )
        return [{"intent": "generic_process", "entities": [goal_input]}]

    async def generate_plan_steps(
        self,
        subgoal: dict[str, Any],
        context: dict[str, Any],
        feedback: str | None = None,
    ) -> list[PlanStep]:
        # Advanced prompt with context and feedback loop
        prompt = (
            "You are Prometheus, a world-class AI agent planner. "
            "Given the following subgoal, user context, and any prior feedback, generate a robust, multi-step plan. "
            "Each step should have a clear description, action_type (e.g., 'QUERY_KN', 'API_CALL', 'CODE_EXECUTION', 'SEND_EMAIL', 'SUMMARIZE_DATA', 'GENERIC_PROCESS'), and parameters. "
            "If the subgoal is ambiguous, include a 'clarification' step. "
            "If the user has a known preference or style, adapt the plan accordingly. "
            "If you have feedback from previous attempts, use it to improve the plan.\n"
            f"Subgoal: {subgoal}\n"
            f"User context: {context}\n"
            f"Feedback: {feedback or 'None'}\n"
            "Plan steps (JSON array):"
        )
        response = await self._call_llm(prompt)
        parsed = self._extract_json_from_llm(response, expect_array=True)
        if parsed is not None and isinstance(parsed, list):
            steps = []
            for step in parsed:
                steps.append(
                    PlanStep(
                        id=uuid4(),
                        description=step.get("description", ""),
                        action_type=step.get("action_type", "GENERIC_PROCESS"),
                        parameters=step.get("parameters", {}),
                        status=StatusEnum.PENDING,
                    )
                )
            return steps
        self.logger.warning(
            "LLM did not return valid JSON array of steps. Returning a single generic step."
        )
        return [
            PlanStep(
                id=uuid4(),
                description=f"Attempt generic processing for subgoal: {subgoal}",
                action_type="GENERIC_PROCESS",
                parameters=subgoal,
                status=StatusEnum.PENDING,
            )
        ]

    async def _call_llm(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are Prometheus, a world-class AI agent planner. Always output valid JSON. Do not wrap in markdown code fences.",
            },
            {"role": "user", "content": prompt},
        ]
        params = default_params(temperature=0.15, max_tokens=1024)
        return await self._llm.generate(messages, model=self.model, params=params)

    def _extract_json_from_llm(self, response: str, expect_array: bool = False):
        """
        Robust JSON extraction from LLM response.
        Handles: direct JSON, markdown code fences, embedded JSON in text.
        """
        import json
        import re

        # Strategy 1: Try direct parse
        try:
            result = json.loads(response.strip())
            if expect_array and not isinstance(result, list):
                return None
            return result
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code fence
        fence_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        fence_match = re.search(fence_pattern, response)
        if fence_match:
            try:
                result = json.loads(fence_match.group(1).strip())
                if expect_array and not isinstance(result, list):
                    return None
                return result
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find JSON array/object in response
        bracket = r"\[.*\]" if expect_array else r"\{.*\}"
        match = re.search(bracket, response, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                return result
            except json.JSONDecodeError as e:
                self.logger.debug(f"Regex-extracted JSON failed to parse: {e}")

        self.logger.error(
            f"Failed to extract JSON from LLM response: {response[:200]}..."
        )
        return None


class LLMStrategicPlanner:
    """
    LLM-based planner for robust, context-aware, and multi-intent planning.
    """

    def __init__(
        self, llm_client: OpenRouterLLMClient, logger: logging.Logger | None = None
    ):
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(
            "LLMStrategicPlanner (provider-agnostic LLM, advanced prompt engineering) initialized."
        )

    async def create_plan(
        self,
        goal_id: UUID,
        goal_input: str,
        user_context: dict[str, Any],
        feedback: str | None = None,
    ) -> Plan | None:
        self.logger.info(
            f"LLMStrategicPlanner: Creating plan for goal {goal_id} - '{goal_input[:50]}...'"
        )
        subgoals = await self.llm_client.extract_intents_and_entities(
            goal_input, user_context
        )
        all_steps = []
        for subgoal in subgoals:
            steps = await self.llm_client.generate_plan_steps(
                subgoal, user_context, feedback
            )
            all_steps.extend(steps)
        if not all_steps:
            self.logger.error(
                f"LLMStrategicPlanner failed to generate any steps for goal {goal_id}"
            )
            return None
        plan = Plan(goal_id=goal_id, steps=all_steps, status=StatusEnum.READY)
        self.logger.info(
            f"LLMStrategicPlanner: Generated plan {plan.id} with {len(all_steps)} steps for goal {goal_id}."
        )
        return plan
