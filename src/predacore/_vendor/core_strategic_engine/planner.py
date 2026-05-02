"""
Implementation of the Hierarchical Strategic Planner (v1).
"""
import logging
import re  # Import regex module

# from .service import AbstractHierarchicalPlanner # Import the abstract class if defined separately
from dataclasses import dataclass, field  # Added dataclasses
from typing import Any
from uuid import UUID

import spacy

# Import necessary components and models. The Knowledge Nexus service was
# retired (PR 1.7); the planner used to query it for entity context but
# now operates from the goal text + LLM-driven decomposition alone.
from predacore._vendor.common.models import Plan, PlanStep, StatusEnum

# --- Add these new structures after the imports ---


@dataclass
class Task:
    """Represents a task in the HTN."""

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class Method:
    """Represents a method to decompose an abstract task."""

    name: str
    task_to_decompose: str  # Name of the abstract task this method applies to
    # Subtasks can be other abstract tasks or primitive task names
    subtasks: list[Task] = field(default_factory=list)
    # Optional: Add preconditions later if needed
    # preconditions: Callable[[Dict[str, Any]], bool] = lambda state: True


# --- Keep the existing PlanStep and Plan models ---
# from common.models import Plan, PlanStep, StatusEnum


# --- Keep the AbstractHierarchicalPlanner class ---
class AbstractHierarchicalPlanner:
    async def create_plan(
        self, goal_id: UUID, goal_input: str, user_context: dict[str, Any]
    ) -> Plan | None:
        raise NotImplementedError


# --- Modify the HierarchicalStrategicPlannerV1 class ---


class HierarchicalStrategicPlannerV1(AbstractHierarchicalPlanner):
    """
    Planner using a basic Hierarchical Task Network (HTN) approach.
    """

    def __init__(
        self,
        kn_stub: Any = None,  # retained for backward compat — KN was retired
        logger: logging.Logger | None = None,
    ):
        # kn_stub is accepted but unused; callers that still pass it (the
        # MCTS planner does, threading None through from subsystem_init)
        # don't need to change.
        self.kn_stub = None
        self.logger = logger or logging.getLogger(__name__)
        # --- Existing spaCy loading ---
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("spaCy English model loaded for robust goal parsing.")
        except Exception as e:
            self.logger.warning(f"spaCy model not available: {e}")
            self.nlp = None

        # --- NEW: Define Primitive Tasks (actions that can be directly executed) ---
        self.primitive_tasks = {
            "QUERY_KN_ACTION",
            "ADD_RELATION_KN_ACTION",
            "ENSURE_NODE_KN_ACTION",  # Renamed for clarity
            "SUMMARIZE_DATA_ACTION",  # Renamed for clarity
            "DISAMBIGUATE_ENTITY_ACTION",  # Renamed for clarity
            "CLASSIFY_GOAL_ACTION",  # Renamed for clarity
            "GENERIC_PROCESS_ACTION",  # Renamed for clarity
            # Add more primitive actions as needed
        }

        # --- NEW: Define HTN Methods (simple examples) ---
        self.methods = {
            # Method to handle the overall goal
            "method_achieve_goal": Method(
                name="method_achieve_goal",
                task_to_decompose="ACHIEVE_GOAL",
                subtasks=[
                    Task(
                        name="PARSE_AND_ROUTE_GOAL", parameters={"goal_input": None}
                    )  # Parameter filled at runtime
                ],
            ),
            # Method to parse the goal and decide the next step based on intent
            "method_parse_and_route": Method(
                name="method_parse_and_route",
                task_to_decompose="PARSE_AND_ROUTE_GOAL",
                # Contains a single placeholder subtask; the _decompose method dynamically
                # replaces it based on the parsed intent (see lines 287-303)
                subtasks=[Task(name="ROUTE_PLACEHOLDER", parameters={})],
            ),
            # Method for handling a 'query_knowledge' intent
            "method_handle_query": Method(
                name="method_handle_query",
                task_to_decompose="HANDLE_QUERY",  # Abstract task for this intent
                subtasks=[
                    Task(
                        name="QUERY_KN_ACTION",
                        parameters={
                            "query_text": None,
                            "labels": ["Concept", "Entity"],
                        },
                    ),
                    Task(
                        name="DISAMBIGUATE_ENTITY_ACTION",
                        parameters={"query_term": None, "ambiguous_node_ids": None},
                    ),  # Conditional
                    Task(
                        name="SUMMARIZE_DATA_ACTION",
                        parameters={"target_entity": None, "node_ids": None},
                    ),
                ],
            ),
            # Method for handling 'add_relation' intent
            "method_handle_add_relation": Method(
                name="method_handle_add_relation",
                task_to_decompose="HANDLE_ADD_RELATION",  # Abstract task
                subtasks=[
                    Task(
                        name="ENSURE_NODE_KN_ACTION",
                        parameters={"name": None, "labels": ["Entity"]},
                    ),  # Subject
                    Task(
                        name="ENSURE_NODE_KN_ACTION",
                        parameters={"name": None, "labels": ["Entity"]},
                    ),  # Object
                    Task(
                        name="ADD_RELATION_KN_ACTION",
                        parameters={
                            "subject_name": None,
                            "relation_type": None,
                            "object_name": None,
                        },
                    ),
                ],
            ),
            # Method for handling 'summarize_knowledge' intent
            "method_handle_summarize": Method(
                name="method_handle_summarize",
                task_to_decompose="HANDLE_SUMMARIZE",  # Abstract task
                subtasks=[
                    Task(
                        name="QUERY_KN_ACTION",
                        parameters={
                            "query_text": None,
                            "labels": ["Concept", "Entity"],
                        },
                    ),
                    Task(
                        name="SUMMARIZE_DATA_ACTION",
                        parameters={"target_entity": None, "node_ids": None},
                    ),
                ],
            ),
            # Method for handling generic goals
            "method_handle_generic": Method(
                name="method_handle_generic",
                task_to_decompose="HANDLE_GENERIC",  # Abstract task
                subtasks=[
                    Task(name="CLASSIFY_GOAL_ACTION", parameters={"input": None}),
                    Task(name="GENERIC_PROCESS_ACTION", parameters={"input": None}),
                ],
            ),
            # Add more methods for different abstract tasks
        }
        self.logger.info(
            "HierarchicalStrategicPlannerV1 initialized with basic HTN structures."
        )

    # --- Keep existing _parse_goal and _query_knowledge methods ---
    # (We might need to adjust _query_knowledge later to fit the HTN flow better)
    async def _parse_goal(self, goal_input: str) -> list[dict[str, Any]]:
        """Robust goal parsing using spaCy NLP if available, fallback to keyword/regex. Supports multi-intent/entity goals."""
        self.logger.debug(f"Parsing goal: '{goal_input[:50]}...'")
        subgoals = []

        # Split on common conjunctions for compound goals (e.g., "and", "then", ";")
        # This is a simple heuristic; more advanced logic could use dependency parsing.
        parts = re.split(r"\b(?:and|then|;)\b", goal_input, flags=re.IGNORECASE)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            parsed = {
                "raw": part,
                "intent": "unknown",
                "entities": [],
                "subject": None,
                "relation": None,
                "object": None,
            }  # Added relation fields
            if self.nlp:
                doc = self.nlp(part)
                verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
                noun_chunks = [chunk.text for chunk in doc.noun_chunks]
                entities = [ent.text for ent in doc.ents]

                # --- Intent detection logic (simplified for brevity, keep original if preferred) ---
                if any(
                    v in ["search", "find", "look", "query", "what is", "who is"]
                    for v in verbs
                ):
                    parsed["intent"] = "query_knowledge"
                    parsed["entities"] = entities or noun_chunks
                elif any(v in ["add", "relate", "link", "connect"] for v in verbs):
                    parsed["intent"] = "add_relation"
                    # Simplified relation extraction (keep original if more robust)
                    if len(entities) >= 2:
                        parsed["subject"] = entities[0]
                        parsed["object"] = entities[-1]  # Assume last entity is object
                        # Find relation verb between subject and object
                        relation_found = False
                        for token in doc:
                            if (
                                token.head.text == parsed["subject"]
                                and token.dep_ == "dobj"
                                and token.head.head.text == parsed["object"]
                            ):
                                parsed["relation"] = token.head.lemma_.upper()
                                relation_found = True
                                break
                            if (
                                token.head.text == parsed["object"]
                                and token.dep_ == "nsubj"
                                and token.head.head.text == parsed["subject"]
                            ):
                                parsed["relation"] = token.lemma_.upper()
                                relation_found = True
                                break
                        if not relation_found:  # Fallback if complex structure
                            for token in doc:
                                if token.pos_ == "VERB" and token.lemma_ not in [
                                    "add",
                                    "relate",
                                    "link",
                                    "connect",
                                ]:
                                    parsed["relation"] = token.lemma_.upper()
                                    break
                    else:  # Fallback regex from original
                        text_part = part.split(":")[-1].strip()
                        relation_keywords = [
                            "relates to",
                            "part of",
                            "is a",
                            "uses",
                            "has",
                            "contains",
                        ]
                        relation_pattern_str = (
                            r"^(.*?)\s+("
                            + "|".join(re.escape(kw) for kw in relation_keywords)
                            + r")\s+(.*)$"
                        )
                        relation_pattern = re.compile(
                            relation_pattern_str, re.IGNORECASE
                        )
                        match = relation_pattern.search(text_part)
                        if match:
                            parsed["subject"] = match.group(1).strip()
                            relation_kw = match.group(2).strip().lower()
                            parsed["relation"] = relation_kw.upper().replace(" ", "_")
                            parsed["object"] = match.group(3).strip()

                elif any(v in ["summarize", "overview", "report"] for v in verbs):
                    parsed["intent"] = "summarize_knowledge"
                    parsed["entities"] = entities or noun_chunks
                else:
                    parsed["intent"] = "generic_process"
                    parsed["entities"] = entities or noun_chunks

                self.logger.debug(f"spaCy NLP parsed subgoal: {parsed}")

            else:
                # --- Fallback keyword/regex logic (simplified for brevity, keep original if preferred) ---
                part_lower = part.lower()
                if (
                    "find information about" in part_lower
                    or "search for" in part_lower
                    or "what is" in part_lower
                ):
                    parsed["intent"] = "query_knowledge"
                    parsed["entities"] = [
                        part.split("about")[-1].split("for")[-1].split("is")[-1].strip()
                    ]
                elif (
                    "add relation" in part_lower
                    or " relation:" in part_lower
                    or re.search(
                        r"\b(relates to|part of|is a|uses|has|contains)\b", part_lower
                    )
                ):
                    parsed["intent"] = "add_relation"
                    # Fallback regex from original
                    text_part = part.split(":")[-1].strip()
                    relation_keywords = [
                        "relates to",
                        "part of",
                        "is a",
                        "uses",
                        "has",
                        "contains",
                    ]
                    relation_pattern_str = (
                        r"^(.*?)\s+("
                        + "|".join(re.escape(kw) for kw in relation_keywords)
                        + r")\s+(.*)$"
                    )
                    relation_pattern = re.compile(relation_pattern_str, re.IGNORECASE)
                    match = relation_pattern.search(text_part)
                    if match:
                        parsed["subject"] = match.group(1).strip()
                        relation_kw = match.group(2).strip().lower()
                        parsed["relation"] = relation_kw.upper().replace(" ", "_")
                        parsed["object"] = match.group(3).strip()
                elif "summarize" in part_lower:
                    parsed["intent"] = "summarize_knowledge"
                    parsed["entities"] = [part.split("summarize")[-1].strip()]
                else:
                    parsed["intent"] = "generic_process"

                self.logger.debug(f"Keyword/Regex parsed subgoal: {parsed}")

            subgoals.append(parsed)
        return subgoals

    async def _query_knowledge(self, entity: str) -> dict[str, Any]:
        """Stub: KN service was retired in PR 1.7.

        The planner used to call out to the gRPC Knowledge Nexus to fetch
        entity nodes for query/summarize intents. The replacement story
        lives in ``predacore.memory.UnifiedMemoryStore`` (entity extraction
        + retrieval), but wiring that into the HTN goal-routing path is a
        separate piece of work. For now this returns an empty result and
        downstream HTN routing falls back to the goal text alone.
        """
        self.logger.debug(
            "Knowledge Nexus query suppressed (service retired): entity=%s",
            entity,
        )
        return {"relevant_nodes": []}

    # --- Remove old _select_strategy and _generate_steps ---

    # --- NEW: Recursive Decomposition Logic ---
    async def _decompose(
        self,
        task: Task,
        current_plan: list[PlanStep],
        knowledge_context: dict[str, Any],
    ):
        """Recursively decomposes tasks using methods until only primitive actions remain."""
        self.logger.debug(
            f"Decomposing task: {task.name} with params {task.parameters}"
        )

        if task.name in self.primitive_tasks:
            self.logger.debug(f"Task {task.name} is primitive. Adding to plan.")
            # Convert primitive task to PlanStep
            # We need a mapping or logic here
            plan_step = self._create_plan_step_from_primitive(task, knowledge_context)
            if plan_step:
                current_plan.append(plan_step)
            return True  # Indicate success

        # Find applicable methods for the abstract task
        applicable_methods = [
            m for m in self.methods.values() if m.task_to_decompose == task.name
        ]

        if not applicable_methods:
            self.logger.error(
                f"No methods found to decompose abstract task: {task.name}"
            )
            return False  # Indicate failure

        # Simple strategy: Use the first applicable method
        # TODO: Implement more sophisticated method selection later
        selected_method = applicable_methods[0]
        self.logger.debug(
            f"Selected method '{selected_method.name}' for task '{task.name}'"
        )

        # Decompose subtasks defined in the method
        for subtask_template in selected_method.subtasks:
            # Create a concrete subtask instance, filling parameters from the parent task
            # or knowledge context
            subtask_instance = Task(
                name=subtask_template.name,
                parameters=self._fill_subtask_parameters(
                    subtask_template.parameters, task.parameters, knowledge_context
                ),
            )

            # --- Dynamic Subtask Generation for PARSE_AND_ROUTE_GOAL ---
            if task.name == "PARSE_AND_ROUTE_GOAL":
                # This task needs special handling based on parsing result
                parsed_goal = task.parameters.get("parsed_goal")
                if not parsed_goal:
                    self.logger.error(
                        "PARSE_AND_ROUTE_GOAL task missing 'parsed_goal' parameter."
                    )
                    return False

                intent = parsed_goal.get("intent", "generic_process")
                self.logger.debug(f"Routing based on intent: {intent}")
                if intent == "query_knowledge":
                    subtask_instance = Task(name="HANDLE_QUERY", parameters=parsed_goal)
                elif intent == "add_relation":
                    subtask_instance = Task(
                        name="HANDLE_ADD_RELATION", parameters=parsed_goal
                    )
                elif intent == "summarize_knowledge":
                    subtask_instance = Task(
                        name="HANDLE_SUMMARIZE", parameters=parsed_goal
                    )
                else:  # generic_process or unknown
                    subtask_instance = Task(
                        name="HANDLE_GENERIC", parameters=parsed_goal
                    )

            # --- Conditional Subtask Execution (Example for Disambiguation) ---
            if subtask_instance.name == "DISAMBIGUATE_ENTITY_ACTION":
                node_ids = knowledge_context.get(
                    f"nodes_{subtask_instance.parameters.get('query_term')}", []
                )
                if len(node_ids) <= 1:
                    self.logger.debug(
                        f"Skipping disambiguation for '{subtask_instance.parameters.get('query_term')}' - not ambiguous."
                    )
                    continue  # Skip this subtask if not needed
                else:
                    # Update parameters with actual node IDs found
                    subtask_instance.parameters["ambiguous_node_ids"] = node_ids

            # Recursively decompose the instantiated subtask
            success = await self._decompose(
                subtask_instance, current_plan, knowledge_context
            )
            if not success:
                self.logger.error(
                    f"Failed to decompose subtask: {subtask_instance.name}"
                )
                return False  # Propagate failure up

        return True  # Indicate successful decomposition of this branch

    def _fill_subtask_parameters(
        self,
        subtask_params_template: dict[str, Any],
        parent_params: dict[str, Any],
        knowledge: dict[str, Any],
    ) -> dict[str, Any]:
        """Fills parameters for a subtask instance."""
        filled_params = {}
        for key, template_value in subtask_params_template.items():
            # If template_value is None, try to get it from parent or knowledge
            if template_value is None:
                if key in parent_params:
                    filled_params[key] = parent_params[key]
                elif (
                    key == "ambiguous_node_ids"
                ):  # Special handling for knowledge context
                    query_term = (
                        parent_params.get("query_term")
                        or parent_params.get("entities", [None])[0]
                    )
                    if query_term:
                        filled_params[key] = knowledge.get(f"nodes_{query_term}", [])
                elif key == "node_ids":  # Special handling for knowledge context
                    target_entity = (
                        parent_params.get("target_entity")
                        or parent_params.get("entities", [None])[0]
                    )
                    if target_entity:
                        filled_params[key] = knowledge.get(f"nodes_{target_entity}", [])
                # Add more specific logic for pulling from knowledge context if needed
                else:
                    # Parameter not found, keep as None or log warning
                    filled_params[key] = None
                    self.logger.warning(
                        f"Parameter '{key}' could not be filled for subtask."
                    )
            else:
                # Use the template value directly (could be a default)
                filled_params[key] = template_value
        # Also copy relevant parameters directly from parent if not explicitly in template
        # (e.g., subject, object, relation for ADD_RELATION_KN_ACTION)
        relevant_parent_keys = [
            "subject",
            "relation",
            "object",
            "subject_name",
            "relation_type",
            "object_name",
            "entities",
            "query_text",
            "target_entity",
            "input",
            "raw",
        ]
        for key in relevant_parent_keys:
            if key in parent_params and key not in filled_params:
                filled_params[key] = parent_params[key]

        # Map parent param names to subtask param names if needed
        if "subject" in parent_params and "subject_name" in subtask_params_template:
            filled_params["subject_name"] = parent_params["subject"]
        if "object" in parent_params and "object_name" in subtask_params_template:
            filled_params["object_name"] = parent_params["object"]
        if "relation" in parent_params and "relation_type" in subtask_params_template:
            filled_params["relation_type"] = parent_params["relation"]
        if "entities" in parent_params and "query_text" in subtask_params_template:
            filled_params["query_text"] = (
                parent_params["entities"][0] if parent_params["entities"] else None
            )
        if "entities" in parent_params and "target_entity" in subtask_params_template:
            filled_params["target_entity"] = (
                parent_params["entities"][0] if parent_params["entities"] else None
            )
        if "raw" in parent_params and "input" in subtask_params_template:
            filled_params["input"] = parent_params["raw"]

        return filled_params

    def _create_plan_step_from_primitive(
        self, task: Task, knowledge: dict[str, Any]
    ) -> PlanStep | None:
        """Maps a primitive task to a PlanStep object."""
        params = task.parameters
        action_type = task.name.replace(
            "_ACTION", ""
        )  # Convention: remove _ACTION suffix

        # --- Logic adapted from old _generate_steps ---
        if task.name == "QUERY_KN_ACTION":
            query_term = params.get("query_text")
            if not query_term:
                return None
            return PlanStep(
                description=f"Query Knowledge Nexus for '{query_term}'",
                action_type=action_type,
                parameters=params,
                status=StatusEnum.PENDING,
            )
        elif task.name == "ADD_RELATION_KN_ACTION":
            subj = params.get("subject_name")
            rel = params.get("relation_type")
            obj = params.get("object_name")
            if not (subj and rel and obj):
                return None
            return PlanStep(
                description=f"Add relation '{subj} {rel} {obj}' to KN",
                action_type=action_type,
                parameters=params,
                status=StatusEnum.PENDING,
            )
        elif task.name == "ENSURE_NODE_KN_ACTION":
            node_name = params.get("name")
            if not node_name:
                return None
            return PlanStep(
                description=f"Ensure node '{node_name}' exists in KN",
                action_type=action_type,
                parameters=params,
                status=StatusEnum.PENDING,
            )
        elif task.name == "SUMMARIZE_DATA_ACTION":
            target = params.get("target_entity")
            if not target:
                return None
            # Pass relevant node IDs found during planning
            params["node_ids"] = knowledge.get(f"nodes_{target}", [])
            return PlanStep(
                description=f"Summarize information related to '{target}'",
                action_type=action_type,
                parameters=params,
                status=StatusEnum.PENDING,
            )
        elif task.name == "DISAMBIGUATE_ENTITY_ACTION":
            query_term = params.get("query_term")
            node_ids = params.get("ambiguous_node_ids", [])
            if not query_term or len(node_ids) <= 1:
                return None  # Should have been skipped in _decompose
            return PlanStep(
                description=f"Disambiguate entity '{query_term}' among found nodes.",
                action_type=action_type,
                parameters=params,
                status=StatusEnum.PENDING,
            )
        elif task.name == "CLASSIFY_GOAL_ACTION":
            goal_input = params.get("input", "")
            return PlanStep(
                description=f"Classify or route generic goal: {goal_input[:50]}...",
                action_type=action_type,
                parameters=params,
                status=StatusEnum.PENDING,
            )
        elif task.name == "GENERIC_PROCESS_ACTION":
            goal_input = params.get("input", "")
            return PlanStep(
                description="Attempt generic processing based on classification",
                action_type=action_type,
                parameters=params,
                status=StatusEnum.PENDING,
            )
        else:
            self.logger.warning(
                f"No PlanStep mapping found for primitive task: {task.name}"
            )
            return None

    # --- OVERWRITE create_plan ---
    async def create_plan(
        self, goal_id: UUID, goal_input: str, user_context: dict[str, Any]
    ) -> Plan | None:
        """
        Generates a plan using HTN decomposition.
        """
        self.logger.info(
            f"Planner HTN: Creating plan for goal {goal_id} - '{goal_input[:50]}...'"
        )
        final_plan_steps: list[PlanStep] = []
        knowledge_context: dict[
            str, Any
        ] = {}  # Store intermediate results like node IDs

        # 1. Parse Goal (potentially multiple subgoals)
        parsed_subgoals = await self._parse_goal(goal_input)

        if not parsed_subgoals:
            self.logger.error(f"Planner failed to parse goal: {goal_input}")
            # Add a fallback generic step instead of returning None
            final_plan_steps.append(
                PlanStep(
                    description=f"Fallback: Attempt generic processing for goal: {goal_input}",
                    action_type="GENERIC_PROCESS",
                    parameters={"input": goal_input},
                    status=StatusEnum.PENDING,
                )
            )
            plan = Plan(
                goal_id=goal_id, steps=final_plan_steps, status=StatusEnum.READY
            )
            return plan

        # For simplicity, handle the first parsed subgoal for now
        # TODO: Extend to handle multiple sequential or parallel subgoals later
        initial_parsed_goal = parsed_subgoals[0]

        # --- Pre-fetch initial knowledge if needed by routing ---
        # Example: If query intent, find nodes before decomposition starts fully
        if (
            initial_parsed_goal["intent"] == "query_knowledge"
            and initial_parsed_goal["entities"]
        ):
            entity = initial_parsed_goal["entities"][0]
            kn_result = await self._query_knowledge(entity)
            knowledge_context[f"nodes_{entity}"] = kn_result.get("relevant_nodes", [])
        elif (
            initial_parsed_goal["intent"] == "summarize_knowledge"
            and initial_parsed_goal["entities"]
        ):
            entity = initial_parsed_goal["entities"][0]
            kn_result = await self._query_knowledge(entity)
            knowledge_context[f"nodes_{entity}"] = kn_result.get("relevant_nodes", [])

        # 2. Define Initial Abstract Task
        # Start with a task that parses and routes based on the first parsed goal
        top_task = Task(
            name="PARSE_AND_ROUTE_GOAL", parameters={"parsed_goal": initial_parsed_goal}
        )
        # Alternatively, start with ACHIEVE_GOAL and let its method call PARSE_AND_ROUTE
        # top_task = Task(name="ACHIEVE_GOAL", parameters={"goal_input": goal_input})

        # 3. Start Decomposition
        success = await self._decompose(top_task, final_plan_steps, knowledge_context)

        if not success or not final_plan_steps:
            self.logger.error(
                f"Planner failed to decompose goal or generate steps for goal {goal_id}"
            )
            # Optionally, add a fallback generic step
            final_plan_steps.append(
                PlanStep(
                    description=f"Fallback: Attempt generic processing for goal: {goal_input}",
                    action_type="GENERIC_PROCESS",
                    parameters={"input": goal_input},
                    status=StatusEnum.PENDING,
                )
            )
            # return None # Indicate planning failure if no fallback desired

        # 4. Create Final Plan Object
        plan = Plan(goal_id=goal_id, steps=final_plan_steps, status=StatusEnum.READY)
        self.logger.info(
            f"Planner HTN: Generated plan {plan.id} with {len(final_plan_steps)} steps for goal {goal_id}."
        )
        return plan
