"""
Comprehensive tests for PredaCore orchestration layer — Phase 5.

Tests: agents/collaboration.py, agents/engine.py, agents/meta_cognition.py,
agents/self_improvement.py, agents/autonomy.py, agents/daf_bridge.py,
and gateway.py.

Target: 80+ tests covering all core orchestration infrastructure.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Imports under test — agents
# ---------------------------------------------------------------------------
from predacore.agents.collaboration import (
    AgentResult,
    AgentSpec,
    AgentTeam,
    CollaborationPattern,
    CollaborationResult,
    ResultAggregator,
    TaskDistributor,
    TaskPayload,
)
from predacore.agents.engine import (
    AGENT_TYPES,
    AgentInstance,
    AgentType,
    CapabilityRouter,
    DynamicAgentSpec,
    PerformanceTracker,
    TaskStore,
    compile_dynamic_agent_prompt,
    create_dynamic_agent_spec,
)
from predacore.agents.meta_cognition import (
    _ECHO_PATTERNS,
    _FABRICATION_INDICATORS,
    ResponseEvaluation,
    ToolCall,
    _stable_args_key,
    _text_similarity,
    detect_loop,
    evaluate_response,
    should_ask_for_help,
)
from predacore.agents.self_improvement import (
    DEFAULT_PROTECTED_FILES,
    AuditLog,
    ImprovementProposal,
    ProposalCategory,
    RiskLevel,
    SelfImprovementEngine,
)

# ---------------------------------------------------------------------------
# Imports under test — gateway
# ---------------------------------------------------------------------------
from predacore.gateway import (
    IDENTITY_CACHE_MAX_SIZE,
    IDENTITY_CACHE_TTL_SECONDS,
    RATE_LIMIT_CLEANUP_USER_THRESHOLD,
    RATE_LIMIT_WINDOW_SECONDS,
    SESSION_LOCKS_MAX,
    USER_RATE_LIMIT_PER_MINUTE,
    ChannelAdapter,
    Gateway,
    IncomingMessage,
    OutgoingMessage,
)

# ===========================================================================
# Section 1: CollaborationPattern enum
# ===========================================================================


class TestCollaborationPattern:
    """Tests for the CollaborationPattern enum."""

    def test_four_patterns_exist(self):
        assert len(CollaborationPattern) == 4

    def test_fan_out_value(self):
        assert CollaborationPattern.FAN_OUT.value == "fan_out"

    def test_pipeline_value(self):
        assert CollaborationPattern.PIPELINE.value == "pipeline"

    def test_consensus_value(self):
        assert CollaborationPattern.CONSENSUS.value == "consensus"

    def test_supervisor_value(self):
        assert CollaborationPattern.SUPERVISOR.value == "supervisor"

    def test_pattern_from_value(self):
        assert CollaborationPattern("fan_out") is CollaborationPattern.FAN_OUT
        assert CollaborationPattern("pipeline") is CollaborationPattern.PIPELINE


# ===========================================================================
# Section 2: Collaboration data models
# ===========================================================================


class TestAgentSpec:
    """Tests for the AgentSpec dataclass."""

    def test_auto_generated_id(self):
        spec = AgentSpec()
        assert spec.agent_id.startswith("agent-")
        assert len(spec.agent_id) > 6

    def test_explicit_id_preserved(self):
        spec = AgentSpec(agent_id="my-agent")
        assert spec.agent_id == "my-agent"

    def test_default_fields(self):
        spec = AgentSpec()
        assert spec.agent_type == ""
        assert spec.capabilities == []

    def test_unique_ids(self):
        ids = {AgentSpec().agent_id for _ in range(20)}
        assert len(ids) == 20


class TestTaskPayload:
    """Tests for the TaskPayload dataclass."""

    def test_auto_generated_task_id(self):
        t = TaskPayload()
        assert t.task_id.startswith("task-")

    def test_explicit_task_id(self):
        t = TaskPayload(task_id="custom-task")
        assert t.task_id == "custom-task"

    def test_default_timeout(self):
        t = TaskPayload()
        assert t.timeout == 30.0

    def test_default_empty_context(self):
        t = TaskPayload()
        assert t.context == {}

    def test_custom_fields(self):
        t = TaskPayload(prompt="hello", timeout=10.0, context={"key": "val"})
        assert t.prompt == "hello"
        assert t.timeout == 10.0
        assert t.context["key"] == "val"


class TestAgentResult:
    """Tests for the AgentResult dataclass."""

    def test_default_success(self):
        r = AgentResult()
        assert r.success is True
        assert r.error == ""
        assert r.output is None
        assert r.latency_ms == 0.0

    def test_failure_result(self):
        r = AgentResult(success=False, error="timeout")
        assert r.success is False
        assert r.error == "timeout"


class TestCollaborationResult:
    """Tests for the CollaborationResult dataclass."""

    def test_default_pattern(self):
        cr = CollaborationResult()
        assert cr.pattern is CollaborationPattern.FAN_OUT

    def test_success_count(self):
        results = [
            AgentResult(success=True),
            AgentResult(success=True),
            AgentResult(success=False, error="fail"),
        ]
        cr = CollaborationResult(results=results)
        assert cr.success_count == 2
        assert cr.error_count == 1

    def test_empty_results(self):
        cr = CollaborationResult()
        assert cr.success_count == 0
        assert cr.error_count == 0

    def test_consensus_reached_default(self):
        cr = CollaborationResult()
        assert cr.consensus_reached is True


# ===========================================================================
# Section 3: AgentType frozen dataclass
# ===========================================================================


class TestAgentType:
    """Tests for the AgentType frozen dataclass."""

    def test_frozen(self):
        at = AgentType(
            type_id="test",
            description="test agent",
            system_prompt="you are a test",
            allowed_tools=frozenset(["tool1"]),
            capabilities=("cap1",),
        )
        with pytest.raises(FrozenInstanceError):
            at.type_id = "modified"  # type: ignore[misc]

    def test_defaults(self):
        at = AgentType(
            type_id="test",
            description="d",
            system_prompt="s",
            allowed_tools=frozenset(),
            capabilities=(),
        )
        assert at.max_steps == 5
        assert at.temperature == 0.7

    def test_to_dict(self):
        at = AgentType(
            type_id="test",
            description="test agent",
            system_prompt="prompt",
            allowed_tools=frozenset(["b", "a"]),
            capabilities=("cap1", "cap2"),
            max_steps=10,
        )
        d = at.to_dict()
        assert d["type_id"] == "test"
        assert d["allowed_tools"] == ["a", "b"]  # sorted
        assert d["capabilities"] == ["cap1", "cap2"]
        assert d["max_steps"] == 10


class TestDynamicAgentSpec:
    """Tests for the DynamicAgentSpec frozen dataclass."""

    def test_frozen(self):
        spec = DynamicAgentSpec(
            spec_id="dyn-1",
            base_type="coder",
            collaboration_role="coder",
            specialization="python",
            mission="write code",
        )
        with pytest.raises(FrozenInstanceError):
            spec.base_type = "other"  # type: ignore[misc]

    def test_defaults(self):
        spec = DynamicAgentSpec(
            spec_id="test",
            base_type="coder",
            collaboration_role="coder",
            specialization="test",
            mission="test",
        )
        assert spec.success_criteria == ()
        assert spec.output_schema == {}
        assert spec.memory_scopes == ("global", "team")
        assert spec.allowed_tools == ()
        assert spec.max_steps is None
        assert spec.temperature is None

    def test_to_transport_dict(self):
        spec = DynamicAgentSpec(
            spec_id="s1",
            base_type="researcher",
            collaboration_role="lead",
            specialization="web",
            mission="find info",
            success_criteria=("accurate",),
        )
        d = spec.to_transport_dict()
        assert d["spec_id"] == "s1"
        assert d["base_type"] == "researcher"
        assert d["success_criteria"] == ["accurate"]
        assert d["memory_scopes"] == ["global", "team"]


# ===========================================================================
# Section 4: AGENT_TYPES registry
# ===========================================================================


class TestAgentTypesRegistry:
    """Tests for the AGENT_TYPES global registry."""

    EXPECTED_TYPES = [
        "researcher", "coder", "desktop_agent", "mobile_agent",
        "creative", "analyst", "planner", "devops", "communicator",
        "memory_agent", "generalist",
    ]

    def test_all_expected_types_registered(self):
        for type_id in self.EXPECTED_TYPES:
            assert type_id in AGENT_TYPES, f"Missing agent type: {type_id}"

    def test_at_least_11_types(self):
        assert len(AGENT_TYPES) >= 11

    def test_all_have_system_prompt(self):
        for type_id, at in AGENT_TYPES.items():
            assert at.system_prompt, f"{type_id} has empty system_prompt"

    def test_all_have_allowed_tools(self):
        for type_id, at in AGENT_TYPES.items():
            assert isinstance(at.allowed_tools, frozenset), f"{type_id} tools not frozenset"
            assert len(at.allowed_tools) >= 1, f"{type_id} has no allowed tools"

    def test_all_have_capabilities(self):
        for type_id, at in AGENT_TYPES.items():
            assert isinstance(at.capabilities, tuple), f"{type_id} capabilities not tuple"
            assert len(at.capabilities) >= 1, f"{type_id} has no capabilities"

    def test_researcher_has_web_tools(self):
        r = AGENT_TYPES["researcher"]
        assert "web_search" in r.allowed_tools
        assert "deep_search" in r.allowed_tools

    def test_coder_has_exec_tools(self):
        c = AGENT_TYPES["coder"]
        assert "python_exec" in c.allowed_tools
        assert "execute_code" in c.allowed_tools

    def test_coder_lower_temperature(self):
        c = AGENT_TYPES["coder"]
        assert c.temperature < 0.7

    def test_desktop_agent_has_vision(self):
        d = AGENT_TYPES["desktop_agent"]
        assert "screen_vision" in d.allowed_tools

    def test_generalist_broad_tool_access(self):
        g = AGENT_TYPES["generalist"]
        assert len(g.allowed_tools) >= 8

    def test_devops_has_git_tools(self):
        d = AGENT_TYPES["devops"]
        assert "git_context" in d.allowed_tools
        assert "run_command" in d.allowed_tools


# ===========================================================================
# Section 5: CapabilityRouter
# ===========================================================================


class TestCapabilityRouter:
    """Tests for the CapabilityRouter keyword routing."""

    def test_search_routes_to_researcher(self):
        assert CapabilityRouter.route("search for python docs") == "researcher"

    def test_code_routes_to_coder(self):
        assert CapabilityRouter.route("write code to sort a list") == "coder"

    def test_click_routes_to_desktop_agent(self):
        assert CapabilityRouter.route("click the save button") == "desktop_agent"

    def test_android_routes_to_mobile_agent(self):
        assert CapabilityRouter.route("open android settings") == "mobile_agent"

    def test_image_routes_to_creative(self):
        assert CapabilityRouter.route("generate image of a cat") == "creative"

    def test_analyze_routes_to_analyst(self):
        assert CapabilityRouter.route("analyze this data") == "analyst"

    def test_plan_routes_to_planner(self):
        assert CapabilityRouter.route("plan a project roadmap") == "planner"

    def test_git_routes_to_devops(self):
        assert CapabilityRouter.route("git commit changes") == "devops"

    def test_speak_routes_to_communicator(self):
        assert CapabilityRouter.route("speak this text aloud") == "communicator"

    def test_remember_routes_to_memory_agent(self):
        assert CapabilityRouter.route("remember this for later") == "memory_agent"

    def test_unknown_falls_back_to_generalist(self):
        assert CapabilityRouter.route("do something vague and indefinite") == "generalist"

    def test_multi_routes_top_n(self):
        results = CapabilityRouter.route_multi("search and code a solution", n=2)
        assert len(results) <= 2
        assert isinstance(results, list)

    def test_multi_fallback_generalist(self):
        results = CapabilityRouter.route_multi("something vague")
        assert results == ["generalist"]

    def test_case_insensitive(self):
        assert CapabilityRouter.route("SEARCH the web") == "researcher"

    def test_multi_keyword_highest_wins(self):
        # "search" and "research" both map to researcher; "code" maps to coder
        result = CapabilityRouter.route("search and research a topic then code")
        assert result == "researcher"  # 2 hits vs 1


# ===========================================================================
# Section 6: compile_dynamic_agent_prompt()
# ===========================================================================


class TestCompileDynamicAgentPrompt:
    """Tests for compile_dynamic_agent_prompt()."""

    def _make_spec(self, **overrides):
        defaults = dict(
            spec_id="test-spec",
            base_type="researcher",
            collaboration_role="lead_researcher",
            specialization="web research",
            mission="find accurate information",
            allowed_tools=("web_search", "deep_search"),
            memory_scopes=("global", "team"),
        )
        defaults.update(overrides)
        return DynamicAgentSpec(**defaults)

    def test_returns_tuple_of_two_strings(self):
        spec = self._make_spec()
        result = compile_dynamic_agent_prompt(spec)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)

    def test_system_prompt_includes_archetype(self):
        spec = self._make_spec()
        system, _ = compile_dynamic_agent_prompt(spec)
        archetype = AGENT_TYPES["researcher"]
        assert archetype.system_prompt in system

    def test_system_prompt_includes_role(self):
        spec = self._make_spec()
        system, _ = compile_dynamic_agent_prompt(spec)
        assert "lead_researcher" in system

    def test_system_prompt_includes_mission(self):
        spec = self._make_spec()
        system, _ = compile_dynamic_agent_prompt(spec)
        assert "find accurate information" in system

    def test_system_prompt_includes_tools(self):
        spec = self._make_spec()
        system, _ = compile_dynamic_agent_prompt(spec)
        assert "web_search" in system

    def test_user_prompt_includes_original_task(self):
        spec = self._make_spec()
        _, user = compile_dynamic_agent_prompt(spec, original_task="Research AI trends")
        assert "Research AI trends" in user

    def test_user_prompt_includes_shared_context(self):
        spec = self._make_spec()
        _, user = compile_dynamic_agent_prompt(spec, shared_context="Agent 1 found X")
        assert "Agent 1 found X" in user

    def test_success_criteria_in_user_prompt(self):
        spec = self._make_spec(success_criteria=("be thorough", "cite sources"))
        _, user = compile_dynamic_agent_prompt(spec)
        assert "be thorough" in user
        assert "cite sources" in user

    def test_output_schema_in_user_prompt(self):
        spec = self._make_spec(output_schema={"findings": "list of results"})
        _, user = compile_dynamic_agent_prompt(spec)
        assert "findings" in user

    def test_unknown_base_type_falls_back_to_generalist(self):
        spec = self._make_spec(base_type="nonexistent_type")
        system, _ = compile_dynamic_agent_prompt(spec)
        generalist = AGENT_TYPES["generalist"]
        assert generalist.system_prompt in system


# ===========================================================================
# Section 7: evaluate_response() — meta-cognition
# ===========================================================================


class TestEvaluateResponse:
    """Tests for the evaluate_response() function."""

    def test_empty_response(self):
        ev = evaluate_response("What is X?", "", [], [])
        assert ev.confidence == 0.0
        assert "empty_response" in ev.issues

    def test_none_response_treated_as_empty(self):
        ev = evaluate_response("What is X?", None, [], [])  # type: ignore[arg-type]
        assert ev.confidence == 0.0

    def test_good_response_high_confidence(self):
        ev = evaluate_response("Tell me about cats", "Cats are domestic felines.", [], [])
        assert ev.confidence >= 0.5
        assert ev.should_send is True

    def test_fabrication_risk_no_tools(self):
        ev = evaluate_response(
            "What is the current price?",
            "The current price is $50.32 as of today.",
            [],
            [],
        )
        assert ev.has_fabrication_risk is True
        assert "fabrication_risk_no_tools" in ev.issues
        assert ev.grounded is False

    def test_no_fabrication_with_tools(self):
        ev = evaluate_response(
            "What is the current price?",
            "The current price is $50.32 as of today.",
            ["web_search"],
            ["Price data: $50.32"],
        )
        # With tools used, fabrication indicators are not checked
        assert ev.has_fabrication_risk is False

    def test_echo_detection(self):
        user_msg = "Can you help me understand the quantum physics of black holes?"
        # Echo requires: 1) _ECHO_PATTERN match at start  2) text_similarity > 0.5
        # Pattern: "you want|you asked|you're asking" at start
        response = "You asked about the quantum physics of black holes and related topics"
        ev = evaluate_response(user_msg, response, [], [])
        assert ev.is_echo is True
        assert "echo_detected" in ev.issues

    def test_echo_not_triggered_for_short_user_msg(self):
        ev = evaluate_response("Hi", "Sure, you'd like to say hi", [], [])
        # User message < 20 chars, so echo check is skipped
        assert ev.is_echo is False

    def test_grounding_check_with_tool_results(self):
        # Response with terms completely unrelated to tool results
        ev = evaluate_response(
            "What files are in the directory?",
            "The quantum entanglement manifold demonstrates interesting topological properties "
            "across multiple dimensional boundaries in theoretical physics frameworks. "
            "Additional research indicates correlations between subatomic particle behavior.",
            ["list_directory"],
            ["file1.txt file2.py readme.md config.yaml setup.sh"],
        )
        assert "low_grounding_overlap" in ev.issues

    def test_brief_response_to_complex_question(self):
        long_question = "Please analyze the complete architecture of the system including " * 5
        ev = evaluate_response(long_question, "OK done.", [], [])
        assert "response_too_brief" in ev.issues

    def test_confidence_clamped_to_0_1(self):
        ev = evaluate_response("test", "test", [], [])
        assert 0.0 <= ev.confidence <= 1.0

    def test_should_send_threshold(self):
        ev = ResponseEvaluation(confidence=0.3)
        assert ev.should_send is True
        ev2 = ResponseEvaluation(confidence=0.29)
        assert ev2.should_send is False

    def test_echo_blocks_should_send(self):
        ev = ResponseEvaluation(confidence=1.0, is_echo=True)
        assert ev.should_send is False


# ===========================================================================
# Section 8: detect_loop() — meta-cognition
# ===========================================================================


class TestDetectLoop:
    """Tests for the detect_loop() function."""

    def test_short_history_no_loop(self):
        history = [("web_search", {"query": "hello"})]
        assert detect_loop(history) is False

    def test_repeated_same_call(self):
        history = [
            ("web_search", {"query": "test"}),
            ("web_search", {"query": "test"}),
            ("web_search", {"query": "test"}),
        ]
        assert detect_loop(history, max_repeats=3) is True

    def test_oscillation_pattern(self):
        history = [
            ("web_search", {"query": "a"}),
            ("read_file", {"path": "b"}),
            ("web_search", {"query": "a"}),
            ("read_file", {"path": "b"}),
        ]
        assert detect_loop(history) is True

    def test_consecutive_identical(self):
        history = [
            ("memory_store", {"key": "x", "value": "y"}),
            ("memory_store", {"key": "x", "value": "y"}),
            ("memory_store", {"key": "x", "value": "y"}),
        ]
        assert detect_loop(history, max_repeats=3) is True

    def test_different_calls_no_loop(self):
        history = [
            ("web_search", {"query": "a"}),
            ("read_file", {"path": "b"}),
            ("write_file", {"path": "c", "content": "d"}),
        ]
        assert detect_loop(history) is False

    def test_thrashing_same_tool_many_times(self):
        """Same tool called 8+ times with >75% of recent calls."""
        history = [("read_file", {"path": "/p"})] * 9
        assert detect_loop(history) is True

    def test_stall_detection_with_max_iterations(self):
        # max_iterations=20, threshold = max(15, int(20*0.8)) = 16
        history = [("tool_a", {"arg": str(i)}) for i in range(16)]
        assert detect_loop(history, max_iterations=20) is True

    def test_stall_detection_default_threshold(self):
        # Without max_iterations, threshold = 15
        history = [("tool_a", {"arg": str(i)}) for i in range(15)]
        assert detect_loop(history) is True

    def test_below_stall_threshold(self):
        # Need diverse enough calls to not trigger thrashing detection
        history = [("tool_a", {"arg": str(i)}) for i in range(5)]
        assert detect_loop(history) is False

    def test_action_tools_use_action_in_fingerprint(self):
        """browser_control with different actions should NOT be flagged as same call."""
        history = [
            ("browser_control", {"action": "navigate", "url": "https://a.com"}),
            ("browser_control", {"action": "get_page_tree"}),
            ("browser_control", {"action": "click", "selector": "#btn"}),
        ]
        assert detect_loop(history) is False

    def test_action_tools_same_action_detected(self):
        """Same action repeated should be caught."""
        history = [
            ("browser_control", {"action": "click", "selector": "#btn"}),
            ("browser_control", {"action": "click", "selector": "#btn"}),
            ("browser_control", {"action": "click", "selector": "#btn"}),
        ]
        assert detect_loop(history, max_repeats=3) is True

    def test_custom_max_repeats(self):
        history = [("web_search", {"query": "same"})] * 5
        assert detect_loop(history, max_repeats=6) is False
        assert detect_loop(history, max_repeats=5) is True

    def test_action_tools_exploration_not_thrashing(self):
        """Action-based tools with 3+ unique actions are exploring, not thrashing."""
        history = [
            ("browser_control", {"action": "navigate", "url": "https://a.com"}),
            ("browser_control", {"action": "get_page_tree"}),
            ("browser_control", {"action": "click", "selector": "#x"}),
            ("browser_control", {"action": "read_text"}),
            ("browser_control", {"action": "navigate", "url": "https://b.com"}),
            ("browser_control", {"action": "get_page_tree"}),
            ("browser_control", {"action": "click", "selector": "#y"}),
            ("browser_control", {"action": "read_text"}),
        ]
        # 8 calls but 4 unique actions -> exploration, not thrashing
        assert detect_loop(history, max_repeats=10) is False


# ===========================================================================
# Section 8b: should_ask_for_help()
# ===========================================================================


class TestShouldAskForHelp:
    """Tests for should_ask_for_help()."""

    def test_multiple_errors_triggers_help(self):
        result = should_ask_for_help(
            tool_errors=["err1", "err2", "err3"],
            iteration_count=1,
            max_iterations=10,
            tool_history=[],
        )
        assert result is not None
        assert "tool errors" in result.lower()

    def test_no_help_needed(self):
        result = should_ask_for_help(
            tool_errors=[],
            iteration_count=1,
            max_iterations=10,
            tool_history=[("web_search", {"q": "hi"})],
        )
        assert result is None

    def test_running_out_of_iterations(self):
        result = should_ask_for_help(
            tool_errors=[],
            iteration_count=8,
            max_iterations=10,
            tool_history=[("t", {"a": str(i)}) for i in range(3)],
        )
        assert result is not None
        assert "iterations" in result.lower()


# ===========================================================================
# Section 8c: Meta-cognition helpers
# ===========================================================================


class TestMetaCognitionHelpers:
    """Tests for _text_similarity and _stable_args_key."""

    def test_text_similarity_identical(self):
        assert _text_similarity("hello world", "hello world") == 1.0

    def test_text_similarity_empty(self):
        assert _text_similarity("", "hello") == 0.0
        assert _text_similarity("hello", "") == 0.0

    def test_text_similarity_partial(self):
        sim = _text_similarity("hello world", "hello earth")
        assert 0.0 < sim < 1.0

    def test_stable_args_key_empty(self):
        assert _stable_args_key({}) == ""

    def test_stable_args_key_sorted(self):
        k1 = _stable_args_key({"b": 2, "a": 1})
        k2 = _stable_args_key({"a": 1, "b": 2})
        assert k1 == k2

    def test_stable_args_key_truncated(self):
        big = {"key": "x" * 500}
        key = _stable_args_key(big)
        assert len(key) <= 200


# ===========================================================================
# Section 9: SelfImprovementEngine
# ===========================================================================


class TestSelfImprovementEngine:
    """Tests for the SelfImprovementEngine."""

    def _make_engine(self, tmp_path, **kwargs):
        outcome_store = AsyncMock()
        outcome_store.get_failure_patterns = AsyncMock(return_value=[])
        outcome_store.get_provider_stats = AsyncMock(return_value=[])
        outcome_store.get_feedback_summary = AsyncMock(return_value={"good": 0, "bad": 0})
        outcome_store.get_recent = AsyncMock(return_value=[])
        defaults = dict(
            home_dir=tmp_path,
            outcome_store=outcome_store,
            max_daily_modifications=3,
            enable_auto_apply=True,
        )
        defaults.update(kwargs)
        return SelfImprovementEngine(**defaults), outcome_store

    def test_protected_files_default(self):
        assert "config.py" in DEFAULT_PROTECTED_FILES
        assert "SOUL_SEED.md" in DEFAULT_PROTECTED_FILES
        assert "security.py" in DEFAULT_PROTECTED_FILES
        assert "__init__.py" in DEFAULT_PROTECTED_FILES

    @pytest.mark.asyncio
    async def test_analyze_failures_no_patterns(self, tmp_path):
        engine, _ = self._make_engine(tmp_path)
        proposals = await engine.analyze_failures()
        assert proposals == []

    @pytest.mark.asyncio
    async def test_analyze_failures_tool_pattern(self, tmp_path):
        engine, store = self._make_engine(tmp_path)
        store.get_failure_patterns.return_value = [
            {"tool_name": "web_search", "failure_count": 5, "error_samples": ["timeout"]}
        ]
        proposals = await engine.analyze_failures()
        assert len(proposals) >= 1
        assert proposals[0].category == ProposalCategory.TOOL_CONFIG

    @pytest.mark.asyncio
    async def test_apply_blocked_by_protected_file(self, tmp_path):
        engine, _ = self._make_engine(tmp_path)
        proposal = ImprovementProposal(
            target_file="config.py",
            risk_level=RiskLevel.LOW,
            auto_applicable=True,
        )
        result = await engine.apply_proposal(proposal)
        assert result is False

    @pytest.mark.asyncio
    async def test_apply_blocked_by_risk_level(self, tmp_path):
        engine, _ = self._make_engine(tmp_path)
        proposal = ImprovementProposal(
            target_file="some_file.py",
            risk_level=RiskLevel.HIGH,
            auto_applicable=True,
        )
        result = await engine.apply_proposal(proposal)
        assert result is False

    @pytest.mark.asyncio
    async def test_apply_blocked_by_daily_limit(self, tmp_path):
        engine, _ = self._make_engine(tmp_path, max_daily_modifications=0)
        proposal = ImprovementProposal(
            target_file="safe_file.py",
            risk_level=RiskLevel.LOW,
            auto_applicable=True,
        )
        result = await engine.apply_proposal(proposal)
        assert result is False

    @pytest.mark.asyncio
    async def test_apply_blocked_when_auto_apply_disabled(self, tmp_path):
        engine, _ = self._make_engine(tmp_path, enable_auto_apply=False)
        proposal = ImprovementProposal(
            target_file="safe_file.py",
            risk_level=RiskLevel.LOW,
        )
        result = await engine.apply_proposal(proposal)
        assert result is False

    def test_get_summary(self, tmp_path):
        engine, _ = self._make_engine(tmp_path)
        summary = engine.get_summary()
        assert "proposals_pending" in summary
        assert "max_daily_modifications" in summary
        assert summary["auto_apply_enabled"] is True


# ===========================================================================
# Section 9b: AuditLog
# ===========================================================================


class TestAuditLog:
    """Tests for the AuditLog class."""

    def test_log_creates_file(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.log("test_action", {"detail": "value"})
        assert (tmp_path / "audit.jsonl").exists()

    def test_get_recent(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.log("action1", {"k": "v1"})
        log.log("action2", {"k": "v2"})
        entries = log.get_recent(limit=10)
        assert len(entries) == 2
        assert entries[0]["action"] == "action1"
        assert entries[1]["action"] == "action2"

    def test_get_recent_empty(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        entries = log.get_recent()
        assert entries == []

    def test_count_today(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.log("apply_proposal", {})
        log.log("analyze_failures", {})
        # Only "apply_proposal" and "apply_proposed" count
        assert log.count_today() == 1

    @pytest.mark.asyncio
    async def test_alog(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        await log.alog("async_action", {"key": "val"})
        entries = log.get_recent()
        assert len(entries) == 1
        assert entries[0]["action"] == "async_action"


# ===========================================================================
# Section 9c: ImprovementProposal
# ===========================================================================


class TestImprovementProposal:
    """Tests for the ImprovementProposal dataclass."""

    def test_auto_generated_id(self):
        p = ImprovementProposal()
        assert len(p.proposal_id) == 10

    def test_auto_created_at(self):
        p = ImprovementProposal()
        assert p.created_at > 0

    def test_to_dict(self):
        p = ImprovementProposal(
            category=ProposalCategory.CODE_FIX,
            risk_level=RiskLevel.HIGH,
            evidence=["sample error"],
        )
        d = p.to_dict()
        assert d["category"] == "code_fix"
        assert d["risk_level"] == "high"
        assert d["evidence"] == ["sample error"]

    def test_proposal_categories(self):
        assert len(ProposalCategory) == 5
        assert ProposalCategory.PROMPT_TUNING.value == "prompt_tuning"
        assert ProposalCategory.MEMORY_CLEANUP.value == "memory_cleanup"

    def test_risk_levels(self):
        assert len(RiskLevel) == 3
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"


# ===========================================================================
# Section 10: IncomingMessage and OutgoingMessage
# ===========================================================================


class TestIncomingMessage:
    """Tests for the IncomingMessage dataclass."""

    def test_auto_generated_message_id(self):
        msg = IncomingMessage(channel="cli", user_id="u1", text="hello")
        assert msg.message_id  # non-empty
        assert len(msg.message_id) == 36  # UUID format

    def test_unique_message_ids(self):
        ids = {
            IncomingMessage(channel="cli", user_id="u1", text="hi").message_id
            for _ in range(10)
        }
        assert len(ids) == 10

    def test_default_fields(self):
        msg = IncomingMessage(channel="telegram", user_id="u1", text="test")
        assert msg.session_id is None
        assert msg.attachments == []
        assert msg.metadata == {}
        assert msg.timestamp > 0

    def test_explicit_fields(self):
        msg = IncomingMessage(
            channel="discord",
            user_id="u1",
            text="hello",
            session_id="s1",
            attachments=[{"type": "image"}],
            metadata={"key": "val"},
        )
        assert msg.session_id == "s1"
        assert msg.attachments == [{"type": "image"}]
        assert msg.metadata["key"] == "val"


class TestOutgoingMessage:
    """Tests for the OutgoingMessage dataclass."""

    def test_required_fields(self):
        msg = OutgoingMessage(channel="cli", user_id="u1", text="hi", session_id="s1")
        assert msg.channel == "cli"
        assert msg.text == "hi"

    def test_default_fields(self):
        msg = OutgoingMessage(channel="cli", user_id="u1", text="hi", session_id="s1")
        assert msg.metadata == {}
        assert msg.tool_calls == []
        assert msg.thinking is None
        assert msg.timestamp > 0

    def test_tool_calls_list(self):
        msg = OutgoingMessage(
            channel="cli", user_id="u1", text="done",
            session_id="s1",
            tool_calls=[{"name": "web_search", "args": {}}],
        )
        assert len(msg.tool_calls) == 1


# ===========================================================================
# Section 11: ChannelAdapter ABC
# ===========================================================================


class TestChannelAdapter:
    """Tests for the ChannelAdapter abstract base class."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ChannelAdapter()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        class MockAdapter(ChannelAdapter):
            channel_name = "mock"

            async def start(self):
                pass

            async def stop(self):
                pass

            async def send(self, message):
                pass

        adapter = MockAdapter()
        assert adapter.channel_name == "mock"

    def test_set_message_handler(self):
        class MockAdapter(ChannelAdapter):
            channel_name = "mock"

            async def start(self):
                pass

            async def stop(self):
                pass

            async def send(self, message):
                pass

        adapter = MockAdapter()
        handler = AsyncMock()
        adapter.set_message_handler(handler)
        assert adapter._message_handler is handler

    def test_missing_method_raises(self):
        """A subclass missing abstract methods cannot be instantiated."""
        class IncompleteAdapter(ChannelAdapter):
            channel_name = "broken"

            async def start(self):
                pass
            # Missing stop() and send()

        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore[abstract]


# ===========================================================================
# Section 12: Gateway rate limiting
# ===========================================================================


class TestGatewayRateLimiting:
    """Tests for Gateway._check_user_rate_limit logic."""

    def _make_gateway(self):
        """Create a minimal Gateway with mocked dependencies."""
        config = MagicMock()
        config.sessions_dir = "/tmp/test_sessions"
        config.home_dir = "/tmp/test_home"
        config.security.max_concurrent_tasks = 5
        config.security.task_timeout_seconds = 60
        config.security.require_auth = False

        process_fn = AsyncMock(return_value="response")

        with patch("predacore.gateway.SessionStore"), \
             patch("predacore.gateway.LaneQueue"), \
             patch("predacore.gateway.ChannelHealthMonitor"), \
             patch("predacore.gateway.IdentityService"), \
             patch("predacore.gateway.LRUCache"), \
             patch("predacore.gateway.RateLimiter"), \
             patch("predacore.gateway.default_api_limits", return_value=[]), \
             patch("predacore.gateway.AuthMiddleware", return_value=None), \
             patch.dict(os.environ, {"PREDACORE_JWT_SECRET": ""}, clear=False):
            gw = Gateway(config, process_fn)

        return gw

    @pytest.mark.asyncio
    async def test_first_request_allowed(self):
        gw = self._make_gateway()
        assert await gw._check_user_rate_limit("user1") is True

    @pytest.mark.asyncio
    async def test_under_limit_allowed(self):
        gw = self._make_gateway()
        for _ in range(USER_RATE_LIMIT_PER_MINUTE - 1):
            assert await gw._check_user_rate_limit("user1") is True

    @pytest.mark.asyncio
    async def test_at_limit_blocked(self):
        gw = self._make_gateway()
        for _ in range(USER_RATE_LIMIT_PER_MINUTE):
            await gw._check_user_rate_limit("user1")
        assert await gw._check_user_rate_limit("user1") is False

    @pytest.mark.asyncio
    async def test_different_users_independent(self):
        gw = self._make_gateway()
        for _ in range(USER_RATE_LIMIT_PER_MINUTE):
            await gw._check_user_rate_limit("user1")
        # user1 is blocked
        assert await gw._check_user_rate_limit("user1") is False
        # user2 is still allowed
        assert await gw._check_user_rate_limit("user2") is True

    @pytest.mark.asyncio
    async def test_expired_timestamps_pruned(self):
        gw = self._make_gateway()
        # Manually inject old timestamps
        old_time = time.time() - RATE_LIMIT_WINDOW_SECONDS - 1
        gw._user_request_times["user1"] = [old_time] * USER_RATE_LIMIT_PER_MINUTE
        # Old timestamps should be pruned, so new request is allowed
        assert await gw._check_user_rate_limit("user1") is True

    @pytest.mark.asyncio
    async def test_stale_user_cleanup(self):
        gw = self._make_gateway()
        # Inject many stale users
        old_time = time.time() - RATE_LIMIT_WINDOW_SECONDS - 10
        for i in range(RATE_LIMIT_CLEANUP_USER_THRESHOLD + 10):
            gw._user_request_times[f"stale_{i}"] = [old_time]
        # Trigger cleanup by making a new request
        await gw._check_user_rate_limit("active_user")
        # Stale users should have been cleaned up
        assert len(gw._user_request_times) < RATE_LIMIT_CLEANUP_USER_THRESHOLD + 10


# ===========================================================================
# Section 12b: Gateway constants
# ===========================================================================


class TestGatewayConstants:
    """Tests for gateway module-level constants."""

    def test_rate_limit_per_minute(self):
        assert USER_RATE_LIMIT_PER_MINUTE == 30

    def test_rate_limit_window(self):
        assert RATE_LIMIT_WINDOW_SECONDS == 60.0

    def test_session_locks_max(self):
        assert SESSION_LOCKS_MAX == 10_000

    def test_identity_cache_max(self):
        assert IDENTITY_CACHE_MAX_SIZE == 500

    def test_identity_cache_ttl(self):
        assert IDENTITY_CACHE_TTL_SECONDS == 600


# ===========================================================================
# Section 13: AgentTeam orchestrator
# ===========================================================================


class TestAgentTeam:
    """Tests for the AgentTeam multi-agent orchestrator."""

    @pytest.mark.asyncio
    async def test_fan_out(self):
        team = AgentTeam()
        team.add_agent(
            AgentSpec(agent_id="a1"),
            AsyncMock(return_value="result_a"),
        )
        team.add_agent(
            AgentSpec(agent_id="a2"),
            AsyncMock(return_value="result_b"),
        )
        cr = await team.fan_out(TaskPayload(prompt="test"))
        assert cr.pattern is CollaborationPattern.FAN_OUT
        assert cr.success_count == 2
        assert len(cr.final_output) == 2
        assert cr.total_latency_ms > 0

    @pytest.mark.asyncio
    async def test_pipeline(self):
        team = AgentTeam()
        team.add_agent(AgentSpec(agent_id="a1"), AsyncMock(return_value="step1_output"))
        team.add_agent(AgentSpec(agent_id="a2"), AsyncMock(return_value="step2_output"))
        cr = await team.pipeline(TaskPayload(prompt="start"), order=["a1", "a2"])
        assert cr.pattern is CollaborationPattern.PIPELINE
        assert cr.final_output == "step2_output"
        assert len(cr.results) == 2

    @pytest.mark.asyncio
    async def test_pipeline_stops_on_failure(self):
        async def fail_fn(task):
            raise RuntimeError("broken")

        team = AgentTeam()
        team.add_agent(AgentSpec(agent_id="a1"), fail_fn)
        team.add_agent(AgentSpec(agent_id="a2"), AsyncMock(return_value="should not run"))
        cr = await team.pipeline(TaskPayload(prompt="start"), order=["a1", "a2"])
        assert len(cr.results) == 1
        assert cr.results[0].success is False
        assert cr.final_output is None

    @pytest.mark.asyncio
    async def test_consensus(self):
        team = AgentTeam()
        team.add_agent(AgentSpec(agent_id="a1"), AsyncMock(return_value="yes"))
        team.add_agent(AgentSpec(agent_id="a2"), AsyncMock(return_value="yes"))
        team.add_agent(AgentSpec(agent_id="a3"), AsyncMock(return_value="no"))
        cr = await team.consensus(TaskPayload(prompt="vote"), quorum=0.5)
        assert cr.pattern is CollaborationPattern.CONSENSUS
        assert cr.final_output == "yes"
        assert cr.consensus_reached is True

    @pytest.mark.asyncio
    async def test_consensus_no_quorum(self):
        team = AgentTeam()
        team.add_agent(AgentSpec(agent_id="a1"), AsyncMock(return_value="a"))
        team.add_agent(AgentSpec(agent_id="a2"), AsyncMock(return_value="b"))
        team.add_agent(AgentSpec(agent_id="a3"), AsyncMock(return_value="c"))
        cr = await team.consensus(TaskPayload(prompt="vote"), quorum=0.5)
        # Each gets 1/3 votes, none reaches 0.5 quorum
        assert cr.consensus_reached is False

    @pytest.mark.asyncio
    async def test_supervise(self):
        team = AgentTeam()
        team.add_agent(AgentSpec(agent_id="worker"), AsyncMock(return_value="draft"))
        team.add_agent(AgentSpec(agent_id="supervisor"), AsyncMock(return_value="final"))
        cr = await team.supervise(
            TaskPayload(prompt="task"),
            worker_id="worker",
            supervisor_id="supervisor",
        )
        assert cr.pattern is CollaborationPattern.SUPERVISOR
        assert cr.final_output == "final"
        assert len(cr.results) == 2

    @pytest.mark.asyncio
    async def test_supervise_missing_worker(self):
        team = AgentTeam()
        team.add_agent(AgentSpec(agent_id="supervisor"), AsyncMock(return_value="review"))
        cr = await team.supervise(
            TaskPayload(prompt="task"),
            worker_id="missing_worker",
            supervisor_id="supervisor",
        )
        assert cr.results[0].success is False
        assert "Not found" in cr.results[0].error

    @pytest.mark.asyncio
    async def test_agent_timeout(self):
        async def slow_fn(task):
            await asyncio.sleep(10)

        team = AgentTeam()
        team.add_agent(AgentSpec(agent_id="slow"), slow_fn)
        cr = await team.fan_out(TaskPayload(prompt="test", timeout=0.01))
        assert cr.results[0].success is False
        assert "Timeout" in cr.results[0].error

    def test_agent_count(self):
        team = AgentTeam()
        assert team.agent_count == 0
        team.add_agent(AgentSpec(agent_id="a1"), AsyncMock())
        assert team.agent_count == 1


# ===========================================================================
# Section 14: ResultAggregator and TaskDistributor
# ===========================================================================


class TestResultAggregator:
    """Tests for ResultAggregator static methods."""

    def test_union(self):
        results = [
            AgentResult(output="a", success=True),
            AgentResult(output="b", success=True),
            AgentResult(output="c", success=False),
        ]
        assert ResultAggregator.union(results) == ["a", "b"]

    def test_majority_vote(self):
        results = [
            AgentResult(output="yes", success=True),
            AgentResult(output="yes", success=True),
            AgentResult(output="no", success=True),
        ]
        assert ResultAggregator.majority_vote(results) == "yes"

    def test_majority_vote_all_failures(self):
        results = [AgentResult(success=False)] * 3
        assert ResultAggregator.majority_vote(results) is None

    def test_weighted_vote(self):
        results = [
            AgentResult(agent_id="expert", output="yes", success=True),
            AgentResult(agent_id="novice", output="no", success=True),
        ]
        weights = {"expert": 10.0, "novice": 1.0}
        assert ResultAggregator.weighted_vote(results, weights) == "yes"


class TestTaskDistributor:
    """Tests for TaskDistributor static methods."""

    def test_replicate(self):
        task = TaskPayload(prompt="do X")
        agents = [AgentSpec(agent_id="a1"), AgentSpec(agent_id="a2")]
        assignments = TaskDistributor.replicate(task, agents)
        assert len(assignments) == 2
        assert assignments[0][1] is task
        assert assignments[1][1] is task

    def test_split_by_sections(self):
        prompt = "line1\nline2\nline3\nline4"
        agents = [AgentSpec(agent_id="a1"), AgentSpec(agent_id="a2")]
        assignments = TaskDistributor.split_by_sections(prompt, agents)
        assert len(assignments) == 2
        # Each agent gets a non-empty portion
        for agent, task in assignments:
            assert task.prompt  # non-empty


# ===========================================================================
# Section 15: create_dynamic_agent_spec()
# ===========================================================================


class TestCreateDynamicAgentSpec:
    """Tests for create_dynamic_agent_spec factory."""

    def test_defaults(self):
        spec = create_dynamic_agent_spec(base_type="researcher")
        assert spec.base_type == "researcher"
        assert spec.spec_id.startswith("dyn-")
        assert spec.memory_scopes == ("global", "team")

    def test_unknown_base_type_falls_back(self):
        spec = create_dynamic_agent_spec(base_type="nonexistent")
        assert spec.base_type == "generalist"

    def test_allowed_tools_scoped_to_archetype(self):
        spec = create_dynamic_agent_spec(
            base_type="researcher",
            allowed_tools=["web_search", "python_exec"],  # python_exec not in researcher
        )
        # Only tools that are in researcher's allowed_tools should be kept
        assert "web_search" in spec.allowed_tools
        assert "python_exec" not in spec.allowed_tools

    def test_no_matching_tools_falls_back_to_all(self):
        spec = create_dynamic_agent_spec(
            base_type="researcher",
            allowed_tools=["nonexistent_tool"],
        )
        # Falls back to all archetype tools
        assert len(spec.allowed_tools) == len(AGENT_TYPES["researcher"].allowed_tools)

    def test_success_criteria_limited_to_5(self):
        spec = create_dynamic_agent_spec(
            base_type="coder",
            success_criteria=["a", "b", "c", "d", "e", "f", "g"],
        )
        assert len(spec.success_criteria) == 5

    def test_max_steps_minimum_1(self):
        spec = create_dynamic_agent_spec(base_type="coder", max_steps=0)
        assert spec.max_steps >= 1

    def test_temperature_from_archetype(self):
        spec = create_dynamic_agent_spec(base_type="coder")
        assert spec.temperature == AGENT_TYPES["coder"].temperature


# ===========================================================================
# Section 16: Engine support classes
# ===========================================================================


class TestAgentInstance:
    """Tests for the AgentInstance dataclass."""

    def test_defaults(self):
        inst = AgentInstance()
        assert inst.status == "idle"
        assert inst.task_count == 0
        assert inst.created_at > 0

    def test_avg_latency(self):
        inst = AgentInstance(task_count=5, total_latency_ms=500.0)
        assert inst.avg_latency_ms == 100.0

    def test_avg_latency_zero_tasks(self):
        inst = AgentInstance(task_count=0, total_latency_ms=0)
        assert inst.avg_latency_ms == 0.0  # max(task_count, 1) prevents division by zero

    def test_success_rate(self):
        inst = AgentInstance(success_count=8, error_count=2)
        assert inst.success_rate == 0.8

    def test_success_rate_zero_total(self):
        inst = AgentInstance(success_count=0, error_count=0)
        assert inst.success_rate == 0.0


class TestTaskStore:
    """Tests for the TaskStore in-memory store."""

    @pytest.mark.asyncio
    async def test_create_and_get(self):
        store = TaskStore()
        record = await store.create("t1", "a1", "researcher", "do stuff")
        assert record.task_id == "t1"
        assert record.status == "running"

        got = await store.get("t1")
        assert got is not None
        assert got.task_id == "t1"

    @pytest.mark.asyncio
    async def test_complete(self):
        store = TaskStore()
        await store.create("t1", "a1", "coder", "write code")
        await store.complete("t1", "output text")
        record = await store.get("t1")
        assert record.status == "completed"
        assert record.output == "output text"
        assert record.latency_ms > 0

    @pytest.mark.asyncio
    async def test_fail(self):
        store = TaskStore()
        await store.create("t1", "a1", "coder", "write code")
        await store.fail("t1", "syntax error")
        record = await store.get("t1")
        assert record.status == "failed"
        assert record.error == "syntax error"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        store = TaskStore()
        assert await store.get("nope") is None

    @pytest.mark.asyncio
    async def test_recent(self):
        store = TaskStore()
        for i in range(5):
            await store.create(f"t{i}", "a1", "researcher", f"task {i}")
        recent = await store.recent(limit=3)
        assert len(recent) == 3


class TestPerformanceTracker:
    """Tests for the PerformanceTracker."""

    def test_record_and_stats(self):
        tracker = PerformanceTracker()
        tracker.record("researcher", 100.0, True)
        tracker.record("researcher", 200.0, True)
        tracker.record("researcher", 300.0, False)
        stats = tracker.stats("researcher")
        assert stats["tasks"] == 3
        assert stats["avg_latency_ms"] == 200.0
        assert stats["success_rate"] == pytest.approx(2 / 3)

    def test_empty_stats(self):
        tracker = PerformanceTracker()
        stats = tracker.stats("unknown")
        assert stats["tasks"] == 0
        assert stats["success_rate"] == 1.0

    def test_all_stats(self):
        tracker = PerformanceTracker()
        tracker.record("a", 10, True)
        tracker.record("b", 20, False)
        all_s = tracker.all_stats()
        assert "a" in all_s
        assert "b" in all_s

    def test_best_type_for(self):
        tracker = PerformanceTracker()
        # researcher: fast, high success
        for _ in range(5):
            tracker.record("researcher", 50.0, True)
        # coder: slow, low success
        for _ in range(5):
            tracker.record("coder", 500.0, False)
        best = tracker.best_type_for(["researcher", "coder"])
        assert best == "researcher"

    def test_best_type_empty(self):
        tracker = PerformanceTracker()
        assert tracker.best_type_for([]) == "generalist"


# ===========================================================================
# Section 17: Gateway session lock LRU
# ===========================================================================


class TestGatewaySessionLock:
    """Tests for Gateway._get_session_lock LRU behavior."""

    def _make_gateway(self):
        config = MagicMock()
        config.sessions_dir = "/tmp/test_sessions"
        config.home_dir = "/tmp/test_home"
        config.security.max_concurrent_tasks = 5
        config.security.task_timeout_seconds = 60
        config.security.require_auth = False

        process_fn = AsyncMock(return_value="response")

        with patch("predacore.gateway.SessionStore"), \
             patch("predacore.gateway.LaneQueue"), \
             patch("predacore.gateway.ChannelHealthMonitor"), \
             patch("predacore.gateway.IdentityService"), \
             patch("predacore.gateway.LRUCache"), \
             patch("predacore.gateway.RateLimiter"), \
             patch("predacore.gateway.default_api_limits", return_value=[]), \
             patch("predacore.gateway.AuthMiddleware", return_value=None), \
             patch.dict(os.environ, {"PREDACORE_JWT_SECRET": ""}, clear=False):
            gw = Gateway(config, process_fn)

        return gw

    @pytest.mark.asyncio
    async def test_same_session_returns_same_lock(self):
        gw = self._make_gateway()
        lock1 = await gw._get_session_lock("session_abc")
        lock2 = await gw._get_session_lock("session_abc")
        assert lock1 is lock2

    @pytest.mark.asyncio
    async def test_different_sessions_different_locks(self):
        gw = self._make_gateway()
        lock1 = await gw._get_session_lock("s1")
        lock2 = await gw._get_session_lock("s2")
        assert lock1 is not lock2

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        gw = self._make_gateway()
        gw._session_locks_max = 3
        await gw._get_session_lock("s1")
        await gw._get_session_lock("s2")
        await gw._get_session_lock("s3")
        # Adding s4 should evict s1 (oldest, unlocked)
        await gw._get_session_lock("s4")
        assert "s1" not in gw._session_locks
        assert "s4" in gw._session_locks


# ===========================================================================
# Section 18: Fabrication patterns
# ===========================================================================


class TestFabricationPatterns:
    """Tests for _FABRICATION_INDICATORS regex patterns."""

    def test_current_price_matches(self):
        text = "The current price is $42"
        assert any(p.search(text) for p in _FABRICATION_INDICATORS)

    def test_as_of_today_matches(self):
        text = "As of today, the rate has changed"
        assert any(p.search(text) for p in _FABRICATION_INDICATORS)

    def test_i_found_matches(self):
        text = "I found that the file contains errors"
        assert any(p.search(text) for p in _FABRICATION_INDICATORS)

    def test_output_shows_matches(self):
        text = "The output shows success"
        assert any(p.search(text) for p in _FABRICATION_INDICATORS)

    def test_normal_text_no_match(self):
        text = "Cats are popular pets around the world."
        assert not any(p.search(text) for p in _FABRICATION_INDICATORS)


# ===========================================================================
# Section 19: Echo patterns
# ===========================================================================


class TestEchoPatterns:
    """Tests for _ECHO_PATTERNS regex patterns."""

    def test_sure_you_asked(self):
        text = "Sure, you asked me to do this"
        assert any(p.search(text) for p in _ECHO_PATTERNS)

    def test_i_understand(self):
        text = "I understand, you want to deploy"
        assert any(p.search(text) for p in _ECHO_PATTERNS)

    def test_you_want(self):
        text = "You want me to create a function"
        assert any(p.search(text) for p in _ECHO_PATTERNS)

    def test_normal_response_no_echo(self):
        text = "Here is the code you requested:\ndef hello(): pass"
        assert not any(p.search(text) for p in _ECHO_PATTERNS)


# ===========================================================================
# Section 20: ToolCall dataclass
# ===========================================================================


class TestToolCall:
    """Tests for the ToolCall dataclass (meta_cognition)."""

    def test_fields(self):
        tc = ToolCall(name="web_search", args_hash="abc123", iteration=5)
        assert tc.name == "web_search"
        assert tc.args_hash == "abc123"
        assert tc.iteration == 5
