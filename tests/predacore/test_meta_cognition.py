"""
Tests for Meta-Cognition Engine — Phase 4.

Tests cover:
  - Response evaluation (grounding, echo detection, fabrication risk)
  - Loop detection (repeated calls, oscillation)
  - Should-ask-for-help logic
"""

from src.predacore.agents.meta_cognition import (
    ResponseEvaluation,
    detect_loop,
    evaluate_response,
    should_ask_for_help,
)

# ── Response evaluation ──────────────────────────────────────────────

class TestResponseEvaluation:
    def test_good_response_high_confidence(self):
        ev = evaluate_response(
            user_message="What files are in this directory?",
            response="Here are the files: main.py, utils.py, config.py",
            tools_used=["run_command"],
            tool_results=["main.py\nutils.py\nconfig.py"],
        )
        assert ev.confidence >= 0.7
        assert ev.should_send
        assert not ev.is_echo
        assert not ev.has_fabrication_risk

    def test_empty_response(self):
        ev = evaluate_response("hello", "", [], [])
        assert ev.confidence == 0.0
        assert "empty_response" in ev.issues
        assert not ev.should_send

    def test_echo_detection(self):
        ev = evaluate_response(
            user_message="How do I configure the database connection string?",
            response="Sure, you're asking how to configure the database connection string.",
            tools_used=[],
            tool_results=[],
        )
        assert ev.is_echo
        assert "echo_detected" in ev.issues

    def test_fabrication_risk_no_tools(self):
        ev = evaluate_response(
            user_message="What is the current price of Bitcoin?",
            response="The current price is $45,000.",
            tools_used=[],
            tool_results=[],
        )
        assert ev.has_fabrication_risk
        assert not ev.grounded
        assert "fabrication_risk_no_tools" in ev.issues

    def test_no_fabrication_with_tools(self):
        ev = evaluate_response(
            user_message="What is the current price of Bitcoin?",
            response="The current price is $45,000.",
            tools_used=["web_search"],
            tool_results=["Bitcoin price: $45,000"],
        )
        assert not ev.has_fabrication_risk

    def test_brief_response_to_complex_question(self):
        long_question = "Can you explain the differences between SQL and NoSQL databases, including their use cases, performance characteristics, and scaling strategies?"
        ev = evaluate_response(long_question, "Yes.", [], [])
        assert "response_too_brief" in ev.issues

    def test_well_grounded_response(self):
        ev = evaluate_response(
            user_message="Read the config file",
            response="The configuration file contains database settings, API keys, and logging configuration. The database host is localhost with port 5432.",
            tools_used=["read_file"],
            tool_results=["database:\n  host: localhost\n  port: 5432\nlogging:\n  level: INFO\napi_keys:\n  service: key123"],
        )
        assert ev.grounded
        assert ev.confidence >= 0.7

    def test_should_send_threshold(self):
        ev = ResponseEvaluation(confidence=0.3)
        assert ev.should_send
        ev2 = ResponseEvaluation(confidence=0.29)
        assert not ev2.should_send
        ev3 = ResponseEvaluation(confidence=0.5, is_echo=True)
        assert not ev3.should_send


# ── Loop detection ────────────────────────────────────────────────────

class TestLoopDetection:
    def test_no_loop_short_history(self):
        history = [("read_file", {"path": "/a"})]
        assert detect_loop(history) is False

    def test_no_loop_varied_calls(self):
        history = [
            ("read_file", {"path": "/a"}),
            ("write_file", {"path": "/b", "content": "x"}),
            ("run_command", {"command": "ls"}),
        ]
        assert detect_loop(history) is False

    def test_same_call_repeated(self):
        history = [
            ("read_file", {"path": "/a"}),
            ("read_file", {"path": "/a"}),
            ("read_file", {"path": "/a"}),
        ]
        assert detect_loop(history) is True

    def test_different_args_no_loop(self):
        history = [
            ("read_file", {"path": "/a"}),
            ("read_file", {"path": "/b"}),
            ("read_file", {"path": "/c"}),
        ]
        assert detect_loop(history) is False

    def test_oscillation_pattern(self):
        history = [
            ("read_file", {"path": "/a"}),
            ("write_file", {"path": "/a", "content": "x"}),
            ("read_file", {"path": "/a"}),
            ("write_file", {"path": "/a", "content": "x"}),
        ]
        assert detect_loop(history) is True

    def test_custom_max_repeats(self):
        history = [
            ("read_file", {"path": "/a"}),
            ("read_file", {"path": "/a"}),
        ]
        assert detect_loop(history, max_repeats=3) is False
        assert detect_loop(history, max_repeats=2) is True


# ── Should ask for help ───────────────────────────────────────────────

class TestShouldAskForHelp:
    def test_no_help_needed(self):
        result = should_ask_for_help(
            tool_errors=[], iteration_count=1,
            max_iterations=10, tool_history=[],
        )
        assert result is None

    def test_multiple_errors(self):
        result = should_ask_for_help(
            tool_errors=["err1", "err2", "err3"],
            iteration_count=1, max_iterations=10,
            tool_history=[],
        )
        assert result is not None
        assert "3 tool errors" in result

    def test_near_max_iterations(self):
        result = should_ask_for_help(
            tool_errors=[], iteration_count=9,
            max_iterations=10,
            tool_history=[],
        )
        assert result is not None
        assert "9 of 10" in result

    def test_loop_triggers_help(self):
        history = [
            ("read_file", {"path": "/a"}),
            ("read_file", {"path": "/a"}),
            ("read_file", {"path": "/a"}),
        ]
        result = should_ask_for_help(
            tool_errors=[], iteration_count=3,
            max_iterations=10, tool_history=history,
        )
        assert result is not None
        assert "loop" in result.lower()

    def test_single_error_no_help(self):
        result = should_ask_for_help(
            tool_errors=["one error"],
            iteration_count=1, max_iterations=10,
            tool_history=[],
        )
        assert result is None
