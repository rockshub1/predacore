"""
Tests for JARVIS Tool Pipeline — sequential chaining, parallel fan-out,
variable substitution, conditions, and error handling.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from jarvis.tools.pipeline import (
    PipelineStep,
    PipelineResult,
    ToolPipeline,
    _MAX_PIPELINE_STEPS,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_dispatcher(responses: dict[str, str] | None = None):
    """Create a mock dispatcher that returns predefined responses."""
    default_responses = {
        "read_file": "file content line1\nline2\nline3",
        "python_exec": "42",
        "web_search": "search results here",
        "list_directory": "file_a.py\nfile_b.py",
        "git_context": "branch: main\nstatus: clean",
    }
    resp = {**default_responses, **(responses or {})}

    dispatcher = MagicMock()

    async def mock_dispatch(tool_name, arguments, origin="pipeline"):
        if tool_name in resp:
            return resp[tool_name]
        return f"[Unknown tool: {tool_name}]"

    dispatcher.dispatch = AsyncMock(side_effect=mock_dispatch)
    return dispatcher


def _run(coro):
    """Run an async function synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════
# PipelineStep Tests
# ═══════════════════════════════════════════════════════════════════


class TestPipelineStep:
    def test_from_dict_basic(self):
        step = PipelineStep.from_dict({"tool": "read_file", "args": {"path": "/tmp/x"}})
        assert step.tool == "read_file"
        assert step.args == {"path": "/tmp/x"}
        assert step.on_error == "stop"

    def test_from_dict_with_options(self):
        step = PipelineStep.from_dict({
            "tool": "web_search",
            "args": {"query": "test"},
            "name": "search_step",
            "condition": "not_empty",
            "on_error": "continue",
            "timeout": 30.0,
        })
        assert step.name == "search_step"
        assert step.condition == "not_empty"
        assert step.on_error == "continue"
        assert step.timeout == 30.0

    def test_from_dict_defaults(self):
        step = PipelineStep.from_dict({"tool": "test"})
        assert step.args == {}
        assert step.name == ""
        assert step.condition == ""
        assert step.on_error == "stop"
        assert step.timeout == 0


# ═══════════════════════════════════════════════════════════════════
# Sequential Pipeline Tests
# ═══════════════════════════════════════════════════════════════════


class TestSequentialPipeline:
    def test_single_step(self):
        dispatcher = _make_dispatcher()
        pipeline = ToolPipeline(dispatcher)
        result = _run(pipeline.execute([
            {"tool": "read_file", "args": {"path": "/tmp/x"}},
        ]))
        assert result.ok
        assert result.steps_completed == 1
        assert result.total_steps == 1
        assert result.final_output == "file content line1\nline2\nline3"

    def test_two_step_chain(self):
        dispatcher = _make_dispatcher()
        pipeline = ToolPipeline(dispatcher)
        result = _run(pipeline.execute([
            {"tool": "read_file", "args": {"path": "/tmp/x"}},
            {"tool": "python_exec", "args": {"code": "process('{{prev}}')"}},
        ]))
        assert result.ok
        assert result.steps_completed == 2

    def test_variable_substitution_prev(self):
        dispatcher = _make_dispatcher()
        pipeline = ToolPipeline(dispatcher)
        _run(pipeline.execute([
            {"tool": "read_file", "args": {"path": "/tmp/x"}},
            {"tool": "python_exec", "args": {"code": "data='{{prev}}'"}},
        ]))
        # Check that dispatch was called with substituted args
        call_args = dispatcher.dispatch.call_args_list
        second_call = call_args[1]
        assert "file content line1" in second_call[0][1]["code"]

    def test_variable_substitution_step_ref(self):
        dispatcher = _make_dispatcher()
        pipeline = ToolPipeline(dispatcher)
        result = _run(pipeline.execute([
            {"tool": "read_file", "args": {"path": "/tmp/x"}, "name": "reader"},
            {"tool": "web_search", "args": {"query": "test"}},
            {"tool": "python_exec", "args": {"code": "{{step.reader}}"}},
        ]))
        assert result.ok
        assert result.steps_completed == 3

    def test_empty_pipeline(self):
        dispatcher = _make_dispatcher()
        pipeline = ToolPipeline(dispatcher)
        result = _run(pipeline.execute([]))
        assert not result.ok
        assert "no steps" in result.error.lower()

    def test_too_many_steps(self):
        dispatcher = _make_dispatcher()
        pipeline = ToolPipeline(dispatcher)
        steps = [{"tool": "read_file", "args": {"path": f"/tmp/{i}"}}
                 for i in range(_MAX_PIPELINE_STEPS + 1)]
        result = _run(pipeline.execute(steps))
        assert not result.ok
        assert "too long" in result.error.lower()

    def test_step_without_tool_name(self):
        dispatcher = _make_dispatcher()
        pipeline = ToolPipeline(dispatcher)
        result = _run(pipeline.execute([
            {"tool": "", "args": {}},
        ]))
        assert not result.ok
        assert "no tool name" in result.error.lower()


# ═══════════════════════════════════════════════════════════════════
# Error Handling Tests
# ═══════════════════════════════════════════════════════════════════


class TestPipelineErrors:
    def test_stop_on_error(self):
        dispatcher = _make_dispatcher({"bad_tool": "[Tool error: something broke]"})
        pipeline = ToolPipeline(dispatcher)
        result = _run(pipeline.execute([
            {"tool": "read_file", "args": {"path": "/tmp/x"}},
            {"tool": "bad_tool", "args": {}},
            {"tool": "python_exec", "args": {"code": "should not run"}},
        ]))
        assert not result.ok
        assert result.aborted_at == 1
        assert result.steps_completed == 1  # Only first step completed

    def test_continue_on_error(self):
        dispatcher = _make_dispatcher({"bad_tool": "[Tool error: broke]"})
        pipeline = ToolPipeline(dispatcher)
        result = _run(pipeline.execute([
            {"tool": "bad_tool", "args": {}, "on_error": "continue"},
            {"tool": "read_file", "args": {"path": "/tmp/x"}},
        ]))
        assert result.ok  # Pipeline completed
        assert result.steps_completed == 2

    def test_exception_stop_on_error(self):
        dispatcher = _make_dispatcher()
        dispatcher.dispatch = AsyncMock(side_effect=RuntimeError("boom"))
        pipeline = ToolPipeline(dispatcher)
        result = _run(pipeline.execute([
            {"tool": "read_file", "args": {}},
        ]))
        assert not result.ok
        assert "boom" in result.error

    def test_exception_continue_on_error(self):
        call_count = 0

        async def mixed_dispatch(tool_name, arguments, origin="pipeline"):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first fails")
            return "second succeeds"

        dispatcher = MagicMock()
        dispatcher.dispatch = AsyncMock(side_effect=mixed_dispatch)
        pipeline = ToolPipeline(dispatcher)
        result = _run(pipeline.execute([
            {"tool": "failing", "args": {}, "on_error": "continue"},
            {"tool": "succeeding", "args": {}},
        ]))
        assert result.steps_completed == 1  # Second one succeeded


# ═══════════════════════════════════════════════════════════════════
# Condition Tests
# ═══════════════════════════════════════════════════════════════════


class TestPipelineConditions:
    def test_contains_condition_passes(self):
        dispatcher = _make_dispatcher()
        pipeline = ToolPipeline(dispatcher)
        result = _run(pipeline.execute([
            {"tool": "read_file", "args": {"path": "/tmp/x"}},
            {"tool": "python_exec", "args": {"code": "ok"}, "condition": "contains:line1"},
        ]))
        assert result.ok
        assert result.steps_completed == 2

    def test_contains_condition_skips(self):
        dispatcher = _make_dispatcher()
        pipeline = ToolPipeline(dispatcher)
        result = _run(pipeline.execute([
            {"tool": "read_file", "args": {"path": "/tmp/x"}},
            {"tool": "python_exec", "args": {"code": "nope"}, "condition": "contains:MISSING"},
        ]))
        assert result.ok
        # Step was skipped — check results
        assert any(r["status"] == "skipped" for r in result.results)

    def test_not_empty_condition(self):
        dispatcher = _make_dispatcher()
        pipeline = ToolPipeline(dispatcher)
        result = _run(pipeline.execute([
            {"tool": "read_file", "args": {"path": "/tmp/x"}},
            {"tool": "python_exec", "args": {}, "condition": "not_empty"},
        ]))
        assert result.ok
        assert result.steps_completed == 2

    def test_lines_gt_condition(self):
        dispatcher = _make_dispatcher()
        pipeline = ToolPipeline(dispatcher)
        # read_file returns 3 lines
        result = _run(pipeline.execute([
            {"tool": "read_file", "args": {"path": "/tmp/x"}},
            {"tool": "python_exec", "args": {}, "condition": "lines>2"},
        ]))
        assert result.ok
        assert result.steps_completed == 2

    def test_lines_lt_condition_skips(self):
        dispatcher = _make_dispatcher()
        pipeline = ToolPipeline(dispatcher)
        result = _run(pipeline.execute([
            {"tool": "read_file", "args": {"path": "/tmp/x"}},
            {"tool": "python_exec", "args": {}, "condition": "lines<2"},
        ]))
        # 3 lines is NOT < 2, so skipped
        assert any(r["status"] == "skipped" for r in result.results)


# ═══════════════════════════════════════════════════════════════════
# Parallel Pipeline Tests
# ═══════════════════════════════════════════════════════════════════


class TestParallelPipeline:
    def test_parallel_all_succeed(self):
        dispatcher = _make_dispatcher()
        pipeline = ToolPipeline(dispatcher)
        result = _run(pipeline.execute_parallel([
            {"tool": "web_search", "args": {"query": "a"}},
            {"tool": "read_file", "args": {"path": "/tmp/x"}},
            {"tool": "git_context", "args": {}},
        ]))
        assert result.ok
        assert result.steps_completed == 3
        assert result.total_steps == 3

    def test_parallel_partial_failure(self):
        responses = {
            "web_search": "results",
            "bad_tool": "[Tool error: failed]",
        }
        # Override dispatch to handle unknown tools
        dispatcher = _make_dispatcher(responses)

        async def custom_dispatch(tool_name, arguments, origin="pipeline_parallel"):
            if tool_name == "failing_tool":
                raise RuntimeError("boom")
            return responses.get(tool_name, f"[Unknown tool: {tool_name}]")

        dispatcher.dispatch = AsyncMock(side_effect=custom_dispatch)
        pipeline = ToolPipeline(dispatcher)
        result = _run(pipeline.execute_parallel([
            {"tool": "web_search", "args": {"query": "a"}},
            {"tool": "failing_tool", "args": {}},
        ]))
        assert not result.ok
        assert result.steps_completed == 1

    def test_parallel_empty(self):
        dispatcher = _make_dispatcher()
        pipeline = ToolPipeline(dispatcher)
        result = _run(pipeline.execute_parallel([]))
        assert not result.ok
        assert "no steps" in result.error.lower()


# ═══════════════════════════════════════════════════════════════════
# PipelineResult Tests
# ═══════════════════════════════════════════════════════════════════


class TestPipelineResult:
    def test_to_dict(self):
        result = PipelineResult(
            ok=True,
            steps_completed=2,
            total_steps=2,
            results=[
                {"step": 0, "tool": "read_file", "status": "ok",
                 "output": "content", "elapsed_ms": 15.0},
                {"step": 1, "tool": "python_exec", "status": "ok",
                 "output": "42", "elapsed_ms": 30.0},
            ],
            final_output="42",
            elapsed_ms=45.5,
        )
        d = result.to_dict()
        assert d["ok"] is True
        assert d["steps_completed"] == 2
        assert len(d["step_results"]) == 2
        assert d["final_output"] == "42"

    def test_to_dict_truncates_output(self):
        long_output = "x" * 5000
        result = PipelineResult(
            ok=True, steps_completed=1, total_steps=1,
            results=[{"step": 0, "tool": "t", "status": "ok",
                       "output": long_output, "elapsed_ms": 10.0}],
            final_output=long_output, elapsed_ms=10.0,
        )
        d = result.to_dict()
        assert len(d["final_output"]) <= 2000
        assert len(d["step_results"][0]["output_preview"]) <= 500


# ═══════════════════════════════════════════════════════════════════
# Variable Substitution Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestVariableSubstitution:
    def test_prev_lines_variable(self):
        dispatcher = _make_dispatcher()
        pipeline = ToolPipeline(dispatcher)
        _run(pipeline.execute([
            {"tool": "read_file", "args": {"path": "/tmp/x"}},
            {"tool": "python_exec", "args": {"code": "lines={{prev_lines}}"}},
        ]))
        call_args = dispatcher.dispatch.call_args_list[1]
        assert "lines=3" in call_args[0][1]["code"]

    def test_prev_len_variable(self):
        dispatcher = _make_dispatcher({"read_file": "hello"})
        pipeline = ToolPipeline(dispatcher)
        _run(pipeline.execute([
            {"tool": "read_file", "args": {"path": "/tmp/x"}},
            {"tool": "python_exec", "args": {"code": "len={{prev_len}}"}},
        ]))
        call_args = dispatcher.dispatch.call_args_list[1]
        assert "len=5" in call_args[0][1]["code"]

    def test_unknown_variable_left_intact(self):
        dispatcher = _make_dispatcher()
        pipeline = ToolPipeline(dispatcher)
        _run(pipeline.execute([
            {"tool": "read_file", "args": {"path": "/tmp/x"}},
            {"tool": "python_exec", "args": {"code": "{{unknown_var}}"}},
        ]))
        call_args = dispatcher.dispatch.call_args_list[1]
        assert "{{unknown_var}}" in call_args[0][1]["code"]

    def test_nested_dict_substitution(self):
        dispatcher = _make_dispatcher({"read_file": "nested_value"})
        pipeline = ToolPipeline(dispatcher)
        _run(pipeline.execute([
            {"tool": "read_file", "args": {"path": "/tmp/x"}},
            {"tool": "python_exec", "args": {"config": {"key": "{{prev}}"}}},
        ]))
        call_args = dispatcher.dispatch.call_args_list[1]
        assert call_args[0][1]["config"]["key"] == "nested_value"

    def test_list_substitution(self):
        dispatcher = _make_dispatcher({"read_file": "list_val"})
        pipeline = ToolPipeline(dispatcher)
        _run(pipeline.execute([
            {"tool": "read_file", "args": {"path": "/tmp/x"}},
            {"tool": "python_exec", "args": {"items": ["{{prev}}", "static"]}},
        ]))
        call_args = dispatcher.dispatch.call_args_list[1]
        assert call_args[0][1]["items"][0] == "list_val"
        assert call_args[0][1]["items"][1] == "static"
