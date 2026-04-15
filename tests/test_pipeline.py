"""Tests for jarvis.tools.pipeline — ToolPipeline sequential + parallel execution."""
from __future__ import annotations

import asyncio
import pytest

from jarvis.tools.pipeline import (
    PipelineStep,
    PipelineResult,
    ToolPipeline,
    _MAX_PIPELINE_STEPS,
)


# ── Fake dispatcher for testing ──────────────────────────────────────


class FakeDispatcher:
    """Minimal dispatcher that records calls and returns canned responses."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.calls: list[tuple[str, dict, str]] = []
        self._responses = responses or {}
        self._default = "ok"

    async def dispatch(
        self,
        tool_name: str,
        arguments: dict,
        *,
        origin: str = "test",
    ) -> str:
        self.calls.append((tool_name, dict(arguments), origin))
        if tool_name in self._responses:
            resp = self._responses[tool_name]
            if callable(resp):
                return resp(arguments)
            return resp
        return self._default


class ErrorDispatcher(FakeDispatcher):
    """Dispatcher that raises on specific tools."""

    def __init__(self, error_tools: set[str] | None = None):
        super().__init__()
        self._error_tools = error_tools or set()

    async def dispatch(self, tool_name, arguments, *, origin="test"):
        self.calls.append((tool_name, dict(arguments), origin))
        if tool_name in self._error_tools:
            raise RuntimeError(f"Simulated failure in {tool_name}")
        return f"ok:{tool_name}"


class SlowDispatcher(FakeDispatcher):
    """Dispatcher that adds delay to simulate slow tools."""

    def __init__(self, delay: float = 0.1):
        super().__init__()
        self._delay = delay

    async def dispatch(self, tool_name, arguments, *, origin="test"):
        self.calls.append((tool_name, dict(arguments), origin))
        await asyncio.sleep(self._delay)
        return f"slow_result:{tool_name}"


# ── PipelineStep Tests ───────────────────────────────────────────────


class TestPipelineStep:
    def test_from_dict_minimal(self):
        step = PipelineStep.from_dict({"tool": "read_file"})
        assert step.tool == "read_file"
        assert step.args == {}
        assert step.name == ""
        assert step.condition == ""
        assert step.on_error == "stop"
        assert step.timeout == 0

    def test_from_dict_full(self):
        step = PipelineStep.from_dict({
            "tool": "web_search",
            "args": {"query": "test"},
            "name": "search_step",
            "condition": "not_empty",
            "on_error": "continue",
            "timeout": 30,
        })
        assert step.tool == "web_search"
        assert step.args == {"query": "test"}
        assert step.name == "search_step"
        assert step.condition == "not_empty"
        assert step.on_error == "continue"
        assert step.timeout == 30.0

    def test_from_dict_defaults(self):
        step = PipelineStep.from_dict({})
        assert step.tool == ""
        assert step.on_error == "stop"


# ── Sequential Pipeline Tests ────────────────────────────────────────


class TestSequentialPipeline:
    @pytest.mark.asyncio
    async def test_single_step(self):
        dispatcher = FakeDispatcher({"read_file": "file contents here"})
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "read_file", "args": {"path": "test.txt"}},
        ])

        assert result.ok is True
        assert result.steps_completed == 1
        assert result.total_steps == 1
        assert result.final_output == "file contents here"
        assert len(dispatcher.calls) == 1
        assert dispatcher.calls[0][0] == "read_file"

    @pytest.mark.asyncio
    async def test_two_step_chain(self):
        dispatcher = FakeDispatcher({
            "read_file": "hello world",
            "python_exec": "result: 11",
        })
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "read_file", "args": {"path": "data.txt"}},
            {"tool": "python_exec", "args": {"code": "len('{{prev}}')"}},
        ])

        assert result.ok is True
        assert result.steps_completed == 2
        assert result.final_output == "result: 11"
        # Verify variable substitution happened
        assert dispatcher.calls[1][1]["code"] == "len('hello world')"

    @pytest.mark.asyncio
    async def test_variable_substitution_prev(self):
        dispatcher = FakeDispatcher()
        dispatcher._default = "test_output"
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "step_a", "args": {}},
            {"tool": "step_b", "args": {"input": "got: {{prev}}"}},
        ])

        assert result.ok
        assert dispatcher.calls[1][1]["input"] == "got: test_output"

    @pytest.mark.asyncio
    async def test_variable_substitution_step_by_index(self):
        responses = {}
        call_count = [0]

        class IndexDispatcher:
            calls = []

            async def dispatch(self, tool_name, arguments, *, origin="test"):
                self.calls.append((tool_name, dict(arguments), origin))
                call_count[0] += 1
                return f"output_{call_count[0]}"

        dispatcher = IndexDispatcher()
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "a", "args": {}, "name": "first"},
            {"tool": "b", "args": {}},
            {"tool": "c", "args": {"ref0": "{{step.0}}", "ref_name": "{{step.first}}"}},
        ])

        assert result.ok
        assert result.steps_completed == 3
        # Step c should reference step 0's output
        assert dispatcher.calls[2][1]["ref0"] == "output_1"
        assert dispatcher.calls[2][1]["ref_name"] == "output_1"

    @pytest.mark.asyncio
    async def test_empty_pipeline(self):
        pipeline = ToolPipeline(FakeDispatcher())
        result = await pipeline.execute([])
        assert result.ok is False
        assert "no steps" in result.error.lower()

    @pytest.mark.asyncio
    async def test_too_many_steps(self):
        steps = [{"tool": f"t{i}", "args": {}} for i in range(_MAX_PIPELINE_STEPS + 1)]
        pipeline = ToolPipeline(FakeDispatcher())
        result = await pipeline.execute(steps)
        assert result.ok is False
        assert "too long" in result.error.lower()

    @pytest.mark.asyncio
    async def test_step_with_no_tool_name(self):
        pipeline = ToolPipeline(FakeDispatcher())
        result = await pipeline.execute([{"tool": "", "args": {}}])
        assert result.ok is False
        assert "no tool name" in result.error.lower()

    @pytest.mark.asyncio
    async def test_invalid_step_type(self):
        pipeline = ToolPipeline(FakeDispatcher())
        result = await pipeline.execute(["not_a_dict"])  # type: ignore
        assert result.ok is False
        assert "invalid step type" in result.error.lower()

    @pytest.mark.asyncio
    async def test_error_stops_pipeline(self):
        dispatcher = ErrorDispatcher(error_tools={"step_b"})
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "step_a", "args": {}},
            {"tool": "step_b", "args": {}, "on_error": "stop"},
            {"tool": "step_c", "args": {}},
        ])

        assert result.ok is False
        assert result.aborted_at == 1
        assert result.steps_completed == 1
        assert len(dispatcher.calls) == 2  # step_c never called

    @pytest.mark.asyncio
    async def test_error_continue_mode(self):
        dispatcher = ErrorDispatcher(error_tools={"step_b"})
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "step_a", "args": {}},
            {"tool": "step_b", "args": {}, "on_error": "continue"},
            {"tool": "step_c", "args": {}},
        ])

        # Pipeline should complete despite step_b failure
        assert len(dispatcher.calls) == 3
        assert result.results[1]["status"] == "error"

    @pytest.mark.asyncio
    async def test_error_output_in_result(self):
        dispatcher = FakeDispatcher({"bad": "[Tool error: file not found]"})
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "bad", "args": {}, "on_error": "stop"},
        ])

        assert result.ok is False
        assert result.aborted_at == 0


# ── Condition Tests ──────────────────────────────────────────────────


class TestPipelineConditions:
    @pytest.mark.asyncio
    async def test_condition_contains_pass(self):
        dispatcher = FakeDispatcher()
        dispatcher._default = "hello world here"
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "a", "args": {}},
            {"tool": "b", "args": {}, "condition": "contains:hello"},
        ])

        assert result.ok
        assert result.steps_completed == 2
        assert len(dispatcher.calls) == 2

    @pytest.mark.asyncio
    async def test_condition_contains_skip(self):
        dispatcher = FakeDispatcher()
        dispatcher._default = "hello world here"
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "a", "args": {}},
            {"tool": "b", "args": {}, "condition": "contains:missing_text"},
        ])

        assert result.ok
        assert len(dispatcher.calls) == 1  # b was skipped
        assert result.results[1]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_condition_not_empty(self):
        dispatcher = FakeDispatcher()
        dispatcher._default = "data"
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "a", "args": {}},
            {"tool": "b", "args": {}, "condition": "not_empty"},
        ])

        assert result.ok
        assert result.steps_completed == 2

    @pytest.mark.asyncio
    async def test_condition_not_error(self):
        dispatcher = FakeDispatcher()
        dispatcher._default = "clean output"
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "a", "args": {}},
            {"tool": "b", "args": {}, "condition": "not_error"},
        ])

        assert result.ok
        assert result.steps_completed == 2

    @pytest.mark.asyncio
    async def test_condition_is_error_skip(self):
        dispatcher = FakeDispatcher()
        dispatcher._default = "clean output"
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "a", "args": {}},
            {"tool": "b", "args": {}, "condition": "is_error"},
        ])

        # b should be skipped because prev output isn't an error
        assert result.results[1]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_condition_lines_gt(self):
        dispatcher = FakeDispatcher()
        dispatcher._default = "line1\nline2\nline3\nline4"
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "a", "args": {}},
            {"tool": "b", "args": {}, "condition": "lines>2"},
        ])

        assert result.steps_completed == 2  # 4 lines > 2

    @pytest.mark.asyncio
    async def test_condition_lines_lt_skip(self):
        dispatcher = FakeDispatcher()
        dispatcher._default = "line1\nline2\nline3\nline4"
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "a", "args": {}},
            {"tool": "b", "args": {}, "condition": "lines<2"},
        ])

        assert result.results[1]["status"] == "skipped"  # 4 lines not < 2

    def test_check_condition_static(self):
        """Test _check_condition directly."""
        check = ToolPipeline._check_condition

        assert check("contains:hello", "hello world") is True
        assert check("contains:missing", "hello world") is False
        assert check("not_contains:missing", "hello world") is True
        assert check("not_contains:hello", "hello world") is False
        assert check("starts_with:hello", "hello world") is True
        assert check("starts_with:world", "hello world") is False
        assert check("not_empty", "data") is True
        assert check("not_empty", "   ") is False
        assert check("is_error", "[Error: bad thing]") is True
        assert check("is_error", "clean output") is False
        assert check("not_error", "clean output") is True
        assert check("lines>2", "a\nb\nc") is True
        assert check("lines>5", "a\nb\nc") is False
        assert check("lines<5", "a\nb\nc") is True
        assert check("unknown_condition", "whatever") is True  # defaults True


# ── Parallel Pipeline Tests ──────────────────────────────────────────


class TestParallelPipeline:
    @pytest.mark.asyncio
    async def test_parallel_basic(self):
        dispatcher = FakeDispatcher({
            "web_search": "search results",
            "read_file": "file contents",
        })
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute_parallel([
            {"tool": "web_search", "args": {"query": "test"}},
            {"tool": "read_file", "args": {"path": "data.txt"}},
        ])

        assert result.ok
        assert result.steps_completed == 2
        assert result.total_steps == 2
        assert len(dispatcher.calls) == 2

    @pytest.mark.asyncio
    async def test_parallel_all_concurrent(self):
        """Verify steps actually run concurrently, not sequentially."""
        dispatcher = SlowDispatcher(delay=0.1)
        pipeline = ToolPipeline(dispatcher)

        import time
        t0 = time.time()
        result = await pipeline.execute_parallel([
            {"tool": "a", "args": {}},
            {"tool": "b", "args": {}},
            {"tool": "c", "args": {}},
        ])
        elapsed = time.time() - t0

        assert result.ok
        assert result.steps_completed == 3
        # If truly parallel, should take ~0.1s not ~0.3s
        assert elapsed < 0.25, f"Parallel took {elapsed:.2f}s — not concurrent!"

    @pytest.mark.asyncio
    async def test_parallel_partial_failure(self):
        dispatcher = ErrorDispatcher(error_tools={"bad_tool"})
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute_parallel([
            {"tool": "good_tool", "args": {}},
            {"tool": "bad_tool", "args": {}},
        ])

        assert result.ok is False
        assert result.steps_completed == 1
        assert result.total_steps == 2
        # Both should have been attempted
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_parallel_empty(self):
        pipeline = ToolPipeline(FakeDispatcher())
        result = await pipeline.execute_parallel([])
        assert result.ok is False
        assert "no steps" in result.error.lower()

    @pytest.mark.asyncio
    async def test_parallel_too_many(self):
        steps = [{"tool": f"t{i}", "args": {}} for i in range(_MAX_PIPELINE_STEPS + 1)]
        pipeline = ToolPipeline(FakeDispatcher())
        result = await pipeline.execute_parallel(steps)
        assert result.ok is False
        assert "too many" in result.error.lower()

    @pytest.mark.asyncio
    async def test_parallel_combined_output(self):
        dispatcher = FakeDispatcher({
            "a": "result_a",
            "b": "result_b",
        })
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute_parallel([
            {"tool": "a", "args": {}},
            {"tool": "b", "args": {}},
        ])

        assert "result_a" in result.final_output
        assert "result_b" in result.final_output


# ── PipelineResult Tests ─────────────────────────────────────────────


class TestPipelineResult:
    def test_to_dict_format(self):
        result = PipelineResult(
            ok=True,
            steps_completed=2,
            total_steps=2,
            results=[
                {"step": 0, "tool": "a", "status": "ok", "output": "x" * 1000, "elapsed_ms": 50.123},
                {"step": 1, "tool": "b", "status": "ok", "output": "y", "elapsed_ms": 30.456},
            ],
            final_output="y",
            elapsed_ms=80.579,
        )
        d = result.to_dict()

        assert d["ok"] is True
        assert d["steps_completed"] == 2
        assert d["total_steps"] == 2
        assert d["elapsed_ms"] == 80.6  # rounded
        assert len(d["step_results"]) == 2
        # Output preview capped at 500 chars
        assert len(d["step_results"][0]["output_preview"]) <= 500

    def test_to_dict_with_error(self):
        result = PipelineResult(
            ok=False,
            steps_completed=1,
            total_steps=3,
            results=[],
            final_output="",
            elapsed_ms=10.0,
            aborted_at=1,
            error="Step 1 failed",
        )
        d = result.to_dict()
        assert d["ok"] is False
        assert d["aborted_at"] == 1
        assert d["error"] == "Step 1 failed"


# ── Variable Substitution Edge Cases ─────────────────────────────────


class TestVariableSubstitution:
    @pytest.mark.asyncio
    async def test_prev_lines_variable(self):
        dispatcher = FakeDispatcher()
        dispatcher._default = "line1\nline2\nline3"
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "a", "args": {}},
            {"tool": "b", "args": {"count": "{{prev_lines}}"}},
        ])

        assert result.ok
        assert dispatcher.calls[1][1]["count"] == "3"

    @pytest.mark.asyncio
    async def test_prev_len_variable(self):
        dispatcher = FakeDispatcher()
        dispatcher._default = "hello"
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "a", "args": {}},
            {"tool": "b", "args": {"size": "{{prev_len}}"}},
        ])

        assert result.ok
        assert dispatcher.calls[1][1]["size"] == "5"

    @pytest.mark.asyncio
    async def test_unknown_variable_left_as_is(self):
        dispatcher = FakeDispatcher()
        dispatcher._default = "ok"
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "a", "args": {}},
            {"tool": "b", "args": {"x": "{{unknown_var}}"}},
        ])

        assert result.ok
        assert dispatcher.calls[1][1]["x"] == "{{unknown_var}}"

    @pytest.mark.asyncio
    async def test_nested_dict_substitution(self):
        dispatcher = FakeDispatcher()
        dispatcher._default = "test_value"
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "a", "args": {}},
            {"tool": "b", "args": {"nested": {"inner": "{{prev}}"}}},
        ])

        assert result.ok
        assert dispatcher.calls[1][1]["nested"]["inner"] == "test_value"

    @pytest.mark.asyncio
    async def test_list_substitution(self):
        dispatcher = FakeDispatcher()
        dispatcher._default = "item"
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            {"tool": "a", "args": {}},
            {"tool": "b", "args": {"items": ["{{prev}}", "static"]}},
        ])

        assert result.ok
        assert dispatcher.calls[1][1]["items"] == ["item", "static"]

    @pytest.mark.asyncio
    async def test_piplinestep_objects_work(self):
        dispatcher = FakeDispatcher()
        dispatcher._default = "data"
        pipeline = ToolPipeline(dispatcher)

        result = await pipeline.execute([
            PipelineStep(tool="a", args={"x": 1}),
            PipelineStep(tool="b", args={"input": "{{prev}}"}),
        ])

        assert result.ok
        assert result.steps_completed == 2
        assert dispatcher.calls[1][1]["input"] == "data"
