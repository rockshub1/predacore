"""
Tests for Parallel Tool Execution — Phase 6.

Tests cover:
  - Tool dependency classification (independent vs dependent)
  - Mutating tools go to dependent queue
  - Read-only tools go to independent queue
  - Path conflict detection between reads and writes
  - Single tool call passthrough
  - Empty tool call list
"""

from src.predacore.core import _classify_tool_dependencies


def _tc(name: str, **args) -> dict:
    """Helper to create a tool call dict."""
    return {"name": name, "arguments": args}


class TestClassifyToolDependencies:
    def test_single_tool(self):
        calls = [_tc("read_file", path="/a")]
        indep, dep = _classify_tool_dependencies(calls)
        assert len(indep) == 1
        assert len(dep) == 0

    def test_empty_list(self):
        indep, dep = _classify_tool_dependencies([])
        assert indep == []
        assert dep == []

    def test_all_read_only(self):
        calls = [
            _tc("read_file", path="/a"),
            _tc("read_file", path="/b"),
            _tc("web_search", query="python"),
        ]
        indep, dep = _classify_tool_dependencies(calls)
        assert len(indep) == 3
        assert len(dep) == 0

    def test_all_mutating(self):
        calls = [
            _tc("write_file", path="/a", content="x"),
            _tc("run_command", command="ls"),
        ]
        indep, dep = _classify_tool_dependencies(calls)
        assert len(indep) == 0
        assert len(dep) == 2

    def test_mixed_no_path_conflict(self):
        calls = [
            _tc("read_file", path="/a"),
            _tc("web_search", query="hello"),
            _tc("write_file", path="/b", content="x"),
        ]
        indep, dep = _classify_tool_dependencies(calls)
        assert len(indep) == 2  # read_file /a + web_search
        assert len(dep) == 1    # write_file /b

    def test_path_conflict_moves_to_dependent(self):
        calls = [
            _tc("read_file", path="/same/file.txt"),
            _tc("write_file", path="/same/file.txt", content="x"),
        ]
        indep, dep = _classify_tool_dependencies(calls)
        # read_file conflicts with write_file on same path
        assert len(indep) == 0
        assert len(dep) == 2

    def test_no_path_conflict_different_files(self):
        calls = [
            _tc("read_file", path="/a.txt"),
            _tc("write_file", path="/b.txt", content="x"),
        ]
        indep, dep = _classify_tool_dependencies(calls)
        assert len(indep) == 1  # read /a.txt
        assert len(dep) == 1    # write /b.txt

    def test_multiple_reads_same_file(self):
        calls = [
            _tc("read_file", path="/a"),
            _tc("read_file", path="/a"),
        ]
        indep, dep = _classify_tool_dependencies(calls)
        assert len(indep) == 2
        assert len(dep) == 0

    def test_python_exec_is_mutating(self):
        calls = [
            _tc("python_exec", code="print(1)"),
            _tc("read_file", path="/a"),
        ]
        indep, dep = _classify_tool_dependencies(calls)
        assert len(indep) == 1  # read_file
        assert len(dep) == 1    # python_exec

    def test_execute_code_is_mutating(self):
        calls = [
            _tc("execute_code", code="console.log(1)", runtime="node"),
        ]
        indep, dep = _classify_tool_dependencies(calls)
        assert len(dep) == 1

    def test_delete_file_is_mutating(self):
        calls = [
            _tc("delete_file", path="/tmp/x"),
            _tc("read_file", path="/tmp/y"),
        ]
        indep, dep = _classify_tool_dependencies(calls)
        assert len(dep) == 1
        assert dep[0]["name"] == "delete_file"

    def test_desktop_control_is_mutating(self):
        calls = [
            _tc("desktop_control", action="screenshot"),
        ]
        indep, dep = _classify_tool_dependencies(calls)
        assert len(dep) == 1

    def test_marketplace_install_is_mutating(self):
        calls = [
            _tc("marketplace_install", skill_name="test"),
            _tc("web_search", query="hello"),
        ]
        indep, dep = _classify_tool_dependencies(calls)
        assert len(indep) == 1
        assert len(dep) == 1

    def test_multiple_unknown_tools_are_independent(self):
        """Tools not in _MUTATING_TOOLS are treated as read-only."""
        calls = [
            _tc("custom_tool_a", x=1),
            _tc("custom_tool_b", y=2),
        ]
        indep, dep = _classify_tool_dependencies(calls)
        assert len(indep) == 2
        assert len(dep) == 0
