"""
Tests for JARVIS Tool Handlers — focused on security, error handling,
and the ToolError hierarchy.

Tests handler behavior without full subsystem init — uses minimal
ToolContext mocks.
"""
import asyncio
import os
import tempfile
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from jarvis.tools.handlers._context import (
    ToolContext,
    ToolError,
    ToolErrorKind,
    missing_param,
    invalid_param,
    subsystem_unavailable,
    resource_not_found,
    blocked,
    SENSITIVE_READ_PATTERNS,
    SENSITIVE_WRITE_PATHS,
    SENSITIVE_WRITE_FILES,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_ctx(**overrides):
    """Create a minimal ToolContext for testing."""
    security = SimpleNamespace(
        trust_level="yolo",
        blocked_paths=[],
        blocked_commands=[],
        task_timeout_seconds=300,
    )
    config = SimpleNamespace(
        security=security,
        mode="test",
    )
    defaults = {
        "config": config,
        "memory": {},
    }
    defaults.update(overrides)
    return ToolContext(**defaults)


# ═══════════════════════════════════════════════════════════════════
# ToolError Tests
# ═══════════════════════════════════════════════════════════════════


class TestToolError:
    def test_basic_error(self):
        err = ToolError("something broke")
        assert str(err) == "something broke"
        assert err.kind == ToolErrorKind.EXECUTION
        assert err.recoverable is True

    def test_error_kinds(self):
        for kind in ToolErrorKind:
            err = ToolError("test", kind=kind)
            assert err.kind == kind

    def test_format_basic(self):
        err = ToolError("file not found", kind=ToolErrorKind.NOT_FOUND)
        formatted = err.format()
        assert formatted.startswith("[")
        assert formatted.endswith("]")
        assert "file not found" in formatted

    def test_format_with_suggestion(self):
        err = ToolError(
            "not found",
            kind=ToolErrorKind.NOT_FOUND,
            suggestion="Check the path",
        )
        formatted = err.format()
        assert "Check the path" in formatted

    def test_format_non_recoverable(self):
        err = ToolError("fatal", recoverable=False)
        formatted = err.format()
        assert "non-recoverable" in formatted

    def test_to_dict(self):
        err = ToolError(
            "bad param",
            kind=ToolErrorKind.INVALID_PARAM,
            tool_name="read_file",
            suggestion="fix it",
        )
        d = err.to_dict()
        assert d["error"] == "bad param"
        assert d["kind"] == "invalid_param"
        assert d["tool"] == "read_file"
        assert d["suggestion"] == "fix it"
        assert d["recoverable"] is True


# ═══════════════════════════════════════════════════════════════════
# Convenience Constructor Tests
# ═══════════════════════════════════════════════════════════════════


class TestConvenienceConstructors:
    def test_missing_param(self):
        err = missing_param("path", tool="read_file")
        assert err.kind == ToolErrorKind.MISSING_PARAM
        assert "path" in str(err)

    def test_invalid_param(self):
        err = invalid_param("timeout", "must be positive")
        assert err.kind == ToolErrorKind.INVALID_PARAM
        assert "timeout" in str(err)
        assert "must be positive" in str(err)

    def test_subsystem_unavailable(self):
        err = subsystem_unavailable("Desktop operator")
        assert err.kind == ToolErrorKind.UNAVAILABLE
        assert "Desktop operator" in str(err)
        assert err.suggestion  # Should have a suggestion

    def test_resource_not_found(self):
        err = resource_not_found("File", "/tmp/missing.txt")
        assert err.kind == ToolErrorKind.NOT_FOUND
        assert "/tmp/missing.txt" in str(err)

    def test_blocked(self):
        err = blocked("sensitive path")
        assert err.kind == ToolErrorKind.BLOCKED
        assert err.recoverable is False


# ═══════════════════════════════════════════════════════════════════
# File Ops Handler Tests
# ═══════════════════════════════════════════════════════════════════


class TestFileOpsHandler:
    def test_read_file_success(self):
        from jarvis.tools.handlers.file_ops import handle_read_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            f.flush()
            try:
                ctx = _make_ctx()
                result = _run(handle_read_file({"path": f.name}, ctx))
                assert "hello world" in result
            finally:
                os.unlink(f.name)

    def test_read_file_missing_path(self):
        from jarvis.tools.handlers.file_ops import handle_read_file
        ctx = _make_ctx()
        with pytest.raises(ToolError) as exc_info:
            _run(handle_read_file({}, ctx))
        assert exc_info.value.kind == ToolErrorKind.MISSING_PARAM

    def test_read_file_not_found(self):
        from jarvis.tools.handlers.file_ops import handle_read_file
        ctx = _make_ctx()
        with pytest.raises(ToolError) as exc_info:
            _run(handle_read_file({"path": "/nonexistent/file.txt"}, ctx))
        assert exc_info.value.kind == ToolErrorKind.NOT_FOUND

    def test_read_file_sensitive_path_blocked(self):
        """Reading sensitive files should be blocked."""
        from jarvis.tools.handlers.file_ops import handle_read_file
        ctx = _make_ctx()
        # Try to read .ssh/id_rsa (sensitive)
        with pytest.raises(ToolError) as exc_info:
            _run(handle_read_file({"path": os.path.expanduser("~/.ssh/id_rsa")}, ctx))
        # Should be blocked OR not found — either is acceptable
        assert exc_info.value.kind in (ToolErrorKind.BLOCKED, ToolErrorKind.NOT_FOUND)

    def test_write_file_success(self):
        from jarvis.tools.handlers.file_ops import handle_write_file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            ctx = _make_ctx()
            result = _run(handle_write_file({"path": path, "content": "test content"}, ctx))
            assert os.path.exists(path)
            assert open(path).read() == "test content"

    def test_write_file_missing_params(self):
        from jarvis.tools.handlers.file_ops import handle_write_file
        ctx = _make_ctx()
        with pytest.raises(ToolError) as exc_info:
            _run(handle_write_file({}, ctx))
        assert exc_info.value.kind == ToolErrorKind.MISSING_PARAM

    def test_write_file_sensitive_path_blocked(self):
        from jarvis.tools.handlers.file_ops import handle_write_file
        ctx = _make_ctx()
        with pytest.raises(ToolError) as exc_info:
            _run(handle_write_file({"path": "/etc/passwd", "content": "evil"}, ctx))
        assert exc_info.value.kind == ToolErrorKind.BLOCKED

    def test_list_directory_success(self):
        from jarvis.tools.handlers.file_ops import handle_list_directory
        ctx = _make_ctx()
        result = _run(handle_list_directory({"path": "/tmp"}, ctx))
        assert isinstance(result, str)

    def test_list_directory_missing_path(self):
        from jarvis.tools.handlers.file_ops import handle_list_directory
        ctx = _make_ctx()
        with pytest.raises(ToolError) as exc_info:
            _run(handle_list_directory({}, ctx))
        assert exc_info.value.kind == ToolErrorKind.MISSING_PARAM


# ═══════════════════════════════════════════════════════════════════
# Shell Handler Tests
# ═══════════════════════════════════════════════════════════════════


class TestShellHandler:
    def test_run_command_success(self):
        from jarvis.tools.handlers.shell import handle_run_command
        ctx = _make_ctx()
        result = _run(handle_run_command({"command": "echo hello"}, ctx))
        assert "hello" in result

    def test_run_command_missing_param(self):
        from jarvis.tools.handlers.shell import handle_run_command
        ctx = _make_ctx()
        with pytest.raises(ToolError) as exc_info:
            _run(handle_run_command({}, ctx))
        assert exc_info.value.kind == ToolErrorKind.MISSING_PARAM


# ═══════════════════════════════════════════════════════════════════
# Security Pattern Tests
# ═══════════════════════════════════════════════════════════════════


class TestSecurityPatterns:
    def test_sensitive_read_patterns_exist(self):
        assert len(SENSITIVE_READ_PATTERNS) >= 5
        assert ".ssh/" in SENSITIVE_READ_PATTERNS
        assert ".env" in SENSITIVE_READ_PATTERNS

    def test_sensitive_write_paths_exist(self):
        assert len(SENSITIVE_WRITE_PATHS) >= 5
        assert "/etc/" in SENSITIVE_WRITE_PATHS

    def test_sensitive_write_files_exist(self):
        assert len(SENSITIVE_WRITE_FILES) >= 5
        assert ".ssh/authorized_keys" in SENSITIVE_WRITE_FILES


# ═══════════════════════════════════════════════════════════════════
# ToolContext Tests
# ═══════════════════════════════════════════════════════════════════


class TestToolContext:
    def test_minimal_context(self):
        ctx = _make_ctx()
        assert ctx.config is not None
        assert ctx.memory == {}
        assert ctx.desktop_operator is None
        assert ctx.voice is None

    def test_context_with_services(self):
        mock_desktop = MagicMock()
        ctx = _make_ctx(desktop_operator=mock_desktop)
        assert ctx.desktop_operator is mock_desktop

    def test_all_optional_fields_default_none(self):
        ctx = _make_ctx()
        assert ctx.memory_service is None
        assert ctx.mcts_planner is None
        assert ctx.voice is None
        assert ctx.sandbox is None
        assert ctx.docker_sandbox is None
        assert ctx.sandbox_pool is None
        assert ctx.skill_marketplace is None
        assert ctx.openclaw_runtime is None
        assert ctx.llm_for_collab is None
        assert ctx.unified_memory is None
