"""
Tests for Docker sandbox wiring into JARVIS core ToolExecutor.

Tests the integration of SubprocessSandboxManager and DockerSandboxManager
into the ToolExecutor, verifying sandbox selection logic, result formatting,
and tool dispatch.
"""
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Minimal config stubs ─────────────────────────────────────────────

@dataclass
class SecurityConfig:
    trust_level: str = "yolo"
    permission_mode: str = "auto"
    approval_timeout: int = 30
    remember_approvals: bool = False
    docker_sandbox: bool = False
    allowed_tools: list[str] = field(default_factory=list)
    blocked_tools: list[str] = field(default_factory=list)
    max_concurrent_tasks: int = 5
    task_timeout_seconds: int = 300


@dataclass
class DaemonConfig:
    enabled: bool = False


@dataclass
class ChannelConfig:
    telegram_token: str = ""
    discord_token: str = ""
    whatsapp_token: str = ""
    webchat_port: int = 3000


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "test"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    context_window: int = 32000


@dataclass
class MemoryConfig:
    enabled: bool = False
    backend: str = "sqlite"
    path: str = "/tmp/test.db"


@dataclass
class AntigravityConfig:
    enabled: bool = False
    api_key: str = ""


@dataclass
class JARVISConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    channels: ChannelConfig = field(default_factory=ChannelConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    antigravity: AntigravityConfig = field(default_factory=AntigravityConfig)
    mode: str = "cli"
    identity_dir: str = ""
    data_dir: str = "/tmp/jarvis_test"
    home_dir: str = "/tmp/jarvis_test"


# ── Tests ────────────────────────────────────────────────────────────

class TestFormatSandboxResult:
    """Test the static _format_sandbox_result method."""

    def _import_tool_executor(self):
        """Import ToolExecutor with mocked dependencies."""
        with patch("src.jarvis.core.JARVISConfig", JARVISConfig):
            from src.jarvis.core import ToolExecutor
            return ToolExecutor

    def test_success_with_stdout(self):
        TE = self._import_tool_executor()
        result = {"status": "SUCCESS", "stdout": "Hello World\n", "stderr": "", "error_message": ""}
        assert TE._format_sandbox_result(result) == "Hello World\n"

    def test_success_empty_output(self):
        TE = self._import_tool_executor()
        result = {"status": "SUCCESS", "stdout": "", "stderr": "", "error_message": ""}
        assert TE._format_sandbox_result(result) == "[Code completed with no output]"

    def test_failed_with_stderr(self):
        TE = self._import_tool_executor()
        result = {"status": "FAILED", "stdout": "", "stderr": "syntax error", "error_message": "Return code 1"}
        out = TE._format_sandbox_result(result)
        assert "[STDERR]" in out
        assert "syntax error" in out
        assert "[Status: FAILED]" in out
        assert "[Error: Return code 1]" in out

    def test_timeout_status(self):
        TE = self._import_tool_executor()
        result = {"status": "TIMEOUT", "stdout": "", "stderr": "", "error_message": "Execution timed out"}
        out = TE._format_sandbox_result(result)
        assert "[Status: TIMEOUT]" in out
        assert "timed out" in out

    def test_truncation(self):
        TE = self._import_tool_executor()
        result = {"status": "SUCCESS", "stdout": "x" * 60000, "stderr": "", "error_message": ""}
        out = TE._format_sandbox_result(result)
        assert "...[truncated]..." in out
        assert len(out) < 60000


class TestToolExecutorSandboxInit:
    """Test sandbox initialization in ToolExecutor."""

    def test_subprocess_sandbox_by_default(self):
        """Without Docker, subprocess sandbox should be created."""
        config = JARVISConfig()
        config.security.docker_sandbox = False
        with patch("src.jarvis.auth.sandbox.SubprocessSandboxManager") as MockSub:
            MockSub.return_value = MagicMock()
            with patch("src.jarvis.auth.sandbox.DockerSandboxManager"):
                from src.jarvis.core import ToolExecutor
                te = ToolExecutor(config)
                assert te._sandbox is not None
                assert te._docker_sandbox is None

    def test_docker_sandbox_when_configured(self):
        """With docker_sandbox=True, Docker sandbox should be created."""
        config = JARVISConfig()
        config.security.docker_sandbox = True
        with patch("src.jarvis.auth.sandbox.SubprocessSandboxManager") as MockSub:
            MockSub.return_value = MagicMock()
            with patch("src.jarvis.auth.sandbox.DockerSandboxManager") as MockDocker:
                MockDocker.return_value = MagicMock()
                from src.jarvis.core import ToolExecutor
                te = ToolExecutor(config)
                assert te._sandbox is not None
                assert te._docker_sandbox is not None

    def test_docker_fallback_on_error(self):
        """If Docker fails to init, sandbox degrades gracefully (docker_client=None)."""
        config = JARVISConfig()
        config.security.docker_sandbox = True
        from src.jarvis.core import ToolExecutor
        te = ToolExecutor(config)
        assert te._sandbox is not None
        # DockerSandboxManager catches init errors internally — it creates
        # an instance with docker_client=None rather than raising.
        if te._docker_sandbox is not None:
            assert te._docker_sandbox.docker_client is None


class TestPythonExecTool:
    """Test python_exec tool with sandbox delegation."""

    @pytest.mark.asyncio
    async def test_delegates_to_subprocess_sandbox(self):
        """python_exec should delegate to the subprocess sandbox."""
        config = JARVISConfig()
        mock_sandbox = AsyncMock()
        mock_sandbox.run = AsyncMock(return_value={
            "status": "SUCCESS", "stdout": "42\n", "stderr": "", "error_message": ""
        })
        with patch("src.jarvis.auth.sandbox.SubprocessSandboxManager", return_value=mock_sandbox):
            with patch("src.jarvis.auth.sandbox.DockerSandboxManager"):
                from src.jarvis.core import ToolExecutor
                te = ToolExecutor(config)
                te._sandbox = mock_sandbox

                result = await te.execute("python_exec", {"code": "print(42)"})
                assert "42" in result
                mock_sandbox.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_timeout(self):
        """python_exec should pass timeout through."""
        config = JARVISConfig()
        mock_sandbox = AsyncMock()
        mock_sandbox.run = AsyncMock(return_value={
            "status": "SUCCESS", "stdout": "", "stderr": "", "error_message": ""
        })
        with patch("src.jarvis.auth.sandbox.SubprocessSandboxManager", return_value=mock_sandbox):
            with patch("src.jarvis.auth.sandbox.DockerSandboxManager"):
                from src.jarvis.core import ToolExecutor
                te = ToolExecutor(config)
                te._sandbox = mock_sandbox

                await te.execute("python_exec", {"code": "pass", "timeout": 60})
                call_kwargs = mock_sandbox.run.call_args
                assert call_kwargs.kwargs.get("timeout") == 60 or call_kwargs[1].get("timeout") == 60


class TestExecuteCodeTool:
    """Test execute_code multi-language tool."""

    @pytest.mark.asyncio
    async def test_requires_docker_for_non_python(self):
        """execute_code should reject non-Python without Docker."""
        config = JARVISConfig()
        with patch("src.jarvis.auth.sandbox.SubprocessSandboxManager", return_value=MagicMock()):
            with patch("src.jarvis.auth.sandbox.DockerSandboxManager"):
                from src.jarvis.core import ToolExecutor
                te = ToolExecutor(config)
                te._docker_sandbox = None

                result = await te.execute("execute_code", {"code": "console.log('hi')", "runtime": "node"})
                assert "requires Docker sandbox" in result

    @pytest.mark.asyncio
    async def test_python_fallback_without_docker(self):
        """execute_code with runtime=python should fall back to python_exec."""
        config = JARVISConfig()
        mock_sandbox = AsyncMock()
        mock_sandbox.run = AsyncMock(return_value={
            "status": "SUCCESS", "stdout": "hi\n", "stderr": "", "error_message": ""
        })
        with patch("src.jarvis.auth.sandbox.SubprocessSandboxManager", return_value=mock_sandbox):
            with patch("src.jarvis.auth.sandbox.DockerSandboxManager"):
                from src.jarvis.core import ToolExecutor
                te = ToolExecutor(config)
                te._sandbox = mock_sandbox
                te._docker_sandbox = None

                result = await te.execute("execute_code", {"code": "print('hi')", "runtime": "python"})
                assert "hi" in result

    @pytest.mark.asyncio
    async def test_delegates_to_docker_sandbox(self):
        """execute_code should use Docker sandbox for multi-language."""
        config = JARVISConfig()
        mock_docker = AsyncMock()
        mock_docker.run_runtime = AsyncMock(return_value={
            "status": "SUCCESS", "stdout": "Hello from Node!\n", "stderr": "", "error_message": ""
        })
        with patch("src.jarvis.auth.sandbox.SubprocessSandboxManager", return_value=MagicMock()):
            with patch("src.jarvis.auth.sandbox.DockerSandboxManager", return_value=mock_docker):
                from src.jarvis.core import ToolExecutor
                te = ToolExecutor(config)
                te._docker_sandbox = mock_docker

                result = await te.execute("execute_code", {
                    "code": "console.log('Hello from Node!')",
                    "runtime": "node",
                    "timeout": 45,
                })
                assert "Hello from Node!" in result
                mock_docker.run_runtime.assert_called_once_with(
                    runtime="node",
                    code="console.log('Hello from Node!')",
                    timeout=45,
                    network_allowed=False,
                )


class TestToolDefinitions:
    """Verify tool definitions are properly registered."""

    def test_execute_code_in_builtin_tools(self):
        from src.jarvis.core import BUILTIN_TOOLS
        names = [t["name"] for t in BUILTIN_TOOLS]
        assert "execute_code" in names
        assert "python_exec" in names

    def test_execute_code_has_runtime_param(self):
        from src.jarvis.core import BUILTIN_TOOLS
        ec = next(t for t in BUILTIN_TOOLS if t["name"] == "execute_code")
        assert "runtime" in ec["parameters"]["properties"]
        assert "runtime" in ec["parameters"]["required"]

    def test_python_exec_has_timeout_param(self):
        from src.jarvis.core import BUILTIN_TOOLS
        pe = next(t for t in BUILTIN_TOOLS if t["name"] == "python_exec")
        assert "timeout" in pe["parameters"]["properties"]
        assert "network_allowed" in pe["parameters"]["properties"]
