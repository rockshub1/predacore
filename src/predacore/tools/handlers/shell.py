"""Shell & code execution handlers: run_command, python_exec, execute_code."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from ._context import (
    ToolContext,
    ToolError,
    ToolErrorKind,
    missing_param,
    invalid_param,
    blocked,
    subsystem_unavailable,
)

logger = logging.getLogger(__name__)

# Patterns that match obviously destructive commands.
# No end-of-line anchors — catches "rm -rf / && echo done" and "rm -rf / | tee log".
DANGEROUS_COMMAND_PATTERNS = [
    re.compile(r"\brm\s+(-\S*)?-r\S*\s+/\s*($|[;&|])", re.IGNORECASE),
    re.compile(r"\brm\s+(-\S*)?-r\S*\s+/\*", re.IGNORECASE),
    re.compile(r"\bdd\s+.*\bof\s*=\s*/dev/[sh]d", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;", re.IGNORECASE),
    re.compile(r"\bchmod\s+(-\S+\s+)*777\s+/\s*($|[;&|])", re.IGNORECASE),
    re.compile(r">\s*/dev/[sh]d", re.IGNORECASE),
    re.compile(r"\bformat\s+[A-Za-z]:", re.IGNORECASE),
]

_MAX_OUTPUT_BYTES = 10 * 1024 * 1024  # 10 MB

# Env vars that must NEVER reach a shell subprocess. If the agent's env holds
# credentials, they must stay in the agent process — a sub-shell may echo them
# via `env`, log them in shell history, or pass them to a command that uploads
# (curl, scp, rsync). Filter at the boundary; safer than trusting the agent not
# to write commands that would read them.
_SENSITIVE_ENV_PREFIXES = (
    "ANTHROPIC_", "OPENAI_", "GOOGLE_", "GEMINI_", "GROQ_", "MISTRAL_",
    "COHERE_", "TOGETHER_", "HF_", "HUGGINGFACE_", "REPLICATE_",
    "AWS_", "AZURE_", "GCP_", "GOOGLE_APPLICATION_CREDENTIALS",
    "GITHUB_TOKEN", "GITLAB_TOKEN", "NPM_TOKEN", "PYPI_TOKEN",
    "DATABASE_URL", "REDIS_URL", "MONGODB_URI",
    "SLACK_", "DISCORD_",
    "CLAUDE_", "PREDACORE_",
)


def _sanitized_env() -> dict[str, str]:
    """Return parent env stripped of credentials before handing to a subshell."""
    return {
        k: v for k, v in os.environ.items()
        if not any(k.upper().startswith(p) or k.upper() == p.rstrip("_")
                   for p in _SENSITIVE_ENV_PREFIXES)
        and "TOKEN" not in k.upper()
        and "SECRET" not in k.upper()
        and "API_KEY" not in k.upper()
        and "PASSWORD" not in k.upper()
    }


async def handle_run_command(args: dict[str, Any], ctx: ToolContext) -> str:
    """Execute a shell command and return stdout/stderr.
    
    Validates against dangerous command patterns, respects trust-level
    timeout policies, and truncates output at 50KB. In YOLO mode,
    timeout can be disabled by passing ``timeout_seconds <= 0``.
    
    Args:
        args: ``{"command": str, "cwd": str, "timeout_seconds": int}``
        ctx: Tool context with security config.
    
    Raises:
        ToolError(MISSING_PARAM): if command is empty.
        ToolError(BLOCKED): if command matches a dangerous pattern.
        ToolError(TIMEOUT): if command exceeds timeout.
    """
    command = args.get("command", "")
    if not command:
        raise missing_param("command", tool="run_command")
    logger.info("run_command: %s", command[:200])
    cwd = args.get("cwd", None)
    timeout_raw = args.get("timeout_seconds", None)

    for pattern in DANGEROUS_COMMAND_PATTERNS:
        if pattern.search(command):
            raise blocked(
                f"command matched dangerous pattern: {pattern.pattern}",
                tool="run_command",
            )

    default_timeout = int(ctx.config.security.task_timeout_seconds or 300)
    if ctx.config.security.trust_level == "yolo":
        default_timeout = max(default_timeout, 600)

    timeout_seconds: float | None
    if timeout_raw is None:
        timeout_seconds = float(default_timeout)
    else:
        try:
            timeout_seconds = float(timeout_raw)
        except (TypeError, ValueError):
            timeout_seconds = float(default_timeout)

    if timeout_seconds is not None and timeout_seconds <= 0:
        if ctx.config.security.trust_level == "yolo":
            timeout_seconds = None
        else:
            timeout_seconds = float(default_timeout)

    if timeout_seconds is not None:
        timeout_seconds = min(timeout_seconds, 24 * 60 * 60)

    shell_executable = os.getenv("SHELL", "")
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            executable=shell_executable or None,
            env=_sanitized_env(),
        )
        if timeout_seconds is None:
            stdout, stderr = await proc.communicate()
        else:
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
                raise ToolError(
                    f"Command timed out after {int(timeout_seconds)} seconds",
                    kind=ToolErrorKind.TIMEOUT,
                    tool_name="run_command",
                )

        if stdout and len(stdout) > _MAX_OUTPUT_BYTES:
            stdout = stdout[:_MAX_OUTPUT_BYTES] + b"\n[output truncated at byte limit]"
        if stderr and len(stderr) > _MAX_OUTPUT_BYTES:
            stderr = stderr[:_MAX_OUTPUT_BYTES] + b"\n[output truncated at byte limit]"

        result = ""
        if stdout:
            result += stdout.decode(errors="replace")
        if stderr:
            result += "\n[STDERR]\n" + stderr.decode(errors="replace")
        if proc.returncode != 0:
            result += f"\n[Exit code: {proc.returncode}]"
        if len(result) > 50000:
            result = result[:25000] + "\n...[truncated]...\n" + result[-5000:]
        return result or "[Command completed with no output]"
    except ToolError:
        raise
    except asyncio.TimeoutError:
        raise ToolError(
            f"Command timed out after {int(timeout_seconds or default_timeout)} seconds",
            kind=ToolErrorKind.TIMEOUT,
            tool_name="run_command",
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise ToolError(
            f"Command error: {e}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="run_command",
        ) from e


_ALLOWED_RUNTIMES = {
    "python", "node", "go", "ruby", "php", "r", "julia",
    "rust", "java", "kotlin", "c", "cpp", "typescript", "bash",
}


async def handle_python_exec(args: dict[str, Any], ctx: ToolContext) -> str:
    """Execute Python code via the sandbox manager (Docker or subprocess)."""
    code = args.get("code", "")
    if not code:
        raise missing_param("code", tool="python_exec")
    timeout = args.get("timeout", 30)
    network_allowed = args.get("network_allowed", False)
    timeout = min(timeout, 300)

    sandbox = ctx.docker_sandbox or ctx.sandbox
    if sandbox:
        result = await sandbox.run(
            code=code, input_args=None,
            timeout=timeout, network_allowed=network_allowed,
        )
        if ctx.format_sandbox_result:
            return ctx.format_sandbox_result(result)
        return str(result)[:50000]

    _BLOCKED_PATTERNS = [
        "import subprocess", "import os", "os.system", "os.popen",
        "os.exec", "__import__", "eval(", "exec(", "compile(",
        "import socket", "import shutil", "shutil.rmtree",
        "open('/etc", "open('/var", "open('/sys",
        "importlib", "getattr(__builtins__", "exec(open",
        "__subclasses__", "__globals__", "__builtins__",
        # Obfuscation vectors
        "getattr(", "from subprocess", "from os import",
        "from shutil import", "from socket import",
        "breakpoint(", "code.interact", "pty.spawn",
    ]
    code_lower = code.lower()
    for pattern in _BLOCKED_PATTERNS:
        if pattern.lower() in code_lower:
            raise blocked(
                f"code contains restricted pattern '{pattern}'. "
                "Install Docker for unrestricted sandboxed execution.",
                tool="python_exec",
            )

    import tempfile

    _MEM_LIMIT_BYTES = 1024 * 1024 * 1024     # 1 GB (was 256 MB)
    _MAX_STDOUT_BYTES = 1_000_000              # 1 MB (was 50 KB)
    _MAX_STDERR_BYTES = 200_000                # 200 KB (was 10 KB)

    wrapper_prefix = (
        "import resource, sys\n"
        "try:\n"
        f"    resource.setrlimit(resource.RLIMIT_AS, ({_MEM_LIMIT_BYTES}, {_MEM_LIMIT_BYTES}))\n"
        "except (ValueError, AttributeError, OSError):\n"
        "    pass\n"
        "# --- user code below ---\n"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper_prefix)
        f.write(code)
        f.flush()
        tmp_path = f.name
    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            raise ToolError(
                f"Python execution timed out after {timeout} seconds",
                kind=ToolErrorKind.TIMEOUT,
                tool_name="python_exec",
            )
        result = ""
        if stdout:
            out_text = stdout.decode(errors="replace")
            if len(out_text) > _MAX_STDOUT_BYTES:
                out_text = out_text[:_MAX_STDOUT_BYTES] + f"\n... [output truncated at {_MAX_STDOUT_BYTES} bytes]"
            result += out_text
        if stderr:
            err_text = stderr.decode(errors="replace")
            if len(err_text) > _MAX_STDERR_BYTES:
                err_text = err_text[:_MAX_STDERR_BYTES] + f"\n... [stderr truncated at {_MAX_STDERR_BYTES} bytes]"
            result += "\n[STDERR]\n" + err_text
        return result or "[Script completed with no output]"
    except ToolError:
        raise
    except (OSError, subprocess.SubprocessError) as e:
        raise ToolError(
            f"Python execution error: {e}",
            kind=ToolErrorKind.EXECUTION,
            tool_name="python_exec",
        ) from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def handle_execute_code(args: dict[str, Any], ctx: ToolContext) -> str:
    """Execute code in any supported language via Docker sandbox."""
    code = args.get("code", "")
    if not code:
        raise missing_param("code", tool="execute_code")
    runtime = str(args.get("runtime", "python")).strip().lower()
    if runtime not in _ALLOWED_RUNTIMES:
        raise invalid_param(
            "runtime",
            f"unsupported runtime: {runtime}. Allowed: {', '.join(sorted(_ALLOWED_RUNTIMES))}",
            tool="execute_code",
        )
    timeout = args.get("timeout", 30)
    network_allowed = args.get("network_allowed", False)

    if not ctx.docker_sandbox:
        if runtime == "python":
            return await handle_python_exec(
                {"code": code, "timeout": timeout, "network_allowed": network_allowed},
                ctx,
            )
        raise ToolError(
            f"execute_code requires Docker sandbox for '{runtime}' runtime",
            kind=ToolErrorKind.UNAVAILABLE,
            tool_name="execute_code",
            suggestion="Set docker_sandbox=true in config or use python_exec for Python",
        )

    result = await ctx.docker_sandbox.run_runtime(
        runtime=runtime, code=code,
        timeout=timeout, network_allowed=network_allowed,
    )
    if ctx.format_sandbox_result:
        return ctx.format_sandbox_result(result)
    return str(result)[:50000]
