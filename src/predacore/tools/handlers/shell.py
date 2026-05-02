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
    blocked,
    invalid_param,
    missing_param,
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


# ──────────────────────────────────────────────────────────────────────────
# D13 — Memory auto-sync after git mutations
#
# When the agent runs `git checkout / merge / rebase / reset / pull /
# switch / cherry-pick / revert`, the working tree may change in ways
# that `git status` alone can't reveal — a clean `git pull` that fast-
# forwards 50 commits leaves a clean working tree, so post-hoc status
# queries miss the actual file changes. Capture HEAD BEFORE the command
# runs; pass the captured SHA to UnifiedMemoryStore.sync_git_changes()
# AFTER success so it can diff old..new HEAD as well as scan working-
# tree dirt. Without this, sync would only catch uncommitted changes
# (matching the lab hook's blind spot — fixed here in B5).
# ──────────────────────────────────────────────────────────────────────────

_GIT_MUTATION_RE = re.compile(
    r"\bgit\s+(?:checkout|switch|merge|rebase|reset|pull|cherry-pick|revert)\b",
    re.IGNORECASE,
)


async def _capture_git_head(cwd: str | None) -> str | None:
    """Return the current HEAD SHA via `git rev-parse HEAD` in ``cwd``.

    Bounded to a 2-second budget so a slow git command can't delay the
    user's actual command. Returns None on failure (not a git repo, git
    binary missing, timeout, etc.) — caller treats None as "no prior
    HEAD captured" and sync still runs in working-tree-only mode.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=cwd,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
    except (FileNotFoundError, asyncio.TimeoutError, OSError):
        return None
    if proc.returncode != 0:
        return None
    sha = out.decode(errors="replace").strip()
    return sha or None


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

    # D13 — capture HEAD BEFORE git mutations so the post-command sync
    # can diff old..new HEAD (catches committed changes from `git pull` /
    # `git checkout other_branch` that working-tree status can't see).
    git_mutation = bool(_GIT_MUTATION_RE.search(command))
    prior_head: str | None = None
    if git_mutation and ctx.unified_memory:
        prior_head = await _capture_git_head(cwd)

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
            except asyncio.TimeoutError as exc:
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
                raise ToolError(
                    f"Command timed out after {int(timeout_seconds)} seconds",
                    kind=ToolErrorKind.TIMEOUT,
                    tool_name="run_command",
                ) from exc

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

        # D13 — after a SUCCESSFUL git mutation, sync memory with whatever
        # changed. Pass prior_head so sync_git_changes() can run a real
        # `git diff prior..HEAD --name-status` (catches committed changes,
        # not just uncommitted dirt). Failures are non-fatal — memory issues
        # must NOT alter what the agent sees as the command result.
        if git_mutation and proc.returncode == 0 and ctx.unified_memory:
            try:
                # L5 — tag synced chunks with the current project_id so
                # cross-repo memory stays partitioned. Resolve from the
                # command's cwd (the actual repo root the user is in).
                try:
                    from predacore.memory.project_id import default_project
                    proj = default_project(cwd=cwd or ".")
                except ImportError:
                    proj = "default"
                await ctx.unified_memory.sync_git_changes(
                    cwd or ".",
                    prior_head=prior_head,
                    project_id=proj,
                )
            except Exception as exc:  # broad: memory must not break shell
                logger.debug(
                    "Memory sync_git_changes after %r failed: %s",
                    command[:80], exc,
                )

        if len(result) > 50000:
            result = result[:25000] + "\n...[truncated]...\n" + result[-5000:]
        return result or "[Command completed with no output]"
    except ToolError:
        raise
    except asyncio.TimeoutError as exc:
        raise ToolError(
            f"Command timed out after {int(timeout_seconds or default_timeout)} seconds",
            kind=ToolErrorKind.TIMEOUT,
            tool_name="run_command",
        ) from exc
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

    # Trust-tiered resource caps. yolo keeps the wide 1 GB / 1 MB ceilings
    # for genuine large data work; ask_everytime drops to 256 MB / 50 KB so
    # a runaway python_exec can't OOM the daemon or flood the agent's
    # context with megabytes of stdout. A single shared cap = global blast
    # radius if any one agent leaks memory.
    _trust = getattr(ctx.config.security, "trust_level", "ask_everytime")
    if _trust == "yolo":
        _MEM_LIMIT_BYTES = 1024 * 1024 * 1024     # 1 GB
        _MAX_STDOUT_BYTES = 1_000_000              # 1 MB
        _MAX_STDERR_BYTES = 200_000                # 200 KB
    else:
        _MEM_LIMIT_BYTES = 256 * 1024 * 1024       # 256 MB
        _MAX_STDOUT_BYTES = 50_000                 # 50 KB
        _MAX_STDERR_BYTES = 10_000                 # 10 KB

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
        except asyncio.TimeoutError as exc:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            raise ToolError(
                f"Python execution timed out after {timeout} seconds",
                kind=ToolErrorKind.TIMEOUT,
                tool_name="python_exec",
            ) from exc
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
