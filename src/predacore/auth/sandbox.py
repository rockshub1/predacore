"""
Sandbox Isolation — subprocess, Docker, and per-session container sandboxes.

Provides three levels of code execution isolation:
  - SubprocessSandboxManager: Local subprocess (dev/testing only)
  - DockerSandboxManager: Full Docker container isolation with multi-runtime support
  - SessionSandboxPool: Per-session ephemeral containers with volume persistence

Usage:
    # Simple execution
    sandbox = SubprocessSandboxManager()
    result = await sandbox.run(code, input_args=None, timeout=30)

    # Docker execution
    docker_sb = DockerSandboxManager()
    result = await docker_sb.run(code, input_args=None, timeout=30)
    result = await docker_sb.run_runtime("node", js_code, timeout=30)

    # Per-session pool
    pool = SessionSandboxPool()
    session_sb = await pool.acquire("session-123")
    result = await session_sb.execute("print('Hello')")
    await pool.release("session-123")
"""
from __future__ import annotations

import asyncio
import logging
import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import docker
    from docker.errors import DockerException
except ImportError:  # pragma: no cover - optional dependency
    docker = None  # type: ignore
    DockerException = Exception  # type: ignore

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Runtime Configuration — Single source of truth for all sandbox types
# ══════════════════════════════════════════════════════════════════════

_RUNTIME_IMAGES: dict[str, str] = {
    "python": "python:3.11-slim",
    "node": "node:20-alpine",
    "typescript": "node:20-alpine",
    "ruby": "ruby:3.2-alpine",
    "go": "golang:1.22-alpine",
    "rust": "rust:latest",
    "java": "eclipse-temurin:21-jdk",
    "kotlin": "gradle:jdk17",
    "c": "gcc:latest",
    "cpp": "gcc:latest",
    "php": "php:8.2-cli-alpine",
    "r": "r-base:latest",
    "julia": "julia:latest",
    "bash": "bash:5",
}

_RUNTIME_EXTENSIONS: dict[str, str] = {
    "python": ".py", "node": ".js", "typescript": ".ts", "ruby": ".rb",
    "go": ".go", "rust": ".rs", "java": ".java", "kotlin": ".kt",
    "c": ".c", "cpp": ".cpp", "php": ".php", "r": ".R", "julia": ".jl",
    "bash": ".sh",
}

_RUNTIME_FILENAMES: dict[str, str] = {
    "python": "sandbox_code.py", "node": "script.js", "go": "main.go",
    "ruby": "script.rb", "php": "script.php", "r": "script.R",
    "julia": "script.jl", "rust": "main.rs", "java": "Main.java",
    "kotlin": "main.kt", "c": "main.c", "cpp": "main.cpp",
    "typescript": "script.ts", "bash": "script.sh",
}

_COMPILED_LANGS: frozenset[str] = frozenset({"rust", "java", "kotlin", "c", "cpp", "go"})

_RUNTIME_ALIASES: dict[str, str] = {"javascript": "node", "shell": "bash"}


# ══════════════════════════════════════════════════════════════════════
# Base Sandbox Managers
# ══════════════════════════════════════════════════════════════════════


class AbstractSandboxManager:
    """Base interface for sandbox managers."""

    async def run(
        self,
        code: str,
        input_args: list[Any] | dict[str, Any] | None,
        timeout: int,
    ) -> dict:
        raise NotImplementedError


class SubprocessSandboxManager(AbstractSandboxManager):
    """
    Minimal subprocess-based sandbox.
    NOTE: This is not hardened; intended for local/dev use only.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        if os.environ.get("PREDACORE_ENV") == "production":
            self.logger.warning(
                "SubprocessSandboxManager is NOT hardened — use "
                "DockerSandboxManager for production deployments"
            )

    async def run(
        self,
        code: str,
        input_args: list[Any] | dict[str, Any] | None,
        timeout: int,
        *,
        network_allowed: bool = False,
    ) -> dict:
        temp_dir = tempfile.mkdtemp()
        code_path = os.path.join(temp_dir, "sandbox_code.py")
        try:
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)
            # Minimal env — do NOT inherit secrets from parent process
            _SAFE_ENV_KEYS = {"PATH", "HOME", "USER", "LANG", "TERM", "TMPDIR", "SHELL"}
            env = {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}
            env["PYTHONPATH"] = ""
            if not network_allowed:
                env["NO_NETWORK"] = "1"

            def _set_resource_limits() -> None:
                """Apply resource limits to the sandboxed subprocess (best-effort)."""
                try:
                    # Max CPU time: match the user-requested timeout
                    resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 5, int(timeout) + 10))
                except (ValueError, OSError):
                    pass
                try:
                    # Max file size: 50 MB
                    resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 1024 * 1024, 50 * 1024 * 1024))
                except (ValueError, OSError):
                    pass
                try:
                    # Max open files: 128
                    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
                except (ValueError, OSError):
                    pass

            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                code_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=temp_dir,
                env=env,
                preexec_fn=_set_resource_limits,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return {
                    "status": "TIMEOUT",
                    "stdout": "",
                    "stderr": "",
                    "result": None,
                    "error_message": "Execution timed out",
                }
            status = "SUCCESS" if proc.returncode == 0 else "FAILED"
            return {
                "status": status,
                "stdout": stdout.decode("utf-8"),
                "stderr": stderr.decode("utf-8"),
                "result": None,
                "error_message": ""
                if status == "SUCCESS"
                else f"Return code {proc.returncode}",
            }
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            self.logger.error("Subprocess sandbox error: %s", e, exc_info=True)
            return {
                "status": "ERROR",
                "stdout": "",
                "stderr": "",
                "result": None,
                "error_message": str(e),
            }
        finally:
            try:
                for name in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, name))
                    except OSError:
                        pass
                os.rmdir(temp_dir)
            except OSError:
                pass


class DockerSandboxManager(AbstractSandboxManager):
    """Secure sandbox using Docker containers for code execution."""

    # Commands for /sandbox/ mount strategy (used by run_runtime)
    _CMD_MAP = {
        "python": ["python", "/sandbox/sandbox_code.py"],
        "node": ["node", "/sandbox/script.js"],
        "go": [
            "sh", "-c",
            "mkdir -p /sandbox/.tmp /sandbox/.gocache && "
            "GOCACHE=/sandbox/.gocache GOTMPDIR=/sandbox/.tmp GO111MODULE=off go run /sandbox/main.go",
        ],
        "ruby": ["ruby", "/sandbox/script.rb"],
        "php": ["php", "/sandbox/script.php"],
        "r": ["Rscript", "/sandbox/script.R"],
        "julia": ["julia", "/sandbox/script.jl"],
        "rust": ["sh", "-c", "cd /sandbox && rustc main.rs -o main && ./main"],
        "java": ["sh", "-c", "cd /sandbox && javac Main.java && java Main"],
        "kotlin": ["sh", "-c", "cd /sandbox && kotlinc main.kt -include-runtime -d main.jar && java -jar main.jar"],
        "c": ["sh", "-c", "cd /sandbox && gcc main.c -o main && ./main"],
        "cpp": ["sh", "-c", "cd /sandbox && g++ main.cpp -o main && ./main"],
        "typescript": ["sh", "-c", "cd /sandbox && npx --yes tsx script.ts"],
    }

    def __init__(
        self,
        logger: logging.Logger | None = None,
        image: str = "python:3.10-slim",
        cpu_limit: float = 0.5,
        mem_limit: str = "256m",
        pids_limit: int = 256,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.image = image
        self.cpu_limit = cpu_limit
        self.mem_limit = mem_limit
        self.pids_limit = pids_limit
        self.docker_client = None

        if docker is None:
            self.logger.warning("python 'docker' package not installed.")
            return

        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            self.logger.info("DockerSandboxManager initialized (image=%s)", self.image)
        except (DockerException, OSError, ConnectionError) as e:
            self.docker_client = None
            self.logger.warning("Docker client init failed: %s", e)

    def _docker_common_args(self, network_allowed: bool) -> dict[str, Any]:
        return {
            "network_mode": ("bridge" if network_allowed else "none"),
            "detach": True,
            "cpu_period": 100000,
            "cpu_quota": int(self.cpu_limit * 100000),
            "mem_limit": self.mem_limit,
            "stdout": True,
            "stderr": True,
            "remove": False,
            "pids_limit": self.pids_limit,
            "read_only": True,
            "security_opt": ["no-new-privileges"],
            "cap_drop": ["ALL"],
            "tmpfs": {"/tmp": "rw,noexec,nosuid,nodev"},
        }

    async def run(
        self,
        code: str,
        input_args: list[Any] | dict[str, Any] | None,
        timeout: int,
        *,
        network_allowed: bool = False,
    ) -> dict:
        if self.docker_client is None:
            return {
                "status": "ERROR", "stdout": "", "stderr": "", "result": None,
                "error_message": "Docker client unavailable.",
            }

        temp_dir = None
        container = None
        try:
            temp_dir = tempfile.mkdtemp()
            code_path = os.path.join(temp_dir, "sandbox_code.py")
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)
            volumes = {temp_dir: {"bind": "/sandbox", "mode": "ro"}}
            command = ["python", "/sandbox/sandbox_code.py"]
            args = self._docker_common_args(network_allowed)
            container = self.docker_client.containers.run(
                self.image, command, volumes=volumes, **args,
            )
            try:
                exit_status = container.wait(timeout=timeout)
                stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8")
            except (DockerException, ConnectionError, OSError) as e:
                self.logger.warning("Docker execution timed out: %s", e)
                container.kill()
                return {
                    "status": "TIMEOUT", "stdout": "", "stderr": "", "result": None,
                    "error_message": f"Execution timed out or failed: {e}",
                }
            if exit_status.get("StatusCode") == 0:
                return {
                    "status": "SUCCESS", "stdout": stdout, "stderr": stderr,
                    "result": None, "error_message": "",
                }
            return {
                "status": "FAILED", "stdout": stdout, "stderr": stderr, "result": None,
                "error_message": f"Return code {exit_status.get('StatusCode')}",
            }
        except DockerException as e:
            self.logger.error("Docker error: %s", e, exc_info=True)
            return {"status": "ERROR", "stdout": "", "stderr": "", "result": None, "error_message": f"Docker error: {e}"}
        except (OSError, ConnectionError) as e:
            self.logger.error("Unexpected sandbox error: %s", e, exc_info=True)
            return {"status": "ERROR", "stdout": "", "stderr": "", "result": None, "error_message": str(e)}
        finally:
            if temp_dir and os.path.exists(temp_dir):
                try:
                    for name in os.listdir(temp_dir):
                        try:
                            os.remove(os.path.join(temp_dir, name))
                        except OSError:
                            pass
                    os.rmdir(temp_dir)
                except OSError:
                    pass
            if container:
                try:
                    container.remove(force=True)
                except (DockerException, OSError):
                    pass

    async def run_runtime(
        self, runtime: str, code: str, timeout: int, *, network_allowed: bool = False
    ) -> dict:
        """Execute code in a language-specific container."""
        if self.docker_client is None:
            return {
                "status": "ERROR", "stdout": "", "stderr": "", "result": None,
                "error_message": "Docker client unavailable.",
            }

        runtime = _RUNTIME_ALIASES.get((runtime or "python").lower(), (runtime or "python").lower())
        image = _RUNTIME_IMAGES.get(runtime)
        filename = _RUNTIME_FILENAMES.get(runtime)
        command = self._CMD_MAP.get(runtime)
        if not image or not filename or not command:
            return {
                "status": "ERROR", "stdout": "", "stderr": "", "result": None,
                "error_message": f"Unsupported runtime: {runtime}",
            }

        temp_dir = None
        container = None
        try:
            temp_dir = tempfile.mkdtemp()
            code_path = os.path.join(temp_dir, filename)
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)
            mode = "rw" if runtime in _COMPILED_LANGS else "ro"
            volumes = {temp_dir: {"bind": "/sandbox", "mode": mode}}
            args = self._docker_common_args(network_allowed)
            if runtime in _COMPILED_LANGS:
                args["tmpfs"] = {"/tmp": "rw,nosuid,nodev"}
            container = self.docker_client.containers.run(
                image, command, volumes=volumes, **args,
            )
            try:
                exit_status = container.wait(timeout=timeout)
                stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8")
            except (DockerException, ConnectionError, OSError) as e:
                self.logger.warning("%s execution timed out: %s", runtime, e)
                container.kill()
                return {
                    "status": "TIMEOUT", "stdout": "", "stderr": "", "result": None,
                    "error_message": f"Execution timed out: {e}",
                }
            if exit_status.get("StatusCode") == 0:
                return {
                    "status": "SUCCESS", "stdout": stdout, "stderr": stderr,
                    "result": None, "error_message": "",
                }
            return {
                "status": "FAILED", "stdout": stdout, "stderr": stderr, "result": None,
                "error_message": f"Return code {exit_status.get('StatusCode')}",
            }
        except DockerException as e:
            self.logger.error("Docker error (%s): %s", runtime, e)
            return {"status": "ERROR", "stdout": "", "stderr": "", "result": None, "error_message": f"Docker error: {e}"}
        except (OSError, ConnectionError) as e:
            self.logger.error("Unexpected %s sandbox error: %s", runtime, e)
            return {"status": "ERROR", "stdout": "", "stderr": "", "result": None, "error_message": str(e)}
        finally:
            if temp_dir and os.path.exists(temp_dir):
                try:
                    for name in os.listdir(temp_dir):
                        try:
                            os.remove(os.path.join(temp_dir, name))
                        except OSError:
                            pass
                    os.rmdir(temp_dir)
                except OSError:
                    pass
            if container:
                try:
                    container.remove(force=True)
                except (DockerException, OSError):
                    pass


# ══════════════════════════════════════════════════════════════════════
# Per-Session Sandbox (Docker containers with volume persistence)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class SandboxConfig:
    """Configuration for a per-session sandbox."""

    image: str = "python:3.11-slim"
    cpu_limit: float = 0.5
    mem_limit: str = "256m"
    pids_limit: int = 256
    timeout: int = 30
    network_allowed: bool = False
    max_output_bytes: int = 65536  # 64KB


@dataclass
class SessionSandbox:
    """
    Represents a sandbox associated with a user session.

    This is a logical wrapper — each call to execute() runs in a
    fresh container for security, but sessions can share a volume
    for stateful code execution (e.g., variables, files).
    """

    session_id: str
    config: SandboxConfig
    created_at: float = field(default_factory=time.time)
    execution_count: int = 0
    total_cpu_time: float = 0.0
    last_used_at: float = 0.0
    _volume_name: str | None = None

    @property
    def volume_name(self) -> str:
        """Get or create the session's volume name."""
        if not self._volume_name:
            import re as _re
            safe_id = _re.sub(r'[^a-zA-Z0-9_-]', '_', self.session_id)[:32]
            self._volume_name = f"predacore-sandbox-{safe_id}"
        return self._volume_name

    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout: int | None = None,
        network_allowed: bool | None = None,
    ) -> dict[str, Any]:
        """
        Execute code in an isolated Docker container.

        Args:
            code: Source code to execute.
            language: Programming language (python, node, go, etc.).
            timeout: Execution timeout in seconds.
            network_allowed: Override network isolation.

        Returns:
            Dict with 'stdout', 'stderr', 'exit_code', and 'execution_time'.
        """
        import os
        import subprocess
        import tempfile

        effective_timeout = timeout or self.config.timeout
        effective_network = (
            network_allowed
            if network_allowed is not None
            else self.config.network_allowed
        )

        # Language-specific configuration
        lang_config = _get_language_config(language)
        if not lang_config:
            return {
                "stdout": "",
                "stderr": f"Unsupported language: {language}",
                "exit_code": 1,
                "execution_time": 0.0,
            }

        # Write code to a temp file
        suffix = lang_config["extension"]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            code_path = f.name

        try:
            # Build docker run command
            cmd = ["docker", "run", "--rm"]

            # Resource limits
            cmd.extend(
                [
                    "--cpus",
                    str(self.config.cpu_limit),
                    "--memory",
                    self.config.mem_limit,
                    "--pids-limit",
                    str(self.config.pids_limit),
                ]
            )

            # Network isolation
            if not effective_network:
                cmd.extend(["--network", "none"])

            # Security: drop all capabilities, read-only root
            cmd.extend(
                [
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--read-only",
                    "--tmpfs",
                    "/tmp:rw,noexec,nosuid,size=64m",
                ]
            )

            # Mount code file
            container_code_path = f"/tmp/code{suffix}"
            cmd.extend(["-v", f"{code_path}:{container_code_path}:ro"])

            # Session volume for persistent state (optional)
            cmd.extend(["-v", f"{self.volume_name}:/workspace"])

            # Image and execution command
            cmd.append(lang_config["image"])
            cmd.extend(lang_config["cmd"](container_code_path))

            # Execute
            start_time = time.monotonic()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            elapsed = time.monotonic() - start_time

            self.execution_count += 1
            self.total_cpu_time += elapsed
            self.last_used_at = time.time()

            stdout = result.stdout[: self.config.max_output_bytes]
            stderr = result.stderr[: self.config.max_output_bytes]

            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": result.returncode,
                "execution_time": round(elapsed, 3),
            }

        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Execution timed out after {effective_timeout}s",
                "exit_code": -1,
                "execution_time": float(effective_timeout),
            }
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            return {
                "stdout": "",
                "stderr": f"Sandbox error: {e}",
                "exit_code": -1,
                "execution_time": 0.0,
            }
        finally:
            os.unlink(code_path)

    async def cleanup(self) -> None:
        """Remove the session volume."""
        import subprocess

        try:
            subprocess.run(
                ["docker", "volume", "rm", "-f", self.volume_name],
                capture_output=True,
                timeout=10,
            )
            logger.debug("Cleaned up sandbox volume: %s", self.volume_name)
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning("Failed to cleanup volume %s: %s", self.volume_name, e)

    def get_stats(self) -> dict[str, Any]:
        """Get sandbox usage statistics."""
        return {
            "session_id": self.session_id,
            "execution_count": self.execution_count,
            "total_cpu_time": round(self.total_cpu_time, 3),
            "uptime": round(time.time() - self.created_at, 1),
            "last_used_at": self.last_used_at,
        }


class SessionSandboxPool:
    """
    Pool manager for per-session sandboxes.

    Handles creation, reuse, and cleanup of session sandboxes.
    Implements idle timeout to automatically clean up unused sandboxes.
    """

    def __init__(
        self,
        default_config: SandboxConfig | None = None,
        idle_timeout: float = 3600.0,  # 1 hour
        max_sessions: int = 50,
    ):
        self._config = default_config or SandboxConfig()
        self._sandboxes: dict[str, SessionSandbox] = {}
        self._idle_timeout = idle_timeout
        self._max_sessions = max_sessions
        self._cleanup_task: asyncio.Task | None = None

    async def acquire(
        self,
        session_id: str,
        config: SandboxConfig | None = None,
    ) -> SessionSandbox:
        """
        Get or create a sandbox for a session.

        Returns an existing sandbox if one exists for the session,
        otherwise creates a new one.
        """
        if session_id in self._sandboxes:
            return self._sandboxes[session_id]

        # Enforce max sessions
        if len(self._sandboxes) >= self._max_sessions:
            await self._evict_oldest()

        sandbox = SessionSandbox(
            session_id=session_id,
            config=config or self._config,
        )
        self._sandboxes[session_id] = sandbox
        logger.info(
            "Created sandbox for session %s (%d active)",
            session_id,
            len(self._sandboxes),
        )
        return sandbox

    async def release(self, session_id: str) -> None:
        """Release and cleanup a session sandbox."""
        sandbox = self._sandboxes.pop(session_id, None)
        if sandbox:
            await sandbox.cleanup()
            logger.info("Released sandbox for session %s", session_id)

    async def start_cleanup_loop(self) -> None:
        """Start the background cleanup loop for idle sandboxes."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup_loop(self) -> None:
        """Stop the cleanup loop and release all sandboxes."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        # Cleanup all active sandboxes
        for session_id in list(self._sandboxes.keys()):
            await self.release(session_id)

    async def _cleanup_loop(self) -> None:
        """Periodically clean up idle sandboxes."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                now = time.time()
                expired = [
                    sid
                    for sid, sb in self._sandboxes.items()
                    if (
                        now - sb.last_used_at > self._idle_timeout
                        and sb.last_used_at > 0
                    )
                    or (
                        now - sb.created_at > self._idle_timeout
                        and sb.last_used_at == 0
                    )
                ]
                for sid in expired:
                    await self.release(sid)
                    logger.info("Evicted idle sandbox: %s", sid)
            except asyncio.CancelledError:
                break
            except (OSError, DockerException) as e:
                logger.error("Sandbox cleanup error: %s", e)

    async def _evict_oldest(self) -> None:
        """Evict the oldest sandbox to make room for a new one."""
        if not self._sandboxes:
            return
        oldest = min(self._sandboxes.items(), key=lambda x: x[1].created_at)
        await self.release(oldest[0])

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "active_sessions": len(self._sandboxes),
            "max_sessions": self._max_sessions,
            "idle_timeout": self._idle_timeout,
            "sandboxes": {sid: sb.get_stats() for sid, sb in self._sandboxes.items()},
        }


def _get_language_config(language: str) -> dict[str, Any] | None:
    """Get Docker image and command for a language runtime.

    Uses module-level _RUNTIME_IMAGES / _RUNTIME_EXTENSIONS as single
    source of truth (shared with DockerSandboxManager).
    """
    lang = _RUNTIME_ALIASES.get(language.lower(), language.lower())
    image = _RUNTIME_IMAGES.get(lang)
    ext = _RUNTIME_EXTENSIONS.get(lang)
    if not image or not ext:
        return None

    # Command builders for /tmp/code mount strategy (used by SessionSandbox)
    cmd_builders: dict[str, Any] = {
        "python": lambda p: ["python", p],
        "node": lambda p: ["node", p],
        "typescript": lambda p: ["npx", "tsx", p],
        "ruby": lambda p: ["ruby", p],
        "go": lambda p: ["go", "run", p],
        "rust": lambda p: ["sh", "-c", f'rustc "{p}" -o /workspace/out && /workspace/out'],
        "java": lambda p: ["sh", "-c", f'cd /tmp && javac code.java && java Main'],
        "kotlin": lambda p: ["sh", "-c", f'cd /tmp && kotlinc code.kt -include-runtime -d main.jar && java -jar main.jar'],
        "c": lambda p: ["sh", "-c", f'gcc "{p}" -o /workspace/out && /workspace/out'],
        "cpp": lambda p: ["sh", "-c", f'g++ "{p}" -o /workspace/out && /workspace/out'],
        "php": lambda p: ["php", p],
        "r": lambda p: ["Rscript", p],
        "julia": lambda p: ["julia", p],
        "bash": lambda p: ["bash", p],
    }
    cmd_fn = cmd_builders.get(lang)
    if not cmd_fn:
        return None

    return {"image": image, "extension": ext, "cmd": cmd_fn}
