"""
Codex CLI Proxy — OpenAI-compatible API wrapper around `codex exec`.

This service exposes /v1/chat/completions and /v1/models so Jarvis (or any
OpenAI-compatible client) can talk to Codex CLI as if it were an API.
"""
from __future__ import annotations

import asyncio
import argparse
import hmac
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request

try:
    from jarvis.llm_providers.router import LLMInterface
except ImportError:  # pragma: no cover - test/local import fallback
    from src.jarvis.llm_providers.router import LLMInterface

app = FastAPI()
logger = logging.getLogger(__name__)

PROXY_API_KEY = os.getenv("CODEX_PROXY_API_KEY", "")
DEFAULT_CODEX_MODEL = os.getenv("CODEX_MODEL", "gpt-5.4")
REPO_ROOT = Path(__file__).resolve().parents[3]
MCP_SERVER_NAME = os.getenv("CODEX_PROXY_MCP_SERVER_NAME", "jarvis")
MCP_SERVER_MODULE = os.getenv(
    "CODEX_PROXY_MCP_SERVER_MODULE", "src.jarvis.tools.mcp_server"
)
USE_JARVIS_MCP = os.getenv("CODEX_PROXY_USE_MCP", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


class CodexCLIError(RuntimeError):
    """Structured error from the local Codex CLI subprocess."""

    def __init__(self, message: str, *, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def _check_api_key(request: Request):
    if not PROXY_API_KEY:
        return
    key = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(key, PROXY_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


_TOOL_REFUSAL_RE = re.compile(
    r"(\bexposes no\b|\bno\b\s+(desktop|browser|tool|tools|notes?)\s+"
    r"(control|access)\b|\bdoesn['’]?t\s+expose\b|\bdoesn['’]?t\s+"
    r"support\b|\bno\s+.*\bbridge\b|\bno\s+.*\bharness\b|"
    r"\b(can(?:not|'t)|unable|not able to|do not|don't have)\b"
    r"[\s\S]{0,80}\b(use|access|control|open|click|save|run)\b"
    r"[\s\S]{0,40}\b(tools?|browser|desktop|notes?)\b)",
    re.I,
)


def _jarvis_mcp_overrides() -> list[str]:
    pythonpath_parts = [str(REPO_ROOT)]
    existing_pythonpath = os.getenv("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    pythonpath = os.pathsep.join(pythonpath_parts)
    return [
        "-c",
        f"mcp_servers.{MCP_SERVER_NAME}.command={json.dumps(sys.executable)}",
        "-c",
        f"mcp_servers.{MCP_SERVER_NAME}.args="
        f"{json.dumps(['-m', MCP_SERVER_MODULE])}",
        "-c",
        f"mcp_servers.{MCP_SERVER_NAME}.env.PYTHONPATH={json.dumps(pythonpath)}",
    ]


def _run_codex_cli(prompt: str, model: str, *, use_mcp: bool = False) -> str:
    codex_bin = shutil.which("codex")
    if not codex_bin:
        raise RuntimeError("Codex CLI not found in PATH.")

    with NamedTemporaryFile(prefix="jarvis_codex_", suffix=".txt", delete=False) as tmp:
        output_path = tmp.name

    cmd = [
        codex_bin,
        "exec",
        "-m",
        model,
        "--color",
        "never",
        "--disable",
        "shell_tool",
        "-s",
        "workspace-write",
        "--skip-git-repo-check",
        "-o",
        output_path,
        "-",
    ]
    if use_mcp:
        cmd.extend(_jarvis_mcp_overrides())

    try:
        proc = subprocess.run(
            cmd,
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            cwd=str(REPO_ROOT),
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise CodexCLIError("Codex CLI timed out after 120 seconds", status_code=504)
    try:
        raw_output = Path(output_path).read_text(encoding="utf-8").strip()
    except OSError:
        raw_output = ""
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass

    if proc.returncode != 0 or not raw_output:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        out = proc.stdout.decode("utf-8", errors="replace").strip()
        combined = "\n".join(part for part in (err, out) if part).strip()
        detail = combined or f"exit code {proc.returncode}"
        lowered = detail.lower()
        if "usage limit" in lowered or "usage_limit_reached" in lowered:
            raise CodexCLIError(
                "Codex usage limit reached for the active ChatGPT account. "
                "Switch accounts or upgrade the account plan, then try again.",
                status_code=429,
            )
        if "429" in lowered or "rate limit" in lowered:
            raise CodexCLIError(
                "Codex rate limit reached. Please wait and try again.",
                status_code=429,
            )
        raise CodexCLIError(f"Codex CLI error: {detail[:500]}", status_code=500)

    return raw_output


def _build_tool_prompted(
    messages: list[dict[str, Any]],
    tools: list[dict] | None,
    *,
    use_mcp: bool = False,
) -> str:
    prompt_messages = list(messages)
    if use_mcp:
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "Jarvis tools are connected via MCP in this session. "
                    "Use MCP tools directly when they help. "
                    "Do not claim tool unavailability if a suitable Jarvis tool exists."
                ),
            },
            *prompt_messages,
        ]
        return LLMInterface._build_cli_prompt(prompt_messages, None)

    normalized = None
    if tools:
        # OpenAI-style tools: [{"type":"function","function":{...}}]
        if isinstance(tools, list) and tools and "function" in tools[0]:
            normalized = [t.get("function", {}) for t in tools if t.get("function")]
        else:
            normalized = tools
    return LLMInterface._build_cli_prompt(prompt_messages, normalized)


def _tool_calls_to_openai(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments", {})
        out.append(
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )
    return out


@app.get("/v1/models", dependencies=[Depends(_check_api_key)])
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_CODEX_MODEL,
                "object": "model",
                "owned_by": "codex-cli",
            }
        ],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(_check_api_key)])
async def chat_completions(request: Request) -> dict[str, Any]:
    body = await request.json()
    model = body.get("model") or DEFAULT_CODEX_MODEL
    messages = body.get("messages") or []
    tools = body.get("tools")
    use_mcp = bool(tools) and USE_JARVIS_MCP

    prompt = _build_tool_prompted(messages, tools, use_mcp=use_mcp)
    try:
        response_text = await asyncio.to_thread(
            _run_codex_cli, prompt, model, use_mcp=use_mcp
        )
    except CodexCLIError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if use_mcp:
        clean_text = response_text.strip()
        parsed_calls = []
    else:
        clean_text, parsed_calls = LLMInterface._parse_tool_calls(response_text)

    # MCP is preferred, but keep the old tool-call shim as a fallback if Codex still
    # claims tools are unavailable or declines to use them.
    if use_mcp and _TOOL_REFUSAL_RE.search(clean_text or ""):
        logger.warning("Codex MCP mode refused tools; retrying via text tool-call bridge.")
        use_mcp = False
        prompt = _build_tool_prompted(messages, tools, use_mcp=False)
        try:
            response_text = await asyncio.to_thread(
                _run_codex_cli, prompt, model, use_mcp=False
            )
        except CodexCLIError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        clean_text, parsed_calls = LLMInterface._parse_tool_calls(response_text)

    # If tools are available but text mode still refused, retry once with stronger instruction.
    if tools and not use_mcp and not parsed_calls and _TOOL_REFUSAL_RE.search(clean_text or ""):
        messages = list(messages)
        messages.append(
            {
                "role": "user",
                "content": (
                    "[System: The listed tools are available. Do not claim tool "
                    "unavailability. If a listed tool can solve the request, emit "
                    "the appropriate tool_call block now.]"
                ),
            }
        )
        prompt = _build_tool_prompted(messages, tools, use_mcp=False)
        try:
            response_text = await asyncio.to_thread(
                _run_codex_cli, prompt, model, use_mcp=False
            )
        except CodexCLIError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        clean_text, parsed_calls = LLMInterface._parse_tool_calls(response_text)

    tool_calls = _tool_calls_to_openai(parsed_calls)
    finish_reason = "tool_calls" if tool_calls else "stop"

    return {
        "id": f"chatcmpl_{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": clean_text,
                    "tool_calls": tool_calls,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    import uvicorn

    def _default_log_file() -> Path:
        return Path.home() / ".prometheus" / "logs" / "codex-proxy.log"

    def _detach_proxy(host: str, port: int, log_file: Path) -> int:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        module_name = (
            __spec__.name if __spec__ is not None else "src.jarvis.evals.codex_proxy"
        )
        with open(log_file, "a", encoding="utf-8") as log_fh:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    module_name,
                    "--host",
                    host,
                    "--port",
                    str(port),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ},
            )
        return proc.pid

    parser = argparse.ArgumentParser(description="Codex CLI OpenAI-compatible proxy")
    parser.add_argument(
        "--host",
        default=os.getenv("CODEX_PROXY_HOST", "127.0.0.1"),
        help="Host to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("CODEX_PROXY_PORT", "7071")),
        help="Port to bind (default: 7071)",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Start the proxy in a detached background process",
    )
    parser.add_argument(
        "--log-file",
        default=str(_default_log_file()),
        help="Detached-mode log file path",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.detach:
        pid = _detach_proxy(args.host, args.port, Path(args.log_file).expanduser())
        print(pid)
        raise SystemExit(0)

    logger.info(
        "Starting Codex proxy on %s:%d with default model %s",
        args.host,
        args.port,
        DEFAULT_CODEX_MODEL,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
