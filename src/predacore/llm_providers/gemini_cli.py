"""
Gemini CLI Provider — Local-first interaction via the `gemini` binary.

Uses text-based tool calling: PredaCore injects tool definitions into the prompt
and parses <tool_call> blocks from the response. PredaCore controls the full
tool loop directly — no external tool-routing protocol.

Useful for:
- Using models authenticated via `gemini auth login`
- Accessing experimental models not yet in the public API
- Bypassing standard API key requirements if you have CLI access
- Free inference — no API key needed
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from collections.abc import Callable
from typing import Any

from .base import LLMProvider
from .text_tool_adapter import build_full_text_prompt, parse_tool_calls
from .types import AssistantResponse, Message, ToolResultRef

logger = logging.getLogger(__name__)


class GeminiCLIProvider(LLMProvider):
    """Provider that shells out to the 'gemini' CLI tool.

    Tools are injected as text in the prompt. PredaCore parses <tool_call>
    blocks from the response and handles the tool loop itself — full control,
    no external tool-routing protocol needed.
    """

    name = "gemini-cli"
    supports_native_tools = False  # Text-based tool calling — PredaCore controls the loop

    # ------------------------------------------------------------------
    # Tool-turn serialization (Phase A refactor — 2026-04-21).
    #
    # Gemini CLI is text-only: tool calls arrive embedded in model output as
    # ``<tool_call>{...}</tool_call>`` XML blocks and results go back as plain
    # text in user turns. The CLI model doesn't understand the ``"tool"``
    # role, so we override to route results into ``role="user"`` turns
    # with formatted text content. For assistant turns we re-inject the
    # model's own ``<tool_call>`` blocks into ``content`` so the text prompt
    # on the next turn echoes what the model originally emitted.
    # ------------------------------------------------------------------

    def append_assistant_turn(
        self,
        messages: list[Message],
        response: AssistantResponse,
    ) -> None:
        import json

        content = response.content
        for tc in response.tool_calls:
            block = json.dumps({"name": tc.name, "arguments": tc.arguments})
            content += f"\n<tool_call>{block}</tool_call>"
        messages.append(
            Message(
                role="assistant",
                content=content,
                tool_calls=list(response.tool_calls),
            )
        )

    def append_tool_results_turn(
        self,
        messages: list[Message],
        results: list[ToolResultRef],
    ) -> None:
        if not results:
            return
        lines = []
        for r in results:
            status = "ERROR" if r.is_error else "OK"
            lines.append(f"[tool_result: {r.name}] ({status})\n{r.result}")
        messages.append(
            Message(
                role="user",  # CLI model doesn't know the "tool" role
                content="\n\n".join(lines),
                tool_results=list(results),
            )
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream_fn: Callable | None = None,
    ) -> dict[str, Any]:
        """Execute chat via the Gemini CLI with text-based tool support."""
        gemini_bin = shutil.which("gemini")
        if not gemini_bin:
            raise ConnectionError(
                "Gemini CLI binary not found. Please install it or check your PATH."
            )

        # Build prompt — tools are injected as text instructions
        full_prompt = build_full_text_prompt(messages, tools)

        # Determine model
        model = self.config.model or "gemini-2.5-flash"

        # Construct command
        _trust = os.getenv("PREDACORE_TRUST_LEVEL", "normal").lower()
        cmd = [gemini_bin, "-p", "", "-o", "text"]
        if _trust == "yolo":
            cmd.append("--yolo")
        if model and model != "auto":
            cmd.extend(["-m", model])

        logger.info(
            "Gemini CLI: model=%s, prompt=%d chars, tools=%d",
            model, len(full_prompt), len(tools) if tools else 0,
        )

        # Execute
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=full_prompt.encode("utf-8")),
                timeout=600,
            )
        except asyncio.TimeoutError as exc:
            proc.kill()
            raise TimeoutError("Gemini CLI timed out after 600s") from exc

        if proc.returncode != 0:
            err_raw = stderr.decode().strip() or stdout.decode().strip()
            err_lines = err_raw.split("\n")
            key_lines = [
                ln for ln in err_lines
                if ln.strip() and not ln.strip().startswith("at ") and "file:///" not in ln
            ]
            err_msg = "; ".join(key_lines[:3]) if key_lines else err_raw[:200]
            raise RuntimeError(f"Gemini CLI failed (code {proc.returncode}): {err_msg}")

        response_text = stdout.decode("utf-8", errors="replace").strip()

        # Parse tool calls from text response (PredaCore controls the loop)
        clean_text, tool_calls = parse_tool_calls(response_text)

        if stream_fn and clean_text:
            try:
                result = stream_fn(clean_text)
                if asyncio.iscoroutine(result):
                    await result
            except (TypeError, ValueError, OSError):
                pass

        return {
            "content": clean_text,
            "tool_calls": tool_calls,
            "finish_reason": "stop" if not tool_calls else "tool_calls",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "model": model,
        }
