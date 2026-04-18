"""
Shared text-based tool calling adapter.

Single source of truth for injecting tool definitions into text prompts
and parsing <tool_call> blocks from LLM responses. Used as a fallback
when providers don't support native function calling, or when a model
rejects native tool schemas.
"""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)


def build_tool_prompt(tools: list[dict] | None) -> str:
    """Build a text prompt section describing available tools.

    Returns a string to append to the system prompt. Includes:
      - Mandatory XML tag format with examples
      - Tool names, descriptions, and parameter schemas
      - Anti-refusal language so models don't reject tool usage
    """
    if not tools:
        return ""

    lines = [
        "",
        "## TOOL USE INSTRUCTIONS (MANDATORY)",
        "",
        "You have access to the tools listed below. When you need to perform an action,",
        "you MUST emit a <tool_call> XML block. Do NOT describe what you would do — actually do it.",
        "Do NOT say 'I will use tool X' without emitting the <tool_call> block.",
        "",
        "### Format",
        "```",
        '<tool_call>{"name": "TOOL_NAME", "arguments": {"param1": "value1"}}</tool_call>',
        "```",
        "",
        "### Examples",
        "",
        "To run a shell command:",
        '<tool_call>{"name": "run_command", "arguments": {"command": "brew install --cask spotify"}}</tool_call>',
        "",
        "To read a file:",
        '<tool_call>{"name": "read_file", "arguments": {"path": "/etc/hosts"}}</tool_call>',
        "",
        "To search the web:",
        '<tool_call>{"name": "web_search", "arguments": {"query": "latest news"}}</tool_call>',
        "",
        "### Rules",
        "1. ALWAYS use <tool_call> tags — never just describe the tool call in prose.",
        "2. You may include brief explanation BEFORE the <tool_call> block.",
        "3. You may emit multiple <tool_call> blocks in one response.",
        "4. After a tool result is returned, summarize the outcome for the user.",
        "5. ONLY use tool names from the list below. Do NOT invent tool names.",
        "",
        "### Available Tools",
    ]

    for tool in tools:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        lines.append(f"  - **{name}**: {desc}")

        # Show parameter schema so the model knows what to pass
        params = tool.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))

        if props:
            lines.append("      Parameters:")
            for pname, pdef in props.items():
                ptype = pdef.get("type", "any")
                pdesc = pdef.get("description", "")
                req_marker = " (required)" if pname in required else ""
                line = f"        - {pname} ({ptype}{req_marker}): {pdesc}"
                if "enum" in pdef:
                    enum_vals = ", ".join(str(v) for v in pdef["enum"])
                    line += f" — allowed: {enum_vals}"
                lines.append(line)

    return "\n".join(lines)


def build_full_text_prompt(
    messages: list[dict], tools: list[dict] | None
) -> str:
    """Flatten messages + tool definitions into a single text prompt.

    Used by CLI providers (e.g. Gemini CLI) that take one string as input
    rather than structured messages.
    """
    parts = []

    if tools:
        parts.append(build_tool_prompt(tools))

    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        parts.append(f"\n[{role}]\n{content}")

    parts.append("\n[ASSISTANT]")
    return "\n".join(parts)


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Parse <tool_call> JSON blocks from LLM text output.

    Returns:
        (clean_text, tool_calls) where clean_text has tool_call blocks removed
        and tool_calls is a list of {"name": str, "arguments": dict}.
    """
    tool_calls: list[dict] = []

    for match in _TOOL_CALL_PATTERN.finditer(text):
        try:
            json_str = match.group(1)
            # Try parsing as-is first to preserve single quotes inside values
            try:
                call = json.loads(json_str)
            except json.JSONDecodeError:
                # Last resort: fix single-quoted JSON keys/string delimiters
                # Only replace quotes that appear to be JSON delimiters, not those in values
                try:
                    import re as _re
                    # Replace single quotes that are JSON structural delimiters
                    fixed = _re.sub(r"(?<=[\[{,:])\s*'|'\s*(?=[}\],:])", '"', json_str)
                    call = json.loads(fixed)
                except json.JSONDecodeError:
                    logger.warning("Skipping unparseable tool_call block: %s", json_str[:200])
                    continue
            if "name" in call:
                tool_calls.append(
                    {
                        "name": call["name"],
                        "arguments": call.get("arguments", {}),
                    }
                )
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Skipping malformed tool_call block: %s", e)
            continue

    clean = _TOOL_CALL_PATTERN.sub("", text).strip()
    return clean, tool_calls
