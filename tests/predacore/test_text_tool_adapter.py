"""Tests for the shared text-based tool calling adapter."""
from src.predacore.llm_providers.text_tool_adapter import (
    build_tool_prompt,
    build_full_text_prompt,
    parse_tool_calls,
)


SAMPLE_TOOLS = [
    {
        "name": "desktop_control",
        "description": "Control the local desktop.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Desktop action to execute",
                    "enum": ["open_app", "press_key"],
                },
                "app_name": {"type": "string", "description": "App name"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "file_read",
        "description": "Read a file from disk.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
            },
            "required": ["path"],
        },
    },
]


# --- build_tool_prompt ---


def test_build_tool_prompt_empty_none():
    assert build_tool_prompt(None) == ""


def test_build_tool_prompt_empty_list():
    assert build_tool_prompt([]) == ""


def test_build_tool_prompt_includes_tool_names():
    prompt = build_tool_prompt(SAMPLE_TOOLS)
    assert "desktop_control" in prompt
    assert "file_read" in prompt


def test_build_tool_prompt_includes_descriptions():
    prompt = build_tool_prompt(SAMPLE_TOOLS)
    assert "Control the local desktop." in prompt
    assert "Read a file from disk." in prompt


def test_build_tool_prompt_includes_enums():
    prompt = build_tool_prompt(SAMPLE_TOOLS)
    assert "allowed: open_app, press_key" in prompt


def test_build_tool_prompt_anti_refusal():
    prompt = build_tool_prompt(SAMPLE_TOOLS)
    assert "MUST emit a <tool_call>" in prompt
    assert "Do NOT say" in prompt


def test_build_tool_prompt_includes_xml_format():
    prompt = build_tool_prompt(SAMPLE_TOOLS)
    assert "<tool_call>" in prompt
    assert "</tool_call>" in prompt


# --- parse_tool_calls ---


def test_parse_tool_calls_single():
    text = 'Sure! <tool_call>{"name": "file_read", "arguments": {"path": "/tmp/x"}}</tool_call>'
    clean, calls = parse_tool_calls(text)
    assert clean == "Sure!"
    assert len(calls) == 1
    assert calls[0]["name"] == "file_read"
    assert calls[0]["arguments"] == {"path": "/tmp/x"}


def test_parse_tool_calls_multiple():
    text = (
        'I will do two things.\n'
        '<tool_call>{"name": "file_read", "arguments": {"path": "a.txt"}}</tool_call>\n'
        '<tool_call>{"name": "desktop_control", "arguments": {"action": "open_app"}}</tool_call>'
    )
    clean, calls = parse_tool_calls(text)
    assert "I will do two things." in clean
    assert len(calls) == 2
    assert calls[0]["name"] == "file_read"
    assert calls[1]["name"] == "desktop_control"


def test_parse_tool_calls_single_quotes():
    text = "<tool_call>{'name': 'file_read', 'arguments': {'path': '/tmp/x'}}</tool_call>"
    clean, calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "file_read"


def test_parse_tool_calls_malformed_skipped():
    text = '<tool_call>this is not json</tool_call> and more text'
    clean, calls = parse_tool_calls(text)
    assert len(calls) == 0
    assert "and more text" in clean


def test_parse_tool_calls_no_blocks():
    text = "Just a normal response with no tool calls."
    clean, calls = parse_tool_calls(text)
    assert clean == text
    assert calls == []


def test_parse_tool_calls_missing_name_skipped():
    text = '<tool_call>{"arguments": {"path": "/tmp"}}</tool_call>'
    clean, calls = parse_tool_calls(text)
    assert len(calls) == 0


# --- build_full_text_prompt ---


def test_build_full_text_prompt_includes_tools_and_messages():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    prompt = build_full_text_prompt(messages, SAMPLE_TOOLS)
    assert "desktop_control" in prompt
    assert "[SYSTEM]" in prompt
    assert "[USER]" in prompt
    assert "[ASSISTANT]" in prompt
    assert "Hello" in prompt


def test_build_full_text_prompt_no_tools():
    messages = [{"role": "user", "content": "Hi"}]
    prompt = build_full_text_prompt(messages, None)
    assert "[USER]" in prompt
    assert "Hi" in prompt
    assert "tool_call" not in prompt
