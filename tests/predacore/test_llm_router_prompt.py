"""
Pin the tool-prompt contract: the prompt that text-mode LLM providers
(gemini_cli, ollama, etc.) see when tools are available must include:
  - The "TOOL USE INSTRUCTIONS" header so the model knows what's expected
  - Enum constraint hints ("allowed: <values>") so the model picks valid args

Wave 12 (2026-05-11): the original test called ``LLMInterface._build_tool_prompt``
which was refactored away. The same contract lives at
``predacore.llm_providers.text_tool_adapter.build_tool_prompt`` (a
module-level function) and is consumed by ``gemini_cli.py:107`` via
``build_full_text_prompt``. This rewrite pins the contract against the
current location.
"""
from __future__ import annotations

from predacore.llm_providers.text_tool_adapter import build_tool_prompt


def test_tool_prompt_includes_instructions_header_and_enums():
    prompt = build_tool_prompt(
        [
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
            }
        ]
    )

    # Header text models latch onto when deciding to emit tool calls.
    assert "TOOL USE INSTRUCTIONS" in prompt
    # Enum hint so the model picks a valid action.
    assert "allowed: open_app, press_key" in prompt
    # Required-flag marker so the model knows action is mandatory.
    assert "(required)" in prompt


def test_tool_prompt_empty_when_no_tools():
    """Empty/None tools → empty string (caller can safely concat)."""
    assert build_tool_prompt(None) == ""
    assert build_tool_prompt([]) == ""
