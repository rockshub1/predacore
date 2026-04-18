from src.predacore.llm_providers.router import LLMInterface


def test_tool_prompt_includes_harness_override_and_enums():
    prompt = LLMInterface._build_tool_prompt(
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

    assert "TOOL USE INSTRUCTIONS" in prompt
    assert "allowed: open_app, press_key" in prompt
