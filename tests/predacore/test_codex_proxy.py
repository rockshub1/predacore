from fastapi.testclient import TestClient

from src.predacore.evals import codex_proxy


def test_build_tool_prompted_uses_mcp_without_text_tool_instructions():
    prompt = codex_proxy._build_tool_prompted(
        [{"role": "user", "content": "Read a file if needed."}],
        [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        use_mcp=True,
    )

    assert "PredaCore tools are connected via MCP" in prompt
    assert "TOOL USE INSTRUCTIONS" not in prompt
    assert "<tool_call>" not in prompt


def test_predacore_mcp_overrides_include_command_args_and_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/tmp/existing")
    overrides = codex_proxy._predacore_mcp_overrides()
    joined = " ".join(overrides)

    assert f"mcp_servers.{codex_proxy.MCP_SERVER_NAME}.command=" in joined
    assert codex_proxy.MCP_SERVER_MODULE in joined
    assert str(codex_proxy.REPO_ROOT) in joined
    assert "/tmp/existing" in joined


def test_chat_completions_uses_mcp_mode_for_tool_requests(monkeypatch):
    calls = []

    def fake_run(prompt, model, *, use_mcp=False):
        calls.append({"prompt": prompt, "model": model, "use_mcp": use_mcp})
        return "Done after reading the file."

    monkeypatch.setattr(codex_proxy, "_run_codex_cli", fake_run)
    monkeypatch.setattr(codex_proxy, "USE_PREDACORE_MCP", True)

    client = TestClient(codex_proxy.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "Read pyproject.toml."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["message"]["content"] == "Done after reading the file."
    assert body["choices"][0]["message"]["tool_calls"] == []
    assert len(calls) == 1
    assert calls[0]["use_mcp"] is True
    assert "TOOL USE INSTRUCTIONS" not in calls[0]["prompt"]


def test_chat_completions_falls_back_to_text_bridge_after_mcp_refusal(monkeypatch):
    calls = []
    responses = iter(
        [
            "I do not have access to any tools in this environment.",
            '<tool_call>{"name":"read_file","arguments":{"path":"x.txt"}}</tool_call>',
        ]
    )

    def fake_run(prompt, model, *, use_mcp=False):
        calls.append({"prompt": prompt, "model": model, "use_mcp": use_mcp})
        return next(responses)

    monkeypatch.setattr(codex_proxy, "_run_codex_cli", fake_run)
    monkeypatch.setattr(codex_proxy, "USE_PREDACORE_MCP", True)

    client = TestClient(codex_proxy.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "Read x.txt."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    tool_calls = body["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "read_file"
    assert calls[0]["use_mcp"] is True
    assert calls[1]["use_mcp"] is False
    assert "TOOL USE INSTRUCTIONS" in calls[1]["prompt"]
