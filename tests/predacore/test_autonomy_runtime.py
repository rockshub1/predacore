from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.predacore.agents.autonomy import (
    KillSwitchActivatedError,
    OpenClawBridgeRuntime,
)
from src.predacore.config import (
    PredaCoreConfig,
    LaunchProfileConfig,
    LLMConfig,
    MemoryConfig,
    OpenClawBridgeConfig,
)


@pytest.fixture
def runtime_config(tmp_path: Path) -> PredaCoreConfig:
    home = tmp_path / ".prometheus"
    (home / "sessions").mkdir(parents=True)
    (home / "skills").mkdir(parents=True)
    (home / "logs").mkdir(parents=True)
    (home / "memory").mkdir(parents=True)
    return PredaCoreConfig(
        home_dir=str(home),
        sessions_dir=str(home / "sessions"),
        skills_dir=str(home / "skills"),
        logs_dir=str(home / "logs"),
        llm=LLMConfig(provider="gemini-cli"),
        memory=MemoryConfig(persistence_dir=str(home / "memory")),
        launch=LaunchProfileConfig(
            profile="public_beast",
            approvals_required=False,
            egm_mode="off",
            default_code_network=True,
            enable_openclaw_bridge=True,
        ),
        openclaw=OpenClawBridgeConfig(
            base_url="https://bridge.example.com",
            task_path="/v1/responses",
            status_path="/v1/tasks/{task_id}",
            model="openclaw",
            agent_id="main",
            max_retries=2,
            retry_backoff_seconds=0.1,
            poll_interval_seconds=0.1,
            max_poll_seconds=5,
        ),
    )


@pytest.mark.asyncio
async def test_delegate_uses_idempotency_cache(runtime_config: PredaCoreConfig):
    runtime = OpenClawBridgeRuntime(runtime_config)
    runtime._http_post_json = AsyncMock(  # type: ignore[attr-defined]
        return_value=({"status": "completed", "result": {"ok": True}}, 200, 1)
    )

    first = await runtime.delegate(task="test-task", context={"k": "v"})
    second = await runtime.delegate(task="test-task", context={"k": "v"})

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert runtime._http_post_json.await_count == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_delegate_openresponses_payload_shape(runtime_config: PredaCoreConfig):
    runtime = OpenClawBridgeRuntime(runtime_config)
    captured: dict[str, Any] = {}

    async def _fake_post_json(
        *,
        endpoint: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: int,
    ) -> Any:
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        captured["headers"] = headers
        captured["timeout_seconds"] = timeout_seconds
        return {"status": "completed", "id": "resp_1"}, 200, 1

    runtime._http_post_json = _fake_post_json  # type: ignore[assignment]

    result = await runtime.delegate(
        task="build status page",
        context={"service": "api_gateway", "priority": "high"},
    )

    assert result["status"] == "completed"
    assert captured["endpoint"] == "https://bridge.example.com/v1/responses"
    assert captured["payload"]["model"] == "openclaw"
    assert captured["payload"]["input"] == "build status page"
    assert "Task context (JSON):" in captured["payload"]["instructions"]
    assert "api_gateway" in captured["payload"]["metadata"]["context_json"]
    assert captured["headers"]["x-openclaw-agent-id"] == "main"


@pytest.mark.asyncio
async def test_delegate_openresponses_does_not_poll_without_status_url(
    runtime_config: PredaCoreConfig,
):
    runtime = OpenClawBridgeRuntime(runtime_config)
    runtime._http_post_json = AsyncMock(  # type: ignore[attr-defined]
        return_value=({"status": "in_progress", "id": "resp_1"}, 200, 1)
    )
    runtime._http_get_json = AsyncMock(  # type: ignore[attr-defined]
        side_effect=AssertionError("openresponses path should not poll by task id")
    )

    result = await runtime.delegate(task="check status", context={})

    assert result["polls"] == 0
    assert result["status_url"] == ""
    assert result["status"] == "in_progress"
    assert runtime._http_get_json.await_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_http_post_json_retries_transient_failures(
    runtime_config: PredaCoreConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = OpenClawBridgeRuntime(runtime_config)
    runtime._retry_sleep = AsyncMock(return_value=None)  # type: ignore[attr-defined]

    class _FakeResponse:
        def __init__(self, status_code: int, payload: dict[str, Any]):
            self.status_code = status_code
            self._payload = payload
            self.text = str(payload)

        def json(self) -> dict[str, Any]:
            return self._payload

    class _FakeAsyncClient:
        _calls = 0

        def __init__(self, *args: Any, **kwargs: Any):
            pass

        async def __aenter__(self) -> _FakeAsyncClient:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

        async def post(self, endpoint: str, json: dict[str, Any], headers: dict[str, str], **kwargs):
            _FakeAsyncClient._calls += 1
            if _FakeAsyncClient._calls == 1:
                return _FakeResponse(503, {"status": "queued", "id": "job-1"})
            return _FakeResponse(200, {"status": "completed", "id": "job-1"})

    # Mock httpx with required exception types
    fake_httpx = type("FakeHTTPX", (), {
        "AsyncClient": _FakeAsyncClient,
        "TransportError": type("TransportError", (Exception,), {}),
        "TimeoutException": type("TimeoutException", (Exception,), {}),
    })
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    body, status_code, attempts = await runtime._http_post_json(
        endpoint="https://bridge.example.com/v1/responses",
        payload={"task": "t"},
        headers={"Content-Type": "application/json"},
        timeout_seconds=10,
    )

    assert status_code == 200
    assert attempts == 2
    assert body["status"] == "completed"


@pytest.mark.asyncio
async def test_delegate_polls_until_terminal_state(runtime_config: PredaCoreConfig):
    runtime_config.openclaw.task_path = "/v1/tasks/oneshot"
    runtime = OpenClawBridgeRuntime(runtime_config)
    runtime._http_post_json = AsyncMock(  # type: ignore[attr-defined]
        return_value=({"status": "queued", "task_id": "job-42"}, 202, 1)
    )

    poll_payloads: list[dict[str, Any]] = [
        {"status": "running", "task_id": "job-42"},
        {"status": "completed", "task_id": "job-42", "result": {"answer": "ok"}},
    ]

    async def _fake_get_json(
        *,
        endpoint: str,
        headers: dict[str, str],
        timeout_seconds: int,
    ) -> Any:
        payload = poll_payloads.pop(0)
        return payload, 200

    runtime._http_get_json = _fake_get_json  # type: ignore[assignment]
    runtime._settings.poll_interval_seconds = 0

    result = await runtime.delegate(task="long-running", context={"scope": "test"})

    assert result["status"] == "completed"
    assert result["polls"] == 2
    assert result["response"]["result"]["answer"] == "ok"
    assert result["status_url"].endswith("/v1/tasks/job-42")


@pytest.mark.asyncio
async def test_delegate_blocked_by_kill_switch(
    runtime_config: PredaCoreConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = OpenClawBridgeRuntime(runtime_config)
    runtime._http_post_json = AsyncMock(  # type: ignore[attr-defined]
        return_value=({"status": "completed"}, 200, 1)
    )
    monkeypatch.setenv("PREDACORE_KILL_SWITCH", "1")

    with pytest.raises(KillSwitchActivatedError):
        await runtime.delegate(task="blocked", context={})

    assert runtime._http_post_json.await_count == 0  # type: ignore[attr-defined]
