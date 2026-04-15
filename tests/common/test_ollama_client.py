"""
Tests for the OllamaClient LLM provider.

Tests creation, configuration, error handling. Does NOT require
a running Ollama server — uses monkeypatch and mocking.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jarvis._vendor.common.llm import LLMClient, OllamaClient, get_default_llm_client


class TestOllamaClientCreation:
    """Test OllamaClient initialization."""

    def test_default_config(self):
        client = OllamaClient()
        assert client.model == "llama3.2"
        assert client.base_url == "http://localhost:11434"
        assert client.endpoint == "http://localhost:11434/v1/chat/completions"
        assert client.keep_alive == "5m"

    def test_custom_config(self):
        client = OllamaClient(
            base_url="http://myserver:9999",
            model="mistral",
            keep_alive="10m",
        )
        assert client.model == "mistral"
        assert client.base_url == "http://myserver:9999"
        assert client.endpoint == "http://myserver:9999/v1/chat/completions"
        assert client.keep_alive == "10m"

    def test_env_var_config(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://remote:8080")
        monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "30m")
        client = OllamaClient()
        assert client.base_url == "http://remote:8080"
        assert client.keep_alive == "30m"

    def test_trailing_slash_stripped(self):
        client = OllamaClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"

    def test_isinstance_llm_client(self):
        client = OllamaClient()
        assert isinstance(client, LLMClient)


class TestGetDefaultLLMClientOllama:
    """Test get_default_llm_client factory for Ollama."""

    def test_ollama_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_MODEL", "codellama")
        client = get_default_llm_client()
        assert isinstance(client, OllamaClient)
        assert client.model == "codellama"

    def test_ollama_provider_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "OLLAMA")
        client = get_default_llm_client()
        assert isinstance(client, OllamaClient)

    def test_ollama_provider_with_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_BASE_URL", "http://gpubox:11434")
        client = get_default_llm_client()
        assert isinstance(client, OllamaClient)
        assert client.base_url == "http://gpubox:11434"

    def test_ollama_default_model(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        client = get_default_llm_client()
        assert client.model == "llama3.2"


class TestOllamaClientHealthCheck:
    """Test OllamaClient utility methods (mocked HTTP)."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        client = OllamaClient()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=Exception("connection refused")):
            result = await client.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "models": [
                {"name": "llama3.2"},
                {"name": "mistral"},
                {"name": "codellama"},
            ]
        }
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            models = await client.list_models()
        assert models == ["llama3.2", "mistral", "codellama"]

    @pytest.mark.asyncio
    async def test_list_models_failure_returns_empty(self):
        client = OllamaClient()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=Exception("fail")):
            models = await client.list_models()
        assert models == []


class TestOllamaClientGenerate:
    """Test OllamaClient.generate (mocked HTTP)."""

    @pytest.mark.asyncio
    async def test_generate_success(self):
        client = OllamaClient(model="llama3.2")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello! How can I help you?"}}]
        }
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.generate([{"role": "user", "content": "Hi"}])
        assert result == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_generate_connect_error(self):
        import httpx
        client = OllamaClient()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock,
                   side_effect=httpx.ConnectError("refused")):
            with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
                await client.generate([{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_generate_model_not_found(self):
        import httpx
        client = OllamaClient(model="nonexistent-model")
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_resp
        )
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(RuntimeError, match="not found on Ollama"):
                await client.generate([{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_generate_with_params(self):
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "result"}}]
        }

        captured_payload = {}
        async def capture_post(url, json=None):
            captured_payload.update(json)
            return mock_resp

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=capture_post):
            await client.generate(
                [{"role": "user", "content": "Hi"}],
                params={"temperature": 0.7, "max_tokens": 100}
            )
        assert captured_payload["temperature"] == 0.7
        assert captured_payload["max_tokens"] == 100
        assert captured_payload["stream"] is False
