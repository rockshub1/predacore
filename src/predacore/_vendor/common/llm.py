"""
Provider-agnostic LLM client abstraction.

Supports OpenAI, OpenRouter, Gemini API, Gemini CLI, and Ollama (local).
Reads configuration from environment variables:

- LLM_PROVIDER: "openai" | "openrouter" | "gemini" | "gemini-cli" | "ollama" (default: gemini-cli)
- LLM_MODEL: model name/id (e.g., "gpt-5", "gemini-2.0-flash", "llama3.2")
- LLM_BASE_URL: optional override for API base URL
- OPENAI_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY: provider API keys
- OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
- OLLAMA_KEEP_ALIVE: keep model loaded duration (default: 5m)
- LLM_REASONING: "minimal" | "low" | "medium" | "high" (optional)
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from functools import partial
from typing import Any

import httpx


class LLMClient:
    def __init__(self, model: str | None = None, logger: Any | None = None):
        self.model = model
        self.logger = logger

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError


class GeminiCLIClient(LLMClient):
    """
    Gemini client that uses the gemini CLI directly via subprocess.
    Uses credentials from 'gemini auth login' - no API key required.
    """

    def __init__(self, model: str | None = None, logger: Any | None = None):
        super().__init__(model=model or "gemini-3-pro-preview", logger=logger)
        self._gemini_path = "gemini"  # Assumes gemini is in PATH
        timeout_raw = (os.getenv("GEMINI_CLI_TIMEOUT_SECONDS") or "45").strip()
        self._timeout_seconds = int(timeout_raw) if timeout_raw.isdigit() else 45
        if self._timeout_seconds < 5:
            self._timeout_seconds = 5
        self._fallback_models = self._parse_fallback_models()

    def _parse_fallback_models(self) -> list[str]:
        raw = os.getenv("LLM_FALLBACK_MODELS", "gemini-3-flash-preview,gemini-2.5-pro")
        return [m.strip() for m in raw.split(",") if m.strip()]

    def _candidate_models(self, requested_model: str | None) -> list[str]:
        ordered = [requested_model or self.model, *self._fallback_models]
        deduped: list[str] = []
        for name in ordered:
            if name and name not in deduped:
                deduped.append(name)
        return deduped

    @staticmethod
    def _should_try_fallback(result: str) -> bool:
        lowered = result.lower()
        return (
            "not found" in lowered
            or "unknown model" in lowered
            or "invalid model" in lowered
            or "unsupported model" in lowered
            or "does not exist" in lowered
            or "exhausted your capacity" in lowered
            or "quota" in lowered
            or "rate limit" in lowered
            or "resource_exhausted" in lowered
            or "temporarily unavailable" in lowered
            or "overloaded" in lowered
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        # Build prompt from messages
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.insert(0, f"System: {content}\n\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n\n")
            else:
                prompt_parts.append(f"{content}")

        prompt = "".join(prompt_parts)

        loop = asyncio.get_running_loop()
        models = self._candidate_models(model)
        last_error = "Error: Gemini CLI returned no response."

        for model_name in models:
            result = await loop.run_in_executor(
                None,
                partial(self._call_gemini_cli, prompt, model_name),
            )
            if not result.startswith("Error:"):
                return result
            last_error = result
            if self.logger:
                self.logger.warning(
                    "Gemini CLI attempt failed for model=%s: %s",
                    model_name,
                    result,
                )
            if "timeout" in result.lower():
                return result
            if not self._should_try_fallback(result):
                return result

        return last_error

    def _call_gemini_cli(self, prompt: str, model_name: str) -> str:
        """Call gemini CLI synchronously (runs in executor)."""
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                [
                    self._gemini_path,
                    "--model",
                    model_name,
                    "--output-format",
                    "text",
                    "-",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            stdout, stderr = process.communicate(
                input=prompt, timeout=self._timeout_seconds
            )
            if process.returncode != 0:
                if self.logger:
                    self.logger.error(
                        "Gemini CLI error (%s): %s", model_name, stderr.strip()
                    )
                return f"Error: {stderr.strip() or f'Gemini CLI exited {process.returncode}'}"

            # Filter out the "Loaded cached credentials" line
            output_lines = stdout.strip().split("\n")
            filtered_lines = [
                line for line in output_lines if not line.startswith("Loaded cached")
            ]
            return "\n".join(filtered_lines).strip()
        except subprocess.TimeoutExpired:
            if process is not None:
                try:
                    if hasattr(os, "killpg") and process.pid != os.getpid():
                        # Graceful SIGTERM first, then SIGKILL after a short wait
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                except Exception:
                    process.kill()
                process.communicate()
            return f"Error: Gemini CLI timeout after {self._timeout_seconds}s"
        except FileNotFoundError:
            return "Error: gemini CLI not found. Run 'gemini auth login' first."


class OpenRouterClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        logger: Any | None = None,
    ):
        super().__init__(model=model, logger=logger)
        self.api_key = api_key
        self.base_url = (base_url or "https://openrouter.ai").rstrip("/")
        self.endpoint = f"{self.base_url}/api/v1/chat/completions"

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
        }
        if params:
            payload.update({k: v for k, v in params.items() if v is not None})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(self.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


class OpenAIClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        logger: Any | None = None,
    ):
        super().__init__(model=model, logger=logger)
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com").rstrip("/")
        # Using Chat Completions for broad compatibility
        self.endpoint = f"{self.base_url}/v1/chat/completions"

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
        }
        if params:
            # Pass through common parameters (temperature, max_tokens, etc.)
            payload.update({k: v for k, v in params.items() if v is not None})

        # Map reasoning effort if provided (best-effort; ignored if unsupported)
        reasoning_effort = (params or {}).get("reasoning_effort")
        if reasoning_effort:
            payload.setdefault("extra_body", {})
            payload["extra_body"]["reasoning_effort"] = reasoning_effort

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(self.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


class GeminiClient(LLMClient):
    """Google Gemini API client using OAuth or API key authentication."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        logger: Any | None = None,
    ):
        super().__init__(model=model or "gemini-2.0-flash", logger=logger)
        self.api_key = api_key
        self._access_token: str | None = None
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

        # If no API key, try to use OAuth credentials from google-auth
        if not self.api_key:
            self._init_oauth()

    def _init_oauth(self):
        """Initialize OAuth credentials from Application Default Credentials."""
        try:
            import google.auth
            from google.auth.transport.requests import Request

            # This picks up credentials from:
            # 1. GOOGLE_APPLICATION_CREDENTIALS env var
            # 2. gcloud auth application-default login
            # 3. Gemini CLI cached credentials
            credentials, project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/generative-language"]
            )

            # Refresh to get access token
            credentials.refresh(Request())
            self._access_token = credentials.token
            self._credentials = credentials
            if self.logger:
                self.logger.info("Gemini OAuth initialized successfully")
        except Exception as e:
            # Fallback: try to read from gemini CLI cache directly
            try:
                import json
                from pathlib import Path

                cache_path = Path.home() / ".gemini" / "credentials.json"
                if cache_path.exists():
                    creds = json.loads(cache_path.read_text())
                    self._access_token = creds.get("access_token")
                    if self.logger:
                        self.logger.info("Using Gemini CLI cached credentials")
            except Exception as cache_err:
                if self.logger:
                    self.logger.warning(
                        f"Could not init OAuth: {e}. Cache fallback also failed: {cache_err}"
                    )

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        model_name = model or self.model

        # Refresh OAuth token if expired
        if hasattr(self, '_credentials') and self._credentials and getattr(self._credentials, 'expired', False):
            try:
                from google.auth.transport.requests import Request
                self._credentials.refresh(Request())
                self._access_token = self._credentials.token
            except Exception as e:
                if self.logger:
                    self.logger.warning("OAuth token refresh failed: %s", e)

        # Build endpoint URL based on auth method
        if self.api_key:
            endpoint = f"{self.base_url}/models/{model_name}:generateContent"
            headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        elif self._access_token:
            endpoint = f"{self.base_url}/models/{model_name}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._access_token}",
            }
        else:
            return "Error: No Gemini API key or OAuth credentials available"

        # Convert OpenAI-style messages to Gemini format
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:  # user
                contents.append({"role": "user", "parts": [{"text": content}]})

        payload: dict[str, Any] = {"contents": contents}

        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        # Add generation config if params provided
        if params:
            gen_config = {}
            if "temperature" in params:
                gen_config["temperature"] = params["temperature"]
            if "max_tokens" in params:
                gen_config["maxOutputTokens"] = params["max_tokens"]
            if gen_config:
                payload["generationConfig"] = gen_config

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # Extract text from Gemini response
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return ""


def _get_reasoning_effort() -> str | None:
    val = os.getenv("LLM_REASONING")
    if not val:
        return None
    val = val.strip().lower()
    if val in {"minimal", "low", "medium", "high"}:
        return val
    return None


class NvidiaClient(LLMClient):
    """NVIDIA NIM API client.

    Uses the OpenAI-compatible chat completions endpoint at
    https://integrate.api.nvidia.com/v1.

    Configuration:
        NVIDIA_API_KEY: NIM API key (nvapi-...)
        LLM_MODEL: model id (default: meta/llama-3.3-70b-instruct)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str | None = None,
        logger: Any | None = None,
    ):
        super().__init__(model=model or "meta/llama-3.3-70b-instruct", logger=logger)
        self.api_key = api_key
        self.base_url = (base_url or "https://integrate.api.nvidia.com").rstrip("/")
        self.endpoint = f"{self.base_url}/v1/chat/completions"

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        import re

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
        }
        if params:
            payload.update({k: v for k, v in params.items() if v is not None})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(self.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"] or ""
            # Strip <think>...</think> blocks from reasoning models
            content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
            return content


class OllamaClient(LLMClient):
    """Ollama client for local LLM inference.

    Connects to a local (or remote) Ollama server using
    the OpenAI-compatible chat completions API.

    Configuration:
        OLLAMA_HOST: server URL (default: http://localhost:11434)
        OLLAMA_KEEP_ALIVE: keep model loaded duration (default: 5m)
        LLM_MODEL: Ollama model name (default: llama3.2)
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        keep_alive: str | None = None,
        logger: Any | None = None,
    ):
        super().__init__(model=model or "llama3.2", logger=logger)
        self.base_url = (
            base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ).rstrip("/")
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        self.keep_alive = keep_alive or os.getenv("OLLAMA_KEEP_ALIVE", "5m")

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
        }
        if params:
            payload.update({k: v for k, v in params.items() if v is not None})

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(self.endpoint, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Start it with: ollama serve"
            ) from exc
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise RuntimeError(
                    f"Model '{model or self.model}' not found on Ollama. "
                    f"Pull it with: ollama pull {model or self.model}"
                ) from e
            raise

    async def list_models(self) -> list[str]:
        """List available models on the Ollama server."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    async def health_check(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(self.base_url)
                return resp.status_code == 200
        except Exception:
            return False


def get_default_llm_client(logger: Any | None = None) -> LLMClient:
    provider = (os.getenv("LLM_PROVIDER") or "gemini-cli").strip().lower()
    model = os.getenv("LLM_MODEL") or None
    base_url = os.getenv("LLM_BASE_URL") or None

    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set for LLM provider 'openai'")
        return OpenAIClient(api_key=key, base_url=base_url, model=model, logger=logger)
    elif provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set for LLM provider 'openrouter'"
            )
        return OpenRouterClient(
            api_key=key, base_url=base_url, model=model, logger=logger
        )
    elif provider == "gemini":
        # Use Gemini REST API with API key
        key = os.getenv("GEMINI_API_KEY")
        return GeminiClient(api_key=key, model=model, logger=logger)
    elif provider == "nvidia":
        key = os.getenv("NVIDIA_API_KEY")
        if not key:
            raise RuntimeError("NVIDIA_API_KEY not set for LLM provider 'nvidia'")
        return NvidiaClient(api_key=key, base_url=base_url, model=model, logger=logger)
    elif provider == "ollama":
        return OllamaClient(base_url=base_url, model=model, logger=logger)
    else:  # gemini-cli (default) - uses gemini auth login
        return GeminiCLIClient(model=model, logger=logger)


def default_params(temperature: float = 0.2, max_tokens: int = 1024) -> dict[str, Any]:
    return {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": _get_reasoning_effort(),
    }
