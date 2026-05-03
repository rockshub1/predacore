"""
Verify v1.5.0 new OpenAI-compatible providers are wired correctly.

Kimi K2 (Moonshot), Qwen 3 (Alibaba DashScope), Hyperbolic, and
Perplexity Sonar all speak OpenAI Chat Completions on the wire — a
single PROVIDER_ENDPOINTS entry per provider is enough. These tests
verify the entries exist and resolve to the right base URL + env var.
"""
from __future__ import annotations

import pytest

from predacore.llm_providers.base import ProviderConfig
from predacore.llm_providers.openai import PROVIDER_ENDPOINTS, OpenAIProvider


@pytest.mark.parametrize(
    "name, expected_base_url, expected_env_key",
    [
        ("kimi", "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
        ("qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
        ("hyperbolic", "https://api.hyperbolic.xyz/v1", "HYPERBOLIC_API_KEY"),
        ("perplexity", "https://api.perplexity.ai", "PERPLEXITY_API_KEY"),
    ],
)
def test_new_provider_endpoint_registered(name, expected_base_url, expected_env_key):
    """Each new provider has its base_url + env var entry in PROVIDER_ENDPOINTS."""
    assert name in PROVIDER_ENDPOINTS, f"{name} missing from PROVIDER_ENDPOINTS"
    ep = PROVIDER_ENDPOINTS[name]
    assert ep["base_url"] == expected_base_url
    assert ep["env_key"] == expected_env_key
    assert ep["default_model"]  # non-empty default model — sourced from model_registry


def test_kimi_provider_resolves_endpoint(monkeypatch):
    """Kimi provider resolves to api.moonshot.cn with MOONSHOT_API_KEY."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-kimi-test")
    cfg = ProviderConfig(
        model="kimi-k2.6",
        extras={"provider": "kimi"},
    )
    prov = OpenAIProvider(cfg)
    api_key, base_url, default_model = prov._resolve_endpoint()
    assert api_key == "sk-kimi-test"
    assert base_url == "https://api.moonshot.cn/v1"
    # Default tracks the model_registry's first entry — accept any kimi-* id
    assert default_model.startswith("kimi") or default_model.startswith("moonshot")


def test_qwen_provider_reads_dashscope_env(monkeypatch):
    """Qwen reads DASHSCOPE_API_KEY (not QWEN_API_KEY)."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-qwen-test")
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    cfg = ProviderConfig(extras={"provider": "qwen"})
    prov = OpenAIProvider(cfg)
    api_key, base_url, _ = prov._resolve_endpoint()
    assert api_key == "sk-qwen-test"
    assert "dashscope" in base_url


def test_hyperbolic_provider_default_model(monkeypatch):
    """Hyperbolic ships a recognizable open-weight model as default."""
    monkeypatch.setenv("HYPERBOLIC_API_KEY", "sk-hyp-test")
    cfg = ProviderConfig(extras={"provider": "hyperbolic"})
    prov = OpenAIProvider(cfg)
    _, _, default_model = prov._resolve_endpoint()
    # Default tracks model_registry.MODELS["hyperbolic"][0] — any current
    # entry should belong to one of the major open-weight families.
    assert default_model and any(
        family in default_model.lower()
        for family in ("qwen", "llama", "deepseek", "mistral")
    )


def test_perplexity_provider_default_model(monkeypatch):
    """Perplexity defaults to a Sonar model (built-in web search)."""
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test")
    cfg = ProviderConfig(extras={"provider": "perplexity"})
    prov = OpenAIProvider(cfg)
    _, _, default_model = prov._resolve_endpoint()
    assert "sonar" in default_model.lower()
