"""Tests for dynamic model discovery in provider_config.py.

Uses respx to intercept HTTP calls so no real Ollama/OpenRouter instance
is needed.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from repowise.server import provider_config as pc


def _reset_caches():
    """Clear in-process discovery caches between tests."""
    pc._OLLAMA_MODELS_CACHE = None
    pc._OPENROUTER_MODELS_CACHE = None


# ---------------------------------------------------------------------------
# Ollama discovery
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_ollama_models_returns_names():
    _reset_caches()
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [
                    {"name": "llama3.2"},
                    {"name": "codellama"},
                    {"name": "qwen2.5-coder"},
                ]
            },
        )
    )
    models = pc._fetch_ollama_models("http://localhost:11434")
    assert models == ["llama3.2", "codellama", "qwen2.5-coder"]


@respx.mock
def test_fetch_ollama_models_empty_when_server_down():
    _reset_caches()
    respx.get("http://localhost:11434/api/tags").mock(side_effect=httpx.ConnectError("refused"))
    models = pc._fetch_ollama_models("http://localhost:11434")
    assert models == []


@respx.mock
def test_fetch_ollama_models_strips_trailing_slash_from_base_url():
    _reset_caches()
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "mistral"}]})
    )
    models = pc._fetch_ollama_models("http://localhost:11434/")
    assert models == ["mistral"]


@respx.mock
def test_get_ollama_models_cached_falls_back_to_catalog_on_failure(monkeypatch):
    _reset_caches()
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    respx.get("http://localhost:11434/api/tags").mock(side_effect=httpx.ConnectError("refused"))
    models = pc._get_ollama_models_cached()
    catalog_defaults = list(pc._CATALOG_BY_ID["ollama"]["models"])
    assert models == catalog_defaults


@respx.mock
def test_get_ollama_models_cached_uses_cache(monkeypatch):
    _reset_caches()
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "gemma3"}]})
    )
    first = pc._get_ollama_models_cached()
    # Second call — mock would raise if it were hit again
    respx.get("http://localhost:11434/api/tags").mock(side_effect=RuntimeError("should not call"))
    second = pc._get_ollama_models_cached()
    assert first == second == ["gemma3"]


# ---------------------------------------------------------------------------
# OpenRouter discovery
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_openrouter_models_filters_by_include_patterns():
    _reset_caches()
    respx.get("https://openrouter.ai/api/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"id": "anthropic/claude-sonnet-4.6"},
                    {"id": "google/gemini-2-flash"},
                    {"id": "some-obscure/model-xyz"},  # should be filtered
                    {"id": "meta-llama/llama-4-scout"},
                ]
            },
        )
    )
    models = pc._fetch_openrouter_models("test-api-key")
    assert "anthropic/claude-sonnet-4.6" in models
    assert "google/gemini-2-flash" in models
    assert "meta-llama/llama-4-scout" in models
    assert "some-obscure/model-xyz" not in models


@respx.mock
def test_fetch_openrouter_models_falls_back_on_http_error():
    _reset_caches()
    respx.get("https://openrouter.ai/api/v1/models").mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )
    models = pc._fetch_openrouter_models("bad-key")
    assert models == pc._OPENROUTER_FALLBACK_MODELS


@respx.mock
def test_fetch_openrouter_models_falls_back_on_connect_error():
    _reset_caches()
    respx.get("https://openrouter.ai/api/v1/models").mock(
        side_effect=httpx.ConnectError("refused")
    )
    models = pc._fetch_openrouter_models("test-key")
    assert models == pc._OPENROUTER_FALLBACK_MODELS


@respx.mock
def test_get_openrouter_models_cached_no_key_returns_fallback(monkeypatch):
    _reset_caches()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    models = pc._get_openrouter_models_cached()
    assert models == pc._OPENROUTER_FALLBACK_MODELS


@respx.mock
def test_get_openrouter_models_cached_uses_cache(monkeypatch):
    _reset_caches()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    respx.get("https://openrouter.ai/api/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"id": "anthropic/claude-sonnet-4.6"}]},
        )
    )
    first = pc._get_openrouter_models_cached()
    respx.get("https://openrouter.ai/api/v1/models").mock(
        side_effect=RuntimeError("should not call")
    )
    second = pc._get_openrouter_models_cached()
    assert first == second


# ---------------------------------------------------------------------------
# list_provider_status — integration
# ---------------------------------------------------------------------------


@respx.mock
def test_list_provider_status_injects_ollama_models(monkeypatch, tmp_path):
    _reset_caches()
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("REPOWISE_CONFIG_DIR", str(tmp_path))
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(
            200,
            json={"models": [{"name": "custom-model"}, {"name": "another-model"}]},
        )
    )
    status = pc.list_provider_status()
    ollama = next(p for p in status["providers"] if p["id"] == "ollama")
    assert "custom-model" in ollama["models"]
    assert "another-model" in ollama["models"]


@respx.mock
def test_list_provider_status_injects_openrouter_models(monkeypatch, tmp_path):
    _reset_caches()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("REPOWISE_CONFIG_DIR", str(tmp_path))
    respx.get("https://openrouter.ai/api/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"id": "anthropic/claude-opus-4"}, {"id": "openai/gpt-4o"}]},
        )
    )
    status = pc.list_provider_status()
    openrouter = next(p for p in status["providers"] if p["id"] == "openrouter")
    assert "anthropic/claude-opus-4" in openrouter["models"]
    assert "openai/gpt-4o" in openrouter["models"]


def test_list_provider_status_uses_catalog_for_unconfigured_ollama(monkeypatch, tmp_path):
    _reset_caches()
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("REPOWISE_CONFIG_DIR", str(tmp_path))
    status = pc.list_provider_status()
    ollama = next(p for p in status["providers"] if p["id"] == "ollama")
    # Ollama is requires_key=False so configured=True — but with no base_url,
    # discovery hits localhost and will fail, falling back to catalog defaults.
    # Either catalog defaults or a superset is fine.
    assert len(ollama["models"]) >= 1
