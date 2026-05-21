"""Provider configuration management — API keys, active provider, model selection.

Stores configuration in a server-side JSON file. Environment variables take
precedence over stored keys for each provider.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider catalog (hardcoded — add new providers here)
# ---------------------------------------------------------------------------

PROVIDER_CATALOG: list[dict[str, Any]] = [
    {
        "id": "gemini",
        "name": "Google Gemini",
        "default_model": "gemini-3.1-flash-lite-preview",
        "models": [
            "gemini-3.1-flash-lite-preview",
            "gemini-3-flash-preview",
            "gemini-3.1-pro-preview",
        ],
        "env_keys": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "requires_key": True,
    },
    {
        "id": "anthropic",
        "name": "Anthropic",
        "default_model": "claude-sonnet-4-6",
        "models": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"],
        "env_keys": ["ANTHROPIC_API_KEY"],
        "requires_key": True,
    },
    {
        "id": "openai",
        "name": "OpenAI",
        "default_model": "gpt-5.4-nano",
        "models": ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"],
        "env_keys": ["OPENAI_API_KEY"],
        "requires_key": True,
    },
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "default_model": "anthropic/claude-sonnet-4.6",
        "models": [
            "anthropic/claude-sonnet-4.6",
            "google/gemini-3.1-flash-lite-preview",
            "meta-llama/llama-4-maverick",
            "openai/gpt-4o",
        ],
        "env_keys": ["OPENROUTER_API_KEY"],
        "requires_key": True,
    },
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "default_model": "deepseek-v4-flash",
        "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
        "env_keys": ["DEEPSEEK_API_KEY"],
        "requires_key": True,
    },
    {
        "id": "ollama",
        "name": "Ollama (Local)",
        "default_model": "llama3.2",
        "models": ["llama3.2", "codellama", "deepseek-coder-v2", "qwen2.5-coder"],
        "env_keys": [],
        "requires_key": False,
    },
    {
        "id": "litellm",
        "name": "LiteLLM",
        "default_model": "groq/llama-3.1-70b-versatile",
        "models": ["groq/llama-3.1-70b-versatile"],
        "env_keys": [],
        "requires_key": True,
    },
]

_CATALOG_BY_ID = {p["id"]: p for p in PROVIDER_CATALOG}

# ---------------------------------------------------------------------------
# Dynamic model discovery — Ollama and OpenRouter
# ---------------------------------------------------------------------------

# In-process cache keyed by config value: (config_key, models_list, fetched_at_epoch).
# Invalidated immediately when base_url / api_key changes, not just on TTL expiry.
_OLLAMA_MODELS_CACHE: tuple[str, list[str], float] | None = None
_OPENROUTER_MODELS_CACHE: tuple[str, list[str], float] | None = None
_DISCOVERY_CACHE_TTL = 3600.0  # seconds

# Curated OpenRouter models shown when live fetch fails or returns nothing useful.
_OPENROUTER_FALLBACK_MODELS: list[str] = [
    "anthropic/claude-sonnet-4.6",
    "google/gemini-3.1-flash-lite-preview",
    "meta-llama/llama-4-maverick",
    "openai/gpt-4o",
]

# Only surface free or low-cost OpenRouter models when auto-discovering.
# Models whose IDs contain any of these strings are included; all others are
# filtered out so the list stays manageable.
_OPENROUTER_INCLUDE_PATTERNS = (
    "claude",
    "gemini",
    "llama",
    "gpt-4o",
    "mistral",
    "deepseek",
    "qwen",
)


def _fetch_ollama_models(base_url: str) -> list[str]:
    """Fetch available model names from a running Ollama instance via GET /api/tags."""
    try:
        import httpx

        url = base_url.rstrip("/") + "/api/tags"
        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", []) if m.get("name")]
    except Exception as exc:
        logger.warning("ollama_model_discovery_failed: %s", exc)
        return []


def _fetch_openrouter_models(api_key: str) -> list[str]:
    """Fetch available models from the OpenRouter API, filtered to useful ones."""
    try:
        import httpx

        resp = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_ids: list[str] = [m["id"] for m in data.get("data", []) if m.get("id")]
        filtered = [
            mid
            for mid in raw_ids
            if any(pat in mid.lower() for pat in _OPENROUTER_INCLUDE_PATTERNS)
        ]
        return filtered if filtered else _OPENROUTER_FALLBACK_MODELS
    except Exception as exc:
        logger.warning("openrouter_model_discovery_failed: %s", exc)
        return _OPENROUTER_FALLBACK_MODELS


def _get_ollama_models_cached() -> list[str]:
    """Return Ollama model list, re-fetching when cache is stale."""
    global _OLLAMA_MODELS_CACHE

    base_url = os.environ.get("OLLAMA_BASE_URL", "").strip()
    if not base_url:
        base_url = "http://localhost:11434"

    now = time.time()
    if _OLLAMA_MODELS_CACHE is not None:
        cached_base_url, models, fetched_at = _OLLAMA_MODELS_CACHE
        if cached_base_url == base_url and now - fetched_at < _DISCOVERY_CACHE_TTL:
            return models

    models = _fetch_ollama_models(base_url)
    if not models:
        # Fall back to catalog defaults when Ollama is unreachable
        models = list(_CATALOG_BY_ID["ollama"]["models"])
    _OLLAMA_MODELS_CACHE = (base_url, models, now)
    return models


def _get_openrouter_models_cached() -> list[str]:
    """Return OpenRouter model list, re-fetching when cache is stale."""
    global _OPENROUTER_MODELS_CACHE

    api_key = _get_key_for_provider("openrouter")
    if not api_key:
        return _OPENROUTER_FALLBACK_MODELS

    now = time.time()
    if _OPENROUTER_MODELS_CACHE is not None:
        cached_api_key, models, fetched_at = _OPENROUTER_MODELS_CACHE
        if cached_api_key == api_key and now - fetched_at < _DISCOVERY_CACHE_TTL:
            return models

    models = _fetch_openrouter_models(api_key)
    _OPENROUTER_MODELS_CACHE = (api_key, models, now)
    return models


# ---------------------------------------------------------------------------
# Config file I/O
# ---------------------------------------------------------------------------


def _config_path() -> Path:
    config_dir = os.environ.get("REPOWISE_CONFIG_DIR", "")
    if config_dir:
        return Path(config_dir) / "provider_config.json"
    return Path("provider_config.json")


def _load_config() -> dict[str, Any]:
    path = _config_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read provider config, using defaults")
    return {}


def _save_config(config: dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _get_key_for_provider(provider_id: str) -> str | None:
    """Get API key: env var takes precedence, then stored config."""
    catalog = _CATALOG_BY_ID.get(provider_id)
    if not catalog:
        return None

    # Check env vars first
    for env_key in catalog.get("env_keys", []):
        val = os.environ.get(env_key)
        if val:
            return val

    # Check stored config
    config = _load_config()
    keys = config.get("keys", {})
    return keys.get(provider_id)


def _get_base_url_for_provider(provider_id: str) -> str | None:
    """Resolve a provider base_url from environment variables."""
    env_map = {
        "anthropic": ["ANTHROPIC_BASE_URL"],
        "openai": ["OPENAI_BASE_URL"],
        "gemini": ["GEMINI_BASE_URL"],
        "ollama": ["OLLAMA_BASE_URL"],
        "deepseek": ["DEEPSEEK_BASE_URL"],
        "litellm": ["LITELLM_BASE_URL", "LITELLM_API_BASE"],
    }
    for env_var in env_map.get(provider_id, []):
        val = os.environ.get(env_var)
        if val:
            return val
    return None


def list_provider_status() -> dict[str, Any]:
    """Return the full provider status including active selection."""
    config = _load_config()
    active_id = config.get("active_provider")
    active_model = config.get("active_model")

    # Auto-detect active if not set
    if not active_id:
        for p in PROVIDER_CATALOG:
            if _get_key_for_provider(p["id"]) or not p["requires_key"]:
                active_id = p["id"]
                active_model = p["default_model"]
                break

    providers = []
    for p in PROVIDER_CATALOG:
        has_key = bool(_get_key_for_provider(p["id"]))
        configured = has_key or not p["requires_key"]

        # Dynamic model discovery for Ollama and OpenRouter.
        if p["id"] == "ollama" and configured:
            models = _get_ollama_models_cached()
        elif p["id"] == "openrouter" and configured:
            models = _get_openrouter_models_cached()
        else:
            models = p["models"]

        providers.append(
            {
                "id": p["id"],
                "name": p["name"],
                "models": models,
                "default_model": p["default_model"],
                "configured": configured,
            }
        )

    return {
        "active": {
            "provider": active_id,
            "model": active_model
            or (_CATALOG_BY_ID.get(active_id, {}).get("default_model") if active_id else None),
        },
        "providers": providers,
    }


def get_active_provider() -> tuple[str | None, str | None]:
    """Return (provider_id, model) for the currently active provider."""
    status = list_provider_status()
    active = status["active"]
    return active["provider"], active["model"]


def set_active_provider(provider_id: str, model: str | None = None) -> None:
    """Set the active provider and model. Persists to config file."""
    if provider_id not in _CATALOG_BY_ID:
        raise ValueError(f"Unknown provider: {provider_id}")
    config = _load_config()
    config["active_provider"] = provider_id
    config["active_model"] = model or _CATALOG_BY_ID[provider_id]["default_model"]
    _save_config(config)


def set_api_key(provider_id: str, key: str | None) -> None:
    """Store or remove an API key for a provider."""
    if provider_id not in _CATALOG_BY_ID:
        raise ValueError(f"Unknown provider: {provider_id}")
    config = _load_config()
    keys = config.setdefault("keys", {})
    if key:
        keys[provider_id] = key
    else:
        keys.pop(provider_id, None)
    _save_config(config)


def get_chat_provider_instance():
    """Create a provider instance for chat using the active config.

    Returns a provider that implements both BaseProvider and ChatProvider.
    """
    from repowise.core.providers.llm.registry import get_provider

    provider_id, model = get_active_provider()
    if not provider_id:
        raise ValueError("No active provider configured. Set an API key first.")

    api_key = _get_key_for_provider(provider_id)
    base_url = _get_base_url_for_provider(provider_id)
    catalog = _CATALOG_BY_ID[provider_id]

    kwargs: dict[str, Any] = {"model": model or catalog["default_model"]}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    return get_provider(provider_id, with_rate_limiter=False, **kwargs)
