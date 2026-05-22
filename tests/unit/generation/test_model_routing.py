"""Tests for model routing — tier-based provider selection in PageGenerator."""

from __future__ import annotations

import pytest

from repowise.core.generation.models import PAGE_TYPE_TIER
from repowise.core.generation.page_generator import PageGenerator
from repowise.core.generation.context_assembler import ContextAssembler
from repowise.core.generation.models import GenerationConfig
from repowise.core.providers.llm.mock import MockProvider


# ---------------------------------------------------------------------------
# PAGE_TYPE_TIER completeness
# ---------------------------------------------------------------------------


def test_page_type_tier_covers_all_generation_levels():
    from repowise.core.generation.models import GENERATION_LEVELS

    for page_type in GENERATION_LEVELS:
        assert page_type in PAGE_TYPE_TIER, f"{page_type!r} missing from PAGE_TYPE_TIER"


def test_page_type_tier_only_valid_tiers():
    valid = {"cheap", "medium", "premium"}
    for page_type, tier in PAGE_TYPE_TIER.items():
        assert tier in valid, f"{page_type!r} has unknown tier {tier!r}"


def test_premium_pages():
    assert PAGE_TYPE_TIER["repo_overview"] == "premium"
    assert PAGE_TYPE_TIER["architecture_diagram"] == "premium"
    assert PAGE_TYPE_TIER["onboarding"] == "premium"


def test_cheap_pages():
    assert PAGE_TYPE_TIER["file_page"] == "cheap"
    assert PAGE_TYPE_TIER["symbol_spotlight"] == "cheap"


def test_medium_pages():
    assert PAGE_TYPE_TIER["module_page"] == "medium"
    assert PAGE_TYPE_TIER["scc_page"] == "medium"
    assert PAGE_TYPE_TIER["infra_page"] == "medium"
    assert PAGE_TYPE_TIER["api_contract"] == "medium"


# ---------------------------------------------------------------------------
# PageGenerator._provider_for
# ---------------------------------------------------------------------------


def _make_gen(tier_providers=None):
    config = GenerationConfig(max_tokens=256, token_budget=500, max_concurrency=1)
    provider = MockProvider()
    assembler = ContextAssembler(config)
    return PageGenerator(provider, assembler, config, tier_providers=tier_providers)


def test_provider_for_no_tiers_returns_primary():
    gen = _make_gen()
    for page_type in PAGE_TYPE_TIER:
        assert gen._provider_for(page_type) is gen._provider


def test_provider_for_cheap_tier():
    cheap = MockProvider(model="cheap-model")
    medium = MockProvider(model="medium-model")
    premium = MockProvider(model="premium-model")
    gen = _make_gen({"cheap": cheap, "medium": medium, "premium": premium})

    assert gen._provider_for("file_page") is cheap
    assert gen._provider_for("symbol_spotlight") is cheap


def test_provider_for_medium_tier():
    cheap = MockProvider(model="cheap-model")
    medium = MockProvider(model="medium-model")
    premium = MockProvider(model="premium-model")
    gen = _make_gen({"cheap": cheap, "medium": medium, "premium": premium})

    assert gen._provider_for("module_page") is medium
    assert gen._provider_for("scc_page") is medium
    assert gen._provider_for("infra_page") is medium


def test_provider_for_premium_tier():
    cheap = MockProvider(model="cheap-model")
    medium = MockProvider(model="medium-model")
    premium = MockProvider(model="premium-model")
    gen = _make_gen({"cheap": cheap, "medium": medium, "premium": premium})

    assert gen._provider_for("repo_overview") is premium
    assert gen._provider_for("architecture_diagram") is premium
    assert gen._provider_for("onboarding") is premium


def test_provider_for_missing_tier_falls_back_to_primary():
    """When a tier is absent from the map, _provider_for falls back to self._provider."""
    premium = MockProvider(model="premium-model")
    gen = _make_gen({"premium": premium})  # only premium wired

    # cheap + medium → fallback to primary
    assert gen._provider_for("file_page") is gen._provider
    assert gen._provider_for("module_page") is gen._provider
    assert gen._provider_for("repo_overview") is premium


def test_provider_for_unknown_page_type_falls_back_to_primary():
    cheap = MockProvider(model="cheap-model")
    gen = _make_gen({"cheap": cheap})
    # Unknown page type → no tier entry → falls back to self._provider
    assert gen._provider_for("nonexistent_type") is gen._provider


# ---------------------------------------------------------------------------
# Cache key varies by tier provider model
# ---------------------------------------------------------------------------


def test_cache_key_differs_when_tier_model_differs():
    cheap = MockProvider(model="cheap-model")
    premium = MockProvider(model="premium-model")
    gen = _make_gen({"cheap": cheap, "premium": premium})

    key_cheap = gen._compute_cache_key("file_page", "same prompt")
    key_premium = gen._compute_cache_key("repo_overview", "same prompt")
    assert key_cheap != key_premium


def test_cache_key_same_page_type_same_prompt():
    """Same page_type + same model + same prompt → same cache key."""
    cheap = MockProvider(model="cheap-model")
    gen = _make_gen({"cheap": cheap})

    k1 = gen._compute_cache_key("file_page", "prompt A")
    k2 = gen._compute_cache_key("file_page", "prompt A")
    assert k1 == k2


# ---------------------------------------------------------------------------
# Backwards compatibility — single-provider mode
# ---------------------------------------------------------------------------


def test_single_provider_mode_is_default():
    """Creating PageGenerator without tier_providers still works identically."""
    gen = _make_gen()
    assert gen._tier_providers == {}
    for pt in PAGE_TYPE_TIER:
        assert gen._provider_for(pt) is gen._provider
