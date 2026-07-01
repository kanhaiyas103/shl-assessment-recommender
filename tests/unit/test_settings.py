"""Tests for environment configuration invariants."""

import pytest
from pydantic import ValidationError

from shl_agent.models.enums import LlmProvider
from shl_agent.utils.settings import Settings, get_settings


def test_default_settings_are_safe_for_foundation_checks() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_provider is LlmProvider.DISABLED
    assert settings.max_conversation_messages == 8
    assert settings.recommendation_limit == 10


def test_enabled_provider_requires_credentials() -> None:
    with pytest.raises(ValidationError, match="llm_api_key"):
        Settings(
            _env_file=None,
            llm_provider=LlmProvider.GEMINI,
            llm_model="gemini-model",
        )


def test_openrouter_requires_base_url() -> None:
    with pytest.raises(ValidationError, match="llm_base_url"):
        Settings(
            _env_file=None,
            llm_provider=LlmProvider.OPENROUTER,
            llm_model="openrouter-model",
            llm_api_key="secret",
        )


def test_enabled_provider_requires_model() -> None:
    with pytest.raises(ValidationError, match="llm_model"):
        Settings(
            _env_file=None,
            llm_provider=LlmProvider.GEMINI,
            llm_api_key="secret",
        )


def test_valid_openrouter_configuration() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider=LlmProvider.OPENROUTER,
        llm_model="provider/model",
        llm_api_key="secret",
        llm_base_url="https://openrouter.ai/api/v1",
    )

    assert settings.llm_provider is LlmProvider.OPENROUTER


def test_get_settings_caches_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHL_APP_ENV", "test")
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second
    get_settings.cache_clear()
