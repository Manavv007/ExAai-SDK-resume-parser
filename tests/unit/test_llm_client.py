"""LLM provider selection and OpenRouter model wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.config import get_settings
from agent.llm_client import (
    classify_llm_error,
    create_adk_model,
    is_rate_limit_error,
    openrouter_model_id,
    openrouter_models_to_try,
    resolve_llm_provider,
)


def test_resolve_llm_provider_auto_prefers_openrouter_when_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "sk-or-test")
    get_settings.cache_clear()
    settings = get_settings()

    assert resolve_llm_provider(settings) == "openrouter"
    assert openrouter_model_id(settings) == "openrouter/free"


def test_openrouter_models_to_try_includes_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL_ID", "openrouter/free")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODEL_IDS", "openai/gpt-oss-20b:free")
    get_settings.cache_clear()
    settings = get_settings()

    assert openrouter_models_to_try(settings) == [
        "openrouter/free",
        "openrouter/openai/gpt-oss-20b:free",
    ]


def test_create_adk_model_openrouter(test_settings, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("litellm")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL_ID", "openrouter/free")
    get_settings.cache_clear()
    settings = get_settings()

    model = create_adk_model(settings)
    assert getattr(model, "model", None) == "openrouter/free"


def test_is_rate_limit_error_detects_openrouter_429() -> None:
    exc = Exception(
        'litellm.RateLimitError: {"error":{"message":"Provider returned error","code":429}}'
    )
    assert is_rate_limit_error(exc) is True


def test_classify_llm_error_rate_limit() -> None:
    code, message = classify_llm_error(Exception("RateLimitError: 429"))
    assert code == "LLM_RATE_LIMIT"
    assert "retry" in message.lower()


def test_openrouter_completion_retries_on_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    get_settings.cache_clear()
    settings = get_settings()

    ok_response = MagicMock()
    rate_error = Exception("litellm.RateLimitError: code 429")

    with (
        patch("litellm.completion", side_effect=[rate_error, ok_response]) as mock_completion,
        patch("agent.llm_client.time.sleep") as mock_sleep,
    ):
        from agent.llm_client import _openrouter_completion

        result = _openrouter_completion(
            model="openrouter/free",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-or-test",
            settings=settings,
        )

    assert result is ok_response
    assert mock_completion.call_count == 2
    mock_sleep.assert_called_once()
