"""LLM provider selection and OpenRouter model wiring."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.config import get_settings
from agent.llm_client import (
    classify_llm_error,
    create_adk_model,
    effective_max_agent_turns,
    increment_llm_call_count,
    is_openrouter_free_tier,
    is_tool_use_unsupported_error,
    openrouter_agent_model_id,
    openrouter_model_id,
    openrouter_models_to_try,
    reset_llm_call_count,
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


def test_effective_max_agent_turns_caps_openrouter_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL_ID", "openrouter/free")
    monkeypatch.setenv("MAX_AGENT_TURNS", "8")
    monkeypatch.setenv("OPENROUTER_FREE_MAX_AGENT_TURNS", "3")
    get_settings.cache_clear()
    settings = get_settings()

    assert is_openrouter_free_tier(settings) is True
    assert effective_max_agent_turns(settings) == 3


def test_openrouter_agent_model_id_defaults_for_free_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL_ID", "openrouter/free")
    monkeypatch.delenv("OPENROUTER_AGENT_MODEL_ID", raising=False)
    get_settings.cache_clear()
    settings = get_settings()

    assert openrouter_agent_model_id(settings) == "openrouter/openai/gpt-oss-20b:free"


def test_classify_tool_use_error() -> None:
    exc = Exception('404 {"message":"No endpoints found that support tool use"}')
    assert is_tool_use_unsupported_error(exc) is True
    code, message = classify_llm_error(exc)
    assert code == "LLM_TOOL_UNSUPPORTED"
    assert "OPENROUTER_AGENT_MODEL_ID" in message


def test_create_adk_model_openrouter_disables_retries(
    test_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL_ID", "openrouter/free")
    monkeypatch.setenv("OPENROUTER_AGENT_MODEL_ID", "openai/gpt-oss-20b:free")
    get_settings.cache_clear()
    settings = get_settings()

    model = create_adk_model(settings)
    assert getattr(model, "model", None) == "openrouter/openai/gpt-oss-20b:free"
    assert model._additional_args.get("num_retries") == 0


def test_is_rate_limit_error_detects_openrouter_429() -> None:
    from agent.llm_client import is_rate_limit_error

    exc = Exception(
        'litellm.RateLimitError: {"error":{"message":"Provider returned error","code":429}}'
    )
    assert is_rate_limit_error(exc) is True


def test_classify_llm_error_rate_limit_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
    get_settings.cache_clear()

    code, message = classify_llm_error(Exception("RateLimitError: 429"))
    assert code == "LLM_RATE_LIMIT"
    assert "gemini" in message.lower()
    assert "openrouter" not in message.lower()


def test_classify_llm_error_rate_limit_openrouter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "sk-or-test")
    get_settings.cache_clear()

    code, message = classify_llm_error(Exception("RateLimitError: 429"))
    assert code == "LLM_RATE_LIMIT"
    assert "openrouter" in message.lower()


def test_llm_call_counter() -> None:
    reset_llm_call_count()
    assert increment_llm_call_count(model="openrouter/free", source="test") == 1
    assert increment_llm_call_count(model="openrouter/free", source="test") == 2


def test_openrouter_completion_single_attempt_on_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm")
    get_settings.cache_clear()
    reset_llm_call_count()

    rate_error = Exception("litellm.RateLimitError: code 429")

    with patch("litellm.completion", side_effect=rate_error) as mock_completion:
        from agent.llm_client import _openrouter_completion

        with pytest.raises(Exception, match="429"):
            _openrouter_completion(
                model="openrouter/free",
                messages=[{"role": "user", "content": "hi"}],
                api_key="sk-or-test",
            )

    assert mock_completion.call_count == 1
