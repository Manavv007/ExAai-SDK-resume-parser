"""LLM provider selection and OpenRouter model wiring."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.config import get_settings
from agent.llm_client import (
    DEFAULT_GROQ_MODEL,
    classify_llm_error,
    create_adk_model,
    create_genai_client,
    effective_max_agent_turns,
    gemini_configured,
    gemini_vertex_active,
    groq_model_id,
    increment_llm_call_count,
    is_openrouter_free_tier,
    is_tool_use_unsupported_error,
    model_version_label,
    openrouter_agent_model_id,
    openrouter_model_id,
    openrouter_models_to_try,
    reset_llm_call_count,
    resolve_llm_provider,
    sync_llm_env,
)


def test_resolve_llm_provider_auto_prefers_groq_when_key_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("GROQ_MODEL_ID", raising=False)
    monkeypatch.delenv("GROQ_AGENT_MODEL_ID", raising=False)
    get_settings.cache_clear()
    settings = get_settings()

    assert resolve_llm_provider(settings) == "groq"
    assert groq_model_id(settings) == DEFAULT_GROQ_MODEL


def test_resolve_llm_provider_auto_prefers_openrouter_when_only_openrouter_key_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
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
    from agent.llm_client import get_llm_call_trace

    reset_llm_call_count()
    assert increment_llm_call_count(model="openrouter/free", source="test") == 1
    assert increment_llm_call_count(model="openrouter/free", source="test") == 2
    trace = get_llm_call_trace()
    assert len(trace) == 2
    assert trace[0]["source"] == "test"
    assert trace[1]["n"] == 2


def test_generate_json_skips_gemini_after_rate_limit_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "sk-or-test")
    get_settings.cache_clear()

    with (
        patch("google.genai.Client") as mock_client_cls,
        patch(
            "agent.llm_client._generate_json_with_litellm_fallbacks",
            return_value={"recommendation": "hold"},
        ) as mock_litellm,
    ):
        from agent.llm_client import generate_json, mark_gemini_rate_limited, reset_llm_call_count

        reset_llm_call_count()
        mark_gemini_rate_limited()
        result = generate_json("score this resume")

    assert result["recommendation"] == "hold"
    mock_client_cls.assert_not_called()
    mock_litellm.assert_called_once()


def test_generate_json_falls_back_to_openrouter_on_gemini_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.genai.errors import APIError

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "sk-or-test")
    get_settings.cache_clear()

    quota_error = APIError(429, {"error": {"message": "RESOURCE_EXHAUSTED"}}, None)

    with (
        patch("google.genai.Client") as mock_client_cls,
        patch(
            "agent.llm_client._generate_json_via_litellm",
            return_value={"recommendation": "advance"},
        ) as mock_litellm,
    ):
        mock_client_cls.return_value.models.generate_content.side_effect = quota_error
        from agent.llm_client import generate_json

        result = generate_json("score this resume")

    assert result["recommendation"] == "advance"
    assert mock_client_cls.return_value.models.generate_content.call_count == 1
    mock_litellm.assert_called_once()
    assert mock_litellm.call_args.kwargs["provider"] == "openrouter"


def test_litellm_fallback_tries_groq_after_openrouter_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    get_settings.cache_clear()

    groq_error = RuntimeError("groq model unavailable")
    with patch(
        "agent.llm_client._generate_json_via_litellm",
        side_effect=[groq_error, {"recommendation": "hold"}],
    ) as mock_litellm:
        from agent.llm_client import (
            _generate_json_with_litellm_fallbacks,
            mark_gemini_rate_limited,
            reset_llm_call_count,
        )

        reset_llm_call_count()
        mark_gemini_rate_limited()
        result = _generate_json_with_litellm_fallbacks(
            "score this resume",
            settings=get_settings(),
        )

    assert result["recommendation"] == "hold"
    assert mock_litellm.call_count == 2
    assert mock_litellm.call_args_list[0].kwargs["provider"] == "groq"
    assert mock_litellm.call_args_list[1].kwargs["provider"] == "openrouter"


def test_effective_max_agent_turns_bumps_orchestration_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("SCREENING_MODE", "agent")
    monkeypatch.setenv("AGENT_EVIDENCE_ORCHESTRATION_ENABLED", "true")
    monkeypatch.setenv("MAX_AGENT_TURNS", "8")
    get_settings.cache_clear()
    settings = get_settings()

    assert effective_max_agent_turns(settings) == 12


def test_effective_max_agent_turns_caps_groq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("MAX_AGENT_TURNS", "8")
    monkeypatch.setenv("GROQ_MAX_AGENT_TURNS", "3")
    get_settings.cache_clear()
    settings = get_settings()

    assert effective_max_agent_turns(settings) == 3


def test_generate_json_falls_back_to_openrouter_when_groq_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "sk-or-test")
    get_settings.cache_clear()

    rate_error = Exception("litellm.RateLimitError: code 429")

    with (
        patch(
            "agent.llm_client._generate_json_via_litellm",
            side_effect=[rate_error, {"recommendation": "advance"}],
        ) as mock_litellm,
    ):
        from agent.llm_client import generate_json

        result = generate_json("score this resume")

    assert result["recommendation"] == "advance"
    assert mock_litellm.call_count == 2
    assert mock_litellm.call_args_list[0].kwargs["provider"] == "groq"
    assert mock_litellm.call_args_list[1].kwargs["provider"] == "openrouter"


def test_create_adk_model_groq_uses_litellm(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    pytest.importorskip("litellm")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("GROQ_AGENT_MODEL_ID", "llama-3.3-70b-versatile")
    get_settings.cache_clear()
    settings = get_settings()

    model = create_adk_model(settings)
    assert getattr(model, "model", None) == "groq/llama-3.3-70b-versatile"
    assert model._agent_models[0] == "groq/llama-3.3-70b-versatile"
    assert model._additional_args.get("num_retries") == 0


def test_litellm_completion_single_attempt_on_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm")
    get_settings.cache_clear()
    reset_llm_call_count()

    rate_error = Exception("litellm.RateLimitError: code 429")

    with patch("litellm.completion", side_effect=rate_error) as mock_completion:
        from agent.llm_client import _litellm_completion

        with pytest.raises(Exception, match="429"):
            _litellm_completion(
                model="groq/llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "hi"}],
                api_key="gsk-test",
            )

    assert mock_completion.call_count == 1


def test_gemini_vertex_configured_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GCP_PROJECT_ID", "exaai-sdk")
    monkeypatch.setenv("GCP_REGION", "asia-south1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    get_settings.cache_clear()
    settings = get_settings()

    assert gemini_vertex_active(settings) is True
    assert gemini_configured(settings) is True
    assert model_version_label(settings).startswith("exaai-adk/vertex/")


def test_vertex_project_can_differ_from_sandbox_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_USE_VERTEXAI", "true")
    monkeypatch.setenv("VERTEX_GCP_PROJECT_ID", "serin-490413")
    monkeypatch.setenv("GCP_PROJECT_ID", "exaai-sdk")
    monkeypatch.setenv("GCP_REGION", "asia-south1")
    get_settings.cache_clear()
    settings = get_settings()

    from agent.config import resolve_vertex_gcp_project

    assert resolve_vertex_gcp_project(settings) == "serin-490413"
    assert settings.gcp_project_id == "exaai-sdk"


def test_sync_llm_env_vertex_clears_api_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_USE_VERTEXAI", "true")
    monkeypatch.setenv("VERTEX_GCP_PROJECT_ID", "serin-490413")
    monkeypatch.setenv("GCP_PROJECT_ID", "exaai-sdk")
    monkeypatch.setenv("GCP_REGION", "asia-south1")
    monkeypatch.setenv("GEMINI_API_KEY", "should-not-leak")
    get_settings.cache_clear()
    settings = get_settings()

    sync_llm_env(settings)

    import os

    assert os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "1"
    assert os.environ.get("GOOGLE_CLOUD_PROJECT") == "serin-490413"
    assert os.environ.get("GOOGLE_CLOUD_LOCATION") == "asia-south1"
    assert "GOOGLE_API_KEY" not in os.environ
    assert "GEMINI_API_KEY" not in os.environ


def test_create_genai_client_vertex_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_USE_VERTEXAI", "true")
    monkeypatch.setenv("VERTEX_GCP_PROJECT_ID", "serin-490413")
    monkeypatch.setenv("GCP_PROJECT_ID", "exaai-sdk")
    monkeypatch.setenv("GCP_REGION", "asia-south1")
    get_settings.cache_clear()
    settings = get_settings()

    with (
        patch("google.genai.Client") as mock_client_cls,
        patch("agent.gcp_credentials.load_gcp_credentials", return_value="creds"),
    ):
        create_genai_client(settings)
        mock_client_cls.assert_called_once_with(
            vertexai=True,
            project="serin-490413",
            location="asia-south1",
            credentials="creds",
        )
