"""LLM provider wiring (Gemini direct, Groq/OpenRouter via LiteLLM)."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from agent.config import Settings

logger = logging.getLogger("exaai_adk.llm_client")

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_AGENT_MODEL = "openai/gpt-oss-20b:free"
DEFAULT_GROQ_MODEL = "groq/llama-3.3-70b-versatile"
DEFAULT_GROQ_AGENT_MODEL = "groq/llama-3.3-70b-versatile"
OPENROUTER_FREE_ROUTERS = frozenset({"openrouter/free", "free"})
LITELLM_PROVIDERS = frozenset({"openrouter", "groq"})

_llm_call_state = threading.local()


def reset_llm_call_count() -> None:
    _llm_call_state.count = 0
    _llm_call_state.trace = []
    _llm_call_state.gemini_rate_limited = False


def mark_gemini_rate_limited() -> None:
    """Skip further Gemini calls this screening run (quota/429)."""
    _llm_call_state.gemini_rate_limited = True


def is_gemini_rate_limited() -> bool:
    return bool(getattr(_llm_call_state, "gemini_rate_limited", False))


def get_llm_call_count() -> int:
    return int(getattr(_llm_call_state, "count", 0))


def get_llm_call_trace() -> list[dict[str, str | int]]:
    trace = getattr(_llm_call_state, "trace", None)
    return list(trace) if isinstance(trace, list) else []


def increment_llm_call_count(*, model: str, source: str) -> int:
    count = get_llm_call_count() + 1
    _llm_call_state.count = count
    trace = get_llm_call_trace()
    trace.append(
        {
            "n": count,
            "source": source,
            "model": model,
            "ts_ms": int(time.time() * 1000),
        }
    )
    _llm_call_state.trace = trace
    logger.info("LLM call #%s source=%s model=%s", count, source, model)
    return count


def attach_llm_usage_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Merge llm_calls + llm_call_trace into screening metadata."""
    out = dict(metadata) if isinstance(metadata, dict) else {}
    count = get_llm_call_count()
    if count:
        out["llm_calls"] = count
    trace = get_llm_call_trace()
    if trace:
        out["llm_call_trace"] = trace
    return out


def resolve_llm_provider(settings: Settings) -> str:
    """Use explicit LLM_PROVIDER; auto prefers Groq, then OpenRouter, then Gemini."""
    if settings.llm_provider in ("gemini", "openrouter", "groq"):
        return settings.llm_provider
    if settings.groq_api_key.strip():
        return "groq"
    if settings.open_router_api_key.strip():
        return "openrouter"
    return "gemini"


def _provider_credentials(provider: str, settings: Settings) -> tuple[str, str | None]:
    if provider == "openrouter":
        return settings.open_router_api_key.strip(), OPENROUTER_API_BASE
    if provider == "groq":
        return settings.groq_api_key.strip(), None
    raise ValueError(f"Unsupported LiteLLM provider: {provider}")


def _normalize_openrouter_model(model: str) -> str:
    stripped = model.strip() or "openrouter/free"
    if stripped.startswith("openrouter/"):
        return stripped
    return f"openrouter/{stripped}"


def _is_free_openrouter_slug(model: str) -> bool:
    lowered = model.strip().lower()
    return lowered in OPENROUTER_FREE_ROUTERS or lowered.endswith(":free") or "/free" in lowered


def is_openrouter_free_tier(settings: Settings, *, for_agent: bool = False) -> bool:
    """True when the configured OpenRouter model is a free-tier route."""
    if resolve_llm_provider(settings) != "openrouter":
        return False
    model = openrouter_agent_model_id(settings) if for_agent else openrouter_model_id(settings)
    return _is_free_openrouter_slug(model)


def openrouter_agent_model_id(settings: Settings) -> str:
    """
    Model for ADK agent tool calling.

    ``openrouter/free`` cannot satisfy native tool-use on many free providers, so
    agent mode uses ``openrouter_agent_model_id`` (default gpt-oss-20b:free).
    """
    explicit = settings.openrouter_agent_model_id.strip()
    if explicit:
        return _normalize_openrouter_model(explicit)
    primary = settings.openrouter_model_id.strip().lower() or "openrouter/free"
    if primary in OPENROUTER_FREE_ROUTERS:
        return _normalize_openrouter_model(DEFAULT_OPENROUTER_AGENT_MODEL)
    return openrouter_model_id(settings)


def effective_max_agent_turns(settings: Settings) -> int:
    """Cap agent LLM round-trips on rate-limited free tiers (Groq RPM, OpenRouter free)."""
    if resolve_llm_provider(settings) == "groq":
        return min(settings.max_agent_turns, settings.groq_max_agent_turns)
    if is_openrouter_free_tier(settings, for_agent=True):
        return min(settings.max_agent_turns, settings.openrouter_free_max_agent_turns)
    return settings.max_agent_turns


def openrouter_model_id(settings: Settings) -> str:
    return _normalize_openrouter_model(settings.openrouter_model_id)


def _normalize_groq_model(model: str) -> str:
    stripped = model.strip() or DEFAULT_GROQ_MODEL.removeprefix("groq/")
    if stripped.startswith("groq/"):
        return stripped
    return f"groq/{stripped}"


def groq_model_id(settings: Settings) -> str:
    return _normalize_groq_model(settings.groq_model_id)


def groq_agent_model_id(settings: Settings) -> str:
    explicit = settings.groq_agent_model_id.strip()
    if explicit:
        return _normalize_groq_model(explicit)
    return _normalize_groq_model(DEFAULT_GROQ_AGENT_MODEL)


def _groq_models_from_settings(
    settings: Settings,
    *,
    primary: str,
    fallback_csv: str,
) -> list[str]:
    models: list[str] = []
    for raw in (
        primary,
        *[part.strip() for part in fallback_csv.split(",") if part.strip()],
    ):
        normalized = raw if raw.startswith("groq/") else f"groq/{raw}"
        if normalized not in models:
            models.append(normalized)
    return models


def groq_models_to_try(settings: Settings) -> list[str]:
    return _groq_models_from_settings(
        settings,
        primary=groq_model_id(settings),
        fallback_csv=settings.groq_fallback_model_ids,
    )


def groq_agent_models_to_try(settings: Settings) -> list[str]:
    return _groq_models_from_settings(
        settings,
        primary=groq_agent_model_id(settings),
        fallback_csv=settings.groq_fallback_model_ids,
    )


def scoring_models_to_try(settings: Settings, provider: str) -> list[str]:
    if provider == "groq":
        return groq_models_to_try(settings)
    return openrouter_models_to_try(settings)


def agent_model_id_for_provider(settings: Settings, provider: str) -> str:
    if provider == "groq":
        return groq_agent_model_id(settings)
    return openrouter_agent_model_id(settings)


def scoring_model_id_for_provider(settings: Settings, provider: str) -> str:
    if provider == "groq":
        return groq_model_id(settings)
    return openrouter_model_id(settings)


def openrouter_models_to_try(settings: Settings) -> list[str]:
    """Primary model plus configured fallbacks (deduped, order preserved)."""
    models: list[str] = []
    for raw in (
        openrouter_model_id(settings),
        *[
            part.strip()
            for part in settings.openrouter_fallback_model_ids.split(",")
            if part.strip()
        ],
    ):
        normalized = raw if raw.startswith("openrouter/") else f"openrouter/{raw}"
        if normalized not in models:
            models.append(normalized)
    return models


def is_rate_limit_error(exc: BaseException) -> bool:
    """True for OpenRouter/Gemini quota or upstream 429 responses."""
    name = type(exc).__name__
    if name in {
        "RateLimitError",
        "RateLimitException",
        "APIError",
        "ClientError",
        "_ResourceExhaustedError",
    }:
        status_code = getattr(exc, "code", getattr(exc, "status_code", None))
        if status_code in (429, "429"):
            return True
    text = str(exc).lower()
    return (
        "ratelimit" in text
        or "rate limit" in text
        or "resource_exhausted" in text
        or 'code":429' in text
        or " 429" in text
        or text.startswith("429")
    )


def is_tool_use_unsupported_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "support tool use" in text
        or "support tool" in text
        or "tool use" in text
        or "tool_use_failed" in text
        or "failed to call a function" in text
        or "<function=" in text
    )


def is_agent_recoverable_llm_error(exc: BaseException) -> bool:
    """Agent may fall back to pipeline scoring after these LLM failures."""
    return is_rate_limit_error(exc) or is_tool_use_unsupported_error(exc)


def classify_llm_error(exc: BaseException) -> tuple[str, str]:
    """Map an LLM exception to (error_code, user_message)."""
    from agent.config import get_settings

    provider = resolve_llm_provider(get_settings())

    if is_rate_limit_error(exc):
        if provider == "gemini":
            settings = get_settings()
            return (
                "LLM_RATE_LIMIT",
                f"Gemini API quota exceeded for {settings.gemini_model_id}. "
                "Wait a few minutes and retry, enable billing at https://ai.google.dev, "
                "set LLM_PROVIDER=groq with GROQ_API_KEY, or use SCREENING_MODE=pipeline.",
            )
        if provider == "groq":
            return (
                "LLM_RATE_LIMIT",
                "Groq per-minute token/request limit reached (not necessarily the daily cap). "
                "Use SCREENING_MODE=pipeline for one LLM call, "
                "set GROQ_MODEL_ID=llama-3.1-8b-instant, wait 60s and retry, "
                "or enable OPEN_ROUTER_API_KEY as backup.",
            )
        return (
            "LLM_RATE_LIMIT",
            "OpenRouter free tier is temporarily rate-limited. Wait 60s and retry, "
            "or set LLM_PROVIDER=groq / SCREENING_MODE=pipeline.",
        )
    if is_tool_use_unsupported_error(exc):
        if provider == "groq":
            return (
                "LLM_TOOL_UNSUPPORTED",
                "Groq model failed ADK tool calling (tool_use_failed). "
                "Set GROQ_AGENT_MODEL_ID=llama-3.3-70b-versatile, LLM_PROVIDER=openrouter "
                f"with OPENROUTER_AGENT_MODEL_ID={DEFAULT_OPENROUTER_AGENT_MODEL}, "
                "or SCREENING_MODE=pipeline.",
            )
        return (
            "LLM_TOOL_UNSUPPORTED",
            "Model does not support native tool calling required by agent mode. "
            f"Set OPENROUTER_AGENT_MODEL_ID={DEFAULT_OPENROUTER_AGENT_MODEL}, "
            "LLM_PROVIDER=gemini, or SCREENING_MODE=pipeline.",
        )
    return ("LLM_ERROR", str(exc))


def gemini_key_suffix(settings: Settings | None = None) -> str:
    """Last 4 chars of the active Gemini key (for health checks, not logging full secrets)."""
    from agent.config import get_settings

    key = (settings or get_settings()).gemini_api_key.strip()
    return key[-4:] if len(key) >= 4 else "****"


def sync_llm_env(settings: Settings) -> None:
    """Expose API keys to google-genai / LiteLLM via process environment."""
    gemini_key = settings.gemini_api_key.strip()
    if gemini_key:
        # google-genai prefers GOOGLE_API_KEY when both are set; keep them identical.
        os.environ["GEMINI_API_KEY"] = gemini_key
        os.environ["GOOGLE_API_KEY"] = gemini_key
        logger.info(
            "Gemini credentials synced from .env (suffix …%s, model=%s)",
            gemini_key_suffix(settings),
            settings.gemini_model_id,
        )
    else:
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GOOGLE_API_KEY", None)

    openrouter_key = settings.open_router_api_key.strip()
    if openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = openrouter_key

    groq_key = settings.groq_api_key.strip()
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key


def create_adk_model(settings: Settings) -> Any:
    """Return Gemini model id string or LiteLlm wrapper for the screening agent."""
    provider = resolve_llm_provider(settings)
    if provider in LITELLM_PROVIDERS:
        api_key, api_base = _provider_credentials(provider, settings)
        if not api_key:
            key_name = "GROQ_API_KEY" if provider == "groq" else "OPEN_ROUTER_API_KEY"
            raise RuntimeError(f"{key_name} is not configured")
        from google.adk.models.lite_llm import LiteLlm

        agent_models = (
            groq_agent_models_to_try(settings)
            if provider == "groq"
            else [agent_model_id_for_provider(settings, provider)]
        )

        class CountingLiteLlm(LiteLlm):
            """LiteLlm with per-turn call counting; num_retries=0 avoids 429 amplification."""

            _agent_models: tuple[str, ...] = tuple(agent_models)

            async def generate_content_async(self, llm_request: Any, stream: bool = False):
                import asyncio

                last_exc: BaseException | None = None
                for index, model_name in enumerate(self._agent_models):
                    self.model = model_name
                    try:
                        increment_llm_call_count(model=model_name, source="adk_agent")
                        async for chunk in super().generate_content_async(
                            llm_request,
                            stream=stream,
                        ):
                            yield chunk
                        return
                    except Exception as exc:
                        last_exc = exc
                        if (
                            not is_agent_recoverable_llm_error(exc)
                            or index >= len(self._agent_models) - 1
                        ):
                            raise
                        logger.warning(
                            "ADK agent model %s failed (%s); trying %s",
                            model_name,
                            "rate_limit" if is_rate_limit_error(exc) else "tool_use",
                            self._agent_models[index + 1],
                        )
                        await asyncio.sleep(2.0)
                if last_exc is not None:
                    raise last_exc

        agent_model = agent_models[0]
        fallbacks = agent_models[1:]
        logger.info("ADK agent %s model: %s (fallbacks: %s)", provider, agent_model, fallbacks)
        litellm_kwargs: dict[str, Any] = {
            "model": agent_model,
            "api_key": api_key,
            "num_retries": 0,
        }
        if api_base:
            litellm_kwargs["api_base"] = api_base
        return CountingLiteLlm(**litellm_kwargs)
    if not settings.gemini_api_key.strip():
        raise RuntimeError("GEMINI_API_KEY is not configured")
    return settings.gemini_model_id


def model_version_label(settings: Settings, *, for_agent: bool = False) -> str:
    provider = resolve_llm_provider(settings)
    if provider in LITELLM_PROVIDERS:
        if for_agent:
            model = agent_model_id_for_provider(settings, provider)
        else:
            model = scoring_model_id_for_provider(settings, provider)
        return f"exaai-adk/{model}"
    return f"exaai-adk/{settings.gemini_model_id}"


def _litellm_response_format(
    provider: str,
    *,
    schema_name: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Groq free models reliably support json_object; OpenRouter can use json_schema."""
    if provider == "groq":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
            "strict": True,
        },
    }


def _litellm_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    api_base: str | None = None,
    **kwargs: Any,
) -> Any:
    """Single LiteLLM attempt — never retry 429 on the same model."""
    import litellm

    increment_llm_call_count(model=model, source="scorer")
    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "api_key": api_key,
        "num_retries": 0,
        **kwargs,
    }
    if api_base:
        call_kwargs["api_base"] = api_base
    return litellm.completion(**call_kwargs)


def _litellm_completion_with_fallbacks(
    *,
    provider: str,
    messages: list[dict[str, str]],
    api_key: str,
    api_base: str | None,
    settings: Settings,
    **kwargs: Any,
) -> Any:
    import time

    models = scoring_models_to_try(settings, provider)
    last_exc: BaseException | None = None
    for index, model in enumerate(models):
        try:
            return _litellm_completion(
                model=model,
                messages=messages,
                api_key=api_key,
                api_base=api_base,
                **kwargs,
            )
        except Exception as exc:
            last_exc = exc
            if not is_rate_limit_error(exc):
                raise
            logger.warning("%s model %s rate-limited; trying next model", provider, model)
            if index < len(models) - 1:
                time.sleep(2.0)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{provider} completion failed without an exception")


def _message_text_from_litellm_response(response: Any) -> str:
    message = response.choices[0].message
    text = message.content or ""
    if not text and getattr(message, "tool_calls", None):
        text = message.tool_calls[0].function.arguments or ""
    return text


def _generate_json_via_litellm(
    contents: str,
    *,
    settings: Settings,
    provider: str,
) -> dict[str, Any]:
    from agent.tools.scorer import _parse_json_response, _scoring_response_schema

    api_key, api_base = _provider_credentials(provider, settings)
    if not api_key:
        key_name = "GROQ_API_KEY" if provider == "groq" else "OPEN_ROUTER_API_KEY"
        raise RuntimeError(f"{key_name} is not configured")

    response = _litellm_completion_with_fallbacks(
        provider=provider,
        messages=[{"role": "user", "content": contents}],
        api_key=api_key,
        api_base=api_base,
        settings=settings,
        response_format=_litellm_response_format(
            provider,
            schema_name="scoring_response",
            schema=_scoring_response_schema(),
        ),
        temperature=0.1,
        max_tokens=8192,
    )
    return _parse_json_response(_message_text_from_litellm_response(response))


def complete_json_for_provider(
    prompt: str,
    *,
    settings: Settings,
    provider: str,
    schema_name: str,
    schema: dict[str, Any],
    max_tokens: int = 1000,
) -> dict[str, Any]:
    """LiteLLM JSON completion for auxiliary tasks (e.g. GitHub summaries)."""
    import json

    api_key, api_base = _provider_credentials(provider, settings)
    if not api_key:
        key_name = "GROQ_API_KEY" if provider == "groq" else "OPEN_ROUTER_API_KEY"
        raise RuntimeError(f"{key_name} is not configured")

    response = _litellm_completion_with_fallbacks(
        provider=provider,
        messages=[{"role": "user", "content": prompt}],
        api_key=api_key,
        api_base=api_base,
        settings=settings,
        response_format=_litellm_response_format(
            provider,
            schema_name=schema_name,
            schema=schema,
        ),
        temperature=0.1,
        max_tokens=max_tokens,
    )
    return json.loads(_message_text_from_litellm_response(response))


def _generate_json_via_openrouter(
    contents: str,
    *,
    settings: Settings,
) -> dict[str, Any]:
    return _generate_json_via_litellm(contents, settings=settings, provider="openrouter")


def _generate_json_via_groq(
    contents: str,
    *,
    settings: Settings,
) -> dict[str, Any]:
    return _generate_json_via_litellm(contents, settings=settings, provider="groq")


def _litellm_fallback_providers(settings: Settings) -> list[str]:
    providers: list[str] = []
    if settings.open_router_api_key.strip():
        providers.append("openrouter")
    if settings.groq_api_key.strip():
        providers.append("groq")
    return providers


def _providers_to_try_for_scoring(settings: Settings) -> list[str]:
    """Primary provider first, then other configured LiteLLM providers."""
    primary = resolve_llm_provider(settings)
    ordered: list[str] = []
    for candidate in (primary, *_litellm_fallback_providers(settings)):
        if candidate in LITELLM_PROVIDERS and candidate not in ordered:
            ordered.append(candidate)
    return ordered


def _generate_json_with_litellm_fallbacks(
    contents: str,
    *,
    settings: Settings,
) -> dict[str, Any]:
    import time

    last_exc: BaseException | None = None
    for provider in _providers_to_try_for_scoring(settings):
        try:
            return _generate_json_via_litellm(contents, settings=settings, provider=provider)
        except Exception as exc:
            last_exc = exc
            if not is_rate_limit_error(exc):
                raise
            logger.warning(
                "%s scoring rate-limited on all models; trying next provider",
                provider,
            )
            time.sleep(2.0)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LiteLLM scoring failed without an exception")


def generate_json(prompt: str, *, correction: str | None = None) -> dict[str, Any]:
    """Provider-agnostic JSON generation for pipeline scoring."""
    from agent.config import get_settings
    from agent.tools.scorer import _parse_json_response, _scoring_response_schema

    settings = get_settings()
    contents = prompt
    if correction:
        contents = f"{prompt}\n\nCORRECTION:\n{correction}"

    provider = resolve_llm_provider(settings)
    if provider in LITELLM_PROVIDERS:
        return _generate_json_with_litellm_fallbacks(contents, settings=settings)

    if is_gemini_rate_limited():
        if _litellm_fallback_providers(settings):
            logger.warning("Gemini already rate-limited this run; using LiteLLM fallback")
            return _generate_json_with_litellm_fallbacks(contents, settings=settings)

    import time

    from google import genai
    from google.genai import types
    from google.genai.errors import APIError, ServerError

    if not settings.gemini_api_key.strip():
        raise RuntimeError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=settings.gemini_api_key)

    response = None
    last_gemini_error: BaseException | None = None

    for attempt in range(2):
        try:
            increment_llm_call_count(model=settings.gemini_model_id, source="scorer")
            response = client.models.generate_content(
                model=settings.gemini_model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=_scoring_response_schema(),
                    max_output_tokens=8192,
                    temperature=0.1,
                ),
            )
            break
        except (APIError, ServerError) as e:
            last_gemini_error = e
            if is_rate_limit_error(e):
                mark_gemini_rate_limited()
                logger.warning("Gemini rate limit on scoring; skipping retries")
                break
            status_code = getattr(e, "code", getattr(e, "status_code", None))
            if attempt == 0 and (status_code in (503,) or "503" in str(e)):
                logger.warning("Gemini API 503 on scoring; one retry in 1.5s")
                time.sleep(1.5)
                continue
            break

    if response is not None:
        return _parse_json_response(response.text or "")

    if last_gemini_error and is_rate_limit_error(last_gemini_error):
        for fallback_provider in _litellm_fallback_providers(settings):
            logger.warning(
                "Gemini quota/rate limit exhausted; falling back to %s for scoring",
                fallback_provider,
            )
            return _generate_json_via_litellm(
                contents,
                settings=settings,
                provider=fallback_provider,
            )

    if last_gemini_error is not None:
        raise last_gemini_error

    raise RuntimeError("Gemini call failed with no response")
