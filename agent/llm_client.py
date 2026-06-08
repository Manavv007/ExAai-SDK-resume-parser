"""LLM provider wiring (Gemini direct or OpenRouter via LiteLLM)."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from agent.config import Settings

logger = logging.getLogger("exaai_adk.llm_client")

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_AGENT_MODEL = "openai/gpt-oss-20b:free"
OPENROUTER_FREE_ROUTERS = frozenset({"openrouter/free", "free"})

_llm_call_state = threading.local()


def reset_llm_call_count() -> None:
    _llm_call_state.count = 0


def get_llm_call_count() -> int:
    return int(getattr(_llm_call_state, "count", 0))


def increment_llm_call_count(*, model: str, source: str) -> int:
    count = get_llm_call_count() + 1
    _llm_call_state.count = count
    logger.info("LLM call #%s source=%s model=%s", count, source, model)
    return count


def resolve_llm_provider(settings: Settings) -> str:
    """Use explicit LLM_PROVIDER; auto picks OpenRouter when its key is set."""
    if settings.llm_provider in ("gemini", "openrouter"):
        return settings.llm_provider
    if settings.open_router_api_key.strip():
        return "openrouter"
    return "gemini"


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
    """Cap agent LLM round-trips on OpenRouter free tier to avoid burning quota."""
    if is_openrouter_free_tier(settings, for_agent=True):
        return min(settings.max_agent_turns, settings.openrouter_free_max_agent_turns)
    return settings.max_agent_turns


def openrouter_model_id(settings: Settings) -> str:
    return _normalize_openrouter_model(settings.openrouter_model_id)


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
    if name in {"RateLimitError", "RateLimitException"}:
        return True
    text = str(exc).lower()
    return "ratelimit" in text or "rate limit" in text or 'code":429' in text or " 429" in text


def is_tool_use_unsupported_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "support tool use" in text or "support tool" in text or "tool use" in text


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
                "or try GEMINI_MODEL_ID=gemini-2.5-flash with SCREENING_MODE=pipeline "
                "(one LLM call per request).",
            )
        return (
            "LLM_RATE_LIMIT",
            "OpenRouter free tier is temporarily rate-limited. Wait 60s and retry, "
            "or set LLM_PROVIDER=gemini / SCREENING_MODE=pipeline.",
        )
    if is_tool_use_unsupported_error(exc):
        return (
            "LLM_TOOL_UNSUPPORTED",
            "OpenRouter model does not support native tool calling required by agent mode. "
            f"Set OPENROUTER_AGENT_MODEL_ID={DEFAULT_OPENROUTER_AGENT_MODEL} "
            "(default), LLM_PROVIDER=gemini, or SCREENING_MODE=pipeline.",
        )
    return ("LLM_ERROR", str(exc))


def sync_llm_env(settings: Settings) -> None:
    """Expose API keys to google-genai / LiteLLM via process environment."""
    gemini_key = settings.gemini_api_key.strip()
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
        os.environ["GOOGLE_API_KEY"] = gemini_key

    openrouter_key = settings.open_router_api_key.strip()
    if openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = openrouter_key


def create_adk_model(settings: Settings) -> Any:
    """Return Gemini model id string or LiteLlm wrapper for the screening agent."""
    provider = resolve_llm_provider(settings)
    if provider == "openrouter":
        api_key = settings.open_router_api_key.strip()
        if not api_key:
            raise RuntimeError("OPEN_ROUTER_API_KEY is not configured")
        from google.adk.models.lite_llm import LiteLlm

        class CountingLiteLlm(LiteLlm):
            """LiteLlm with per-turn call counting; num_retries=0 avoids 429 amplification."""

            async def generate_content_async(self, llm_request: Any, stream: bool = False):
                increment_llm_call_count(model=str(self.model), source="adk_agent")
                async for chunk in super().generate_content_async(llm_request, stream=stream):
                    yield chunk

        agent_model = openrouter_agent_model_id(settings)
        logger.info("ADK agent OpenRouter model: %s", agent_model)
        return CountingLiteLlm(
            model=agent_model,
            api_key=api_key,
            api_base=OPENROUTER_API_BASE,
            num_retries=0,
        )
    if not settings.gemini_api_key.strip():
        raise RuntimeError("GEMINI_API_KEY is not configured")
    return settings.gemini_model_id


def model_version_label(settings: Settings, *, for_agent: bool = False) -> str:
    provider = resolve_llm_provider(settings)
    if provider == "openrouter":
        model = openrouter_agent_model_id(settings) if for_agent else openrouter_model_id(settings)
        return f"exaai-adk/{model}"
    return f"exaai-adk/{settings.gemini_model_id}"


def _openrouter_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    **kwargs: Any,
) -> Any:
    """Single OpenRouter attempt — never retry 429 on the same model."""
    import litellm

    increment_llm_call_count(model=model, source="scorer")
    return litellm.completion(
        model=model,
        messages=messages,
        api_key=api_key,
        api_base=OPENROUTER_API_BASE,
        num_retries=0,
        **kwargs,
    )


def _openrouter_completion_with_fallbacks(
    *,
    messages: list[dict[str, str]],
    api_key: str,
    settings: Settings,
    **kwargs: Any,
) -> Any:
    models = openrouter_models_to_try(settings)
    last_exc: BaseException | None = None
    for model in models:
        try:
            return _openrouter_completion(
                model=model,
                messages=messages,
                api_key=api_key,
                **kwargs,
            )
        except Exception as exc:
            last_exc = exc
            if not is_rate_limit_error(exc):
                raise
            logger.warning("OpenRouter model %s rate-limited; trying next model", model)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("OpenRouter completion failed without an exception")


def generate_json(prompt: str, *, correction: str | None = None) -> dict[str, Any]:
    """Provider-agnostic JSON generation for pipeline scoring."""
    from agent.config import get_settings
    from agent.tools.scorer import _parse_json_response, _scoring_response_schema

    settings = get_settings()
    contents = prompt
    if correction:
        contents = f"{prompt}\n\nCORRECTION:\n{correction}"

    provider = resolve_llm_provider(settings)
    if provider == "openrouter":
        api_key = settings.open_router_api_key.strip()
        if not api_key:
            raise RuntimeError("OPEN_ROUTER_API_KEY is not configured")

        response = _openrouter_completion_with_fallbacks(
            messages=[{"role": "user", "content": contents}],
            api_key=api_key,
            settings=settings,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "scoring_response",
                    "schema": _scoring_response_schema(),
                    "strict": True,
                },
            },
            temperature=0.1,
            max_tokens=8192,
        )
        message = response.choices[0].message
        text = message.content or ""
        if not text and getattr(message, "tool_calls", None):
            text = message.tool_calls[0].function.arguments or ""
        return _parse_json_response(text)

    import time

    from google import genai
    from google.genai import types
    from google.genai.errors import APIError, ServerError

    if not settings.gemini_api_key.strip():
        raise RuntimeError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=settings.gemini_api_key)

    max_retries = 3
    delay = 1.5
    response = None

    for attempt in range(max_retries):
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
            status_code = getattr(e, "code", getattr(e, "status_code", None))
            if (
                status_code in (503, 429) or "503" in str(e) or "429" in str(e)
            ) and attempt < max_retries - 1:
                logger.warning(
                    f"Gemini API returned transient error (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {delay}s... Error: {e}"
                )
                time.sleep(delay)
                delay *= 2.0
            else:
                raise

    if response is None:
        raise RuntimeError("Gemini call failed with no response")
    return _parse_json_response(response.text or "")
