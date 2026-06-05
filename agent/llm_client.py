"""LLM provider wiring (Gemini direct or OpenRouter via LiteLLM)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from agent.config import Settings

logger = logging.getLogger("exaai_adk.llm_client")

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def resolve_llm_provider(settings: Settings) -> str:
    """Use explicit LLM_PROVIDER; auto picks OpenRouter when its key is set."""
    if settings.llm_provider in ("gemini", "openrouter"):
        return settings.llm_provider
    if settings.open_router_api_key.strip():
        return "openrouter"
    return "gemini"


def openrouter_model_id(settings: Settings) -> str:
    model = settings.openrouter_model_id.strip() or "openrouter/free"
    if model.startswith("openrouter/"):
        return model
    return f"openrouter/{model}"


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
    return "ratelimit" in text or "rate limit" in text or "code\":429" in text or " 429" in text


def classify_llm_error(exc: BaseException) -> tuple[str, str]:
    """Map an LLM exception to (error_code, user_message)."""
    if is_rate_limit_error(exc):
        return (
            "LLM_RATE_LIMIT",
            "OpenRouter free tier is temporarily rate-limited. Wait 30-60 seconds and "
            "retry, or set OPENROUTER_MODEL_ID=openrouter/free to auto-pick another "
            "free model. Paid OpenRouter credits raise limits.",
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

        return LiteLlm(
            model=openrouter_model_id(settings),
            api_key=api_key,
            api_base=OPENROUTER_API_BASE,
            num_retries=settings.llm_max_retries,
        )
    if not settings.gemini_api_key.strip():
        raise RuntimeError("GEMINI_API_KEY is not configured")
    return settings.gemini_model_id


def model_version_label(settings: Settings) -> str:
    provider = resolve_llm_provider(settings)
    if provider == "openrouter":
        return f"exaai-adk/{openrouter_model_id(settings)}"
    return f"exaai-adk/{settings.gemini_model_id}"


def _openrouter_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    settings: Settings,
    **kwargs: Any,
) -> Any:
    import litellm

    last_exc: BaseException | None = None
    for attempt in range(settings.llm_max_retries):
        try:
            return litellm.completion(
                model=model,
                messages=messages,
                api_key=api_key,
                api_base=OPENROUTER_API_BASE,
                **kwargs,
            )
        except Exception as exc:
            last_exc = exc
            if not is_rate_limit_error(exc) or attempt >= settings.llm_max_retries - 1:
                raise
            delay = settings.llm_retry_backoff_seconds * (2**attempt)
            logger.warning(
                "OpenRouter rate limit on %s (attempt %s/%s); retrying in %.1fs",
                model,
                attempt + 1,
                settings.llm_max_retries,
                delay,
            )
            time.sleep(delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("OpenRouter completion failed without an exception")


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
                settings=settings,
                **kwargs,
            )
        except Exception as exc:
            last_exc = exc
            if not is_rate_limit_error(exc):
                raise
            logger.warning("OpenRouter model %s rate-limited; trying fallback", model)
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

    from google import genai
    from google.genai import types

    if not settings.gemini_api_key.strip():
        raise RuntimeError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=settings.gemini_api_key)
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
    return _parse_json_response(response.text or "")
