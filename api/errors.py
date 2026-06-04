"""Map pipeline exceptions to HTTP error responses."""

from __future__ import annotations


def screening_error_from_exception(exc: BaseException) -> tuple[int, str, str]:
    """
    Return (http_status, error_code, user_message) for a pipeline failure.
    """
    try:
        from google.genai.errors import ClientError
    except ImportError:
        ClientError = None  # type: ignore[misc, assignment]

    if ClientError is not None and isinstance(exc, ClientError):
        status = getattr(exc, "code", None) or 500
        if status == 429:
            return (
                503,
                "LLM_RATE_LIMIT",
                "Gemini API quota exceeded. Check billing at https://ai.google.dev "
                "or wait and retry. You can also try GEMINI_MODEL_ID=gemini-2.5-flash.",
            )
        if status in {401, 403}:
            return (
                503,
                "LLM_AUTH_ERROR",
                "Gemini API key rejected. Verify GEMINI_API_KEY in .env.",
            )
        return (
            503,
            "LLM_ERROR",
            f"Gemini API error ({status}). Check server logs for details.",
        )

    if isinstance(exc, RuntimeError) and "GEMINI_API_KEY" in str(exc):
        return (503, "LLM_NOT_CONFIGURED", str(exc))

    return (500, "INTERNAL_ERROR", "Screening failed. Check server logs for details.")
