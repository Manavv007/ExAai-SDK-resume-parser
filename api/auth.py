"""API key validation for /screen."""

from __future__ import annotations

from typing import Annotated

from fastapi import Form, HTTPException, Request


def _normalize_token(value: str) -> str:
    """Strip whitespace and optional Bearer prefix (Swagger/curl vary)."""
    token = value.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token


def _extract_token(request: Request, api_key: str | None) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.strip():
        return _normalize_token(auth)

    if api_key and api_key.strip():
        return _normalize_token(api_key)

    return None


async def require_api_key(
    request: Request,
    api_key: Annotated[
        str | None,
        Form(
            description=(
                "API token from server .env API_KEYS. "
                "In Swagger UI, paste your token here (e.g. dev-local-key-change-me). "
                "Alternatively send Authorization: Bearer <token> from curl or your app."
            ),
        ),
    ] = None,
) -> None:
    """Accept Bearer header or api_key form field (form field is required for Swagger UI)."""
    from agent.config import get_settings

    settings = get_settings()
    allowed = settings.parsed_api_keys()
    if not allowed:
        raise HTTPException(status_code=503, detail="API_KEYS not configured")

    token = _extract_token(request, api_key)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Use api_key form field or Authorization: Bearer <token>",
        )
    if token not in allowed:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Must match a value in server API_KEYS (.env).",
        )
