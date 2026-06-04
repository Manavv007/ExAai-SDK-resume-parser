"""Exa AI URL content fetching."""

from __future__ import annotations

from agent.config import get_settings


def fetch_url_text(url: str) -> str:
    """Fetch page text for one URL via Exa contents API."""
    settings = get_settings()
    if not settings.exa_api_key.strip():
        raise RuntimeError("EXA_API_KEY is not configured")

    from exa_py import Exa

    client = Exa(api_key=settings.exa_api_key)
    response = client.get_contents(
        urls=[url],
        text=True,
    )

    results = getattr(response, "results", None) or []
    if not results:
        return ""

    first = results[0]
    return getattr(first, "text", None) or ""
