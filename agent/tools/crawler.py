"""Exa AI URL content fetching (batch-aware)."""

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


def fetch_url_text_batch(urls: list[str]) -> dict[str, str]:
    """Fetch page text for multiple URLs in a single Exa API call.

    Returns ``{url: text}`` — URLs with no result map to ``""``.
    """
    settings = get_settings()
    if not settings.exa_api_key.strip():
        raise RuntimeError("EXA_API_KEY is not configured")
    if not urls:
        return {}

    from exa_py import Exa

    client = Exa(api_key=settings.exa_api_key)
    response = client.get_contents(
        urls=urls,
        text=True,
    )

    results = getattr(response, "results", None) or []
    by_url: dict[str, str] = {}
    for item in results:
        url = getattr(item, "url", None) or ""
        text = getattr(item, "text", None) or ""
        if url:
            by_url[url] = text

    # Ensure every requested URL has an entry (empty string if missing)
    for url in urls:
        by_url.setdefault(url, "")
    return by_url
