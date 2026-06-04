"""Profile URL enrichment (Exa + security checks + cache)."""

from __future__ import annotations

import asyncio
from typing import Any

from agent.cache.url_cache import UrlCache
from agent.config import get_settings
from agent.security.allowlist import check_allowlist
from agent.security.ssrf_guard import validate_url
from agent.tools.crawler import fetch_url_text
from agent.tools.sanitizer import sanitize_external_content

_cache: UrlCache | None = None


def get_url_cache() -> UrlCache:
    global _cache
    if _cache is None:
        _cache = UrlCache()
    return _cache


class _StateView:
    """Minimal dict-like state for shared fetch logic with ADK tools."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.state = data


def fetch_profile_url(state: dict[str, Any], url: str) -> dict[str, Any]:
    """
    Fetch one profile URL into ``state['enriched_contents']``.

    Returns a status dict (same shape as the ADK tool).
    """
    settings = get_settings()
    allowed_urls = set(state.get("profile_urls") or [])
    if url not in allowed_urls:
        return {"ok": False, "url": url, "error": "url_not_in_candidate_list"}

    ssrf = validate_url(url)
    if not ssrf.allowed:
        return {"ok": False, "url": url, "error": ssrf.reason}

    allow = check_allowlist(url)
    if not allow.allowed:
        return {"ok": False, "url": url, "error": allow.reason}

    cache = get_url_cache()
    cached = cache.get(url)
    if cached is not None:
        raw = cached
    else:
        try:
            raw = fetch_url_text(url)
            cache.set(url, raw)
        except Exception as exc:
            return {"ok": False, "url": url, "error": "exa_fetch_failed", "message": str(exc)}

    sanitized = sanitize_external_content(raw, url, max_chars=settings.content_token_cap)
    trust_by_url = state.get("profile_trust_by_url") or {}
    profile_trust = trust_by_url.get(url, "scoring_limited")
    enriched: list[dict[str, Any]] = list(state.get("enriched_contents") or [])
    enriched.append(
        {
            "url": url,
            "content": sanitized,
            "domain_category": allow.domain_category,
            "profile_trust": profile_trust,
            "ok": True,
        }
    )
    state["enriched_contents"] = enriched

    return {
        "ok": True,
        "url": url,
        "domain_category": allow.domain_category,
    }


async def enrich_profile_urls_async(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch all candidate profile URLs concurrently (bounded)."""
    settings = get_settings()
    urls = list(state.get("profile_urls") or [])[: settings.max_urls_per_resume]
    semaphore = asyncio.Semaphore(6)

    async def _one(url: str) -> dict[str, Any]:
        async with semaphore:
            return await asyncio.to_thread(fetch_profile_url, state, url)

    if not urls:
        return []
    return await asyncio.gather(*[_one(url) for url in urls])


def enrich_profile_urls(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Sync wrapper for enrichment (no running event loop)."""
    return asyncio.run(enrich_profile_urls_async(state))
