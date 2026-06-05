"""Profile URL enrichment (Exa + security checks + cache)."""

from __future__ import annotations

import asyncio
from typing import Any

from agent.cache.url_cache import UrlCache
from agent.config import get_settings
from agent.security.allowlist import check_allowlist
from agent.security.profile_identity import ProfileTrust
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


def _content_preview(content: str, *, limit: int = 500) -> str:
    if len(content) <= limit:
        return content
    return content[:limit] + "…"


def fetch_profile_url_data(
    state: dict[str, Any],
    url: str,
    *,
    allow_untrusted: bool = False,
) -> dict[str, Any]:
    """
    Fetch and sanitize one profile URL without mutating ``enriched_contents``.

    Used by single-url and batch fetch paths so parallel batch calls stay safe.
    Untrusted URLs are never fetched (Exa); pipeline adds prompt stubs via
    ``_stub_untrusted_profile_entry`` instead.
    """
    settings = get_settings()
    allowed_urls = set(state.get("profile_urls") or [])
    if url not in allowed_urls:
        return {"ok": False, "url": url, "error": "url_not_in_candidate_list"}

    trust_by_url = state.get("profile_trust_by_url") or {}
    if not allow_untrusted and trust_by_url.get(url) == ProfileTrust.SCORING_UNTRUSTED.value:
        return {
            "ok": False,
            "url": url,
            "error": "profile_untrusted",
            "message": "URL marked scoring_untrusted; do not fetch for scoring.",
        }

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
            return {
                "ok": False,
                "url": url,
                "error": "exa_fetch_failed",
                "message": str(exc),
            }

    sanitized = sanitize_external_content(raw, url, max_chars=settings.content_token_cap)
    profile_trust = trust_by_url.get(url, ProfileTrust.SCORING_LIMITED.value)
    entry = {
        "url": url,
        "content": sanitized,
        "domain_category": allow.domain_category,
        "profile_trust": profile_trust,
        "ok": True,
    }
    return {
        "ok": True,
        "url": url,
        "domain_category": allow.domain_category,
        "profile_trust": profile_trust,
        "entry": entry,
        "content_preview": _content_preview(sanitized),
    }


def _stub_untrusted_profile_entry(url: str) -> dict[str, Any]:
    """Record an untrusted URL for scoring prompts without an Exa fetch."""
    allow = check_allowlist(url)
    return {
        "url": url,
        "content": "",
        "domain_category": allow.domain_category if allow.allowed else "unknown",
        "profile_trust": ProfileTrust.SCORING_UNTRUSTED.value,
        "ok": True,
        "skipped_fetch": True,
    }


def fetch_profile_url(state: dict[str, Any], url: str) -> dict[str, Any]:
    """
    Fetch one profile URL into ``state['enriched_contents']``.

    Returns a status dict (same shape as the ADK tool).
    """
    data = fetch_profile_url_data(state, url, allow_untrusted=False)
    if not data.get("ok"):
        return data

    enriched: list[dict[str, Any]] = list(state.get("enriched_contents") or [])
    enriched.append(data["entry"])
    state["enriched_contents"] = enriched

    return {
        "ok": True,
        "url": url,
        "domain_category": data.get("domain_category"),
        "profile_trust": data.get("profile_trust"),
    }


def _dedupe_preserve_order(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        ordered.append(url)
    return ordered


def _enriched_url_set(state: dict[str, Any]) -> set[str]:
    return {
        str(item.get("url"))
        for item in (state.get("enriched_contents") or [])
        if item.get("url")
    }


def plan_batch_profile_fetches(
    state: dict[str, Any],
    urls: list[str],
    *,
    skip_untrusted: bool = True,
) -> tuple[list[str], list[dict[str, Any]], int]:
    """
    Choose eligible URLs for a batch fetch within the session budget.

    Skips untrusted URLs, URLs not on the candidate list, and URLs already
    present in ``enriched_contents``. Caps new fetches so total unique enriched
    URLs per session does not exceed ``max_urls_per_resume``.

    Returns (eligible_urls, skipped_meta, truncated_count).
    """
    settings = get_settings()
    allowed = set(state.get("profile_urls") or [])
    trust_by_url = state.get("profile_trust_by_url") or {}
    already_enriched = _enriched_url_set(state)
    session_budget = max(0, settings.max_urls_per_resume - len(already_enriched))
    skipped: list[dict[str, Any]] = []
    eligible: list[str] = []

    for url in _dedupe_preserve_order(urls):
        if url not in allowed:
            skipped.append(
                {
                    "url": url,
                    "ok": False,
                    "error": "url_not_in_candidate_list",
                }
            )
            continue
        if url in already_enriched:
            skipped.append(
                {
                    "url": url,
                    "ok": False,
                    "error": "already_fetched",
                    "message": "URL already enriched in this session.",
                }
            )
            continue
        if skip_untrusted and trust_by_url.get(url) == ProfileTrust.SCORING_UNTRUSTED.value:
            skipped.append(
                {
                    "url": url,
                    "ok": False,
                    "error": "profile_untrusted",
                    "message": "Ignored: scoring_untrusted profile.",
                }
            )
            continue
        eligible.append(url)

    truncated = max(0, len(eligible) - session_budget)
    if session_budget <= 0:
        eligible = []
    elif truncated:
        eligible = eligible[:session_budget]

    return eligible, skipped, truncated


async def fetch_profile_urls_batch_async(
    state: dict[str, Any],
    urls: list[str],
) -> dict[str, Any]:
    """Fetch multiple profile URLs in parallel (bounded concurrency)."""
    eligible, skipped, truncated = plan_batch_profile_fetches(
        state,
        urls,
        skip_untrusted=True,
    )
    if not eligible:
        return {
            "ok": True,
            "fetched": 0,
            "skipped": skipped,
            "truncated": truncated,
            "results": [],
            "message": "No eligible URLs to fetch.",
        }

    semaphore = asyncio.Semaphore(6)

    async def _one(url: str) -> dict[str, Any]:
        async with semaphore:
            return await asyncio.to_thread(fetch_profile_url_data, state, url)

    raw_results = await asyncio.gather(*[_one(url) for url in eligible])
    enriched: list[dict[str, Any]] = list(state.get("enriched_contents") or [])
    results: list[dict[str, Any]] = []

    for item in raw_results:
        if item.get("ok"):
            enriched.append(item["entry"])
            results.append(
                {
                    "ok": True,
                    "url": item["url"],
                    "domain_category": item.get("domain_category"),
                    "profile_trust": item.get("profile_trust"),
                    "content_preview": item.get("content_preview"),
                }
            )
        else:
            results.append(item)

    state["enriched_contents"] = enriched
    fetched = sum(1 for item in results if item.get("ok"))

    return {
        "ok": True,
        "fetched": fetched,
        "skipped": skipped,
        "truncated": truncated,
        "results": results,
        "message": (
            f"Fetched {fetched} profile(s). "
            "Full sanitized content stored in session for scoring."
        ),
    }


def fetch_profile_urls_batch(state: dict[str, Any], urls: list[str]) -> dict[str, Any]:
    """Sync wrapper for batch profile fetch (ADK tools)."""
    return asyncio.run(fetch_profile_urls_batch_async(state, urls))


async def enrich_profile_urls_async(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch trusted/limited profile URLs concurrently; stub untrusted without Exa."""
    settings = get_settings()
    urls = list(state.get("profile_urls") or [])[: settings.max_urls_per_resume]
    trust_by_url = state.get("profile_trust_by_url") or {}
    enriched: list[dict[str, Any]] = list(state.get("enriched_contents") or [])
    to_fetch: list[str] = []
    results: list[dict[str, Any]] = []

    for url in urls:
        if trust_by_url.get(url) == ProfileTrust.SCORING_UNTRUSTED.value:
            entry = _stub_untrusted_profile_entry(url)
            enriched.append(entry)
            results.append(
                {
                    "ok": True,
                    "url": url,
                    "profile_trust": entry["profile_trust"],
                    "skipped_fetch": True,
                    "message": "Untrusted profile; Exa fetch skipped.",
                }
            )
        else:
            to_fetch.append(url)

    state["enriched_contents"] = enriched
    if not to_fetch:
        return results

    semaphore = asyncio.Semaphore(6)

    async def _one(url: str) -> dict[str, Any]:
        async with semaphore:
            return await asyncio.to_thread(fetch_profile_url, state, url)

    fetched = await asyncio.gather(*[_one(url) for url in to_fetch])
    results.extend(fetched)
    return results


def enrich_profile_urls(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Sync wrapper for enrichment (no running event loop)."""
    return asyncio.run(enrich_profile_urls_async(state))
