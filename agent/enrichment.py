"""Profile URL enrichment (Exa + security checks + cache).

Batch-aware: all uncached URLs are fetched in a single Exa API call
instead of N separate round-trips.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent.cache.url_cache import UrlCache
from agent.config import get_settings
from agent.logging_config import trace_event
from agent.security.allowlist import check_allowlist
from agent.security.profile_identity import ProfileTrust
from agent.security.ssrf_guard import validate_url
from agent.tools.crawler import fetch_url_html_for_link_discovery, fetch_url_text, fetch_url_text_batch
from agent.tools.github_analyzer import (
    ensure_github_analysis_after_discovery,
    normalize_github_profile_url,
    normalize_github_repo_url,
    sync_github_identity,
)
from agent.tools.link_extractor import (
    extract_urls_from_html,
    extract_urls_from_text,
    is_profile_discovery_url,
    merge_url_candidates,
    normalize_url,
)
from agent.tools.portfolio_signal import is_portfolio_like_url
from agent.tools.sanitizer import sanitize_external_content

_cache: UrlCache | None = None
logger = logging.getLogger("exaai_adk.enrichment")
_MAX_DISCOVERED_LINKS = 10


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


def _build_enriched_entry(
    *,
    url: str,
    raw: str,
    trust_by_url: dict[str, str],
    allow_result: Any,
    settings: Any,
) -> dict[str, Any]:
    """Sanitize raw content and build an enriched_contents entry."""
    sanitized = sanitize_external_content(raw, url, max_chars=settings.content_token_cap)
    profile_trust = trust_by_url.get(url, ProfileTrust.SCORING_LIMITED.value)
    return {
        "url": url,
        "content": sanitized,
        "domain_category": allow_result.domain_category,
        "profile_trust": profile_trust,
        "ok": True,
    }


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
    trace_event(
        logger,
        "enrichment_url_start",
        url=url,
        allow_untrusted=allow_untrusted,
    )
    settings = get_settings()
    allowed_urls = set(state.get("profile_urls") or [])
    if url not in allowed_urls:
        trace_event(logger, "enrichment_url_rejected", url=url, reason="url_not_in_candidate_list")
        return {"ok": False, "url": url, "error": "url_not_in_candidate_list"}

    trust_by_url = state.get("profile_trust_by_url") or {}
    if not allow_untrusted and trust_by_url.get(url) == ProfileTrust.SCORING_UNTRUSTED.value:
        trace_event(logger, "enrichment_url_rejected", url=url, reason="profile_untrusted")
        return {
            "ok": False,
            "url": url,
            "error": "profile_untrusted",
            "message": "URL marked scoring_untrusted; do not fetch for scoring.",
        }

    ssrf = validate_url(url)
    if not ssrf.allowed:
        trace_event(logger, "enrichment_url_rejected", url=url, reason=ssrf.reason)
        return {"ok": False, "url": url, "error": ssrf.reason}

    allow = check_allowlist(url)
    if not allow.allowed:
        trace_event(logger, "enrichment_url_rejected", url=url, reason=allow.reason)
        return {"ok": False, "url": url, "error": allow.reason}

    cache = get_url_cache()
    cached = cache.get(url)
    if cached is not None:
        raw = cached
        trace_event(logger, "enrichment_cache_hit", url=url)
    else:
        try:
            raw = fetch_url_text(url)
            cache.set(url, raw)
            trace_event(logger, "enrichment_fetch_success", url=url, raw_chars=len(raw))
        except Exception as exc:
            trace_event(
                logger,
                "enrichment_fetch_error",
                url=url,
                reason="exa_fetch_failed",
                error=str(exc),
            )
            return {
                "ok": False,
                "url": url,
                "error": "exa_fetch_failed",
                "message": str(exc),
            }

    entry = _build_enriched_entry(
        url=url,
        raw=raw,
        trust_by_url=trust_by_url,
        allow_result=allow,
        settings=settings,
    )
    return {
        "ok": True,
        "url": url,
        "domain_category": allow.domain_category,
        "profile_trust": entry["profile_trust"],
        "entry": entry,
        "content_preview": _content_preview(entry["content"]),
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


def _failed_fetch_entry(
    url: str,
    *,
    trust_by_url: dict[str, str],
    error: str,
    message: str = "",
) -> dict[str, Any]:
    """Record a crawl attempt that did not return usable content."""
    allow = check_allowlist(url)
    entry: dict[str, Any] = {
        "url": url,
        "content": "",
        "domain_category": allow.domain_category if allow.allowed else "unknown",
        "profile_trust": trust_by_url.get(url, ProfileTrust.SCORING_LIMITED.value),
        "ok": False,
        "fetch_error": error,
    }
    if message:
        entry["fetch_message"] = message
    return entry


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
        str(item.get("url")) for item in (state.get("enriched_contents") or []) if item.get("url")
    }


def _merge_unique_urls(existing: list[str] | None, new_urls: list[str]) -> list[str]:
    merged: list[str] = list(existing or [])
    seen = {str(url) for url in merged}
    for url in new_urls:
        normalized = normalize_url(str(url or "")) or str(url or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return merged


def _candidates_include_code_platform(candidates: list[str]) -> bool:
    for candidate in candidates:
        if normalize_github_repo_url(candidate) or normalize_github_profile_url(candidate):
            return True
        lowered = candidate.lower()
        if "gitlab.com" in lowered or "bitbucket.org" in lowered:
            return True
    return False


def _collect_link_candidates(
    page_url: str,
    content: str,
    *,
    candidate_name: str = "",
    known_handles: list[str] | None = None,
    max_urls: int = 200,
) -> list[str]:
    """Gather outbound URLs from Exa text, with HTML fallback for portfolio hubs."""
    candidates = extract_urls_from_text(content, max_urls=max_urls)
    portfolio_like = is_portfolio_like_url(
        page_url,
        content,
        candidate_name=candidate_name,
        known_handles=known_handles,
    )
    if portfolio_like and not _candidates_include_code_platform(candidates):
        try:
            html = fetch_url_html_for_link_discovery(page_url)
            html_candidates = extract_urls_from_html(html, base_url=page_url, max_urls=max_urls)
            candidates = merge_url_candidates(candidates, html_candidates)
            if html_candidates:
                trace_event(
                    logger,
                    "enrichment_html_link_supplement",
                    url=page_url,
                    html_link_count=len(html_candidates),
                )
        except Exception as exc:
            trace_event(
                logger,
                "enrichment_html_link_supplement_failed",
                url=page_url,
                error=str(exc),
            )
    return candidates


def _discover_links_from_entries(
    entries: list[dict[str, Any]],
    *,
    max_links: int = _MAX_DISCOVERED_LINKS,
    candidate_name: str = "",
    known_handles: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return (github_repo_links, non_github_links) from portfolio-like pages.

    ``candidate_name`` and ``known_handles`` are forwarded to
    ``is_portfolio_like_url`` so identity-based scoring can verify that a
    crawled page belongs to this specific candidate.
    """
    discovered_github: list[str] = []
    discovered_non_github: list[str] = []
    seen_github: set[str] = set()
    seen_non_github: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("ok"):
            continue
        page_url = str(entry.get("url") or "")
        content = str(entry.get("content") or "")
        extracted_candidates = _collect_link_candidates(
            page_url,
            content,
            candidate_name=candidate_name,
            known_handles=known_handles,
        )
        has_outbound_links = any(candidate != page_url for candidate in extracted_candidates)
        if not is_portfolio_like_url(
            page_url,
            content,
            candidate_name=candidate_name,
            known_handles=known_handles,
        ) and not has_outbound_links:
            continue
        for candidate in extracted_candidates:
            repo_url = normalize_github_repo_url(candidate)
            if repo_url:
                key = repo_url.lower()
                if key not in seen_github:
                    seen_github.add(key)
                    discovered_github.append(repo_url)
                continue
            profile_url = normalize_github_profile_url(candidate)
            if profile_url:
                key = profile_url.lower()
                if key not in seen_non_github:
                    seen_non_github.add(key)
                    discovered_non_github.append(profile_url)
                continue
            normalized = normalize_url(candidate)
            if not normalized or normalized == page_url or not is_profile_discovery_url(normalized):
                continue
            key = normalized.lower()
            if key in seen_non_github:
                continue
            seen_non_github.add(key)
            discovered_non_github.append(normalized)
            if len(discovered_non_github) >= max_links:
                break
        if len(discovered_non_github) >= max_links:
            break
    return discovered_github, discovered_non_github


def _append_discovered_profile_meta(state: dict[str, Any], urls: list[str]) -> None:
    if not urls:
        return
    meta = list(state.get("profile_url_meta") or [])
    existing = {str(item.get("url") or "") for item in meta if isinstance(item, dict)}
    for url in urls:
        if url in existing:
            continue
        meta.append({"url": url, "source": "crawl_discovered", "platform": "discovered"})
        existing.add(url)
    state["profile_url_meta"] = meta


def _ensure_trust_defaults(state: dict[str, Any], urls: list[str]) -> None:
    if not urls:
        return
    trust_by_url = dict(state.get("profile_trust_by_url") or {})
    for url in urls:
        trust_by_url.setdefault(url, ProfileTrust.SCORING_LIMITED.value)
    state["profile_trust_by_url"] = trust_by_url


def _extract_discovered_links_to_state(
    state: dict[str, Any],
    *,
    source_entries: list[dict[str, Any]],
) -> tuple[list[str], list[str], dict[str, Any]]:
    """
    Discover links from crawled entries and merge into session state.

    Does not call Exa for follow-up URLs — the agent (or pipeline pre-enrich)
    decides whether to fetch discovered profile pages next.
    """
    resume_structured = state.get("resume_structured") or {}
    candidate_name: str = str(resume_structured.get("candidate_name") or "")
    github_username: str = str(state.get("github_username") or "")
    known_handles: list[str] = [h for h in [github_username] if h]

    discovered_github, discovered_non_github = _discover_links_from_entries(
        source_entries,
        candidate_name=candidate_name,
        known_handles=known_handles,
    )
    discovered_github = _merge_unique_urls(
        list(state.get("discovered_github_repo_urls") or []),
        discovered_github,
    )
    state["discovered_github_repo_urls"] = discovered_github
    github = state.get("github_repo_analyses")
    if isinstance(github, dict):
        github["discovered_github_repo_urls"] = list(discovered_github)
        state["github_repo_analyses"] = github
    if discovered_github:
        trace_event(
            logger,
            "enrichment_discovered_github_links",
            discovered_count=len(discovered_github),
        )
    sync_github_identity(state)

    if discovered_non_github:
        discovered_non_github = _filter_profile_discovery_urls(discovered_non_github)
        merged_profiles = _merge_unique_urls(
            list(state.get("profile_urls") or []),
            discovered_non_github,
        )
        state["profile_urls"] = merged_profiles
        discovered_profiles = _merge_unique_urls(
            list(state.get("discovered_profile_urls") or []),
            discovered_non_github,
        )
        state["discovered_profile_urls"] = discovered_profiles
        _append_discovered_profile_meta(state, discovered_non_github)
        _ensure_trust_defaults(state, discovered_non_github)
        sync_github_identity(state)

    meta = {
        "discovered_non_github_count": len(discovered_non_github),
        "discovered_github_count": len(discovered_github),
    }
    return discovered_github, discovered_non_github, meta


def fetch_budget_remaining(state: dict[str, Any]) -> int:
    """Unique Exa fetches still allowed this session."""
    settings = get_settings()
    return max(0, settings.max_urls_per_resume - len(_enriched_url_set(state)))


def suggested_next_profile_urls(state: dict[str, Any], *, limit: int = 10) -> list[str]:
    """Profile URLs discovered or listed but not yet enriched in session."""
    enriched = _enriched_url_set(state)
    candidates = _merge_unique_urls(
        _high_value_follow_up_urls(list(state.get("discovered_profile_urls") or []), limit=limit),
        list(state.get("profile_urls") or []),
    )
    return [url for url in candidates if url not in enriched][:limit]


def build_fetch_profiles_tool_payload(
    state: dict[str, Any],
    *,
    base: dict[str, Any],
) -> dict[str, Any]:
    """Attach session discovery + budget fields for agent orchestration."""
    payload = dict(base)
    payload.update(
        {
            "discovered_github_repo_urls": list(state.get("discovered_github_repo_urls") or []),
            "discovered_profile_urls": list(state.get("discovered_profile_urls") or []),
            "suggested_next_urls": suggested_next_profile_urls(state),
            "github_username": state.get("github_username"),
            "fetch_budget_remaining": fetch_budget_remaining(state),
        }
    )
    return payload


def _filter_profile_discovery_urls(urls: list[str]) -> list[str]:
    """Drop CDN/static asset noise before merging discovered profile URLs."""
    return [url for url in urls if is_profile_discovery_url(url)]


def _high_value_follow_up_urls(urls: list[str], *, limit: int = 3) -> list[str]:
    """Prioritize profile pages worth a follow-up Exa fetch (not GitHub repos)."""
    from agent.tools.github_analyzer import normalize_github_profile_url

    priority_hosts = (
        "linkedin.com/in/",
        "behance.net",
        "dribbble.com",
        "figma.com",
        "artstation.com",
        "scholar.google.com",
        "researchgate.net",
        "orcid.org",
    )
    ranked: list[str] = []
    seen: set[str] = set()

    def add(url: str) -> None:
        if url in seen or not is_profile_discovery_url(url):
            return
        seen.add(url)
        ranked.append(url)

    for url in urls:
        lowered = url.lower()
        if normalize_github_profile_url(url):
            add(url)
            continue
        if any(marker in lowered for marker in priority_hosts):
            add(url)

    for url in urls:
        if len(ranked) >= limit:
            break
        add(url)

    return ranked[:limit]


async def _run_discovered_link_pass(
    state: dict[str, Any],
    *,
    source_entries: list[dict[str, Any]],
    auto_follow_discovered: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Discover links from portfolio pages; optionally fetch non-GitHub links once.

    Returns (extra_results, extra_entries, discovery_meta).
    """
    _discovered_github, discovered_non_github, meta = _extract_discovered_links_to_state(
        state,
        source_entries=source_entries,
    )
    await ensure_github_analysis_after_discovery(state)
    if not discovered_non_github:
        return [], [], meta

    follow_urls: list[str] = []
    if auto_follow_discovered:
        follow_urls = _filter_profile_discovery_urls(discovered_non_github[:_MAX_DISCOVERED_LINKS])
    else:
        follow_urls = _high_value_follow_up_urls(discovered_non_github)

    if not follow_urls:
        return [], [], meta
    eligible, skipped, truncated = plan_batch_profile_fetches(
        state,
        follow_urls,
        skip_untrusted=True,
    )
    meta.update(
        {
            "discovered_skipped_count": len(skipped),
            "discovered_truncated": truncated,
            "auto_follow_mode": "all" if auto_follow_discovered else "high_value",
            "follow_up_urls": follow_urls,
        }
    )
    if not eligible:
        return [], [], meta

    settings = get_settings()
    trust_by_url = state.get("profile_trust_by_url") or {}
    cache = get_url_cache()
    url_to_raw, _ = await asyncio.to_thread(_fetch_batch_with_cache, eligible, cache)
    batch_results = _build_batch_results(eligible, url_to_raw, trust_by_url, settings)
    entries = [
        item.get("entry")
        for item in batch_results
        if isinstance(item.get("entry"), dict)
    ]
    trace_event(
        logger,
        "enrichment_discovered_links_fetched",
        discovered_non_github_count=len(discovered_non_github),
        fetched=len([item for item in batch_results if item.get("ok")]),
        skipped_count=len(skipped),
        truncated=truncated,
    )
    sync_github_identity(state)
    return (
        [{k: v for k, v in item.items() if k != "entry"} for item in batch_results],
        entries,
        meta,
    )


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


# ---------------------------------------------------------------------------
# Batch fetch helpers
# ---------------------------------------------------------------------------


def _classify_and_check_url(
    url: str,
    *,
    allowed_urls: set[str],
    trust_by_url: dict[str, str],
    skip_untrusted: bool,
) -> dict[str, Any]:
    """Run SSRF + allowlist checks for a single URL. Returns a status dict."""
    if url not in allowed_urls:
        trace_event(
            logger,
            "enrichment_batch_rejected",
            url=url,
            reason="url_not_in_candidate_list",
        )
        return {"ok": False, "url": url, "error": "url_not_in_candidate_list"}

    if skip_untrusted and trust_by_url.get(url) == ProfileTrust.SCORING_UNTRUSTED.value:
        trace_event(logger, "enrichment_batch_rejected", url=url, reason="profile_untrusted")
        return {
            "ok": False,
            "url": url,
            "error": "profile_untrusted",
            "message": "URL marked scoring_untrusted; do not fetch for scoring.",
        }

    ssrf = validate_url(url)
    if not ssrf.allowed:
        trace_event(logger, "enrichment_batch_rejected", url=url, reason=ssrf.reason)
        return {"ok": False, "url": url, "error": ssrf.reason}

    allow = check_allowlist(url)
    if not allow.allowed:
        trace_event(logger, "enrichment_batch_rejected", url=url, reason=allow.reason)
        return {"ok": False, "url": url, "error": allow.reason}

    trace_event(logger, "enrichment_batch_eligible", url=url, category=allow.domain_category)
    return {"ok": True, "url": url, "allow": allow}


def _fetch_batch_with_cache(
    urls: list[str],
    cache: UrlCache,
) -> tuple[dict[str, str], list[str]]:
    """Separate cached vs uncached URLs; batch-fetch only the uncached ones.

    Returns (url_to_raw_text, list_of_urls_that_were_cached).
    """
    cached_urls: list[str] = []
    uncached_urls: list[str] = []
    url_to_raw: dict[str, str] = {}

    for url in urls:
        cached = cache.get(url)
        if cached is not None:
            url_to_raw[url] = cached
            cached_urls.append(url)
        else:
            uncached_urls.append(url)

    if uncached_urls:
        try:
            fetched = fetch_url_text_batch(uncached_urls)
            for url, text in fetched.items():
                url_to_raw[url] = text
                if text:
                    cache.set(url, text)
        except Exception:
            # If batch fails entirely, mark all uncached as failed
            for url in uncached_urls:
                url_to_raw[url] = ""

    return url_to_raw, cached_urls


def _build_batch_results(
    eligible_urls: list[str],
    url_to_raw: dict[str, str],
    trust_by_url: dict[str, str],
    settings: Any,
) -> list[dict[str, Any]]:
    """Build result dicts for each URL from raw fetched text."""
    results: list[dict[str, Any]] = []
    for url in eligible_urls:
        raw = url_to_raw.get(url, "")
        if not raw:
            results.append(
                {
                    "ok": False,
                    "url": url,
                    "error": "exa_fetch_failed",
                    "message": "Empty response from Exa.",
                    "entry": _failed_fetch_entry(
                        url,
                        trust_by_url=trust_by_url,
                        error="exa_fetch_failed",
                        message="Empty response from Exa.",
                    ),
                }
            )
            continue

        allow = check_allowlist(url)
        entry = _build_enriched_entry(
            url=url,
            raw=raw,
            trust_by_url=trust_by_url,
            allow_result=allow,
            settings=settings,
        )
        results.append(
            {
                "ok": True,
                "url": url,
                "domain_category": allow.domain_category,
                "profile_trust": entry["profile_trust"],
                "content_preview": _content_preview(entry["content"]),
                "entry": entry,
            }
        )
    return results


async def fetch_profile_urls_batch_async(
    state: dict[str, Any],
    urls: list[str],
    *,
    auto_follow_discovered: bool = False,
) -> dict[str, Any]:
    """Fetch multiple profile URLs in a single batch Exa API call.

    Security checks (SSRF, allowlist) still run per-URL before the batch
    fetch. Cached URLs are served from SQLite without any API call.

    When ``auto_follow_discovered`` is False (agent mode default), discovered
    links are stored in session for a follow-up ``fetch_profiles`` call.
    """
    trace_event(
        logger,
        "enrichment_batch_start",
        requested_count=len(urls),
    )
    eligible, skipped, truncated = plan_batch_profile_fetches(
        state,
        urls,
        skip_untrusted=True,
    )
    if not eligible:
        trace_event(
            logger,
            "enrichment_batch_noop",
            skipped_count=len(skipped),
            truncated=truncated,
        )
        return build_fetch_profiles_tool_payload(
            state,
            base={
                "ok": True,
                "fetched": 0,
                "skipped": skipped,
                "truncated": truncated,
                "results": [],
                "discovered_non_github_count": 0,
                "discovered_github_count": len(state.get("discovered_github_repo_urls") or []),
                "message": "No eligible URLs to fetch.",
            },
        )

    settings = get_settings()
    trust_by_url = state.get("profile_trust_by_url") or {}
    cache = get_url_cache()

    # Run the batch fetch in a thread to avoid blocking the event loop
    url_to_raw, _cached_urls = await asyncio.to_thread(_fetch_batch_with_cache, eligible, cache)

    results = _build_batch_results(eligible, url_to_raw, trust_by_url, settings)

    # Merge successful and failed crawl attempts into state for sources_crawled.
    enriched: list[dict[str, Any]] = list(state.get("enriched_contents") or [])
    for item in results:
        entry = item.get("entry")
        if isinstance(entry, dict):
            enriched.append(entry)
    state["enriched_contents"] = enriched

    # Depth-1 discovery from portfolio pages (agent controls follow-up fetches).
    extra_results, extra_entries, discovery_meta = await _run_discovered_link_pass(
        state,
        source_entries=[
            item.get("entry")
            for item in results
            if isinstance(item.get("entry"), dict)
        ],
        auto_follow_discovered=auto_follow_discovered,
    )
    await ensure_github_analysis_after_discovery(state)
    if extra_entries:
        enriched.extend(extra_entries)
        state["enriched_contents"] = enriched
        results.extend(
            {
                "ok": bool(item.get("ok")),
                "url": item.get("url"),
                "domain_category": item.get("domain_category"),
                "profile_trust": item.get("profile_trust"),
                "content_preview": item.get("content_preview"),
            }
            for item in extra_results
        )

    fetched = sum(1 for item in results if item.get("ok"))
    trace_event(
        logger,
        "enrichment_batch_end",
        fetched=fetched,
        eligible_count=len(eligible),
        skipped_count=len(skipped),
        truncated=truncated,
        discovered_non_github_count=int(discovery_meta.get("discovered_non_github_count") or 0),
        discovered_github_count=int(discovery_meta.get("discovered_github_count") or 0),
        fetch_budget_remaining=fetch_budget_remaining(state),
        github_username=state.get("github_username"),
    )

    return build_fetch_profiles_tool_payload(
        state,
        base={
            "ok": True,
            "fetched": fetched,
            "skipped": skipped,
            "truncated": truncated,
            "results": [{k: v for k, v in item.items() if k != "entry"} for item in results],
            "discovered_non_github_count": int(
                discovery_meta.get("discovered_non_github_count") or 0
            ),
            "discovered_github_count": int(discovery_meta.get("discovered_github_count") or 0),
            "message": (
                f"Fetched {fetched} profile(s). Full sanitized content stored in session "
                "for scoring."
            ),
        },
    )


def fetch_profile_urls_batch(
    state: dict[str, Any],
    urls: list[str],
    *,
    auto_follow_discovered: bool = False,
) -> dict[str, Any]:
    """Sync wrapper for batch profile fetch (ADK tools)."""
    return asyncio.run(
        fetch_profile_urls_batch_async(
            state,
            urls,
            auto_follow_discovered=auto_follow_discovered,
        )
    )


async def enrich_profile_urls_async(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch trusted/limited profile URLs in one batch; stub untrusted without Exa.

    All uncached URLs are fetched in a single Exa API call instead of N
    separate round-trips.
    """
    settings = get_settings()
    urls = list(state.get("profile_urls") or [])[: settings.max_urls_per_resume]
    trust_by_url = state.get("profile_trust_by_url") or {}
    enriched: list[dict[str, Any]] = list(state.get("enriched_contents") or [])
    results: list[dict[str, Any]] = []

    # Separate untrusted (stub) from fetchable URLs
    to_fetch: list[str] = []
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

    # Batch-fetch all URLs in one API call (cached URLs are free)
    cache = get_url_cache()
    url_to_raw, _cached_urls = await asyncio.to_thread(_fetch_batch_with_cache, to_fetch, cache)

    batch_results = _build_batch_results(to_fetch, url_to_raw, trust_by_url, settings)

    for item in batch_results:
        entry = item.get("entry")
        if isinstance(entry, dict):
            enriched.append(entry)
    state["enriched_contents"] = enriched
    results.extend({k: v for k, v in item.items() if k != "entry"} for item in batch_results)

    extra_results, extra_entries, _ = await _run_discovered_link_pass(
        state,
        source_entries=[
            item.get("entry")
            for item in batch_results
            if isinstance(item.get("entry"), dict)
        ],
        auto_follow_discovered=True,
    )
    if extra_entries:
        enriched.extend(extra_entries)
        state["enriched_contents"] = enriched
        results.extend(extra_results)
    return results


def enrich_profile_urls(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Sync wrapper for enrichment (no running event loop)."""
    return asyncio.run(enrich_profile_urls_async(state))
