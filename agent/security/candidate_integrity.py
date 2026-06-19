"""Candidate link integrity scoring — informational only; does not affect fit score."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlparse

from agent.security.profile_identity import slug_lists_plausibly_same_person, slug_tokens_from_url
from agent.tools.github_analyzer import normalize_github_profile_url, normalize_github_repo_url
from agent.tools.link_extractor import (
    extract_urls_from_text,
    is_profile_discovery_url,
    normalize_url,
)

SignalStatus = Literal["pass", "fail", "warn", "insufficient_data"]
IntegrityIndication = Literal["good", "bad", "not_enough_evidence"]

_SIGNAL_IDS = (
    "github_account_timeline",
    "linkedin_contact_links",
    "github_profile_readme_links",
)

_CONTACT_SECTION_RE = re.compile(
    r"(?is)(contact|website|portfolio|connect|links?|social)"
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

_PERSONAL_PROFILE_HOSTS = (
    "github.com",
    "linkedin.com",
    "behance.net",
    "dribbble.com",
    "gitlab.com",
    "kaggle.com",
)


def _parse_iso_datetime(value: str) -> datetime | None:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    try:
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _status_to_indication(status: SignalStatus) -> IntegrityIndication:
    if status == "pass":
        return "good"
    if status in ("fail", "warn"):
        return "bad"
    return "not_enough_evidence"


def _platform_bucket(url: str) -> str | None:
    normalized = normalize_url(url)
    if not normalized:
        return None
    host = (urlparse(normalized).netloc or "").lower().replace("www.", "")
    if host.endswith("github.com"):
        if normalize_github_repo_url(normalized):
            return f"github_repo:{normalized.rstrip('/').lower()}"
        profile = normalize_github_profile_url(normalized)
        if profile:
            return f"github_profile:{profile.rstrip('/').lower()}"
        return None
    if "linkedin.com" in host:
        parts = [p for p in urlparse(normalized).path.split("/") if p]
        if parts and parts[0].lower() == "in" and len(parts) > 1:
            return f"linkedin:{parts[1].lower()}"
        return None
    for marker in ("behance.net", "dribbble.com", "gitlab.com", "kaggle.com"):
        if host.endswith(marker) or host == marker:
            path = urlparse(normalized).path.strip("/").split("/")[0]
            if path:
                return f"{marker}:{path.lower()}"
            return f"{marker}:{normalized.rstrip('/').lower()}"
    if is_profile_discovery_url(normalized):
        return f"other:{normalized.rstrip('/').lower()}"
    return None


def _resume_personal_profile_urls(profile_urls: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in profile_urls:
        normalized = normalize_url(str(raw or ""))
        if not normalized or normalized in seen:
            continue
        if normalize_github_repo_url(normalized):
            continue
        bucket = _platform_bucket(normalized)
        if bucket is None:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _extract_profile_urls_from_text(text: str, *, contact_biased: bool = False) -> list[str]:
    if not text.strip():
        return []
    source = text
    if contact_biased:
        lines = text.splitlines()
        picked: list[str] = []
        for idx, line in enumerate(lines):
            if _CONTACT_SECTION_RE.search(line):
                picked.extend(lines[idx : idx + 8])
        if picked:
            source = "\n".join(picked)
    urls = extract_urls_from_text(source, max_urls=40)
    out: list[str] = []
    seen: set[str] = set()
    for url in urls:
        normalized = normalize_url(url)
        if not normalized or not is_profile_discovery_url(normalized):
            continue
        if normalize_github_repo_url(normalized):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def compare_url_sets(resume_urls: list[str], platform_urls: list[str]) -> dict[str, Any]:
    resume_buckets = {_platform_bucket(url): url for url in resume_urls if _platform_bucket(url)}
    platform_buckets = {
        _platform_bucket(url): url for url in platform_urls if _platform_bucket(url)
    }
    overlap: list[str] = []
    resume_only: list[str] = []
    platform_only: list[str] = []
    conflicting: list[dict[str, str]] = []

    resume_by_host: dict[str, list[tuple[str, str]]] = {}
    platform_by_host: dict[str, list[tuple[str, str]]] = {}
    for bucket, url in resume_buckets.items():
        host = bucket.split(":", 1)[0]
        resume_by_host.setdefault(host, []).append((bucket, url))
    for bucket, url in platform_buckets.items():
        host = bucket.split(":", 1)[0]
        platform_by_host.setdefault(host, []).append((bucket, url))

    matched_resume: set[str] = set()
    matched_platform: set[str] = set()
    for host, resume_items in resume_by_host.items():
        platform_items = platform_by_host.get(host, [])
        if not platform_items:
            continue
        resume_slugs = [slug_tokens_from_url(url) for _, url in resume_items]
        platform_slugs = [slug_tokens_from_url(url) for _, url in platform_items]
        if any(
            bucket in platform_buckets and bucket in resume_buckets
            for bucket, _ in resume_items
        ):
            for bucket, url in resume_items:
                if bucket in platform_buckets:
                    overlap.append(url)
                    matched_resume.add(bucket)
                    matched_platform.add(bucket)
            continue
        if slug_lists_plausibly_same_person(resume_slugs + platform_slugs):
            for bucket, url in resume_items:
                overlap.append(url)
                matched_resume.add(bucket)
            for bucket, url in platform_items:
                matched_platform.add(bucket)
            continue
        if len(resume_items) == 1 and len(platform_items) == 1:
            conflicting.append(
                {
                    "platform": host,
                    "resume_url": resume_items[0][1],
                    "platform_url": platform_items[0][1],
                }
            )

    for bucket, url in resume_buckets.items():
        if bucket not in matched_resume:
            resume_only.append(url)
    for bucket, url in platform_buckets.items():
        if bucket not in matched_platform:
            platform_only.append(url)

    return {
        "overlap": overlap,
        "resume_only": resume_only,
        "platform_only": platform_only,
        "conflicting": conflicting,
    }


def parse_resume_timeline_anchor(resume_structured: dict[str, Any]) -> datetime | None:
    years: list[int] = []
    for block in list(resume_structured.get("education") or []) + list(
        resume_structured.get("experience_highlights") or []
    ):
        for match in _YEAR_RE.finditer(str(block)):
            years.append(int(match.group(0)))
    exp_years = resume_structured.get("experience_years")
    if isinstance(exp_years, int) and exp_years > 0:
        years.append(datetime.now().year - exp_years)
    if not years:
        return None
    return datetime(min(years), 1, 1)


def _signal(
    signal_id: str,
    status: SignalStatus,
    *,
    evidence: str = "",
    source_urls: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "signal_id": signal_id,
        "indication": _status_to_indication(status),
        "evidence": evidence[:500],
        "source_urls": list(source_urls or [])[:10],
    }


def _eval_github_account_timeline(github_repo_analyses: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(github_repo_analyses, dict):
        return _signal(
            "github_account_timeline",
            "insufficient_data",
            evidence="No GitHub analysis.",
        )
    user_profile = dict(github_repo_analyses.get("user_profile") or {})
    if not user_profile.get("html_url"):
        username = str(
            github_repo_analyses.get("username") or user_profile.get("login") or ""
        ).strip()
        if username:
            user_profile["html_url"] = f"https://github.com/{username}"
    timeline = github_repo_analyses.get("activity_timeline") or {}
    account_created = _parse_iso_datetime(str(user_profile.get("created_at") or ""))
    earliest_activity = _parse_iso_datetime(str(timeline.get("earliest_activity_at") or ""))
    if not account_created:
        return _signal(
            "github_account_timeline",
            "insufficient_data",
            evidence="GitHub account creation date unavailable.",
            source_urls=[str(user_profile.get("html_url") or "")] if user_profile else [],
        )
    if not earliest_activity:
        return _signal(
            "github_account_timeline",
            "insufficient_data",
            evidence="No resume-listed GitHub repo activity timeline available.",
            source_urls=[str(user_profile.get("html_url") or "")],
        )
    if account_created > earliest_activity:
        return _signal(
            "github_account_timeline",
            "fail",
            evidence=(
                f"account_created={user_profile.get('created_at')}; "
                f"earliest_repo_activity={timeline.get('earliest_activity_at')}"
            ),
            source_urls=[str(user_profile.get("html_url") or "")],
        )
    return _signal(
        "github_account_timeline",
        "pass",
        evidence=(
            f"GitHub account predates earliest resume-listed repo activity "
            f"({timeline.get('earliest_activity_at')})."
        ),
        source_urls=[str(user_profile.get("html_url") or "")],
    )


def _linkedin_enriched_entry(enriched_contents: list[dict[str, Any]]) -> dict[str, Any] | None:
    for entry in enriched_contents:
        if not isinstance(entry, dict) or not entry.get("ok", True):
            continue
        url = str(entry.get("url") or "").lower()
        if "linkedin.com/in/" in url:
            return entry
    return None


def _eval_linkedin_contact_links(
    resume_urls: list[str],
    enriched_contents: list[dict[str, Any]],
) -> dict[str, Any]:
    entry = _linkedin_enriched_entry(enriched_contents)
    if entry is None:
        return _signal(
            "linkedin_contact_links",
            "insufficient_data",
            evidence="LinkedIn profile was not crawled; contact links unavailable.",
        )
    content = str(entry.get("content") or "")
    linkedin_urls = _extract_profile_urls_from_text(content, contact_biased=True)
    if not linkedin_urls:
        linkedin_urls = _extract_profile_urls_from_text(content, contact_biased=False)
    comparison = compare_url_sets(resume_urls, linkedin_urls)
    source = [str(entry.get("url") or "")]
    if comparison["conflicting"]:
        return _signal(
            "linkedin_contact_links",
            "fail",
            evidence=(
                "LinkedIn contact links conflict with resume profile URLs: "
                + "; ".join(
                    (
                        f"{item['platform']} resume={item['resume_url']} "
                        f"linkedin={item['platform_url']}"
                    )
                    for item in comparison["conflicting"][:2]
                )
            ),
            source_urls=source,
        )
    if comparison["overlap"]:
        return _signal(
            "linkedin_contact_links",
            "pass",
            evidence="LinkedIn contact/profile links corroborate resume URLs.",
            source_urls=source + comparison["overlap"][:3],
        )
    if linkedin_urls and resume_urls and not comparison["overlap"]:
        return _signal(
            "linkedin_contact_links",
            "fail",
            evidence="LinkedIn lists personal profile links that do not match the resume.",
            source_urls=source + linkedin_urls[:3],
        )
    if linkedin_urls and comparison["platform_only"]:
        return _signal(
            "linkedin_contact_links",
            "warn",
            evidence="Partial overlap between LinkedIn links and resume profiles.",
            source_urls=source,
        )
    return _signal(
        "linkedin_contact_links",
        "insufficient_data",
        evidence="No outbound personal profile links found on LinkedIn page text.",
        source_urls=source,
    )


def _github_surfaces(github_repo_analyses: dict[str, Any] | None) -> tuple[list[str], str]:
    urls: list[str] = []
    bio_blob = ""
    if not isinstance(github_repo_analyses, dict):
        return urls, bio_blob
    user_profile = github_repo_analyses.get("user_profile") or {}
    bio_blob = " ".join(
        str(user_profile.get(key) or "")
        for key in ("bio", "blog", "html_url")
    )
    readme = str(github_repo_analyses.get("profile_readme") or "")
    if readme:
        bio_blob = f"{bio_blob}\n{readme}"
        urls.extend(_extract_profile_urls_from_text(readme))
    urls.extend(_extract_profile_urls_from_text(bio_blob))
    for analysis in github_repo_analyses.get("repo_analyses") or []:
        if not isinstance(analysis, dict):
            continue
        for sample in analysis.get("code_samples") or []:
            if isinstance(sample, str) and sample.lower().startswith("readme preview:"):
                urls.extend(_extract_profile_urls_from_text(sample))
    return urls, bio_blob


def _eval_github_profile_readme_links(
    resume_urls: list[str],
    github_repo_analyses: dict[str, Any] | None,
) -> dict[str, Any]:
    github_urls, bio_blob = _github_surfaces(github_repo_analyses)
    user_profile = (
        dict((github_repo_analyses or {}).get("user_profile") or {})
        if isinstance(github_repo_analyses, dict)
        else {}
    )
    if not user_profile.get("html_url"):
        username = str(
            (github_repo_analyses or {}).get("username") or user_profile.get("login") or ""
        ).strip()
        if username:
            user_profile["html_url"] = f"https://github.com/{username}"
    source = [user_profile["html_url"]] if user_profile.get("html_url") else []
    if not github_urls and not bio_blob.strip():
        return _signal(
            "github_profile_readme_links",
            "insufficient_data",
            evidence="GitHub profile README/bio unavailable.",
            source_urls=source,
        )
    comparison = compare_url_sets(resume_urls, github_urls)
    if comparison["conflicting"]:
        return _signal(
            "github_profile_readme_links",
            "fail",
            evidence="GitHub profile/README links conflict with resume profile URLs.",
            source_urls=source + github_urls[:3],
        )
    if comparison["overlap"]:
        return _signal(
            "github_profile_readme_links",
            "pass",
            evidence="GitHub profile/README links corroborate resume portfolio URLs.",
            source_urls=source + comparison["overlap"][:3],
        )
    if github_urls:
        return _signal(
            "github_profile_readme_links",
            "warn",
            evidence="GitHub profile surfaces list links that only partially match the resume.",
            source_urls=source + github_urls[:3],
        )
    return _signal(
        "github_profile_readme_links",
        "insufficient_data",
        evidence="No outbound profile links found in GitHub profile surfaces.",
        source_urls=source,
    )


def _indicators_from_signals(signals: list[dict[str, Any]]) -> dict[str, IntegrityIndication]:
    by_id = {
        str(signal.get("signal_id") or ""): signal.get("indication")
        for signal in signals
        if isinstance(signal, dict)
    }
    out: dict[str, IntegrityIndication] = {}
    for signal_id in _SIGNAL_IDS:
        raw = by_id.get(signal_id, "not_enough_evidence")
        out[signal_id] = (
            raw if raw in ("good", "bad", "not_enough_evidence") else "not_enough_evidence"
        )
    return out


def compute_overall_integrity(indicators: dict[str, IntegrityIndication]) -> IntegrityIndication:
    """
    Roll up per-signal integrity into one overall indication.

    - Any signal ``bad`` → overall ``bad``.
    - All signals ``good`` → overall ``good``.
    - Mix of ``good`` and ``not_enough_evidence`` (no ``bad``) → overall ``good``.
    - All ``not_enough_evidence`` → overall ``not_enough_evidence``.
    """
    per_signal = [indicators.get(signal_id, "not_enough_evidence") for signal_id in _SIGNAL_IDS]
    if any(value == "bad" for value in per_signal):
        return "bad"
    if all(value == "good" for value in per_signal):
        return "good"
    if any(value == "good" for value in per_signal):
        return "good"
    return "not_enough_evidence"


def _overall_reasoning_phrase(overall: IntegrityIndication) -> str:
    if overall == "good":
        return "overall: good."
    if overall == "bad":
        return "overall: issue detected."
    return "overall: not enough evidence."


def _build_reasoning(signals: list[dict[str, Any]], *, overall: IntegrityIndication) -> str:
    parts: list[str] = [_overall_reasoning_phrase(overall)]
    for signal in signals:
        signal_id = str(signal.get("signal_id") or "")
        indication = str(signal.get("indication") or "not_enough_evidence")
        if indication == "good":
            parts.append(f"{signal_id}: good.")
        elif indication == "bad":
            parts.append(f"{signal_id}: issue detected.")
        else:
            parts.append(f"{signal_id}: not enough evidence.")
    text = " ".join(parts)
    return text[:500]


def _indicators_from_signals(signals: list[dict[str, Any]]) -> dict[str, IntegrityIndication]:
    by_id = {
        str(signal.get("signal_id") or ""): signal.get("indication")
        for signal in signals
        if isinstance(signal, dict)
    }
    out: dict[str, IntegrityIndication] = {}
    for signal_id in _SIGNAL_IDS:
        raw = by_id.get(signal_id, "not_enough_evidence")
        out[signal_id] = (
            raw if raw in ("good", "bad", "not_enough_evidence") else "not_enough_evidence"
        )
    return out


def compute_candidate_integrity(
    *,
    profile_urls: list[str],
    enriched_contents: list[dict[str, Any]],
    github_repo_analyses: dict[str, Any] | None,
    resume_structured: dict[str, Any] | None = None,
    profile_trust_by_url: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return integrity indicators, reasoning, and per-signal evidence.

    Does not affect fit score.
    """
    _ = profile_trust_by_url  # reserved for future trust-weighted signals
    resume_structured = resume_structured if isinstance(resume_structured, dict) else {}
    resume_profiles = _resume_personal_profile_urls(list(profile_urls or []))
    enriched = [entry for entry in (enriched_contents or []) if isinstance(entry, dict)]

    signals = [
        _eval_github_account_timeline(github_repo_analyses),
        _eval_linkedin_contact_links(resume_profiles, enriched),
        _eval_github_profile_readme_links(resume_profiles, github_repo_analyses),
    ]

    indicators = _indicators_from_signals(signals)
    overall = compute_overall_integrity(indicators)
    indicators["overall"] = overall
    reasoning = _build_reasoning(signals, overall=overall)
    user_profile = (
        (github_repo_analyses or {}).get("user_profile") or {}
        if isinstance(github_repo_analyses, dict)
        else {}
    )
    return {
        "indicators": indicators,
        "reasoning": reasoning,
        "signals": signals,
        "github_user_fetched": bool(user_profile.get("created_at")),
        "resume_timeline_anchor": (
            parse_resume_timeline_anchor(resume_structured).isoformat()
            if parse_resume_timeline_anchor(resume_structured)
            else None
        ),
    }
