"""Role-aware portfolio verification and deterministic scoring guardrails."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal
from urllib.parse import urlparse

from agent.tools.link_extractor import normalize_url

logger = logging.getLogger("exaai_adk.portfolio_signal")

RoleCategory = Literal[
    "software_engineering",
    "aiml",
    "data_science",
    "design",
    "research_academic",
    "non_portfolio",
]

VALID_ROLE_CATEGORIES: frozenset[str] = frozenset(
    {
        "software_engineering",
        "aiml",
        "data_science",
        "design",
        "research_academic",
        "non_portfolio",
    }
)

SignalStrength = Literal[
    "strong",
    "strong_personal_website",
    "weak",
    "none",
    "not_applicable",
]

_HARD_CAP_ROLE_CATEGORIES: frozenset[str] = frozenset(
    {"software_engineering", "aiml", "design"}
)
_NONE_SIGNAL_HARD_CAP = 75

_GENERIC_DOMAINS: frozenset[str] = frozenset(
    {
        "gmail.com",
        "googlemail.com",
        "yahoo.com",
        "outlook.com",
        "hotmail.com",
        "icloud.com",
        "protonmail.com",
        "proton.me",
        "linkedin.com",
        "twitter.com",
        "x.com",
        "facebook.com",
        "instagram.com",
        "medium.com",
        "youtube.com",
        "youtu.be",
        "drive.google.com",
        "dropbox.com",
        "s3.amazonaws.com",
        "linktr.ee",
        "reddit.com",
        "wikipedia.org",
        "github.com",
        "gitlab.com",
        "bitbucket.org",
        "kaggle.com",
        "behance.net",
        "dribbble.com",
        "figma.com",
        "scholar.google.com",
        "researchgate.net",
        "orcid.org",
        "arxiv.org",
    }
)

_ROLE_CONFIGS: dict[str, dict[str, Any]] = {
    "software_engineering": {
        "required_platforms": ("github.com", "gitlab.com", "bitbucket.org"),
        "penalty_label": "No verifiable code portfolio (GitHub/GitLab/Bitbucket)",
        "base_penalty": 15,
    },
    "aiml": {
        "required_platforms": ("github.com", "gitlab.com", "kaggle.com"),
        "penalty_label": "No verifiable machine learning portfolio (GitHub/Kaggle)",
        "base_penalty": 15,
    },
    "data_science": {
        "required_platforms": ("github.com", "kaggle.com"),
        "penalty_label": "No verifiable data analysis portfolio (Kaggle/GitHub)",
        "base_penalty": 10,
    },
    "design": {
        "required_platforms": ("behance.net", "dribbble.com", "figma.com"),
        "penalty_label": "No verifiable design portfolio (Behance/Dribbble/Figma)",
        "base_penalty": 15,
    },
    "research_academic": {
        "required_platforms": (
            "scholar.google.com",
            "researchgate.net",
            "arxiv.org",
            "orcid.org",
        ),
        "penalty_label": "No verifiable research profile (Google Scholar/ORCID/ResearchGate)",
        "base_penalty": 10,
    },
    "non_portfolio": {
        "required_platforms": (),
        "penalty_label": "No portfolio requirements for this role category",
        "base_penalty": 0,
    },
}

_CRAWL_ERROR_SIGNATURES: tuple[str, ...] = (
    "cloudflare",
    "enable javascript",
    "access denied",
    "403 forbidden",
    "robot check",
    "captcha",
    "security challenge",
    "404 not found",
    "page not found",
    "under maintenance",
    "rate limit",
    "too many requests",
)

_NON_PORTFOLIO_TITLE_HINTS: tuple[str, ...] = (
    "human resources",
    " hr ",
    "recruiter",
    "talent acquisition",
    "sales",
    "account executive",
    "business development",
    "project manager",
    "program manager",
    "customer success",
    "operations manager",
    "office manager",
    "executive assistant",
)

_AIML_HINTS: tuple[str, ...] = (
    "machine learning",
    "ml engineer",
    "ai engineer",
    "deep learning",
    "llm",
    "nlp engineer",
    "computer vision",
)

_DATA_SCIENCE_HINTS: tuple[str, ...] = (
    "data scientist",
    "data science",
    "data analyst",
    "analytics engineer",
    "business intelligence",
)

_DESIGN_HINTS: tuple[str, ...] = (
    "designer",
    "ux designer",
    "ui designer",
    "product designer",
    "graphic designer",
    "visual designer",
)

_RESEARCH_HINTS: tuple[str, ...] = (
    "research scientist",
    "researcher",
    "postdoc",
    "phd",
    "professor",
    "faculty",
)

_SOFTWARE_HINTS: tuple[str, ...] = (
    "software engineer",
    "software developer",
    "backend engineer",
    "frontend engineer",
    "full stack",
    "fullstack",
    "devops",
    "sre",
    "platform engineer",
    "site reliability",
)


def normalize_role_category(value: str | None) -> RoleCategory:
    """Coerce arbitrary role labels to a supported category."""
    cleaned = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "software": "software_engineering",
        "engineering": "software_engineering",
        "swe": "software_engineering",
        "ml": "aiml",
        "ai_ml": "aiml",
        "machine_learning": "aiml",
        "data": "data_science",
        "research": "research_academic",
        "academic": "research_academic",
        "none": "non_portfolio",
        "general": "non_portfolio",
    }
    cleaned = aliases.get(cleaned, cleaned)
    if cleaned in VALID_ROLE_CATEGORIES:
        return cleaned  # type: ignore[return-value]
    return "non_portfolio"


def infer_role_category(
    *,
    job_title: str | None = None,
    domain: str | None = None,
    jd_text: str = "",
    must_have: list[str] | None = None,
    nice_to_have: list[str] | None = None,
) -> RoleCategory:
    """Heuristic role_category when LLM classification is unavailable."""
    title = f" {str(job_title or '').lower()} "
    corpus = " ".join(
        [
            str(job_title or ""),
            str(domain or ""),
            jd_text,
            " ".join(must_have or []),
            " ".join(nice_to_have or []),
        ]
    ).lower()

    if any(hint in title or hint in corpus for hint in _NON_PORTFOLIO_TITLE_HINTS):
        return "non_portfolio"
    # Prefer SWE detection over coarse domain tagging (some JDs get mislabeled "design").
    # This prevents design portfolio requirements from triggering on SDE/SWE intern roles.
    if any(hint in title or hint in corpus for hint in _SOFTWARE_HINTS):
        return "software_engineering"
    if any(hint in corpus for hint in _AIML_HINTS):
        return "aiml"
    if any(hint in corpus for hint in _DATA_SCIENCE_HINTS):
        return "data_science"
    if domain == "design" or any(hint in corpus for hint in _DESIGN_HINTS):
        return "design"
    if domain == "academic" or any(hint in corpus for hint in _RESEARCH_HINTS):
        return "research_academic"
    if domain == "technical":
        return "software_engineering"
    return "non_portfolio"


def resolve_role_category(jd_structured: dict[str, Any] | Any | None) -> RoleCategory:
    """Read role_category from structured JD or infer it safely."""
    if not isinstance(jd_structured, dict):
        jd_structured = jd_structured.__dict__ if jd_structured is not None else {}

    explicit = jd_structured.get("role_category")
    if isinstance(explicit, str) and explicit.strip():
        return normalize_role_category(explicit)

    return infer_role_category(
        job_title=jd_structured.get("job_title"),
        domain=jd_structured.get("domain"),
        must_have=list(jd_structured.get("must_have") or []),
        nice_to_have=list(jd_structured.get("nice_to_have") or []),
    )


def _domain_from_url(url: str) -> str | None:
    normalized = normalize_url(url)
    if not normalized:
        return None
    host = urlparse(normalized).netloc.lower().replace("www.", "")
    return host or None


def is_personal_website(url: str) -> bool:
    """True when URL looks like a self-hosted portfolio (not generic social/email)."""
    domain = _domain_from_url(url)
    if not domain:
        return False
    for generic in _GENERIC_DOMAINS:
        if domain == generic or domain.endswith(f".{generic}"):
            return False
    return "." in domain


def _unwrap_external_content(content: str) -> str:
    """Strip sanitizer delimiters before crawl-quality heuristics."""
    text = str(content or "")
    match = re.search(
        r"===BEGIN EXTERNAL CONTENT:[^=]+===\s*(.*?)\s*===END EXTERNAL CONTENT===",
        text,
        flags=re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return text.strip()


def assess_crawl_quality(content: Any) -> tuple[bool, str]:
    """Filter soft crawl failures, blocks, and empty pages."""
    if not content or not isinstance(content, str):
        return False, "empty_response"

    body = _unwrap_external_content(content)
    if len(body) < 150:
        return False, "too_short"

    lowered = body.lower()
    for signature in _CRAWL_ERROR_SIGNATURES:
        if signature in lowered:
            return False, f"scrape_blocked_or_error_signature_({signature})"

    return True, "valid_content"


def enriched_contents_to_map(enriched_contents: list[dict[str, Any]] | None) -> dict[str, str]:
    """Map profile URL -> raw/sanitized crawl body."""
    mapped: dict[str, str] = {}
    for item in enriched_contents or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        normalized = normalize_url(url) or url
        mapped[normalized] = str(item.get("content") or "")
    return mapped


def resolve_experience_years(resume_structured: dict[str, Any] | Any | None) -> int:
    if not isinstance(resume_structured, dict):
        resume_structured = (
            resume_structured.__dict__ if resume_structured is not None else {}
        )
    raw = resume_structured.get("experience_years")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def evaluate_portfolio_signal(
    *,
    role_category: str,
    profile_urls: list[str],
    enriched_contents: list[dict[str, Any]] | dict[str, str] | None,
    experience_years: int = 0,
) -> dict[str, Any]:
    """Evaluate portfolio footprint and compute deterministic penalty metadata."""
    category = normalize_role_category(role_category)
    config = _ROLE_CONFIGS.get(category, _ROLE_CONFIGS["non_portfolio"])
    required: tuple[str, ...] = tuple(config["required_platforms"])

    if isinstance(enriched_contents, dict):
        content_by_url = dict(enriched_contents)
    else:
        content_by_url = enriched_contents_to_map(
            enriched_contents if isinstance(enriched_contents, list) else None
        )

    if not required:
        return {
            "role_category": category,
            "required_platforms_found": [],
            "required_platforms_missing": [],
            "penalty_points": 0,
            "penalty_applied": False,
            "penalty_reason": None,
            "signal_strength": "not_applicable",
            "crawl_status_log": {},
            "personal_portfolio_url": None,
        }

    normalized_urls = [normalize_url(url) or url for url in profile_urls if url]
    found_required: list[str] = []
    personal_portfolio_url: str | None = None

    for url in normalized_urls:
        url_lower = url.lower()
        for platform in required:
            if platform in url_lower and platform not in found_required:
                found_required.append(platform)
        if is_personal_website(url):
            personal_portfolio_url = url

    crawled_platforms: list[str] = []
    personal_portfolio_crawled = False
    crawl_status_log: dict[str, str] = {}

    for url in normalized_urls:
        content = content_by_url.get(url, "")
        is_valid, reason = assess_crawl_quality(content)
        crawl_status_log[url] = reason
        if not is_valid:
            continue
        url_lower = url.lower()
        for platform in required:
            if platform in url_lower and platform not in crawled_platforms:
                crawled_platforms.append(platform)
        if personal_portfolio_url and url == personal_portfolio_url:
            personal_portfolio_crawled = True

    missing_platforms = [platform for platform in required if platform not in found_required]

    if crawled_platforms:
        signal_strength: SignalStrength = "strong"
    elif personal_portfolio_crawled:
        signal_strength = "strong_personal_website"
    elif found_required or personal_portfolio_url:
        signal_strength = "weak"
    else:
        signal_strength = "none"

    base_penalty = int(config["base_penalty"])
    if experience_years <= 2:
        penalty_scaler = 0.7
    elif experience_years >= 8:
        penalty_scaler = 1.2
    else:
        penalty_scaler = 1.0

    if signal_strength == "strong":
        calculated_penalty = 0.0
    elif signal_strength == "strong_personal_website":
        calculated_penalty = base_penalty * 0.25 * penalty_scaler
    elif signal_strength == "weak":
        calculated_penalty = base_penalty * 0.5 * penalty_scaler
    elif signal_strength == "none":
        calculated_penalty = base_penalty * penalty_scaler
    else:
        calculated_penalty = 0.0

    final_penalty = int(round(calculated_penalty))
    result = {
        "role_category": category,
        "required_platforms_found": found_required,
        "required_platforms_missing": missing_platforms,
        "penalty_points": final_penalty,
        "penalty_applied": final_penalty > 0,
        "penalty_reason": config["penalty_label"] if final_penalty > 0 else None,
        "signal_strength": signal_strength,
        "crawl_status_log": crawl_status_log,
        "personal_portfolio_url": personal_portfolio_url,
        "experience_years": experience_years,
        "penalty_scaler": penalty_scaler,
    }
    logger.info(
        "Portfolio signal role=%s strength=%s penalty=%s platforms_found=%s",
        category,
        signal_strength,
        final_penalty,
        found_required,
    )
    return result


def build_portfolio_prompt_section(portfolio_signal: dict[str, Any]) -> str:
    """Qualitative guidance for the LLM scorer (no math in-model)."""
    if not portfolio_signal.get("penalty_applied"):
        return (
            "PORTFOLIO VERIFICATION STATUS:\n"
            "Verified: crawlable proof-of-work profiles match the role category. "
            "You may weigh verified portfolio evidence positively, but do not inflate "
            "scores from keyword stuffing alone."
        )

    penalty_points = int(portfolio_signal.get("penalty_points") or 0)
    return (
        "PORTFOLIO VERIFICATION WARNING:\n"
        f"Alert: {portfolio_signal.get('penalty_reason')}\n"
        f"Signal assessment: {str(portfolio_signal.get('signal_strength') or '').upper()}\n\n"
        "INSTRUCTIONS FOR SCORING ALIGNMENT:\n"
        "- The candidate lacks fully verifiable portfolio proof for this role category.\n"
        "- Do not let resume keyword matches alone inflate confidence or match_scores.\n"
        "- Note missing or unverifiable portfolio evidence in your qualitative reasoning.\n"
        f"- IMPORTANT: score resume claims normally; a deterministic post-processor will "
        f"subtract {penalty_points} points outside the model."
    )


def build_portfolio_red_flags(portfolio_signal: dict[str, Any]) -> list[dict[str, str]]:
    """Map portfolio penalties to platform red_flag contract (flag/severity/evidence)."""
    if not portfolio_signal.get("penalty_applied"):
        return []

    strength = str(portfolio_signal.get("signal_strength") or "")
    severity = "high" if strength == "none" else "medium"
    missing = portfolio_signal.get("required_platforms_missing") or []
    rec_platforms = ", ".join(missing) if missing else "the required role platforms"
    description = (
        f"{portfolio_signal.get('penalty_reason')} "
        f"(evaluated strength: {strength})."
    ).strip()
    recommendation = (
        f"Ask the candidate to supply active links for {rec_platforms} during initial screen."
    )
    return [
        {
            "flag": "missing_portfolio_verification",
            "severity": severity,
            "evidence": f"{description} Recommendation: {recommendation}"[:500],
        }
    ]


def apply_portfolio_penalties(
    score: int,
    portfolio_signal: dict[str, Any],
) -> tuple[int, int, bool]:
    """
    Apply deterministic penalty and optional hard-cap.

    Returns (adjusted_score, penalty_applied_points, hard_cap_applied).
    """
    if not portfolio_signal.get("penalty_applied"):
        return max(0, min(100, score)), 0, False

    penalty_points = int(portfolio_signal.get("penalty_points") or 0)
    adjusted = max(0, min(100, score - penalty_points))
    hard_cap_applied = False

    category = str(portfolio_signal.get("role_category") or "")
    strength = str(portfolio_signal.get("signal_strength") or "")
    if strength == "none" and category in _HARD_CAP_ROLE_CATEGORIES:
        if adjusted > _NONE_SIGNAL_HARD_CAP:
            adjusted = _NONE_SIGNAL_HARD_CAP
            hard_cap_applied = True

    applied = max(0, score - adjusted)
    return adjusted, applied, hard_cap_applied
