"""Role-aware portfolio verification and deterministic scoring guardrails."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal
from urllib.parse import urlparse

from agent.tools.link_extractor import is_profile_discovery_url, normalize_url

logger = logging.getLogger("exaai_adk.portfolio_signal")

# ---------------------------------------------------------------------------
# Portfolio scoring constants
# ---------------------------------------------------------------------------

# Score must reach this threshold for a page to be considered portfolio-like.
_PORTFOLIO_SCORE_THRESHOLD = 40

# Known platform categories whose domains are definitionally portfolio hubs.
_PORTFOLIO_ALLOWLIST_CATS: frozenset[str] = frozenset(
    {"portfolio", "code", "design", "writing", "academic", "video", "music", "business"}
)

# URL path segments that signal a portfolio page — only trusted at shallow depth.
_PORTFOLIO_ROOT_PATHS: tuple[str, ...] = ("/portfolio", "/projects")

# First-person work ownership phrases — strong evidence of a personal showcase.
_FIRST_PERSON_WORK: tuple[str, ...] = (
    "i built",
    "i developed",
    "i created",
    "i designed",
    "i worked on",
    "i made",
    "i wrote",
    "my project",
    "my work",
    "my portfolio",
    "my github",
)

# Multi-word showcase phrases — specific enough to imply intentional portfolio content.
_SHOWCASE_PHRASES: tuple[str, ...] = (
    "live demo",
    "view project",
    "source code",
    "built with",
    "tech stack",
    "featured projects",
    "open source",
    "project showcase",
    "side project",
    "case study",
)

# Phrases indicating the page owner is open to work / contact — personal page signal.
_HIRE_SIGNALS: tuple[str, ...] = (
    "hire me",
    "available for",
    "open to work",
    "get in touch",
    "contact me",
    "freelance",
)

# Corporate "we/our" language — strong evidence the page is NOT a personal portfolio.
_CORPORATE_SIGNALS: tuple[str, ...] = (
    "our team",
    "our products",
    "our services",
    "we offer",
    "our solution",
    "our platform",
)

# Conversion / marketing copy — typical of company sites, not personal portfolios.
_CONVERSION_SIGNALS: tuple[str, ...] = (
    "sign up",
    "get started",
    "subscribe now",
    "buy now",
    "add to cart",
    "free trial",
)

# Job listing boilerplate — rules out a page that is a job post rather than a portfolio.
_JOB_LISTING_SIGNALS: tuple[str, ...] = (
    "apply now",
    "job requirements",
    "we are hiring",
    "equal opportunity employer",
)

# Hard error / block page signatures — page content is not usable.
_ERROR_SIGNALS: tuple[str, ...] = (
    "access denied",
    "page not found",
    "403 forbidden",
    "captcha",
    "enable javascript",
)

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
        "base_penalty": 15,
    },
    "design": {
        "required_platforms": ("behance.net", "dribbble.com", "figma.com"),
        "penalty_label": "No verifiable design portfolio (Behance/Dribbble/Figma)",
        "base_penalty": 15,
    },
    "research_academic": {
        "required_platforms": (
            "github.com",
            "scholar.google.com",
            "researchgate.net",
            "arxiv.org",
            "orcid.org",
        ),
        "penalty_label": "No verifiable research profile (Google Scholar/ORCID/ResearchGate)",
        "base_penalty": 15,
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
    """True when URL looks like a self-hosted custom domain (not generic social/email).

    Note: this is a *negative* check — it confirms the domain is not a known
    generic platform, but cannot verify ownership by a specific candidate.
    Use ``is_portfolio_like_url`` with ``candidate_name`` / ``known_handles``
    for ownership-aware detection.
    """
    if not is_profile_discovery_url(url):
        return False
    domain = _domain_from_url(url)
    if not domain:
        return False
    for generic in _GENERIC_DOMAINS:
        if domain == generic or domain.endswith(f".{generic}"):
            return False
    return "." in domain


# ---------------------------------------------------------------------------
# Portfolio scoring helpers
# ---------------------------------------------------------------------------


def _url_path_depth(url: str) -> int:
    """Number of non-empty path segments in the URL."""
    try:
        return len([s for s in urlparse(url).path.split("/") if s])
    except Exception:
        return 99


def _count_github_repo_urls(content: str) -> int:
    """Count distinct ``github.com/<user>/<repo>`` patterns — not bare domain mentions."""
    pattern = re.compile(r"github\.com/[\w\-]+/[\w\-]+", re.IGNORECASE)
    return len(set(pattern.findall(content)))


def _name_matches_domain(candidate_name: str, domain: str) -> bool:
    """Return True if the candidate's name tokens appear in the domain string.

    Splits the full name into tokens (first name, last name, etc.) and checks
    whether any token of ≥ 3 characters appears verbatim in the domain.  This
    catches patterns like ``johnsmith.dev``, ``john-smith.me``, ``jsmith.io``.
    """
    if not candidate_name or not domain:
        return False
    domain_clean = domain.lower().replace("-", "").replace(".", "")
    for part in re.split(r"[\s.\-_]+", candidate_name.lower()):
        part = part.strip()
        if len(part) >= 3 and part in domain_clean:
            return True
    return False


def _handle_matches_content(handles: list[str], content: str) -> bool:
    """Return True if any known handle appears in the crawled page content.

    A known handle (e.g. GitHub username ``janedoe``) appearing in the page
    content is strong evidence that the page belongs to this candidate.
    """
    if not handles or not content:
        return False
    lowered = content.lower()
    for handle in handles:
        if handle and len(handle) >= 3 and handle.lower() in lowered:
            return True
    return False


def _portfolio_score(
    url: str,
    content: str,
    candidate_name: str,
    known_handles: list[str],
) -> int:
    """
    Accumulate weighted evidence that a crawled page is a personal portfolio hub.

    Positive signals raise the score; anti-signals (corporate copy, error pages,
    job listings) lower it.  The caller compares the result against
    ``_PORTFOLIO_SCORE_THRESHOLD`` to produce a boolean decision.

    Score sources
    -------------
    URL-level (no content needed, high precision)
      +50  known portfolio-hub domain from allowlist (behance, codepen, etc.)
      +40  custom personal domain (non-generic, not a known platform)
      +35  hosted doc/workspace service (Notion, Google Docs)
      +20  /portfolio or /projects path at depth ≤ 2

    Identity verification (requires candidate context)
      +30  candidate name tokens found in the domain
      +25  known handle (GitHub username etc.) appears in page content

    Content signals (post-crawl)
      +10  each first-person work phrase, capped at +25 total
      +30/+20/+10  ≥3/≥2/≥1 distinct GitHub repo URLs in content
      +8   each showcase phrase ("live demo", "built with", etc.)
      +15  hire/contact-me intent

    Anti-signals
      -10  each corporate "we/our" phrase
      -15  each conversion CTA ("sign up", "buy now", etc.)
      -15  each job-listing boilerplate phrase
      -25  each hard error / block signature
    """
    score = 0
    lowered_url = url.lower()

    # -- Allowlist: known portfolio-hub domain ---------------------------------
    try:
        from agent.security.allowlist import get_domain_category, normalize_hostname
        host = normalize_hostname(url)
        if host:
            cat = get_domain_category(host)
            if cat in _PORTFOLIO_ALLOWLIST_CATS:
                score += 50
    except Exception:
        pass

    # -- Custom personal domain (not a known platform) -------------------------
    if is_personal_website(url):
        score += 40

    # -- Hosted doc / workspace services --------------------------------------
    if "docs.google.com/document/" in lowered_url:
        score += 35
    if "notion.so/" in lowered_url or ".notion.site/" in lowered_url:
        score += 35

    # -- Root-depth path hints ------------------------------------------------
    if _url_path_depth(url) <= 2 and any(hint in lowered_url for hint in _PORTFOLIO_ROOT_PATHS):
        score += 20

    # -- Identity: candidate name in domain -----------------------------------
    domain = _domain_from_url(url) or ""
    if candidate_name and _name_matches_domain(candidate_name, domain):
        score += 30
        logger.debug("portfolio_score: name match in domain url=%s", url)

    # -- Identity: known handle in page content --------------------------------
    if known_handles and content and _handle_matches_content(known_handles, content):
        score += 25
        logger.debug("portfolio_score: handle match in content url=%s", url)

    if not content:
        return score

    lowered = content.lower()

    # -- First-person work language (capped to avoid runaway) -----------------
    fp_hits = sum(1 for phrase in _FIRST_PERSON_WORK if phrase in lowered)
    score += min(fp_hits * 10, 25)

    # -- GitHub repo URL density -----------------------------------------------
    repos = _count_github_repo_urls(content)
    if repos >= 3:
        score += 30
    elif repos >= 2:
        score += 20
    elif repos >= 1:
        score += 10

    # -- Showcase phrases ------------------------------------------------------
    score += sum(8 for phrase in _SHOWCASE_PHRASES if phrase in lowered)

    # -- Hire / contact intent -------------------------------------------------
    if any(s in lowered for s in _HIRE_SIGNALS):
        score += 15

    # -- Anti-signals ----------------------------------------------------------
    score -= sum(10 for s in _CORPORATE_SIGNALS if s in lowered)
    score -= sum(15 for s in _CONVERSION_SIGNALS if s in lowered)
    score -= sum(15 for s in _JOB_LISTING_SIGNALS if s in lowered)
    score -= sum(25 for s in _ERROR_SIGNALS if s in lowered)

    return score


def is_portfolio_like_url(
    url: str,
    content: str = "",
    *,
    candidate_name: str = "",
    known_handles: list[str] | None = None,
) -> bool:
    """Evidence-weighted detection of personal portfolio hub pages.

    Accumulates positive signals (custom domain, known platform, first-person
    work language, GitHub repo link density, showcase phrases) and subtracts
    anti-signals (corporate copy, conversion CTAs, job listings, error pages).
    Returns True only when the total evidence score reaches
    ``_PORTFOLIO_SCORE_THRESHOLD`` (default 40).

    Parameters
    ----------
    url:
        The crawled page URL.
    content:
        Full sanitized text of the crawled page (available post-crawl).
    candidate_name:
        Candidate's full name from the parsed resume (e.g. ``"Jane Doe"``).
        When supplied, name tokens are matched against the URL domain to verify
        ownership (e.g. ``janedoe.dev`` scores higher for candidate "Jane Doe").
    known_handles:
        List of known platform handles for this candidate (e.g. GitHub username).
        When a handle appears in page content it strongly suggests ownership.

    Notes
    -----
    Existing callers that pass only ``(url)`` or ``(url, content)`` are
    unaffected — ``candidate_name`` and ``known_handles`` default to empty.
    """
    score = _portfolio_score(
        str(url or ""),
        str(content or ""),
        str(candidate_name or ""),
        list(known_handles or []),
    )
    logger.debug(
        "portfolio_score url=%s score=%d threshold=%d",
        url,
        score,
        _PORTFOLIO_SCORE_THRESHOLD,
    )
    return score >= _PORTFOLIO_SCORE_THRESHOLD


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


def classify_url_crawl_status(url: str, content: Any, *, was_enriched: bool) -> str:
    """Map a profile URL to an Exa-oriented crawl status label."""
    from agent.tools.github_analyzer import (
        normalize_github_profile_url,
        normalize_github_repo_url,
    )

    if normalize_github_repo_url(url):
        return "not_applicable_github_repo"
    if normalize_github_profile_url(url):
        return "not_applicable_github_profile"
    if not is_profile_discovery_url(url):
        return "not_applicable_asset"
    if not was_enriched:
        return "not_crawled"
    if not content or not isinstance(content, str):
        return "empty_response"
    is_valid, reason = assess_crawl_quality(content)
    return "valid_content" if is_valid else reason


def build_github_status_log(
    profile_urls: list[str],
    *,
    github_repo_analyses: dict[str, Any] | None,
) -> dict[str, str]:
    """Summarize GitHub API validation separate from Exa crawl_status_log."""
    from agent.tools.github_analyzer import (
        normalize_github_profile_url,
        normalize_github_repo_url,
    )

    status: dict[str, str] = {}
    github = github_repo_analyses if isinstance(github_repo_analyses, dict) else {}
    username = str(github.get("username") or "").strip()
    repo_analyses = {
        str(item.get("url") or ""): item
        for item in (github.get("repo_analyses") or [])
        if isinstance(item, dict)
    }
    sandbox_by_url = {
        str(item.get("url") or item.get("repo_url") or ""): item
        for item in (github.get("sandbox_reports") or [])
        if isinstance(item, dict)
    }

    if username:
        profile_url = f"https://github.com/{username}"
        if repo_analyses:
            status[profile_url] = "validated_api"
        else:
            status[profile_url] = "username_resolved"

    for url in profile_urls:
        normalized = normalize_url(url) or url
        repo_url = normalize_github_repo_url(normalized)
        if not repo_url:
            profile_only = normalize_github_profile_url(normalized)
            if profile_only and profile_only not in status:
                status[profile_only] = "discovered_not_validated"
            continue
        if repo_url in sandbox_by_url:
            report = sandbox_by_url[repo_url]
            status[repo_url] = "sandbox_ok" if report.get("clone_ok") else "sandbox_failed"
        elif repo_url in repo_analyses:
            status[repo_url] = "api_analyzed"
        else:
            status[repo_url] = "discovered_not_validated"

    return status


def build_sandbox_status_log(
    github_repo_analyses: dict[str, Any] | None,
) -> dict[str, str]:
    """Sandbox clone/eval status keyed by repository URL."""
    github = github_repo_analyses if isinstance(github_repo_analyses, dict) else {}
    status: dict[str, str] = {}
    for item in github.get("sandbox_reports") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("repo_url") or "").strip()
        if not url:
            continue
        if item.get("timed_out"):
            status[url] = "timed_out"
        elif item.get("clone_ok"):
            status[url] = "clone_ok"
        else:
            status[url] = "clone_failed"
    return status


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
    resume_structured: dict[str, Any] | None = None,
    github_repo_analyses: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate portfolio footprint and compute deterministic penalty metadata."""
    category = normalize_role_category(role_category)
    config = _ROLE_CONFIGS.get(category, _ROLE_CONFIGS["non_portfolio"])
    required: tuple[str, ...] = tuple(config["required_platforms"])

    if isinstance(enriched_contents, dict):
        content_by_url = dict(enriched_contents)
        enriched_url_set = set(content_by_url.keys())
    else:
        content_by_url = enriched_contents_to_map(
            enriched_contents if isinstance(enriched_contents, list) else None
        )
        enriched_url_set = {
            str(item.get("url") or "")
            for item in (enriched_contents or [])
            if isinstance(item, dict) and item.get("url")
        }

    if not required:
        return {
            "role_category": category,
            "required_platforms_found": [],
            "required_platforms_missing": [],
            "penalty_points": 0,
            "penalty_applied": False,
            "penalty_reason": None,
            "crawl_status_log": {},
            "github_status_log": {},
            "sandbox_status_log": {},
            "personal_portfolio_url": None,
        }

    normalized_urls = [normalize_url(url) or url for url in profile_urls if url]
    found_required: list[str] = []
    personal_portfolio_url: str | None = None
    verified_personal_portfolio = False

    candidate_name = ""
    github_username = ""
    if isinstance(resume_structured, dict):
        candidate_name = str(resume_structured.get("candidate_name") or "")
        github_username = str(resume_structured.get("github_username") or "")
    github_data = github_repo_analyses if isinstance(github_repo_analyses, dict) else {}
    if not github_username:
        github_username = str(github_data.get("username") or "")
    known_handles = [h for h in [github_username] if h]

    for url in normalized_urls:
        url_lower = url.lower()
        for platform in required:
            if platform in url_lower and platform not in found_required:
                found_required.append(platform)
        if is_personal_website(url):
            personal_portfolio_url = url

    crawl_status_log: dict[str, str] = {}

    for url in normalized_urls:
        content = content_by_url.get(url, "")
        crawl_status_log[url] = classify_url_crawl_status(
            url,
            content,
            was_enriched=url in enriched_url_set,
        )
        is_valid, _reason = assess_crawl_quality(content)
        if (
            category == "design"
            and not found_required
            and is_valid
            and is_portfolio_like_url(
                url,
                content,
                candidate_name=candidate_name,
                known_handles=known_handles,
            )
        ):
            verified_personal_portfolio = True
            personal_portfolio_url = personal_portfolio_url or url

    github_status_log = build_github_status_log(
        normalized_urls,
        github_repo_analyses=github_data,
    )
    sandbox_status_log = build_sandbox_status_log(github_data)

    missing_platforms = [
        platform for platform in required if platform not in found_required
    ]

    base_penalty = int(config["base_penalty"])
    # Presence policy: required platform link OR verified personal portfolio (design).
    if found_required:
        final_penalty = 0
    elif category == "design" and verified_personal_portfolio:
        final_penalty = 0
        found_required = ["verified_personal_portfolio"]
        missing_platforms = []
    else:
        final_penalty = base_penalty
    result = {
        "role_category": category,
        "required_platforms_found": found_required,
        "required_platforms_missing": missing_platforms,
        "penalty_points": final_penalty,
        "penalty_applied": final_penalty > 0,
        "penalty_reason": config["penalty_label"] if final_penalty > 0 else None,
        "crawl_status_log": crawl_status_log,
        "github_status_log": github_status_log,
        "sandbox_status_log": sandbox_status_log,
        "personal_portfolio_url": personal_portfolio_url,
        "verified_personal_portfolio": verified_personal_portfolio,
        "experience_years": experience_years,
        "penalty_scaler": 1.0,
    }
    logger.info(
        "Portfolio signal role=%s penalty=%s platforms_found=%s",
        category,
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
        "Verification status: no required platform links were provided.\n\n"
        "INSTRUCTIONS FOR SCORING ALIGNMENT:\n"
        "- The candidate did not provide required role-platform portfolio links.\n"
        "- Do not let resume keyword matches alone inflate confidence or match_scores.\n"
        "- Note missing portfolio platform links in your qualitative reasoning.\n"
        f"- IMPORTANT: score resume claims normally; a deterministic post-processor will "
        f"subtract {penalty_points} points outside the model."
    )


def build_portfolio_red_flags(portfolio_signal: dict[str, Any]) -> list[dict[str, str]]:
    """Map portfolio penalties to platform red_flag contract (flag/severity/evidence)."""
    if not portfolio_signal.get("penalty_applied"):
        return []

    severity = "high"
    missing = portfolio_signal.get("required_platforms_missing") or []
    rec_platforms = ", ".join(missing) if missing else "the required role platforms"
    description = str(portfolio_signal.get("penalty_reason") or "").strip()
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
    found_required = portfolio_signal.get("required_platforms_found") or []
    if not found_required and category in _HARD_CAP_ROLE_CATEGORIES:
        if adjusted > _NONE_SIGNAL_HARD_CAP:
            adjusted = _NONE_SIGNAL_HARD_CAP
            hard_cap_applied = True

    applied = max(0, score - adjusted)
    return adjusted, applied, hard_cap_applied
