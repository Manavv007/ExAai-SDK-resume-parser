"""Domain allowlist for crawl targets."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

# category -> domain suffixes (hostname equals suffix or is a subdomain of it)
ALLOWLIST_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "portfolio": (
        "behance.net",
        "dribbble.com",
        "adobe.com",
        "cargocollective.com",
        "format.com",
        "squarespace.com",
        "wixsite.com",
        "webflow.io",
        "contra.com",
    ),
    "code": (
        "github.com",
        "github.io",
        "gitlab.com",
        "bitbucket.org",
        "codepen.io",
        "replit.com",
        "hackerrank.com",
        "leetcode.com",
        "kaggle.com",
        "huggingface.co",
        "npmjs.com",
        "pypi.org",
    ),
    "professional": (
        "linkedin.com",
        "wellfound.com",
        "toptal.com",
        "upwork.com",
        "angel.co",
    ),
    "writing": (
        "medium.com",
        "substack.com",
        "dev.to",
        "hashnode.dev",
        "mirror.xyz",
        "ghost.io",
        "wordpress.com",
        "blogspot.com",
    ),
    "academic": (
        "scholar.google.com",
        "researchgate.net",
        "academia.edu",
        "orcid.org",
        "arxiv.org",
        "ssrn.com",
        "ncbi.nlm.nih.gov",
    ),
    "design": (
        "figma.com",
        "notion.site",
        "notion.so",
        "canva.com",
        "artstation.com",
        "deviantart.com",
    ),
    "video": (
        "vimeo.com",
        "youtube.com",
        "youtu.be",
        "imdb.com",
    ),
    "music": (
        "soundcloud.com",
        "bandcamp.com",
        "spotify.com",
        "music.apple.com",
    ),
    "business": (
        "crunchbase.com",
        "producthunt.com",
        "indiehackers.com",
    ),
    "legal": (
        "avvo.com",
        "martindale.com",
    ),
    "certification": (
        "credly.com",
        "accredible.com",
        "coursera.org",
        "learn.linkedin.com",
        "cloud.google.com",
        "aws.amazon.com",
    ),
}

_SUFFIX_TO_CATEGORY: dict[str, str] = {
    suffix: category
    for category, suffixes in ALLOWLIST_BY_CATEGORY.items()
    for suffix in suffixes
}

ALLOWED_SUFFIXES: frozenset[str] = frozenset(_SUFFIX_TO_CATEGORY.keys())


@dataclass(frozen=True)
class AllowlistResult:
    allowed: bool
    domain_category: str | None = None
    hostname: str | None = None
    reason: str | None = None


def normalize_hostname(url: str) -> str | None:
    parsed = urlparse(url)
    if not parsed.hostname:
        return None
    host = parsed.hostname.lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def get_domain_category(hostname: str) -> str | None:
    for suffix, category in _SUFFIX_TO_CATEGORY.items():
        if hostname == suffix or hostname.endswith(f".{suffix}"):
            return category
    return None


def is_allowlisted(url: str) -> bool:
    return check_allowlist(url).allowed


def check_allowlist(url: str) -> AllowlistResult:
    hostname = normalize_hostname(url)
    if not hostname:
        return AllowlistResult(allowed=False, reason="missing_hostname")

    category = get_domain_category(hostname)
    if category is None:
        return AllowlistResult(
            allowed=False,
            hostname=hostname,
            reason="domain_not_allowlisted",
        )

    return AllowlistResult(
        allowed=True,
        hostname=hostname,
        domain_category=category,
    )
