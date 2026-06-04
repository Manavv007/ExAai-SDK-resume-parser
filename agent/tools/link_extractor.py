"""Link extraction, normalization, platform inference, and deduplication."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from agent.config import get_settings
from agent.tools.parser import JdStructured

LinkSource = Literal["explicit", "inferred"]

_URL_PATTERN = re.compile(r"https?://[^\s\]>)\}\"']+", re.IGNORECASE)
_BARE_DOMAIN_PATTERN = re.compile(
    r"(?i)\b((?:github|gitlab|linkedin)\.com/[\w\-./%]+|"
    r"(?:linkedin\.com/in/[\w\-]+)|(?:github\.com/[\w\-]+))"
)
_HANDLE_PATTERN = re.compile(r"@([A-Za-z0-9_\-.]{2,})")

_TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "fbclid",
        "gclid",
    }
)

_DOMAIN_INFERENCE: dict[str, list[tuple[str, str]]] = {
    "technical": [
        ("github.com", "https://github.com/{handle}"),
        ("gitlab.com", "https://gitlab.com/{handle}"),
        ("hackerrank.com", "https://www.hackerrank.com/{handle}"),
        ("kaggle.com", "https://www.kaggle.com/{handle}"),
        ("huggingface.co", "https://huggingface.co/{handle}"),
    ],
    "design": [
        ("behance.net", "https://www.behance.net/{handle}"),
        ("dribbble.com", "https://dribbble.com/{handle}"),
        ("artstation.com", "https://www.artstation.com/{handle}"),
    ],
    "writing": [
        ("medium.com", "https://medium.com/@{handle}"),
        ("substack.com", "https://{handle}.substack.com"),
        ("dev.to", "https://dev.to/{handle}"),
    ],
    "academic": [
        ("orcid.org", "https://orcid.org/{handle}"),
        ("researchgate.net", "https://www.researchgate.net/profile/{handle}"),
    ],
    "music": [
        ("soundcloud.com", "https://soundcloud.com/{handle}"),
        ("bandcamp.com", "https://{handle}.bandcamp.com"),
    ],
    "film": [
        ("vimeo.com", "https://vimeo.com/{handle}"),
        ("youtube.com", "https://www.youtube.com/@{handle}"),
    ],
    "business": [
        ("crunchbase.com", "https://www.crunchbase.com/person/{handle}"),
        ("producthunt.com", "https://www.producthunt.com/@{handle}"),
        ("wellfound.com", "https://wellfound.com/u/{handle}"),
    ],
}


@dataclass(frozen=True)
class ExtractedLink:
    url: str
    source: LinkSource
    platform: str | None = None


def normalize_url(url: str) -> str | None:
    """Normalize URL: enforce https, strip tracking params, trim trailing punctuation."""
    cleaned = url.strip().rstrip(".,;)")
    if not cleaned:
        return None

    if not cleaned.lower().startswith(("http://", "https://")):
        cleaned = f"https://{cleaned}"

    parsed = urlparse(cleaned)
    if not parsed.netloc:
        return None

    query = parse_qs(parsed.query, keep_blank_values=False)
    filtered = {k: v for k, v in query.items() if k.lower() not in _TRACKING_PARAMS}
    new_query = urlencode(filtered, doseq=True)

    path = parsed.path or ""
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    normalized = urlunparse(
        (
            "https",
            parsed.netloc.lower(),
            path,
            "",
            new_query,
            "",
        )
    )
    return normalized


def _extract_handles(text: str) -> list[str]:
    handles: list[str] = []
    for match in _HANDLE_PATTERN.finditer(text):
        handle = match.group(1).strip().strip("/")
        if handle and handle not in handles:
            handles.append(handle)
    return handles[:5]


def _extract_explicit_urls(text: str, pdf_hyperlinks: list[str] | None) -> list[str]:
    found: list[str] = []
    for pattern in (_URL_PATTERN, _BARE_DOMAIN_PATTERN):
        for match in pattern.finditer(text):
            found.append(match.group(0))

    for link in pdf_hyperlinks or []:
        found.append(link)

    return found


def _infer_links(domain: str, handles: list[str]) -> list[ExtractedLink]:
    templates = _DOMAIN_INFERENCE.get(domain, [])
    if not templates or not handles:
        return []

    inferred: list[ExtractedLink] = []
    handle = handles[0]
    for _host, template in templates[:4]:
        url = normalize_url(template.format(handle=handle))
        if url:
            inferred.append(
                ExtractedLink(url=url, source="inferred", platform=_host),
            )
    return inferred


def extract_links(
    resume_text: str,
    *,
    jd: JdStructured | None = None,
    pdf_hyperlinks: list[str] | None = None,
    max_urls: int | None = None,
) -> list[ExtractedLink]:
    """
    Extract and infer candidate profile URLs from resume text.

    ``resume_text`` should be raw or redacted body text; hyperlinks from PDF
    annotations are passed separately so they are not lost during PII redaction.
    """
    limit = max_urls if max_urls is not None else get_settings().max_urls_per_resume
    domain = jd.domain if jd else _detect_resume_domain(resume_text)

    explicit_raw = _extract_explicit_urls(resume_text, pdf_hyperlinks)
    handles = _extract_handles(resume_text)

    seen: set[str] = set()
    results: list[ExtractedLink] = []

    def add(url: str | None, source: LinkSource, platform: str | None = None) -> None:
        if not url or url in seen or len(results) >= limit:
            return
        seen.add(url)
        results.append(ExtractedLink(url=url, source=source, platform=platform))

    for raw in explicit_raw:
        normalized = normalize_url(raw)
        if normalized:
            host = urlparse(normalized).netloc
            add(normalized, "explicit", host)

    inferred_hosts = {link.platform for link in results if link.source == "explicit"}

    for inferred in _infer_links(domain, handles):
        if inferred.platform and inferred.platform in inferred_hosts:
            continue
        add(inferred.url, inferred.source, inferred.platform)

    return results


def _detect_resume_domain(text: str) -> str:
    lowered = text.lower()
    if any(k in lowered for k in ("figma", "behance", "dribbble", "ux", "ui design")):
        return "design"
    if any(k in lowered for k in ("research", "publication", "phd", "thesis")):
        return "academic"
    if any(k in lowered for k in ("python", "kubernetes", "api", "backend", "git")):
        return "technical"
    return "general"
