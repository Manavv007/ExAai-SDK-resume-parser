"""Link extraction, normalization, platform inference, and deduplication."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

from agent.config import get_settings
from agent.tools.parser import JdStructured

LinkSource = Literal["explicit", "inferred"]

_URL_PATTERN = re.compile(r"https?://[^\s\]>)\}\"']+", re.IGNORECASE)
_HTML_ATTR_URL_PATTERN = re.compile(
    r"""(?:href|src)\s*=\s*["']([^"']+)["']""",
    re.IGNORECASE,
)
_BARE_DOMAIN_PATTERN = re.compile(
    r"(?i)\b((?:github|gitlab|linkedin)\.com/[\w\-./%]+|"
    r"(?:linkedin\.com/in/[\w\-]+)|(?:github\.com/[\w\-]+))"
)
# Social handles only — exclude email addresses (e.g. user@gmail.com → not @gmail.com).
_HANDLE_PATTERN = re.compile(r"(?<![A-Za-z0-9._%+\-])@([A-Za-z0-9_\-]{2,39})(?![A-Za-z0-9._%+\-])")
_EMAIL_DOMAIN_HANDLES = frozenset(
    {
        "gmail.com",
        "googlemail.com",
        "outlook.com",
        "hotmail.com",
        "yahoo.com",
        "icloud.com",
        "protonmail.com",
        "live.com",
        "msn.com",
    }
)

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

_PROFILE_NOISE_QUERY_PARAMS = frozenset(
    {
        "locale",
        "lang",
        "language",
        "hl",
        "lng",
        "set_locale",
        "country",
        "region",
        "ref",
        "referrer",
        "trk",
        "trk_info",
        "original_referer",
        "originalreferer",
    }
)

# First path segment that signals site navigation, not a personal profile handle.
_SITE_SECTION_FIRST_SEGMENTS = frozenset(
    {
        "about",
        "blog",
        "company",
        "explore",
        "features",
        "feed",
        "for_you",
        "freelance",
        "gallery",
        "groups",
        "help",
        "hire",
        "joblist",
        "jobs",
        "learning",
        "login",
        "marketplace",
        "misc",
        "news",
        "orgs",
        "posts",
        "projects",
        "pulse",
        "register",
        "resources",
        "school",
        "search",
        "settings",
        "signup",
        "support",
        "topics",
        "trending",
        "enterprise",
    }
)

_PROFILE_ROUTE_PREFIXES = frozenset(
    {
        "in",
        "u",
        "user",
        "users",
        "profile",
        "people",
        "channel",
        "c",
    }
)

_GITHUB_NON_PROFILE_SEGMENTS = frozenset(
    {
        "orgs",
        "settings",
        "marketplace",
        "topics",
        "features",
        "enterprise",
        "sponsors",
        "login",
        "signup",
        "explore",
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


_NON_CRAWLABLE_PREFIXES = ("mailto:", "tel:", "sms:", "javascript:", "data:")
_NON_CRAWLABLE_NETLOCS = frozenset({"mailto", "tel", "sms", "javascript", "data"})

_STATIC_ASSET_EXTENSIONS = frozenset(
    {
        ".js",
        ".css",
        ".map",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".svg",
        ".ico",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".mp4",
        ".webm",
        ".mp3",
        ".wav",
        ".zip",
        ".gz",
        ".json",
        ".pdf",
        ".doc",
        ".docx",
    }
)

_CDN_SUBDOMAIN_MARKERS = (
    "cdn.",
    "-cdn-",
    "-cf.",
    "static.",
    "assets.",
    "s3-",
    "mir-",
)

_CDN_HOST_SUFFIXES = (
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "cdnjs.cloudflare.com",
    "ajax.googleapis.com",
    "use.typekit.net",
    "kit.fontawesome.com",
)

_NON_PROFILE_HOST_SUFFIXES = (
    "feedburner.com",
    "evidon.com",
    "doubleclick.net",
    "googletagmanager.com",
    "google-analytics.com",
    "formspree.io",
    "getform.io",
    "formcarry.com",
    "web3forms.com",
    "typeform.com",
    "calendly.com",
    "tally.so",
)

# Path segments on a custom portfolio host that embed another site's profile URL.
_EMBEDDED_EXTERNAL_PROFILE_MARKERS = frozenset(
    {
        "github.com",
        "gitlab.com",
        "linkedin.com",
        "twitter.com",
        "x.com",
        "instagram.com",
        "facebook.com",
        "youtube.com",
        "medium.com",
    }
)

# Hosts that look like filenames (e.g. https://script.js) before base-URL resolution.
_FAKE_HOST_RE = re.compile(
    r"^(?:script|style|main|index|app|bundle|vendor|chunk|dev_avatar)(?:[.\-_].*)?$",
    re.IGNORECASE,
)
_ASSET_NETLOC_SUFFIXES = (
    ".js",
    ".css",
    ".map",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
)


def normalize_url(url: str) -> str | None:
    """Normalize URL: enforce https, strip tracking params, trim trailing punctuation."""
    cleaned = url.strip().rstrip(".,;)")
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered.startswith(_NON_CRAWLABLE_PREFIXES):
        return None

    if not lowered.startswith(("http://", "https://")):
        cleaned = f"https://{cleaned}"

    parsed = urlparse(cleaned)
    if not parsed.netloc:
        return None

    netloc = parsed.netloc.lower().replace("www.", "")
    if netloc in _NON_CRAWLABLE_NETLOCS or netloc.startswith(("mailto", "tel", "sms")):
        return None
    if "@" in netloc and not netloc.endswith(
        tuple(f".{host}" for host in ("github.com", "gitlab.com", "linkedin.com"))
    ):
        return None

    # Reject truncated Google Docs document URLs produced when a PDF parser
    # splits a long URL across lines.  A valid document ID is exactly 44
    # base64url characters; anything shorter is a broken fragment.
    if "docs.google.com" in netloc and "/document/d/" in parsed.path:
        path_parts = [p for p in parsed.path.split("/") if p]
        # path_parts: ['document', 'd', '<id>', ...]
        try:
            doc_id = path_parts[path_parts.index("d") + 1]
        except (ValueError, IndexError):
            doc_id = ""
        if len(doc_id) < 44:
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


def profile_url_identity_key(url: str) -> tuple[str, str] | None:
    """Stable dedupe key for the same personal profile page (ignores query/www)."""
    canonical = canonical_profile_url(url) or normalize_url(url)
    if not canonical:
        return None
    parsed = urlparse(canonical)
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/").lower() or "/"
    return host, path


def _path_looks_like_personal_profile(host: str, path_parts: list[str]) -> bool:
    """True when the URL path is a candidate-owned profile root, not site navigation."""
    if not path_parts:
        return False

    first = path_parts[0].lower()

    if host.endswith(("github.com", "gitlab.com")):
        return len(path_parts) == 1 and first not in _GITHUB_NON_PROFILE_SEGMENTS

    if "linkedin.com" in host:
        return first == "in" and len(path_parts) >= 2

    if first in _SITE_SECTION_FIRST_SEGMENTS:
        return False

    if first in _PROFILE_ROUTE_PREFIXES and len(path_parts) >= 2:
        return True

    if first.startswith("@"):
        return True

    if len(path_parts) == 1:
        segment = first
        if "." in segment:
            ext = segment.rsplit(".", 1)[-1].lower()
            if ext in {"html", "htm", "php", "asp", "aspx", "jsp", "json", "xml"}:
                return False
        return True

    if len(path_parts) == 2 and first in {"profile"}:
        return True

    return False


_PORTFOLIO_PATH_HOSTS = frozenset(
    {
        "behance.net",
        "dribbble.com",
        "figma.com",
        "artstation.com",
    }
)

# Scraped JS fragments and template garbage — not candidate profile pages.
_PROFILE_PATH_GARBAGE_RE = re.compile(
    r"[(){}\[\]|\\+*?]|function\s|\.replace\(|n\.target",
    re.IGNORECASE,
)


def is_fetchable_personal_profile_url(url: str) -> bool:
    """True when a portfolio-host URL is a personal handle page, not site navigation."""
    canonical = canonical_profile_url(url) or normalize_url(url)
    if not canonical:
        return False
    if _PROFILE_PATH_GARBAGE_RE.search(canonical):
        return False
    parsed = urlparse(canonical)
    host = parsed.netloc.lower().removeprefix("www.")
    if is_junk_discovery_url(canonical):
        return False
    if not any(host == marker or host.endswith(f".{marker}") for marker in _PORTFOLIO_PATH_HOSTS):
        return True
    path_parts = [part for part in parsed.path.split("/") if part]
    return _path_looks_like_personal_profile(host, path_parts)


def canonical_profile_url(url: str) -> str | None:
    """Normalize a portfolio/profile URL for fetch and dedupe (drops locale/noise params)."""
    normalized = normalize_url(url)
    if not normalized:
        return None
    parsed = urlparse(normalized)
    host = parsed.netloc.lower().removeprefix("www.")
    path_parts = [part for part in parsed.path.split("/") if part]

    noise_params = _TRACKING_PARAMS | _PROFILE_NOISE_QUERY_PARAMS
    query = parse_qs(parsed.query, keep_blank_values=False)
    filtered = {key: value for key, value in query.items() if key.lower() not in noise_params}

    # Personal profile pages: query string is almost always locale/tracking noise.
    if _path_looks_like_personal_profile(host, path_parts):
        filtered = {}

    path = parsed.path or ""
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    new_query = urlencode(filtered, doseq=True)
    return urlunparse(("https", host, path, "", new_query, ""))


def collapse_profile_urls(urls: list[str]) -> list[str]:
    """Keep one canonical URL per personal-profile identity (first occurrence wins)."""
    merged: list[str] = []
    seen: set[tuple[str, str]] = set()
    for raw in urls:
        canonical = canonical_profile_url(str(raw or "")) or normalize_url(str(raw or ""))
        if not canonical:
            continue
        key = profile_url_identity_key(canonical)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(canonical)
    return merged


def _host_looks_like_real_site(netloc: str) -> bool:
    """Reject bare filenames mistaken for domains (e.g. script.js, dev_avatar.jpg)."""
    host = netloc.lower().replace("www.", "")
    if not host or host in {"localhost", "127.0.0.1"}:
        return True
    if any(host.endswith(suffix) for suffix in _ASSET_NETLOC_SUFFIXES):
        return False
    if _FAKE_HOST_RE.match(host.split(".")[0] if "." in host else host):
        return False
    labels = host.split(".")
    if len(labels) < 2:
        return False
    tld = labels[-1]
    if len(tld) < 2 or not tld.isalpha():
        return False
    return True


def is_static_asset_url(url: str) -> bool:
    """True for stylesheet/script/image/font URLs that are not profile pages."""
    normalized = normalize_url(url)
    if not normalized:
        return True
    path = urlparse(normalized).path.lower()
    for ext in _STATIC_ASSET_EXTENSIONS:
        if path.endswith(ext):
            return True
    return False


def is_cdn_or_asset_host(url: str) -> bool:
    normalized = normalize_url(url)
    if not normalized:
        return True
    host = urlparse(normalized).netloc.lower().replace("www.", "")
    if any(host == suffix or host.endswith(f".{suffix}") for suffix in _NON_PROFILE_HOST_SUFFIXES):
        return True
    if any(marker in host for marker in _CDN_SUBDOMAIN_MARKERS):
        return True
    if any(host == suffix or host.endswith(f".{suffix}") for suffix in _CDN_HOST_SUFFIXES):
        return True
    return not _host_looks_like_real_site(host)


def has_embedded_external_profile_path(url: str) -> bool:
    """True when a custom-domain URL path embeds github.com/linkedin.com as a segment."""
    normalized = normalize_url(url)
    if not normalized:
        return False
    parsed = urlparse(normalized)
    host = parsed.netloc.lower().removeprefix("www.")
    if any(
        host == marker or host.endswith(f".{marker}")
        for marker in _EMBEDDED_EXTERNAL_PROFILE_MARKERS
    ):
        return False
    parts = [part.lower() for part in parsed.path.split("/") if part]
    return bool(parts) and parts[0] in _EMBEDDED_EXTERNAL_PROFILE_MARKERS


def unwrap_embedded_external_profile_url(url: str) -> str | None:
    """Rewrite portfolio-relative paths like ``/github.com/user/repo`` to the real host URL."""
    normalized = normalize_url(url)
    if not normalized or not has_embedded_external_profile_path(normalized):
        return None
    parsed = urlparse(normalized)
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return None
    marker = parts[0].lower()
    if marker == "github.com" and len(parts) >= 2:
        owner = parts[1]
        if len(parts) == 2:
            profile = f"https://github.com/{owner}"
            return profile if is_profile_discovery_url(profile) else None
        repo = parts[2]
        repo_url = f"https://github.com/{owner}/{repo}"
        return repo_url if is_profile_discovery_url(repo_url) else None
    if marker == "linkedin.com" and len(parts) >= 3 and parts[1].lower() == "in":
        linkedin = f"https://linkedin.com/in/{parts[2]}"
        return linkedin if is_profile_discovery_url(linkedin) else None
    return None


def is_junk_discovery_url(url: str) -> bool:
    """Reject form handlers, resume PDFs, and portfolio-relative external profile links."""
    normalized = normalize_url(url)
    if not normalized:
        return True
    if is_static_asset_url(normalized):
        return True
    if is_cdn_or_asset_host(normalized):
        return True
    return has_embedded_external_profile_path(normalized)


def is_profile_discovery_url(url: str) -> bool:
    """True when a URL is worth keeping for profile discovery / Exa follow-up."""
    normalized = normalize_url(url)
    if not normalized:
        return False
    if is_junk_discovery_url(normalized):
        return False
    return True


def resolve_profile_url(candidate: str, base_url: str | None = None) -> str | None:
    """Normalize a link candidate, resolving relative paths against ``base_url``."""
    cleaned = candidate.strip().rstrip(".,;)")
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered.startswith(_NON_CRAWLABLE_PREFIXES):
        return None

    resolved = cleaned
    if base_url and not lowered.startswith(("http://", "https://")):
        if lowered.startswith("//"):
            resolved = f"https:{cleaned}"
        else:
            resolved = urljoin(base_url if base_url.endswith("/") else f"{base_url}/", cleaned)

    normalized = normalize_url(resolved)
    if not normalized or not is_profile_discovery_url(normalized):
        return None
    return normalized


def _is_valid_social_handle(handle: str) -> bool:
    """Reject email domains and other non-username @ tokens."""
    if not handle or "." in handle:
        return False
    lowered = handle.lower()
    if lowered in _EMAIL_DOMAIN_HANDLES:
        return False
    if lowered in {"email", "mail", "contact"}:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_\-]+", handle))


def _handle_from_explicit_github(urls: list[str]) -> str | None:
    """Prefer GitHub username from an explicit profile/repo URL on the resume."""
    for raw in urls:
        normalized = normalize_url(raw)
        if not normalized or "github.com" not in normalized:
            continue
        parts = [p for p in urlparse(normalized).path.split("/") if p]
        if not parts:
            continue
        candidate = parts[0]
        if candidate.lower() in {"orgs", "settings", "marketplace", "topics"}:
            continue
        if _is_valid_social_handle(candidate):
            return candidate
    return None


def _extract_handles(text: str, *, explicit_urls: list[str]) -> list[str]:
    handles: list[str] = []

    github_user = _handle_from_explicit_github(explicit_urls)
    if github_user:
        handles.append(github_user)

    for match in _HANDLE_PATTERN.finditer(text):
        handle = match.group(1).strip().strip("/")
        if _is_valid_social_handle(handle) and handle not in handles:
            handles.append(handle)
    return handles[:5]


def _resume_mentions_platform(resume_text: str, host: str) -> bool:
    """Only infer a platform URL if the resume actually references that site."""
    lowered = resume_text.lower()
    key = host.split(".")[0]
    return key in lowered or host in lowered


def _extract_explicit_urls(text: str, pdf_hyperlinks: list[str] | None) -> list[str]:
    found: list[str] = []
    for pattern in (_URL_PATTERN, _BARE_DOMAIN_PATTERN):
        for match in pattern.finditer(text):
            found.append(match.group(0))

    for link in pdf_hyperlinks or []:
        found.append(link)

    return found


def _infer_links(
    domain: str,
    handles: list[str],
    resume_text: str,
    *,
    explicit_hosts: set[str],
) -> list[ExtractedLink]:
    """
    Guess profile URLs only when we have a real username handle.

    Platforms are included only if the resume mentions that site or the
    candidate already linked to it explicitly (avoids gmail.com → kaggle.com).
    """
    templates = _DOMAIN_INFERENCE.get(domain, [])
    if not templates or not handles:
        return []

    inferred: list[ExtractedLink] = []
    handle = handles[0]
    for host, template in templates:
        if host in explicit_hosts:
            continue
        if not _resume_mentions_platform(resume_text, host):
            continue
        url = normalize_url(template.format(handle=handle))
        if url:
            inferred.append(
                ExtractedLink(url=url, source="inferred", platform=host),
            )
        if len(inferred) >= 2:
            break
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
    handles = _extract_handles(resume_text, explicit_urls=explicit_raw)

    seen: set[str] = set()
    results: list[ExtractedLink] = []
    explicit_hosts: set[str] = set()

    def add(url: str | None, source: LinkSource, platform: str | None = None) -> None:
        if not url or url in seen or len(results) >= limit:
            return
        seen.add(url)
        results.append(ExtractedLink(url=url, source=source, platform=platform))

    for raw in explicit_raw:
        normalized = normalize_url(raw)
        if not normalized:
            continue
        unwrapped = unwrap_embedded_external_profile_url(normalized)
        if unwrapped:
            host = urlparse(unwrapped).netloc
            explicit_hosts.add(host)
            add(unwrapped, "explicit", host)
            continue
        if not is_profile_discovery_url(normalized):
            continue
        host = urlparse(normalized).netloc
        explicit_hosts.add(host)
        add(normalized, "explicit", host)

    if get_settings().infer_profile_urls:
        for inferred in _infer_links(
            domain,
            handles,
            resume_text,
            explicit_hosts=explicit_hosts,
        ):
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


def _append_normalized_urls(
    urls: list[str],
    seen: set[str],
    candidates: list[str],
    *,
    max_urls: int,
    base_url: str | None = None,
) -> bool:
    """Append unique normalized URLs. Returns True when the cap is reached."""
    for candidate in candidates:
        if base_url:
            normalized = resolve_profile_url(candidate, base_url)
        else:
            normalized = normalize_url(candidate)
            if normalized and not is_profile_discovery_url(normalized):
                normalized = None
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        urls.append(normalized)
        if len(urls) >= max_urls:
            return True
    return False


def extract_urls_from_text(text: str, *, max_urls: int = 50) -> list[str]:
    """Extract normalized URLs from arbitrary text blocks (e.g., portfolio pages)."""
    if not text:
        return []
    seen: set[str] = set()
    urls: list[str] = []
    for pattern in (_URL_PATTERN, _BARE_DOMAIN_PATTERN):
        for match in pattern.finditer(text):
            candidate = match.group(0).rstrip("=.,;)>")
            if _append_normalized_urls(urls, seen, [candidate], max_urls=max_urls):
                return urls
    return urls


def extract_urls_from_html(
    html: str,
    *,
    base_url: str | None = None,
    max_urls: int = 50,
) -> list[str]:
    """Extract normalized URLs from raw HTML (href/src attributes and inline URLs).

    Used when Exa text extraction omits hyperlinks from JS-rendered portfolio sites.
    Relative ``href``/``src`` values are resolved against ``base_url`` when provided.
    """
    if not html:
        return []
    seen: set[str] = set()
    urls: list[str] = []
    attr_candidates = [match.group(1) for match in _HTML_ATTR_URL_PATTERN.finditer(html)]
    if _append_normalized_urls(
        urls, seen, attr_candidates, max_urls=max_urls, base_url=base_url
    ):
        return urls
    for pattern in (_URL_PATTERN, _BARE_DOMAIN_PATTERN):
        for match in pattern.finditer(html):
            candidate = match.group(0).rstrip("=.,;)>")
            if _append_normalized_urls(
                urls, seen, [candidate], max_urls=max_urls, base_url=base_url
            ):
                return urls
    return urls


def merge_url_candidates(*candidate_lists: list[str]) -> list[str]:
    """Merge URL candidate lists preserving first-seen order."""
    merged: list[str] = []
    seen: set[str] = set()
    for candidates in candidate_lists:
        for candidate in candidates:
            normalized = normalize_url(candidate)
            if not normalized or not is_profile_discovery_url(normalized):
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
    return merged
