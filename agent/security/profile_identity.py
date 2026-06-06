"""Profile URL identity trust: corroborate resume vs linked profiles before scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol
from urllib.parse import urlparse

from agent.config import get_settings


class _ProfileLinkLike(Protocol):
    url: str
    source: str

_EMAIL_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9._%+\-])([A-Za-z0-9._%+\-]+)@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})"
)
_PROFILE_HOSTS = frozenset(
    {
        "github.com",
        "www.github.com",
        "gitlab.com",
        "www.gitlab.com",
        "linkedin.com",
        "www.linkedin.com",
        "hackerrank.com",
        "www.hackerrank.com",
        "kaggle.com",
        "www.kaggle.com",
        "behance.net",
        "www.behance.net",
        "dribbble.com",
        "www.dribbble.com",
    }
)
_MIN_TOKEN_LEN = 2
_STOP_TOKENS = frozenset(
    {
        "com",
        "www",
        "the",
        "and",
        "inc",
        "llc",
        "ltd",
        "profile",
        "user",
        "in",
    }
)

PROFILE_UNTRUSTED_SCORE_CAP = 45

IDENTITY_SCORING_RULES = (
    "Resume-first scoring: requirement evidence must cite the redacted resume unless "
    "external content is marked SCORING_TRUSTED. "
    "SCORING_LIMITED blocks may support a criterion only when the resume also states "
    "the same skill or project. "
    "SCORING_UNTRUSTED / omitted profiles must not increase match_score or overall score. "
    "PROFILE OMITTED means the system could not automatically corroborate that URL with "
    "the resume name/handles — it is NOT a claim the link is fake or belongs to someone else. "
    "Only add profile_identity_mismatch when a GitHub/LinkedIn-class profile host is "
    "SCORING_UNTRUSTED. Do not flag email, phone, or demo-space URLs. "
    "Do not claim 'all URLs' are mismatched when some blocks are SCORING_TRUSTED."
)


class ProfileTrust(StrEnum):
    SCORING_TRUSTED = "scoring_trusted"
    SCORING_LIMITED = "scoring_limited"
    SCORING_UNTRUSTED = "scoring_untrusted"


@dataclass(frozen=True)
class IdentityBundle:
    """Name/email tokens from resume (not profile URL slugs — those are verified separately)."""

    name_tokens: tuple[str, ...] = ()
    email_tokens: tuple[str, ...] = ()

    def identity_tokens(self) -> frozenset[str]:
        return frozenset(self.name_tokens + self.email_tokens)


@dataclass(frozen=True)
class ProfileTrustAssessment:
    url: str
    trust: ProfileTrust
    source: str
    slug_tokens: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


def _normalize_token(raw: str) -> str | None:
    token = re.sub(r"[^a-z0-9]", "", raw.lower())
    if len(token) < _MIN_TOKEN_LEN or token in _STOP_TOKENS:
        return None
    if token.isdigit():
        return None
    return token


def _tokenize_text(text: str) -> list[str]:
    tokens: list[str] = []
    for part in re.split(r"[\s_\-./@+]+", text):
        normalized = _normalize_token(part)
        if normalized and normalized not in tokens:
            tokens.append(normalized)
    return tokens


def _name_tokens_from_resume(text: str) -> list[str]:
    tokens: list[str] = []
    try:
        from agent.security.pii_redactor import _cached_analyze

        for result in _cached_analyze(text, entities=["PERSON"], language="en"):
            snippet = text[result.start : result.end]
            for token in _tokenize_text(snippet):
                if token not in tokens:
                    tokens.append(token)
    except Exception:
        pass
    return tokens


def _email_tokens_from_resume(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _EMAIL_RE.finditer(text):
        local = match.group(1).split("+")[0]
        for token in _tokenize_text(local.replace(".", " ")):
            if token not in tokens:
                tokens.append(token)
    return tokens


def _slug_tokens_from_url(
    url: str,
    *,
    identity: IdentityBundle | None = None,
) -> list[str]:
    from agent.tools.link_extractor import normalize_url

    normalized = normalize_url(url)
    if not normalized:
        return []
    parsed = urlparse(normalized)
    host = (parsed.netloc or "").lower().replace("www.", "")
    parts = [p for p in parsed.path.split("/") if p]

    slug_parts: list[str] = []
    if host.endswith("github.com") or host.endswith("gitlab.com"):
        if parts and parts[0].lower() not in {
            "orgs",
            "settings",
            "marketplace",
            "topics",
        }:
            slug_parts.append(parts[0])
    elif "linkedin.com" in host:
        if parts and parts[0].lower() == "in" and len(parts) > 1:
            slug_parts.append(parts[1])
        elif parts:
            slug_parts.append(parts[-1])

    tokens: list[str] = []
    for part in slug_parts:
        for token in _tokenize_text(part):
            if token not in tokens:
                tokens.append(token)
        compact = re.sub(r"[^a-z0-9]", "", part.lower())
        if identity:
            for id_token in identity.identity_tokens():
                if len(id_token) >= 3 and id_token in compact and id_token not in tokens:
                    tokens.append(id_token)
    return tokens


def _collect_resume_profile_slugs(resume_text: str) -> list[list[str]]:
    from agent.tools.link_extractor import normalize_url

    slugs: list[list[str]] = []
    for match in re.finditer(
        r"https?://[^\s\]>)\}\"']+|(?:github|gitlab|linkedin)\.com/[\w\-./%]+",
        resume_text,
        re.IGNORECASE,
    ):
        raw = match.group(0)
        if not raw.lower().startswith("http"):
            raw = f"https://{raw}"
        normalized = normalize_url(raw)
        if not normalized:
            continue
        host = urlparse(normalized).netloc.lower()
        if not any(host == h or host.endswith("." + h) for h in _PROFILE_HOSTS):
            continue
        tokens = _slug_tokens_from_url(normalized)
        if tokens:
            slugs.append(tokens)
    return slugs


def extract_identity_bundle(resume_text: str) -> IdentityBundle:
    """Build identity tokens from raw resume (run before link lists are finalized)."""
    return IdentityBundle(
        name_tokens=tuple(_name_tokens_from_resume(resume_text)),
        email_tokens=tuple(_email_tokens_from_resume(resume_text)),
    )


def _tokens_related(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) >= 3 and len(b) >= 3 and (a in b or b in a):
        return True
    return False


def _token_sets_related(left: frozenset[str] | set[str], right: frozenset[str] | set[str]) -> bool:
    if not left or not right:
        return False
    for a in left:
        for b in right:
            if _tokens_related(a, b):
                return True
    return False


def _explicit_slugs_consistent(slug_lists: list[list[str]]) -> bool:
    if len(slug_lists) <= 1:
        return True
    for i in range(len(slug_lists)):
        for j in range(i + 1, len(slug_lists)):
            if not _token_sets_related(frozenset(slug_lists[i]), frozenset(slug_lists[j])):
                return False
    return True


def _slug_matches_identity(slug_tokens: list[str], bundle: IdentityBundle) -> bool:
    """Profile slug aligns with resume name/email (e.g. Manav vs Manavv007)."""
    return _token_sets_related(frozenset(slug_tokens), bundle.identity_tokens())


def _slug_cross_linked(slug_tokens: list[str], other_slug_lists: list[list[str]]) -> bool:
    """Another explicit profile on the resume shares tokens with this slug."""
    for other in other_slug_lists:
        if _token_sets_related(frozenset(slug_tokens), frozenset(other)):
            return True
    return False


def _identity_bundle_is_weak(bundle: IdentityBundle) -> bool:
    return len(bundle.identity_tokens()) == 0


def _clear_mismatch(slug_tokens: list[str], bundle: IdentityBundle) -> bool:
    """Strong disagreement: resume has name/email but profile slug shares none."""
    if _identity_bundle_is_weak(bundle):
        return False
    if not slug_tokens:
        return False
    return not _slug_matches_identity(slug_tokens, bundle)


def assess_profile_links(
    resume_text: str,
    links: list[_ProfileLinkLike],
) -> list[ProfileTrustAssessment]:
    """
    Assign scoring trust per profile URL.

    Explicit URLs are not blindly trusted; they require corroboration with resume identity
    or other profile slugs on the same resume (handles Manav vs Manavv007).
    """
    bundle = extract_identity_bundle(resume_text)
    explicit_profiles: list[tuple[str, list[str]]] = []
    for link in links:
        if link.source == "explicit":
            tokens = _slug_tokens_from_url(link.url, identity=bundle)
            if tokens:
                explicit_profiles.append((link.url, tokens))

    explicit_slug_lists = [tokens for _, tokens in explicit_profiles]
    cross_links_ok = _explicit_slugs_consistent(explicit_slug_lists)
    assessments: list[ProfileTrustAssessment] = []

    for link in links:
        slug_tokens = _slug_tokens_from_url(link.url, identity=bundle)
        reasons: list[str] = []
        others = [tokens for url, tokens in explicit_profiles if url != link.url]

        if link.source == "inferred":
            if _slug_matches_identity(slug_tokens, bundle):
                trust = ProfileTrust.SCORING_LIMITED
                reasons.append("inferred_url_weakly_matches_identity")
            else:
                trust = ProfileTrust.SCORING_UNTRUSTED
                reasons.append("inferred_url_no_identity_match")
            assessments.append(
                ProfileTrustAssessment(
                    url=link.url,
                    trust=trust,
                    source=link.source,
                    slug_tokens=tuple(slug_tokens),
                    reasons=tuple(reasons),
                )
            )
            continue

        if not cross_links_ok:
            trust = ProfileTrust.SCORING_UNTRUSTED
            reasons.append("conflicting_profile_slugs_on_resume")
        elif _slug_matches_identity(slug_tokens, bundle):
            trust = ProfileTrust.SCORING_TRUSTED
            reasons.append("profile_corroborated_with_resume_identity")
        elif others and _slug_cross_linked(slug_tokens, others):
            trust = ProfileTrust.SCORING_TRUSTED
            reasons.append("profile_cross_linked_with_other_resume_urls")
        elif _identity_bundle_is_weak(bundle):
            trust = ProfileTrust.SCORING_LIMITED
            reasons.append("explicit_url_identity_not_established_on_resume")
        elif _clear_mismatch(slug_tokens, bundle):
            trust = ProfileTrust.SCORING_UNTRUSTED
            reasons.append("profile_slug_does_not_match_resume_identity")
        else:
            trust = ProfileTrust.SCORING_LIMITED
            reasons.append("explicit_url_partial_identity_match")

        assessments.append(
            ProfileTrustAssessment(
                url=link.url,
                trust=trust,
                source=link.source,
                slug_tokens=tuple(slug_tokens),
                reasons=tuple(reasons),
            )
        )

    return assessments


def build_identity_red_flags(
    assessments: list[ProfileTrustAssessment],
) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    for item in assessments:
        if item.trust != ProfileTrust.SCORING_UNTRUSTED:
            continue
        evidence = f"{item.url}: {', '.join(item.reasons)}"[:200]
        flags.append(
            {
                "flag": "profile_identity_mismatch",
                "severity": "high",
                "evidence": (
                    "Automated check could not corroborate this profile URL with the "
                    f"resume identity (not a fraud determination): {evidence}"
                )[:500],
            }
        )
    return flags


def _is_identity_profile_host(url: str) -> bool:
    """True for hosts where slug/resume mismatch implies untrusted external evidence."""
    from agent.tools.link_extractor import normalize_url

    normalized = normalize_url(url)
    if not normalized:
        return False
    host = urlparse(normalized).netloc.lower().replace("www.", "")
    return any(host == profile_host or host.endswith("." + profile_host) for profile_host in _PROFILE_HOSTS)


def should_cap_score_for_identity(assessments: list[ProfileTrustAssessment]) -> bool:
    """Cap overall score when a resume-listed profile host is untrusted for scoring."""
    return any(
        a.trust == ProfileTrust.SCORING_UNTRUSTED and _is_identity_profile_host(a.url)
        for a in assessments
    )


def apply_identity_score_cap(score: int) -> int:
    return min(score, PROFILE_UNTRUSTED_SCORE_CAP)


def assessments_to_dicts(assessments: list[ProfileTrustAssessment]) -> list[dict[str, Any]]:
    return [
        {
            "url": a.url,
            "trust": a.trust.value,
            "source": a.source,
            "reasons": list(a.reasons),
            "slug_tokens": list(a.slug_tokens),
        }
        for a in assessments
    ]


def trust_map_from_assessments(
    assessments: list[ProfileTrustAssessment],
) -> dict[str, str]:
    return {a.url: a.trust.value for a in assessments}


def format_enriched_content_for_scoring(
    *,
    url: str,
    content: str,
    profile_trust: str,
) -> str:
    """Shape external block for the judge prompt based on trust tier."""
    mode = get_settings().profile_scoring_mode.strip().lower()
    if profile_trust == ProfileTrust.SCORING_UNTRUSTED.value:
        return (
            f"===PROFILE OMITTED ({url})===\n"
            "Identity not corroborated with resume. Do not use for scoring.\n"
            "===END PROFILE OMITTED==="
        )
    if profile_trust == ProfileTrust.SCORING_LIMITED.value:
        if mode == "balanced" and content.strip():
            return (
                f"===UNVERIFIED PROFILE ({url})===\n"
                f"{content}\n"
                "===END UNVERIFIED PROFILE===\n"
                "Use only if the redacted resume corroborates the same skills or projects."
            )
        return (
            f"===UNVERIFIED PROFILE ({url})===\n"
            "Content withheld; resume must corroborate any related skills.\n"
            "===END UNVERIFIED PROFILE==="
        )
    return content


def merge_identity_red_flags(
    model_flags: list[Any],
    identity_flags: list[dict[str, str]],
) -> list[Any]:
    merged = list(model_flags) if isinstance(model_flags, list) else []
    seen = {f.get("flag") for f in merged if isinstance(f, dict)}
    for flag in identity_flags:
        key = (flag.get("flag"), flag.get("evidence"))
        if key[0] in seen:
            continue
        merged.append(flag)
        seen.add(flag.get("flag"))
    return merged
