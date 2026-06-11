"""Harden LLM scoring output before resume-screening-result-v1 validation."""

from __future__ import annotations

from typing import Any

from agent.tools.parser import VALID_REQUIREMENT_TYPES

_VALID_SEVERITIES = frozenset({"low", "medium", "high"})
_VALID_RELEVANCE = frozenset({"high", "medium", "low"})
_DEFAULT_EVIDENCE = "No explicit evidence found in resume or profiles."
_PLACEHOLDER_REQUIREMENTS = frozenset(
    {
        "requirement",
        "fit",
        "role fit",
        "criterion",
        "technical_skill",
        "must_have",
        "nice_to_have",
    }
)


def coerce_score(value: Any, *, default: int = 0) -> int:
    """Coerce model scores (float/str) to 0-100 integers."""
    if isinstance(value, bool):
        return default
    try:
        if isinstance(value, (int, float)):
            return max(0, min(100, int(round(value))))
        if isinstance(value, str) and value.strip():
            return max(0, min(100, int(round(float(value.strip())))))
    except (TypeError, ValueError):
        pass
    return default


def quantize_score(score: int, *, step: int = 5) -> int:
    """Snap a 0-100 score to the nearest step (e.g. step=5 → 70, 75, 80)."""
    if step <= 1:
        return max(0, min(100, score))
    quantized = int(round(score / step) * step)
    return max(0, min(100, quantized))


def _matches_cover_rubric(
    requirement_matches: list[dict[str, Any]],
    rubric: list[Any],
) -> bool:
    """True when every rubric row has a real LLM match (not a padded placeholder)."""
    if not rubric or len(requirement_matches) < len(rubric):
        return False
    for match in requirement_matches[: len(rubric)]:
        if str(match.get("evidence") or "").strip() == _DEFAULT_EVIDENCE:
            return False
    return True


def resolve_overall_score(
    *,
    llm_score: int,
    derived_score: int,
    rubric: list[Any],
    requirement_matches: list[dict[str, Any]],
    rubric_derived: bool,
    sandbox_llm_scoring: bool = False,
    has_sandbox_reports: bool = False,
) -> int:
    """Pick the overall score from LLM output vs weighted rubric mean."""
    if sandbox_llm_scoring and has_sandbox_reports and llm_score > 0:
        return llm_score

    has_full_rubric = _matches_cover_rubric(requirement_matches, rubric)
    if rubric_derived and has_full_rubric and derived_score > 0:
        return derived_score
    if derived_score > 0:
        return max(llm_score, derived_score)
    return llm_score


def _rubric_item_name(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("criterion") or item.get("requirement") or "").strip()
    return str(getattr(item, "criterion", None) or getattr(item, "requirement", None) or "").strip()


def is_placeholder_requirement(text: str) -> bool:
    cleaned = text.strip().lower()
    return not cleaned or cleaned in _PLACEHOLDER_REQUIREMENTS


def _resolve_requirement_label(
    item: dict[str, Any],
    *,
    rubric: list[Any],
    index: int,
) -> str:
    """Map agent output to the rubric criterion text (fixes literal 'requirement' placeholders)."""
    rubric_name = _rubric_item_name(rubric[index]) if index < len(rubric) else ""
    if rubric_name:
        return rubric_name
    from_model = str(item.get("requirement") or item.get("criterion") or "").strip()
    if from_model and not is_placeholder_requirement(from_model):
        return from_model
    return "requirement"


def _rubric_item_type(item: Any) -> str:
    if isinstance(item, dict):
        req_type = item.get("requirement_type")
    else:
        req_type = getattr(item, "requirement_type", None)
    if isinstance(req_type, str) and req_type in VALID_REQUIREMENT_TYPES:
        return req_type
    return "technical_skill"


def sanitize_requirement_matches(
    raw: Any,
    rubric: list[Any],
    *,
    score_step: int = 1,
) -> list[dict[str, Any]]:
    """Normalize requirement_matches; align length to rubric when possible."""
    items = raw if isinstance(raw, list) else []
    sanitized: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        requirement = _resolve_requirement_label(item, rubric=rubric, index=index)

        req_type = str(item.get("requirement_type") or "").strip().lower()
        if req_type not in VALID_REQUIREMENT_TYPES:
            req_type = (
                _rubric_item_type(rubric[index]) if index < len(rubric) else "technical_skill"
            )

        evidence = str(item.get("evidence") or "").strip().replace("\n", " ")[:200]
        if not evidence:
            evidence = _DEFAULT_EVIDENCE

        raw_match_score = coerce_score(item.get("match_score"))
        entry: dict[str, Any] = {
            "requirement": requirement or "requirement",
            "requirement_type": req_type,
            "match_score": quantize_score(raw_match_score, step=score_step),
            "evidence": evidence,
        }
        source_quote = item.get("source_quote")
        if isinstance(source_quote, str) and source_quote.strip():
            entry["source_quote"] = source_quote.strip()[:500]
        sanitized.append(entry)

    if rubric:
        while len(sanitized) < len(rubric):
            index = len(sanitized)
            sanitized.append(
                {
                    "requirement": _rubric_item_name(rubric[index]) or "requirement",
                    "requirement_type": _rubric_item_type(rubric[index]),
                    "match_score": 0,
                    "evidence": _DEFAULT_EVIDENCE,
                }
            )
        sanitized = sanitized[: len(rubric)]

    if not sanitized:
        sanitized.append(
            {
                "requirement": "role fit",
                "requirement_type": "technical_skill",
                "match_score": 0,
                "evidence": _DEFAULT_EVIDENCE,
            }
        )

    return sanitized


def sanitize_red_flags(raw: Any) -> list[dict[str, Any]]:
    items = raw if isinstance(raw, list) else []
    sanitized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        flag = str(item.get("flag") or "").strip()
        evidence = str(item.get("evidence") or "").strip().replace("\n", " ")[:500]
        if not flag or not evidence:
            continue
        severity = str(item.get("severity") or "medium").strip().lower()
        if severity not in _VALID_SEVERITIES:
            severity = "medium"
        sanitized.append({"flag": flag, "severity": severity, "evidence": evidence})
    return sanitized


def _normalize_url(url: str) -> str:
    cleaned = url.strip()
    if cleaned and not cleaned.startswith(("http://", "https://")):
        return f"https://{cleaned}"
    return cleaned


def _title_from_profile_meta(
    url: str,
    profile_url_meta: list[dict[str, Any]] | None,
) -> str | None:
    for item in profile_url_meta or []:
        if not isinstance(item, dict):
            continue
        if _normalize_url(str(item.get("url") or "")) != url:
            continue
        platform = item.get("platform")
        if isinstance(platform, str) and platform.strip():
            return platform.strip()[:200]
    return None


def sanitize_sources_crawled(
    raw: Any,
    *,
    enriched_fallback: list[dict[str, Any]],
    profile_urls_fallback: list[str] | None = None,
    profile_url_meta: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    items = raw if isinstance(raw, list) else []
    sanitized: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        url = _normalize_url(str(item.get("url") or ""))
        if not url or url in seen:
            continue
        seen.add(url)
        relevance = str(item.get("relevance") or "medium").strip().lower()
        if relevance not in _VALID_RELEVANCE:
            relevance = "medium"
        entry: dict[str, Any] = {"url": url, "relevance": relevance}
        title = item.get("title")
        if isinstance(title, str) and title.strip():
            entry["title"] = title.strip()[:200]
        sanitized.append(entry)

    if sanitized:
        return sanitized

    for item in enriched_fallback:
        url = _normalize_url(str(item.get("url") or ""))
        if not url or url in seen:
            continue
        seen.add(url)
        if item.get("skipped_fetch"):
            relevance = "low"
        elif item.get("ok", True):
            relevance = "high"
        else:
            relevance = "low"
        entry: dict[str, Any] = {"url": url, "relevance": relevance}
        title = item.get("domain_category")
        if isinstance(title, str) and title.strip():
            entry["title"] = title.strip()[:200]
        sanitized.append(entry)

    if sanitized:
        return sanitized

    for raw_url in profile_urls_fallback or []:
        url = _normalize_url(str(raw_url or ""))
        if not url or url in seen:
            continue
        seen.add(url)
        entry = {"url": url, "relevance": "medium"}
        title = _title_from_profile_meta(url, profile_url_meta)
        if title:
            entry["title"] = title
        sanitized.append(entry)
    return sanitized


def optional_metadata_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value >= 0:
        return int(round(value))
    return None


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Drop null optional fields — jsonschema rejects null for integer/boolean slots."""
    return {key: value for key, value in metadata.items() if value is not None}
