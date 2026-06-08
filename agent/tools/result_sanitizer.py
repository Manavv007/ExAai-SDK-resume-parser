"""Harden LLM scoring output before resume-screening-result-v1 validation."""

from __future__ import annotations

from typing import Any

from agent.tools.parser import VALID_REQUIREMENT_TYPES

_VALID_SEVERITIES = frozenset({"low", "medium", "high"})
_VALID_RELEVANCE = frozenset({"high", "medium", "low"})
_DEFAULT_EVIDENCE = "No explicit evidence found in resume or profiles."


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


def _rubric_item_name(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("criterion") or item.get("requirement") or "").strip()
    return str(getattr(item, "criterion", None) or getattr(item, "requirement", None) or "").strip()


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
) -> list[dict[str, Any]]:
    """Normalize requirement_matches; align length to rubric when possible."""
    items = raw if isinstance(raw, list) else []
    sanitized: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        requirement = str(item.get("requirement") or "").strip()
        if not requirement and index < len(rubric):
            requirement = _rubric_item_name(rubric[index]) or "requirement"

        req_type = str(item.get("requirement_type") or "").strip().lower()
        if req_type not in VALID_REQUIREMENT_TYPES:
            req_type = (
                _rubric_item_type(rubric[index]) if index < len(rubric) else "technical_skill"
            )

        evidence = str(item.get("evidence") or "").strip().replace("\n", " ")[:200]
        if not evidence:
            evidence = _DEFAULT_EVIDENCE

        entry: dict[str, Any] = {
            "requirement": requirement or "requirement",
            "requirement_type": req_type,
            "match_score": coerce_score(item.get("match_score")),
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


def sanitize_sources_crawled(
    raw: Any,
    *,
    enriched_fallback: list[dict[str, Any]],
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
        sanitized.append(
            {
                "url": url,
                "relevance": "high" if item.get("ok", True) else "low",
                **(
                    {"title": str(item["domain_category"]).strip()[:200]}
                    if isinstance(item.get("domain_category"), str)
                    and str(item.get("domain_category")).strip()
                    else {}
                ),
            }
        )
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
