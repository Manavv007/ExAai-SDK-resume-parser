"""Presidio-based PII redaction for resume and JD text."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine

# Entity types redacted in resume/JD body (not in separately extracted link lists).
DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "DATE_TIME",
    "URL",
    "NRP",
    "AGE",
]


@dataclass
class RedactionSummary:
    """Internal metrics only — do not expose raw PII in API responses."""

    fields_redacted: list[str] = field(default_factory=list)
    redaction_count: int = 0
    counts_by_type: dict[str, int] = field(default_factory=dict)


# Lower number = kept when spans overlap (e.g. EMAIL wins over URL in an address).
_ENTITY_PRIORITY: dict[str, int] = {
    "EMAIL_ADDRESS": 0,
    "PHONE_NUMBER": 1,
    "PERSON": 2,
    "NRP": 3,
    "AGE": 4,
    "LOCATION": 5,
    "DATE_TIME": 6,
    "URL": 7,
}


@lru_cache
def _get_analyzer() -> AnalyzerEngine:
    return AnalyzerEngine()


def _non_overlapping_results(results: list) -> list:
    """Drop overlapping spans, keeping higher-priority entity types."""
    ordered = sorted(
        results,
        key=lambda r: (
            _ENTITY_PRIORITY.get(r.entity_type, 99),
            -(r.end - r.start),
        ),
    )
    kept: list = []
    for candidate in ordered:
        if any(candidate.start < k.end and candidate.end > k.start for k in kept):
            continue
        kept.append(candidate)
    return kept


def redact_text(
    text: str,
    *,
    entities: list[str] | None = None,
    redact_urls: bool = True,
    language: str = "en",
) -> tuple[str, RedactionSummary]:
    """
    Redact PII from plain text.

    Set ``redact_urls=False`` when processing an extracted link list so URLs
    remain fetchable while body text is still redacted separately.
    """
    if not text or not text.strip():
        return text, RedactionSummary()

    target_entities = list(entities or DEFAULT_ENTITIES)
    if not redact_urls and "URL" in target_entities:
        target_entities = [e for e in target_entities if e != "URL"]

    analyzer = _get_analyzer()
    results = analyzer.analyze(
        text=text,
        language=language,
        entities=target_entities,
    )

    if not results:
        return text, RedactionSummary()

    results = _non_overlapping_results(results)

    counts_by_type: dict[str, int] = defaultdict(int)
    counters: dict[str, int] = defaultdict(int)
    redacted = text

    for result in sorted(results, key=lambda r: r.start, reverse=True):
        entity_type = result.entity_type
        counters[entity_type] += 1
        counts_by_type[entity_type] += 1
        placeholder = f"[{entity_type}_{counters[entity_type]}]"
        redacted = redacted[: result.start] + placeholder + redacted[result.end :]

    summary = RedactionSummary(
        fields_redacted=sorted(counts_by_type.keys()),
        redaction_count=len(results),
        counts_by_type=dict(counts_by_type),
    )
    return redacted, summary
