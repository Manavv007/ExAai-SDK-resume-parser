"""Session state contract for the screening pipeline.

Used by prep, enrichment (ADK tools), scoring, and the HTTP API.
Keys are written by one stage and read by later stages — do not store raw PII
except ``resume_text`` (redacted).
"""

from __future__ import annotations

from typing import Any, TypedDict


class ScreeningSessionState(TypedDict, total=False):
    application_id: str
    job_id: str
    request_id: str
    resume_text: str
    jd_raw: str
    jd_structured: dict[str, Any]
    profile_urls: list[str]
    profile_url_meta: list[dict[str, Any]]
    rubric: list[dict[str, str]]
    rubric_preamble: str
    enriched_contents: list[dict[str, Any]]
    redaction_count: int
    prep_latency_ms: int
    processing_time_ms: int
    correction_prompt: str | None
    retry_count: int
    start_time: float


# Documented keys for maintainers (TypedDict is not enforced at runtime).
SESSION_STATE_KEYS = """
Inputs (set at request):
  application_id, job_id, request_id

Prep outputs:
  resume_text, jd_raw, jd_structured, profile_urls, profile_url_meta,
  rubric, rubric_preamble, redaction_count, prep_latency_ms, enriched_contents=[]

Enrichment outputs:
  enriched_contents[] — {{url, content, domain_category, ok?}}

Scoring / validation:
  correction_prompt, retry_count, processing_time_ms
"""
