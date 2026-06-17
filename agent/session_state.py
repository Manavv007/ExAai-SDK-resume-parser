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
    profile_trust: list[dict[str, Any]]
    profile_trust_by_url: dict[str, str]
    identity_red_flags: list[dict[str, str]]
    profile_identity_cap_score: bool
    rubric: list[dict[str, str]]
    rubric_preamble: str
    enriched_contents: list[dict[str, Any]]
    screening_result: dict[str, Any]
    redaction_count: int
    prep_latency_ms: int
    processing_time_ms: int
    correction_prompt: str | None
    retry_count: int
    start_time: float
    github_username: str | None
    github_repo_analyses: dict[str, Any]
    discovered_profile_urls: list[str]
    discovered_github_repo_urls: list[str]


# Documented keys for maintainers (TypedDict is not enforced at runtime).
SESSION_STATE_KEYS = """
Inputs (set at request):
  application_id, job_id, request_id

Prep outputs:
  resume_text, jd_raw, jd_structured, profile_urls, profile_url_meta,
  profile_trust, profile_trust_by_url, identity_red_flags, profile_identity_cap_score,
  rubric, rubric_preamble, redaction_count, prep_latency_ms, enriched_contents=[]

Enrichment outputs:
  enriched_contents[] — {{url, content, domain_category, ok?}}

Agent path (Phase 3+):
  Prep state is copied into ADK session.state before Runner starts.
  Initial user message from build_agent_user_message() carries JD, resume, rubric,
  and profile_trust_by_url; tools read/write the same session dict.
  Agent tools: list_candidate_profile_urls, fetch_profiles, submit_screening_result
  (fetch_profile_content optional for single URL).
  screening_result — resume-screening-result-v1 after submit_screening_result succeeds.
  discovered_profile_urls, discovered_github_repo_urls — depth-1 links discovered from
  portfolio pages during enrichment.

Pipeline-only scoring / validation:
  correction_prompt, retry_count, processing_time_ms
"""
