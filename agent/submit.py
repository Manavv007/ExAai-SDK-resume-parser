"""Finalize and validate agent-submitted screening JSON."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from agent.prep_context import merge_with_prep_state
from agent.enrichment import resume_profile_urls
from agent.tools.rubric_builder import resolve_session_rubric
from agent.tools.scorer import normalize_screening_result
from agent.tools.validator import validate_result_detailed


def process_screening_submission(
    state: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize agent output, apply score caps, and validate platform contract.

    Returns ``{ok: true, screening_result: ...}`` or ``{ok: false, errors: [...]}``.
    """
    if not isinstance(raw, dict):
        return {
            "ok": False,
            "errors": ["result must be a JSON object"],
        }

    merged_state = merge_with_prep_state(state)
    application_id = str(merged_state.get("application_id") or "")
    job_id = str(merged_state.get("job_id") or "")
    try:
        UUID(application_id)
        UUID(job_id)
    except ValueError:
        return {
            "ok": False,
            "errors": ["session application_id and job_id must be valid UUIDs"],
        }

    rubric_models = resolve_session_rubric(merged_state)

    try:
        normalized = normalize_screening_result(
            raw,
            application_id=application_id,
            job_id=job_id,
            resume_text=str(merged_state.get("resume_text") or ""),
            rubric=rubric_models,
            enriched_contents=list(merged_state.get("enriched_contents") or []),
            processing_time_ms=merged_state.get("processing_time_ms"),
            identity_red_flags=list(merged_state.get("identity_red_flags") or []),
            profile_identity_cap_score=bool(merged_state.get("profile_identity_cap_score")),
            github_repo_analyses=merged_state.get("github_repo_analyses"),
            profile_urls=list(merged_state.get("profile_urls") or []),
            resume_profile_urls=resume_profile_urls(merged_state),
            profile_url_meta=list(merged_state.get("profile_url_meta") or []),
            jd_structured=merged_state.get("jd_structured") or {},
            resume_structured=merged_state.get("resume_structured") or {},
            discovered_github_repo_urls=list(merged_state.get("discovered_github_repo_urls") or []),
            discovered_profile_urls=list(merged_state.get("discovered_profile_urls") or []),
            profile_trust_by_url=dict(merged_state.get("profile_trust_by_url") or {}),
            screening_mode=str(merged_state.get("screening_mode") or ""),
            portfolio_role_category=merged_state.get("portfolio_role_category"),
            portfolio_role_reasoning=merged_state.get("portfolio_role_reasoning"),
            portfolio_role_source=merged_state.get("portfolio_role_source"),
            portfolio_role_platforms=list(merged_state.get("portfolio_role_platforms") or []),
            portfolio_role_label=merged_state.get("portfolio_role_label"),
        )
    except (ValueError, TypeError) as exc:
        return {"ok": False, "errors": [str(exc)]}

    outcome = validate_result_detailed(normalized)
    if not outcome.ok:
        import logging

        logging.getLogger("exaai_adk.submit").warning(
            "Screening submission validation failed: %s",
            "; ".join(outcome.errors),
        )
        return {"ok": False, "errors": outcome.errors}

    return {
        "ok": True,
        "screening_result": normalized,
        "message": "Screening result accepted and stored in session.",
    }
