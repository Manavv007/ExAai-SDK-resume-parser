"""Finalize and validate agent-submitted screening JSON."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from agent.tools.rubric_builder import build_rubric
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

    application_id = str(state.get("application_id") or "")
    job_id = str(state.get("job_id") or "")
    try:
        UUID(application_id)
        UUID(job_id)
    except ValueError:
        return {
            "ok": False,
            "errors": ["session application_id and job_id must be valid UUIDs"],
        }

    jd_structured = state.get("jd_structured") or {}
    rubric = state.get("rubric")
    rubric_models = (
        build_rubric(jd_structured) if not rubric else rubric
    )

    try:
        normalized = normalize_screening_result(
            raw,
            application_id=application_id,
            job_id=job_id,
            resume_text=str(state.get("resume_text") or ""),
            rubric=rubric_models,
            enriched_contents=list(state.get("enriched_contents") or []),
            processing_time_ms=state.get("processing_time_ms"),
            identity_red_flags=list(state.get("identity_red_flags") or []),
            profile_identity_cap_score=bool(state.get("profile_identity_cap_score")),
        )
    except (ValueError, TypeError) as exc:
        return {"ok": False, "errors": [str(exc)]}

    outcome = validate_result_detailed(normalized)
    if not outcome.ok:
        return {"ok": False, "errors": outcome.errors}

    return {
        "ok": True,
        "screening_result": normalized,
        "message": "Screening result accepted and stored in session.",
    }
