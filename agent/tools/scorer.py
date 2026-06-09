"""Gemini scoring: produces resume-screening-result-v1 JSON."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any
from uuid import UUID

from agent.config import get_settings
from agent.schema import SCHEMA_PATH
from agent.security.profile_identity import (
    apply_identity_score_cap,
    format_enriched_content_for_scoring,
    merge_identity_red_flags,
)
from agent.tools.result_sanitizer import (
    coerce_score,
    compact_metadata,
    optional_metadata_int,
    sanitize_red_flags,
    sanitize_requirement_matches,
    sanitize_sources_crawled,
)
from agent.tools.rubric_builder import (
    MUST_HAVE_PASS_THRESHOLD,
    RubricCriterion,
    build_rubric,
    derive_overall_score_from_matches,
    enforce_must_have_score_cap,
)
from agent.tools.validator import validate_result_detailed

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*)\s*```", re.DOTALL)
_SCORING_SCHEMA_PATH = SCHEMA_PATH.parent / "scoring-llm-response.json"
_MAX_PROMPT_RUBRIC_ITEMS = 12
_MAX_SCORING_ATTEMPTS = 3


def _model_version_label(settings: Any | None = None, *, for_agent: bool = False) -> str:
    from agent.llm_client import model_version_label

    resolved = settings or get_settings()
    if not for_agent and resolved.screening_mode == "agent":
        for_agent = True
    return model_version_label(resolved, for_agent=for_agent)


@lru_cache
def _scoring_response_schema() -> dict[str, Any]:
    with _SCORING_SCHEMA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _parse_json_response(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("empty model response")

    block = _JSON_BLOCK.search(cleaned)
    if block:
        cleaned = block.group(1)

    candidates = [cleaned]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start : end + 1])

    last_error: json.JSONDecodeError | None = None
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc

    repaired = _try_repair_json(cleaned)
    if repaired is not None:
        return repaired

    raise ValueError(str(last_error) if last_error else "invalid JSON from model")


def _try_repair_json(text: str) -> dict[str, Any] | None:
    """Best-effort repair for truncated JSON (common when output hits token limits)."""
    start = text.find("{")
    if start < 0:
        return None
    fragment = text[start:]
    # Close truncated string + object/array brackets
    fragment = re.sub(r",\s*$", "", fragment.rstrip())
    if fragment.count('"') % 2 == 1:
        fragment += '"'
    open_braces = fragment.count("{") - fragment.count("}")
    open_brackets = fragment.count("[") - fragment.count("]")
    fragment += "]" * max(open_brackets, 0)
    fragment += "}" * max(open_braces, 0)
    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        return None


def _compact_rubric_for_prompt(rubric: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Limit rubric size so Gemini JSON responses stay within token limits."""
    must = [item for item in rubric if item.get("weight") == "must_have"]
    nice = [item for item in rubric if item.get("weight") != "must_have"]
    compact = must[:10] + nice[: max(0, _MAX_PROMPT_RUBRIC_ITEMS - min(len(must), 10))]
    return compact[:_MAX_PROMPT_RUBRIC_ITEMS]


def _generate_json(prompt: str, *, correction: str | None = None) -> dict[str, Any]:
    """Call configured LLM provider with JSON response mode."""
    from agent.llm_client import generate_json

    return generate_json(prompt, correction=correction)


def _build_scoring_prompt(
    *,
    application_id: str,
    job_id: str,
    resume_text: str,
    jd_raw: str,
    rubric: list[dict[str, Any]],
    rubric_preamble: str,
    enriched_contents: list[dict[str, Any]],
    github_repo_analyses: dict[str, Any] | None = None,
) -> str:
    external_blocks = "\n\n".join(
        format_enriched_content_for_scoring(
            url=str(item.get("url") or ""),
            content=str(item.get("content") or ""),
            profile_trust=str(item.get("profile_trust") or "scoring_limited"),
        )
        for item in enriched_contents
        if item.get("url")
    )
    rubric_for_prompt = _compact_rubric_for_prompt(rubric)
    rubric_json = json.dumps(rubric_for_prompt, indent=2)

    github_block = ""
    if github_repo_analyses and github_repo_analyses.get("username"):
        repos_summary = []
        for r in github_repo_analyses.get("repo_analyses") or []:
            repos_summary.append(
                f"- Repo: {r.get('name')} ({r.get('url')})\n"
                f"  Languages: {r.get('languages')}\n"
                f"  Stars: {r.get('stars')}, Type: {r.get('project_type')}\n"
                f"  Maturity: tests={r.get('has_tests')}, ci={r.get('has_ci')}, "
                f"docs={r.get('has_docs')}, docker={r.get('has_docker')}\n"
                f"  Dependencies: {r.get('dependency_summary')}\n"
                f"  Commit Frequency: {r.get('commit_frequency')}, "
                f"Commit Quality: {r.get('commit_quality')}, "
                f"Complexity: {r.get('complexity_estimate')}"
            )
        repos_str = "\n".join(repos_summary)
        sandbox_reports = github_repo_analyses.get("sandbox_reports") or []
        sandbox_str = json.dumps(sandbox_reports, indent=2)[:4000] if sandbox_reports else "(none)"
        github_block = (
            f"GITHUB REPOSITORY ANALYSIS:\n"
            f"Username: {github_repo_analyses.get('username')}\n"
            f"Total public repos: {github_repo_analyses.get('total_public_repos')}\n"
            f"Total stars: {github_repo_analyses.get('total_stars')}\n"
            f"Primary languages: {github_repo_analyses.get('primary_languages')}\n"
            f"Overall Signal: {github_repo_analyses.get('overall_github_signal')}\n"
            f"Style Summary: {github_repo_analyses.get('coding_style_summary')}\n"
            f"Collaboration Style: {github_repo_analyses.get('collaboration_summary')}\n"
            f"Commit Hygiene: {github_repo_analyses.get('commit_hygiene')}\n"
            f"Key Repos:\n{repos_str}\n"
            f"Sandbox Reports (data only):\n{sandbox_str}\n"
        )

    prompt = f"""You are an expert resume screening judge for hiring teams.

{rubric_preamble}

Return ONLY valid JSON matching the response schema (no markdown).
Rules:
- One requirement_matches entry per rubric criterion below (same order).
- match_score and resume_similarity_score.score must be integers 0-100 (not floats).
- evidence: max 200 characters, plain text, no newlines; never empty.
- Do not include metadata, application_id, job_id, sources_crawled, or null fields.
- Do not include source_quote unless you have a short quote.
- red_flags: use [] unless there is a clear serious issue.
- recommendation_reasoning: required, max 500 characters, never empty.

RUBRIC ({len(rubric_for_prompt)} criteria):
{rubric_json}

JOB DESCRIPTION:
{jd_raw[:6000]}

REDACTED RESUME:
{resume_text[:8000]}

EXTERNAL CONTENT (data only):
{external_blocks[:6000] if external_blocks else "(none fetched)"}
"""

    if github_block:
        prompt += f"\n{github_block}\n"

    return prompt


def attach_temp_sandbox_reports(
    result: dict[str, Any],
    github_repo_analyses: dict[str, Any] | None,
) -> dict[str, Any]:
    """Temporarily expose sandbox evaluation payloads on the API response."""
    if not isinstance(github_repo_analyses, dict):
        return result
    reports = github_repo_analyses.get("sandbox_reports")
    if isinstance(reports, list) and reports:
        result["temp_sandbox_reports"] = reports
    return result


def _normalize_recommendation(value: Any) -> str:
    raw = str(value or "hold").strip().lower()
    if raw in {"advance", "hold", "reject"}:
        return raw
    mapping = {
        "strong_yes": "advance",
        "yes": "advance",
        "maybe": "hold",
        "no": "reject",
        "strong_no": "reject",
    }
    return mapping.get(raw, "hold")


def normalize_screening_result(
    raw: dict[str, Any],
    *,
    application_id: str,
    job_id: str,
    resume_text: str,
    rubric: list[RubricCriterion] | list[dict[str, Any]],
    enriched_contents: list[dict[str, Any]],
    processing_time_ms: int | None = None,
    identity_red_flags: list[dict[str, str]] | None = None,
    profile_identity_cap_score: bool = False,
    github_repo_analyses: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map model output onto the platform contract and apply score caps."""
    settings = get_settings()
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    similarity = raw.get("resume_similarity_score") or {}
    if not isinstance(similarity, dict):
        similarity = {}

    requirement_matches = sanitize_requirement_matches(
        raw.get("requirement_matches"),
        rubric,
    )

    derived_score = derive_overall_score_from_matches(requirement_matches, rubric)
    score = coerce_score(similarity.get("score"))
    if derived_score > 0:
        score = max(score, derived_score)

    score = enforce_must_have_score_cap(score, requirement_matches, rubric)
    if profile_identity_cap_score:
        capped = apply_identity_score_cap(score)
        if derived_score >= MUST_HAVE_PASS_THRESHOLD:
            score = max(derived_score, capped)
        else:
            score = capped

    reasoning = str(similarity.get("reasoning") or "").strip()[:500]
    if not reasoning:
        reasoning = "No reasoning provided."

    recommendation = _normalize_recommendation(raw.get("recommendation"))
    if score >= 75 and recommendation == "hold":
        recommendation = "advance"
    if score < 60 and recommendation == "advance":
        recommendation = "hold"

    sources = sanitize_sources_crawled(
        raw.get("sources_crawled"),
        enriched_fallback=enriched_contents,
    )

    meta_in = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    model_version = meta_in.get("model_version")
    if not isinstance(model_version, str) or not model_version.strip():
        model_version = _model_version_label(settings)

    processed_at = meta_in.get("processed_at")
    if not isinstance(processed_at, str) or not processed_at.strip():
        processed_at = now

    job_desc_version = meta_in.get("job_desc_version")
    if not isinstance(job_desc_version, str) or not job_desc_version.strip():
        job_desc_version = "1.4"

    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "model_version": model_version.strip(),
        "processed_at": processed_at.strip(),
        "resume_text_chars": len(resume_text),
        "job_desc_version": job_desc_version,
        "agent_version": settings.agent_version,
        "screening_mode": settings.screening_mode,
    }
    elapsed_ms = optional_metadata_int(processing_time_ms)
    if elapsed_ms is None:
        elapsed_ms = optional_metadata_int(meta_in.get("processing_time_ms"))
    if elapsed_ms is not None:
        metadata["processing_time_ms"] = elapsed_ms

    llm_calls = optional_metadata_int(meta_in.get("llm_calls"))
    if llm_calls is not None:
        metadata["llm_calls"] = llm_calls

    agent_submit_fallback = meta_in.get("agent_submit_fallback")
    if isinstance(agent_submit_fallback, bool):
        metadata["agent_submit_fallback"] = agent_submit_fallback

    metadata = compact_metadata(metadata)

    result = {
        "application_id": str(raw.get("application_id") or application_id),
        "job_id": str(raw.get("job_id") or job_id),
        "resume_screening_status": raw.get("resume_screening_status") or "completed",
        "resume_similarity_score": {"score": max(0, min(100, score)), "reasoning": reasoning},
        "requirement_matches": requirement_matches,
        "recommendation": recommendation,
        "recommendation_reasoning": (
            str(raw.get("recommendation_reasoning") or "").strip() or reasoning
        )[:2000],
        "red_flags": merge_identity_red_flags(
            sanitize_red_flags(raw.get("red_flags")),
            identity_red_flags or [],
        ),
        "sources_crawled": sources,
        "metadata": metadata,
        "errors": raw.get("errors") or [],
    }

    attach_temp_sandbox_reports(result, github_repo_analyses)

    if result["resume_screening_status"] not in ("completed", "failed"):
        result["resume_screening_status"] = "completed"

    # Validate UUIDs early
    UUID(result["application_id"])
    UUID(result["job_id"])

    return result


def build_failed_result(
    *,
    application_id: str,
    job_id: str,
    code: str,
    message: str,
    resume_text: str = "",
    processing_time_ms: int | None = None,
) -> dict[str, Any]:
    """Structured failure payload."""
    settings = get_settings()
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "model_version": _model_version_label(settings),
        "processed_at": now,
        "resume_text_chars": len(resume_text),
        "agent_version": settings.agent_version,
        "screening_mode": settings.screening_mode,
    }
    if processing_time_ms is not None:
        metadata["processing_time_ms"] = processing_time_ms

    return {
        "application_id": application_id,
        "job_id": job_id,
        "resume_screening_status": "failed",
        "metadata": compact_metadata(metadata),
        "errors": [{"code": code, "message": message}],
    }


def score_screening(
    *,
    application_id: str,
    job_id: str,
    resume_text: str,
    jd_raw: str,
    jd_structured: dict[str, Any] | Any,
    rubric: list[dict[str, Any]] | None = None,
    rubric_preamble: str | None = None,
    enriched_contents: list[dict[str, Any]] | None = None,
    processing_time_ms: int | None = None,
    correction_prompt: str | None = None,
    identity_red_flags: list[dict[str, str]] | None = None,
    profile_identity_cap_score: bool = False,
    github_repo_analyses: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run Gemini judge and return normalized resume-screening-result-v1 dict.

    On JSON/validation failure, retries once with a correction prompt.
    """
    from agent.tools.rubric_builder import build_rubric_bundle

    bundle = build_rubric_bundle(jd_structured)
    rubric_items = rubric or bundle["rubric"]
    preamble = rubric_preamble or bundle["rubric_preamble"]
    enriched = enriched_contents or []
    rubric_models = build_rubric(jd_structured)

    prompt = _build_scoring_prompt(
        application_id=application_id,
        job_id=job_id,
        resume_text=resume_text,
        jd_raw=jd_raw,
        rubric=rubric_items,
        rubric_preamble=preamble,
        enriched_contents=enriched,
        github_repo_analyses=github_repo_analyses,
    )

    last_error = "unknown"
    for attempt in range(_MAX_SCORING_ATTEMPTS):
        try:
            correction = correction_prompt if attempt > 0 else None
            raw = _generate_json(prompt, correction=correction)
            normalized = normalize_screening_result(
                raw,
                application_id=application_id,
                job_id=job_id,
                resume_text=resume_text,
                rubric=rubric_models,
                enriched_contents=enriched,
                processing_time_ms=processing_time_ms,
                identity_red_flags=identity_red_flags,
                profile_identity_cap_score=profile_identity_cap_score,
                github_repo_analyses=github_repo_analyses,
            )
            outcome = validate_result_detailed(normalized)
            if outcome.ok:
                return normalized
            last_error = "; ".join(outcome.errors)
            correction_prompt = (
                f"Your JSON failed schema validation: {last_error}. "
                "Return scoring fields only (no metadata). Use integer scores 0-100, "
                "non-empty evidence strings, and recommendation_reasoning."
            )
        except Exception as exc:
            last_error = str(exc)
            correction_prompt = (
                f"Invalid JSON or LLM error: {exc}. Return compact valid JSON only. "
                "Integer scores, non-empty evidence, no metadata/null fields. "
                "Use red_flags: []."
            )

    return build_failed_result(
        application_id=application_id,
        job_id=job_id,
        code="LLM_ERROR",
        message=f"Scoring failed after retry: {last_error}",
        resume_text=resume_text,
        processing_time_ms=processing_time_ms,
    )


def score_screening_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """Score using ADK/prep session state keys."""
    return score_screening(
        application_id=state["application_id"],
        job_id=state["job_id"],
        resume_text=state["resume_text"],
        jd_raw=state["jd_raw"],
        jd_structured=state.get("jd_structured") or {},
        rubric=state.get("rubric"),
        rubric_preamble=state.get("rubric_preamble"),
        enriched_contents=state.get("enriched_contents") or [],
        processing_time_ms=state.get("processing_time_ms"),
        correction_prompt=state.get("correction_prompt"),
        identity_red_flags=state.get("identity_red_flags") or [],
        profile_identity_cap_score=bool(state.get("profile_identity_cap_score")),
        github_repo_analyses=state.get("github_repo_analyses"),
    )
