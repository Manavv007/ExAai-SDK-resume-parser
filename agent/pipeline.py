"""Screening pipeline: prep → enrich → score → validate → audit."""

from __future__ import annotations

import time
from typing import Any

from google.adk import Agent
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService

from agent.adk_tools import (
    fetch_profile_content,
    fetch_profiles,
    list_candidate_profile_urls,
    submit_screening_result,
)
from agent.agent_runner import SCREENING_AGENT_INSTRUCTION
from agent.audit.logger import log_screening_result
from agent.config import get_settings
from agent.llm_client import create_adk_model
from agent.prep import prepare_screening_state
from agent.session_state import SESSION_STATE_KEYS
from agent.tools.scorer import build_failed_result, score_screening_from_state
from agent.tools.validator import validate_result_detailed

SCREENING_AGENT_TOOLS = [
    list_candidate_profile_urls,
    fetch_profiles,
    submit_screening_result,
    fetch_profile_content,
]


def create_screening_agent(
    *,
    before_model_callback: Any | None = None,
) -> Agent:
    """ADK agent with enrichment tools and submit_screening_result (Phase 3)."""
    settings = get_settings()
    agent_kwargs: dict[str, Any] = {
        "name": "resume_screener",
        "model": create_adk_model(settings),
        "description": "Screens resumes with Exa-backed profile enrichment and structured verdict",
        "instruction": SCREENING_AGENT_INSTRUCTION,
        "tools": SCREENING_AGENT_TOOLS,
    }
    if before_model_callback is not None:
        agent_kwargs["before_model_callback"] = before_model_callback
    return Agent(**agent_kwargs)


def create_runner(
    *,
    agent: Agent | None = None,
    auto_create_session: bool = False,
) -> Runner:
    app = App(name="exaai_adk", root_agent=agent or create_screening_agent())
    return Runner(
        app=app,
        session_service=InMemorySessionService(),
        auto_create_session=auto_create_session,
    )


root_agent = create_screening_agent()


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def score_with_validation(state: dict[str, Any]) -> dict[str, Any]:
    """
    Score and validate; retry once with correction prompt on schema failure.
    """
    correction: str | None = None
    for attempt in range(2):
        state["retry_count"] = attempt
        if correction:
            state["correction_prompt"] = correction

        result = score_screening_from_state(state)
        outcome = validate_result_detailed(result)
        if outcome.ok:
            if result.get("resume_screening_status") != "failed":
                result["resume_screening_status"] = "completed"
            return result

        correction = (
            "Your JSON failed schema validation: "
            + "; ".join(outcome.errors)
            + ". Return only valid resume-screening-result-v1 JSON."
        )

    return build_failed_result(
        application_id=state["application_id"],
        job_id=state["job_id"],
        code="VALIDATION_ERROR",
        message=correction or "Validation failed",
        resume_text=state.get("resume_text", ""),
        processing_time_ms=state.get("processing_time_ms"),
    )


async def run_screening_pipeline_async(state: dict[str, Any]) -> dict[str, Any]:
    """Deterministic path: enrich all profile URLs, then score with Gemini."""
    from agent.enrichment import enrich_profile_urls_async

    await enrich_profile_urls_async(state)
    return score_with_validation(state)


async def run_screening_async(
    *,
    application_id: str,
    job_id: str,
    resume_bytes: bytes,
    resume_filename: str,
    jd_bytes: bytes | None = None,
    jd_filename: str = "",
    jd_text: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Async screening entry; mode from SCREENING_MODE (pipeline or agent)."""
    start = time.monotonic()
    state: dict[str, Any] = prepare_screening_state(
        application_id=application_id,
        job_id=job_id,
        resume_bytes=resume_bytes,
        resume_filename=resume_filename,
        jd_bytes=jd_bytes,
        jd_filename=jd_filename,
        jd_text=jd_text,
    )
    state["request_id"] = request_id
    state["start_time"] = start
    state["retry_count"] = 0

    settings = get_settings()
    if settings.screening_mode == "agent":
        from agent.agent_runner import run_screening_agent_async

        result = await run_screening_agent_async(state)
    else:
        result = await run_screening_pipeline_async(state)

    state["processing_time_ms"] = _elapsed_ms(start)
    if result.get("metadata"):
        result["metadata"]["processing_time_ms"] = state["processing_time_ms"]

    log_screening_result(state, result, request_id=request_id)
    return result


def run_screening(
    *,
    application_id: str,
    job_id: str,
    resume_bytes: bytes,
    resume_filename: str,
    jd_bytes: bytes | None = None,
    jd_filename: str = "",
    jd_text: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Sync wrapper for CLI/tests."""
    import asyncio

    return asyncio.run(
        run_screening_async(
            application_id=application_id,
            job_id=job_id,
            resume_bytes=resume_bytes,
            resume_filename=resume_filename,
            jd_bytes=jd_bytes,
            jd_filename=jd_filename,
            jd_text=jd_text,
            request_id=request_id,
        )
    )


def state_contract_doc() -> str:
    """Return session state documentation string."""
    return SESSION_STATE_KEYS
