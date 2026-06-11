"""Screening pipeline: prep → enrich → score → validate → audit."""

from __future__ import annotations

import time
from typing import Any

from google.adk import Agent
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService

from agent.adk_tools import (
    analyze_github,
    fetch_profiles,
    get_github_repo_structures,
    run_sandbox_analysis,
    submit_screening_result,
)
from agent.agent_runner import SCREENING_AGENT_INSTRUCTION
from agent.audit.logger import log_screening_result
from agent.config import get_settings
from agent.deferred_screening import schedule_deferred_sandbox_finalization
from agent.llm_client import (
    attach_llm_usage_metadata,
    create_adk_model,
    increment_llm_call_count,
    reset_llm_call_count,
    resolve_llm_provider,
)
from agent.prep import prepare_screening_state, retrieve_github_thread
from agent.prep_context import clear_prep_state, register_prep_state
from agent.sandbox_gating import (
    agent_evidence_orchestration_active,
    await_sandbox_for_scoring,
    clear_sandbox_task,
    ensure_sandbox_before_scoring,
    sandbox_overlap_active,
    start_sandbox_task,
)
from agent.session_state import SESSION_STATE_KEYS
from agent.tools.sandbox_scoring import reconcile_sandbox_penalty_in_result
from agent.tools.scorer import (
    attach_temp_sandbox_reports,
    build_failed_result,
    score_screening_from_state,
)
from agent.tools.validator import validate_result_detailed

def _screening_agent_tools() -> list[Any]:
    tools: list[Any] = [
        fetch_profiles,
        submit_screening_result,
        analyze_github,
    ]
    if agent_evidence_orchestration_active():
        tools.insert(0, get_github_repo_structures)
        tools.insert(2, run_sandbox_analysis)
    return tools


def screening_agent_tools() -> list[Any]:
    """Return ADK tools for the current settings."""
    return _screening_agent_tools()


def _counting_before_model_callback(callback_context: Any, llm_request: Any) -> Any:
    """Count native Gemini ADK turns (LiteLLM agent models count in generate_content_async)."""
    settings = get_settings()
    if resolve_llm_provider(settings) == "gemini":
        increment_llm_call_count(model=settings.gemini_model_id, source="adk_agent")
    return None


def create_screening_agent(
    *,
    before_model_callback: Any | None = None,
) -> Agent:
    """ADK agent with enrichment tools and submit_screening_result (Phase 3)."""
    settings = get_settings()

    def chained_before_model_callback(callback_context: Any, llm_request: Any) -> Any:
        _counting_before_model_callback(callback_context, llm_request)
        if before_model_callback is not None:
            return before_model_callback(
                callback_context=callback_context,
                llm_request=llm_request,
            )
        return None

    agent_kwargs: dict[str, Any] = {
        "name": "resume_screener",
        "model": create_adk_model(settings),
        "description": "Screens resumes with Exa-backed profile enrichment and structured verdict",
        "instruction": SCREENING_AGENT_INSTRUCTION,
        "tools": screening_agent_tools(),
        "before_model_callback": chained_before_model_callback,
    }
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


_root_agent: Agent | None = None


def get_root_agent() -> Agent:
    """Lazy singleton for ADK CLI/docs; avoids API key lookup at import time."""
    global _root_agent
    if _root_agent is None:
        _root_agent = create_screening_agent()
    return _root_agent


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def score_with_validation(
    state: dict[str, Any],
    *,
    max_attempts: int = 2,
    max_llm_attempts: int | None = None,
    compact_sandbox_prompt: bool = False,
) -> dict[str, Any]:
    """
    Score and validate; retry once with correction prompt on schema failure.
    """
    correction: str | None = None
    for attempt in range(max(1, max_attempts)):
        state["retry_count"] = attempt
        if correction:
            state["correction_prompt"] = correction

        result = score_screening_from_state(
            state,
            max_llm_attempts=max_llm_attempts,
            compact_sandbox_prompt=compact_sandbox_prompt,
        )
        outcome = validate_result_detailed(result)
        if outcome.ok:
            if result.get("resume_screening_status") != "failed":
                result["resume_screening_status"] = "completed"
            return result

        correction = (
            "Your JSON failed schema validation: "
            + "; ".join(outcome.errors)
            + ". Return scoring fields only (resume_similarity_score, "
            "requirement_matches, recommendation, recommendation_reasoning, red_flags). "
            "Do not include metadata, application_id, job_id, or null fields."
        )

    return build_failed_result(
        application_id=state["application_id"],
        job_id=state["job_id"],
        code="VALIDATION_ERROR",
        message=correction or "Validation failed",
        resume_text=state.get("resume_text", ""),
        processing_time_ms=state.get("processing_time_ms"),
    )


async def _await_github_prep(application_id: str) -> None:
    """Block until background GitHub API prep (and inline sandbox when configured) finishes."""
    import asyncio
    import logging

    github_thread, github_error = retrieve_github_thread(application_id)
    if github_thread is None:
        return
    await asyncio.to_thread(github_thread.join)
    if github_error:
        logging.getLogger("exaai_adk.prep").error(
            "Error during GitHub deep analysis: %s",
            github_error[0],
        )


async def run_screening_pipeline_async(state: dict[str, Any]) -> dict[str, Any]:
    """Deterministic path: enrich all profile URLs, then score with Gemini."""
    from agent.enrichment import enrich_profile_urls_async

    await enrich_profile_urls_async(state)
    await await_sandbox_for_scoring(state)
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
    reset_llm_call_count()
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
    if not request_id:
        import uuid

        request_id = uuid.uuid4().hex
    state["request_id"] = request_id
    state["start_time"] = start
    state["retry_count"] = 0

    settings = get_settings()
    screening_mode = settings.screening_mode
    state["screening_mode"] = screening_mode

    application_id = str(state.get("application_id") or "")
    register_prep_state(state)
    result: dict[str, Any] = {}
    try:
        await _await_github_prep(application_id)

        if not agent_evidence_orchestration_active(settings):
            if sandbox_overlap_active():
                start_sandbox_task(state)
            else:
                await ensure_sandbox_before_scoring(state)

        register_prep_state(state)

        if screening_mode == "agent":
            from agent.agent_runner import run_screening_agent_async

            result = await run_screening_agent_async(state)
        else:
            result = await run_screening_pipeline_async(state)
    finally:
        await await_sandbox_for_scoring(state)
        register_prep_state(state)
        clear_sandbox_task(application_id)
        clear_prep_state(application_id)

    state["processing_time_ms"] = _elapsed_ms(start)
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        result["metadata"] = metadata
    metadata["processing_time_ms"] = state["processing_time_ms"]
    result["metadata"] = attach_llm_usage_metadata(metadata)

    github_repo_analyses = state.get("github_repo_analyses")
    result = reconcile_sandbox_penalty_in_result(result, github_repo_analyses)
    attach_temp_sandbox_reports(result, github_repo_analyses)
    result = schedule_deferred_sandbox_finalization(state, result)
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
