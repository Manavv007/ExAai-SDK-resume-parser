"""Background sandbox finalization for provisional screening results."""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import Any

from agent.config import get_settings
from agent.sandbox.models import RepoExecutionReport
from agent.screening_store import ScreeningResultStore
from agent.tools.scorer import attach_temp_sandbox_reports

logger = logging.getLogger("exaai_adk.deferred_screening")

_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()


def persist_screening_result(
    *,
    application_id: str,
    job_id: str,
    status: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    store = ScreeningResultStore()
    return store.save(
        application_id=application_id,
        job_id=job_id,
        status=status,
        result=result,
    )


def schedule_deferred_sandbox_finalization(
    state: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Persist a provisional result and schedule sandbox-backed finalization if needed."""
    settings = get_settings()
    application_id = str(state.get("application_id") or result.get("application_id") or "")
    job_id = str(state.get("job_id") or result.get("job_id") or "")
    github_repo_analyses = state.get("github_repo_analyses")
    selected_urls = []
    if isinstance(github_repo_analyses, dict):
        urls = github_repo_analyses.get("selected_sandbox_repo_urls")
        if isinstance(urls, list):
            selected_urls = [str(url) for url in urls if url]

    if not settings.sandbox_deferred_enabled or not selected_urls:
        persist_screening_result(
            application_id=application_id,
            job_id=job_id,
            status=str(result.get("resume_screening_status") or "completed"),
            result=result,
        )
        return result

    provisional = copy.deepcopy(result)
    provisional["resume_screening_status"] = "processing"
    provisional["temp_sandbox_reports"] = [
        RepoExecutionReport(
            repo=_repo_name_from_url(url),
            url=url,
            provider=str(getattr(settings, "sandbox_provider", "cloud_run")),
            clone_ok=False,
            summary="Sandbox evaluation has been scheduled in the background.",
            skipped_reason="Deferred sandbox evaluation pending.",
        ).compact()
        for url in selected_urls
    ]
    persist_screening_result(
        application_id=application_id,
        job_id=job_id,
        status="processing",
        result=provisional,
    )

    task = asyncio.create_task(
        _finalize_screening_in_background(copy.deepcopy(state), copy.deepcopy(result))
    )
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return provisional


async def _finalize_screening_in_background(
    state: dict[str, Any],
    provisional_source: dict[str, Any],
) -> None:
    """Run sandbox evaluation after the response returns, then rescore once."""
    application_id = str(
        state.get("application_id") or provisional_source.get("application_id") or ""
    )
    job_id = str(state.get("job_id") or provisional_source.get("job_id") or "")
    store = ScreeningResultStore()

    try:
        from agent.pipeline import score_with_validation
        from agent.tools.github_analyzer import _evaluate_sandbox_repos

        settings = get_settings()
        github_repo_analyses = state.get("github_repo_analyses")
        selected_urls = []
        if isinstance(github_repo_analyses, dict):
            urls = github_repo_analyses.get("selected_sandbox_repo_urls")
            if isinstance(urls, list):
                selected_urls = [str(url) for url in urls if url]
        if not selected_urls:
            store.save(
                application_id=application_id,
                job_id=job_id,
                status=str(provisional_source.get("resume_screening_status") or "completed"),
                result=provisional_source,
            )
            return

        reports = await _evaluate_sandbox_repos(selected_urls, settings)
        repo_analyses = dict(github_repo_analyses or {})
        repo_analyses["sandbox_reports"] = reports
        state["github_repo_analyses"] = repo_analyses

        final_result = await asyncio.to_thread(score_with_validation, state, max_attempts=1)
        if final_result.get("resume_screening_status") != "completed":
            logger.warning(
                "Deferred sandbox final scoring did not complete cleanly for %s/%s; "
                "falling back to provisional result with real sandbox reports.",
                application_id,
                job_id,
            )
            final_result = copy.deepcopy(provisional_source)
            final_result["resume_screening_status"] = "completed"

        attach_temp_sandbox_reports(final_result, repo_analyses)
        store.save(
            application_id=application_id,
            job_id=job_id,
            status=str(final_result.get("resume_screening_status") or "completed"),
            result=final_result,
        )
    except Exception:
        logger.exception(
            "Deferred sandbox finalization failed for application_id=%s job_id=%s",
            application_id,
            job_id,
        )
        failed = copy.deepcopy(provisional_source)
        failed["resume_screening_status"] = "completed"
        store.save(
            application_id=application_id,
            job_id=job_id,
            status="completed",
            result=failed,
        )


def _repo_name_from_url(url: str) -> str:
    stripped = (url or "").rstrip("/").removesuffix(".git")
    parts = [part for part in stripped.split("/") if part]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return stripped or "unknown"
