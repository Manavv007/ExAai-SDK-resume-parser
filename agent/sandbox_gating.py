"""Wait for sandbox evaluation before candidate scoring when required."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from agent.config import ResolvedSandboxPreRunMode, Settings, get_settings
from agent.prep_context import register_prep_state

logger = logging.getLogger("exaai_adk.sandbox_gating")

_DEFERRED_PENDING_REASON = "Deferred sandbox evaluation pending."

_SANDBOX_TASKS: dict[str, asyncio.Task[None]] = {}
_SANDBOX_STATE_REFS: dict[str, dict[str, Any]] = {}


def sandbox_llm_scoring_active(settings: Settings | None = None) -> bool:
    """True when sandbox risk should be judged by the LLM, not deterministic penalties."""
    resolved = settings or get_settings()
    return bool(getattr(resolved, "sandbox_llm_scoring_enabled", True))


def resolve_sandbox_pre_run_mode(
    settings: Settings | None = None,
) -> ResolvedSandboxPreRunMode:
    """
    Resolve whether prep should run sandbox before the agent starts.

    When orchestration is on, ``auto`` defaults to ``none`` so the agent picks
    focus_paths via run_sandbox_analysis.
    """
    resolved = settings or get_settings()
    raw = str(getattr(resolved, "sandbox_pre_run_mode", "auto")).strip().lower()
    if raw in ("none", "risk_only", "full"):
        return raw  # type: ignore[return-value]
    if agent_evidence_orchestration_active(resolved):
        return "none"
    return "full"


async def run_sandbox_pre_run_for_orchestration(state: dict[str, Any]) -> None:
    """Optional sandbox before the agent when evidence orchestration is enabled."""
    settings = get_settings()
    if not agent_evidence_orchestration_active(settings):
        return

    mode = resolve_sandbox_pre_run_mode(settings)
    if mode == "none":
        logger.info(
            "Skipping sandbox pre-run (SANDBOX_PRE_RUN_MODE=%s); "
            "agent should call run_sandbox_analysis with focus_paths",
            settings.sandbox_pre_run_mode,
        )
        return
    if mode == "risk_only":
        from agent.adk_tools import run_risk_only_sandbox_pre_pass
        from agent.prep_context import merge_with_prep_state

        prep_merged = merge_with_prep_state(state)
        if await run_risk_only_sandbox_pre_pass(prep_merged):
            if prep_merged.get("github_repo_analyses"):
                state["github_repo_analyses"] = prep_merged["github_repo_analyses"]
            if prep_merged.get("github_repo_structures"):
                state["github_repo_structures"] = prep_merged["github_repo_structures"]
            if prep_merged.get("sandbox_risk_only_pre_pass"):
                state["sandbox_risk_only_pre_pass"] = True
        register_prep_state(state)
        return

    from agent.adk_tools import ensure_sandbox_evidence
    from agent.prep_context import merge_with_prep_state

    prep_merged = merge_with_prep_state(state)
    await ensure_sandbox_evidence(prep_merged)
    if prep_merged.get("github_repo_analyses"):
        state["github_repo_analyses"] = prep_merged["github_repo_analyses"]
    register_prep_state(state)


def agent_evidence_orchestration_active(settings: Settings | None = None) -> bool:
    """True when the screening agent orchestrates GitHub/Exa/sandbox tool calls."""
    resolved = settings or get_settings()
    if str(getattr(resolved, "screening_mode", "pipeline")).strip().lower() != "agent":
        return False
    return bool(getattr(resolved, "agent_evidence_orchestration_enabled", True))


def effective_agent_run_timeout_seconds(
    settings: Settings | None = None,
    state: dict[str, Any] | None = None,
) -> int:
    """Wall-clock budget for the ADK agent loop (must cover sandbox tool latency)."""
    resolved = settings or get_settings()
    base = max(30, int(getattr(resolved, "agent_run_timeout_seconds", 120) or 120))
    if not agent_evidence_orchestration_active(resolved):
        return base

    from agent.llm_client import effective_max_agent_turns

    sandbox_wait = float(getattr(resolved, "sandbox_wait_seconds", 45) or 45)
    turn_budget = effective_max_agent_turns(resolved) * 25

    repo_count = 1
    if isinstance(state, dict):
        github = state.get("github_repo_analyses")
        if isinstance(github, dict):
            urls = github.get("selected_sandbox_repo_urls")
            if isinstance(urls, list) and urls:
                repo_count = max(1, len(urls))

    structure_budget = min(180, 20 * repo_count)
    orchestrated_floor = int(sandbox_wait + structure_budget + turn_budget + 120)
    return max(base, orchestrated_floor)


def sandbox_overlap_active(settings: Settings | None = None) -> bool:
    """True when sandbox should run in parallel with agent/pipeline scoring."""
    resolved = settings or get_settings()
    if (
        resolved.sandbox_deferred_enabled
        or not resolved.sandbox_overlap_enabled
        or sandbox_llm_scoring_active(resolved)
    ):
        return False
    enabled_val = resolved.github_clone_analysis_enabled
    if isinstance(enabled_val, bool):
        return enabled_val
    text = str(enabled_val).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    return text in ("auto", "hybrid")


def sandbox_mode_for_settings(settings: Settings | None = None) -> Literal["inline", "deferred"]:
    """Prep/GitHub analysis mode: inline blocks until sandbox reports exist."""
    resolved = settings or get_settings()
    if (
        resolved.sandbox_deferred_enabled
        or sandbox_overlap_active(resolved)
        or agent_evidence_orchestration_active(resolved)
    ):
        return "deferred"
    return "inline"


def _clone_analysis_enabled(settings: Settings) -> bool:
    enabled_val = settings.github_clone_analysis_enabled
    if isinstance(enabled_val, bool):
        return enabled_val
    text = str(enabled_val).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    return text in ("auto", "hybrid")


def sandbox_required_for_state(state: dict[str, Any], settings: Settings | None = None) -> bool:
    """True when scoring should wait for sandbox reports (not deferred fast-path)."""
    resolved = settings or get_settings()
    if agent_evidence_orchestration_active(resolved):
        return False
    if resolved.sandbox_deferred_enabled or not _clone_analysis_enabled(resolved):
        return False
    github = state.get("github_repo_analyses")
    if not isinstance(github, dict):
        return False
    urls = github.get("selected_sandbox_repo_urls")
    return isinstance(urls, list) and bool(urls)


def _reports_cover_urls(urls: list[str], reports: list[Any]) -> bool:
    if len(reports) < len(urls):
        return False
    for report in reports:
        if not isinstance(report, dict):
            return False
        if report.get("skipped_reason") == _DEFERRED_PENDING_REASON:
            return False
    return True


async def _ensure_sandbox_reports_for_urls(
    state: dict[str, Any],
    urls: list[str],
    *,
    settings: Settings | None = None,
) -> None:
    """Evaluate sandbox for ``urls`` and merge reports into ``github_repo_analyses``."""
    resolved = settings or get_settings()
    if not urls:
        return

    github = state.get("github_repo_analyses")
    if not isinstance(github, dict):
        github = {}
        state["github_repo_analyses"] = github

    from agent.tools.github_analyzer import _evaluate_sandbox_repos

    reports = list(github.get("sandbox_reports") or [])
    by_url: dict[str, dict[str, Any]] = {
        str(report["url"]): report
        for report in reports
        if isinstance(report, dict) and report.get("url")
    }

    async def evaluate_urls(urls_to_eval: list[str], *, label: str) -> None:
        if not urls_to_eval:
            return
        logger.info(
            "%s sandbox on %s repo(s) before scoring application_id=%s",
            label,
            len(urls_to_eval),
            state.get("application_id"),
        )
        for report in await _evaluate_sandbox_repos(urls_to_eval, resolved):
            url = str(report.get("url") or "")
            if url:
                by_url[url] = report

    if _reports_cover_urls(urls, [by_url[u] for u in urls if u in by_url]):
        return

    await evaluate_urls([u for u in urls if u not in by_url], label="Evaluating")
    await evaluate_urls(
        [
            u
            for u in urls
            if by_url.get(u, {}).get("skipped_reason") == _DEFERRED_PENDING_REASON
        ],
        label="Finishing deferred",
    )
    await evaluate_urls(
        [u for u in urls if by_url.get(u, {}).get("timed_out") is True],
        label="Retrying timed-out",
    )

    merged = dict(github)
    if not merged.get("selected_sandbox_repo_urls"):
        merged["selected_sandbox_repo_urls"] = urls
    merged["sandbox_reports"] = [by_url[u] for u in urls if u in by_url]
    state["github_repo_analyses"] = merged


async def ensure_sandbox_before_scoring(state: dict[str, Any]) -> None:
    """Run or finish sandbox evaluation before the judge scores the candidate."""
    settings = get_settings()
    if not sandbox_required_for_state(state, settings):
        return

    github = state.get("github_repo_analyses")
    if not isinstance(github, dict):
        return

    urls = [str(url) for url in github.get("selected_sandbox_repo_urls") or [] if url]
    if not urls:
        return

    await _ensure_sandbox_reports_for_urls(state, urls, settings=settings)


async def force_ensure_sandbox_before_scoring(state: dict[str, Any]) -> None:
    """Run sandbox before scoring even when orchestration defers prep-time evaluation."""
    settings = get_settings()
    if not _clone_analysis_enabled(settings):
        return

    from agent.adk_tools import _github_repo_urls_from_state

    urls = _github_repo_urls_from_state(state)
    if not urls:
        logger.warning(
            "force_ensure_sandbox_before_scoring: no repo URLs application_id=%s",
            state.get("application_id"),
        )
        return

    await _ensure_sandbox_reports_for_urls(state, urls, settings=settings)


def start_sandbox_task(state: dict[str, Any]) -> asyncio.Task[None] | None:
    """Start background sandbox evaluation for overlap mode."""
    if not sandbox_overlap_active():
        return None
    if not sandbox_required_for_state(state):
        return None

    application_id = str(state.get("application_id") or "").strip()
    if not application_id:
        return None

    existing = _SANDBOX_TASKS.get(application_id)
    if existing is not None and not existing.done():
        return existing

    logger.info(
        "Starting parallel sandbox evaluation for application_id=%s",
        application_id,
    )
    task = asyncio.create_task(ensure_sandbox_before_scoring(state))
    _SANDBOX_TASKS[application_id] = task
    _SANDBOX_STATE_REFS[application_id] = state
    return task


def _sync_sandbox_reports(target: dict[str, Any], source: dict[str, Any]) -> None:
    source_github = source.get("github_repo_analyses")
    if not isinstance(source_github, dict):
        return
    reports = source_github.get("sandbox_reports")
    if not reports:
        return
    target_github = target.get("github_repo_analyses")
    if isinstance(target_github, dict):
        merged_github = dict(target_github)
    else:
        merged_github = {}
    merged_github["sandbox_reports"] = reports
    target["github_repo_analyses"] = merged_github


async def await_sandbox_for_scoring(state: dict[str, Any]) -> None:
    """Wait for sandbox reports before scoring or normalizing results."""
    application_id = str(state.get("application_id") or "").strip()
    task = _SANDBOX_TASKS.pop(application_id, None) if application_id else None
    canonical_state = _SANDBOX_STATE_REFS.pop(application_id, None) if application_id else None

    if task is not None:
        await task
        if canonical_state is not None:
            _sync_sandbox_reports(state, canonical_state)
    elif sandbox_required_for_state(state):
        await ensure_sandbox_before_scoring(state)

    register_prep_state(state)


def clear_sandbox_task(application_id: str) -> None:
    """Remove any leftover sandbox task registry entry."""
    application_id = application_id.strip()
    _SANDBOX_TASKS.pop(application_id, None)
    _SANDBOX_STATE_REFS.pop(application_id, None)
