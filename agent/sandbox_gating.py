"""Wait for sandbox evaluation before candidate scoring when required."""

from __future__ import annotations

import logging
from typing import Any, Literal

from agent.config import Settings, get_settings

logger = logging.getLogger("exaai_adk.sandbox_gating")

_DEFERRED_PENDING_REASON = "Deferred sandbox evaluation pending."


def sandbox_mode_for_settings(settings: Settings | None = None) -> Literal["inline", "deferred"]:
    """Prep/GitHub analysis mode: inline blocks until sandbox reports exist."""
    resolved = settings or get_settings()
    return "deferred" if resolved.sandbox_deferred_enabled else "inline"


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
        for report in await _evaluate_sandbox_repos(urls_to_eval, settings):
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
    merged["sandbox_reports"] = [by_url[u] for u in urls if u in by_url]
    state["github_repo_analyses"] = merged
