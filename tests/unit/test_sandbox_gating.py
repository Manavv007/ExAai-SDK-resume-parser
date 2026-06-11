"""Sandbox must complete before scoring when deferred mode is off."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent.sandbox_gating import (
    await_sandbox_for_scoring,
    clear_sandbox_task,
    effective_agent_run_timeout_seconds,
    ensure_sandbox_before_scoring,
    sandbox_mode_for_settings,
    sandbox_overlap_active,
    sandbox_required_for_state,
    start_sandbox_task,
)


def test_sandbox_mode_inline_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SANDBOX_DEFERRED_ENABLED", raising=False)
    monkeypatch.delenv("SANDBOX_OVERLAP_ENABLED", raising=False)
    from agent.config import get_settings

    get_settings.cache_clear()
    assert sandbox_mode_for_settings() == "inline"


def test_sandbox_overlap_active_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_CLONE_ANALYSIS_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_DEFERRED_ENABLED", "false")
    monkeypatch.setenv("SANDBOX_OVERLAP_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_LLM_SCORING_ENABLED", "false")
    from agent.config import get_settings

    get_settings.cache_clear()
    assert sandbox_overlap_active() is True
    assert sandbox_mode_for_settings() == "deferred"


def test_sandbox_overlap_inactive_when_deferred_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_CLONE_ANALYSIS_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_DEFERRED_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_OVERLAP_ENABLED", "true")
    from agent.config import get_settings

    get_settings.cache_clear()
    assert sandbox_overlap_active() is False


def test_sandbox_required_when_clone_enabled_and_urls_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_CLONE_ANALYSIS_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_DEFERRED_ENABLED", "false")
    from agent.config import get_settings

    get_settings.cache_clear()
    state = {
        "github_repo_analyses": {
            "selected_sandbox_repo_urls": ["https://github.com/u/r1"],
            "sandbox_reports": [],
        }
    }
    assert sandbox_required_for_state(state) is True


def test_sandbox_not_required_when_deferred_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_CLONE_ANALYSIS_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_DEFERRED_ENABLED", "true")
    from agent.config import get_settings

    get_settings.cache_clear()
    state = {
        "github_repo_analyses": {
            "selected_sandbox_repo_urls": ["https://github.com/u/r1"],
        }
    }
    assert sandbox_required_for_state(state) is False


@pytest.mark.asyncio
async def test_ensure_sandbox_runs_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_CLONE_ANALYSIS_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_DEFERRED_ENABLED", "false")
    from agent.config import get_settings

    get_settings.cache_clear()
    state = {
        "application_id": "11111111-1111-4111-8111-111111111111",
        "github_repo_analyses": {
            "selected_sandbox_repo_urls": ["https://github.com/u/r1"],
            "sandbox_reports": [],
        },
    }
    fake_report = {
        "repo": "u/r1",
        "url": "https://github.com/u/r1",
        "clone_ok": True,
        "summary": "done",
    }

    with patch(
        "agent.tools.github_analyzer._evaluate_sandbox_repos",
        new=AsyncMock(return_value=[fake_report]),
    ) as mock_eval:
        await ensure_sandbox_before_scoring(state)

    mock_eval.assert_awaited_once()
    assert state["github_repo_analyses"]["sandbox_reports"] == [fake_report]


@pytest.mark.asyncio
async def test_start_and_await_sandbox_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_CLONE_ANALYSIS_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_DEFERRED_ENABLED", "false")
    monkeypatch.setenv("SANDBOX_OVERLAP_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_LLM_SCORING_ENABLED", "false")
    from agent.config import get_settings

    get_settings.cache_clear()
    application_id = "11111111-1111-4111-8111-111111111111"
    state = {
        "application_id": application_id,
        "github_repo_analyses": {
            "selected_sandbox_repo_urls": ["https://github.com/u/r1"],
            "sandbox_reports": [],
        },
    }
    fake_report = {
        "repo": "u/r1",
        "url": "https://github.com/u/r1",
        "clone_ok": True,
        "summary": "done",
    }

    with patch(
        "agent.tools.github_analyzer._evaluate_sandbox_repos",
        new=AsyncMock(return_value=[fake_report]),
    ):
        task = start_sandbox_task(state)
        assert task is not None
        await await_sandbox_for_scoring(state)

    assert state["github_repo_analyses"]["sandbox_reports"] == [fake_report]
    clear_sandbox_task(application_id)


def test_resolve_sandbox_pre_run_mode_defaults_none_for_orchestration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCREENING_MODE", "agent")
    monkeypatch.setenv("AGENT_EVIDENCE_ORCHESTRATION_ENABLED", "true")
    from agent.config import get_settings
    from agent.sandbox_gating import resolve_sandbox_pre_run_mode

    get_settings.cache_clear()
    assert resolve_sandbox_pre_run_mode(get_settings()) == "none"


def test_resolve_sandbox_pre_run_mode_full_when_explicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCREENING_MODE", "agent")
    monkeypatch.setenv("AGENT_EVIDENCE_ORCHESTRATION_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_PRE_RUN_MODE", "full")
    from agent.config import get_settings
    from agent.sandbox_gating import resolve_sandbox_pre_run_mode

    get_settings.cache_clear()
    assert resolve_sandbox_pre_run_mode(get_settings()) == "full"


@pytest.mark.asyncio
async def test_run_sandbox_pre_run_skips_when_mode_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCREENING_MODE", "agent")
    monkeypatch.setenv("AGENT_EVIDENCE_ORCHESTRATION_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_PRE_RUN_MODE", "none")
    from agent.config import get_settings
    from agent.sandbox_gating import run_sandbox_pre_run_for_orchestration

    get_settings.cache_clear()
    state = {
        "application_id": "00000000-0000-0000-0000-000000000001",
        "github_repo_analyses": {"selected_sandbox_repo_urls": ["https://github.com/u/r1"]},
    }

    with patch(
        "agent.adk_tools.ensure_sandbox_evidence",
        new=AsyncMock(),
    ) as mock_ensure:
        await run_sandbox_pre_run_for_orchestration(state)
        mock_ensure.assert_not_called()


def test_effective_agent_run_timeout_extends_for_orchestration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCREENING_MODE", "agent")
    monkeypatch.setenv("AGENT_EVIDENCE_ORCHESTRATION_ENABLED", "true")
    monkeypatch.setenv("AGENT_RUN_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("SANDBOX_WAIT_SECONDS", "180")
    monkeypatch.setenv("MAX_AGENT_TURNS", "8")
    from agent.config import get_settings

    get_settings.cache_clear()
    state = {
        "github_repo_analyses": {
            "selected_sandbox_repo_urls": [
                "https://github.com/u/r1",
                "https://github.com/u/r2",
                "https://github.com/u/r3",
            ]
        }
    }
    timeout = effective_agent_run_timeout_seconds(get_settings(), state)
    assert timeout > 120
    assert timeout >= 180
