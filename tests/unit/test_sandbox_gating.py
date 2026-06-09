"""Sandbox must complete before scoring when deferred mode is off."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent.sandbox_gating import (
    ensure_sandbox_before_scoring,
    sandbox_mode_for_settings,
    sandbox_required_for_state,
)


def test_sandbox_mode_inline_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SANDBOX_DEFERRED_ENABLED", raising=False)
    from agent.config import get_settings

    get_settings.cache_clear()
    assert sandbox_mode_for_settings() == "inline"


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
