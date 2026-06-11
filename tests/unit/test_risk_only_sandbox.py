"""Tests for risk-only sandbox pre-pass (vulns without file excerpts)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent.sandbox.evaluator.repo_profiler import profile_repository
from agent.tools.repo_focus import (
    build_risk_only_focus_spec,
    is_risk_only_evaluation,
)


def test_build_risk_only_focus_spec() -> None:
    spec = build_risk_only_focus_spec(
        repo_role="aligned",
        candidate_tags=["backend_engineer"],
        file_paths=["app/main.py"],
    )
    assert is_risk_only_evaluation(spec) is True
    assert spec["focus_paths"] == []
    assert spec["top_files_count"] == 0


def test_profile_repository_risk_only_skips_file_samples(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("api_key = 'secret'\n", encoding="utf-8")
    (repo / "requirements.txt").write_text("flask\n", encoding="utf-8")

    empty_tool = lambda _repo_dir: {}  # noqa: E731
    monkeypatch.setattr("agent.sandbox.evaluator.repo_profiler.run_scc", empty_tool)
    monkeypatch.setattr("agent.sandbox.evaluator.repo_profiler.run_pip_audit", empty_tool)
    monkeypatch.setattr("agent.sandbox.evaluator.repo_profiler.run_npm_audit", empty_tool)
    monkeypatch.setattr("agent.sandbox.evaluator.repo_profiler.run_trivy_fs", empty_tool)
    monkeypatch.setattr("agent.sandbox.evaluator.repo_profiler.run_semgrep", empty_tool)
    monkeypatch.setattr("agent.sandbox.evaluator.repo_profiler.run_checkov", empty_tool)
    monkeypatch.setattr("agent.sandbox.evaluator.repo_profiler.run_hadolint", empty_tool)
    monkeypatch.setattr(
        "agent.sandbox.evaluator.repo_profiler.calculate_git_metrics",
        lambda _repo_dir: {
            "commit_count": 0,
            "unique_authors": 0,
            "days_since_last_commit": None,
            "top_author_commit_share": 0.0,
            "sole_author": False,
            "history_is_shallow": True,
        },
    )
    monkeypatch.setattr(
        "agent.sandbox.evaluator.repo_profiler.build_security_profile",
        lambda _repo_dir, _hits: {
            "secret_pattern_hits": 0,
            "secret_hygiene": "ok",
            "has_env_file": False,
            "has_env_example": False,
        },
    )
    monkeypatch.setattr(
        "agent.sandbox.evaluator.repo_profiler.calculate_secret_hits",
        lambda _repo_dir: 0,
    )
    monkeypatch.setattr(
        "agent.sandbox.evaluator.repo_profiler.build_documentation_profile",
        lambda _repo_dir: {
            "has_setup_instructions": False,
            "has_architecture_notes": False,
            "readme_quality": "minimal",
        },
    )

    spec = build_risk_only_focus_spec(
        repo_role="aligned",
        candidate_tags=["backend_engineer"],
    )
    profile, _findings = profile_repository(repo, ["python"], focus_spec=spec)

    assert profile["evaluation_mode"] == "risk_only"
    assert profile["sample_files"] == []
    assert profile["top_files"] == []
    assert "external_tool_signals" in profile


@pytest.mark.asyncio
async def test_run_risk_only_sandbox_pre_pass_marks_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCREENING_MODE", "agent")
    monkeypatch.setenv("AGENT_EVIDENCE_ORCHESTRATION_ENABLED", "true")
    from agent.adk_tools import run_risk_only_sandbox_pre_pass
    from agent.config import get_settings

    get_settings.cache_clear()

    state = {
        "application_id": "00000000-0000-0000-0000-000000000001",
        "github_repo_analyses": {
            "selected_sandbox_repo_urls": ["https://github.com/dev/service"],
            "candidate_tags": ["backend_engineer"],
        },
    }

    fake_report = {
        "repo": "dev/service",
        "url": "https://github.com/dev/service",
        "clone_ok": True,
        "evaluation_mode": "risk_only",
        "repo_profile": {"evaluation_mode": "risk_only", "top_files": [], "sample_files": []},
        "findings": [],
    }

    with (
        patch(
            "agent.adk_tools._fetch_repo_tree_paths",
            new=AsyncMock(return_value=(["app/main.py"], {})),
        ),
        patch(
            "agent.adk_tools._evaluate_sandbox_repos",
            new=AsyncMock(return_value=[fake_report]),
        ),
    ):
        ran = await run_risk_only_sandbox_pre_pass(state)

    assert ran is True
    assert state["sandbox_risk_only_pre_pass"] is True
    assert state["github_repo_analyses"]["sandbox_risk_only_pre_pass"] is True
    assert state.get("sandbox_completed_by_agent") is not True


@pytest.mark.asyncio
async def test_run_sandbox_pre_run_risk_only_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCREENING_MODE", "agent")
    monkeypatch.setenv("AGENT_EVIDENCE_ORCHESTRATION_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_PRE_RUN_MODE", "risk_only")
    from agent.config import get_settings
    from agent.sandbox_gating import run_sandbox_pre_run_for_orchestration

    get_settings.cache_clear()
    state = {
        "application_id": "00000000-0000-0000-0000-000000000001",
        "github_repo_analyses": {
            "selected_sandbox_repo_urls": ["https://github.com/dev/service"],
        },
    }

    with patch(
        "agent.adk_tools.run_risk_only_sandbox_pre_pass",
        new=AsyncMock(return_value=True),
    ) as mock_pre:
        await run_sandbox_pre_run_for_orchestration(state)
        mock_pre.assert_awaited_once()
