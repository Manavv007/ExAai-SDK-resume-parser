"""Parallel sandbox + agent scoring overlap."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.adk_tools import submit_screening_result
from agent.agent_runner import build_agent_user_message
from agent.sandbox_gating import start_sandbox_task

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
APP_ID = "11111111-1111-4111-8111-111111111111"
JOB_ID = "22222222-2222-4222-8222-222222222222"


@pytest.mark.asyncio
async def test_submit_awaits_sandbox_and_applies_penalty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_CLONE_ANALYSIS_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_DEFERRED_ENABLED", "false")
    monkeypatch.setenv("SANDBOX_OVERLAP_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_LLM_SCORING_ENABLED", "false")
    from agent.config import get_settings
    from agent.prep_context import register_prep_state

    get_settings.cache_clear()

    fake_report = {
        "repo": "u/chaos",
        "url": "https://github.com/u/chaos",
        "classification": "aligned",
        "clone_ok": True,
        "summary": "risky",
        "repo_profile": {
            "security_profile": {"secret_hygiene": "weak", "secret_pattern_hits": 2},
            "external_tool_signals": {},
        },
        "quality_signals": {"has_tests": False, "has_ci": False},
        "findings": [],
    }
    state = {
        "application_id": APP_ID,
        "job_id": JOB_ID,
        "resume_text": "Python engineer.",
        "jd_raw": "Python role",
        "jd_structured": {},
        "rubric": [
            {
                "criterion": "Python",
                "weight": "must_have",
                "requirement_type": "technical_skill",
            }
        ],
        "github_repo_analyses": {
            "selected_sandbox_repo_urls": ["https://github.com/u/chaos"],
            "sandbox_reports": [],
        },
        "enriched_contents": [],
        "identity_red_flags": [],
        "profile_identity_cap_score": False,
    }
    register_prep_state(state)

    eval_started = asyncio.Event()
    eval_finished = asyncio.Event()

    async def slow_eval(urls, settings, **_kwargs):
        eval_started.set()
        await asyncio.sleep(0.05)
        eval_finished.set()
        return [fake_report]

    ctx = MagicMock()
    ctx.state = dict(state)

    payload = {
        "resume_similarity_score": {"score": 80, "reasoning": "Strong fit."},
        "requirement_matches": [
            {
                "requirement": "Python",
                "requirement_type": "technical_skill",
                "match_score": 85,
                "evidence": "Resume lists Python.",
            }
        ],
        "recommendation": "advance",
        "recommendation_reasoning": "Good match.",
        "red_flags": [],
    }

    with patch(
        "agent.tools.github_analyzer._evaluate_sandbox_repos",
        new=AsyncMock(side_effect=slow_eval),
    ):
        task = start_sandbox_task(state)
        assert task is not None
        submit_task = asyncio.create_task(submit_screening_result(payload, ctx))
        await asyncio.wait_for(eval_started.wait(), timeout=1.0)
        result = await submit_task

    assert eval_finished.is_set()
    assert result["ok"] is True
    score = result["screening_result"]["resume_similarity_score"]["score"]
    assert score < 80


def test_agent_brief_notes_pending_sandbox_when_overlap_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_OVERLAP_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_DEFERRED_ENABLED", "false")
    monkeypatch.setenv("SANDBOX_LLM_SCORING_ENABLED", "false")
    from agent.config import get_settings

    get_settings.cache_clear()

    state = {
        "application_id": APP_ID,
        "job_id": JOB_ID,
        "resume_text": "Engineer",
        "jd_raw": "Role",
        "rubric_preamble": "",
        "rubric": [],
        "profile_urls": [],
        "profile_trust_by_url": {},
        "identity_red_flags": [],
        "profile_identity_cap_score": False,
        "github_repo_analyses": {
            "username": "janedoe",
            "repo_analyses": [],
            "selected_sandbox_repo_urls": ["https://github.com/u/r1"],
            "sandbox_reports": [],
        },
    }

    message = build_agent_user_message(state)

    assert "evaluation in progress" in message.lower()


@pytest.mark.asyncio
@patch("agent.agent_runner.run_screening_agent_async", new_callable=AsyncMock)
@patch("agent.pipeline.run_screening_pipeline_async", new_callable=AsyncMock)
@patch("agent.pipeline.ensure_sandbox_before_scoring", new_callable=AsyncMock)
@patch("agent.pipeline.start_sandbox_task")
async def test_run_screening_starts_sandbox_task_without_pre_await(
    mock_start_task: MagicMock,
    mock_ensure: AsyncMock,
    mock_pipeline: AsyncMock,
    mock_agent: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.pipeline import run_screening_async

    monkeypatch.setenv("SCREENING_MODE", "agent")
    monkeypatch.setenv("SANDBOX_OVERLAP_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_DEFERRED_ENABLED", "false")
    monkeypatch.setenv("SANDBOX_LLM_SCORING_ENABLED", "false")
    from agent.config import get_settings

    get_settings.cache_clear()

    completed = json.loads((FIXTURES / "valid_result_completed.json").read_text(encoding="utf-8"))
    mock_agent.return_value = completed
    mock_start_task.return_value = None

    resume = (FIXTURES / "sample_resume.txt").read_bytes()
    jd = (FIXTURES / "sample_jd.txt").read_bytes()

    with patch("agent.pipeline._await_github_prep", new_callable=AsyncMock):
        await run_screening_async(
            application_id=APP_ID,
            job_id=JOB_ID,
            resume_bytes=resume,
            resume_filename="resume.txt",
            jd_bytes=jd,
            jd_filename="jd.txt",
        )

    mock_start_task.assert_called_once()
    mock_ensure.assert_not_awaited()
