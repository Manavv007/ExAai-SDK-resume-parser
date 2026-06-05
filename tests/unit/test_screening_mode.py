"""SCREENING_MODE feature flag (pipeline vs agent)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent.pipeline import run_screening_async

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
APP_ID = "11111111-1111-4111-8111-111111111111"
JOB_ID = "22222222-2222-4222-8222-222222222222"


@pytest.mark.asyncio
@patch("agent.agent_runner.run_screening_agent_async", new_callable=AsyncMock)
@patch("agent.pipeline.run_screening_pipeline_async", new_callable=AsyncMock)
async def test_run_screening_async_uses_agent_mode(
    mock_pipeline: AsyncMock,
    mock_agent: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCREENING_MODE", "agent")
    from agent.config import get_settings

    get_settings.cache_clear()

    completed = json.loads((FIXTURES / "valid_result_completed.json").read_text(encoding="utf-8"))
    mock_agent.return_value = completed

    resume = (FIXTURES / "sample_resume.txt").read_bytes()
    jd = (FIXTURES / "sample_jd.txt").read_bytes()

    result = await run_screening_async(
        application_id=APP_ID,
        job_id=JOB_ID,
        resume_bytes=resume,
        resume_filename="resume.txt",
        jd_bytes=jd,
        jd_filename="jd.txt",
    )

    mock_agent.assert_awaited_once()
    mock_pipeline.assert_not_awaited()
    assert result["resume_screening_status"] == "completed"


@pytest.mark.asyncio
@patch("agent.agent_runner.run_screening_agent_async", new_callable=AsyncMock)
@patch("agent.pipeline.run_screening_pipeline_async", new_callable=AsyncMock)
async def test_run_screening_async_uses_pipeline_mode(
    mock_pipeline: AsyncMock,
    mock_agent: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCREENING_MODE", "pipeline")
    from agent.config import get_settings

    get_settings.cache_clear()

    completed = json.loads((FIXTURES / "valid_result_completed.json").read_text(encoding="utf-8"))
    mock_pipeline.return_value = completed

    resume = (FIXTURES / "sample_resume.txt").read_bytes()
    jd = (FIXTURES / "sample_jd.txt").read_bytes()

    result = await run_screening_async(
        application_id=APP_ID,
        job_id=JOB_ID,
        resume_bytes=resume,
        resume_filename="resume.txt",
        jd_bytes=jd,
        jd_filename="jd.txt",
    )

    mock_pipeline.assert_awaited_once()
    mock_agent.assert_not_awaited()
    assert result["resume_screening_status"] == "completed"
