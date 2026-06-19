"""ADK agent screening path (Runner + tools, mocked LLM)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService

from agent.agent_runner import run_screening_agent_async
from agent.pipeline import create_screening_agent
from agent.prep import prepare_screening_state
from agent.tools.validator import validate_result
from tests.integration.conftest import (
    APP_ID,
    JOB_ID,
    agent_text_response,
    batch_fetch_side_effect,
    build_scripted_runner,
    load_llm_fixture,
)


@pytest.mark.asyncio
@patch(
    "agent.enrichment.fetch_url_text_batch",
    side_effect=batch_fetch_side_effect("Open source Python projects."),
)
async def test_agent_screening_mocked_tool_flow(
    mock_fetch,
    test_settings,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("URL_CACHE_PATH", str(tmp_path / "cache.db"))
    from agent.config import get_settings

    get_settings.cache_clear()

    resume = b"Alex Chen\nPython engineer\nhttps://github.com/alexchen-dev\n"
    jd = b"Senior Python Engineer\nMust have: Python, FastAPI\n"

    state = prepare_screening_state(
        application_id=APP_ID,
        job_id=JOB_ID,
        resume_bytes=resume,
        resume_filename="resume.txt",
        jd_text=jd.decode("utf-8"),
    )
    state["request_id"] = "req-agent-1"
    state["processing_time_ms"] = 1200
    state["screening_mode"] = "agent"

    trusted_url = next(
        (url for url in state["profile_urls"] if "github.com" in url),
        state["profile_urls"][0] if state["profile_urls"] else "https://github.com/alexchen-dev",
    )
    submit_payload = load_llm_fixture(
        requirement="Python",
        requirement_type="technical_skill",
        score=82,
        rubric=state.get("rubric"),
    )

    runner = build_scripted_runner(
        fetch_urls=[trusted_url],
        submit_payload=submit_payload,
    )

    with patch(
        "agent.enrichment.validate_url",
        return_value=type("R", (), {"allowed": True, "reason": None})(),
    ):
        with patch(
            "agent.enrichment.check_allowlist",
            return_value=type(
                "R",
                (),
                {"allowed": True, "reason": None, "domain_category": "code"},
            )(),
        ):
            result = await run_screening_agent_async(state, runner=runner)

    assert result["resume_screening_status"] == "completed"
    assert validate_result(result)
    assert result["application_id"] == APP_ID
    assert result["job_id"] == JOB_ID
    assert mock_fetch.call_count == 1
    assert trusted_url in mock_fetch.call_args[0][0]
    assert len(result.get("sources_crawled") or []) >= 1


@pytest.mark.asyncio
@patch("agent.pipeline.score_with_validation")
async def test_agent_screening_no_submit_returns_failed(
    mock_score,
    test_settings,
) -> None:
    from agent.tools.scorer import build_failed_result

    mock_score.return_value = build_failed_result(
        application_id=APP_ID,
        job_id=JOB_ID,
        code="LLM_ERROR",
        message="fallback failed",
        resume_text="Engineer",
    )
    state = {
        "application_id": APP_ID,
        "job_id": JOB_ID,
        "request_id": "req-agent-fail",
        "resume_text": "Engineer",
        "jd_raw": "Role",
        "rubric": [],
        "rubric_preamble": "",
        "profile_urls": [],
        "profile_trust_by_url": {},
        "identity_red_flags": [],
        "enriched_contents": [],
        "processing_time_ms": 50,
    }

    async def _never_submit(*, callback_context, llm_request):
        return agent_text_response("I forgot to submit.")

    agent = create_screening_agent(before_model_callback=_never_submit)
    runner = Runner(
        app=App(name="exaai_adk", root_agent=agent),
        session_service=InMemorySessionService(),
        auto_create_session=False,
    )

    result = await run_screening_agent_async(
        state,
        runner=runner,
    )

    assert result["resume_screening_status"] == "failed"
    assert result["errors"][0]["code"] == "LLM_ERROR"
    assert "submit_screening_result" in result["errors"][0]["message"]
    mock_score.assert_called_once()


@pytest.mark.asyncio
@patch("agent.pipeline.score_with_validation")
async def test_agent_screening_no_submit_uses_score_fallback(
    mock_score,
    test_settings,
) -> None:
    from tests.integration.conftest import load_llm_fixture

    mock_score.return_value = load_llm_fixture(score=72)

    state = {
        "application_id": APP_ID,
        "job_id": JOB_ID,
        "request_id": "req-agent-fallback",
        "resume_text": "Python engineer with FastAPI experience.",
        "jd_raw": "Senior Python Engineer",
        "rubric": [
            {
                "criterion": "Python",
                "weight": "must_have",
                "requirement_type": "technical_skill",
            }
        ],
        "rubric_preamble": "",
        "profile_urls": [],
        "profile_trust_by_url": {},
        "identity_red_flags": [],
        "enriched_contents": [],
        "processing_time_ms": 50,
    }

    async def _never_submit(*, callback_context, llm_request):
        return agent_text_response("Done.")

    agent = create_screening_agent(before_model_callback=_never_submit)
    runner = Runner(
        app=App(name="exaai_adk", root_agent=agent),
        session_service=InMemorySessionService(),
        auto_create_session=False,
    )

    result = await run_screening_agent_async(state, runner=runner)

    assert result["resume_screening_status"] == "completed"
    assert result["metadata"].get("agent_submit_fallback") is True
    mock_score.assert_called_once()


@pytest.mark.asyncio
@patch("agent.agent_runner._consume_agent_run")
async def test_agent_screening_timeout_returns_failed(
    mock_consume,
    test_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_RUN_TIMEOUT_SECONDS", "1")
    from agent.config import get_settings

    get_settings.cache_clear()

    async def _timeout(*_args, **_kwargs):
        raise TimeoutError

    mock_consume.side_effect = _timeout

    state = {
        "application_id": APP_ID,
        "job_id": JOB_ID,
        "request_id": "req-agent-timeout",
        "resume_text": "Engineer",
        "jd_raw": "Role",
        "rubric": [],
        "rubric_preamble": "",
        "profile_urls": [],
        "profile_trust_by_url": {},
        "identity_red_flags": [],
        "enriched_contents": [],
    }

    result = await run_screening_agent_async(state)

    assert result["resume_screening_status"] == "failed"
    assert result["errors"][0]["code"] == "AGENT_TIMEOUT"
