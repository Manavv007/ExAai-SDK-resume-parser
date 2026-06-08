"""Integration tests: domain fixtures via ADK agent path (SCREENING_MODE=agent)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.pipeline import run_screening_async
from agent.prep import prepare_screening_state
from tests.integration.conftest import (
    APP_ID,
    DOMAIN_CASES,
    JOB_ID,
    DomainCase,
    allowlist_ok,
    assert_no_pii_in_payload,
    assert_valid_completed_result,
    batch_fetch_side_effect,
    build_scripted_runner,
    domain_paths,
    load_llm_fixture,
)


@pytest.mark.parametrize("case", DOMAIN_CASES, ids=[c.key for c in DOMAIN_CASES])
@pytest.mark.asyncio
@patch("agent.pipeline.create_runner")
@patch("agent.enrichment.fetch_url_text_batch")
async def test_domain_agent_mode_completed(
    mock_fetch,
    mock_create_runner,
    case: DomainCase,
    test_settings,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCREENING_MODE", "agent")
    monkeypatch.setenv("URL_CACHE_PATH", str(tmp_path / f"agent-{case.key}.db"))
    from agent.config import get_settings

    get_settings.cache_clear()

    resume_path, jd_path = domain_paths(case.key)
    prep = prepare_screening_state(
        application_id=APP_ID,
        job_id=JOB_ID,
        resume_bytes=resume_path.read_bytes(),
        resume_filename="resume.txt",
        jd_text=jd_path.read_text(encoding="utf-8"),
    )

    fetch_url = next(url for url in prep["profile_urls"] if case.crawl_url_substring in url)
    submit_payload = load_llm_fixture(rubric=prep["rubric"], score=80)
    mock_create_runner.return_value = build_scripted_runner(
        fetch_urls=[fetch_url],
        submit_payload=submit_payload,
    )
    mock_fetch.side_effect = batch_fetch_side_effect(f"External profile content for {case.key}.")

    mock_cache = MagicMock()
    mock_cache.get.return_value = None

    with patch("agent.enrichment.get_url_cache", return_value=mock_cache):
        with patch(
            "agent.enrichment.validate_url",
            return_value=type("R", (), {"allowed": True, "reason": None})(),
        ):
            with patch(
                "agent.enrichment.check_allowlist",
                return_value=allowlist_ok(case.domain_category),
            ):
                result = await run_screening_async(
                    application_id=APP_ID,
                    job_id=JOB_ID,
                    resume_bytes=resume_path.read_bytes(),
                    resume_filename="resume.txt",
                    jd_bytes=jd_path.read_bytes(),
                    jd_filename="jd.txt",
                    request_id=f"req-agent-{case.key}",
                )

    assert_valid_completed_result(result, case)
    assert_no_pii_in_payload(result, case.pii_markers)
    mock_fetch.assert_called_once()
    assert fetch_url in mock_fetch.call_args[0][0]
