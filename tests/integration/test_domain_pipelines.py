"""Integration tests: software, design, and academic domain fixtures."""

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
    domain_paths,
    load_llm_fixture,
)


@pytest.fixture(autouse=True)
def _pipeline_screening_mode(pipeline_mode) -> None:
    """Pipeline domain tests require SCREENING_MODE=pipeline."""


@pytest.mark.parametrize("case", DOMAIN_CASES, ids=[c.key for c in DOMAIN_CASES])
@pytest.mark.asyncio
@patch("agent.tools.scorer._generate_json")
@patch("agent.enrichment.fetch_url_text_batch")
async def test_domain_pipeline_completed(
    mock_fetch,
    mock_generate,
    case: DomainCase,
    test_settings,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("URL_CACHE_PATH", str(tmp_path / f"{case.key}.db"))
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

    assert prep["jd_structured"]["domain"] == case.expected_domain
    assert any(case.crawl_url_substring in u for u in prep["profile_urls"])

    mock_generate.return_value = load_llm_fixture(rubric=prep["rubric"])
    mock_fetch.side_effect = batch_fetch_side_effect(
        f"External profile content for {case.key}."
    )

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
                    request_id=f"req-{case.key}",
                )

    assert_valid_completed_result(result, case)
    assert_no_pii_in_payload(result, case.pii_markers)
