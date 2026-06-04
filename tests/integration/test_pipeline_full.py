"""Full pipeline with mocked Exa and Gemini."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.pipeline import run_screening_async
from agent.tools.validator import validate_result

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.mark.asyncio
@patch("agent.tools.scorer._generate_json")
@patch("agent.enrichment.fetch_url_text", return_value="Open source Python projects.")
async def test_full_pipeline_completed(
    mock_fetch,
    mock_generate,
    test_settings,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("URL_CACHE_PATH", str(tmp_path / "cache.db"))
    from agent.config import get_settings

    get_settings.cache_clear()

    fixture = json.loads((FIXTURES / "valid_result_completed.json").read_text(encoding="utf-8"))
    fixture["requirement_matches"] = [
        {
            "requirement": "Python",
            "requirement_type": "technical_skill",
            "match_score": 85,
            "evidence": "Resume lists Python in multiple roles.",
        }
    ]
    mock_generate.return_value = fixture

    resume = (FIXTURES / "sample_resume.txt").read_bytes()
    jd = (FIXTURES / "sample_jd.txt").read_bytes()

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
            result = await run_screening_async(
                application_id=fixture["application_id"],
                job_id=fixture["job_id"],
                resume_bytes=resume,
                resume_filename="resume.txt",
                jd_bytes=jd,
                jd_filename="jd.txt",
                request_id="req-integration-1",
            )

    assert result["resume_screening_status"] == "completed"
    assert validate_result(result)
    assert result["metadata"]["processing_time_ms"] is not None
    if mock_fetch.called:
        assert result.get("sources_crawled") is not None
