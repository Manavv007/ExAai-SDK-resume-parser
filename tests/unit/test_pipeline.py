import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.pipeline import (
    create_screening_agent,
    root_agent,
    score_with_validation,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_create_screening_agent_has_tools() -> None:
    agent = create_screening_agent()
    assert agent.name == "resume_screener"
    assert len(agent.tools) >= 2


def test_root_agent_export() -> None:
    assert root_agent.name == "resume_screener"


@patch("agent.pipeline.score_screening_from_state")
def test_score_with_validation_keeps_failed_status(mock_score) -> None:
    from agent.tools.scorer import build_failed_result

    mock_score.return_value = build_failed_result(
        application_id="11111111-1111-4111-8111-111111111111",
        job_id="22222222-2222-4222-8222-222222222222",
        code="LLM_ERROR",
        message="Scoring failed after retry: bad json",
    )
    state = {
        "application_id": "11111111-1111-4111-8111-111111111111",
        "job_id": "22222222-2222-4222-8222-222222222222",
        "resume_text": "text",
    }
    result = score_with_validation(state)
    assert result["resume_screening_status"] == "failed"
    assert result["errors"][0]["code"] == "LLM_ERROR"


@patch("agent.pipeline.score_screening_from_state")
def test_score_with_validation_retry(mock_score) -> None:
    good = json.loads((FIXTURES / "valid_result_completed.json").read_text(encoding="utf-8"))
    bad = {
        "application_id": good["application_id"],
        "job_id": good["job_id"],
        "resume_screening_status": "completed",
    }
    mock_score.side_effect = [bad, good]

    state = {
        "application_id": good["application_id"],
        "job_id": good["job_id"],
        "resume_text": "Engineer",
    }
    result = score_with_validation(state)
    assert result["resume_screening_status"] == "completed"
    assert mock_score.call_count == 2
