import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.agent_runner import SCREENING_AGENT_INSTRUCTION
from agent.config import get_settings
from agent.pipeline import (
    create_screening_agent,
    get_root_agent,
    score_with_validation,
    screening_agent_tools,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _tool_name(tool: object) -> str:
    return str(getattr(tool, "name", None) or getattr(tool, "__name__", tool))


def test_create_screening_agent_has_tools(test_settings) -> None:
    agent = create_screening_agent()
    assert agent.name == "resume_screener"
    tool_names = {_tool_name(tool) for tool in agent.tools}
    expected = {fn.__name__ for fn in screening_agent_tools()}
    assert tool_names == expected
    assert len(agent.tools) == len(expected)
    assert "classify_portfolio_role" in tool_names
    assert "fetch_profiles" in tool_names
    assert "submit_screening_result" in tool_names
    assert "analyze_github" in tool_names


def test_create_screening_agent_uses_configured_model(test_settings) -> None:
    agent = create_screening_agent()
    assert agent.model == test_settings.gemini_model_id


def test_create_screening_agent_uses_openrouter_litellm(
    monkeypatch: pytest.MonkeyPatch,
    test_settings,
) -> None:
    pytest.importorskip("litellm")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-openrouter")
    monkeypatch.setenv("OPENROUTER_MODEL_ID", "openrouter/free")
    get_settings.cache_clear()

    agent = create_screening_agent()
    assert getattr(agent.model, "model", None) == "openrouter/openai/gpt-oss-20b:free"


def test_create_screening_agent_instruction_covers_trust(test_settings) -> None:
    agent = create_screening_agent()
    assert agent.instruction == SCREENING_AGENT_INSTRUCTION
    assert "profile_trust_by_url" in agent.instruction
    assert "scoring_untrusted" in agent.instruction.lower()


def test_root_agent_export(test_settings) -> None:
    assert get_root_agent().name == "resume_screener"


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
