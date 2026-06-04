import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from agent.tools.rubric_builder import build_rubric_bundle
from agent.tools.scorer import (
    _parse_json_response,
    normalize_screening_result,
    score_screening,
)
from agent.tools.validator import validate_result

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_parse_json_response_strips_markdown_fence() -> None:
    raw = '```json\n{"score": 1}\n```'
    assert _parse_json_response(raw)["score"] == 1


@patch("agent.tools.scorer._generate_json")
def test_score_screening_success(mock_generate, test_settings) -> None:
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

    jd = (FIXTURES / "sample_jd.txt").read_text(encoding="utf-8")
    bundle = build_rubric_bundle(
        {"domain": "technical", "must_have": ["Python"], "nice_to_have": []}
    )

    result = score_screening(
        application_id=fixture["application_id"],
        job_id=fixture["job_id"],
        resume_text="Backend engineer with Python and PostgreSQL.",
        jd_raw=jd,
        jd_structured={"domain": "technical", "must_have": ["Python"], "nice_to_have": []},
        rubric=bundle["rubric"],
        rubric_preamble=bundle["rubric_preamble"],
        enriched_contents=[
            {
                "url": "https://github.com/example",
                "content": (
                    "===BEGIN EXTERNAL CONTENT: https://github.com/example===\n"
                    "OSS\n===END EXTERNAL CONTENT==="
                ),
                "domain_category": "code",
            }
        ],
    )

    assert result["resume_screening_status"] == "completed"
    assert validate_result(result) is True
    assert result["recommendation"] in {"advance", "hold", "reject"}
    assert result["resume_similarity_score"]["score"] == 78
    assert len(result["requirement_matches"]) >= 1


@patch("agent.tools.scorer._generate_json")
def test_score_screening_retries_on_invalid_json(mock_generate, test_settings) -> None:
    fixture = json.loads((FIXTURES / "valid_result_completed.json").read_text(encoding="utf-8"))
    mock_generate.side_effect = [ValueError("bad"), fixture]

    result = score_screening(
        application_id=fixture["application_id"],
        job_id=fixture["job_id"],
        resume_text="Engineer with Python.",
        jd_raw="Need Python",
        jd_structured={"domain": "technical", "must_have": ["Python"], "nice_to_have": []},
    )

    assert mock_generate.call_count == 2
    assert result["resume_screening_status"] == "completed"


@patch("agent.tools.scorer._generate_json")
def test_score_screening_failed_after_exhausted_retries(mock_generate, test_settings) -> None:
    mock_generate.side_effect = ValueError("still bad")

    result = score_screening(
        application_id="11111111-1111-4111-8111-111111111111",
        job_id="22222222-2222-4222-8222-222222222222",
        resume_text="text",
        jd_raw="jd",
        jd_structured={"domain": "general", "must_have": [], "nice_to_have": []},
    )

    assert mock_generate.call_count == 3
    assert result["resume_screening_status"] == "failed"
    assert result["errors"][0]["code"] == "LLM_ERROR"


def test_normalize_maps_recommendation_aliases() -> None:
    fixture = json.loads((FIXTURES / "valid_result_completed.json").read_text(encoding="utf-8"))
    raw = deepcopy(fixture)
    raw["recommendation"] = "yes"

    normalized = normalize_screening_result(
        raw,
        application_id=fixture["application_id"],
        job_id=fixture["job_id"],
        resume_text="resume",
        rubric=[],
        enriched_contents=[],
    )
    assert normalized["recommendation"] == "advance"
