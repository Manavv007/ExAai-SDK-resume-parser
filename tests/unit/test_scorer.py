import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from agent.tools.rubric_builder import build_rubric_bundle
from agent.tools.scorer import (
    _parse_json_response,
    attach_temp_sandbox_reports,
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
    assert result["resume_similarity_score"]["score"] == 85
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


def test_normalize_includes_temp_sandbox_reports(test_settings) -> None:
    fixture = json.loads((FIXTURES / "valid_result_completed.json").read_text(encoding="utf-8"))
    sandbox_reports = [
        {
            "repo": "testuser/repo1",
            "url": "https://github.com/testuser/repo1",
            "provider": "cloud_run",
            "clone_ok": True,
            "summary": "ok",
        }
    ]
    normalized = normalize_screening_result(
        fixture,
        application_id=fixture["application_id"],
        job_id=fixture["job_id"],
        resume_text="resume",
        rubric=[],
        enriched_contents=[],
        github_repo_analyses={"username": "testuser", "sandbox_reports": sandbox_reports},
    )

    assert normalized["temp_sandbox_reports"] == sandbox_reports
    assert validate_result(normalized) is True


def test_attach_temp_sandbox_reports_noop_without_reports() -> None:
    result = {"resume_screening_status": "completed"}
    assert attach_temp_sandbox_reports(result, {}) is result
    assert "temp_sandbox_reports" not in result


def test_normalize_aligns_overall_score_with_high_rubric_matches() -> None:
    """Regression: identity cap must not floor overall score when resume rubric is strong."""
    rubric = [
        {"criterion": "Python", "weight": "must_have", "requirement_type": "technical_skill"},
        {"criterion": "FastAPI", "weight": "nice_to_have", "requirement_type": "technical_skill"},
    ]
    raw = {
        "resume_similarity_score": {"score": 45, "reasoning": "Capped by profiles."},
        "requirement_matches": [
            {
                "requirement": "Python",
                "requirement_type": "technical_skill",
                "match_score": 95,
                "evidence": "Resume lists Python across multiple projects.",
            },
            {
                "requirement": "FastAPI",
                "requirement_type": "technical_skill",
                "match_score": 100,
                "evidence": "Resume cites FastAPI in production APIs.",
            },
        ],
        "recommendation": "advance",
        "recommendation_reasoning": "Strong resume fit.",
        "red_flags": [],
    }

    normalized = normalize_screening_result(
        raw,
        application_id="11111111-1111-4111-8111-111111111111",
        job_id="22222222-2222-4222-8222-222222222222",
        resume_text="resume",
        rubric=rubric,
        enriched_contents=[],
        profile_identity_cap_score=True,
    )

    assert normalized["resume_similarity_score"]["score"] >= 90
    assert normalized["recommendation"] == "advance"


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


def test_normalize_omits_processing_time_ms_when_unknown(test_settings) -> None:
    fixture = json.loads((FIXTURES / "valid_result_completed.json").read_text(encoding="utf-8"))
    fixture["metadata"] = {
        k: v for k, v in fixture["metadata"].items() if k != "processing_time_ms"
    }

    normalized = normalize_screening_result(
        fixture,
        application_id=fixture["application_id"],
        job_id=fixture["job_id"],
        resume_text="resume body text",
        rubric=[],
        enriched_contents=[],
        processing_time_ms=None,
    )

    assert "processing_time_ms" not in normalized["metadata"]
    assert validate_result(normalized) is True


def test_normalize_strips_null_processing_time_ms_from_llm_metadata(test_settings) -> None:
    """Regression: LLM returns metadata.processing_time_ms: null in pipeline mode
    where processing_time_ms hasn't been set yet (None). This must NOT leak None
    into the output — jsonschema rejects null for integer fields."""
    fixture = json.loads((FIXTURES / "valid_result_completed.json").read_text(encoding="utf-8"))
    fixture["metadata"] = {"processing_time_ms": None, "llm_calls": None}

    normalized = normalize_screening_result(
        fixture,
        application_id=fixture["application_id"],
        job_id=fixture["job_id"],
        resume_text="resume body text",
        rubric=[],
        enriched_contents=[],
        processing_time_ms=None,
    )

    assert "processing_time_ms" not in normalized["metadata"]
    assert "llm_calls" not in normalized["metadata"]
    assert validate_result(normalized) is True


@patch("agent.tools.scorer._generate_json")
def test_score_screening_no_processing_time_ms_validates(mock_generate, test_settings) -> None:
    """Regression: score_screening with processing_time_ms=None (pipeline mode) must
    produce a valid completed result, not fail with 'None is not of type integer'."""
    fixture = json.loads((FIXTURES / "valid_result_completed.json").read_text(encoding="utf-8"))
    fixture["requirement_matches"] = [
        {
            "requirement": "Python",
            "requirement_type": "technical_skill",
            "match_score": 85,
            "evidence": "Resume lists Python in multiple roles.",
        }
    ]
    # Simulate LLM returning metadata with null processing_time_ms
    fixture["metadata"] = {"processing_time_ms": None}
    mock_generate.return_value = fixture

    result = score_screening(
        application_id=fixture["application_id"],
        job_id=fixture["job_id"],
        resume_text="Backend engineer with Python and PostgreSQL.",
        jd_raw="Senior Python engineer needed.",
        jd_structured={"domain": "technical", "must_have": ["Python"], "nice_to_have": []},
        processing_time_ms=None,
    )

    assert result["resume_screening_status"] == "completed"
    assert validate_result(result) is True
    # processing_time_ms should either be absent or a valid integer
    meta = result["metadata"]
    if "processing_time_ms" in meta:
        assert isinstance(meta["processing_time_ms"], int)
