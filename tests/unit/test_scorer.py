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
        jd_structured={
            "domain": "technical",
            "role_category": "non_portfolio",
            "must_have": ["Python"],
            "nice_to_have": [],
        },
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
def test_score_screening_skips_portfolio_penalty_in_pipeline_mode(
    mock_generate, test_settings
) -> None:
    fixture = json.loads((FIXTURES / "valid_result_completed.json").read_text(encoding="utf-8"))
    fixture["resume_similarity_score"] = {"score": 92, "reasoning": "Strong resume claims."}
    mock_generate.return_value = fixture

    result = score_screening(
        application_id=fixture["application_id"],
        job_id=fixture["job_id"],
        resume_text="Senior backend engineer with Python.",
        jd_raw="Senior Software Engineer role requiring Python.",
        jd_structured={
            "job_title": "Senior Software Engineer",
            "domain": "technical",
            "role_category": "software_engineering",
            "must_have": ["Python"],
            "nice_to_have": [],
        },
        enriched_contents=[],
        resume_structured={"experience_years": 10},
    )

    assert result["resume_screening_status"] == "completed"
    breakdown = result.get("evaluation_breakdown") or {}
    portfolio = breakdown.get("portfolio_signal") or {}
    assert portfolio.get("role_category_source") == "skipped_pipeline"
    assert not portfolio.get("penalty_applied")
    assert breakdown.get("portfolio_penalty", 0) == 0


@patch("agent.tools.scorer._generate_json")
def test_score_screening_applies_portfolio_penalty_in_agent_mode(
    mock_generate, test_settings
) -> None:
    fixture = json.loads((FIXTURES / "valid_result_completed.json").read_text(encoding="utf-8"))
    fixture["resume_similarity_score"] = {"score": 92, "reasoning": "Strong resume claims."}
    mock_generate.return_value = fixture

    from agent.tools.scorer import normalize_screening_result

    normalized = normalize_screening_result(
        fixture,
        application_id=fixture["application_id"],
        job_id=fixture["job_id"],
        resume_text="Senior backend engineer with Python.",
        rubric=[
            {
                "criterion": "Python",
                "weight": "must_have",
                "requirement_type": "technical_skill",
            }
        ],
        enriched_contents=[],
        jd_structured={"role_category": "software_engineering"},
        resume_structured={"experience_years": 10},
        screening_mode="agent",
        portfolio_role_category="software_engineering",
        portfolio_role_reasoning="Backend SWE role requires GitHub proof.",
        portfolio_role_source="agent",
    )

    breakdown = normalized.get("evaluation_breakdown") or {}
    portfolio = breakdown.get("portfolio_signal") or {}
    assert portfolio.get("role_category_source") == "agent"
    assert portfolio.get("penalty_applied") is True
    assert breakdown.get("portfolio_penalty", 0) > 0


@patch("agent.tools.scorer._generate_json")
def test_score_screening_retries_on_invalid_json(mock_generate, test_settings) -> None:
    fixture = json.loads((FIXTURES / "valid_result_completed.json").read_text(encoding="utf-8"))
    mock_generate.side_effect = [ValueError("bad"), fixture]

    result = score_screening(
        application_id=fixture["application_id"],
        job_id=fixture["job_id"],
        resume_text="Engineer with Python.",
        jd_raw="Need Python",
        jd_structured={
            "domain": "technical",
            "role_category": "non_portfolio",
            "must_have": ["Python"],
            "nice_to_have": [],
        },
    )

    assert mock_generate.call_count == 2
    assert result["resume_screening_status"] == "completed"


@patch("agent.tools.scorer._generate_json")
def test_score_screening_stops_after_rate_limit(mock_generate, test_settings) -> None:
    from agent.llm_client import mark_gemini_rate_limited, reset_llm_call_count

    reset_llm_call_count()
    mark_gemini_rate_limited()
    mock_generate.side_effect = Exception("RateLimitError: 429")

    result = score_screening(
        application_id="11111111-1111-4111-8111-111111111111",
        job_id="22222222-2222-4222-8222-222222222222",
        resume_text="text",
        jd_raw="jd",
        jd_structured={"domain": "general", "must_have": [], "nice_to_have": []},
        max_llm_attempts=3,
    )

    assert mock_generate.call_count == 1
    assert result["resume_screening_status"] == "failed"


@patch("agent.tools.scorer._generate_json")
def test_score_screening_respects_max_llm_attempts(mock_generate, test_settings) -> None:
    mock_generate.side_effect = ValueError("still bad")

    score_screening(
        application_id="11111111-1111-4111-8111-111111111111",
        job_id="22222222-2222-4222-8222-222222222222",
        resume_text="text",
        jd_raw="jd",
        jd_structured={"domain": "general", "must_have": [], "nice_to_have": []},
        max_llm_attempts=1,
    )

    assert mock_generate.call_count == 1


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

    assert mock_generate.call_count == 2
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


def test_attach_temp_sandbox_reports_includes_empty_list_when_github_ran() -> None:
    result = {"resume_screening_status": "completed"}
    attach_temp_sandbox_reports(
        result,
        {"username": "Manavv007", "sandbox_reports": []},
    )
    assert result["temp_sandbox_reports"] == []


def test_attach_temp_sandbox_reports_noop_without_github_username() -> None:
    result = {"resume_screening_status": "completed"}
    assert attach_temp_sandbox_reports(result, {}) is result
    assert "temp_sandbox_reports" not in result


def test_attach_temp_sandbox_reports_without_username_when_reports_present() -> None:
    sandbox_reports = [
        {
            "repo": "owner/repo",
            "url": "https://github.com/owner/repo",
            "clone_ok": True,
            "repo_profile": {"git_profile": {"commit_count": 5}},
        }
    ]
    result = {"resume_screening_status": "completed"}
    attach_temp_sandbox_reports(result, {"sandbox_reports": sandbox_reports})
    assert result["temp_sandbox_reports"] == sandbox_reports


def test_attach_temp_sandbox_reports_preserves_existing_when_realign_empty() -> None:
    existing = [{"repo": "owner/repo", "url": "https://github.com/owner/repo", "clone_ok": True}]
    result = {"resume_screening_status": "completed", "temp_sandbox_reports": existing}
    attach_temp_sandbox_reports(
        result,
        {
            "username": "owner",
            "selected_sandbox_repo_urls": ["https://github.com/other/missing"],
            "sandbox_reports": [],
        },
    )
    assert result["temp_sandbox_reports"] == existing


def test_normalize_ignores_inflated_llm_overall_score() -> None:
    """Overall score must follow rubric match_scores, not a higher LLM headline score."""
    rubric = [
        {"criterion": "Python", "weight": "must_have", "requirement_type": "technical_skill"},
        {"criterion": "FastAPI", "weight": "nice_to_have", "requirement_type": "technical_skill"},
    ]
    raw = {
        "resume_similarity_score": {"score": 85, "reasoning": "Excellent fit."},
        "requirement_matches": [
            {
                "requirement": "Python",
                "requirement_type": "technical_skill",
                "match_score": 72,
                "evidence": "Resume lists Python.",
            },
            {
                "requirement": "FastAPI",
                "requirement_type": "technical_skill",
                "match_score": 68,
                "evidence": "Some FastAPI exposure.",
            },
        ],
        "recommendation": "advance",
        "recommendation_reasoning": "Good match.",
        "red_flags": [],
    }

    normalized = normalize_screening_result(
        raw,
        application_id="11111111-1111-4111-8111-111111111111",
        job_id="22222222-2222-4222-8222-222222222222",
        resume_text="resume",
        rubric=rubric,
        enriched_contents=[],
    )

    # (70*2 + 70*1) / 3 = 70 after 5-point quantization of 72 and 68
    assert normalized["resume_similarity_score"]["score"] == 70


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
        jd_structured={
            "domain": "technical",
            "role_category": "non_portfolio",
            "must_have": ["Python"],
            "nice_to_have": [],
        },
        processing_time_ms=None,
    )

    assert result["resume_screening_status"] == "completed"
    assert validate_result(result) is True
    # processing_time_ms should either be absent or a valid integer
    meta = result["metadata"]
    if "processing_time_ms" in meta:
        assert isinstance(meta["processing_time_ms"], int)


def test_normalize_attaches_candidate_integrity_without_changing_fit_score() -> None:
    rubric = [
        {"criterion": "Python", "weight": "must_have", "requirement_type": "technical_skill"},
    ]
    raw = {
        "resume_similarity_score": {"score": 82, "reasoning": "Strong Python fit."},
        "requirement_matches": [
            {
                "requirement": "Python",
                "requirement_type": "technical_skill",
                "match_score": 82,
                "evidence": "Resume lists Python.",
            }
        ],
        "recommendation": "advance",
        "recommendation_reasoning": "Good match.",
        "red_flags": [],
    }
    github = {
        "user_profile": {
            "login": "alice",
            "html_url": "https://github.com/alice",
            "created_at": "2018-01-01T00:00:00Z",
        },
        "activity_timeline": {"earliest_activity_at": "2020-01-01T00:00:00Z"},
    }

    baseline = normalize_screening_result(
        raw,
        application_id="11111111-1111-4111-8111-111111111111",
        job_id="22222222-2222-4222-8222-222222222222",
        resume_text="resume",
        rubric=rubric,
        enriched_contents=[],
        github_repo_analyses=None,
        profile_urls=["https://github.com/alice"],
    )
    normalized = normalize_screening_result(
        raw,
        application_id="11111111-1111-4111-8111-111111111111",
        job_id="22222222-2222-4222-8222-222222222222",
        resume_text="resume",
        rubric=rubric,
        enriched_contents=[],
        github_repo_analyses=github,
        profile_urls=["https://github.com/alice"],
    )

    assert (
        normalized["resume_similarity_score"]["score"]
        == baseline["resume_similarity_score"]["score"]
    )
    assert "candidate_integrity" in normalized
    integrity = normalized["candidate_integrity"]
    assert integrity["github_account_timeline"] == "good"
    assert integrity["linkedin_contact_links"] in ("good", "bad", "not_enough_evidence")
    assert normalized["integrity_signals"]
