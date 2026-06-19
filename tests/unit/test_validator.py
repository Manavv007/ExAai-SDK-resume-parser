import json
from copy import deepcopy
from pathlib import Path

import pytest

from agent.schema import SCHEMA_PATH
from agent.schema.models import ResumeScreeningResult
from agent.tools.validator import parse_result, validate_result, validate_result_detailed

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _load(name: str) -> dict:
    with (FIXTURES / name).open(encoding="utf-8") as f:
        return json.load(f)


def test_schema_file_exists() -> None:
    assert SCHEMA_PATH.is_file()
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        schema = json.load(f)
    assert schema["$id"] == "resume-screening-result-v1"


def test_valid_completed_fixture_passes() -> None:
    data = _load("valid_result_completed.json")
    assert validate_result(data) is True
    model = parse_result(data)
    assert model.resume_screening_status.value == "completed"
    assert model.resume_similarity_score is not None
    assert model.resume_similarity_score.score == 78
    assert model.candidate_integrity is not None
    assert model.candidate_integrity.github_account_timeline.value == "good"


def test_valid_failed_fixture_passes() -> None:
    data = _load("valid_result_failed.json")
    assert validate_result(data) is True
    model = parse_result(data)
    assert model.resume_screening_status.value == "failed"
    assert len(model.errors) == 1


def test_completed_missing_recommendation_fails() -> None:
    data = _load("valid_result_completed.json")
    data = deepcopy(data)
    del data["recommendation"]
    outcome = validate_result_detailed(data)
    assert outcome.ok is False
    assert any("recommendation" in e for e in outcome.errors)


def test_score_out_of_range_fails() -> None:
    data = _load("valid_result_completed.json")
    data = deepcopy(data)
    data["resume_similarity_score"]["score"] = 101
    assert validate_result(data) is False


def test_reasoning_too_long_fails() -> None:
    data = _load("valid_result_completed.json")
    data = deepcopy(data)
    data["resume_similarity_score"]["reasoning"] = "x" * 501
    assert validate_result(data) is False


def test_failed_without_errors_fails() -> None:
    data = _load("valid_result_failed.json")
    data = deepcopy(data)
    data["errors"] = []
    outcome = validate_result_detailed(data)
    assert outcome.ok is False


def test_invalid_uuid_fails() -> None:
    data = _load("valid_result_completed.json")
    data = deepcopy(data)
    data["application_id"] = "not-a-uuid"
    assert validate_result(data) is False


def test_pydantic_rejects_completed_without_score() -> None:
    data = _load("valid_result_completed.json")
    data = deepcopy(data)
    del data["resume_similarity_score"]
    with pytest.raises(ValueError, match="resume_similarity_score"):
        parse_result(data)


def test_round_trip_pydantic_model() -> None:
    data = _load("valid_result_completed.json")
    model = ResumeScreeningResult.model_validate(data)
    roundtrip = model.model_dump(mode="json", exclude_none=True)
    assert validate_result(roundtrip) is True
