
from agent.tools.scorer import (
    _compact_rubric_for_prompt,
    _parse_json_response,
    _scoring_response_schema,
    _try_repair_json,
)


def test_scoring_response_schema_loads() -> None:
    schema = _scoring_response_schema()
    assert schema["type"] == "object"
    assert "resume_similarity_score" in schema["properties"]


def test_compact_rubric_limits_items() -> None:
    rubric = [
        {"criterion": f"req{i}", "weight": "must_have", "requirement_type": "technical_skill"}
        for i in range(20)
    ]
    compact = _compact_rubric_for_prompt(rubric)
    assert len(compact) <= 12


def test_try_repair_truncated_json() -> None:
    broken = '{"resume_similarity_score": {"score": 70, "reasoning": "Good fit'
    repaired = _try_repair_json(broken)
    assert repaired is not None
    assert repaired["resume_similarity_score"]["score"] == 70


def test_parse_json_response_uses_repair() -> None:
    broken = '{"a": 1, "b": "text without end'
    result = _parse_json_response(broken)
    assert result["a"] == 1
