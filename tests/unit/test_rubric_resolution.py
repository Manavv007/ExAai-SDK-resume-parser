"""Tests for rubric resolution and incomplete agent output detection."""

from __future__ import annotations

from agent.submit import process_screening_submission
from agent.tools.rubric_builder import (
    build_rubric_bundle,
    requirement_matches_need_rescore,
    resolve_session_rubric,
)


def test_resolve_session_rubric_rebuilds_from_jd_structured() -> None:
    bundle = build_rubric_bundle(
        {
            "domain": "technical",
            "must_have": ["Python fundamentals"],
            "nice_to_have": ["Docker"],
        }
    )
    state = {
        "rubric": [],
        "jd_structured": {
            "domain": "technical",
            "must_have": ["Python fundamentals"],
            "nice_to_have": ["Docker"],
        },
    }

    rubric = resolve_session_rubric(state)

    assert len(rubric) == len(bundle["rubric"])
    assert rubric[0]["criterion"] == "Python fundamentals"


def test_submit_uses_jd_structured_when_rubric_missing_from_session() -> None:
    bundle = build_rubric_bundle(
        {
            "domain": "technical",
            "must_have": ["Python fundamentals"],
            "nice_to_have": [],
        }
    )
    state = {
        "application_id": "00000000-0000-0000-0000-000000000001",
        "job_id": "00000000-0000-0000-0000-000000000002",
        "resume_text": "Python engineer with Flask projects.",
        "jd_structured": {
            "domain": "technical",
            "must_have": ["Python fundamentals"],
            "nice_to_have": [],
        },
        "rubric": [],
        "enriched_contents": [],
    }
    raw = {
        "resume_similarity_score": {"score": 80, "reasoning": "Strong Python background."},
        "requirement_matches": [
            {
                "requirement": "Python fundamentals",
                "requirement_type": "technical_skill",
                "match_score": 85,
                "evidence": "Resume lists Python in multiple projects.",
            }
        ],
        "recommendation": "advance",
        "recommendation_reasoning": "Meets core skills.",
        "red_flags": [],
    }

    outcome = process_screening_submission(state, raw)

    assert outcome["ok"] is True
    matches = outcome["screening_result"]["requirement_matches"]
    assert len(matches) == len(bundle["rubric"])
    assert matches[0]["requirement"] == "Python fundamentals"


def test_requirement_matches_need_rescore_detects_role_fit_placeholder() -> None:
    rubric = [{"criterion": "Python", "weight": "must_have", "requirement_type": "technical_skill"}]
    matches = [
        {
            "requirement": "role fit",
            "requirement_type": "technical_skill",
            "match_score": 0,
            "evidence": "No explicit evidence found in resume or profiles.",
        }
    ]

    assert requirement_matches_need_rescore(matches, rubric) is True
    assert requirement_matches_need_rescore([], rubric) is True
    assert requirement_matches_need_rescore(matches, []) is True


def test_requirement_matches_need_rescore_detects_placeholder_requirement_rows() -> None:
    rubric = [
        {
            "criterion": "Python fundamentals",
            "weight": "must_have",
            "requirement_type": "technical_skill",
        },
        {"criterion": "REST APIs", "weight": "must_have", "requirement_type": "technical_skill"},
    ]
    matches = [
        {
            "requirement": "requirement",
            "requirement_type": "technical_skill",
            "match_score": 100,
            "evidence": "No explicit evidence found in resume or profiles.",
        },
        {
            "requirement": "requirement",
            "requirement_type": "technical_skill",
            "match_score": 100,
            "evidence": "No explicit evidence found in resume or profiles.",
        },
    ]

    assert requirement_matches_need_rescore(matches, rubric) is True


def test_merge_with_prep_state_restores_rubric_for_submit() -> None:
    from agent.prep_context import clear_prep_state, merge_with_prep_state, register_prep_state

    bundle = build_rubric_bundle(
        {
            "domain": "technical",
            "must_have": ["Python fundamentals"],
            "nice_to_have": [],
        }
    )
    prep = {
        "application_id": "00000000-0000-0000-0000-000000000001",
        "job_id": "00000000-0000-0000-0000-000000000002",
        "resume_text": "Python engineer.",
        "jd_raw": "Must have: Python fundamentals",
        "jd_structured": {
            "domain": "technical",
            "must_have": ["Python fundamentals"],
            "nice_to_have": [],
        },
        "rubric": bundle["rubric"],
        "enriched_contents": [],
    }
    register_prep_state(prep)
    try:
        tool_state = {
            "application_id": prep["application_id"],
            "job_id": prep["job_id"],
            "rubric": [],
            "jd_structured": {},
            "jd_raw": "",
            "enriched_contents": [{"url": "https://github.com/example", "content": "OSS"}],
        }
        merged = merge_with_prep_state(tool_state)
        assert len(resolve_session_rubric(merged)) == len(bundle["rubric"])
        assert merged["enriched_contents"] == tool_state["enriched_contents"]
    finally:
        clear_prep_state(prep["application_id"])
