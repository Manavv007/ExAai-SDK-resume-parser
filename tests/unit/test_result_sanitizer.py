"""Tests for LLM output sanitization before schema validation."""

from __future__ import annotations

import json
from pathlib import Path

from agent.tools.result_sanitizer import compact_metadata, sanitize_requirement_matches
from agent.tools.scorer import normalize_screening_result
from agent.tools.validator import validate_result

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_normalize_messy_llm_payload_passes_validation(test_settings) -> None:
    """Common Gemini quirks: floats, null metadata, empty evidence, bad enums."""
    fixture = json.loads((FIXTURES / "valid_result_completed.json").read_text(encoding="utf-8"))
    messy = {
        "resume_similarity_score": {"score": 78.6, "reasoning": "  Good fit.  "},
        "requirement_matches": [
            {
                "requirement": "Python",
                "requirement_type": "technical",
                "match_score": "82.4",
                "evidence": "",
                "source_quote": None,
                "extra_field": "drop-me",
            }
        ],
        "recommendation": "yes",
        "recommendation_reasoning": "",
        "red_flags": [
            {"flag": "gap", "severity": "severe", "evidence": "Two-year gap detected."},
            {"flag": "", "severity": "low", "evidence": "skip"},
        ],
        "sources_crawled": [
            {
                "url": "github.com/example-candidate",
                "title": None,
                "relevance": "strong",
            }
        ],
        "metadata": {
            "processing_time_ms": None,
            "llm_calls": None,
            "agent_submit_fallback": None,
        },
    }

    normalized = normalize_screening_result(
        messy,
        application_id=fixture["application_id"],
        job_id=fixture["job_id"],
        resume_text="resume text",
        rubric=[
            {
                "criterion": "Python",
                "weight": "must_have",
                "requirement_type": "technical_skill",
            }
        ],
        enriched_contents=[],
        processing_time_ms=None,
    )

    assert validate_result(normalized) is True
    assert normalized["resume_similarity_score"]["score"] == 82
    assert normalized["requirement_matches"][0]["match_score"] == 82
    assert normalized["requirement_matches"][0]["evidence"]
    assert normalized["recommendation"] == "advance"
    assert normalized["recommendation_reasoning"]
    assert normalized["red_flags"][0]["severity"] == "medium"
    assert normalized["sources_crawled"][0]["url"].startswith("https://")
    assert "title" not in normalized["sources_crawled"][0]
    assert "processing_time_ms" not in normalized["metadata"]
    assert "llm_calls" not in normalized["metadata"]


def test_sanitize_replaces_placeholder_requirement_with_rubric_criterion() -> None:
    rubric = [
        {
            "criterion": "Python programming",
            "weight": "must_have",
            "requirement_type": "technical_skill",
        },
        {
            "criterion": "Git experience",
            "weight": "nice_to_have",
            "requirement_type": "technical_skill",
        },
    ]
    sanitized = sanitize_requirement_matches(
        [
            {
                "requirement": "requirement",
                "requirement_type": "technical_skill",
                "match_score": 90,
                "evidence": "",
            },
            {
                "requirement": "requirement",
                "requirement_type": "technical_skill",
                "match_score": 80,
                "evidence": "Listed GitHub projects.",
            },
        ],
        rubric,
    )

    assert sanitized[0]["requirement"] == "Python programming"
    assert sanitized[1]["requirement"] == "Git experience"


def test_compact_metadata_strips_null_optional_fields() -> None:
    assert compact_metadata(
        {
            "schema_version": "1.0",
            "model_version": "x",
            "processed_at": "2026-01-01T00:00:00Z",
            "resume_text_chars": 10,
            "agent_version": "0.1.0",
            "processing_time_ms": None,
            "llm_calls": None,
            "agent_submit_fallback": None,
        }
    ) == {
        "schema_version": "1.0",
        "model_version": "x",
        "processed_at": "2026-01-01T00:00:00Z",
        "resume_text_chars": 10,
        "agent_version": "0.1.0",
    }
