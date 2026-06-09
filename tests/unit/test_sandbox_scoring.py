"""Tests for deterministic sandbox score penalties."""

from __future__ import annotations

from agent.tools.rubric_builder import build_rubric_bundle
from agent.tools.sandbox_scoring import (
    apply_sandbox_score_penalty,
    compute_sandbox_score_penalty,
    repo_sandbox_risk_penalty,
)
from agent.tools.scorer import normalize_screening_result


def _chaos_repo_report() -> dict:
    return {
        "repo": "DevKansara97/chaos-repo",
        "clone_ok": True,
        "quality_signals": {
            "has_tests": False,
            "has_ci": False,
            "has_docs": True,
            "has_docker": True,
        },
        "repo_profile": {
            "security_profile": {
                "secret_pattern_hits": 0,
                "has_env_file": True,
                "secret_hygiene": "weak",
            },
            "external_tool_signals": {
                "pip_audit": {"vulnerability_count": 70},
                "npm_audit": {"vulnerability_count": 4},
                "trivy": {"vulnerability_count": 55},
            },
        },
        "findings": [
            {
                "severity": "high",
                "category": "risk",
                "title": "Repository shows weak secret hygiene.",
            }
        ],
    }


def _healthy_repo_report() -> dict:
    return {
        "repo": "DevKansara97/healthy-repo",
        "clone_ok": True,
        "quality_signals": {
            "has_tests": True,
            "has_ci": True,
            "has_docs": True,
            "has_docker": True,
        },
        "repo_profile": {
            "security_profile": {"secret_hygiene": "mixed"},
            "external_tool_signals": {
                "pip_audit": {"vulnerability_count": 1},
                "trivy": {"vulnerability_count": 1},
            },
        },
        "findings": [],
    }


def _fitness_repo_report() -> dict:
    return {
        "repo": "Parinn7/Personalized-Fitness-Tracker-DBMS",
        "clone_ok": True,
        "quality_signals": {
            "has_tests": False,
            "has_ci": False,
            "has_docs": False,
            "has_docker": False,
        },
        "repo_profile": {
            "security_profile": {"secret_hygiene": "mixed"},
            "external_tool_signals": {"trivy": {"vulnerability_count": 0}},
        },
        "findings": [],
    }


def test_chaos_repo_penalty_is_material() -> None:
    assert repo_sandbox_risk_penalty(_chaos_repo_report()) == 25


def test_combined_penalty_matches_candidate_example() -> None:
    reports = [_healthy_repo_report(), _fitness_repo_report(), _chaos_repo_report()]
    penalty = compute_sandbox_score_penalty(reports)
    assert 20 <= penalty <= 23


def test_apply_sandbox_score_penalty_reduces_high_llm_score() -> None:
    adjusted, penalty = apply_sandbox_score_penalty(
        77,
        {
            "username": "DevKansara97",
            "sandbox_reports": [
                _healthy_repo_report(),
                _fitness_repo_report(),
                _chaos_repo_report(),
            ],
        },
    )
    assert penalty >= 16
    assert 55 <= adjusted <= 62


def test_normalize_screening_result_applies_sandbox_penalty() -> None:
    bundle = build_rubric_bundle(
        {
            "domain": "technical",
            "must_have": ["Python fundamentals"],
            "nice_to_have": ["Docker"],
        }
    )
    raw = {
        "resume_similarity_score": {"score": 77, "reasoning": "Strong Python alignment."},
        "requirement_matches": [
            {
                "requirement": "Python fundamentals",
                "requirement_type": "technical_skill",
                "match_score": 95,
                "evidence": "Multiple Python projects.",
            }
        ],
        "recommendation": "advance",
        "recommendation_reasoning": "Good fit overall.",
        "red_flags": [],
    }

    normalized = normalize_screening_result(
        raw,
        application_id="00000000-0000-0000-0000-000000000001",
        job_id="00000000-0000-0000-0000-000000000002",
        resume_text="Python engineer",
        rubric=bundle["rubric"],
        enriched_contents=[],
        github_repo_analyses={
            "username": "DevKansara97",
            "sandbox_reports": [
                _healthy_repo_report(),
                _fitness_repo_report(),
                _chaos_repo_report(),
            ],
        },
    )

    score = normalized["resume_similarity_score"]["score"]
    assert 55 <= score <= 62
    assert normalized["recommendation"] == "hold"
    reasoning = normalized["resume_similarity_score"]["reasoning"]
    assert "Sandbox repo review reduced the score" in reasoning
