"""Tests for deterministic sandbox score penalties."""

from __future__ import annotations

import pytest

from agent.prep_context import (
    merge_github_repo_analyses,
    merge_with_prep_state,
    register_prep_state,
)
from agent.tools.rubric_builder import build_rubric_bundle
from agent.tools.sandbox_scoring import (
    apply_sandbox_score_penalty,
    compute_sandbox_score_ceiling,
    compute_sandbox_score_penalty,
    reconcile_sandbox_penalty_in_result,
    repo_sandbox_risk_penalty,
    repo_sandbox_score_ceiling,
)
from agent.tools.scorer import normalize_screening_result


def _chaos_repo_report() -> dict:
    return {
        "repo": "DevKansara97/chaos-repo",
        "classification": "aligned",
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
        "classification": "aligned",
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
        "classification": "orthogonal",
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


@pytest.fixture
def deterministic_sandbox_scoring(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_LLM_SCORING_ENABLED", "false")
    from agent.config import get_settings

    get_settings.cache_clear()


def test_fitness_repo_without_vulns_has_no_penalty() -> None:
    assert repo_sandbox_risk_penalty(_fitness_repo_report()) == 0


def test_chaos_repo_penalty_is_material(deterministic_sandbox_scoring) -> None:
    penalty = repo_sandbox_risk_penalty(_chaos_repo_report())
    assert 10 <= penalty <= 15


def test_chaos_repo_has_critical_ceiling(deterministic_sandbox_scoring) -> None:
    assert repo_sandbox_score_ceiling(_chaos_repo_report()) == 85


def test_combined_penalty_matches_candidate_example(deterministic_sandbox_scoring) -> None:
    reports = [_healthy_repo_report(), _fitness_repo_report(), _chaos_repo_report()]
    penalty = compute_sandbox_score_penalty(reports)
    assert 10 <= penalty <= 15
    assert compute_sandbox_score_ceiling(reports) == 85


def test_apply_sandbox_score_penalty_applies_when_llm_scoring_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_LLM_SCORING_ENABLED", "true")
    from agent.config import get_settings

    get_settings.cache_clear()
    adjusted, reduction = apply_sandbox_score_penalty(
        95,
        {
            "username": "DevKansara97",
            "sandbox_reports": [
                _healthy_repo_report(),
                _fitness_repo_report(),
                _chaos_repo_report(),
            ],
        },
    )
    assert reduction > 0
    assert 78 <= adjusted <= 83


def test_apply_sandbox_score_penalty_reduces_high_llm_score(deterministic_sandbox_scoring) -> None:
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
    assert penalty > 0
    assert adjusted <= 77


def test_merge_github_repo_analyses_prefers_sandbox_reports_from_prep() -> None:
    prep_github = {
        "username": "dev",
        "sandbox_reports": [{"repo": "u/r1", "url": "https://github.com/u/r1", "clone_ok": True}],
    }
    session_github = {
        "username": "dev",
        "sandbox_reports": [],
        "overall_github_signal": "strong",
    }
    merged = merge_github_repo_analyses(prep_github, session_github)
    assert merged is not None
    assert len(merged["sandbox_reports"]) == 1
    assert merged["overall_github_signal"] == "strong"


def test_merge_with_prep_state_keeps_sandbox_reports_when_session_is_stale() -> None:
    from agent.prep_context import clear_prep_state

    application_id = "00000000-0000-0000-0000-000000000001"
    fake_report = {
        "repo": "u/r1",
        "url": "https://github.com/u/r1",
        "clone_ok": True,
        "quality_signals": {"has_tests": False, "has_ci": False},
    }
    register_prep_state(
        {
            "application_id": application_id,
            "github_repo_analyses": {
                "username": "dev",
                "sandbox_reports": [fake_report],
            },
        }
    )
    try:
        merged = merge_with_prep_state(
            {
                "application_id": application_id,
                "github_repo_analyses": {
                    "username": "dev",
                    "sandbox_reports": [],
                },
            }
        )
        assert merged["github_repo_analyses"]["sandbox_reports"] == [fake_report]
    finally:
        clear_prep_state(application_id)


def test_reconcile_sandbox_penalty_applies_missed_penalty(deterministic_sandbox_scoring) -> None:
    reports = [_healthy_repo_report(), _fitness_repo_report(), _chaos_repo_report()]
    github = {"username": "DevKansara97", "sandbox_reports": reports}
    result = {
        "resume_similarity_score": {"score": 77, "reasoning": "Strong Python alignment."},
        "recommendation": "advance",
        "metadata": {},
    }

    reconciled = reconcile_sandbox_penalty_in_result(result, github)

    assert reconciled["resume_similarity_score"]["score"] < 77
    reasoning = reconciled["resume_similarity_score"]["reasoning"]
    assert "Sandbox repo review reduced the score" in reasoning


def test_reconcile_sandbox_penalty_is_idempotent(deterministic_sandbox_scoring) -> None:
    reports = [_fitness_repo_report()]
    github = {"username": "dev", "sandbox_reports": reports}
    result = {
        "resume_similarity_score": {
            "score": 70,
            "reasoning": (
                "Sandbox repo review reduced the score by 5 points due to engineering-risk signals."
            ),
        },
        "metadata": {},
    }
    reconciled = reconcile_sandbox_penalty_in_result(result, github)
    assert reconciled["resume_similarity_score"]["score"] == 70


def test_normalize_screening_result_applies_sandbox_penalty(deterministic_sandbox_scoring) -> None:
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
    breakdown = normalized.get("evaluation_breakdown") or {}
    assert breakdown.get("final_score_source") == "evaluation_composite"
    assert 10 <= int(breakdown.get("sandbox_penalty") or 0) <= 15
    assert score < 77
    assert normalized["recommendation"] == "hold"
    reasoning = normalized["resume_similarity_score"]["reasoning"]
    assert "Sandbox repo review reduced the score" in reasoning
