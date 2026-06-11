"""Unit tests for deterministic repo portfolio and code-quality scoring."""

from __future__ import annotations

from agent.tools.repo_scoring import (
    build_evaluation_breakdown,
    compute_code_quality_score,
    compute_ownership_confidence,
    recency_score,
    score_sandbox_report,
)


def _sample_report(*, classification: str = "aligned", secret_hits: int = 0) -> dict:
    return {
        "url": "https://github.com/candidate/demo",
        "repo": "candidate/demo",
        "clone_ok": True,
        "classification": classification,
        "repo_profile": {
            "repo_role": classification,
            "has_ci": True,
            "has_tests": False,
            "has_docker": True,
            "git_profile": {
                "commit_count": 33,
                "unique_authors": 1,
                "days_since_last_commit": 14,
                "top_author_commit_share": 0.92,
                "sole_author": True,
                "merge_to_commit_ratio": 0.06,
                "history_is_shallow": False,
            },
            "documentation_profile": {
                "readme_present": True,
                "readme_bytes": 4500,
                "has_setup_instructions": True,
                "has_architecture_section": True,
                "has_docs_dir": True,
            },
            "code_metrics": {
                "type_annotation_ratio": 0.4,
                "error_handling_density": 0.03,
                "avg_cyclomatic_complexity": 3.5,
                "lint_violations_per_kloc": 2.0,
            },
            "security_profile": {"secret_pattern_hits": secret_hits},
            "top_files": [
                {"path": "backend/app/main.py", "content_status": "ok"},
                {"path": "backend/app/services/rag_retriever.py", "content_status": "ok"},
            ],
        },
    }


def test_recency_score_tiers() -> None:
    assert recency_score(10) == 100
    assert recency_score(120) == 60
    assert recency_score(300) == 30
    assert recency_score(400) == 10


def test_ownership_confidence_caps_fork() -> None:
    git_profile = {"top_author_commit_share": 0.9, "history_is_shallow": False}
    assert compute_ownership_confidence(git_profile=git_profile, is_fork=False) > 0.7
    assert compute_ownership_confidence(git_profile=git_profile, is_fork=True) <= 0.30


def test_code_quality_score_any_profile_with_code_metrics() -> None:
    result = compute_code_quality_score(_sample_report()["repo_profile"])
    assert isinstance(result["code_quality_score"], int)
    assert 0 <= result["code_quality_score"] <= 100
    assert result["bonus"] == 2  # docker only; tests missing = no penalty
    assert result["components"]["secrets"] == 100.0


def test_code_quality_penalizes_secrets() -> None:
    clean = compute_code_quality_score(_sample_report(secret_hits=0)["repo_profile"])
    dirty = compute_code_quality_score(_sample_report(secret_hits=3)["repo_profile"])
    assert dirty["code_quality_score"] is not None
    assert clean["code_quality_score"] is not None
    assert dirty["code_quality_score"] < clean["code_quality_score"]


def test_score_sandbox_report_returns_portfolio_and_quality() -> None:
    github = {
        "repo_analyses": [
            {
                "url": "https://github.com/candidate/demo",
                "is_fork": False,
                "github_metadata": {"contributors_count": 1, "license_present": True},
            }
        ]
    }
    scored = score_sandbox_report(_sample_report(), github_repo_analyses=github)
    assert scored is not None
    assert scored["classification"] == "aligned"
    assert scored["repo_final_score"] > 0
    assert isinstance(scored["code_quality_score"], int)


def test_build_evaluation_breakdown_composite() -> None:
    rubric = [{"criterion": "Python", "weight": "must_have", "requirement_type": "technical_skill"}]
    matches = [
        {
            "requirement": "Python",
            "requirement_type": "technical_skill",
            "match_score": 90,
            "evidence": "Used across backend services.",
        }
    ]
    github = {"sandbox_reports": [_sample_report()], "repo_analyses": []}
    breakdown = build_evaluation_breakdown(
        requirement_matches=matches,
        rubric=rubric,
        github_repo_analyses=github,
    )
    assert breakdown is not None
    assert breakdown["jd_fit_score"] == 90
    assert isinstance(breakdown["repo_portfolio_score"], int)
    assert isinstance(breakdown["code_quality_score"], int)
    assert isinstance(breakdown["composite_score"], int)
    assert len(breakdown["repos"]) == 1
