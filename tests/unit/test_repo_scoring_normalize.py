"""Integration: evaluation_breakdown attached during normalize_screening_result."""

from __future__ import annotations

from agent.tools.scorer import normalize_screening_result


def test_normalize_attaches_evaluation_breakdown() -> None:
    rubric = [{"criterion": "Python", "weight": "must_have", "requirement_type": "technical_skill"}]
    raw = {
        "resume_similarity_score": {"score": 88, "reasoning": "Strong Python evidence."},
        "requirement_matches": [
            {
                "requirement": "Python",
                "requirement_type": "technical_skill",
                "match_score": 90,
                "evidence": "FastAPI backend with typed handlers.",
            }
        ],
        "recommendation": "advance",
        "recommendation_reasoning": "Good fit.",
        "red_flags": [],
    }
    github = {
        "username": "candidate",
        "sandbox_reports": [
            {
                "url": "https://github.com/candidate/demo",
                "repo": "candidate/demo",
                "clone_ok": True,
                "classification": "aligned",
                "repo_profile": {
                    "repo_role": "aligned",
                    "has_ci": True,
                    "git_profile": {
                        "commit_count": 20,
                        "unique_authors": 1,
                        "days_since_last_commit": 20,
                        "top_author_commit_share": 0.95,
                        "merge_to_commit_ratio": 0.05,
                    },
                    "documentation_profile": {
                        "readme_present": True,
                        "readme_bytes": 2000,
                        "has_setup_instructions": True,
                    },
                    "code_metrics": {
                        "type_annotation_ratio": 0.5,
                        "error_handling_density": 0.04,
                        "avg_cyclomatic_complexity": 2.5,
                        "lint_violations_per_kloc": 1.0,
                    },
                    "security_profile": {"secret_pattern_hits": 0},
                    "top_files": [{"path": "main.py", "content_status": "ok"}],
                },
            }
        ],
    }

    normalized = normalize_screening_result(
        raw,
        application_id="11111111-1111-4111-8111-111111111111",
        job_id="22222222-2222-4222-8222-222222222222",
        resume_text="Python engineer",
        rubric=rubric,
        enriched_contents=[],
        github_repo_analyses=github,
    )

    breakdown = normalized.get("evaluation_breakdown")
    assert isinstance(breakdown, dict)
    assert breakdown.get("jd_fit_score") == 90
    assert isinstance(breakdown.get("composite_score"), int)
    assert breakdown.get("final_score_source") == "evaluation_composite"
    assert normalized["resume_similarity_score"]["score"] == breakdown["final_score"]


def test_composite_score_not_overridden_by_identity_cap() -> None:
    """Sandbox composite must win over rubric-derived score when profiles are untrusted."""
    rubric = [
        {"criterion": "Python", "weight": "must_have", "requirement_type": "technical_skill"},
        {"criterion": "FastAPI", "weight": "nice_to_have", "requirement_type": "technical_skill"},
    ]
    raw = {
        "resume_similarity_score": {"score": 95, "reasoning": "Excellent JD fit."},
        "requirement_matches": [
            {
                "requirement": "Python",
                "requirement_type": "technical_skill",
                "match_score": 100,
                "evidence": "Multiple Python projects.",
            },
            {
                "requirement": "FastAPI",
                "requirement_type": "technical_skill",
                "match_score": 100,
                "evidence": "FastAPI in repos.",
            },
        ],
        "recommendation": "advance",
        "recommendation_reasoning": "Strong fit.",
        "red_flags": [],
    }
    github = {
        "username": "candidate",
        "sandbox_reports": [
            {
                "url": "https://github.com/candidate/demo",
                "repo": "candidate/demo",
                "clone_ok": True,
                "classification": "aligned",
                "repo_profile": {
                    "repo_role": "aligned",
                    "has_ci": False,
                    "git_profile": {
                        "commit_count": 5,
                        "unique_authors": 1,
                        "days_since_last_commit": 200,
                        "top_author_commit_share": 1.0,
                        "merge_to_commit_ratio": 0.0,
                    },
                    "documentation_profile": {
                        "readme_present": True,
                        "readme_bytes": 500,
                        "has_setup_instructions": False,
                    },
                    "code_metrics": {
                        "type_annotation_ratio": 0.2,
                        "error_handling_density": 0.01,
                        "avg_cyclomatic_complexity": 8.0,
                        "lint_violations_per_kloc": None,
                    },
                    "security_profile": {"secret_pattern_hits": 10, "secret_hygiene": "weak"},
                    "top_files": [{"path": "main.py", "content_status": "ok"}],
                },
            }
        ],
    }

    normalized = normalize_screening_result(
        raw,
        application_id="11111111-1111-4111-8111-111111111111",
        job_id="22222222-2222-4222-8222-222222222222",
        resume_text="Python engineer",
        rubric=rubric,
        enriched_contents=[],
        github_repo_analyses=github,
        profile_identity_cap_score=True,
    )

    breakdown = normalized.get("evaluation_breakdown")
    assert breakdown.get("final_score_source") == "evaluation_composite"
    assert normalized["resume_similarity_score"]["score"] == breakdown["final_score"]
    assert normalized["resume_similarity_score"]["score"] < 90
