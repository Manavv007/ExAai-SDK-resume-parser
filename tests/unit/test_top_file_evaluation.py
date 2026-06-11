"""Tests for top_file_evaluation merge at submit."""

from __future__ import annotations

from agent.tools.rubric_builder import build_rubric_bundle
from agent.tools.scorer import normalize_screening_result
from agent.tools.top_file_evaluation import merge_top_file_evaluation


def _github_with_top_files() -> dict:
    return {
        "username": "dev",
        "sandbox_reports": [
            {
                "repo": "dev/healthy-repo",
                "url": "https://github.com/dev/healthy-repo",
                "classification": "aligned",
                "repo_profile": {
                    "top_files": [
                        {
                            "path": "app/main.py",
                            "importance_rank": 1,
                            "language": "py",
                            "compaction_tier": "raw",
                            "total_lines": 10,
                            "sent_lines": 10,
                            "content_status": "ok",
                            "content": "def main():\n    return 42\n",
                        }
                    ]
                },
            }
        ],
    }


def test_merge_top_file_evaluation_uses_sandbox_metadata() -> None:
    merged = merge_top_file_evaluation(
        [
            {
                "repo_url": "https://github.com/dev/healthy-repo",
                "path": "app/main.py",
                "jd_criteria": ["Python fundamentals"],
                "match_signal": "positive",
                "assessment": "Clear Python entrypoint with a simple function.",
            }
        ],
        _github_with_top_files(),
    )
    assert len(merged) == 1
    row = merged[0]
    assert row["repo"] == "dev/healthy-repo"
    assert row["path"] == "app/main.py"
    assert row["jd_criteria"] == ["Python fundamentals"]
    assert row["match_signal"] == "positive"
    assert "return 42" in row["evidence_snippet"]
    assert row["compaction_tier"] == "raw"


def test_merge_top_file_evaluation_ignores_agent_rows_not_in_sandbox() -> None:
    merged = merge_top_file_evaluation(
        [
            {
                "repo_url": "https://github.com/dev/healthy-repo",
                "path": "app/main.py",
                "jd_criteria": ["Python fundamentals"],
                "match_signal": "positive",
                "assessment": "Valid row.",
            },
            {
                "repo_url": "https://github.com/dev/healthy-repo",
                "path": "src/not_in_sandbox.py",
                "jd_criteria": ["Should be dropped"],
                "match_signal": "negative",
                "assessment": "This path was not sandboxed.",
            },
        ],
        _github_with_top_files(),
    )
    assert len(merged) == 1
    assert merged[0]["path"] == "app/main.py"


def test_merge_top_file_evaluation_defaults_without_agent_rows() -> None:
    merged = merge_top_file_evaluation(None, _github_with_top_files())
    assert len(merged) == 1
    assert merged[0]["match_signal"] == "positive"
    assert merged[0]["assessment"]


def test_normalize_screening_result_includes_top_file_evaluation(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_LLM_SCORING_ENABLED", "true")
    from agent.config import get_settings

    get_settings.cache_clear()

    bundle = build_rubric_bundle(
        {
            "domain": "technical",
            "must_have": ["Python fundamentals"],
            "nice_to_have": [],
        }
    )
    raw = {
        "resume_similarity_score": {"score": 80, "reasoning": "Good fit."},
        "requirement_matches": [
            {
                "requirement": "Python fundamentals",
                "requirement_type": "technical_skill",
                "match_score": 85,
                "evidence": "Python project evidence.",
            }
        ],
        "top_file_evaluation": [
            {
                "repo_url": "https://github.com/dev/healthy-repo",
                "path": "app/main.py",
                "jd_criteria": ["Python fundamentals"],
                "match_signal": "positive",
                "assessment": "Focused file shows readable Python structure.",
            }
        ],
        "recommendation": "advance",
        "recommendation_reasoning": "Strong match.",
        "red_flags": [],
    }

    normalized = normalize_screening_result(
        raw,
        application_id="00000000-0000-0000-0000-000000000001",
        job_id="00000000-0000-0000-0000-000000000002",
        resume_text="Python engineer",
        rubric=bundle["rubric"],
        enriched_contents=[],
        github_repo_analyses=_github_with_top_files(),
    )

    assert "top_file_evaluation" in normalized
    assert len(normalized["top_file_evaluation"]) == 1
    assert normalized["top_file_evaluation"][0]["assessment"].startswith("Focused file")
