from pathlib import Path

def sub(path: str, pairs: list[tuple[str, str]]) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    for old, new in pairs:
        if old not in text:
            raise SystemExit(f"missing in {path}: {old[:80]!r}")
        text = text.replace(old, new, 1)
    p.write_text(text, encoding="utf-8")

sub("agent/agent_runner.py", [
    (
        "4. Turn 3 — submit_screening_result after reading sandbox digest (risk_tier, vulns, secrets, excerpts).",
        "4. Turn 3 — submit_screening_result after reading sandbox digest\n   (risk_tier, vulns, secrets, excerpts).",
    ),
    (
        "   Penalize aligned-repo material risk: CRITICAL/SEVERE risk_tier means resume_similarity_score should",
        "   Penalize aligned-repo material risk: CRITICAL/SEVERE risk_tier means\n   resume_similarity_score should",
    ),
    (
        '        "CONTINUATION REQUIRED — your next response MUST be a submit_screening_result tool call only.",',
        '        "CONTINUATION REQUIRED — your next response MUST be a submit_screening_result "\n        "tool call only.",',
    ),
    (
        '                "1) get_github_repo_structures  2) run_sandbox_analysis per repo with classification "',
        '                "1) get_github_repo_structures  2) run_sandbox_analysis per repo "\n                "with classification "',
    ),
    (
        '                "3) submit_screening_result with top_file_evaluation for each sandbox top_files path\\n"',
        '                "3) submit_screening_result with top_file_evaluation for each sandbox "\n                "top_files path\\n"',
    ),
    (
        '            _log_agent_workflow_state(session_state_dict, label=f"pre-continuation-{continuation_idx + 1}")',
        '            _log_agent_workflow_state(\n                session_state_dict,\n                label=f"pre-continuation-{continuation_idx + 1}",\n            )',
    ),
])

sub("agent/config.py", [
    (
        '        description="LLM sampling temperature for scoring and structured JSON calls (0 = most stable).",',
        '        description=(\n            "LLM sampling temperature for scoring and structured JSON calls "\n            "(0 = most stable)."\n        ),',
    ),
    (
        '        description="Quantize requirement match_score and final overall score to this step (e.g. 5 → 70, 75, 80).",',
        '        description=(\n            "Quantize requirement match_score and final overall score to this step "\n            "(e.g. 5 -> 70, 75, 80)."\n        ),',
    ),
    (
        '            "Populates enriched_contents and sources_crawled even when the agent skips fetch_profiles."',
        '            "Populates enriched_contents and sources_crawled even when the agent "\n            "skips fetch_profiles."',
    ),
])

sub("agent/tools/scorer.py", [
    (
        "from agent.tools.repo_scoring import build_evaluation_breakdown, resolve_score_from_evaluation_breakdown",
        "from agent.tools.repo_scoring import (\n    build_evaluation_breakdown,\n    resolve_score_from_evaluation_breakdown,\n)",
    ),
    (
        "- resume_similarity_score.score is computed from weighted match_scores; score each criterion carefully.",
        "- resume_similarity_score.score is computed from weighted match_scores;\n  score each criterion carefully.",
    ),
    (
        "    if sandbox_penalty > 0 and score != pre_sandbox_score and score_source != \"evaluation_composite\":",
        "    if (\n        sandbox_penalty > 0\n        and score != pre_sandbox_score\n        and score_source != \"evaluation_composite\"\n    ):",
    ),
])

sub("agent/tools/sandbox_prompt.py", [
    (
        '            f"  classification={role}, risk_tier={risk_tier}, clone_ok={report.get(\'clone_ok\')}, "',
        '            f"  classification={role}, risk_tier={risk_tier}, "\n            f"clone_ok={report.get(\'clone_ok\')}, "',
    ),
    (
        '            f"secret_hits={security.get(\'secret_pattern_hits\') or profile.get(\'secret_pattern_hits\') or 0}, "',
        '            f"secret_hits={\n                security.get(\'secret_pattern_hits\')\n                or profile.get(\'secret_pattern_hits\')\n                or 0\n            }, "',
    ),
    (
        "- Read every sandbox repo summary, risk_tier, and focused file excerpts before submit_screening_result.",
        "- Read every sandbox repo summary, risk_tier, and focused file excerpts before\n  submit_screening_result.",
    ),
    (
        "- Repo classifications: aligned (strict), adjacent (moderate), peripheral (low weight), orthogonal (exclude role depth).",
        "- Repo classifications: aligned (strict), adjacent (moderate), peripheral (low weight),\n  orthogonal (exclude role depth).",
    ),
    (
        "- Do NOT lower scores only because a repo lacks tests or CI (common for coursework, DSA, or HDL repos).",
        "- Do NOT lower scores only because a repo lacks tests or CI\n  (common for coursework, DSA, or HDL repos).",
    ),
    (
        "- Treat aligned/adjacent repos with risk_tier SEVERE or CRITICAL as major negatives — not offset by other repos.",
        "- Treat aligned/adjacent repos with risk_tier SEVERE or CRITICAL as major negatives —\n  not offset by other repos.",
    ),
    (
        "  * CRITICAL (50+ combined vulns, or weak secret hygiene with 20+ vulns): overall score should be 60-65.",
        "  * CRITICAL (50+ combined vulns, or weak secret hygiene with 20+ vulns):\n    overall score should be 60-65.",
    ),
    (
        "- Orthogonal repos (e.g. coursework DBMS) must not inflate scores; missing tests/CI there is neutral.",
        "- Orthogonal repos (e.g. coursework DBMS) must not inflate scores;\n  missing tests/CI there is neutral.",
    ),
    (
        "- Cite sandbox evidence in requirement_matches or resume_similarity_score.reasoning when it affects judgment.",
        "- Cite sandbox evidence in requirement_matches or resume_similarity_score.reasoning\n  when it affects judgment.",
    ),
    (
        "- The platform applies mandatory risk caps for severe aligned-repo signals; your score should reflect that risk.",
        "- The platform applies mandatory risk caps for severe aligned-repo signals;\n  your score should reflect that risk.",
    ),
])

sub("tests/unit/test_sandbox_scoring.py", [
    (
        "from agent.prep_context import merge_github_repo_analyses, merge_with_prep_state, register_prep_state",
        "from agent.prep_context import (\n    merge_github_repo_analyses,\n    merge_with_prep_state,\n    register_prep_state,\n)",
    ),
    (
        '    assert "Sandbox repo review reduced the score" in reconciled["resume_similarity_score"]["reasoning"]',
        '    reasoning = reconciled["resume_similarity_score"]["reasoning"]\n    assert "Sandbox repo review reduced the score" in reasoning',
    ),
    (
        '"reasoning": "Sandbox repo review reduced the score by 5 points due to engineering-risk signals.",',
        '"reasoning": (\n                "Sandbox repo review reduced the score by 5 points due to "\n                "engineering-risk signals."\n            ),',
    ),
])

print("ok")