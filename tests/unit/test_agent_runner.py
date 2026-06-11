import json

import pytest

from agent.agent_runner import (
    SCREENING_AGENT_INSTRUCTION,
    _run_heuristic_sandbox_fallback_if_needed,
    build_agent_user_message,
    seed_screening_session,
)
from agent.pipeline import create_runner


def test_screening_instruction_references_trust_tiers() -> None:
    lowered = SCREENING_AGENT_INSTRUCTION.lower()
    assert "profile_trust_by_url" in SCREENING_AGENT_INSTRUCTION
    assert "scoring_untrusted" in lowered
    assert "submit_screening_result" in lowered
    assert "fetch_profiles" in lowered
    assert "get_github_repo_structures" in lowered
    assert "run_sandbox_analysis" in lowered
    assert "1-5 focus_paths" in SCREENING_AGENT_INSTRUCTION
    assert "copy" in lowered and "get_github_repo_structures" in SCREENING_AGENT_INSTRUCTION


def test_build_agent_user_message_includes_screening_context() -> None:
    state = {
        "application_id": "11111111-1111-4111-8111-111111111111",
        "job_id": "22222222-2222-4222-8222-222222222222",
        "resume_text": "Python engineer with six years of experience.",
        "jd_raw": "Senior Python Backend Engineer\nMust have: Python, FastAPI",
        "rubric_preamble": "Score resume-first; avoid bias.",
        "rubric": [
            {
                "criterion": "Python",
                "weight": "must_have",
                "requirement_type": "technical_skill",
            }
        ],
        "profile_urls": ["https://github.com/janedoe"],
        "profile_trust_by_url": {
            "https://github.com/janedoe": "scoring_trusted",
        },
        "identity_red_flags": [],
        "profile_identity_cap_score": False,
        "github_repo_analyses": {
            "username": "janedoe",
            "sandbox_reports": [
                {
                    "repo": "janedoe/demo",
                    "clone_ok": True,
                    "repo_profile": {
                        "security_profile": {"secret_hygiene": "mixed"},
                        "external_tool_signals": {"trivy": {"vulnerability_count": 0}},
                    },
                    "findings": [],
                }
            ],
        },
    }

    message = build_agent_user_message(state)

    assert "11111111-1111-4111-8111-111111111111" in message
    assert "Python engineer" in message
    assert "Senior Python Backend Engineer" in message
    assert "profile_trust_by_url" in message.lower() or "PROFILE_TRUST_BY_URL" in message
    assert "scoring_trusted" in message
    assert "https://github.com/janedoe" in message
    assert "sandbox reports" in message.lower()
    assert "do not lower scores only because a repo lacks tests or ci" in message.lower()
    assert "submit_screening_result" in message
    assert "SUBMIT_PAYLOAD_SHAPE" in message
    assert "FINAL STEP" in message
    assert json.loads(message.split("PROFILE_TRUST_BY_URL:\n", 1)[1].split("\n\nJOB", 1)[0])


def test_build_agent_user_message_pending_sandbox_when_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_OVERLAP_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_DEFERRED_ENABLED", "false")
    monkeypatch.setenv("SANDBOX_LLM_SCORING_ENABLED", "false")
    from agent.config import get_settings

    get_settings.cache_clear()

    state = {
        "application_id": "11111111-1111-4111-8111-111111111111",
        "job_id": "22222222-2222-4222-8222-222222222222",
        "resume_text": "Engineer",
        "jd_raw": "Role",
        "rubric_preamble": "",
        "rubric": [],
        "profile_urls": [],
        "profile_trust_by_url": {},
        "identity_red_flags": [],
        "profile_identity_cap_score": False,
        "github_repo_analyses": {
            "username": "janedoe",
            "repo_analyses": [],
            "selected_sandbox_repo_urls": ["https://github.com/u/r1"],
            "sandbox_reports": [],
        },
    }

    message = build_agent_user_message(state)

    assert "evaluation in progress" in message.lower()


def test_build_agent_user_message_notes_identity_cap() -> None:
    state = {
        "application_id": "11111111-1111-4111-8111-111111111111",
        "job_id": "22222222-2222-4222-8222-222222222222",
        "resume_text": "Engineer",
        "jd_raw": "Role",
        "rubric_preamble": "",
        "rubric": [],
        "profile_urls": ["https://github.com/other"],
        "profile_trust_by_url": {"https://github.com/other": "scoring_untrusted"},
        "identity_red_flags": [
            {
                "flag": "profile_identity_mismatch",
                "severity": "high",
                "evidence": "slug mismatch",
            }
        ],
        "profile_identity_cap_score": True,
    }

    message = build_agent_user_message(state)

    assert "capped at 45" in message
    assert "profile_identity_mismatch" in message


@pytest.mark.asyncio
async def test_seed_screening_session_is_idempotent() -> None:
    runner = create_runner(auto_create_session=False)
    state = {
        "application_id": "11111111-1111-4111-8111-111111111111",
        "job_id": "22222222-2222-4222-8222-222222222222",
        "resume_text": "Engineer",
    }
    await seed_screening_session(
        runner,
        state,
        user_id="11111111-1111-4111-8111-111111111111",
        session_id="sess-1",
    )
    await seed_screening_session(
        runner,
        state,
        user_id="11111111-1111-4111-8111-111111111111",
        session_id="sess-1",
    )
    session = await runner.session_service.get_session(
        app_name=runner.app_name,
        user_id="11111111-1111-4111-8111-111111111111",
        session_id="sess-1",
    )
    assert session is not None
    assert session.state.get("resume_text") == "Engineer"


@pytest.mark.asyncio
async def test_run_heuristic_sandbox_fallback_if_needed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from unittest.mock import AsyncMock, patch

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCREENING_MODE", "agent")
    monkeypatch.setenv("AGENT_EVIDENCE_ORCHESTRATION_ENABLED", "true")
    from agent.config import get_settings

    get_settings.cache_clear()

    prep = {
        "application_id": "11111111-1111-4111-8111-111111111111",
        "github_repo_analyses": {
            "selected_sandbox_repo_urls": ["https://github.com/dev/service"],
        },
    }
    session = {}

    with patch(
        "agent.adk_tools.ensure_sandbox_evidence",
        new=AsyncMock(
            side_effect=lambda state: state.update(
                {
                    "github_repo_analyses": {
                        "sandbox_reports": [{"url": "https://github.com/dev/service"}],
                    }
                }
            )
        ),
    ):
        ran = await _run_heuristic_sandbox_fallback_if_needed(prep, session)

    assert ran is True
