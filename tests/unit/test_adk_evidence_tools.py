"""Tests for agent-orchestrated GitHub/sandbox ADK tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.adk_tools import (
    build_heuristic_fallback_repo_specs,
    ensure_sandbox_evidence,
    execute_sandbox_analysis_for_state,
    get_github_repo_structures,
    run_sandbox_analysis,
)
from agent.prep_context import get_prep_state, register_prep_state


class _AdkLikeState:
    def __init__(self, value: dict) -> None:
        self._value = dict(value)

    def get(self, key, default=None):
        return self._value.get(key, default)

    def __setitem__(self, key, value) -> None:
        self._value[key] = value

    def keys(self):
        return self._value.keys()


def test_register_prep_state_accepts_adk_like_state() -> None:
    register_prep_state(
        _AdkLikeState(
            {
                "application_id": "11111111-1111-4111-8111-111111111111",
                "job_id": "22222222-2222-4222-8222-222222222222",
                "resume_text": "Engineer",
            }
        )
    )
    prep = get_prep_state("11111111-1111-4111-8111-111111111111")
    assert prep is not None
    assert prep["resume_text"] == "Engineer"


@pytest.mark.asyncio
async def test_get_github_repo_structures_returns_classifications(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCREENING_MODE", "agent")
    from agent.config import get_settings

    get_settings.cache_clear()

    ctx = MagicMock()
    ctx.state = {
        "github_repo_analyses": {
            "username": "dev",
            "candidate_tags": ["backend_engineer"],
            "selected_sandbox_repo_urls": ["https://github.com/dev/service"],
            "repo_analyses": [
                {
                    "url": "https://github.com/dev/service",
                    "languages": {"Python": 90.0},
                    "repo_type_tags": ["backend_service"],
                }
            ],
        }
    }

    with patch(
        "agent.adk_tools._fetch_repo_tree_paths",
        new=AsyncMock(
            return_value=(
                ["README.md", "app/api/routes.py", "requirements.txt"],
                {"languages": {"Python": 90.0}},
            )
        ),
    ):
        result = await get_github_repo_structures(ctx)

    assert result["ok"] is True
    assert result["repos"][0]["classification"] == "aligned"
    assert "mandatory_focus_paths" in result["repos"][0]


@pytest.mark.asyncio
async def test_run_sandbox_analysis_merges_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCREENING_MODE", "agent")
    from agent.config import get_settings

    get_settings.cache_clear()

    ctx = MagicMock()
    ctx.state = {
        "github_repo_analyses": {
            "username": "dev",
            "candidate_tags": ["backend_engineer"],
            "selected_sandbox_repo_urls": ["https://github.com/dev/service"],
            "sandbox_reports": [],
        },
        "github_repo_structures": {
            "https://github.com/dev/service": {
                "classification": "aligned",
                "_file_paths": ["README.md", "app/api/routes.py"],
            }
        },
    }

    fake_report = {
        "repo": "dev/service",
        "url": "https://github.com/dev/service",
        "clone_ok": True,
        "repo_profile": {"sample_files": [{"path": "app/api/routes.py", "content_status": "ok"}]},
        "findings": [],
    }

    with patch(
        "agent.adk_tools._evaluate_sandbox_repos",
        new=AsyncMock(return_value=[fake_report]),
    ):
        result = await run_sandbox_analysis(
            [
                {
                    "repo_url": "https://github.com/dev/service",
                    "classification": "aligned",
                    "focus_paths": [{"path": "app/api/routes.py", "max_lines": 120}],
                }
            ],
            ctx,
        )

    assert result["ok"] is True
    assert ctx.state["sandbox_completed_by_agent"] is True
    assert (
        ctx.state["github_repo_analyses"]["sandbox_reports"][0]["url"]
        == "https://github.com/dev/service"
    )


@pytest.mark.asyncio
async def test_run_sandbox_analysis_rejects_too_many_focus_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCREENING_MODE", "agent")
    from agent.config import get_settings

    get_settings.cache_clear()

    ctx = MagicMock()
    ctx.state = {
        "github_repo_analyses": {
            "username": "dev",
            "candidate_tags": ["backend_engineer"],
            "selected_sandbox_repo_urls": ["https://github.com/dev/service"],
        },
        "github_repo_structures": {
            "https://github.com/dev/service": {
                "classification": "aligned",
                "_file_paths": [f"src/file_{index}.py" for index in range(8)],
            }
        },
    }

    focus_paths = [{"path": f"src/file_{index}.py"} for index in range(6)]
    result = await run_sandbox_analysis(
        [
            {
                "repo_url": "https://github.com/dev/service",
                "classification": "aligned",
                "focus_paths": focus_paths,
            }
        ],
        ctx,
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_sandbox_repo_spec"
    assert "maximum is 5" in result["errors"][0]


@pytest.mark.asyncio
async def test_run_sandbox_analysis_rejects_unknown_focus_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCREENING_MODE", "agent")
    from agent.config import get_settings

    get_settings.cache_clear()

    ctx = MagicMock()
    ctx.state = {
        "github_repo_analyses": {
            "username": "dev",
            "selected_sandbox_repo_urls": ["https://github.com/dev/service"],
        },
        "github_repo_structures": {
            "https://github.com/dev/service": {
                "classification": "aligned",
                "_file_paths": ["app/api/routes.py"],
            }
        },
    }

    with patch(
        "agent.adk_tools._evaluate_sandbox_repos",
        new=AsyncMock(return_value=[]),
    ):
        result = await run_sandbox_analysis(
            [
                {
                    "repo_url": "https://github.com/dev/service",
                    "classification": "aligned",
                    "focus_paths": [{"path": "src/does_not_exist.py"}],
                }
            ],
            ctx,
        )

    assert result["ok"] is False
    assert result["error"] == "invalid_sandbox_repo_spec"
    assert "not found" in result["errors"][0]


@pytest.mark.asyncio
async def test_run_sandbox_analysis_requires_focus_paths_under_orchestration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCREENING_MODE", "agent")
    monkeypatch.setenv("AGENT_EVIDENCE_ORCHESTRATION_ENABLED", "true")
    from agent.config import get_settings

    get_settings.cache_clear()

    ctx = MagicMock()
    ctx.state = {
        "github_repo_analyses": {
            "username": "dev",
            "selected_sandbox_repo_urls": ["https://github.com/dev/service"],
        },
        "github_repo_structures": {
            "https://github.com/dev/service": {
                "classification": "aligned",
                "_file_paths": ["app/api/routes.py"],
            }
        },
    }

    result = await run_sandbox_analysis(
        [{"repo_url": "https://github.com/dev/service", "classification": "aligned"}],
        ctx,
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_sandbox_repo_spec"
    assert "focus_paths is required" in result["errors"][0]


@pytest.mark.asyncio
async def test_run_sandbox_analysis_rejects_mismatched_classification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCREENING_MODE", "agent")
    monkeypatch.setenv("AGENT_EVIDENCE_ORCHESTRATION_ENABLED", "true")
    from agent.config import get_settings

    get_settings.cache_clear()

    ctx = MagicMock()
    ctx.state = {
        "github_repo_analyses": {
            "username": "dev",
            "selected_sandbox_repo_urls": ["https://github.com/dev/service"],
        },
        "github_repo_structures": {
            "https://github.com/dev/service": {
                "classification": "orthogonal",
                "_file_paths": ["static/style.css"],
            }
        },
    }

    result = await run_sandbox_analysis(
        [
            {
                "repo_url": "https://github.com/dev/service",
                "classification": "aligned",
                "focus_paths": [{"path": "static/style.css"}],
            }
        ],
        ctx,
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_sandbox_repo_spec"
    assert "does not match" in result["errors"][0]


def test_build_heuristic_fallback_repo_specs_uses_structure_classification() -> None:
    specs = build_heuristic_fallback_repo_specs(
        {
            "github_repo_structures": {
                "https://github.com/dev/service": {"classification": "orthogonal"},
            }
        },
        ["https://github.com/dev/service"],
    )
    assert len(specs) == 1
    assert specs[0]["classification"] == "orthogonal"
    assert "focus_paths" not in specs[0]


@pytest.mark.asyncio
async def test_execute_sandbox_heuristic_fallback_flags_and_caps_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCREENING_MODE", "agent")
    monkeypatch.setenv("AGENT_EVIDENCE_ORCHESTRATION_ENABLED", "true")
    from agent.config import get_settings

    get_settings.cache_clear()

    state = {
        "github_repo_analyses": {
            "username": "dev",
            "candidate_tags": ["backend_engineer"],
            "selected_sandbox_repo_urls": ["https://github.com/dev/service"],
        },
        "github_repo_structures": {
            "https://github.com/dev/service": {
                "classification": "aligned",
                "_file_paths": [
                    "README.md",
                    "requirements.txt",
                    "app/api/routes.py",
                    "app/services/order_service.py",
                ],
            }
        },
    }

    fake_report = {
        "repo": "dev/service",
        "url": "https://github.com/dev/service",
        "clone_ok": True,
        "repo_profile": {"top_files": []},
        "findings": [],
    }

    captured_focus: dict = {}

    async def _capture_eval(urls, settings, file_focus_by_url=None):
        captured_focus.update(file_focus_by_url or {})
        return [fake_report]

    with patch("agent.adk_tools._evaluate_sandbox_repos", new=AsyncMock(side_effect=_capture_eval)):
        result = await execute_sandbox_analysis_for_state(
            state,
            build_heuristic_fallback_repo_specs(state, ["https://github.com/dev/service"]),
            allow_empty_focus_paths=True,
        )

    assert result["ok"] is True
    assert result["sandbox_heuristic_fallback"] is True
    assert state.get("sandbox_heuristic_fallback") is True
    assert state.get("sandbox_completed_by_agent") is not True
    focus_spec = captured_focus["https://github.com/dev/service"]
    assert focus_spec["pick_mode"] == "legacy"
    assert focus_spec["max_files"] == 5
    assert len(focus_spec["focus_paths"]) <= 5


@pytest.mark.asyncio
async def test_ensure_sandbox_evidence_runs_heuristic_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCREENING_MODE", "agent")
    monkeypatch.setenv("AGENT_EVIDENCE_ORCHESTRATION_ENABLED", "true")
    from agent.config import get_settings

    get_settings.cache_clear()

    state = {
        "application_id": "00000000-0000-0000-0000-000000000001",
        "github_repo_analyses": {
            "selected_sandbox_repo_urls": ["https://github.com/dev/service"],
        },
        "github_repo_structures": {
            "https://github.com/dev/service": {
                "classification": "aligned",
                "_file_paths": ["app/api/routes.py"],
            }
        },
    }

    fake_report = {
        "repo": "dev/service",
        "url": "https://github.com/dev/service",
        "clone_ok": True,
        "repo_profile": {},
        "findings": [],
    }

    with patch(
        "agent.adk_tools._evaluate_sandbox_repos",
        new=AsyncMock(return_value=[fake_report]),
    ):
        ran = await ensure_sandbox_evidence(state)

    assert ran is True
    assert state["github_repo_analyses"]["sandbox_heuristic_fallback"] is True
    assert state["github_repo_analyses"]["sandbox_reports"]


@pytest.mark.asyncio
async def test_agent_sandbox_clears_risk_only_pre_pass_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SCREENING_MODE", "agent")
    monkeypatch.setenv("AGENT_EVIDENCE_ORCHESTRATION_ENABLED", "true")
    from agent.config import get_settings

    get_settings.cache_clear()

    state = {
        "github_repo_analyses": {
            "username": "dev",
            "candidate_tags": ["backend_engineer"],
            "selected_sandbox_repo_urls": ["https://github.com/dev/service"],
            "sandbox_reports": [
                {"url": "https://github.com/dev/service", "evaluation_mode": "risk_only"}
            ],
            "sandbox_risk_only_pre_pass": True,
        },
        "sandbox_risk_only_pre_pass": True,
        "github_repo_structures": {
            "https://github.com/dev/service": {
                "classification": "aligned",
                "_file_paths": ["app/api/routes.py"],
            }
        },
    }

    fake_report = {
        "repo": "dev/service",
        "url": "https://github.com/dev/service",
        "clone_ok": True,
        "repo_profile": {"top_files": [{"path": "app/api/routes.py"}]},
        "findings": [],
    }

    with patch(
        "agent.adk_tools._evaluate_sandbox_repos",
        new=AsyncMock(return_value=[fake_report]),
    ):
        result = await execute_sandbox_analysis_for_state(
            state,
            [
                {
                    "repo_url": "https://github.com/dev/service",
                    "classification": "aligned",
                    "focus_paths": [{"path": "app/api/routes.py"}],
                }
            ],
        )

    assert result["ok"] is True
    assert state.get("sandbox_risk_only_pre_pass") is None
    assert "sandbox_risk_only_pre_pass" not in state["github_repo_analyses"]
    assert state.get("sandbox_completed_by_agent") is True
