"""Unit tests for GitHub repository analyzer."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.sandbox.models import RepoExecutionReport
from agent.tools.github_analyzer import (
    _evaluate_sandbox_repos,
    _extract_dependencies_from_manifest,
    _get_jd_keywords,
    _parse_dependencies_from_code,
    _score_repo_relevance,
    _select_sandbox_repo_urls,
    _select_static_repos,
    analyze_github_repos,
    extract_github_repo_urls,
    extract_github_username,
    merge_github_repo_urls,
    normalize_github_profile_url,
    normalize_github_repo_url,
    resolve_github_username,
    sync_github_identity,
)
from agent.tools.github_client import RepoMeta


def test_extract_github_username() -> None:
    urls = [
        "https://github.com/johnsmith",
        "https://www.github.com/janedoe/repo-name",
        "http://github.com/bob",
        "https://linkedin.com/in/someone",
    ]
    assert extract_github_username(urls) == "johnsmith"

    # Test skipping standard GitHub paths
    urls_standard = [
        "https://github.com/sponsors",
        "https://github.com/trending",
        "https://github.com/someuser",
    ]
    assert extract_github_username(urls_standard) == "someuser"


def test_extract_github_repo_urls_normalizes_unique_resume_repos() -> None:
    urls = [
        "https://github.com/testuser/api-service",
        "https://www.github.com/testuser/api-service/",
        "https://github.com/testuser/web-app.git?tab=readme",
        "https://github.com/testuser",
        "https://github.com/topics/python",
    ]

    assert normalize_github_repo_url(urls[0]) == "https://github.com/testuser/api-service"
    assert extract_github_repo_urls(urls) == [
        "https://github.com/testuser/api-service",
        "https://github.com/testuser/web-app",
    ]


def test_normalize_github_profile_url() -> None:
    assert normalize_github_profile_url("https://github.com/Manavv007") == (
        "https://github.com/Manavv007"
    )
    assert normalize_github_profile_url("https://github.com/Manavv007/") == (
        "https://github.com/Manavv007"
    )
    assert normalize_github_profile_url("https://github.com/Manavv007/my-repo") is None
    assert normalize_github_profile_url("https://github.com/topics/python") is None


def test_resolve_github_username_from_discovered_repo_urls() -> None:
    state = {
        "profile_urls": ["https://manavbhavsar-portfolio.vercel.app/"],
        "discovered_github_repo_urls": [
            "https://github.com/manavv007/exaai-adk",
            "https://github.com/manavv007/other-repo",
        ],
    }
    assert resolve_github_username(state) == "manavv007"


def test_sync_github_identity_sets_username_and_analysis_shell() -> None:
    state = {
        "profile_urls": ["https://manavbhavsar-portfolio.vercel.app/"],
        "discovered_github_repo_urls": ["https://github.com/Manavv007/repo-one"],
    }
    username = sync_github_identity(state)
    assert username == "Manavv007"
    assert state["github_username"] == "Manavv007"
    github = state["github_repo_analyses"]
    assert isinstance(github, dict)
    assert github["username"] == "Manavv007"
    assert github["repo_analyses"] == []


def test_merge_github_repo_urls_preserves_order_and_dedupes() -> None:
    merged = merge_github_repo_urls(
        ["https://github.com/a/one", "https://github.com/a/two"],
        ["https://github.com/a/two", "https://github.com/a/three"],
    )
    assert merged == [
        "https://github.com/a/one",
        "https://github.com/a/two",
        "https://github.com/a/three",
    ]


def test_get_jd_keywords() -> None:
    jd_structured = {
        "job_title": "Senior Python Developer",
        "must_have": ["Strong experience in FastAPI and PostgreSQL", "Docker"],
        "nice_to_have": ["AWS experience", "Kubernetes is a plus"],
    }
    keywords = _get_jd_keywords(jd_structured)
    assert "python" in keywords
    assert "fastapi" in keywords
    assert "docker" in keywords
    assert "aws" in keywords
    assert "kubernetes" in keywords


def test_score_repo_relevance() -> None:
    repo = RepoMeta(
        name="my-fastapi-app",
        owner="testuser",
        url="https://github.com/testuser/my-fastapi-app",
        description="A cool app using FastAPI and PostgreSQL",
        language="Python",
        stars=10,
        forks=2,
        is_fork=False,
        default_branch="main",
        updated_at="2026-06-06T12:00:00Z",
        topics=["fastapi", "postgres"],
    )
    jd_keywords = {"python", "fastapi", "docker"}
    score = _score_repo_relevance(repo, jd_keywords)
    # Match: language (python), name (fastapi), description (fastapi),
    # topics (fastapi), stars (10 * 0.1)
    assert score > 20.0


def test_select_static_repos_uses_all_resolved_resume_repos() -> None:
    api_repo = RepoMeta(
        name="api-service",
        owner="testuser",
        url="https://github.com/testuser/api-service",
        description=None,
        language="Python",
        stars=1,
        forks=0,
        is_fork=False,
        default_branch="main",
        updated_at="2026-06-06T12:00:00Z",
    )
    web_repo = RepoMeta(
        name="web-app",
        owner="testuser",
        url="https://github.com/testuser/web-app",
        description=None,
        language="TypeScript",
        stars=100,
        forks=0,
        is_fork=False,
        default_branch="main",
        updated_at="2026-06-06T12:00:00Z",
    )
    settings = MagicMock()
    settings.sandbox_max_resume_repos = 10

    selected, mode = _select_static_repos(
        resolved_resume_repos=[api_repo, web_repo],
        ranked_repos=[web_repo, api_repo],
        resume_repo_urls=[
            "https://github.com/testuser/api-service",
            "https://github.com/testuser/web-app",
        ],
        settings=settings,
    )

    assert [repo.name for repo in selected] == ["api-service", "web-app"]
    assert mode == "resume_repos"


@pytest.mark.asyncio
async def test_resolve_resume_repos_fetches_missing_from_api() -> None:
    from agent.tools.github_analyzer import _resolve_resume_repos_for_analysis

    listed_repo = RepoMeta(
        name="listed",
        owner="testuser",
        url="https://github.com/testuser/listed",
        description=None,
        language="Python",
        stars=1,
        forks=0,
        is_fork=False,
        default_branch="main",
        updated_at="2026-06-06T12:00:00Z",
    )
    fetched_repo = RepoMeta(
        name="resume-only",
        owner="testuser",
        url="https://github.com/testuser/resume-only",
        description=None,
        language="Go",
        stars=2,
        forks=0,
        is_fork=False,
        default_branch="main",
        updated_at="2026-06-06T12:00:00Z",
    )
    client = MagicMock()
    client.get_repo_meta = AsyncMock(return_value=fetched_repo)
    settings = MagicMock()
    settings.sandbox_max_resume_repos = 10

    resolved = await _resolve_resume_repos_for_analysis(
        client=client,
        all_repos=[listed_repo],
        resume_repo_urls=[
            "https://github.com/testuser/listed",
            "https://github.com/testuser/resume-only",
        ],
        settings=settings,
    )

    assert [repo.name for repo in resolved] == ["listed", "resume-only"]
    client.get_repo_meta.assert_awaited_once_with("testuser", "resume-only")


def test_select_sandbox_repo_urls_includes_six_resume_repos() -> None:
    settings = MagicMock()
    settings.sandbox_max_resume_repos = 12
    six_urls = [f"https://github.com/testuser/repo{i}" for i in range(1, 7)]
    selected, mode = _select_sandbox_repo_urls(
        ranked_repos=[],
        resume_repo_urls=six_urls,
        settings=settings,
    )
    assert selected == six_urls
    assert mode == "resume_repos"


def test_select_sandbox_repo_urls_uses_all_resume_repos_before_fallback() -> None:
    ranked_repos = [
        RepoMeta(
            name="ranked",
            owner="testuser",
            url="https://github.com/testuser/ranked",
            description=None,
            language="Python",
            stars=100,
            forks=0,
            is_fork=False,
            default_branch="main",
            updated_at="2026-06-06T12:00:00Z",
        )
    ]
    settings = MagicMock()
    settings.sandbox_max_resume_repos = 5
    settings.sandbox_max_profile_repos = 2

    selected, mode = _select_sandbox_repo_urls(
        ranked_repos=ranked_repos,
        resume_repo_urls=[
            "https://github.com/testuser/one",
            "https://github.com/testuser/two",
        ],
        settings=settings,
    )

    assert selected == [
        "https://github.com/testuser/one",
        "https://github.com/testuser/two",
    ]
    assert mode == "resume_repos"

    settings.sandbox_max_resume_repos = 10
    five_urls = [f"https://github.com/testuser/project{i}" for i in range(1, 6)]
    all_five, five_mode = _select_sandbox_repo_urls(
        ranked_repos=ranked_repos,
        resume_repo_urls=five_urls,
        settings=settings,
    )
    assert all_five == five_urls
    assert five_mode == "resume_repos"

    fallback, fallback_mode = _select_sandbox_repo_urls(
        ranked_repos=ranked_repos,
        resume_repo_urls=[],
        settings=settings,
    )

    assert fallback == ["https://github.com/testuser/ranked"]
    assert fallback_mode == "ranked_profile_repos"


@pytest.mark.asyncio
async def test_evaluate_sandbox_repos_runs_provider_in_parallel() -> None:
    class FakeProvider:
        async def evaluate_repo(self, *, repo_url, repo_name, commands, file_focus=None):
            return RepoExecutionReport(
                repo=repo_name,
                url=repo_url,
                provider="cloud_run",
                clone_ok=True,
                detected_stack=["python"],
                summary="ok",
            )

    settings = MagicMock()
    settings.github_clone_analysis_enabled = True
    settings.sandbox_provider = "cloud_run"

    with patch("agent.sandbox.providers.create_sandbox_provider", return_value=FakeProvider()):
        reports = await _evaluate_sandbox_repos(
            [
                "https://github.com/testuser/one",
                "https://github.com/testuser/two",
            ],
            settings,
        )

    assert [report["repo"] for report in reports] == ["testuser/one", "testuser/two"]
    assert all(report["clone_ok"] for report in reports)


@pytest.mark.asyncio
async def test_evaluate_sandbox_repos_preserves_fast_results_on_batch_timeout() -> None:
    class MixedSpeedProvider:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def evaluate_repo(self, *, repo_url, repo_name, commands, file_focus=None):
            self.calls.append(repo_url)
            if repo_url.endswith("/slow"):
                await asyncio.sleep(1)
            return RepoExecutionReport(repo=repo_name, url=repo_url, clone_ok=True)

    settings = MagicMock()
    settings.sandbox_provider = "cloud_run"
    settings.sandbox_wait_seconds = 0.05
    provider = MixedSpeedProvider()

    with patch("agent.sandbox.providers.create_sandbox_provider", return_value=provider):
        reports = await _evaluate_sandbox_repos(
            [
                "https://github.com/testuser/fast",
                "https://github.com/testuser/slow",
            ],
            settings,
            _retry_pass=1,
        )

    assert reports[0]["clone_ok"] is True
    assert reports[0]["repo"] == "testuser/fast"
    assert reports[1]["timed_out"] is True


@pytest.mark.asyncio
async def test_evaluate_sandbox_repos_returns_timeout_reports() -> None:
    class SlowProvider:
        async def evaluate_repo(self, *, repo_url, repo_name, commands, file_focus=None):
            await asyncio.sleep(1)
            return RepoExecutionReport(repo=repo_name, url=repo_url, clone_ok=True)

    settings = MagicMock()
    settings.github_clone_analysis_enabled = True
    settings.sandbox_provider = "cloud_run"
    settings.sandbox_wait_seconds = 0.01

    with patch("agent.sandbox.providers.create_sandbox_provider", return_value=SlowProvider()):
        reports = await _evaluate_sandbox_repos(
            ["https://github.com/testuser/slow"],
            settings,
        )

    assert reports == [
        {
            "repo": "testuser/slow",
            "url": "https://github.com/testuser/slow",
            "provider": "cloud_run",
            "clone_ok": False,
            "detected_stack": [],
            "repo_profile": {},
            "commands": [],
            "quality_signals": {},
            "risk_flags": [],
            "findings": [],
            "summary": (
                "Sandbox evaluation did not finish within the screening wait budget; "
                "static GitHub evidence was used instead."
            ),
            "overall_assessment": "",
            "confidence": "low",
            "timed_out": True,
            "skipped_reason": "Sandbox wait budget exceeded after 0s.",
        }
    ]


def test_parse_dependencies_from_code() -> None:
    py_code = """
import os
import sys
import fastapi
from sqlalchemy import create_engine
from typing import List
"""
    js_code = """
import React from 'react';
const lodash = require('lodash');
"""
    go_code = """
package main
import (
    "fmt"
    "github.com/gin-gonic/gin"
)
"""
    assert "fastapi" in _parse_dependencies_from_code("app.py", py_code)
    assert "sqlalchemy" in _parse_dependencies_from_code("app.py", py_code)
    assert "os" not in _parse_dependencies_from_code("app.py", py_code)  # stdlib ignored

    assert "react" in _parse_dependencies_from_code("app.js", js_code)
    assert "lodash" in _parse_dependencies_from_code("app.js", js_code)

    assert "gin" in _parse_dependencies_from_code("main.go", go_code)


def test_extract_dependencies_from_manifest() -> None:
    pkg_json = """{
        "dependencies": {
            "express": "^4.18.2",
            "cors": "^2.8.5"
        }
    }"""
    pyproject = """
[tool.poetry.dependencies]
python = "^3.10"
fastapi = "^0.100.0"
"""
    req_txt = """
flask>=2.0.0
pytest==7.1.0
# some comment
"""
    cargo = """
[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = "1.0"
"""
    assert _extract_dependencies_from_manifest("package.json", pkg_json) == {"express", "cors"}
    assert _extract_dependencies_from_manifest("pyproject.toml", pyproject) == {"fastapi"}
    assert _extract_dependencies_from_manifest("requirements.txt", req_txt) == {"flask", "pytest"}
    assert _extract_dependencies_from_manifest("Cargo.toml", cargo) == {"tokio", "serde"}


@pytest.mark.asyncio
async def test_analyze_github_repos() -> None:
    mock_repos = [
        RepoMeta(
            name="repo1",
            owner="testuser",
            url="https://github.com/testuser/repo1",
            description="A Python FastAPI project",
            language="Python",
            stars=10,
            forks=2,
            is_fork=False,
            default_branch="main",
            updated_at="2026-06-06T12:00:00Z",
        )
    ]

    mock_tree = [
        MagicMock(path="README.md", type="blob"),
        MagicMock(path="main.py", type="blob"),
        MagicMock(path="requirements.txt", type="blob"),
        MagicMock(path="tests/test_main.py", type="blob"),
    ]

    def mock_get_content(owner, repo, path):
        if path == "requirements.txt":
            return "fastapi\nuvicorn"
        if path == "README.md":
            return "This is repo1 readme"
        if path == "main.py":
            return "import fastapi\nprint('hello')"
        return ""

    jd_structured = {
        "job_title": "Python Engineer",
        "must_have": ["FastAPI"],
        "nice_to_have": [],
    }

    # Patch GitHubClient calls
    with (
        patch(
            "agent.tools.github_client.GitHubClient.get_user_repos", new_callable=AsyncMock
        ) as mock_get_repos,
        patch(
            "agent.tools.github_client.GitHubClient.get_repo_languages", new_callable=AsyncMock
        ) as mock_langs,
        patch(
            "agent.tools.github_client.GitHubClient.get_repo_tree", new_callable=AsyncMock
        ) as mock_get_tree,
        patch(
            "agent.tools.github_client.GitHubClient.get_file_content", new_callable=AsyncMock
        ) as mock_get_file_content,
        patch(
            "agent.tools.github_client.GitHubClient.get_recent_commits", new_callable=AsyncMock
        ) as mock_get_commits,
        patch(
            "agent.tools.github_client.GitHubClient.get_user_events", new_callable=AsyncMock
        ) as mock_get_events,
        patch(
            "agent.tools.github_analyzer._generate_coding_style_summary", new_callable=AsyncMock
        ) as mock_gen_summary,
        patch("agent.tools.github_analyzer.get_settings") as mock_get_settings,
    ):
        mock_settings = MagicMock()
        mock_settings.github_analysis_enabled = True
        mock_settings.github_llm_summary_enabled = True
        mock_settings.max_repos_to_analyze = 3
        mock_settings.sandbox_max_resume_repos = 5
        mock_settings.sandbox_max_profile_repos = 2
        mock_settings.github_clone_analysis_enabled = False
        mock_settings.max_files_per_repo = 15
        mock_settings.github_content_token_cap = 12000
        mock_settings.gemini_api_key = "test-api-key"
        mock_settings.gemini_model_id = "gemini-2.0-flash"
        mock_settings.llm_provider = "gemini"
        mock_get_settings.return_value = mock_settings

        mock_get_repos.return_value = mock_repos
        mock_langs.return_value = {"Python": 1000}
        mock_get_tree.return_value = mock_tree
        mock_get_file_content.side_effect = mock_get_content
        mock_get_commits.return_value = []
        mock_get_events.return_value = []
        mock_gen_summary.return_value = (
            "Clean FastAPI setup with unit tests.",
            "strong",
            "Active open source collaborator.",
            "Descriptive, high quality commits.",
        )

        analysis = await analyze_github_repos(
            username="testuser",
            repo_urls=["https://github.com/testuser"],
            jd_structured=jd_structured,
        )

        assert analysis["username"] == "testuser"
        assert analysis["total_public_repos"] == 1
        assert analysis["total_stars"] == 10
        assert analysis["overall_github_signal"] == "strong"
        assert analysis["coding_style_summary"] == "Clean FastAPI setup with unit tests."
        assert analysis["collaboration_summary"] == "Active open source collaborator."
        assert analysis["commit_hygiene"] == "Descriptive, high quality commits."

        assert len(analysis["repo_analyses"]) == 1
        repo_an = analysis["repo_analyses"][0]
        assert repo_an["name"] == "repo1"
        assert repo_an["has_tests"] is True
        assert "fastapi" in repo_an["dependency_summary"]


@pytest.mark.asyncio
async def test_analyze_github_repos_attaches_sandbox_reports() -> None:
    mock_repos = [
        RepoMeta(
            name="repo1",
            owner="testuser",
            url="https://github.com/testuser/repo1",
            description="A Python project",
            language="Python",
            stars=1,
            forks=0,
            is_fork=False,
            default_branch="main",
            updated_at="2026-06-06T12:00:00Z",
        )
    ]
    sandbox_reports = [
        {
            "repo": "testuser/repo1",
            "url": "https://github.com/testuser/repo1",
            "provider": "cloud_run",
            "clone_ok": True,
            "detected_stack": ["python"],
            "commands": [],
            "quality_signals": {},
            "risk_flags": [],
            "summary": "ok",
            "timed_out": False,
        }
    ]

    with (
        patch(
            "agent.tools.github_client.GitHubClient.get_user_repos", new_callable=AsyncMock
        ) as mock_get_repos,
        patch(
            "agent.tools.github_client.GitHubClient.get_user_events", new_callable=AsyncMock
        ) as mock_get_events,
        patch(
            "agent.tools.github_analyzer._analyze_single_repo", new_callable=AsyncMock
        ) as mock_analyze_repo,
        patch(
            "agent.tools.github_analyzer._evaluate_sandbox_repos", new_callable=AsyncMock
        ) as mock_sandbox,
        patch("agent.tools.github_analyzer.get_settings") as mock_get_settings,
    ):
        mock_settings = MagicMock()
        mock_settings.github_analysis_enabled = True
        mock_settings.github_llm_summary_enabled = False
        mock_settings.github_clone_analysis_enabled = True
        mock_settings.sandbox_max_resume_repos = 5
        mock_settings.sandbox_max_profile_repos = 2
        mock_settings.max_repos_to_analyze = 3
        mock_settings.github_content_token_cap = 12000
        mock_get_settings.return_value = mock_settings

        mock_get_repos.return_value = mock_repos
        mock_get_events.return_value = []
        from agent.tools.github_analyzer import RepoAnalysis

        mock_analyze_repo.return_value = RepoAnalysis(
            name="repo1",
            url="https://github.com/testuser/repo1",
            description="A Python project",
            languages={"Python": 100.0},
            stars=1,
            is_fork=False,
            project_type="codebase",
            has_tests=True,
            has_ci=False,
            has_docs=True,
            has_docker=False,
            dependency_summary="pytest",
        )
        mock_sandbox.return_value = sandbox_reports

        analysis = await analyze_github_repos(
            username="testuser",
            repo_urls=["https://github.com/testuser/repo1"],
            jd_structured={"job_title": "Python Engineer"},
        )

    assert analysis["selected_sandbox_repo_urls"] == ["https://github.com/testuser/repo1"]
    assert analysis["sandbox_reports"] == sandbox_reports
    assert analysis["repo_selection_mode"] == "resume_repos"
    mock_sandbox.assert_awaited_once()


@pytest.mark.asyncio
async def test_analyze_github_repos_deferred_skips_inline_sandbox_wait() -> None:
    mock_repos = [
        RepoMeta(
            name="repo1",
            owner="testuser",
            url="https://github.com/testuser/repo1",
            description="A Python project",
            language="Python",
            stars=1,
            forks=0,
            is_fork=False,
            default_branch="main",
            updated_at="2026-06-06T12:00:00Z",
        )
    ]

    with (
        patch(
            "agent.tools.github_client.GitHubClient.get_user_repos", new_callable=AsyncMock
        ) as mock_get_repos,
        patch(
            "agent.tools.github_client.GitHubClient.get_user_events", new_callable=AsyncMock
        ) as mock_get_events,
        patch(
            "agent.tools.github_analyzer._analyze_single_repo", new_callable=AsyncMock
        ) as mock_analyze_repo,
        patch(
            "agent.tools.github_analyzer._evaluate_sandbox_repos", new_callable=AsyncMock
        ) as mock_sandbox,
        patch("agent.tools.github_analyzer.get_settings") as mock_get_settings,
    ):
        mock_settings = MagicMock()
        mock_settings.github_analysis_enabled = True
        mock_settings.github_llm_summary_enabled = False
        mock_settings.github_clone_analysis_enabled = True
        mock_settings.sandbox_max_resume_repos = 5
        mock_settings.sandbox_max_profile_repos = 2
        mock_settings.max_repos_to_analyze = 3
        mock_settings.github_content_token_cap = 12000
        mock_get_settings.return_value = mock_settings

        mock_get_repos.return_value = mock_repos
        mock_get_events.return_value = []
        from agent.tools.github_analyzer import RepoAnalysis

        mock_analyze_repo.return_value = RepoAnalysis(
            name="repo1",
            url="https://github.com/testuser/repo1",
            description="A Python project",
            languages={"Python": 100.0},
            stars=1,
            is_fork=False,
            project_type="codebase",
            has_tests=True,
            has_ci=False,
            has_docs=True,
            has_docker=False,
            dependency_summary="pytest",
        )

        analysis = await analyze_github_repos(
            username="testuser",
            repo_urls=["https://github.com/testuser/repo1"],
            jd_structured={"job_title": "Python Engineer"},
            sandbox_mode="deferred",
        )

    assert analysis["selected_sandbox_repo_urls"] == ["https://github.com/testuser/repo1"]
    assert analysis["sandbox_reports"] == []
    mock_sandbox.assert_not_awaited()


def test_analyze_collaboration() -> None:
    from agent.tools.github_analyzer import _analyze_collaboration

    events = [
        {
            "type": "PullRequestEvent",
            "repo": {"name": "other-owner/repo2"},
            "payload": {"action": "opened"},
        },
        {
            "type": "PullRequestReviewEvent",
            "repo": {"name": "testuser/repo1"},
            "payload": {},
        },
        {
            "type": "PushEvent",
            "repo": {"name": "other-owner/repo3"},
            "payload": {},
        },
    ]
    res = _analyze_collaboration("testuser", events)
    assert res["pull_requests_created"] == 1
    assert res["pull_requests_reviewed"] == 1
    assert "other-owner/repo2" in res["external_contributions"]
    assert "other-owner/repo3" in res["external_contributions"]
    assert "testuser/repo1" not in res["external_contributions"]
    assert "opened 1 PRs" in res["summary"]


def test_score_path_relevance() -> None:
    from agent.tools.github_analyzer import _score_path_relevance

    jd_keywords = {"db", "fastapi", "auth"}

    score1 = _score_path_relevance("src/auth/service.py", jd_keywords)
    score2 = _score_path_relevance("docs/readme.txt", jd_keywords)

    assert score1 > score2


def test_extract_relevant_snippet() -> None:
    from agent.tools.github_analyzer import _extract_relevant_snippet

    content = "line 1\nline 2 with database\nline 3\nline 4"
    jd_keywords = {"database"}
    snippet = _extract_relevant_snippet(content, jd_keywords, max_chars=30)
    assert "database" in snippet


def test_generate_coding_style_summary_heuristic() -> None:
    from agent.tools.github_analyzer import _generate_coding_style_summary_heuristic

    repos_data = [
        {
            "name": "repo1",
            "languages": {"Python": 80.0, "JavaScript": 20.0},
            "has_tests": True,
            "has_ci": False,
            "has_docs": True,
            "has_docker": False,
            "commit_quality": "descriptive",
            "stars": 1,
            "complexity_estimate": "simple",
        }
    ]
    summary, signal, collab, hygiene = _generate_coding_style_summary_heuristic(
        "testuser", repos_data, "Opened 1 PR."
    )

    assert "Python" in summary
    assert "JavaScript" in summary
    assert "repo1" in summary
    assert "testing" in summary
    assert "documentation" in summary
    assert signal == "strong"
    assert "detailed commit messages" in hygiene
    assert collab == "Opened 1 PR."


def test_generate_coding_style_summary_heuristic_empty() -> None:
    from agent.tools.github_analyzer import _generate_coding_style_summary_heuristic

    summary, signal, collab, hygiene = _generate_coding_style_summary_heuristic(
        "testuser", [], "No PRs."
    )
    assert "No public repositories" in summary
    assert signal == "none"
    assert collab == "No PRs."
