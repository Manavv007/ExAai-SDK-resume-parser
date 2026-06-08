"""Unit tests for GitHub REST API client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.tools.github_client import GitHubClient


@pytest.fixture(autouse=True)
def reset_rate_limit() -> None:
    GitHubClient._rate_limit_reset_time = 0.0
    yield
    GitHubClient._rate_limit_reset_time = 0.0


@pytest.mark.asyncio
async def test_get_user_repos() -> None:
    client = GitHubClient(token="test-token")
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {
            "name": "repo1",
            "owner": {"login": "testuser"},
            "html_url": "https://github.com/testuser/repo1",
            "description": "desc",
            "language": "Python",
            "stargazers_count": 5,
            "forks_count": 2,
            "fork": False,
            "default_branch": "main",
            "updated_at": "2026-06-06T12:00:00Z",
            "topics": ["python", "api"],
        }
    ]
    mock_response.headers = {"X-RateLimit-Remaining": "5000", "X-RateLimit-Limit": "5000"}
    mock_response.status_code = 200

    with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response
        repos = await client.get_user_repos("testuser")
        assert len(repos) == 1
        assert repos[0].name == "repo1"
        assert repos[0].language == "Python"
        assert repos[0].stars == 5
        assert not repos[0].is_fork


@pytest.mark.asyncio
async def test_get_repo_languages() -> None:
    client = GitHubClient(token="test-token")
    mock_response = MagicMock()
    mock_response.json.return_value = {"Python": 1000, "Go": 500}
    mock_response.status_code = 200

    with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response
        languages = await client.get_repo_languages("testuser", "repo1")
        assert languages == {"Python": 1000, "Go": 500}


@pytest.mark.asyncio
async def test_get_repo_tree() -> None:
    client = GitHubClient(token="test-token")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "tree": [
            {"path": "src/main.py", "mode": "100644", "type": "blob", "sha": "abc", "size": 123}
        ]
    }
    mock_response.status_code = 200

    with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response
        tree = await client.get_repo_tree("testuser", "repo1")
        assert len(tree) == 1
        assert tree[0].path == "src/main.py"
        assert tree[0].type == "blob"


@pytest.mark.asyncio
async def test_get_file_content() -> None:
    client = GitHubClient(token="test-token")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "content": "aW1wb3J0IG9zCg==",  # base64 encoded "import os\n"
        "encoding": "base64",
    }
    mock_response.status_code = 200

    with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response
        content = await client.get_file_content("testuser", "repo1", "src/main.py")
        assert "import os" in content


@pytest.mark.asyncio
async def test_get_recent_commits() -> None:
    client = GitHubClient(token="test-token")
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {
            "sha": "123456",
            "commit": {
                "message": "Commit msg",
                "author": {"name": "Test Author", "date": "2026-06-06T12:00:00Z"},
            },
        }
    ]
    mock_response.status_code = 200

    with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response
        commits = await client.get_recent_commits("testuser", "repo1")
        assert len(commits) == 1
        assert commits[0].sha == "123456"
        assert commits[0].message == "Commit msg"
        assert commits[0].author_name == "Test Author"


@pytest.mark.asyncio
async def test_get_user_events() -> None:
    client = GitHubClient(token="test-token")
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {
            "id": "1",
            "type": "PullRequestEvent",
            "actor": {"login": "testuser"},
            "repo": {"name": "testuser/repo1"},
            "payload": {"action": "opened"},
        }
    ]
    mock_response.status_code = 200

    with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response
        events = await client.get_user_events("testuser")
        assert len(events) == 1
        assert events[0]["type"] == "PullRequestEvent"


@pytest.mark.asyncio
async def test_github_client_rate_limit_circuit_breaker() -> None:
    import time

    # Reset the class-level variable first to be safe
    GitHubClient._rate_limit_reset_time = 0.0

    client = GitHubClient(token="test-token")
    mock_response = MagicMock()
    mock_response.json.return_value = []
    # Remaining 0 starts the circuit breaker!
    mock_response.headers = {
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Limit": "5000",
        "X-RateLimit-Reset": str(time.time() + 100),
    }
    mock_response.status_code = 200

    with patch("httpx.AsyncClient.request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response

        # 1. First call works but sets the rate limit reset time
        repos = await client.get_user_repos("testuser")
        assert repos == []
        assert GitHubClient._rate_limit_reset_time > time.time()

        # 2. Second call should be skipped and fail immediately with HTTPStatusError
        # (caught by get_user_repos and returns empty list)
        repos2 = await client.get_user_repos("testuser")
        assert repos2 == []

        # Verify that mock_request was only called ONCE because the second one was skipped!
        assert mock_request.call_count == 1

    # Clean up class-level variable after the test
    GitHubClient._rate_limit_reset_time = 0.0
