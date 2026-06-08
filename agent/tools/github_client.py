"""Async GitHub REST API client using httpx."""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from agent.config import get_settings

logger = logging.getLogger("exaai_adk.github_client")


@dataclass
class RepoMeta:
    name: str
    owner: str
    url: str
    description: str | None
    language: str | None
    stars: int
    forks: int
    is_fork: bool
    default_branch: str
    updated_at: str
    topics: list[str] = field(default_factory=list)


@dataclass
class TreeEntry:
    path: str
    mode: str
    type: str  # "blob" or "tree"
    sha: str
    size: int | None = None


@dataclass
class CommitMeta:
    sha: str
    message: str
    author_name: str
    date: str


class GitHubClient:
    """Async client for interacting with the GitHub REST API.

    Uses a shared ``httpx.AsyncClient`` connection pool so that TCP/TLS
    connections are reused across requests, saving ~100-200ms per call.
    Use as an async context manager::

        async with GitHubClient() as client:
            repos = await client.get_user_repos("octocat")
    """

    _rate_limit_reset_time = 0.0

    def __init__(self, token: str | None = None, timeout: float | None = None) -> None:
        settings = get_settings()
        self.token = (token or settings.github_token).strip()
        self.timeout = timeout or float(settings.github_api_timeout_seconds)

        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": f"exaai-adk/{settings.agent_version}",
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
        else:
            logger.warning(
                "No GITHUB_TOKEN configured. Unauthenticated GitHub API calls are subject "
                "to strict rate limiting (60 requests/hour)."
            )

        # Shared connection pool — created lazily, closed via close() or __aexit__
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared httpx client, creating it on first use."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=self.headers,
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                ),
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying connection pool."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> GitHubClient:
        self._get_client()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Make an async request to the GitHub API with error handling."""
        now = time.time()
        if now < GitHubClient._rate_limit_reset_time:
            logger.warning(f"GitHub API is currently rate limited. Skipping request to {url}")
            mock_request = httpx.Request(method, url, headers=self.headers)
            mock_response = httpx.Response(
                status_code=403,
                request=mock_request,
                content=(
                    b'{"message": "GitHub API rate limit exceeded '
                    b'(skipped by circuit breaker)"}'
                ),
            )
            raise httpx.HTTPStatusError(
                message="GitHub API rate limit exceeded (skipped)",
                request=mock_request,
                response=mock_response,
            )

        client = self._get_client()

        try:
            response = await client.request(method, url, **kwargs)
            # Log rate limit info if available
            limit = response.headers.get("X-RateLimit-Limit")
            remaining = response.headers.get("X-RateLimit-Remaining")
            if remaining and int(remaining) < 10:
                logger.warning(f"GitHub API rate limit is low: {remaining}/{limit} remaining.")

            if remaining and int(remaining) == 0:
                reset_header = response.headers.get("X-RateLimit-Reset")
                try:
                    GitHubClient._rate_limit_reset_time = (
                        float(reset_header) if reset_header else time.time() + 3600
                    )
                except ValueError:
                    GitHubClient._rate_limit_reset_time = time.time() + 3600
                logger.error(
                    f"GitHub API rate limit hit 0. Circuit breaker active "
                    f"until epoch {GitHubClient._rate_limit_reset_time}"
                )

            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403 and "rate limit" in e.response.text.lower():
                logger.error("GitHub API rate limit exceeded.")
                reset_header = e.response.headers.get("X-RateLimit-Reset")
                try:
                    GitHubClient._rate_limit_reset_time = (
                        float(reset_header) if reset_header else time.time() + 3600
                    )
                except ValueError:
                    GitHubClient._rate_limit_reset_time = time.time() + 3600
            else:
                logger.error(
                    f"GitHub API HTTP error {e.response.status_code} on {url}: {e.response.text}"
                )
            raise
        except Exception as e:
            logger.error(f"GitHub API network error on {url}: {e}")
            raise

    async def get_user_repos(self, username: str) -> list[RepoMeta]:
        """Fetch list of public repositories for a user."""
        url = f"https://api.github.com/users/{username}/repos"
        params = {"per_page": 100, "sort": "updated"}

        try:
            response = await self._request("GET", url, params=params)
            repos_data = response.json()

            repos = []
            for repo in repos_data:
                repos.append(
                    RepoMeta(
                        name=repo.get("name", ""),
                        owner=repo.get("owner", {}).get("login", ""),
                        url=repo.get("html_url", ""),
                        description=repo.get("description"),
                        language=repo.get("language"),
                        stars=repo.get("stargazers_count", 0),
                        forks=repo.get("forks_count", 0),
                        is_fork=repo.get("fork", False),
                        default_branch=repo.get("default_branch", "main"),
                        updated_at=repo.get("updated_at", ""),
                        topics=repo.get("topics", []),
                    )
                )
            return repos
        except Exception as e:
            logger.error(f"Failed to fetch repos for user {username}: {e}")
            return []

    async def get_repo_languages(self, owner: str, repo: str) -> dict[str, int]:
        """Fetch the bytes of code written in each language in a repository."""
        url = f"https://api.github.com/repos/{owner}/{repo}/languages"

        try:
            response = await self._request("GET", url)
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch languages for {owner}/{repo}: {e}")
            return {}

    async def get_repo_tree(self, owner: str, repo: str, branch: str = "main") -> list[TreeEntry]:
        """Fetch recursive file tree of a repository."""
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}"
        params = {"recursive": "1"}

        try:
            response = await self._request("GET", url, params=params)
            tree_data = response.json().get("tree", [])

            entries = []
            for entry in tree_data:
                entries.append(
                    TreeEntry(
                        path=entry.get("path", ""),
                        mode=entry.get("mode", ""),
                        type=entry.get("type", ""),
                        sha=entry.get("sha", ""),
                        size=entry.get("size"),
                    )
                )
            return entries
        except Exception as e:
            logger.error(f"Failed to fetch file tree for {owner}/{repo} branch {branch}: {e}")
            return []

    async def get_file_content(self, owner: str, repo: str, path: str) -> str:
        """Fetch file content from a repository."""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

        try:
            response = await self._request("GET", url)
            data = response.json()
            if "content" in data and data.get("encoding") == "base64":
                content_str = data["content"].replace("\n", "").replace("\r", "")
                content_bytes = base64.b64decode(content_str.encode("utf-8"))
                return content_bytes.decode("utf-8", errors="replace")
            return ""
        except Exception as e:
            logger.error(f"Failed to fetch file content for {owner}/{repo}/{path}: {e}")
            return ""

    async def get_recent_commits(
        self, owner: str, repo: str, author: str | None = None, limit: int = 10
    ) -> list[CommitMeta]:
        """Fetch recent commits in a repository."""
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        params: dict[str, Any] = {"per_page": limit}
        if author:
            params["author"] = author

        try:
            response = await self._request("GET", url, params=params)
            commits_data = response.json()

            commits = []
            for c in commits_data:
                commit_obj = c.get("commit", {})
                author_obj = commit_obj.get("author", {})
                commits.append(
                    CommitMeta(
                        sha=c.get("sha", ""),
                        message=commit_obj.get("message", ""),
                        author_name=author_obj.get("name", ""),
                        date=author_obj.get("date", ""),
                    )
                )
            return commits
        except Exception as e:
            logger.error(f"Failed to fetch commits for {owner}/{repo}: {e}")
            return []

    async def get_user_events(self, username: str) -> list[dict[str, Any]]:
        """Fetch public events for a user (contains PRs, issues, reviews)."""
        url = f"https://api.github.com/users/{username}/events"
        params = {"per_page": 30}

        try:
            response = await self._request("GET", url, params=params)
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch public events for user {username}: {e}")
            return []
