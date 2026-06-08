"""Shared pytest fixtures."""

import os

import pytest

from agent.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default API keys for CI/local pytest (import must not require real secrets)."""
    monkeypatch.setenv("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", "test-gemini"))
    monkeypatch.setenv("EXA_API_KEY", os.environ.get("EXA_API_KEY", "test-exa"))
    monkeypatch.setenv("API_KEYS", os.environ.get("API_KEYS", "test-key"))
    monkeypatch.setenv("LLM_PROVIDER", os.environ.get("LLM_PROVIDER", "gemini"))
    get_settings.cache_clear()
    from agent.tools.github_client import GitHubClient
    GitHubClient._rate_limit_reset_time = 0.0
    yield
    GitHubClient._rate_limit_reset_time = 0.0
    get_settings.cache_clear()


@pytest.fixture
def test_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    monkeypatch.setenv("API_KEYS", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)
    get_settings.cache_clear()
    return get_settings()
