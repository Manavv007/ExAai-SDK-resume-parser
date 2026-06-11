"""Shared pytest fixtures."""

from pathlib import Path

import pytest

from agent.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _test_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Default API keys for CI/local pytest (import must not require real secrets)."""
    monkeypatch.setattr("agent.config._ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    monkeypatch.setenv("API_KEYS", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("SCREENING_MODE", "pipeline")
    monkeypatch.setenv("AGENT_EVIDENCE_ORCHESTRATION_ENABLED", "false")
    monkeypatch.setenv("SCREENING_RESULT_STORE_PATH", str(tmp_path / "screening-results"))
    get_settings.cache_clear()
    from agent.llm_client import reset_llm_call_count
    from agent.tools.github_client import GitHubClient

    reset_llm_call_count()
    GitHubClient._rate_limit_reset_time = 0.0
    yield
    reset_llm_call_count()
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
