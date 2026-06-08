"""Settings and environment sync."""

from __future__ import annotations

import os

import pytest

from agent.config import get_settings


def test_sync_gemini_env_from_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=from-dotenv-key\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.gemini_api_key == "from-dotenv-key"
    assert os.environ.get("GEMINI_API_KEY") == "from-dotenv-key"
    assert os.environ.get("GOOGLE_API_KEY") == "from-dotenv-key"


def test_jd_parse_use_llm_defaults_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JD_PARSE_USE_LLM", raising=False)
    get_settings.cache_clear()
    assert get_settings().jd_parse_use_llm is False


def test_sandbox_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_CLONE_ANALYSIS_ENABLED", raising=False)
    monkeypatch.delenv("SANDBOX_PROVIDER", raising=False)
    monkeypatch.delenv("SANDBOX_MAX_REPOS", raising=False)
    monkeypatch.delenv("SANDBOX_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("SANDBOX_NETWORK_MODE", raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.github_clone_analysis_enabled is False
    assert settings.sandbox_provider == "cloud_run"
    assert settings.sandbox_max_repos == 2
    assert settings.sandbox_max_resume_repos == 5
    assert settings.sandbox_max_profile_repos == 2
    assert settings.sandbox_timeout_seconds == 300
    assert settings.sandbox_wait_seconds == 45
    assert settings.sandbox_poll_interval_seconds == 2.0
    assert settings.sandbox_network_mode == "install_only"
    assert settings.gcp_region == "asia-south1"
    assert settings.cloud_run_sandbox_job_name == "repo-evaluator"
    assert settings.sandbox_report_prefix == "sandbox-reports"


def test_sandbox_config_normalizes_literals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SANDBOX_PROVIDER", " CLOUD_RUN ")
    monkeypatch.setenv("SANDBOX_NETWORK_MODE", " INSTALL_ONLY ")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.sandbox_provider == "cloud_run"
    assert settings.sandbox_network_mode == "install_only"
