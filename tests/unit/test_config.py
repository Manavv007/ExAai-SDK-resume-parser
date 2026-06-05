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
