"""Shared pytest fixtures."""

import pytest

from agent.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def test_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    monkeypatch.setenv("API_KEYS", "test-key")
    get_settings.cache_clear()
    return get_settings()
