"""JD_PARSE_USE_LLM config gates Gemini JD parsing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.config import get_settings
from agent.tools.parser import parse_jd_structured

JD = "Senior Python Engineer\nMust have:\n- Python\nNice to have:\n- Kubernetes\n"


def test_jd_parse_skips_gemini_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    test_settings,
) -> None:
    monkeypatch.setenv("JD_PARSE_USE_LLM", "false")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    get_settings.cache_clear()

    with patch("agent.tools.parser._parse_jd_with_gemini") as mock_llm:
        result = parse_jd_structured(JD)

    mock_llm.assert_not_called()
    assert any("Python" in item for item in result.must_have)


def test_jd_parse_uses_gemini_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    test_settings,
) -> None:
    monkeypatch.setenv("JD_PARSE_USE_LLM", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    get_settings.cache_clear()

    from agent.tools.parser import JdStructured

    fake = JdStructured(
        job_title="Senior Python Engineer",
        domain="technical",
        industry=None,
        seniority=None,
        must_have=["Python"],
        nice_to_have=["Kubernetes"],
        requirements=[],
    )

    with patch("agent.tools.parser._parse_jd_with_gemini", return_value=fake) as mock_llm:
        result = parse_jd_structured(JD)

    mock_llm.assert_called_once()
    assert result.job_title == "Senior Python Engineer"
