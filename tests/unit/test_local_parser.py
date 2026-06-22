"""Local JD and resume parsing (no LLM)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.local_parser import (
    extract_jd_requirements,
    normalize_extracted_text,
    parse_jd_local,
    parse_resume_local,
)
from agent.tools.parser import parse_jd_structured, parse_resume_structured

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_normalize_extracted_text_reflows_pdf_lines() -> None:
    raw = "Senior Python\nengineer with six years\nof backend work."
    assert normalize_extracted_text(raw) == "Senior Python engineer with six years of backend work."


def test_parse_jd_local_sample_fixture() -> None:
    jd_text = (FIXTURES / "sample_jd.txt").read_text(encoding="utf-8")
    jd = parse_jd_local(jd_text)

    assert jd.job_title == "Software Engineering Intern — AI & Data Pipelines"
    assert jd.domain == "technical"
    assert jd.industry == "EXAai Demo Labs"
    assert any("Python" in item for item in jd.must_have)
    assert any("Git" in item for item in jd.must_have)
    assert any("Docker" in item for item in jd.nice_to_have)
    assert len(jd.requirements) >= 5
    assert all(req.requirement_type for req in jd.requirements)


def test_parse_jd_local_software_domain_fixture() -> None:
    jd_text = (FIXTURES / "domains" / "software" / "jd.txt").read_text(encoding="utf-8")
    jd = parse_jd_local(jd_text)

    assert jd.job_title == "Senior Backend Engineer"
    assert jd.seniority == "senior"
    assert any("Python" in item for item in jd.must_have)
    assert any("Kubernetes" in item for item in jd.nice_to_have)


def test_parse_jd_structured_defaults_to_local(
    monkeypatch: pytest.MonkeyPatch,
    test_settings,
) -> None:
    monkeypatch.setenv("JD_PARSE_USE_LLM", "false")
    from agent.config import get_settings

    get_settings.cache_clear()

    jd = parse_jd_structured(
        "Data Engineer\nRequirements:\n- Must have: Python\nNice to have:\n- Preferred: Spark\n"
    )
    assert jd.must_have == ["Python"]
    assert jd.nice_to_have == ["Spark"]
    assert jd.requirements[0].requirement_type == "technical_skill"


def test_parse_resume_local_software_fixture() -> None:
    text = (FIXTURES / "domains" / "software" / "resume.txt").read_text(encoding="utf-8")
    resume = parse_resume_local(text)

    assert resume.candidate_name == "Alex Chen"
    assert resume.headline == "Senior Software Engineer"
    assert resume.experience_years == 6
    assert any(skill.lower() == "python" for skill in resume.skills)
    assert any("FastAPI" in h for h in resume.experience_highlights)


def test_parse_resume_structured_wrapper() -> None:
    text = (FIXTURES / "sample_resume.txt").read_text(encoding="utf-8")
    resume = parse_resume_structured(text)

    assert resume.headline == "Senior Software Engineer"
    assert resume.candidate_name == "Jane Doe"


def test_extract_jd_inline_required_bullets() -> None:
    must, nice = extract_jd_requirements(
        "Role\nRequirements:\n- Required: PostgreSQL\n- Must have: Python, FastAPI\n"
        "Nice to have:\n- Preferred: Docker, Kubernetes\n"
    )
    assert "PostgreSQL" in must
    assert "Python" in must
    assert "FastAPI" in must
    assert "Docker" in nice
    assert "Kubernetes" in nice


def test_compound_requirements_are_not_shredded() -> None:
    """Requirement sentences with parentheses or and/or must stay whole (bug fix)."""
    jd_text = (FIXTURES / "sample_jd.txt").read_text(encoding="utf-8")
    must, nice = extract_jd_requirements(jd_text)
    all_reqs = must + nice

    # Compound bullets must survive as single criteria, not comma fragments.
    assert "Solid Python fundamentals (functions, classes, async basics, pip/venv)" in must
    assert "Basic Git workflow (clone, branch, commit, pull request)" in must
    assert "Exposure to FastAPI, Flask, or Django" in nice
    assert any("degree in CS, engineering, or related field" in r for r in must)

    # The shredded fragments from the bad report must NOT appear as requirements.
    for fragment in (
        "classes",
        "async basics",
        "pip/venv)",
        "branch",
        "commit",
        "pull request)",
        "engineering",
        "or related field",
        "Flask",
        "or Django",
    ):
        assert fragment not in all_reqs, f"requirement was shredded into {fragment!r}"


def test_pure_skill_list_still_splits() -> None:
    """Genuine short skill lists without and/or or parens should still split."""
    must, _ = extract_jd_requirements("Role\nRequirements:\n- Must have: Python, FastAPI, SQL\n")
    assert "Python" in must
    assert "FastAPI" in must
    assert "SQL" in must
