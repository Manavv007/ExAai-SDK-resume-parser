"""Shared helpers for integration and security tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from agent.tools.validator import validate_result

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
DOMAINS = FIXTURES / "domains"

APP_ID = "11111111-1111-4111-8111-111111111111"
JOB_ID = "22222222-2222-4222-8222-222222222222"


@dataclass(frozen=True)
class DomainCase:
    key: str
    expected_domain: str
    pii_markers: tuple[str, ...]
    crawl_url_substring: str
    domain_category: str


DOMAIN_CASES: tuple[DomainCase, ...] = (
    DomainCase(
        key="software",
        expected_domain="technical",
        pii_markers=("alex.chen@example.com", "Alex Chen", "+1 (415) 555-0198"),
        crawl_url_substring="github.com",
        domain_category="code",
    ),
    DomainCase(
        key="design",
        expected_domain="design",
        pii_markers=("morgan.lee@creative.example.com", "Morgan Lee", "+1 (212) 555-0142"),
        crawl_url_substring="behance.net",
        domain_category="portfolio",
    ),
    DomainCase(
        key="academic",
        expected_domain="academic",
        pii_markers=(
            "samira.patel@university.example.com",
            "Samira Patel",
            "Dr. Samira Patel",
        ),
        crawl_url_substring="scholar.google.com",
        domain_category="academic",
    ),
)


def domain_paths(key: str) -> tuple[Path, Path]:
    base = DOMAINS / key
    return base / "resume.txt", base / "jd.txt"


def load_llm_fixture(
    *,
    requirement: str | None = None,
    requirement_type: str = "technical_skill",
    score: int = 78,
    recommendation: str = "advance",
    rubric: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Minimal valid completed payload for mocked Gemini."""
    base = json.loads((FIXTURES / "valid_result_completed.json").read_text(encoding="utf-8"))
    base["application_id"] = APP_ID
    base["job_id"] = JOB_ID
    base["resume_similarity_score"] = {
        "score": score,
        "reasoning": "Candidate aligns with role requirements.",
    }

    if rubric:
        base["requirement_matches"] = [
            {
                "requirement": item.get("criterion") or item.get("requirement") or "fit",
                "requirement_type": item.get("requirement_type") or requirement_type,
                "match_score": max(min(score + 5, 100), 55),
                "evidence": f"Evidence supports {item.get('criterion', 'fit')}.",
            }
            for item in rubric
        ]
    else:
        req = requirement or "role fit"
        base["requirement_matches"] = [
            {
                "requirement": req,
                "requirement_type": requirement_type,
                "match_score": max(min(score + 5, 100), 55),
                "evidence": f"Resume and external sources support {req}.",
            }
        ]

    base["recommendation"] = recommendation
    return base


def allowlist_ok(category: str) -> MagicMock:
    return MagicMock(allowed=True, reason=None, domain_category=category)


def assert_valid_completed_result(result: dict[str, Any], case: DomainCase) -> None:
    assert result["resume_screening_status"] == "completed"
    assert validate_result(result)
    assert result["recommendation"] in {"advance", "hold", "reject"}
    assert result["application_id"] == APP_ID
    assert result["job_id"] == JOB_ID
    assert len(result["requirement_matches"]) >= 1


def assert_no_pii_in_payload(payload: Any, markers: tuple[str, ...]) -> None:
    text = json.dumps(payload) if not isinstance(payload, str) else payload
    lowered = text.lower()
    for marker in markers:
        assert marker.lower() not in lowered, f"PII leaked: {marker}"
