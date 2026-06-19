"""Tests for LLM output sanitization before schema validation."""

from __future__ import annotations

import json
from pathlib import Path

from agent.tools.result_sanitizer import (
    compact_metadata,
    quantize_score,
    resolve_overall_score,
    sanitize_requirement_matches,
)
from agent.tools.scorer import normalize_screening_result
from agent.tools.validator import validate_result

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_normalize_messy_llm_payload_passes_validation(test_settings) -> None:
    """Common Gemini quirks: floats, null metadata, empty evidence, bad enums."""
    fixture = json.loads((FIXTURES / "valid_result_completed.json").read_text(encoding="utf-8"))
    messy = {
        "resume_similarity_score": {"score": 78.6, "reasoning": "  Good fit.  "},
        "requirement_matches": [
            {
                "requirement": "Python",
                "requirement_type": "technical",
                "match_score": "82.4",
                "evidence": "",
                "source_quote": None,
                "extra_field": "drop-me",
            }
        ],
        "recommendation": "yes",
        "recommendation_reasoning": "",
        "red_flags": [
            {"flag": "gap", "severity": "severe", "evidence": "Two-year gap detected."},
            {"flag": "", "severity": "low", "evidence": "skip"},
        ],
        "sources_crawled": [
            {
                "url": "github.com/example-candidate",
                "title": None,
                "relevance": "strong",
            }
        ],
        "metadata": {
            "processing_time_ms": None,
            "llm_calls": None,
            "agent_submit_fallback": None,
        },
    }

    normalized = normalize_screening_result(
        messy,
        application_id=fixture["application_id"],
        job_id=fixture["job_id"],
        resume_text="resume text",
        rubric=[
            {
                "criterion": "Python",
                "weight": "must_have",
                "requirement_type": "technical_skill",
            }
        ],
        enriched_contents=[],
        processing_time_ms=None,
    )

    assert validate_result(normalized) is True
    assert normalized["resume_similarity_score"]["score"] == 80
    assert normalized["requirement_matches"][0]["match_score"] == 80
    assert normalized["requirement_matches"][0]["evidence"]
    assert normalized["recommendation"] == "advance"
    assert normalized["recommendation_reasoning"]
    assert "red_flags" not in normalized
    assert normalized["sources_crawled"][0]["url"].startswith("https://")
    assert "title" not in normalized["sources_crawled"][0]
    assert "processing_time_ms" not in normalized["metadata"]
    assert "llm_calls" not in normalized["metadata"]


def test_sanitize_replaces_placeholder_requirement_with_rubric_criterion() -> None:
    rubric = [
        {
            "criterion": "Python programming",
            "weight": "must_have",
            "requirement_type": "technical_skill",
        },
        {
            "criterion": "Git experience",
            "weight": "nice_to_have",
            "requirement_type": "technical_skill",
        },
    ]
    sanitized = sanitize_requirement_matches(
        [
            {
                "requirement": "requirement",
                "requirement_type": "technical_skill",
                "match_score": 90,
                "evidence": "",
            },
            {
                "requirement": "requirement",
                "requirement_type": "technical_skill",
                "match_score": 80,
                "evidence": "Listed GitHub projects.",
            },
        ],
        rubric,
    )

    assert sanitized[0]["requirement"] == "Python programming"
    assert sanitized[1]["requirement"] == "Git experience"


def test_quantize_score_snaps_to_step() -> None:
    assert quantize_score(72, step=5) == 70
    assert quantize_score(73, step=5) == 75
    assert quantize_score(82, step=5) == 80


def test_resolve_overall_score_prefers_rubric_when_complete() -> None:
    rubric = [{"criterion": "Python", "weight": "must_have"}]
    matches = [{"requirement": "Python", "match_score": 70, "evidence": "x"}]
    assert (
        resolve_overall_score(
            llm_score=85,
            derived_score=70,
            rubric=rubric,
            requirement_matches=matches,
            rubric_derived=True,
        )
        == 70
    )
    assert (
        resolve_overall_score(
            llm_score=85,
            derived_score=70,
            rubric=rubric,
            requirement_matches=matches,
            rubric_derived=False,
        )
        == 85
    )


def test_sanitize_sources_crawled_falls_back_to_enriched_contents() -> None:
    from agent.tools.result_sanitizer import sanitize_sources_crawled

    sources = sanitize_sources_crawled(
        [],
        enriched_fallback=[
            {
                "url": "https://github.com/candidate",
                "domain_category": "code",
                "ok": True,
            },
            {
                "url": "https://linkedin.com/in/candidate",
                "domain_category": "professional",
                "ok": False,
                "fetch_error": "exa_fetch_failed",
            },
        ],
    )

    assert len(sources) == 2
    assert sources[0]["url"] == "https://github.com/candidate"
    assert sources[0]["relevance"] == "high"
    assert sources[0]["title"] == "code"
    assert sources[1]["relevance"] == "low"


def test_sanitize_sources_crawled_falls_back_to_profile_urls() -> None:
    from agent.tools.result_sanitizer import sanitize_sources_crawled

    sources = sanitize_sources_crawled(
        [],
        enriched_fallback=[],
        profile_urls_fallback=["https://github.com/candidate"],
        profile_url_meta=[
            {"url": "https://github.com/candidate", "platform": "github"},
        ],
    )

    assert len(sources) == 1
    assert sources[0]["url"] == "https://github.com/candidate"
    assert sources[0]["title"] == "github"


def test_sanitize_sources_crawled_excludes_unfetched_discovered_junk() -> None:
    from agent.tools.result_sanitizer import sanitize_sources_crawled

    portfolio = "https://manavbhavsar.vercel.app/"
    sources = sanitize_sources_crawled(
        [
            {
                "url": "https://formspree.io/f/xzdqqzdd",
                "relevance": "medium",
                "title": "discovered",
            }
        ],
        enriched_fallback=[
            {"url": portfolio, "domain_category": "portfolio", "ok": True},
            {"url": "https://github.com/Manavv007", "domain_category": "code", "ok": True},
            {
                "url": "https://linkedin.com/in/manavbhavsar0908",
                "domain_category": "professional",
                "ok": True,
            },
        ],
        profile_urls_fallback=[
            portfolio,
            "https://formspree.io",
            "https://manavbhavsar.vercel.app/github.com/Manavv007",
        ],
        resume_profile_urls=[portfolio],
    )

    urls = {item["url"] for item in sources}
    assert "https://formspree.io/f/xzdqqzdd" not in urls
    assert "https://formspree.io" not in urls
    assert "https://manavbhavsar.vercel.app/github.com/Manavv007" not in urls
    assert portfolio in urls
    assert "https://github.com/Manavv007" in urls
    assert "https://linkedin.com/in/manavbhavsar0908" in urls


def test_sanitize_sources_crawled_excludes_discovered_linkedin_noise() -> None:
    from agent.tools.result_sanitizer import sanitize_sources_crawled

    sources = sanitize_sources_crawled(
        [],
        enriched_fallback=[
            {
                "url": "https://github.com/Manavv007",
                "domain_category": "code",
                "ok": True,
            },
            {
                "url": "https://linkedin.com/in/manavbhavsar0908",
                "domain_category": "professional",
                "ok": True,
            },
            {
                "url": "https://www.linkedin.com/school/pdeuofficial",
                "domain_category": "discovered",
                "ok": True,
            },
            {
                "url": "https://linkedin.com/company/nptel",
                "domain_category": "discovered",
                "ok": True,
            },
            {
                "url": "https://www.linkedin.com/posts/foo-activity-123",
                "domain_category": "discovered",
                "ok": True,
            },
        ],
        profile_urls_fallback=[
            "https://github.com/Manavv007",
            "https://linkedin.com/in/manavbhavsar0908",
        ],
    )

    urls = {item["url"] for item in sources}
    assert "https://github.com/Manavv007" in urls
    assert "https://linkedin.com/in/manavbhavsar0908" in urls
    assert not any("/school/" in url for url in urls)
    assert not any("/company/" in url for url in urls)
    assert not any("/posts/" in url for url in urls)


def test_sanitize_sources_crawled_collapses_behance_locale_variants() -> None:
    from agent.tools.result_sanitizer import sanitize_sources_crawled

    sources = sanitize_sources_crawled(
        [],
        enriched_fallback=[
            {
                "url": "https://www.behance.net/archidaga",
                "domain_category": "portfolio",
                "ok": True,
            },
            {
                "url": "https://www.behance.net/archidaga?locale=cs_CZ",
                "domain_category": "discovered",
                "ok": True,
            },
            {
                "url": "https://www.behance.net/archidaga?locale=fr_FR",
                "domain_category": "discovered",
                "ok": True,
            },
            {
                "url": "https://linkedin.com/in/archidaga",
                "domain_category": "professional",
                "ok": True,
            },
            {
                "url": "https://www.linkedin.com/in/archidaga",
                "domain_category": "professional",
                "ok": True,
            },
        ],
        profile_urls_fallback=[
            "https://www.behance.net/archidaga",
            "https://linkedin.com/in/archidaga",
        ],
    )

    assert len(sources) == 2
    urls = [item["url"] for item in sources]
    assert "https://behance.net/archidaga" in urls
    assert "https://linkedin.com/in/archidaga" in urls
    assert not any("locale=" in url for url in urls)


def test_discover_links_from_linkedin_profile_skips_school_and_posts() -> None:
    from agent.enrichment import _discover_links_from_entries

    linkedin_html = """
    Education: https://www.linkedin.com/school/pdeuofficial
    Company: https://linkedin.com/company/nptel
    Post: https://www.linkedin.com/posts/krish-parmar-developer_ai-llm-activity-7419757286421626880-AS3G
    Portfolio: https://manavbhavsar.dev/projects
    """
    entries = [
        {
            "url": "https://linkedin.com/in/manavbhavsar0908",
            "ok": True,
            "content": linkedin_html,
        }
    ]
    _, discovered_non_github = _discover_links_from_entries(entries)
    joined = " ".join(discovered_non_github).lower()
    assert "school" not in joined
    assert "company" not in joined
    assert "/posts/" not in joined


def test_compact_metadata_strips_null_optional_fields() -> None:
    assert compact_metadata(
        {
            "schema_version": "1.0",
            "model_version": "x",
            "processed_at": "2026-01-01T00:00:00Z",
            "resume_text_chars": 10,
            "agent_version": "0.1.0",
            "processing_time_ms": None,
            "llm_calls": None,
            "agent_submit_fallback": None,
        }
    ) == {
        "schema_version": "1.0",
        "model_version": "x",
        "processed_at": "2026-01-01T00:00:00Z",
        "resume_text_chars": 10,
        "agent_version": "0.1.0",
    }
