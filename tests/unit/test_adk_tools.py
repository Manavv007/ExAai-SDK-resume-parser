import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.adk_tools import (
    fetch_profile_content,
    fetch_profiles,
    list_candidate_profile_urls,
    submit_screening_result,
)
from agent.enrichment import plan_batch_profile_fetches
from agent.submit import process_screening_submission
from agent.tools.validator import validate_result

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
APP_ID = "11111111-1111-4111-8111-111111111111"
JOB_ID = "22222222-2222-4222-8222-222222222222"


class _FakeState(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def test_list_candidate_profile_urls_includes_trust() -> None:
    ctx = MagicMock()
    ctx.state = _FakeState(
        profile_urls=["https://github.com/janedoe"],
        profile_url_meta=[{"url": "https://github.com/janedoe", "source": "explicit"}],
        profile_trust_by_url={"https://github.com/janedoe": "scoring_trusted"},
    )
    result = list_candidate_profile_urls(ctx)
    assert result["count"] == 1
    assert "github.com" in result["urls"][0]
    assert result["trust_by_url"]["https://github.com/janedoe"] == "scoring_trusted"


@patch("agent.enrichment.fetch_url_text", return_value="Python projects and OSS contributions.")
def test_fetch_profile_content_success(mock_fetch, test_settings) -> None:
    url = "https://github.com/janedoe"
    ctx = MagicMock()
    ctx.state = _FakeState(profile_urls=[url], enriched_contents=[])

    mock_cache = MagicMock()
    mock_cache.get.return_value = None

    with patch("agent.enrichment.get_url_cache", return_value=mock_cache):
        with patch(
            "agent.enrichment.validate_url",
            return_value=MagicMock(allowed=True, reason=None),
        ):
            with patch(
                "agent.enrichment.check_allowlist",
                return_value=MagicMock(allowed=True, reason=None, domain_category="code"),
            ):
                result = fetch_profile_content(url, ctx)

    assert result["ok"] is True
    assert len(ctx.state["enriched_contents"]) == 1
    mock_fetch.assert_called_once_with(url)


def test_fetch_profile_content_rejects_unknown_url(test_settings) -> None:
    ctx = MagicMock()
    ctx.state = _FakeState(profile_urls=["https://github.com/janedoe"], enriched_contents=[])

    result = fetch_profile_content("https://evil.example.com/x", ctx)
    assert result["ok"] is False
    assert result["error"] == "url_not_in_candidate_list"


@patch(
    "agent.enrichment.fetch_url_text_batch",
    return_value={
        "https://github.com/janedoe": "Portfolio and design systems.",
        "https://www.behance.net/janedoe": "Portfolio and design systems.",
    },
)
def test_fetch_profiles_parallel_success(mock_fetch_batch, test_settings) -> None:
    urls = [
        "https://github.com/janedoe",
        "https://www.behance.net/janedoe",
    ]
    ctx = MagicMock()
    ctx.state = _FakeState(
        profile_urls=urls,
        profile_trust_by_url={u: "scoring_trusted" for u in urls},
        enriched_contents=[],
    )

    mock_cache = MagicMock()
    mock_cache.get.return_value = None

    with patch("agent.enrichment.get_url_cache", return_value=mock_cache):
        with patch(
            "agent.enrichment.validate_url",
            return_value=MagicMock(allowed=True, reason=None),
        ):
            with patch(
                "agent.enrichment.check_allowlist",
                return_value=MagicMock(allowed=True, reason=None, domain_category="code"),
            ):
                result = asyncio.run(fetch_profiles(urls, ctx))

    assert result["ok"] is True
    assert result["fetched"] == 2
    assert len(result["results"]) == 2
    assert len(ctx.state["enriched_contents"]) == 2
    mock_fetch_batch.assert_called_once()


@patch(
    "agent.enrichment.fetch_url_text_batch",
    return_value={"https://github.com/janedoe": "Trusted GitHub profile."},
)
def test_fetch_profiles_skips_untrusted(mock_fetch_batch, test_settings) -> None:
    trusted = "https://github.com/janedoe"
    untrusted = "https://github.com/torvalds"
    ctx = MagicMock()
    ctx.state = _FakeState(
        profile_urls=[trusted, untrusted],
        profile_trust_by_url={
            trusted: "scoring_trusted",
            untrusted: "scoring_untrusted",
        },
        enriched_contents=[],
    )

    mock_cache = MagicMock()
    mock_cache.get.return_value = None

    with patch("agent.enrichment.get_url_cache", return_value=mock_cache):
        with patch(
            "agent.enrichment.validate_url",
            return_value=MagicMock(allowed=True, reason=None),
        ):
            with patch(
                "agent.enrichment.check_allowlist",
                return_value=MagicMock(allowed=True, reason=None, domain_category="code"),
            ):
                result = asyncio.run(fetch_profiles([trusted, untrusted], ctx))

    assert result["ok"] is True
    assert result["fetched"] == 1
    assert len(result["skipped"]) == 1
    assert result["skipped"][0]["error"] == "profile_untrusted"
    mock_fetch_batch.assert_called_once()


def test_fetch_profiles_respects_session_budget(monkeypatch, test_settings) -> None:
    monkeypatch.setenv("MAX_URLS_PER_RESUME", "2")
    from agent.config import get_settings

    get_settings.cache_clear()

    urls = [f"https://github.com/user{i}" for i in range(4)]
    ctx = MagicMock()
    ctx.state = _FakeState(
        profile_urls=urls,
        profile_trust_by_url={u: "scoring_trusted" for u in urls},
        enriched_contents=[],
    )

    eligible, skipped, truncated = plan_batch_profile_fetches(ctx.state, urls)
    assert len(eligible) == 2
    assert truncated == 2
    assert skipped == []


def test_fetch_profiles_skips_already_enriched_urls(test_settings) -> None:
    trusted = "https://github.com/janedoe"
    another = "https://www.behance.net/janedoe"
    ctx = MagicMock()
    ctx.state = _FakeState(
        profile_urls=[trusted, another],
        profile_trust_by_url={
            trusted: "scoring_trusted",
            another: "scoring_trusted",
        },
        enriched_contents=[
            {
                "url": trusted,
                "content": "cached profile",
                "domain_category": "code",
                "profile_trust": "scoring_trusted",
                "ok": True,
            }
        ],
    )

    eligible, skipped, truncated = plan_batch_profile_fetches(ctx.state, [trusted, another])

    assert eligible == [another]
    assert truncated == 0
    assert len(skipped) == 1
    assert skipped[0]["error"] == "already_fetched"


@patch("agent.enrichment.fetch_url_text", return_value="content")
def test_fetch_profile_content_skips_untrusted_without_exa(mock_fetch, test_settings) -> None:
    url = "https://github.com/torvalds"
    ctx = MagicMock()
    ctx.state = _FakeState(
        profile_urls=[url],
        profile_trust_by_url={url: "scoring_untrusted"},
        enriched_contents=[],
    )

    result = fetch_profile_content(url, ctx)

    assert result["ok"] is False
    assert result["error"] == "profile_untrusted"
    mock_fetch.assert_not_called()
    assert ctx.state["enriched_contents"] == []


@patch(
    "agent.enrichment.fetch_url_text_batch",
    return_value={"https://github.com/janedoe": "Linus Torvalds kernel work."},
)
def test_enrich_profile_urls_skips_exa_for_untrusted(mock_fetch_batch, test_settings) -> None:
    from agent.enrichment import enrich_profile_urls

    trusted = "https://github.com/janedoe"
    untrusted = "https://github.com/torvalds"
    state = _FakeState(
        profile_urls=[trusted, untrusted],
        profile_trust_by_url={
            trusted: "scoring_trusted",
            untrusted: "scoring_untrusted",
        },
        enriched_contents=[],
    )

    mock_cache = MagicMock()
    mock_cache.get.return_value = None

    with patch("agent.enrichment.get_url_cache", return_value=mock_cache):
        with patch(
            "agent.enrichment.validate_url",
            return_value=MagicMock(allowed=True, reason=None),
        ):
            with patch(
                "agent.enrichment.check_allowlist",
                return_value=MagicMock(allowed=True, reason=None, domain_category="code"),
            ):
                enrich_profile_urls(state)

    mock_fetch_batch.assert_called_once()
    call_urls = mock_fetch_batch.call_args[0][0]
    assert trusted in call_urls
    assert untrusted not in call_urls
    assert len(state["enriched_contents"]) == 2
    stub = next(item for item in state["enriched_contents"] if item["url"] == untrusted)
    assert stub["profile_trust"] == "scoring_untrusted"
    assert stub.get("skipped_fetch") is True
    assert stub["content"] == ""


def _base_session_state(**extras) -> _FakeState:
    state = _FakeState(
        application_id=APP_ID,
        job_id=JOB_ID,
        resume_text="Python engineer with six years of backend experience.",
        jd_structured={
            "domain": "technical",
            "must_have": ["Python"],
            "nice_to_have": [],
            "requirements": [
                {
                    "text": "Python",
                    "weight": "must_have",
                    "requirement_type": "technical_skill",
                }
            ],
        },
        enriched_contents=[
            {
                "url": "https://github.com/example-candidate",
                "content": "Open-source Python projects.",
                "domain_category": "code",
            }
        ],
        processing_time_ms=8420,
    )
    state.update(extras)
    return state


def _llm_scoring_payload(**overrides) -> dict:
    payload = {
        "resume_similarity_score": {
            "score": 78,
            "reasoning": "Strong Python and distributed systems experience.",
        },
        "requirement_matches": [
            {
                "requirement": "Python",
                "requirement_type": "technical_skill",
                "match_score": 85,
                "evidence": "Resume lists Python in three roles over six years.",
            }
        ],
        "recommendation": "advance",
        "recommendation_reasoning": "Meets must-have technical skills.",
        "red_flags": [],
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_submit_screening_result_accepts_valid_fixture() -> None:
    fixture = json.loads((FIXTURES / "valid_result_completed.json").read_text(encoding="utf-8"))
    ctx = MagicMock()
    ctx.state = _base_session_state()

    with patch("agent.adk_tools.await_sandbox_for_scoring", new_callable=AsyncMock):
        result = await submit_screening_result(fixture, ctx)

    assert result["ok"] is True
    assert "screening_result" in ctx.state
    stored = ctx.state["screening_result"]
    assert stored["application_id"] == APP_ID
    assert stored["job_id"] == JOB_ID
    assert validate_result(stored)


@pytest.mark.asyncio
async def test_submit_screening_result_accepts_llm_shape() -> None:
    ctx = MagicMock()
    ctx.state = _base_session_state()

    with patch("agent.adk_tools.await_sandbox_for_scoring", new_callable=AsyncMock):
        result = await submit_screening_result(_llm_scoring_payload(), ctx)

    assert result["ok"] is True
    stored = ctx.state["screening_result"]
    assert stored["resume_screening_status"] == "completed"
    assert stored["metadata"]["resume_text_chars"] == len(ctx.state["resume_text"])
    assert validate_result(stored)


@pytest.mark.asyncio
async def test_submit_screening_result_rejects_bad_session_uuids() -> None:
    ctx = MagicMock()
    ctx.state = _base_session_state(application_id="not-a-uuid", job_id=JOB_ID)

    with patch("agent.adk_tools.await_sandbox_for_scoring", new_callable=AsyncMock):
        result = await submit_screening_result(_llm_scoring_payload(), ctx)

    assert result["ok"] is False
    assert any("uuid" in err.lower() for err in result["errors"])
    assert "screening_result" not in ctx.state


@pytest.mark.asyncio
async def test_submit_screening_result_sanitizes_invalid_requirement_type() -> None:
    ctx = MagicMock()
    ctx.state = _base_session_state()
    bad = _llm_scoring_payload(
        requirement_matches=[
            {
                "requirement": "Python",
                "requirement_type": "bogus_type",
                "match_score": 85,
                "evidence": "Resume lists Python.",
            }
        ]
    )

    with patch("agent.adk_tools.await_sandbox_for_scoring", new_callable=AsyncMock):
        result = await submit_screening_result(bad, ctx)

    assert result["ok"] is True
    stored = ctx.state["screening_result"]
    assert stored["requirement_matches"][0]["requirement_type"] == "technical_skill"


def test_submit_screening_result_applies_must_have_cap() -> None:
    state = _base_session_state()
    raw = _llm_scoring_payload(
        resume_similarity_score={"score": 90, "reasoning": "Excellent overall."},
        requirement_matches=[
            {
                "requirement": "Python",
                "requirement_type": "technical_skill",
                "match_score": 30,
                "evidence": "Limited Python evidence.",
            }
        ],
    )

    outcome = process_screening_submission(state, raw)

    assert outcome["ok"] is True
    assert outcome["screening_result"]["resume_similarity_score"]["score"] == 30


def test_submit_screening_result_applies_identity_cap() -> None:
    state = _base_session_state(profile_identity_cap_score=True)
    raw = _llm_scoring_payload(
        resume_similarity_score={"score": 90, "reasoning": "Strong candidate."},
    )

    outcome = process_screening_submission(state, raw)

    assert outcome["ok"] is True
    # Resume rubric evidence (85) outweighs profile identity cap when must-haves pass.
    assert outcome["screening_result"]["resume_similarity_score"]["score"] == 85


def test_submit_screening_result_identity_cap_when_rubric_weak() -> None:
    state = _base_session_state(profile_identity_cap_score=True)
    raw = _llm_scoring_payload(
        resume_similarity_score={"score": 90, "reasoning": "Strong candidate."},
        requirement_matches=[
            {
                "requirement": "Python",
                "requirement_type": "technical_skill",
                "match_score": 25,
                "evidence": "Little Python evidence on resume.",
            }
        ],
    )

    outcome = process_screening_submission(state, raw)

    assert outcome["ok"] is True
    assert outcome["screening_result"]["resume_similarity_score"]["score"] == 25


def test_submit_screening_result_includes_temp_sandbox_reports() -> None:
    sandbox_reports = [
        {
            "repo": "testuser/repo1",
            "url": "https://github.com/testuser/repo1",
            "provider": "cloud_run",
            "clone_ok": True,
            "summary": "ok",
        }
    ]
    state = _base_session_state(
        github_repo_analyses={
            "username": "testuser",
            "sandbox_reports": sandbox_reports,
        }
    )
    outcome = process_screening_submission(state, _llm_scoring_payload())

    assert outcome["ok"] is True
    assert outcome["screening_result"]["temp_sandbox_reports"] == sandbox_reports
