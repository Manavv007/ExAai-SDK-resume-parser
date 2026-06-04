"""Profile URL identity trust assessments."""

from agent.security.profile_identity import (
    ProfileTrust,
    _slug_matches_identity,
    assess_profile_links,
    extract_identity_bundle,
    should_cap_score_for_identity,
)
from agent.tools.link_extractor import ExtractedLink, extract_links


def test_manav_name_matches_manavv007_github() -> None:
    resume = """
    Manav Bhavsar
    manav@gmail.com
    GitHub: https://github.com/Manavv007
    LinkedIn: https://linkedin.com/in/manavbhavsar0908
    """
    bundle = extract_identity_bundle(resume)
    assert _slug_matches_identity(["manavv007"], bundle)
    links = extract_links(resume, max_urls=10)
    assessments = assess_profile_links(resume, links)
    gh = [a for a in assessments if "github.com" in a.url][0]
    li = [a for a in assessments if "linkedin.com" in a.url][0]
    assert gh.trust == ProfileTrust.SCORING_TRUSTED
    assert li.trust == ProfileTrust.SCORING_TRUSTED


def test_someone_elses_github_untrusted() -> None:
    resume = """
    Manav Bhavsar
    manav@gmail.com
    Portfolio: https://github.com/torvalds
    """
    links = extract_links(resume, max_urls=5)
    assessments = assess_profile_links(resume, links)
    gh = [a for a in assessments if "torvalds" in a.url][0]
    assert gh.trust == ProfileTrust.SCORING_UNTRUSTED
    assert should_cap_score_for_identity(assessments)


def test_conflicting_explicit_profiles_untrusted() -> None:
    resume = """
    Jane Candidate
    https://github.com/alice-dev
    https://linkedin.com/in/totally-different-person
    """
    links = extract_links(resume, max_urls=10)
    assessments = assess_profile_links(resume, links)
    assert all(a.trust == ProfileTrust.SCORING_UNTRUSTED for a in assessments)


def test_inferred_url_without_match_untrusted() -> None:
    resume = "Contact manav@gmail.com only."
    links = [
        ExtractedLink(
            url="https://github.com/randomother",
            source="inferred",
            platform="github.com",
        )
    ]
    assessments = assess_profile_links(resume, links)
    assert assessments[0].trust == ProfileTrust.SCORING_UNTRUSTED


def test_infer_profile_urls_disabled_by_default(monkeypatch) -> None:
    monkeypatch.setenv("INFER_PROFILE_URLS", "false")
    from agent.config import get_settings

    get_settings.cache_clear()
    text = "GitHub: @coder123 — hackerrank contests."
    links = extract_links(text, max_urls=10)
    assert not any(link.source == "inferred" for link in links)
