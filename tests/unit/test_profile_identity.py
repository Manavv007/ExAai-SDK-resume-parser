"""Profile URL identity trust assessments."""

from agent.security.profile_identity import (
    ProfileTrust,
    _is_personal_identity_profile_url,
    _slug_matches_identity,
    assess_profile_links,
    build_identity_red_flags,
    extract_identity_bundle,
    should_cap_score_for_identity,
)
from agent.tools.link_extractor import ExtractedLink, extract_links


def test_personal_profiles_trusted_when_company_linkedin_also_on_resume() -> None:
    resume = """
    Manav Bhavsar
    manav@gmail.com
    GitHub: https://github.com/Manavv007
    LinkedIn: https://linkedin.com/in/manavbhavsar0908
    University: https://linkedin.com/school/pandit-deendayal-energy-university
    """
    links = extract_links(resume, max_urls=10)
    assessments = assess_profile_links(resume, links)
    gh = [a for a in assessments if "github.com/Manavv007" in a.url][0]
    li = [a for a in assessments if "/in/manavbhavsar0908" in a.url][0]
    assert gh.trust == ProfileTrust.SCORING_TRUSTED
    assert li.trust == ProfileTrust.SCORING_TRUSTED


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


def test_mailto_link_does_not_trigger_identity_score_cap() -> None:
    resume = """
    Manav Bhavsar
    manav@gmail.com
    GitHub: https://github.com/Manavv007
    Email link: mailto:bhavsarmanav14@gmail.com
    """
    links = extract_links(resume, max_urls=10)
    assessments = assess_profile_links(resume, links)
    assert should_cap_score_for_identity(assessments) is False


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


def test_github_repo_url_is_not_personal_identity_profile() -> None:
    assert _is_personal_identity_profile_url("https://github.com/Manavv007") is True
    assert (
        _is_personal_identity_profile_url(
            "https://github.com/Manavv007/S.A.R.A.L.-Scheme-Access-Retrieval-Analysis-Layer-"
        )
        is False
    )


def test_linkedin_non_profile_urls_not_personal_portfolio_crawl() -> None:
    from agent.security.profile_identity import is_personal_portfolio_crawl_url

    assert is_personal_portfolio_crawl_url("https://linkedin.com/in/manavbhavsar0908") is True
    assert is_personal_portfolio_crawl_url("https://www.linkedin.com/school/pdeuofficial") is False
    assert is_personal_portfolio_crawl_url("https://linkedin.com/company/nptel") is False
    assert (
        is_personal_portfolio_crawl_url(
            "https://www.linkedin.com/posts/krish-parmar-developer_ai-llm-activity-7419757286421626880-AS3G"
        )
        is False
    )
    assert is_personal_portfolio_crawl_url("https://behance.net/archidaga") is True
    assert (
        is_personal_portfolio_crawl_url(
            "https://behance.net/joblist?tracking_source=nav20"
        )
        is False
    )
    assert is_personal_portfolio_crawl_url("https://behance.net/?tracking_source=nav20") is False


def test_linkedin_plus_github_repos_no_identity_mismatch_flag() -> None:
    """Project repo URLs must not conflict with a lone LinkedIn profile."""
    resume = """
    Manav Bhavsar
    LinkedIn: https://linkedin.com/in/manavbhavsar0908
    Project: https://github.com/Manavv007/S.A.R.A.L.-Scheme-Access-Retrieval-Analysis-Layer-
    """
    links = extract_links(resume, max_urls=10)
    assessments = assess_profile_links(resume, links)
    li = [a for a in assessments if "/in/manavbhavsar0908" in a.url][0]
    assert li.trust != ProfileTrust.SCORING_UNTRUSTED
    flags = build_identity_red_flags(assessments)
    assert not any(f["flag"] == "profile_identity_mismatch" for f in flags)


def test_github_profile_and_linkedin_plausible_without_presidio_name(monkeypatch) -> None:
    def _no_person(_text: str) -> list[str]:
        return []

    monkeypatch.setattr(
        "agent.security.profile_identity._name_tokens_from_resume",
        _no_person,
    )
    resume = """
    manav@gmail.com
    GitHub: https://github.com/Manavv007
    LinkedIn: https://linkedin.com/in/manavbhavsar0908
    """
    links = extract_links(resume, max_urls=10)
    assessments = assess_profile_links(resume, links)
    assert all(a.trust != ProfileTrust.SCORING_UNTRUSTED for a in assessments)
    assert should_cap_score_for_identity(assessments) is False


def test_conflicting_explicit_profiles_untrusted() -> None:
    resume = """
    Jane Candidate
    https://github.com/alice-dev
    https://linkedin.com/in/totally-different-person
    """
    links = extract_links(resume, max_urls=10)
    assessments = assess_profile_links(resume, links)
    assert all(a.trust == ProfileTrust.SCORING_UNTRUSTED for a in assessments)


def test_behance_explicit_profile_is_exa_fetchable() -> None:
    resume = """
    Jane Designer
    Portfolio: https://www.behance.net/janedesign
    GitHub: https://github.com/totally-other-handle
    """
    links = extract_links(resume, max_urls=10)
    assessments = assess_profile_links(resume, links)
    behance = [a for a in assessments if "behance.net" in a.url][0]
    assert behance.trust == ProfileTrust.SCORING_LIMITED


def test_is_exa_enrichable_profile_url() -> None:
    from agent.security.profile_identity import is_exa_enrichable_profile_url

    assert is_exa_enrichable_profile_url("https://www.behance.net/designer") is True
    assert is_exa_enrichable_profile_url("https://dribbble.com/designer") is True
    assert (
        is_exa_enrichable_profile_url(
            "https://github.com/user/Some-Project-Repo"
        )
        is False
    )


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
