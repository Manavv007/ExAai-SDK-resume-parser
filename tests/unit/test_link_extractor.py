from pathlib import Path

from agent.tools.link_extractor import (
    ExtractedLink,
    canonical_profile_url,
    collapse_profile_urls,
    extract_links,
    extract_urls_from_html,
    extract_urls_from_text,
    is_profile_discovery_url,
    normalize_url,
    profile_url_identity_key,
    resolve_profile_url,
)
from agent.tools.parser import JdStructured, parse_jd_structured

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_normalize_strips_tracking_and_forces_https() -> None:
    url = normalize_url("http://GitHub.com/janedoe?utm_source=x&ref=1")
    assert url == "https://github.com/janedoe?ref=1"


def test_canonical_profile_url_strips_locale_on_personal_profile_paths() -> None:
  for url in (
        "https://www.behance.net/archidaga?locale=cs_CZ",
        "https://dribbble.com/archidaga?lang=en",
        "https://www.artstation.com/archidaga?hl=fr",
        "https://soundcloud.com/archidaga?lang=de",
        "https://jane.dev/?lang=en",
    ):
        canonical = canonical_profile_url(url)
        assert canonical is not None
        assert "?" not in canonical
        assert "locale" not in canonical
        assert "lang" not in canonical


def test_canonical_profile_url_preserves_query_on_non_profile_paths() -> None:
    assert (
        canonical_profile_url("https://github.com/janedoe/awesome-repo?tab=readme")
        == "https://github.com/janedoe/awesome-repo?tab=readme"
    )
    assert (
        canonical_profile_url("https://behance.net/gallery?locale=en_US")
        == "https://behance.net/gallery"
    )


def test_profile_url_identity_key_dedupes_platform_variants() -> None:
    keys = {
        profile_url_identity_key("https://www.behance.net/archidaga"),
        profile_url_identity_key("https://behance.net/archidaga?locale=de_DE"),
        profile_url_identity_key("https://www.linkedin.com/in/archidaga"),
        profile_url_identity_key("https://linkedin.com/in/archidaga"),
    }
    assert None not in keys
    assert len(keys) == 2


def test_collapse_profile_urls_keeps_one_per_identity() -> None:
    urls = collapse_profile_urls(
        [
            "https://www.behance.net/archidaga",
            "https://www.behance.net/archidaga?locale=cs_CZ",
            "https://www.behance.net/archidaga?locale=fr_FR",
            "https://linkedin.com/in/archidaga",
            "https://www.linkedin.com/in/archidaga",
        ]
    )
    assert urls == [
        "https://behance.net/archidaga",
        "https://linkedin.com/in/archidaga",
    ]


def test_normalize_bare_domain() -> None:
    url = normalize_url("github.com/janedoe")
    assert url == "https://github.com/janedoe"


def test_normalize_rejects_mailto_and_tel() -> None:
    assert normalize_url("mailto:candidate@example.com") is None
    assert normalize_url("tel:+15551234567") is None
    assert normalize_url("https://mailto:candidate@example.com") is None


def test_extract_explicit_urls_from_resume() -> None:
    text = (FIXTURES / "sample_resume.txt").read_text(encoding="utf-8")
    links = extract_links(text, max_urls=10)

    urls = {link.url for link in links}
    assert "https://github.com/janedoe" in urls
    assert "https://linkedin.com/in/janedoe" in urls
    github = [link for link in links if "github.com/janedoe" in link.url]
    assert any(link.source == "explicit" for link in github)


def test_pdf_hyperlinks_merged() -> None:
    text = "See my portfolio for details."
    pdf_links = ["https://github.com/from-pdf-annotation"]
    links = extract_links(text, pdf_hyperlinks=pdf_links, max_urls=5)
    assert any("from-pdf-annotation" in link.url for link in links)


def test_infer_technical_platforms_from_handle(monkeypatch) -> None:
    monkeypatch.setenv("INFER_PROFILE_URLS", "true")
    from agent.config import get_settings

    get_settings.cache_clear()
    jd = JdStructured(domain="technical", must_have=["Python"])
    text = "GitHub: @coder123 — active on hackerrank for coding challenges."
    links = extract_links(text, jd=jd, max_urls=10)
    inferred = [link for link in links if link.source == "inferred"]
    assert inferred
    assert any("github.com/coder123" in link.url for link in inferred)
    assert any("hackerrank.com/coder123" in link.url for link in inferred)
    assert not any("kaggle.com" in link.url for link in inferred)


def test_email_domain_not_inferred_as_profile() -> None:
    text = """
    manav@gmail.com
    GitHub: https://github.com/Manavv007
    LinkedIn: linkedin.com/in/manavbhavsar0908
    """
    links = extract_links(text, max_urls=10)
    joined = " ".join(link.url for link in links)
    assert "gitlab.com/gmail" not in joined
    assert "kaggle.com/gmail" not in joined
    assert "hackerrank.com/gmail" not in joined
    assert "https://github.com/manavv007" in joined.lower()


def test_dedupe_urls() -> None:
    text = """
    https://github.com/alice
    https://github.com/alice/
    github.com/alice?utm_campaign=1
    """
    links = extract_links(text, max_urls=10)
    github_links = [link for link in links if "github.com/alice" in link.url]
    assert len(github_links) == 1


def test_max_urls_cap() -> None:
    text = "\n".join(f"https://example{i}.com/page" for i in range(20))
    links = extract_links(text, max_urls=3)
    assert len(links) == 3


def test_jd_domain_drives_inference(monkeypatch) -> None:
    monkeypatch.setenv("INFER_PROFILE_URLS", "true")
    from agent.config import get_settings

    get_settings.cache_clear()
    jd = parse_jd_structured(
        "Graphic Designer\nRequirements:\n- Figma",
        use_llm=False,
    )
    links = extract_links(
        "Portfolio on behance.net — reach me @designstar",
        jd=jd,
        max_urls=8,
    )
    hosts = {link.platform for link in links if link.source == "inferred"}
    assert "behance.net" in hosts or "dribbble.com" in hosts


def test_extracted_link_frozen() -> None:
    link = ExtractedLink(url="https://github.com/x", source="explicit")
    assert link.url == "https://github.com/x"


def test_extract_urls_from_html_hrefs() -> None:
    base = "https://manavbhavsar-portfolio.vercel.app/"
    html = (
        '<a href="https://github.com/Manavv007">GitHub</a>'
        '<a href="/projects">Projects</a>'
        '<script src="script.js"></script>'
        '<img src="dev_avatar.jpg" />'
        '<link href="https://fonts.googleapis.com/css2?family=Outfit" rel="stylesheet">'
    )
    urls = extract_urls_from_html(html, base_url=base, max_urls=20)
    lowered = [u.lower() for u in urls]
    assert "https://github.com/manavv007" in lowered
    assert "https://manavbhavsar-portfolio.vercel.app/projects" in lowered
    assert not any("script.js" in u and "vercel.app" not in u for u in urls)
    assert not any(u.startswith("https://fonts.googleapis.com") for u in urls)


def test_resolve_profile_url_rejects_fake_domains() -> None:
    assert resolve_profile_url("script.js") is None
    assert resolve_profile_url("dev_avatar.jpg") is None
    assert (
        resolve_profile_url("script.js", "https://manavbhavsar-portfolio.vercel.app/")
        is None
    )
    assert is_profile_discovery_url(
        "https://manavbhavsar-portfolio.vercel.app/script.js"
    ) is False
    assert is_profile_discovery_url("https://www.behance.net/manifest.json") is False
    assert is_profile_discovery_url("https://cdn.behance.net/") is False
    assert is_profile_discovery_url("https://mir-s3-cdn-cf.behance.net") is False
    assert is_profile_discovery_url("https://feeds.feedburner.com/behance/vorr") is False


def test_extract_urls_from_text_still_works() -> None:
    text = "See https://github.com/janedoe/portfolio-app and github.com/janedoe"
    urls = extract_urls_from_text(text, max_urls=10)
    joined = " ".join(urls).lower()
    assert "github.com/janedoe" in joined


def test_is_fetchable_personal_profile_url_filters_behance_navigation() -> None:
    from agent.tools.link_extractor import is_fetchable_personal_profile_url

    assert is_fetchable_personal_profile_url("https://behance.net/archidaga") is True
    assert (
        is_fetchable_personal_profile_url(
            "https://behance.net/joblist?tracking_source=nav20"
        )
        is False
    )
    assert is_fetchable_personal_profile_url("https://behance.net/?tracking_source=nav20") is False
    assert is_fetchable_personal_profile_url("https://behance.net/about") is False
    garbage_url = (
        "https://behance.net/archidaga/+(n.target&&n.target.src)"
        ":n}function e(n){return n.replace"
    )
    assert is_fetchable_personal_profile_url(garbage_url) is False


def test_is_profile_discovery_url_rejects_form_handlers_and_wrapped_external_paths() -> None:
    portfolio = "https://manavbhavsar.vercel.app/"
    assert is_profile_discovery_url(portfolio) is True
    assert is_profile_discovery_url("https://formspree.io") is False
    assert is_profile_discovery_url("https://formspree.io/f/xzdqqzdd") is False
    assert (
        is_profile_discovery_url("https://manavbhavsar.vercel.app/github.com/Manavv007")
        is False
    )
    assert (
        is_profile_discovery_url(
            "https://manavbhavsar.vercel.app/linkedin.com/in/manavbhavsar0908"
        )
        is False
    )
    assert (
        is_profile_discovery_url(
            "https://manavbhavsar.vercel.app/ManavResume_20_3%20(2).pdf"
        )
        is False
    )


def test_unwrap_embedded_external_profile_url_rewrites_github_paths() -> None:
    from agent.tools.link_extractor import unwrap_embedded_external_profile_url

    wrapped_repo = (
        "https://manavbhavsar.vercel.app/github.com/Manavv007/"
        "SentinEL-Sentinel-eGeMAPS-openSMILE-"
    )
    assert unwrap_embedded_external_profile_url(wrapped_repo) == (
        "https://github.com/Manavv007/SentinEL-Sentinel-eGeMAPS-openSMILE-"
    )
    assert unwrap_embedded_external_profile_url("https://manavbhavsar.vercel.app/") is None
