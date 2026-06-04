from pathlib import Path

from agent.tools.link_extractor import (
    ExtractedLink,
    extract_links,
    normalize_url,
)
from agent.tools.parser import JdStructured, parse_jd_structured

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_normalize_strips_tracking_and_forces_https() -> None:
    url = normalize_url("http://GitHub.com/janedoe?utm_source=x&ref=1")
    assert url == "https://github.com/janedoe?ref=1"


def test_normalize_bare_domain() -> None:
    url = normalize_url("github.com/janedoe")
    assert url == "https://github.com/janedoe"


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
