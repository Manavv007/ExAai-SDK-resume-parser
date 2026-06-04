import pytest

from agent.security.allowlist import check_allowlist, get_domain_category, is_allowlisted


@pytest.mark.parametrize(
    ("url", "category"),
    [
        ("https://github.com/janedoe", "code"),
        ("https://www.linkedin.com/in/janedoe", "professional"),
        ("https://medium.com/@janedoe", "writing"),
        ("https://behance.net/gallery/123", "portfolio"),
        ("https://scholar.google.com/citations?user=abc", "academic"),
        ("https://user.github.io/portfolio", "code"),
    ],
)
def test_allowlisted_domains(url: str, category: str) -> None:
    result = check_allowlist(url)
    assert result.allowed is True
    assert result.domain_category == category
    assert is_allowlisted(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://evil-site.com/profile",
        "https://pastebin.com/raw/abc",
        "https://google.com/search?q=test",
    ],
)
def test_blocked_domains(url: str) -> None:
    result = check_allowlist(url)
    assert result.allowed is False
    assert result.reason == "domain_not_allowlisted"
    assert is_allowlisted(url) is False


def test_get_domain_category_subdomain() -> None:
    assert get_domain_category("gist.github.com") == "code"


def test_missing_hostname() -> None:
    result = check_allowlist("not-a-valid-url")
    assert result.allowed is False
    assert result.reason == "missing_hostname"
