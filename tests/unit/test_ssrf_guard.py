import pytest

from agent.security.ssrf_guard import (
    MAX_URL_LENGTH,
    clear_dns_cache,
    validate_url,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_dns_cache()
    yield
    clear_dns_cache()


def _public_resolver(_hostname: str) -> list[str]:
    return ["93.184.216.34"]


def _private_resolver(_hostname: str) -> list[str]:
    return ["192.168.1.1"]


def _loopback_resolver(_hostname: str) -> list[str]:
    return ["127.0.0.1"]


def test_https_public_host_allowed() -> None:
    result = validate_url(
        "https://github.com/example-user",
        resolver=_public_resolver,
    )
    assert result.allowed is True
    assert result.hostname == "github.com"


def test_http_scheme_rejected() -> None:
    result = validate_url("http://github.com/user", resolver=_public_resolver)
    assert result.allowed is False
    assert result.reason == "https_required"


def test_literal_ipv4_hostname_rejected() -> None:
    result = validate_url("https://192.168.1.1/path", resolver=_public_resolver)
    assert result.allowed is False
    assert result.reason == "ip_hostname_blocked"


def test_localhost_rejected_without_dns() -> None:
    result = validate_url("https://localhost/admin")
    assert result.allowed is False
    assert result.reason == "localhost_blocked"


def test_private_resolved_ip_rejected() -> None:
    result = validate_url("https://evil.example.com/", resolver=_private_resolver)
    assert result.allowed is False
    assert result.reason == "private_or_reserved_ip"


def test_loopback_resolved_ip_rejected() -> None:
    result = validate_url("https://internal.example.com/", resolver=_loopback_resolver)
    assert result.allowed is False
    assert result.reason == "private_or_reserved_ip"


def test_url_too_long_rejected() -> None:
    long_url = "https://github.com/" + ("a" * MAX_URL_LENGTH)
    result = validate_url(long_url, resolver=_public_resolver)
    assert result.allowed is False
    assert result.reason == "url_too_long"


def test_dns_cache_reuses_resolver() -> None:
    calls: list[str] = []

    def counting_resolver(hostname: str) -> list[str]:
        calls.append(hostname)
        return ["93.184.216.34"]

    url = "https://example.com/page"
    assert validate_url(url, resolver=counting_resolver).allowed is True
    assert validate_url(url, resolver=counting_resolver).allowed is True
    assert calls == ["example.com"]
