"""SSRF validation before outbound URL fetches."""

from __future__ import annotations

import ipaddress
import re
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

MAX_URL_LENGTH = 2048
DNS_CACHE_TTL_SECONDS = 60

# IPv4 literal or bracketed IPv6 in URL host
_IPV4_HOST = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_IPV6_HOST = re.compile(r"^\[([0-9a-fA-F:]+)\]$")

_dns_cache: dict[str, tuple[float, list[str]]] = {}


@dataclass(frozen=True)
class UrlValidationResult:
    allowed: bool
    reason: str | None = None
    hostname: str | None = None
    resolved_ips: tuple[str, ...] = ()


def _cache_get(hostname: str) -> list[str] | None:
    entry = _dns_cache.get(hostname)
    if not entry:
        return None
    expires_at, ips = entry
    if time.monotonic() > expires_at:
        _dns_cache.pop(hostname, None)
        return None
    return ips


def _cache_set(hostname: str, ips: list[str]) -> None:
    _dns_cache[hostname] = (time.monotonic() + DNS_CACHE_TTL_SECONDS, ips)


def clear_dns_cache() -> None:
    """Clear in-memory DNS cache (for tests)."""
    _dns_cache.clear()


def _parse_hostname(url: str) -> str | None:
    parsed = urlparse(url)
    return parsed.hostname


def _hostname_is_literal_ip(hostname: str) -> bool:
    if _IPV4_HOST.match(hostname):
        return True
    if _IPV6_HOST.match(hostname):
        return True
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _ip_is_blocked(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return bool(
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast
    )


def _resolve_hostname(hostname: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(
            hostname,
            None,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise OSError(f"DNS resolution failed: {exc}") from exc

    ips: list[str] = []
    for info in infos:
        sockaddr = info[4]
        if sockaddr:
            ips.append(sockaddr[0])

    return list(dict.fromkeys(ips))


def _resolve_with_cache(
    hostname: str,
    resolver: Callable[[str], list[str]] | None,
) -> list[str]:
    cached = _cache_get(hostname)
    if cached is not None:
        return cached

    if resolver is not None:
        ips = resolver(hostname)
    else:
        ips = _resolve_hostname(hostname)

    _cache_set(hostname, ips)
    return ips


def validate_url(
    url: str,
    *,
    resolver: Callable[[str], list[str]] | None = None,
) -> UrlValidationResult:
    """
    Validate a URL is safe to fetch.

    Rules: HTTPS only, max length 2048, no literal IP hostnames,
    resolved addresses must not be private/reserved/link-local.
    """
    if not url or not url.strip():
        return UrlValidationResult(allowed=False, reason="empty_url")

    if len(url) > MAX_URL_LENGTH:
        return UrlValidationResult(allowed=False, reason="url_too_long")

    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        return UrlValidationResult(allowed=False, reason="https_required")

    hostname = parsed.hostname
    if not hostname:
        return UrlValidationResult(allowed=False, reason="missing_hostname")

    hostname = hostname.lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return UrlValidationResult(allowed=False, reason="localhost_blocked", hostname=hostname)

    if _hostname_is_literal_ip(hostname):
        return UrlValidationResult(allowed=False, reason="ip_hostname_blocked", hostname=hostname)

    try:
        resolved_ips = _resolve_with_cache(hostname, resolver)
    except OSError:
        return UrlValidationResult(
            allowed=False,
            reason="dns_resolution_failed",
            hostname=hostname,
        )

    if not resolved_ips:
        return UrlValidationResult(
            allowed=False,
            reason="dns_resolution_empty",
            hostname=hostname,
        )

    for ip_str in resolved_ips:
        if _ip_is_blocked(ip_str):
            return UrlValidationResult(
                allowed=False,
                reason="private_or_reserved_ip",
                hostname=hostname,
                resolved_ips=tuple(resolved_ips),
            )

    return UrlValidationResult(
        allowed=True,
        hostname=hostname,
        resolved_ips=tuple(resolved_ips),
    )
