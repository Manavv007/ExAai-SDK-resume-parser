"""Crawled content sanitization and injection stripping."""

from __future__ import annotations

import re

_INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions"),
    re.compile(r"(?i)disregard\s+(all\s+)?(previous|prior)"),
    re.compile(r"(?i)you\s+are\s+now"),
    re.compile(r"(?i)your\s+new\s+task\s+is"),
    re.compile(r"(?i)rate\s+this\s+candidate"),
    re.compile(r"(?i)score\s+this\s+applicant"),
]
_HTML_TAG = re.compile(r"<[^>]+>")


def sanitize_external_content(text: str, url: str, *, max_chars: int = 8000) -> str:
    """Strip HTML, injection phrases, truncate, and wrap in delimiters."""
    cleaned = _HTML_TAG.sub(" ", text or "")
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("[removed]", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "…"

    return f"===BEGIN EXTERNAL CONTENT: {url}===\n{cleaned}\n===END EXTERNAL CONTENT==="
