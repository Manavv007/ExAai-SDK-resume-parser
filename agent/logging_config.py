"""Console logging for local dev and Cloud Run."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Send exaai-adk loggers to stderr at the configured level."""
    numeric = getattr(logging, str(level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(levelname)s:%(name)s: %(message)s",
        force=True,
    )
    # Keep third-party HTTP noise down unless DEBUG.
    if numeric > logging.DEBUG:
        for name in ("httpx", "httpcore", "google.auth", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)
