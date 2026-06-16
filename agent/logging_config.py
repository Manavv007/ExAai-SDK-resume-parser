"""Console logging for local dev and Cloud Run."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

_BASE_LOG_RECORD_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
}


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def trace_enabled() -> bool:
    return _is_truthy(os.getenv("AGENT_TRACE_ENABLED"))


def trace_event(logger: logging.Logger, event: str, **fields: object) -> None:
    if not trace_enabled():
        return
    logger.info(event, extra={"event": event, **fields})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if isinstance(event, str) and event:
            payload["event"] = event
        for key, value in record.__dict__.items():
            if key in _BASE_LOG_RECORD_FIELDS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", *, log_format: str = "text") -> None:
    """Send exaai-adk loggers to stderr at the configured level."""
    numeric = getattr(logging, str(level or "INFO").upper(), logging.INFO)
    normalized_format = str(log_format or os.getenv("LOG_FORMAT", "text")).strip().lower()
    if normalized_format == "json":
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logging.basicConfig(
        level=numeric,
        handlers=[handler],
        force=True,
    )
    # Keep third-party HTTP noise down unless DEBUG.
    if numeric > logging.DEBUG:
        for name in ("httpx", "httpcore", "google.auth", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)
