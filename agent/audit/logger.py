"""Structured audit logging — no resume or PII."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("exaai_adk.audit")


def log_screening_result(
    state: dict[str, Any],
    result: dict[str, Any],
    *,
    request_id: str | None = None,
) -> None:
    """Emit one JSON audit line per screening run."""
    similarity = result.get("resume_similarity_score") or {}
    enriched = state.get("enriched_contents") or []
    errors = result.get("errors") or []
    sources = result.get("sources_crawled") or []

    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "request_id": request_id or state.get("request_id"),
        "application_id": result.get("application_id") or state.get("application_id"),
        "job_id": result.get("job_id") or state.get("job_id"),
        "resume_screening_status": result.get("resume_screening_status"),
        "recommendation": result.get("recommendation"),
        "overall_score": similarity.get("score"),
        "model_version": (result.get("metadata") or {}).get("model_version"),
        "processing_time_ms": (result.get("metadata") or {}).get("processing_time_ms")
        or state.get("processing_time_ms"),
        "criteria_count": len(result.get("requirement_matches") or []),
        "sources_attempted": len(state.get("profile_urls") or []),
        "sources_successful": sum(1 for item in enriched if item.get("ok")),
        "sources_crawled_count": len(sources),
        "redaction_count": state.get("redaction_count", 0),
        "error_count": len(errors),
        "retry_count": state.get("retry_count", 0),
    }

    level = logging.ERROR if result.get("resume_screening_status") == "failed" else logging.INFO
    logger.log(level, json.dumps(payload, default=str))
