"""Deterministic prep before the ADK screening agent runs."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from agent.security.pii_redactor import redact_text
from agent.tools.link_extractor import extract_links
from agent.tools.parser import parse_file, parse_jd_structured
from agent.tools.rubric_builder import build_rubric_bundle


def prepare_screening_state(
    *,
    application_id: str,
    job_id: str,
    resume_bytes: bytes,
    resume_filename: str,
    jd_bytes: bytes | None = None,
    jd_filename: str = "",
    jd_text: str | None = None,
) -> dict[str, Any]:
    """
    Parse inputs, redact PII, extract links, structure JD.

    Returns a dict suitable for ADK ``session.state`` (no raw PII in values
    returned to the model except redacted resume text).
    """
    started = time.monotonic()

    resume_doc = parse_file(resume_bytes, resume_filename)
    if jd_text:
        jd_raw = jd_text.strip()
    elif jd_bytes:
        jd_raw = parse_file(jd_bytes, jd_filename).text
    else:
        raise ValueError("Either jd_text or jd_bytes is required")

    jd_structured = parse_jd_structured(jd_raw)
    resume_text, redaction_summary = redact_text(resume_doc.text)

    links = extract_links(
        resume_doc.text,
        jd=jd_structured,
        pdf_hyperlinks=resume_doc.hyperlinks,
    )
    rubric_bundle = build_rubric_bundle(jd_structured)

    return {
        "application_id": application_id,
        "job_id": job_id,
        "resume_text": resume_text,
        "jd_raw": jd_raw,
        "jd_structured": asdict(jd_structured),
        "profile_urls": [link.url for link in links],
        "profile_url_meta": [
            {"url": link.url, "source": link.source, "platform": link.platform}
            for link in links
        ],
        "redaction_count": redaction_summary.redaction_count,
        "enriched_contents": [],
        "rubric": rubric_bundle["rubric"],
        "rubric_preamble": rubric_bundle["rubric_preamble"],
        "prep_latency_ms": int((time.monotonic() - started) * 1000),
    }
