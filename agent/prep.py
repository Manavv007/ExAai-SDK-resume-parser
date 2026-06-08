"""Deterministic prep before the ADK screening agent runs.

Independent work (PII redaction, link extraction, rubric building, profile
identity assessment) is parallelized with a ThreadPoolExecutor so CPU-bound
work runs concurrently without blocking the caller.
"""

from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any

_GITHUB_THREADS: dict[str, tuple[threading.Thread, list[Exception]]] = {}


def retrieve_github_thread(application_id: str) -> tuple[threading.Thread | None, list[Exception]]:
    """Retrieve and pop the GitHub background thread and errors for an application."""
    return _GITHUB_THREADS.pop(application_id, (None, []))

from agent.security.pii_redactor import redact_text
from agent.security.profile_identity import (
    IDENTITY_SCORING_RULES,
    assess_profile_links,
    assessments_to_dicts,
    build_identity_red_flags,
    should_cap_score_for_identity,
    trust_map_from_assessments,
)
from agent.tools.link_extractor import extract_links
from agent.tools.parser import parse_file, parse_jd_structured, parse_resume_structured
from agent.tools.rubric_builder import build_rubric_bundle


def _parallel_prep(
    resume_text_raw: str,
    jd_raw: str,
    pdf_hyperlinks: list[str],
) -> tuple[tuple[str, Any], list, Any, Any]:
    """Run independent CPU-bound prep steps in parallel threads.

    Uses a ThreadPoolExecutor to run PII redaction, link extraction,
    JD structuring, and resume structuring concurrently.  These four
    tasks are completely independent — each takes the raw resume/JD
    text and produces its own output.

    Returns (redacted_text, redaction_summary), links, jd_structured,
    resume_structured.
    """
    with ThreadPoolExecutor(max_workers=4) as pool:
        redact_future = pool.submit(redact_text, resume_text_raw)
        links_future = pool.submit(
            extract_links,
            resume_text_raw,
            jd=None,
            pdf_hyperlinks=pdf_hyperlinks,
        )
        jd_future = pool.submit(parse_jd_structured, jd_raw)
        resume_future = pool.submit(parse_resume_structured, resume_text_raw)

        redacted_text, redaction_summary = redact_future.result()
        links = links_future.result()
        jd_structured = jd_future.result()
        resume_structured = resume_future.result()

    return (redacted_text, redaction_summary), links, jd_structured, resume_structured


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

    Performance: GitHub analysis is started as soon as links are extracted
    and runs concurrently with the rest of prep (PII redaction, JD/resume
    structuring, rubric building).
    """
    import asyncio
    import logging
    import threading

    from agent.tools.github_analyzer import analyze_github_repos, extract_github_username

    prep_logger = logging.getLogger("exaai_adk.prep")
    started = time.monotonic()

    # --- Phase 1: Parse files (I/O-bound, must complete first) ---
    resume_doc = parse_file(resume_bytes, resume_filename)
    if jd_text:
        jd_raw = jd_text.strip()
    elif jd_bytes:
        jd_raw = parse_file(jd_bytes, jd_filename).text
    else:
        raise ValueError("Either jd_text or jd_bytes is required")

    # --- Phase 1.5: Extract links early so GitHub analysis can start ASAP ---
    links = extract_links(
        resume_doc.text,
        jd=None,
        pdf_hyperlinks=resume_doc.hyperlinks,
    )
    link_urls = [link.url for link in links]
    github_username = extract_github_username(link_urls)

    # --- Kick off GitHub analysis in background thread (overlaps with Phase 2/3) ---
    github_repo_analyses: dict[str, Any] = {}
    github_error: list[Exception] = []

    def _run_github_analysis(
        username: str, urls: list[str], jd_raw_text: str
    ) -> None:
        """Run async GitHub analysis in a dedicated event loop thread."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Structure JD for the analyzer (needs the raw text parsed)
            jd_struct = parse_jd_structured(jd_raw_text)
            res = loop.run_until_complete(
                analyze_github_repos(username, urls, asdict(jd_struct))
            )
            github_repo_analyses.update(res)
        except Exception as e:
            github_error.append(e)
        finally:
            loop.close()

    github_thread: threading.Thread | None = None
    if github_username:
        github_thread = threading.Thread(
            target=_run_github_analysis,
            args=(github_username, link_urls, jd_raw),
            daemon=True,
        )
        github_thread.start()
        _GITHUB_THREADS[application_id] = (github_thread, github_error)

    # --- Phase 2: Run independent CPU-bound work in parallel ---
    # PII redaction, JD structuring, and resume structuring run concurrently.
    # Link extraction was already done above to enable early GitHub kickoff.
    with ThreadPoolExecutor(max_workers=3) as pool:
        redact_future = pool.submit(redact_text, resume_doc.text)
        jd_future = pool.submit(parse_jd_structured, jd_raw)
        resume_future = pool.submit(parse_resume_structured, resume_doc.text)

        resume_text, redaction_summary = redact_future.result()
        jd_structured = jd_future.result()
        resume_structured = resume_future.result()

    # --- Phase 3: Dependent work (needs links + jd_structured) ---
    rubric_bundle = build_rubric_bundle(jd_structured)
    profile_assessments = assess_profile_links(resume_doc.text, links)
    profile_trust_by_url = trust_map_from_assessments(profile_assessments)

    return {
        "application_id": application_id,
        "job_id": job_id,
        "resume_text": resume_text,
        "resume_structured": asdict(resume_structured),
        "jd_raw": jd_raw,
        "jd_structured": asdict(jd_structured),
        "profile_urls": link_urls,
        "profile_url_meta": [
            {"url": link.url, "source": link.source, "platform": link.platform}
            for link in links
        ],
        "redaction_count": redaction_summary.redaction_count,
        "enriched_contents": [],
        "rubric": rubric_bundle["rubric"],
        "rubric_preamble": (
            f"{rubric_bundle['rubric_preamble']}\n{IDENTITY_SCORING_RULES}"
        ),
        "profile_trust": assessments_to_dicts(profile_assessments),
        "profile_trust_by_url": profile_trust_by_url,
        "identity_red_flags": build_identity_red_flags(profile_assessments),
        "profile_identity_cap_score": should_cap_score_for_identity(
            profile_assessments
        ),
        "github_username": github_username,
        "github_repo_analyses": github_repo_analyses,
        "prep_latency_ms": int((time.monotonic() - started) * 1000),
    }

