"""Vertex AI Agent Engine custom application.

The screening service is a pipeline (parse -> redact PII -> enrich -> sandbox ->
score -> validate), not a chat agent. Agent Engine's default ``AdkApp`` only exposes
``stream_query(message=...)`` and never runs the deterministic prep/post-processing
layers. This custom template wraps :func:`agent.pipeline.run_screening_async` so the
entire pipeline runs remotely behind a single ``screen`` operation.

Agent Engine request payloads are JSON, so binary resume/JD files are passed as
base64-encoded strings and decoded here.
"""

from __future__ import annotations

from typing import Any


class ResumeScreeningApp:
    """Custom Agent Engine application exposing a ``screen`` operation."""

    def set_up(self) -> None:
        """Run once per instance cold start (heavy init lives here, not __init__)."""
        import os

        # Only /tmp is writable on Agent Engine instances and it is not shared
        # across replicas. Point local caches/result store there.
        os.environ.setdefault("URL_CACHE_PATH", "/tmp/url_cache.db")
        os.environ.setdefault("SCREENING_RESULT_STORE_PATH", "/tmp/screening-results")
        os.makedirs("/tmp/screening-results", exist_ok=True)

        from agent.config import get_settings
        from agent.logging_config import configure_logging

        settings = get_settings()
        configure_logging(settings.log_level, log_format=settings.log_format)

        # Presidio needs a spaCy model. The wheel is declared in requirements, but
        # download here as a fallback so PII redaction never crashes at runtime.
        self._ensure_spacy_model()

    @staticmethod
    def _ensure_spacy_model() -> None:
        import spacy

        for name in ("en_core_web_lg", "en_core_web_sm"):
            if spacy.util.is_package(name):
                return
        try:
            from spacy.cli import download

            download("en_core_web_sm")
        except Exception:  # pragma: no cover - best effort fallback
            pass

    def screen(
        self,
        application_id: str,
        job_id: str,
        resume_b64: str | None = None,
        resume_url: str | None = None,
        resume_filename: str = "resume.pdf",
        resume_auth_token: str | None = None,
        jd_text: str | None = None,
        jd_b64: str | None = None,
        jd_url: str | None = None,
        jd_filename: str = "",
        jd_auth_token: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the full screening pipeline and return resume-screening-result-v1 JSON.

        Provide the resume as EITHER ``resume_b64`` (base64 bytes, good for quick
        tests) OR ``resume_url`` (an ``https://`` signed URL or ``gs://`` URI, the
        production-friendly path that matches job_applications.resume_url).

        Args:
            application_id: UUID from job_applications.
            job_id: UUID from jobs.
            resume_b64: Base64-encoded resume bytes (PDF/DOCX).
            resume_url: https:// or gs:// location of the resume (fetched server-side).
            resume_filename: Filename used to detect format; derived from resume_url
                when not provided.
            resume_auth_token: Optional Bearer token sent as ``Authorization: Bearer
                <token>`` when fetching an ``https://`` resume_url (ignored for gs://).
            jd_text: Plain-text job description (provide this OR jd_b64/jd_url).
            jd_b64: Base64-encoded JD file bytes.
            jd_url: https:// or gs:// location of the JD file (fetched server-side).
            jd_filename: JD filename; derived from jd_url when not provided.
            jd_auth_token: Optional Bearer token for an ``https://`` jd_url.
            request_id: Optional trace id; generated when omitted.
        """
        import asyncio

        from agent.pipeline import run_screening_async

        resume_bytes = self._resolve_bytes(
            resume_b64, resume_url, label="resume", auth_token=resume_auth_token
        )
        if resume_bytes is None:
            raise ValueError("Provide either resume_b64 or resume_url.")
        if resume_url and resume_filename == "resume.pdf":
            derived = self._filename_from_url(resume_url)
            if derived:
                resume_filename = derived

        jd_bytes = self._resolve_bytes(jd_b64, jd_url, label="jd", auth_token=jd_auth_token)
        if jd_url and not jd_filename:
            jd_filename = self._filename_from_url(jd_url) or ""

        return asyncio.run(
            run_screening_async(
                application_id=application_id,
                job_id=job_id,
                resume_bytes=resume_bytes,
                resume_filename=resume_filename,
                jd_bytes=jd_bytes,
                jd_filename=jd_filename,
                jd_text=jd_text,
                request_id=request_id,
            )
        )

    # Cap server-side downloads to avoid abuse / oversized payloads.
    MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

    @classmethod
    def _resolve_bytes(
        cls,
        b64: str | None,
        url: str | None,
        *,
        label: str,
        auth_token: str | None = None,
    ) -> bytes | None:
        """Resolve file bytes from base64 OR a remote URL (base64 takes priority)."""
        import base64

        if b64:
            return base64.b64decode(b64)
        if url:
            return cls._fetch_url_bytes(url, label=label, auth_token=auth_token)
        return None

    @classmethod
    def _fetch_url_bytes(cls, url: str, *, label: str, auth_token: str | None = None) -> bytes:
        """Download bytes from an https:// URL or gs:// URI.

        Only https and gs schemes are allowed (basic SSRF hardening). The caller is
        trusted to pass its own storage location (e.g. a Supabase signed URL or a
        GCS object in the same project). ``auth_token``, when given, is sent as an
        ``Authorization: Bearer`` header on https fetches (ignored for gs://, which
        uses Application Default Credentials).
        """
        url = (url or "").strip()
        if url.startswith("gs://"):
            from google.cloud import storage

            bucket_name, _, blob_name = url[len("gs://"):].partition("/")
            if not bucket_name or not blob_name:
                raise ValueError(f"Invalid GCS URI for {label}: {url!r}")
            blob = storage.Client().bucket(bucket_name).blob(blob_name)
            data = blob.download_as_bytes()
        elif url.startswith("https://"):
            import httpx

            headers = {}
            if auth_token and auth_token.strip():
                headers["Authorization"] = f"Bearer {auth_token.strip()}"
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.content
        else:
            raise ValueError(
                f"{label}_url must be an https:// URL or gs:// URI, got: {url!r}"
            )
        if len(data) > cls.MAX_DOWNLOAD_BYTES:
            raise ValueError(
                f"{label} download exceeds {cls.MAX_DOWNLOAD_BYTES} bytes."
            )
        return data

    @staticmethod
    def _filename_from_url(url: str) -> str:
        """Best-effort filename (with extension) from a URL/URI for format detection."""
        path = (url or "").split("?", 1)[0].split("#", 1)[0].rstrip("/")
        return path.rsplit("/", 1)[-1] if "/" in path else ""

    def register_operations(self) -> dict[str, list[str]]:
        """Expose ``screen`` as a standard (sync, non-streaming) operation."""
        return {"": ["screen"]}
