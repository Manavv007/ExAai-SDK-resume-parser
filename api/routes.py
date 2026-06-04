"""Screening routes."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from agent.pipeline import run_screening_async
from api.auth import require_api_key
from api.errors import screening_error_from_exception
from api.file_validation import (
    FileValidationError,
    validate_upload_bytes,
    validate_uuid,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["screening"])


async def _read_optional_jd_file(request: Request) -> tuple[bytes | None, str]:
    """
    Read optional JD file from multipart form without binding it on the route.

    Swagger UI often sends an invalid string into optional File fields; reading
    manually avoids 422 validation errors when only jd_text is used.
    """
    form = await request.form()
    jd_part = form.get("jd")
    if not isinstance(jd_part, UploadFile):
        return None, ""
    filename = (jd_part.filename or "").strip()
    if not filename:
        return None, ""
    return await jd_part.read(), filename


def _error_response(
    status: int,
    *,
    code: str,
    message: str,
    application_id: str | None = None,
    job_id: str | None = None,
) -> JSONResponse:
    body: dict = {
        "resume_screening_status": "failed",
        "errors": [{"code": code, "message": message}],
    }
    if application_id:
        body["application_id"] = application_id
    if job_id:
        body["job_id"] = job_id
    return JSONResponse(status_code=status, content=body)


@router.post("/screen")
async def screen(
    request: Request,
    application_id: str = Form(...),
    job_id: str = Form(...),
    resume: UploadFile = File(...),
    jd_text: str | None = Form(
        None,
        description="Job description as plain text (recommended in Swagger UI).",
    ),
    api_key: Annotated[
        str | None,
        Form(
            description=(
                "API token from server .env API_KEYS (paste here in Swagger UI). "
                "Example: dev-local-key-change-me"
            ),
        ),
    ] = None,
) -> JSONResponse:
    """Run resume screening pipeline. Returns resume-screening-result-v1."""
    await require_api_key(request, api_key)
    request_id = getattr(request.state, "request_id", None)

    for field, value in (("application_id", application_id), ("job_id", job_id)):
        err = validate_uuid(value, field)
        if err:
            return _error_response(400, code=err.code, message=err.message)

    resume_bytes = await resume.read()
    resume_kind, resume_err = validate_upload_bytes(
        resume_bytes,
        resume.filename or "resume.pdf",
        label="resume",
    )
    if resume_err:
        return _error_response(
            400,
            code=resume_err.code,
            message=resume_err.message,
            application_id=application_id,
            job_id=job_id,
        )

    jd_bytes, jd_filename = await _read_optional_jd_file(request)
    if jd_bytes is not None:
        _, jd_err = validate_upload_bytes(jd_bytes, jd_filename, label="jd")
        if jd_err:
            return _error_response(
                400,
                code=jd_err.code,
                message=jd_err.message,
                application_id=application_id,
                job_id=job_id,
            )
    elif not (jd_text and jd_text.strip()):
        return _error_response(
            400,
            code="INVALID_REQUEST",
            message="Provide jd_text (paste JD) or upload a jd file via curl -F jd=@file.pdf",
            application_id=application_id,
            job_id=job_id,
        )

    try:
        result = await run_screening_async(
            application_id=application_id,
            job_id=job_id,
            resume_bytes=resume_bytes,
            resume_filename=resume.filename or f"resume.{resume_kind}",
            jd_bytes=jd_bytes,
            jd_filename=jd_filename,
            jd_text=jd_text.strip() if jd_text else None,
            request_id=request_id,
        )
    except ValueError as exc:
        logger.warning("screening_validation_error", extra={"request_id": request_id})
        return _error_response(
            400,
            code="INVALID_REQUEST",
            message=str(exc),
            application_id=application_id,
            job_id=job_id,
        )
    except Exception as exc:
        status, code, message = screening_error_from_exception(exc)
        logger.exception(
            "screening_failed",
            extra={"request_id": request_id, "error_code": code},
        )
        return _error_response(
            status,
            code=code,
            message=message,
            application_id=application_id,
            job_id=job_id,
        )

    status_code = 200 if result.get("resume_screening_status") == "completed" else 422
    return JSONResponse(status_code=status_code, content=result)
