"""Upload validation for POST /screen."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

MAX_UPLOAD_BYTES = 5 * 1024 * 1024

PDF_MAGIC = b"%PDF"
DOCX_MAGIC = b"PK\x03\x04"
TEXT_EXTENSIONS = {".txt", ".md", ".text"}


@dataclass
class FileValidationError:
    code: str
    message: str


def validate_uuid(value: str, field: str) -> FileValidationError | None:
    try:
        uuid.UUID(value)
    except ValueError:
        return FileValidationError(
            code="INVALID_REQUEST",
            message=f"{field} must be a valid UUID",
        )
    return None


def _detect_kind(data: bytes, filename: str) -> str | None:
    lower = (filename or "").lower()
    if data.startswith(PDF_MAGIC):
        return "pdf"
    if data.startswith(DOCX_MAGIC) or lower.endswith(".docx"):
        return "docx"
    if lower.endswith(tuple(TEXT_EXTENSIONS)) or not data:
        return "text"
    return None


def validate_upload_bytes(
    data: bytes,
    filename: str,
    *,
    required: bool = True,
    label: str = "file",
) -> tuple[str | None, FileValidationError | None]:
    if not data:
        if required:
            return None, FileValidationError(
                code="INVALID_REQUEST",
                message=f"{label} is empty",
            )
        return None, None

    if len(data) > MAX_UPLOAD_BYTES:
        return None, FileValidationError(
            code="INVALID_REQUEST",
            message=f"{label} exceeds 5MB limit",
        )

    kind = _detect_kind(data, filename)
    if kind is None:
        return None, FileValidationError(
            code="INVALID_REQUEST",
            message=f"{label} must be PDF, DOCX, or plain text",
        )
    return kind, None
