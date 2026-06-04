import uuid

from api.file_validation import (
    MAX_UPLOAD_BYTES,
    validate_upload_bytes,
    validate_uuid,
)


def test_validate_uuid_ok() -> None:
    assert validate_uuid(str(uuid.uuid4()), "application_id") is None


def test_validate_uuid_bad() -> None:
    err = validate_uuid("not-a-uuid", "job_id")
    assert err is not None
    assert err.code == "INVALID_REQUEST"


def test_validate_upload_pdf() -> None:
    kind, err = validate_upload_bytes(b"%PDF-1.4\n", "resume.pdf")
    assert err is None
    assert kind == "pdf"


def test_validate_upload_too_large() -> None:
    data = b"%PDF" + b"x" * MAX_UPLOAD_BYTES
    _, err = validate_upload_bytes(data, "resume.pdf")
    assert err is not None
    assert "5MB" in err.message
