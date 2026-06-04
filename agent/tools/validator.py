"""Output schema validation for resume-screening-result-v1."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from jsonschema import Draft7Validator
from pydantic import ValidationError as PydanticValidationError

from agent.schema import SCHEMA_PATH
from agent.schema.models import ResumeScreeningResult


@dataclass
class ValidationOutcome:
    ok: bool
    errors: list[str] = field(default_factory=list)


@lru_cache
def _get_validator() -> Draft7Validator:
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        schema = json.load(f)
    return Draft7Validator(
        schema,
        format_checker=Draft7Validator.FORMAT_CHECKER,
    )


def validate_result(data: dict[str, Any], *, use_pydantic: bool = True) -> bool:
    """Return True if data conforms to resume-screening-result-v1."""
    return validate_result_detailed(data, use_pydantic=use_pydantic).ok


def validate_result_detailed(
    data: dict[str, Any],
    *,
    use_pydantic: bool = True,
) -> ValidationOutcome:
    """
    Validate against JSON Schema (draft-07) and optionally Pydantic models.

    JSON Schema is the source of truth; Pydantic adds Python-native types and
    status-specific rules (completed vs failed).
    """
    errors: list[str] = []

    validator = _get_validator()
    schema_errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    if schema_errors:
        for err in schema_errors:
            path = ".".join(str(p) for p in err.path) or "(root)"
            errors.append(f"[jsonschema] {path}: {err.message}")
        return ValidationOutcome(ok=False, errors=errors)

    if not use_pydantic:
        return ValidationOutcome(ok=True)

    try:
        ResumeScreeningResult.model_validate(data)
    except PydanticValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(f"[pydantic] {loc}: {err['msg']}")
        return ValidationOutcome(ok=False, errors=errors)

    return ValidationOutcome(ok=True)


def parse_result(data: dict[str, Any]) -> ResumeScreeningResult:
    """Validate and parse into a typed model. Raises on invalid data."""
    outcome = validate_result_detailed(data, use_pydantic=True)
    if not outcome.ok:
        raise ValueError("; ".join(outcome.errors))
    return ResumeScreeningResult.model_validate(data)
