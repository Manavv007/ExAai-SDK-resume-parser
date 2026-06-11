"""Prepare sandbox file-focus payloads for Cloud Run job environment variables."""

from __future__ import annotations

import json
from typing import Any

# Cloud Run container env values are capped (~32 KiB); full repo trees exceed this.
_MAX_FILE_FOCUS_JSON_BYTES = 30_000


def compact_file_focus_for_cloud_run_job(
    file_focus: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Shrink a focus spec before sending as ``FILE_FOCUS_JSON`` to repo-evaluator.

    The evaluator clones the repo and can list paths locally; the API-side tree is
    only needed to resolve agent ``focus_paths`` before the job starts.
    """
    if not isinstance(file_focus, dict):
        return None

    compact: dict[str, Any] = {
        key: value
        for key, value in file_focus.items()
        if key not in ("file_paths", "agent_focus_paths")
    }
    if not compact:
        return None
    return compact


def file_focus_json_for_cloud_run_job(file_focus: dict[str, Any] | None) -> str:
    """Serialize focus spec for Cloud Run, enforcing the env size budget."""
    compact = compact_file_focus_for_cloud_run_job(file_focus)
    if not compact:
        raise ValueError("file_focus is empty after compaction for Cloud Run job")

    encoded = json.dumps(compact, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _MAX_FILE_FOCUS_JSON_BYTES:
        raise ValueError(
            "FILE_FOCUS_JSON exceeds Cloud Run env size limit "
            f"({len(encoded.encode('utf-8'))} bytes > {_MAX_FILE_FOCUS_JSON_BYTES})"
        )
    return encoded
