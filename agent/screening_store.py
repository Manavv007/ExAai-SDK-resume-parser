"""Lightweight persisted store for screening results."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.config import get_settings


class ScreeningResultStore:
    """Persist latest screening result snapshots for polling endpoints."""

    def __init__(self, base_path: str | None = None) -> None:
        settings = get_settings()
        self.base_path = Path(base_path or settings.screening_result_store_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        application_id: str,
        job_id: str,
        status: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        record = {
            "application_id": application_id,
            "job_id": job_id,
            "status": status,
            "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "result": result,
        }
        path = self._path_for(application_id, job_id)
        path.write_text(json.dumps(record, ensure_ascii=True, indent=2), encoding="utf-8")
        return record

    def load(self, *, application_id: str, job_id: str) -> dict[str, Any] | None:
        path = self._path_for(application_id, job_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def _path_for(self, application_id: str, job_id: str) -> Path:
        safe_name = f"{application_id}__{job_id}.json"
        return self.base_path / safe_name
