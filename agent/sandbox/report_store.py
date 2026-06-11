"""Storage helpers for sandbox evaluator reports."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.sandbox.models import RepoExecutionReport


def slug_repo_name(repo_name: str) -> str:
    """Return a stable path-safe slug for a repository name."""
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", repo_name.strip()).strip("-")
    return slug or "repo"


def build_report_uri(
    *,
    bucket: str,
    prefix: str,
    repo_name: str,
    run_id: str | None = None,
) -> str:
    """Build a GCS URI for one sandbox report."""
    if not bucket:
        raise ValueError("Report bucket is required")
    clean_prefix = prefix.strip().strip("/") or "sandbox-reports"
    clean_run_id = run_id or uuid.uuid4().hex
    return f"gs://{bucket}/{clean_prefix}/{slug_repo_name(repo_name)}/{clean_run_id}.json"


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parse a gs:// URI into bucket and blob path."""
    if not uri.startswith("gs://"):
        raise ValueError("GCS URI must start with gs://")
    rest = uri.removeprefix("gs://")
    bucket, sep, blob = rest.partition("/")
    if not bucket or not sep or not blob:
        raise ValueError("GCS URI must include bucket and object path")
    return bucket, blob


@dataclass
class SandboxReportStore:
    """Read and write sandbox reports from local files or GCS."""

    storage_client: Any | None = None
    project_id: str | None = None
    credentials_path: str = ""

    def write(self, report: RepoExecutionReport, destination: str) -> None:
        payload = json.dumps(report.model_dump(mode="json"), indent=2)
        if destination.startswith("gs://"):
            self._write_gcs(payload, destination)
            return

        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    def read(self, source: str) -> RepoExecutionReport:
        if source.startswith("gs://"):
            text = self._read_gcs(source)
        else:
            text = Path(source).read_text(encoding="utf-8")
        data = json.loads(text)
        return RepoExecutionReport.model_validate(data)

    def _write_gcs(self, payload: str, destination: str) -> None:
        bucket_name, blob_name = parse_gcs_uri(destination)
        client = self._storage_client()
        bucket = client.bucket(bucket_name)
        bucket.blob(blob_name).upload_from_string(payload, content_type="application/json")

    def _read_gcs(self, source: str) -> str:
        bucket_name, blob_name = parse_gcs_uri(source)
        client = self._storage_client()
        return client.bucket(bucket_name).blob(blob_name).download_as_text()

    def _storage_client(self) -> Any:
        if self.storage_client is not None:
            return self.storage_client
        try:
            from google.cloud import storage
        except ImportError as exc:  # pragma: no cover - dependency is declared for runtime
            raise RuntimeError("google-cloud-storage is required for gs:// report storage") from exc
        from agent.config import get_settings
        from agent.gcp_credentials import load_sandbox_gcp_credentials

        settings = get_settings()
        if self.credentials_path.strip():
            from agent.gcp_credentials import load_gcp_credentials

            credentials = load_gcp_credentials(settings_path=self.credentials_path)
        else:
            credentials = load_sandbox_gcp_credentials(settings)
        return storage.Client(project=self.project_id, credentials=credentials)
