"""Sandbox report storage tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.sandbox.models import CommandResult, RepoExecutionReport
from agent.sandbox.report_store import (
    SandboxReportStore,
    build_report_uri,
    parse_gcs_uri,
    slug_repo_name,
)


class FakeBlob:
    def __init__(self) -> None:
        self.uploaded = ""
        self.content_type = ""

    def upload_from_string(self, payload: str, *, content_type: str) -> None:
        self.uploaded = payload
        self.content_type = content_type

    def download_as_text(self) -> str:
        return self.uploaded


class FakeBucket:
    def __init__(self) -> None:
        self.blobs: dict[str, FakeBlob] = {}

    def blob(self, name: str) -> FakeBlob:
        if name not in self.blobs:
            self.blobs[name] = FakeBlob()
        return self.blobs[name]


class FakeStorageClient:
    def __init__(self) -> None:
        self.buckets: dict[str, FakeBucket] = {}

    def bucket(self, name: str) -> FakeBucket:
        if name not in self.buckets:
            self.buckets[name] = FakeBucket()
        return self.buckets[name]


def _report() -> RepoExecutionReport:
    return RepoExecutionReport(
        repo="owner/project",
        url="https://github.com/owner/project",
        clone_ok=True,
        commands=[CommandResult(step="clone", command="git clone", ok=True)],
        summary="ok",
    )


def test_slug_repo_name_and_build_report_uri() -> None:
    assert slug_repo_name("owner/project") == "owner-project"
    assert slug_repo_name("  ") == "repo"

    uri = build_report_uri(
        bucket="reports-bucket",
        prefix="/candidate-sandbox/",
        repo_name="owner/project",
        run_id="run-1",
    )

    assert uri == "gs://reports-bucket/candidate-sandbox/owner-project/run-1.json"


def test_parse_gcs_uri_validation() -> None:
    assert parse_gcs_uri("gs://bucket/path/report.json") == ("bucket", "path/report.json")

    with pytest.raises(ValueError):
        parse_gcs_uri("https://bucket/path")
    with pytest.raises(ValueError):
        parse_gcs_uri("gs://bucket")


def test_report_store_local_round_trip(tmp_path: Path) -> None:
    store = SandboxReportStore()
    path = tmp_path / "reports" / "report.json"

    store.write(_report(), str(path))
    loaded = store.read(str(path))

    assert loaded.repo == "owner/project"
    assert loaded.clone_ok is True


def test_report_store_gcs_round_trip() -> None:
    client = FakeStorageClient()
    store = SandboxReportStore(storage_client=client)
    uri = "gs://reports-bucket/sandbox-reports/owner-project/run-1.json"

    store.write(_report(), uri)
    loaded = store.read(uri)

    assert loaded.repo == "owner/project"
    blob = client.buckets["reports-bucket"].blobs["sandbox-reports/owner-project/run-1.json"]
    assert blob.content_type == "application/json"
