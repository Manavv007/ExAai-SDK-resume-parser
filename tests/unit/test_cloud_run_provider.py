"""Cloud Run sandbox provider tests."""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent.config import Settings
from agent.sandbox.base import SandboxCommand
from agent.sandbox.cloud_run_provider import CloudRunJobRef, CloudRunSandboxProvider


class FakeResponse:
    def __init__(self, data: dict[str, Any], status_code: int = 200) -> None:
        self.data = data
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self.data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeHttpClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.posts: list[dict[str, Any]] = []
        self.gets: list[dict[str, Any]] = []

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> FakeResponse:
        self.posts.append({"url": url, "headers": headers, "json": json})
        return self.responses.pop(0)

    async def get(self, url: str, *, headers: dict[str, str]) -> FakeResponse:
        self.gets.append({"url": url, "headers": headers})
        return self.responses.pop(0)


class FakeBlob:
    def __init__(self, text: str) -> None:
        self.text = text

    def download_as_text(self) -> str:
        return self.text


class FakeBucket:
    def __init__(self, text: str) -> None:
        self.text = text
        self.blob_name = ""

    def blob(self, name: str) -> FakeBlob:
        self.blob_name = name
        return FakeBlob(self.text)


class FakeStorageClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.bucket_name = ""
        self.bucket_obj = FakeBucket(text)

    def bucket(self, name: str) -> FakeBucket:
        self.bucket_name = name
        return self.bucket_obj


def _settings() -> Settings:
    return Settings(
        gemini_api_key="test-gemini",
        exa_api_key="test-exa",
        api_keys="test-key",
        llm_provider="gemini",
        gcp_project_id="project-1",
        gcp_region="asia-south1",
        cloud_run_sandbox_job_name="repo-evaluator",
        sandbox_report_bucket="reports-bucket",
        sandbox_report_prefix="candidate-sandbox",
        sandbox_timeout_seconds=123,
        sandbox_poll_interval_seconds=0,
    )


@pytest.mark.asyncio
async def test_cloud_run_provider_runs_job_and_reads_report() -> None:
    report_payload = {
        "repo": "owner/project",
        "url": "https://github.com/owner/project",
        "provider": "cloud_run",
        "clone_ok": True,
        "detected_stack": ["python"],
        "commands": [{"step": "clone", "command": "git clone", "ok": True}],
        "quality_signals": {},
        "risk_flags": [],
        "summary": "ok",
        "timed_out": False,
    }
    http_client = FakeHttpClient(
        [
            FakeResponse({"name": "projects/project-1/locations/asia-south1/operations/op-1"}),
            FakeResponse({"done": False}),
            FakeResponse({"done": True}),
        ]
    )
    storage_client = FakeStorageClient(json.dumps(report_payload))
    provider = CloudRunSandboxProvider(
        settings=_settings(),
        http_client=http_client,
        storage_client=storage_client,
        access_token="token-1",
    )

    report = await provider.evaluate_repo(
        repo_url="https://github.com/owner/project",
        repo_name="owner/project",
        commands=[SandboxCommand(step="test", command="python -m pytest -q")],
    )

    assert report.clone_ok is True
    assert report.detected_stack == ["python"]
    assert len(http_client.posts) == 1
    assert len(http_client.gets) == 2
    post = http_client.posts[0]
    assert post["headers"]["Authorization"] == "Bearer token-1"
    assert post["url"].endswith("/projects/project-1/locations/asia-south1/jobs/repo-evaluator:run")
    overrides = post["json"]["overrides"]
    assert overrides["taskCount"] == 1
    assert overrides["timeout"] == "123s"
    env = {item["name"]: item["value"] for item in overrides["containerOverrides"][0]["env"]}
    assert env["REPO_URL"] == "https://github.com/owner/project"
    assert env["REPO_NAME"] == "owner/project"
    assert env["SANDBOX_TIMEOUT_SECONDS"] == "123"
    assert env["REPORT_OUTPUT_URI"].startswith(
        "gs://reports-bucket/candidate-sandbox/owner-project/"
    )
    assert json.loads(env["COMMAND_PLAN_JSON"]) == [
        {"step": "test", "command": "python -m pytest -q"}
    ]
    assert storage_client.bucket_name == "reports-bucket"
    assert storage_client.bucket_obj.blob_name.startswith("candidate-sandbox/owner-project/")


@pytest.mark.asyncio
async def test_cloud_run_provider_raises_on_operation_error() -> None:
    http_client = FakeHttpClient(
        [
            FakeResponse({"name": "projects/project-1/locations/asia-south1/operations/op-1"}),
            FakeResponse({"done": True, "error": {"message": "job failed"}}),
        ]
    )
    provider = CloudRunSandboxProvider(
        settings=_settings(),
        http_client=http_client,
        storage_client=FakeStorageClient("{}"),
        access_token="token-1",
    )

    with pytest.raises(RuntimeError, match="job failed"):
        await provider.evaluate_repo(
            repo_url="https://github.com/owner/project",
            repo_name="owner/project",
            commands=[],
        )


def test_cloud_run_provider_validates_required_config() -> None:
    provider = CloudRunSandboxProvider(
        job_ref=CloudRunJobRef(
            project_id="",
            region="asia-south1",
            job_name="repo-evaluator",
            report_bucket="reports-bucket",
            report_prefix="sandbox-reports",
        ),
        settings=_settings(),
        access_token="token-1",
    )

    with pytest.raises(ValueError, match="GCP_PROJECT_ID"):
        provider._validate_config()
