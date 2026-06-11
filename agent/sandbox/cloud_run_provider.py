"""Cloud Run Jobs sandbox provider."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx

from agent.config import Settings, get_settings
from agent.sandbox.base import SandboxCommand
from agent.sandbox.focus_transport import file_focus_json_for_cloud_run_job
from agent.sandbox.models import RepoExecutionReport
from agent.sandbox.report_store import SandboxReportStore, build_report_uri

CLOUD_RUN_API = "https://run.googleapis.com/v2"
CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


@dataclass(frozen=True)
class CloudRunJobRef:
    project_id: str
    region: str
    job_name: str
    report_bucket: str
    report_prefix: str = "sandbox-reports"


class CloudRunSandboxProvider:
    """Production sandbox backend that will execute evaluator Cloud Run Jobs."""

    def __init__(
        self,
        job_ref: CloudRunJobRef | None = None,
        *,
        settings: Settings | None = None,
        http_client: Any | None = None,
        storage_client: Any | None = None,
        access_token: str | None = None,
    ) -> None:
        settings = settings or get_settings()
        self._settings = settings
        self.job_ref = job_ref or CloudRunJobRef(
            project_id=settings.gcp_project_id,
            region=settings.gcp_region,
            job_name=settings.cloud_run_sandbox_job_name,
            report_bucket=settings.sandbox_report_bucket,
            report_prefix=settings.sandbox_report_prefix,
        )
        self.timeout_seconds = settings.sandbox_timeout_seconds
        self.poll_interval_seconds = settings.sandbox_poll_interval_seconds
        self.http_client = http_client
        self.report_store = SandboxReportStore(
            storage_client=storage_client,
            project_id=self.job_ref.project_id or None,
            credentials_path=settings.sandbox_google_application_credentials,
        )
        self.access_token = access_token

    async def evaluate_repo(
        self,
        *,
        repo_url: str,
        repo_name: str,
        commands: list[SandboxCommand],
        file_focus: dict[str, Any] | None = None,
    ) -> RepoExecutionReport:
        """Execute a Cloud Run Job and return the evaluator report."""
        self._validate_config()
        report_uri = self._build_report_uri(repo_name)
        operation_name = await self._run_job(
            repo_url=repo_url,
            repo_name=repo_name,
            report_uri=report_uri,
            commands=commands,
            file_focus=file_focus,
        )
        await self._wait_for_operation(operation_name)
        return await asyncio.to_thread(self._read_report, report_uri)

    def _validate_config(self) -> None:
        missing = []
        if not self.job_ref.project_id:
            missing.append("GCP_PROJECT_ID")
        if not self.job_ref.region:
            missing.append("GCP_REGION")
        if not self.job_ref.job_name:
            missing.append("CLOUD_RUN_SANDBOX_JOB_NAME")
        if not self.job_ref.report_bucket:
            missing.append("SANDBOX_REPORT_BUCKET")
        if missing:
            raise ValueError("Missing Cloud Run sandbox config: " + ", ".join(missing))

    async def _run_job(
        self,
        *,
        repo_url: str,
        repo_name: str,
        report_uri: str,
        commands: list[SandboxCommand],
        file_focus: dict[str, Any] | None = None,
    ) -> str:
        env = [
            {"name": "REPO_URL", "value": repo_url},
            {"name": "REPO_NAME", "value": repo_name},
            {"name": "REPORT_OUTPUT_URI", "value": report_uri},
            {"name": "SANDBOX_TIMEOUT_SECONDS", "value": str(self.timeout_seconds)},
        ]
        if commands:
            env.append(
                {
                    "name": "COMMAND_PLAN_JSON",
                    "value": json.dumps(
                        [{"step": command.step, "command": command.command} for command in commands]
                    ),
                }
            )
        if file_focus:
            env.append(
                {
                    "name": "FILE_FOCUS_JSON",
                    "value": file_focus_json_for_cloud_run_job(file_focus),
                }
            )

        payload = {
            "overrides": {
                "containerOverrides": [{"env": env}],
                "taskCount": 1,
                "timeout": f"{self.timeout_seconds}s",
            }
        }
        response = await self._request_json(
            "POST",
            f"{CLOUD_RUN_API}/{self._job_resource_name()}:run",
            json_payload=payload,
        )
        operation_name = response.get("name")
        if not isinstance(operation_name, str) or not operation_name:
            raise RuntimeError("Cloud Run jobs.run response did not include operation name")
        return operation_name

    async def _wait_for_operation(self, operation_name: str) -> None:
        deadline = asyncio.get_running_loop().time() + self.timeout_seconds + 30
        while True:
            operation = await self._request_json(
                "GET",
                f"{CLOUD_RUN_API}/{operation_name}",
            )
            if operation.get("done") is True:
                error = operation.get("error")
                if isinstance(error, dict):
                    message = error.get("message") or error.get("code") or "unknown error"
                    raise RuntimeError(f"Cloud Run sandbox job failed: {message}")
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Timed out waiting for Cloud Run sandbox job operation")
            await asyncio.sleep(self.poll_interval_seconds)

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {await self._get_access_token()}",
            "Content-Type": "application/json",
        }
        if self.http_client is not None:
            response = await self._send_request(
                self.http_client,
                method,
                url,
                headers=headers,
                json_payload=json_payload,
            )
        else:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await self._send_request(
                    client,
                    method,
                    url,
                    headers=headers,
                    json_payload=json_payload,
                )
        if response.status_code >= 400:
            detail = response.text.strip()
            raise RuntimeError(
                f"Cloud Run API {response.status_code} for {url}: {detail[:2000]}"
            )
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Cloud Run API response was not a JSON object")
        return data

    @staticmethod
    async def _send_request(
        client: Any,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_payload: dict[str, Any] | None,
    ) -> Any:
        if method == "POST":
            return await client.post(url, headers=headers, json=json_payload)
        if method == "GET":
            return await client.get(url, headers=headers)
        raise ValueError(f"Unsupported HTTP method: {method}")

    async def _get_access_token(self) -> str:
        if self.access_token:
            return self.access_token
        return await asyncio.to_thread(self._load_access_token)

    def _load_access_token(self) -> str:
        from agent.gcp_credentials import access_token_from_credentials, load_sandbox_gcp_credentials

        credentials = load_sandbox_gcp_credentials(self._settings)
        return access_token_from_credentials(credentials)

    def _read_report(self, report_uri: str) -> RepoExecutionReport:
        return self.report_store.read(report_uri)

    def _job_resource_name(self) -> str:
        return (
            f"projects/{self.job_ref.project_id}/locations/{self.job_ref.region}"
            f"/jobs/{self.job_ref.job_name}"
        )

    def _build_report_uri(self, repo_name: str) -> str:
        return build_report_uri(
            bucket=self.job_ref.report_bucket,
            prefix=self.job_ref.report_prefix,
            repo_name=repo_name,
        )
