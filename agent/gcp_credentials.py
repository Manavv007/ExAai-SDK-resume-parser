"""Load GCP credentials for Vertex AI vs sandbox (may use different service accounts)."""

from __future__ import annotations

import os
from typing import Any

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def resolve_credentials_path(*, settings_path: str = "") -> str | None:
    """Return a service-account JSON path from settings or GOOGLE_APPLICATION_CREDENTIALS."""
    path = (settings_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    return path or None


def load_gcp_credentials(
    *,
    settings_path: str = "",
    scopes: list[str] | None = None,
) -> Any:
    """Load service-account credentials from file, else Application Default Credentials."""
    import google.auth
    from google.auth.transport.requests import Request

    scope_list = scopes or [CLOUD_PLATFORM_SCOPE]
    path = resolve_credentials_path(settings_path=settings_path)
    if path:
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(
            path,
            scopes=scope_list,
        )
    else:
        credentials, _project = google.auth.default(scopes=scope_list)

    if not credentials.valid:
        credentials.refresh(Request())
    return credentials


def access_token_from_credentials(credentials: Any) -> str:
    token = getattr(credentials, "token", None)
    if not token:
        raise RuntimeError("Google credentials did not provide an access token")
    return str(token)


def load_sandbox_gcp_credentials(settings: Any) -> Any:
    """
    Credentials for Cloud Run sandbox + GCS on GCP_PROJECT_ID.

    Prefers SANDBOX_GOOGLE_APPLICATION_CREDENTIALS, then VERTEX/GOOGLE_APPLICATION_CREDENTIALS.
    When Vertex and sandbox use different projects and no sandbox key is set, falls back to
    gcloud user ADC (ignoring service-account JSON env vars).
    """
    from agent.config import resolve_vertex_gcp_project

    explicit = str(getattr(settings, "sandbox_google_application_credentials", "") or "").strip()
    if explicit:
        return load_gcp_credentials(settings_path=explicit)

    for candidate in (
        str(getattr(settings, "vertex_google_application_credentials", "") or "").strip(),
        str(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip(),
    ):
        if candidate:
            return load_gcp_credentials(settings_path=candidate)

    sandbox_project = str(getattr(settings, "gcp_project_id", "") or "").strip()
    vertex_project = resolve_vertex_gcp_project(settings)
    if vertex_project and sandbox_project and vertex_project != sandbox_project:
        return _load_adc_without_service_account_env()

    return load_gcp_credentials(settings_path="")


def _load_adc_without_service_account_env() -> Any:
    """Application Default Credentials from gcloud login, not GOOGLE_APPLICATION_CREDENTIALS."""
    saved = {
        key: os.environ.pop(key, None)
        for key in (
            "GOOGLE_APPLICATION_CREDENTIALS",
            "VERTEX_GOOGLE_APPLICATION_CREDENTIALS",
            "SANDBOX_GOOGLE_APPLICATION_CREDENTIALS",
        )
    }
    try:
        return load_gcp_credentials(settings_path="")
    finally:
        for key, value in saved.items():
            if value:
                os.environ[key] = value
