"""Sandbox provider factory."""

from __future__ import annotations

from agent.config import get_settings
from agent.sandbox.base import SandboxProvider
from agent.sandbox.cloud_run_provider import CloudRunSandboxProvider


def create_sandbox_provider() -> SandboxProvider:
    """Create the configured sandbox provider."""
    settings = get_settings()
    if settings.sandbox_provider == "cloud_run":
        return CloudRunSandboxProvider()
    raise NotImplementedError(
        f"Sandbox provider '{settings.sandbox_provider}' is not implemented in this build."
    )
