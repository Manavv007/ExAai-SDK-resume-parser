"""Sandbox provider factory tests."""

from __future__ import annotations

import pytest

from agent.sandbox.cloud_run_provider import CloudRunSandboxProvider
from agent.sandbox.providers import create_sandbox_provider


def test_create_sandbox_provider_defaults_to_cloud_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SANDBOX_PROVIDER", "cloud_run")

    provider = create_sandbox_provider()

    assert isinstance(provider, CloudRunSandboxProvider)


def test_create_sandbox_provider_rejects_unimplemented_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SANDBOX_PROVIDER", "e2b")

    with pytest.raises(NotImplementedError):
        create_sandbox_provider()
