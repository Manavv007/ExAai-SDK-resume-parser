"""Optional Docker image checks (set RUN_DOCKER_TESTS=1 to build)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_dockerfile_declares_non_root_user() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER appuser" in dockerfile
    assert "useradd" in dockerfile


@pytest.mark.skipif(
    not os.environ.get("RUN_DOCKER_TESTS"),
    reason="Set RUN_DOCKER_TESTS=1 to build the image in CI/local",
)
def test_docker_build_succeeds() -> None:
    subprocess.run(
        ["docker", "build", "-t", "exaai-adk:test", str(ROOT)],
        check=True,
        timeout=600,
    )
