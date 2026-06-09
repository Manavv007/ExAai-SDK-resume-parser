"""Optional subprocess-based integrations for repo profiling."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def run_gitleaks(repo_dir: Path) -> int | None:
    binary = shutil.which("gitleaks")
    if not binary:
        return None
    try:
        completed = subprocess.run(
            [binary, "detect", "--source", str(repo_dir), "--report-format", "json", "--no-banner"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if completed.returncode not in {0, 1}:
            return None
        data = json.loads(completed.stdout or "[]")
        return len(data) if isinstance(data, list) else None
    except Exception:
        return None


def run_trivy_fs(repo_dir: Path) -> dict[str, Any] | None:
    binary = shutil.which("trivy")
    if not binary:
        return None
    try:
        completed = subprocess.run(
            [binary, "fs", "--format", "json", str(repo_dir)],
            capture_output=True,
            text=True,
            check=False,
            timeout=45,
        )
        if completed.returncode not in {0, 1}:
            return None
        return json.loads(completed.stdout or "{}")
    except Exception:
        return None


def run_semgrep(repo_dir: Path) -> dict[str, Any] | None:
    binary = shutil.which("semgrep")
    if not binary:
        return None
    try:
        completed = subprocess.run(
            [binary, "scan", "--config", "p/secrets", "--json", str(repo_dir)],
            capture_output=True,
            text=True,
            check=False,
            timeout=45,
        )
        if completed.returncode not in {0, 1}:
            return None
        return json.loads(completed.stdout or "{}")
    except Exception:
        return None


def run_checkov(repo_dir: Path) -> dict[str, Any] | None:
    binary = shutil.which("checkov")
    if not binary:
        return None
    try:
        completed = subprocess.run(
            [binary, "-d", str(repo_dir), "-o", "json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=45,
        )
        if completed.returncode not in {0, 1}:
            return None
        return json.loads(completed.stdout or "{}")
    except Exception:
        return None


def run_scc(repo_dir: Path) -> dict[str, Any] | None:
    binary = shutil.which("scc")
    if not binary:
        return None
    try:
        completed = subprocess.run(
            [binary, str(repo_dir), "--format", "json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if completed.returncode not in {0, 1}:
            return None
        return json.loads(completed.stdout or "[]")
    except Exception:
        return None


def run_pip_audit(repo_dir: Path) -> dict[str, Any] | None:
    binary = shutil.which("pip-audit")
    if not binary:
        return None
    target = None
    for name in ("requirements.txt", "pyproject.toml"):
        path = repo_dir / name
        if path.exists():
            target = path
            break
    if target is None:
        return None
    try:
        completed = subprocess.run(
            [binary, "-r", str(target), "--format", "json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=45,
        )
        if completed.returncode not in {0, 1}:
            return None
        return json.loads(completed.stdout or "[]")
    except Exception:
        return None


def run_npm_audit(repo_dir: Path) -> dict[str, Any] | None:
    if not (repo_dir / "package.json").exists():
        return None
    binary = shutil.which("npm")
    if not binary:
        return None
    try:
        completed = subprocess.run(
            [binary, "audit", "--json", "--package-lock-only"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=45,
        )
        if completed.returncode not in {0, 1}:
            return None
        return json.loads(completed.stdout or "{}")
    except Exception:
        return None


def run_hadolint(repo_dir: Path) -> dict[str, Any] | None:
    dockerfile = repo_dir / "Dockerfile"
    if not dockerfile.exists():
        return None
    binary = shutil.which("hadolint")
    if not binary:
        return None
    try:
        completed = subprocess.run(
            [binary, str(dockerfile), "-f", "json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if completed.returncode not in {0, 1}:
            return None
        return json.loads(completed.stdout or "[]")
    except Exception:
        return None


def run_interrogate(repo_dir: Path) -> dict[str, Any] | None:
    binary = shutil.which("interrogate")
    if not binary:
        return None
    try:
        completed = subprocess.run(
            [binary, str(repo_dir), "-f", "json", "-q"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if completed.returncode not in {0, 1}:
            return None
        return json.loads(completed.stdout or "{}")
    except Exception:
        return None
