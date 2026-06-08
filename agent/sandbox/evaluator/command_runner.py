"""Subprocess command runner for the Cloud Run evaluator."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from agent.sandbox.base import SandboxCommand
from agent.sandbox.models import CommandResult


def run_sandbox_command(
    command: SandboxCommand,
    *,
    cwd: Path,
    timeout_seconds: int,
) -> CommandResult:
    """Run one evaluator command with bounded output and timeout."""
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command.command,
            cwd=str(cwd),
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return CommandResult(
            step=command.step,
            command=command.command,
            ok=completed.returncode == 0,
            exit_code=completed.returncode,
            duration_ms=_elapsed_ms(started),
            stdout=completed.stdout,
            stderr=completed.stderr,
            failure_type=_classify_failure(
                step=command.step,
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            ),
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            step=command.step,
            command=command.command,
            ok=False,
            duration_ms=_elapsed_ms(started),
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            error=f"Command timed out after {timeout_seconds}s.",
            failure_type="timeout",
        )
    except Exception as exc:
        return CommandResult(
            step=command.step,
            command=command.command,
            ok=False,
            duration_ms=_elapsed_ms(started),
            error=str(exc) or exc.__class__.__name__,
            failure_type=_default_failure_type(command.step),
        )


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _classify_failure(
    *,
    step: str,
    exit_code: int,
    stdout: str,
    stderr: str,
) -> str | None:
    if exit_code == 0:
        return None

    output = f"{stdout}\n{stderr}".lower()
    if exit_code == 137:
        return "resource_killed"
    if "no such file or directory" in output and ("g++" in output or "gcc" in output):
        return "missing_system_dependency"
    if "failed building wheel" in output or "subprocess-exited-with-error" in output:
        return "dependency_build_failure" if step == "install" else _default_failure_type(step)
    return _default_failure_type(step)


def _default_failure_type(step: str) -> str:
    return {
        "clone": "clone_failure",
        "install": "install_failure",
        "build": "build_failure",
        "test": "test_failure",
        "runtime": "runtime_failure",
        "inspect": "inspection_failure",
    }.get(step, "runtime_failure")
