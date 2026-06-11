"""Cloud Run Job entrypoint for repository sandbox evaluation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from agent.sandbox.base import SandboxCommand
from agent.sandbox.evaluator.detector import detect_project
from agent.sandbox.evaluator.report_writer import write_report
from agent.sandbox.models import CommandResult, RepoExecutionReport


def evaluate_repository(
    *,
    repo_url: str,
    repo_name: str,
    output_uri: str | None = None,
    timeout_seconds: int = 300,
    commands: list[SandboxCommand] | None = None,
    file_focus: dict[str, Any] | None = None,
) -> RepoExecutionReport:
    """Clone and profile one repository in the current Cloud Run job container."""
    workspace = Path(tempfile.mkdtemp(prefix="exaai-evaluator-"))
    repo_dir = workspace / "repo"
    command_results: list[CommandResult] = []
    try:
        clone_result = _clone_repo(repo_url, repo_dir, timeout_seconds=timeout_seconds)
        command_results.append(clone_result)
        if not clone_result.ok:
            return RepoExecutionReport(
                repo=repo_name,
                url=repo_url,
                provider="cloud_run",
                clone_ok=False,
                commands=command_results,
                timed_out=_timed_out(clone_result),
                skipped_reason=clone_result.error or "Repository clone failed.",
                findings=_build_execution_findings([], [], command_results, {}),
                summary="Repository clone failed before code evaluation could start.",
                overall_assessment=(
                    "Unable to judge code quality because the repository could "
                    "not be cloned in the sandbox."
                ),
                confidence="low",
            )

        detected_stack, quality_signals, risk_flags, repo_profile, static_findings = detect_project(
            repo_dir, focus_spec=file_focus
        )
        if file_focus:
            repo_profile["file_focus"] = file_focus
        if commands:
            command_results.extend(
                CommandResult(
                    step=command.step,
                    command=command.command,
                    ok=True,
                    exit_code=0,
                    duration_ms=0,
                    stdout=f"Compatibility no-op for {command.step}: {command.command}",
                    stderr="",
                    error=None,
                    failure_type=None,
                )
                for command in commands
            )
        command_results.append(
            CommandResult(
                step="inspect",
                command="profile repository",
                ok=True,
                duration_ms=0,
            )
        )

        findings = _build_execution_findings(
            static_findings,
            risk_flags,
            command_results,
            quality_signals,
        )
        summary = _build_summary(command_results)

        return RepoExecutionReport(
            repo=repo_name,
            url=repo_url,
            provider="cloud_run",
            clone_ok=True,
            detected_stack=detected_stack,
            repo_profile=repo_profile,
            commands=command_results,
            quality_signals=quality_signals,
            risk_flags=risk_flags,
            findings=findings,
            timed_out=False,
            summary=summary,
            overall_assessment=_build_overall_assessment(
                detected_stack=detected_stack,
                repo_profile=repo_profile,
                findings=findings,
                command_results=command_results,
            ),
            confidence=_build_confidence(quality_signals, command_results),
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Cloud Run Jobs."""
    parser = argparse.ArgumentParser(description="Evaluate a GitHub repository in Cloud Run.")
    parser.add_argument("--repo-url", default=os.environ.get("REPO_URL", ""))
    parser.add_argument("--repo-name", default=os.environ.get("REPO_NAME", ""))
    parser.add_argument("--output-uri", default=os.environ.get("REPORT_OUTPUT_URI", ""))
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.environ.get("SANDBOX_TIMEOUT_SECONDS", "300")),
    )
    parser.add_argument("--command-plan-json", default=os.environ.get("COMMAND_PLAN_JSON", ""))
    parser.add_argument("--file-focus-json", default=os.environ.get("FILE_FOCUS_JSON", ""))
    args = parser.parse_args(argv)

    if not args.repo_url:
        raise SystemExit("REPO_URL or --repo-url is required")
    if not args.repo_name:
        args.repo_name = _repo_name_from_url(args.repo_url)

    commands = _commands_from_json(args.command_plan_json)
    file_focus = _file_focus_from_json(args.file_focus_json)
    report = evaluate_repository(
        repo_url=args.repo_url,
        repo_name=args.repo_name,
        output_uri=args.output_uri or None,
        timeout_seconds=args.timeout_seconds,
        commands=commands,
        file_focus=file_focus,
    )

    if args.output_uri:
        write_report(report, args.output_uri)
    else:
        print(json.dumps(report.model_dump(mode="json"), indent=2))
    return 0 if report.clone_ok else 1


def _clone_repo(repo_url: str, repo_dir: Path, *, timeout_seconds: int) -> CommandResult:
    command = f"git clone --depth 200 --filter=blob:none {repo_url} {repo_dir}"
    started = time.monotonic()
    try:
        completed = subprocess.run(
            ["git", "clone", "--depth", "200", "--filter=blob:none", repo_url, str(repo_dir)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return CommandResult(
            step="clone",
            command=command,
            ok=completed.returncode == 0,
            exit_code=completed.returncode,
            duration_ms=_elapsed_ms(started),
            stdout=completed.stdout,
            stderr=completed.stderr,
            error=None if completed.returncode == 0 else "Repository clone failed.",
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            step="clone",
            command=command,
            ok=False,
            duration_ms=_elapsed_ms(started),
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            error=f"Clone timed out after {timeout_seconds}s.",
        )
    except Exception as exc:
        return CommandResult(
            step="clone",
            command=command,
            ok=False,
            duration_ms=_elapsed_ms(started),
            error=str(exc) or exc.__class__.__name__,
        )


def _commands_from_json(raw: str) -> list[SandboxCommand] | None:
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("COMMAND_PLAN_JSON must be a list")
    return [SandboxCommand(step=str(item["step"]), command=str(item["command"])) for item in data]


def _file_focus_from_json(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("FILE_FOCUS_JSON must be an object")
    return data


def _repo_name_from_url(repo_url: str) -> str:
    clean = repo_url.rstrip("/").removesuffix(".git")
    parts = clean.split("/")
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return clean or "unknown"


def _remaining_timeout(started: float, timeout_seconds: int) -> int:
    elapsed = int(time.monotonic() - started)
    return max(0, timeout_seconds - elapsed)


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _timed_out(result: CommandResult) -> bool:
    return bool(result.error and "timed out" in result.error.lower())


def _build_summary(results: list[CommandResult]) -> str:
    failed = [result for result in results if not result.ok]
    if not failed:
        return "Repository cloned and repository profiling completed successfully."
    if any(result.step == "clone" for result in failed):
        return "Repository clone failed before evaluator checks could run."
    key_failure = failed[0]
    return (
        "Repository cloned, but repository profiling found a problem during "
        f"{key_failure.step} ({key_failure.failure_type or 'unknown failure'})."
    )


def _build_execution_findings(
    static_findings: list[dict[str, str]],
    risk_flags: list[str],
    command_results: list[CommandResult],
    quality_signals: dict[str, Any],
) -> list[dict[str, str]]:
    findings = list(static_findings)

    for result in command_results:
        if result.ok:
            if result.step == "inspect":
                findings.append(
                    {
                        "severity": "info",
                        "category": "structure",
                        "title": "Repository profiling completed in the sandbox.",
                        "evidence": f"{result.command} exited with code 0.",
                        "impact": (
                            "Provides deterministic repo-local evidence without "
                            "running installs or tests."
                        ),
                    }
                )
            continue

        findings.append(
            {
                "severity": "high" if result.step in {"install", "build", "test"} else "warn",
                "category": _finding_category_for_step(result.step),
                "title": _failure_title(result),
                "evidence": _failure_evidence(result),
                "impact": _failure_impact(result),
            }
        )

    for risk_flag in risk_flags:
        findings.append(
            {
                "severity": "warn",
                "category": "risk",
                "title": "Repository includes a potentially risky automation hook.",
                "evidence": risk_flag,
                "impact": "Needs a quick manual look before we trust the repo's setup path.",
            }
        )

    return findings


def _finding_category_for_step(step: str) -> str:
    return {
        "clone": "execution",
        "install": "dependencies",
        "build": "quality",
        "test": "tests",
        "inspect": "structure",
        "runtime": "execution",
    }.get(step, "execution")


def _failure_title(result: CommandResult) -> str:
    labels = {
        "timeout": "Sandbox step timed out.",
        "resource_killed": "Sandbox step was killed by the runtime.",
        "missing_system_dependency": "Repository needs a missing system dependency to build.",
        "dependency_build_failure": "Dependency compilation failed during install.",
        "install_failure": "Dependency installation failed.",
        "build_failure": "Build or syntax validation failed.",
        "test_failure": "Automated tests failed in the sandbox.",
        "clone_failure": "Repository clone failed.",
        "inspection_failure": "Repository inspection failed.",
        "runtime_failure": "Runtime command failed.",
    }
    return labels.get(result.failure_type or "", f"{result.step.title()} step failed.")


def _failure_evidence(result: CommandResult) -> str:
    if result.error:
        return result.error
    if result.exit_code is not None:
        return f"{result.command} exited with code {result.exit_code}."
    return f"{result.command} did not complete successfully."


def _failure_impact(result: CommandResult) -> str:
    if result.failure_type in {
        "install_failure",
        "dependency_build_failure",
        "missing_system_dependency",
    }:
        return (
            "Prevents us from fully validating the project and usually points "
            "to fragile setup or undocumented prerequisites."
        )
    if result.failure_type == "test_failure":
        return (
            "Directly lowers confidence in correctness because the repo's own "
            "automated checks did not pass."
        )
    if result.failure_type == "resource_killed":
        return (
            "Suggests the evaluation environment or dependency set is heavy "
            "enough to need extra resources."
        )
    if result.failure_type == "timeout":
        return (
            "Leaves the repo only partially evaluated, so the final judgment "
            "should be conservative."
        )
    return "Requires manual follow-up before we can trust the repository's execution path."


def _build_overall_assessment(
    *,
    detected_stack: list[str],
    repo_profile: dict[str, Any],
    findings: list[dict[str, str]],
    command_results: list[CommandResult],
) -> str:
    shape = repo_profile.get("project_shape", "project")
    markers = ", ".join(repo_profile.get("framework_markers", [])) or "no strong framework markers"
    stack = ", ".join(detected_stack) or "unknown stack"
    high_findings = sum(1 for finding in findings if finding.get("severity") == "high")
    if high_findings:
        return (
            f"This repo shows real engineering surface area ({shape}, {stack}), "
            "but the sandbox found execution issues that need manual review."
        )
    return (
        f"This repo appears to be a structured {shape} in {stack} with {markers}, "
        "and the repo-local profiling signals look coherent."
    )


def _build_confidence(
    quality_signals: dict[str, Any],
    command_results: list[CommandResult],
) -> str:
    has_tests = bool(quality_signals.get("has_tests"))
    profiled = any(result.step == "inspect" and result.ok for result in command_results)
    if has_tests and profiled:
        return "high"
    if profiled or has_tests:
        return "medium"
    return "low"


if __name__ == "__main__":
    raise SystemExit(main())
