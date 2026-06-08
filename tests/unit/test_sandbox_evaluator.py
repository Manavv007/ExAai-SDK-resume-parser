"""Cloud Run evaluator package tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agent.sandbox.base import SandboxCommand
from agent.sandbox.evaluator.command_runner import run_sandbox_command
from agent.sandbox.evaluator.detector import build_command_plan, detect_project
from agent.sandbox.evaluator.main import evaluate_repository
from agent.sandbox.evaluator.report_writer import _parse_gcs_uri, write_report
from agent.sandbox.models import CommandResult, RepoExecutionReport


def test_detect_project_and_command_plan_for_python(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text(
        "requests==2.31.0\npytest==8.3.0\n",
        encoding="utf-8",
    )
    (tmp_path / "app").mkdir()
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("Demo\n", encoding="utf-8")

    stack, quality, risk_flags, repo_profile, findings = detect_project(tmp_path)
    commands = build_command_plan(tmp_path)

    assert stack == ["python"]
    assert quality["has_tests"] is True
    assert quality["has_docs"] is True
    assert risk_flags == []
    assert repo_profile["project_shape"] == "python_project"
    assert "tests" in repo_profile["test_dirs"]
    assert repo_profile["dependency_health"]["dependency_count"] == 2
    assert repo_profile["dependency_health"]["pinned_versions"] is True
    assert repo_profile["dependency_health"]["outdated_dependencies"] is None
    assert repo_profile["architecture"]["layers"] == ["pipeline", "ui"]
    assert repo_profile["architecture"]["separation_of_concerns"] is True
    assert any(finding["category"] == "tests" for finding in findings)
    assert any(finding["category"] == "dependencies" for finding in findings)
    assert commands == [
        SandboxCommand(step="install", command='python -m pip install -e ".[dev]"'),
        SandboxCommand(step="build", command="python -m compileall -q ."),
        SandboxCommand(step="test", command="python -m pytest -q"),
    ]


def test_detect_project_and_command_plan_for_node(tmp_path: Path) -> None:
    package_json = {
        "scripts": {
            "build": "vite build",
            "test": "vitest run",
            "postinstall": "node scripts/setup.js",
        }
    }
    (tmp_path / "package.json").write_text(json.dumps(package_json), encoding="utf-8")
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")

    stack, quality, risk_flags, repo_profile, findings = detect_project(tmp_path)
    commands = build_command_plan(tmp_path)

    assert stack == ["node"]
    assert "package.json install lifecycle script present" in risk_flags
    assert quality["dependency_files"] == ["package.json", "package-lock.json"]
    assert repo_profile["project_shape"] == "application"
    assert repo_profile["dependency_health"]["dependency_count"] == 0
    assert any("stack markers" in finding["title"].lower() for finding in findings)
    assert commands == [
        SandboxCommand(step="install", command="npm ci"),
        SandboxCommand(step="build", command="npm run build"),
        SandboxCommand(step="test", command="npm test -- --watch=false"),
    ]


def test_detect_project_infers_node_dependency_health_and_architecture(tmp_path: Path) -> None:
    package_json = {
        "dependencies": {
            "react": "^18.3.0",
            "express": "4.19.2",
        },
        "devDependencies": {
            "vitest": "~2.0.0",
        },
    }
    (tmp_path / "package.json").write_text(json.dumps(package_json), encoding="utf-8")
    (tmp_path / "frontend").mkdir()
    (tmp_path / "services").mkdir()
    (tmp_path / "__tests__").mkdir()

    _, _, _, repo_profile, findings = detect_project(tmp_path)

    assert repo_profile["dependency_health"]["dependency_count"] == 3
    assert repo_profile["dependency_health"]["pinned_versions"] is False
    assert repo_profile["dependency_health"]["outdated_dependencies"] is None
    assert repo_profile["architecture"]["layers"] == ["services", "ui"]
    assert repo_profile["architecture"]["separation_of_concerns"] is True
    assert any(
        finding["title"] == "Dependency hygiene was inferred from manifest files."
        for finding in findings
    )


def test_run_sandbox_command_captures_success(tmp_path: Path) -> None:
    command = SandboxCommand(
        step="inspect",
        command=f'"{sys.executable}" -c "print(123)"',
    )

    result = run_sandbox_command(command, cwd=tmp_path, timeout_seconds=10)

    assert result.ok is True
    assert result.exit_code == 0
    assert "123" in result.stdout
    assert result.failure_type is None


def test_run_sandbox_command_classifies_failures(tmp_path: Path) -> None:
    command = SandboxCommand(
        step="test",
        command=f'"{sys.executable}" -c "raise SystemExit(1)"',
    )

    result = run_sandbox_command(command, cwd=tmp_path, timeout_seconds=10)

    assert result.ok is False
    assert result.failure_type == "test_failure"


def test_write_report_to_local_file(tmp_path: Path) -> None:
    report = RepoExecutionReport(
        repo="owner/project",
        url="https://github.com/owner/project",
        clone_ok=True,
        commands=[CommandResult(step="clone", command="git clone", ok=True)],
    )
    output = tmp_path / "reports" / "report.json"

    write_report(report, str(output))

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["repo"] == "owner/project"
    assert data["provider"] == "cloud_run"


def test_parse_gcs_uri() -> None:
    assert _parse_gcs_uri("gs://bucket/path/report.json") == ("bucket", "path/report.json")


def test_evaluate_repository_runs_detected_commands(
    monkeypatch,
) -> None:
    def fake_clone(repo_url: str, repo_dir: Path, *, timeout_seconds: int) -> CommandResult:
        repo_dir.mkdir(parents=True)
        (repo_dir / "requirements.txt").write_text("pytest\n", encoding="utf-8")
        (repo_dir / "tests").mkdir()
        (repo_dir / "tests" / "test_app.py").write_text("def test_ok(): pass\n", encoding="utf-8")
        return CommandResult(step="clone", command="git clone", ok=True, exit_code=0)

    def fake_run(command: SandboxCommand, *, cwd: Path, timeout_seconds: int) -> CommandResult:
        return CommandResult(step=command.step, command=command.command, ok=True, exit_code=0)

    monkeypatch.setattr("agent.sandbox.evaluator.main._clone_repo", fake_clone)
    monkeypatch.setattr("agent.sandbox.evaluator.main.run_sandbox_command", fake_run)

    report = evaluate_repository(
        repo_url="https://github.com/owner/project",
        repo_name="owner/project",
        timeout_seconds=30,
    )

    assert report.clone_ok is True
    assert report.detected_stack == ["python"]
    assert report.install_ok is True
    assert report.build_ok is True
    assert report.tests_ok is True
    assert report.quality_signals["has_tests"] is True
    assert report.provider == "cloud_run"
    assert report.repo_profile["project_shape"] == "application"
    assert report.confidence == "high"
    assert report.findings
    assert (
        "structured" in report.overall_assessment.lower()
        or "completed cleanly" in report.overall_assessment.lower()
    )
