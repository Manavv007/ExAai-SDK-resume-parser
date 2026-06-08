"""Sandbox result model contracts."""

from __future__ import annotations

from agent.sandbox.models import CommandResult, RepoExecutionReport, truncate_output


def test_truncate_output_bounds_text() -> None:
    text = "x" * 5000

    truncated = truncate_output(text, max_chars=100)

    assert len(truncated) <= 100
    assert "truncated" in truncated


def test_command_result_truncates_logs_and_compacts() -> None:
    result = CommandResult(
        step="test",
        command=" pytest -q ",
        ok=False,
        exit_code=1,
        duration_ms=123,
        stdout="passed\n",
        stderr="failure details " * 400,
        failure_type="test_failure",
    )

    compact = result.compact()

    assert result.command == "pytest -q"
    assert len(result.stderr) <= 4000
    assert compact["step"] == "test"
    assert compact["ok"] is False
    assert compact["exit_code"] == 1
    assert compact["failure_type"] == "test_failure"
    assert "output_preview" in compact


def test_repo_execution_report_step_helpers_and_compact() -> None:
    report = RepoExecutionReport(
        repo="owner/project",
        url="https://github.com/owner/project",
        clone_ok=True,
        detected_stack=["python", "fastapi"],
        commands=[
            CommandResult(step="clone", command="git clone", ok=True, exit_code=0),
            CommandResult(step="install", command="pip install -r requirements.txt", ok=True),
            CommandResult(step="test", command="pytest -q", ok=False, exit_code=1),
        ],
        quality_signals={"has_tests": True, "dependency_files": ["requirements.txt"]},
        risk_flags=["tests require external database"],
        repo_profile={"project_shape": "service", "framework_markers": ["fastapi"]},
        findings=[
            {
                "severity": "warn",
                "category": "tests",
                "title": "Automated tests failed in the sandbox.",
                "evidence": "pytest -q exited with code 1.",
                "impact": "Directly lowers confidence in correctness.",
            }
        ],
        summary="Buildable Python service with failing integration-style tests.",
        overall_assessment=(
            "This looks like a real service, but the failing test suite is a quality concern."
        ),
        confidence="high",
    )

    compact = report.compact()

    assert report.install_ok is True
    assert report.build_ok is None
    assert report.tests_ok is False
    assert compact["repo"] == "owner/project"
    assert compact["detected_stack"] == ["python", "fastapi"]
    assert compact["repo_profile"]["project_shape"] == "service"
    assert len(compact["commands"]) == 3
    assert compact["risk_flags"] == ["tests require external database"]
    assert compact["findings"][0]["category"] == "tests"
    assert compact["confidence"] == "high"
