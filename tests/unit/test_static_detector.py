"""Unit tests for static code and repository analysis inside detector.py."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from agent.sandbox.evaluator.detector import (
    _calculate_code_metrics,
    _calculate_git_metrics,
    _calculate_secrets,
    _collect_sample_files,
    detect_project,
)


def _init_git_repo(tmp_path: Path) -> None:
    # Initialize a git repository for testing git metrics
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test Author 1"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "author1@test.com"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    # Configure safety to allow committing empty/initial commits
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )


def test_static_git_metrics(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)

    # First commit
    (tmp_path / "file1.txt").write_text("Hello World", encoding="utf-8")
    subprocess.run(["git", "add", "file1.txt"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "First commit"], cwd=str(tmp_path), check=True, capture_output=True
    )

    # Second commit by a different author
    (tmp_path / "file2.txt").write_text("Hello World 2", encoding="utf-8")
    subprocess.run(["git", "add", "file2.txt"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--author=Test Author 2 <author2@test.com>", "-m", "Second commit"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )

    metrics = _calculate_git_metrics(tmp_path)
    assert metrics["commit_count"] == 2
    assert metrics["unique_authors"] == 2
    assert metrics["sole_author"] is False
    assert metrics["top_author_commit_share"] == 0.5
    assert metrics["days_since_last_commit"] >= 0


def test_static_code_metrics_python_ast(tmp_path: Path) -> None:
    python_code = """
def func_without_annotation(a, b):
    if a > b:
        return a
    else:
        for i in range(b):
            if i == 5:
                break
        return b

def func_with_annotation(x: int, y: int) -> int:
    try:
        z = x + y
        if z == 10:
            return z
    except Exception as e:
        pass
    return 0
"""
    (tmp_path / "main.py").write_text(python_code, encoding="utf-8")
    metrics = _calculate_code_metrics(tmp_path)

    # Let's verify cyclomatic complexity:
    # func_without_annotation base (1) + If (1) + For (1) + If (1) = 4 complexity points.
    # func_with_annotation base (1) + If (1) + ExceptHandler (1) = 3 complexity points.
    # Total functions = 2. Total complexity = 7.
    # avg_cyclomatic_complexity = 7 / 2 = 3.5.
    assert metrics["avg_cyclomatic_complexity"] == 3.5

    # Type annotation ratio:
    # func_with_annotation has annotations. func_without_annotation doesn't.
    # Ratio = 1 / 2 = 0.5.
    assert metrics["type_annotation_ratio"] == 0.5


def test_static_code_metrics_non_python(tmp_path: Path) -> None:
    # TS file should count for type annotation ratio
    typescript_code = """
function greet(name: string): string {
    if (name) {
        return "Hello " + name;
    }
    return "Hello Guest";
}
"""
    (tmp_path / "greet.ts").write_text(typescript_code, encoding="utf-8")
    metrics = _calculate_code_metrics(tmp_path)

    # js_ts_files = 1, ts_files = 1. type_annotation_ratio = 1.0.
    assert metrics["type_annotation_ratio"] == 1.0


def test_static_code_densities_and_lints(tmp_path: Path) -> None:
    # 5 LOC total
    # Error handling keywords: 1 ("catch")
    # TODOs/FIXMEs: 1 ("TODO")
    # Lint violation 1: line exceeds 120 chars (130 chars of 'x')
    # Lint violation 2: trailing spaces
    code = (
        "function test() {\n"
        "    try {\n"
        '        console.log("x" * 100); // TODO: check if this exceeds 120 chars '
        "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
        "    } catch (e) {}   \n"
        "}\n"
    )
    (tmp_path / "main.js").write_text(code, encoding="utf-8")
    metrics = _calculate_code_metrics(tmp_path)

    # total_loc = 6 (including last empty line if content.splitlines() has it, or 5)
    # let's assert density ranges or values
    assert metrics["error_handling_density"] > 0
    assert metrics["todo_fixme_density"] > 0
    assert metrics["lint_violations_per_kloc"] > 0


def test_static_secrets(tmp_path: Path) -> None:
    secret_code = """
# AWS Access Key ID
AWS_KEY = "AKIA0000000000000000"

# Slack Webhook URL
SLACK = "https://hooks.slack.example/services/T00000000/B00000000/000000000000000000000000"
"""
    (tmp_path / "secrets.py").write_text(secret_code, encoding="utf-8")

    # Environment file secret
    (tmp_path / ".env").write_text('DB_PASSWORD="super-secret-password-12345"\n', encoding="utf-8")

    hits = _calculate_secrets(tmp_path)
    # AWS Key (1) + Slack Webhook (1) + DB Password assignment (1) = 3 hits
    assert hits >= 3


def test_collect_sample_files(tmp_path: Path) -> None:
    (tmp_path / "small.py").write_text("print('small')", encoding="utf-8")
    (tmp_path / "large.py").write_text("print('large')\n" * 10, encoding="utf-8")
    (tmp_path / "medium.py").write_text("print('medium')\n" * 5, encoding="utf-8")

    samples = _collect_sample_files(tmp_path)
    assert len(samples) == 3
    # Sorted by size: large.py (most lines/bytes), then medium.py, then small.py
    assert samples[0]["path"] == "large.py"
    assert samples[1]["path"] == "medium.py"
    assert samples[2]["path"] == "small.py"
    assert "large" in samples[0]["content_preview"]


def test_detect_project_merges_all_metrics(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (tmp_path / "Dockerfile").write_text("FROM python:3.9\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Test Project\n", encoding="utf-8")
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text("on: push\n", encoding="utf-8")
    (tmp_path / "file.py").write_text("def main():\n    pass\n", encoding="utf-8")

    # Commit the files so we have a commit history
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial"], cwd=str(tmp_path), check=True, capture_output=True
    )

    stack, quality, risk_flags, repo_profile, findings = detect_project(tmp_path)

    # Verify standard values in return signature
    assert "python" in stack
    assert quality["has_docs"] is True
    assert quality["has_ci"] is True
    assert quality["has_tests"] is False

    # Verify merged repo_profile parameters
    assert repo_profile["commit_count"] == 1
    assert repo_profile["unique_authors"] == 1
    assert repo_profile["days_since_last_commit"] >= 0
    assert repo_profile["has_ci"] is True
    assert repo_profile["has_tests"] is False
    assert repo_profile["has_docs"] is True
    assert repo_profile["has_dockerfile"] is True
    assert repo_profile["top_author_commit_share"] == 1.0
    assert repo_profile["sole_author"] is True
    assert repo_profile["avg_cyclomatic_complexity"] == 1.0  # main() has complexity 1
    assert repo_profile["type_annotation_ratio"] == 0.0
    assert repo_profile["error_handling_density"] == 0.0
    assert repo_profile["todo_fixme_density"] == 0.0
    assert repo_profile["lint_violations_per_kloc"] == 0.0
    assert repo_profile["secret_pattern_hits"] == 0
    assert len(repo_profile["sample_files"]) == 1
    assert repo_profile["sample_files"][0]["path"] == "file.py"


def test_detect_project_react_doctor_mocked(tmp_path: Path, monkeypatch) -> None:
    _init_git_repo(tmp_path)
    package_json = {"dependencies": {"react": "^18.2.0"}}
    (tmp_path / "package.json").write_text(json.dumps(package_json), encoding="utf-8")
    (tmp_path / "main.jsx").write_text("const App = () => <div>Hello</div>;\n", encoding="utf-8")

    mock_run_called = []

    class MockCompletedProcess:
        def __init__(self, returncode, stdout, stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    original_run = subprocess.run

    def fake_run(args, **kwargs):
        if args[0] == "git":
            return original_run(args, **kwargs)

        if any("react-doctor" in str(arg) for arg in args):
            mock_run_called.append(args)
            mock_json_out = {
                "score": {"value": 85, "label": "Great"},
                "diagnostics": [
                    {
                        "filePath": "src/App.jsx",
                        "line": 10,
                        "message": "Do not use array index as key",
                        "severity": "warning",
                        "ruleId": "react-doctor/no-array-index-as-key",
                    }
                ],
            }
            return MockCompletedProcess(0, json.dumps(mock_json_out))

        return MockCompletedProcess(1, "")

    monkeypatch.setattr("subprocess.run", fake_run)

    original_run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True)
    original_run(
        ["git", "commit", "-m", "Initial React"], cwd=str(tmp_path), check=True, capture_output=True
    )

    stack, quality, risk_flags, repo_profile, findings = detect_project(tmp_path)

    assert "node" in stack
    assert "react" in repo_profile["framework_markers"]
    assert len(mock_run_called) == 1

    assert repo_profile["react_doctor_score"] == 85
    react_finding = next((f for f in findings if "React Doctor" in f["title"]), None)
    assert react_finding is not None
    assert react_finding["severity"] == "warn"
    assert "no-array-index-as-key" in react_finding["evidence"]
    assert "App.jsx" in react_finding["evidence"]
    assert repo_profile["lint_violations_per_kloc"] == 1000.0
