"""Clone-and-profile repository detection and compatibility helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.sandbox.base import SandboxCommand
from agent.sandbox.evaluator.filesystem_scanner import (
    collect_sample_files as _collect_sample_files,
)
from agent.sandbox.evaluator.filesystem_scanner import (
    collect_source_files,
    exists,
    has_docker_config,
    has_glob,
    has_readme,
    read_package_json,
)
from agent.sandbox.evaluator.git_local_analyzer import (
    calculate_git_metrics as _calculate_git_metrics,
)
from agent.sandbox.evaluator.python_code_analyzer import analyze_python_code
from agent.sandbox.evaluator.repo_profiler import profile_repository
from agent.sandbox.evaluator.secret_scanner import calculate_secret_hits as _calculate_secrets
from agent.sandbox.evaluator.tree_sitter_analyzer import analyze_non_python_code


def detect_project(
    repo_dir: Path,
    *,
    focus_spec: dict[str, Any] | None = None,
) -> tuple[
    list[str],
    dict[str, bool | int | str | list[str]],
    list[str],
    dict[str, Any],
    list[dict[str, str]],
]:
    """Return stack names, quality signals, risk flags, repo profile, and findings."""
    stack: list[str] = []
    risk_flags: list[str] = []
    dependency_files: list[str] = []

    if exists(repo_dir, "pyproject.toml", "requirements.txt", "setup.py"):
        stack.append("python")
    if exists(repo_dir, "package.json"):
        stack.append("node")
    if exists(repo_dir, "go.mod"):
        stack.append("go")
    if exists(repo_dir, "Cargo.toml"):
        stack.append("rust")
    if exists(repo_dir, "pom.xml", "build.gradle", "build.gradle.kts"):
        stack.append("java")

    for name in (
        "pyproject.toml",
        "requirements.txt",
        "package.json",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "go.mod",
        "Cargo.toml",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
    ):
        if (repo_dir / name).exists():
            dependency_files.append(name)

    package_data = read_package_json(repo_dir)
    scripts = package_data.get("scripts") if isinstance(package_data.get("scripts"), dict) else {}
    if scripts and any(name in scripts for name in ("preinstall", "install", "postinstall")):
        risk_flags.append("package.json install lifecycle script present")

    has_tests = has_glob(
        repo_dir,
        "tests/**",
        "test/**",
        "**/*_test.py",
        "**/*.test.js",
        "**/*.test.ts",
        "**/*.spec.js",
        "**/*.spec.ts",
        "**/*Test.java",
    )
    has_ci = has_glob(repo_dir, ".github/workflows/*") or exists(
        repo_dir,
        ".gitlab-ci.yml",
        ".travis.yml",
        "circle.yml",
    )
    has_docs = has_readme(repo_dir) or has_glob(repo_dir, "docs/**")
    has_docker = has_docker_config(repo_dir)
    file_count = sum(1 for path in repo_dir.rglob("*") if path.is_file())

    repo_profile, findings = profile_repository(repo_dir, stack, focus_spec=focus_spec)
    return (
        stack,
        {
            "has_tests": has_tests,
            "has_ci": has_ci,
            "has_docs": has_docs,
            "has_docker": has_docker,
            "file_count": file_count,
            "dependency_files": dependency_files,
        },
        risk_flags,
        repo_profile,
        findings,
    )


def build_command_plan(repo_dir: Path) -> list[SandboxCommand]:
    """Profile-only mode: keep command plan empty unless explicitly overridden."""
    _ = repo_dir
    return []


def _calculate_code_metrics(repo_dir: Path) -> dict[str, Any]:
    stack = []
    if exists(repo_dir, "pyproject.toml", "requirements.txt", "setup.py") or any(
        path.suffix.lower() == ".py" for path in collect_source_files(repo_dir, include_tests=True)
    ):
        stack.append("python")
    python_metrics = analyze_python_code(repo_dir) if "python" in stack else {}
    other_metrics = analyze_non_python_code(repo_dir)
    if python_metrics and python_metrics.get("avg_cyclomatic_complexity") is not None:
        return {
            "avg_cyclomatic_complexity": python_metrics["avg_cyclomatic_complexity"],
            "avg_function_length": python_metrics["avg_function_length"],
            "type_annotation_ratio": python_metrics["type_annotation_ratio"],
            "error_handling_density": python_metrics["error_handling_density"],
            "todo_fixme_density": 0.0,
            "lint_violations_per_kloc": other_metrics.get("lint_violations_per_kloc"),
        }
    return {
        "avg_cyclomatic_complexity": other_metrics.get("avg_cyclomatic_complexity"),
        "avg_function_length": other_metrics.get("avg_function_length"),
        "type_annotation_ratio": other_metrics.get("type_annotation_ratio"),
        "error_handling_density": other_metrics.get("error_handling_density"),
        "todo_fixme_density": 0.0,
        "lint_violations_per_kloc": other_metrics.get("lint_violations_per_kloc"),
    }


__all__ = [
    "build_command_plan",
    "detect_project",
    "_calculate_code_metrics",
    "_calculate_git_metrics",
    "_calculate_secrets",
    "_collect_sample_files",
]
