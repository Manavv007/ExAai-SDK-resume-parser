"""Repo-local clone-and-profile orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.sandbox.evaluator.external_tools import (
    run_checkov,
    run_hadolint,
    run_interrogate,
    run_npm_audit,
    run_pip_audit,
    run_scc,
    run_semgrep,
    run_trivy_fs,
)
from agent.sandbox.evaluator.filesystem_scanner import (
    collect_sample_files,
    dependency_health,
    detect_framework_markers,
    exists,
    find_entrypoints,
    has_docker_compose,
    has_docker_config,
    has_dockerfile,
    has_glob,
    has_readme,
    infer_project_shape,
    infer_repo_type_tags,
    list_dirs,
    list_files,
    read_package_json,
)
from agent.sandbox.evaluator.git_local_analyzer import calculate_git_metrics
from agent.sandbox.evaluator.python_code_analyzer import analyze_python_code
from agent.sandbox.evaluator.secret_scanner import build_security_profile, calculate_secret_hits
from agent.sandbox.evaluator.text_signal_analyzer import (
    build_documentation_profile,
    count_todo_fixme_density,
)
from agent.sandbox.evaluator.top_files import collect_top_files
from agent.sandbox.evaluator.tree_sitter_analyzer import analyze_non_python_code
from agent.tools.repo_focus import is_risk_only_evaluation


def profile_repository(
    repo_dir: Path,
    stack: list[str],
    *,
    focus_spec: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    package_data = read_package_json(repo_dir)
    framework_markers = detect_framework_markers(repo_dir, package_data)
    source_dirs = list_dirs(
        repo_dir,
        ("src", "app", "apps", "lib", "pkg", "server", "backend", "frontend", "ui", "services"),
    )
    test_dirs = list_dirs(repo_dir, ("tests", "test", "__tests__", "spec"))
    config_files = list_files(
        repo_dir,
        (
            "pyproject.toml",
            "requirements.txt",
            "setup.py",
            "package.json",
            "tsconfig.json",
            "Dockerfile",
            ".ruff.toml",
            "ruff.toml",
            "mypy.ini",
            ".pre-commit-config.yaml",
            ".editorconfig",
            ".gitignore",
            "docker-compose.yml",
            "docker-compose.yaml",
        ),
    )
    dep_health = dependency_health(repo_dir, package_data)
    architecture = _architecture_profile(repo_dir, source_dirs, test_dirs, framework_markers)
    project_shape = infer_project_shape(repo_dir, package_data, stack)
    repo_type_tags = infer_repo_type_tags(
        stack=stack,
        framework_markers=framework_markers,
        architecture_layers=architecture["layers"],
        project_shape=project_shape,
        package_data=package_data,
        repo_dir=repo_dir,
    )
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
    git_profile = calculate_git_metrics(repo_dir)
    secret_hits = calculate_secret_hits(repo_dir)
    security_profile = build_security_profile(repo_dir, secret_hits)
    documentation_profile = build_documentation_profile(repo_dir)
    scc_data = run_scc(repo_dir)
    pip_audit_data = run_pip_audit(repo_dir)
    npm_audit_data = run_npm_audit(repo_dir)
    trivy_data = run_trivy_fs(repo_dir)
    semgrep_data = run_semgrep(repo_dir)
    checkov_data = run_checkov(repo_dir)
    hadolint_data = run_hadolint(repo_dir)
    risk_only = is_risk_only_evaluation(focus_spec)
    if risk_only:
        interrogate_data = {}
        sample_files: list[dict[str, Any]] = []
        top_files: list[dict[str, Any]] = []
        code_metrics = {
            "avg_cyclomatic_complexity": None,
            "avg_function_length": None,
            "type_annotation_ratio": None,
            "error_handling_density": None,
            "todo_fixme_density": 0.0,
            "lint_violations_per_kloc": None,
        }
    else:
        interrogate_data = run_interrogate(repo_dir)
        sample_files = collect_sample_files(repo_dir, focus_spec)
        top_files = collect_top_files(
            repo_dir,
            focus_spec,
            max_files=int((focus_spec or {}).get("top_files_count") or 5),
        )
        python_metrics = analyze_python_code(repo_dir) if "python" in stack else {}
        tree_metrics = (
            analyze_non_python_code(repo_dir) if any(lang != "python" for lang in stack) else {}
        )
        code_metrics = _merge_code_metrics(python_metrics, tree_metrics)
        code_metrics["todo_fixme_density"] = count_todo_fixme_density(
            [item["content_preview"] for item in sample_files],
            sum(item["lines"] for item in sample_files),
        )
        code_metrics["lint_violations_per_kloc"] = tree_metrics.get("lint_violations_per_kloc")
    dockerfile_present = has_dockerfile(repo_dir)
    docker_compose_present = has_docker_compose(repo_dir)
    docker_present = has_docker_config(repo_dir)
    repo_profile = {
        "project_shape": project_shape,
        "repo_type_tags": repo_type_tags,
        "framework_markers": framework_markers,
        "entrypoints": find_entrypoints(repo_dir, package_data),
        "source_dirs": source_dirs,
        "test_dirs": test_dirs,
        "config_files": config_files,
        "dependency_health": dep_health,
        "architecture": architecture,
        "git_profile": git_profile,
        "code_metrics": code_metrics,
        "security_profile": security_profile,
        "documentation_profile": documentation_profile,
        "external_tool_signals": {
            "scc": _summarize_scc(scc_data),
            "pip_audit": _summarize_pip_audit(pip_audit_data),
            "npm_audit": _summarize_npm_audit(npm_audit_data),
            "trivy": _summarize_trivy(trivy_data),
            "semgrep": _summarize_semgrep(semgrep_data),
            "checkov": _summarize_checkov(checkov_data),
            "hadolint": _summarize_hadolint(hadolint_data),
            "interrogate": _summarize_interrogate(interrogate_data),
        },
        "commit_count": git_profile["commit_count"],
        "unique_authors": git_profile["unique_authors"],
        "days_since_last_commit": git_profile["days_since_last_commit"],
        "has_ci": has_ci,
        "has_tests": has_tests,
        "has_docs": has_docs,
        "has_dockerfile": dockerfile_present,
        "has_docker_compose": docker_compose_present,
        "has_docker": docker_present,
        "top_author_commit_share": git_profile["top_author_commit_share"],
        "sole_author": git_profile["sole_author"],
        "history_is_shallow": git_profile["history_is_shallow"],
        "avg_cyclomatic_complexity": code_metrics["avg_cyclomatic_complexity"],
        "avg_function_length": code_metrics["avg_function_length"],
        "type_annotation_ratio": code_metrics["type_annotation_ratio"],
        "error_handling_density": code_metrics["error_handling_density"],
        "todo_fixme_density": code_metrics["todo_fixme_density"],
        "lint_violations_per_kloc": code_metrics["lint_violations_per_kloc"],
        "secret_pattern_hits": security_profile["secret_pattern_hits"],
        "sample_files": sample_files,
        "top_files": top_files,
    }
    if risk_only:
        repo_profile["evaluation_mode"] = "risk_only"

    repo_role = str((focus_spec or {}).get("repo_role") or "unknown")
    findings = _build_findings(
        has_tests=has_tests,
        has_ci=has_ci,
        has_docs=has_docs,
        has_docker=docker_present,
        has_dockerfile=dockerfile_present,
        has_docker_compose=docker_compose_present,
        framework_markers=framework_markers,
        dependency_health=dep_health,
        architecture=architecture,
        security_profile=security_profile,
        documentation_profile=documentation_profile,
        scc_data=scc_data,
        pip_audit_data=pip_audit_data,
        npm_audit_data=npm_audit_data,
        trivy_data=trivy_data,
        semgrep_data=semgrep_data,
        checkov_data=checkov_data,
        hadolint_data=hadolint_data,
        interrogate_data=interrogate_data,
        repo_role=repo_role,
    )
    return repo_profile, findings


def _architecture_profile(
    repo_dir: Path,
    source_dirs: list[str],
    test_dirs: list[str],
    framework_markers: list[str],
) -> dict[str, Any]:
    layers: list[str] = []
    directory_candidates = {
        "ui": ("ui", "frontend", "client", "web", "app"),
        "services": ("services", "service", "api", "backend", "server"),
        "guardrails": ("guardrails", "rails", "safety"),
        "pipeline": ("pipeline", "workflows", "jobs"),
        "data": ("data", "models", "schema"),
    }
    lower_dirs = {path.name.lower() for path in repo_dir.rglob("*") if path.is_dir()}
    lower_files = {
        path.name.lower()
        for path in repo_dir.rglob("*")
        if path.is_file() and path.suffix in {".py", ".ts", ".js"}
    }
    combined = lower_dirs | lower_files
    for layer, names in directory_candidates.items():
        if any(name in combined for name in names):
            layers.append(layer)
    if "streamlit" in framework_markers and "ui" not in layers:
        layers.append("ui")
    if (
        "fastapi" in framework_markers or "flask" in framework_markers
    ) and "services" not in layers:
        layers.append("services")
    return {
        "layers": sorted(layers),
        "separation_of_concerns": bool(source_dirs) and bool(test_dirs) and len(layers) >= 2,
        "frontend_backend_split": "ui" in layers and "services" in layers,
    }


def _merge_code_metrics(
    python_metrics: dict[str, Any],
    tree_metrics: dict[str, Any],
) -> dict[str, Any]:
    base = {
        "avg_cyclomatic_complexity": None,
        "avg_function_length": None,
        "type_annotation_ratio": None,
        "error_handling_density": None,
        "todo_fixme_density": 0.0,
        "lint_violations_per_kloc": None,
    }
    if python_metrics and python_metrics.get("avg_cyclomatic_complexity") is not None:
        base.update(python_metrics)
        base["lint_violations_per_kloc"] = None
        return base
    if tree_metrics:
        base.update(tree_metrics)
    return base


def _build_findings(
    *,
    has_tests: bool,
    has_ci: bool,
    has_docs: bool,
    has_docker: bool,
    has_dockerfile: bool,
    has_docker_compose: bool,
    framework_markers: list[str],
    dependency_health: dict[str, Any],
    architecture: dict[str, Any],
    security_profile: dict[str, Any],
    documentation_profile: dict[str, Any],
    scc_data: dict[str, Any] | None,
    pip_audit_data: dict[str, Any] | None,
    npm_audit_data: dict[str, Any] | None,
    trivy_data: dict[str, Any] | None,
    semgrep_data: dict[str, Any] | None,
    checkov_data: dict[str, Any] | None,
    hadolint_data: dict[str, Any] | None,
    interrogate_data: dict[str, Any] | None,
    repo_role: str = "unknown",
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    role = str(repo_role or "unknown").lower()
    penalize_missing_tests = role in ("aligned", "adjacent")

    if has_tests:
        findings.append(
            {
                "severity": "info",
                "category": "tests",
                "title": "Repository includes an automated test surface.",
                "evidence": "Test directories/files detected.",
                "impact": "Gives us a direct way to validate candidate code behavior.",
            }
        )
    elif penalize_missing_tests:
        findings.append(
            {
                "severity": "warn",
                "category": "tests",
                "title": "Repository does not expose an obvious automated test suite.",
                "evidence": "No common test paths were detected.",
                "impact": ("Reduces confidence in correctness for this role-aligned repository."),
            }
        )
    if has_ci:
        findings.append(
            {
                "severity": "info",
                "category": "quality",
                "title": "Repository includes CI configuration.",
                "evidence": "Detected workflow or CI config files in the repository root.",
                "impact": "Suggests a repeatable validation workflow.",
            }
        )
    if has_docs:
        findings.append(
            {
                "severity": "info",
                "category": "quality",
                "title": "Repository includes developer-facing documentation.",
                "evidence": "README or docs directory detected.",
                "impact": "Suggests some attention to maintainability and onboarding.",
            }
        )
    if has_docker:
        evidence_parts: list[str] = []
        if has_dockerfile:
            evidence_parts.append("Dockerfile")
        if has_docker_compose:
            evidence_parts.append("docker-compose")
        findings.append(
            {
                "severity": "info",
                "category": "structure",
                "title": "Repository includes container configuration.",
                "evidence": ", ".join(evidence_parts) or "container config detected",
                "impact": (
                    "Suggests the project can be run or deployed in a reproducible environment."
                ),
            }
        )
    if framework_markers:
        findings.append(
            {
                "severity": "info",
                "category": "structure",
                "title": "Project stack markers were identified.",
                "evidence": f"Framework/tool markers: {', '.join(framework_markers)}",
                "impact": "Helps us identify the repo type and engineering context.",
            }
        )
    if dependency_health["dependency_count"]:
        findings.append(
            {
                "severity": "info" if dependency_health["pinned_versions"] else "warn",
                "category": "dependencies",
                "title": "Dependency hygiene was inferred from manifest files.",
                "evidence": (
                    f"{dependency_health['dependency_count']} dependencies detected; "
                    f"pinned_versions={dependency_health['pinned_versions']}."
                ),
                "impact": "Version pinning improves reproducibility and reduces drift.",
            }
        )
    if architecture["layers"]:
        findings.append(
            {
                "severity": "info",
                "category": "structure",
                "title": "Repository shows identifiable architecture layers.",
                "evidence": f"Detected layers: {', '.join(architecture['layers'])}",
                "impact": "Clear layering usually makes the codebase easier to reason about.",
            }
        )
    if security_profile["secret_pattern_hits"] > 0 or security_profile["has_env_file"]:
        findings.append(
            {
                "severity": "high",
                "category": "risk",
                "title": "Repository shows weak secret hygiene.",
                "evidence": (
                    f"secret_pattern_hits={security_profile['secret_pattern_hits']}, "
                    f"has_env_file={security_profile['has_env_file']}"
                ),
                "impact": (
                    "Committed secrets or unsafe env handling are strong professionalism concerns."
                ),
            }
        )
    elif documentation_profile["has_setup_instructions"]:
        findings.append(
            {
                "severity": "info",
                "category": "quality",
                "title": "README includes setup guidance.",
                "evidence": "Install/setup/getting-started text detected.",
                "impact": "Suggests empathy for other developers using the repo.",
            }
        )
    trivy_summary = _summarize_trivy(trivy_data)
    scc_summary = _summarize_scc(scc_data)
    if scc_summary["code_lines"] is not None:
        findings.append(
            {
                "severity": "info",
                "category": "structure",
                "title": "Source line counting scan was available.",
                "evidence": (
                    f"code_lines={scc_summary['code_lines']}, "
                    f"comment_ratio={scc_summary['comment_ratio']}"
                ),
                "impact": "Fast language and comment statistics help calibrate repo scale.",
            }
        )
    pip_summary = _summarize_pip_audit(pip_audit_data)
    if pip_summary["vulnerability_count"]:
        findings.append(
            {
                "severity": "warn",
                "category": "risk",
                "title": "Python dependency audit reported known vulnerabilities.",
                "evidence": f"pip_audit_vulnerabilities={pip_summary['vulnerability_count']}",
                "impact": "Known package vulnerabilities are a direct maintenance signal.",
            }
        )
    npm_summary = _summarize_npm_audit(npm_audit_data)
    if npm_summary["vulnerability_count"]:
        findings.append(
            {
                "severity": "warn",
                "category": "risk",
                "title": "Node dependency audit reported known vulnerabilities.",
                "evidence": f"npm_audit_vulnerabilities={npm_summary['vulnerability_count']}",
                "impact": "Detects dependency risk without fully installing the repo.",
            }
        )
    if trivy_summary["vulnerability_count"]:
        findings.append(
            {
                "severity": "warn",
                "category": "risk",
                "title": "Dependency vulnerability scan reported known CVEs.",
                "evidence": f"trivy_vulnerability_count={trivy_summary['vulnerability_count']}",
                "impact": "Known CVEs are a useful maturity and maintenance signal.",
            }
        )
    semgrep_summary = _summarize_semgrep(semgrep_data)
    if semgrep_summary["result_count"]:
        findings.append(
            {
                "severity": "warn",
                "category": "risk",
                "title": "Semgrep reported potential secret or security issues.",
                "evidence": f"semgrep_results={semgrep_summary['result_count']}",
                "impact": "Automated secret and security rules flagged patterns in the repo.",
            }
        )
    checkov_summary = _summarize_checkov(checkov_data)
    if checkov_summary["failed_checks"]:
        findings.append(
            {
                "severity": "warn",
                "category": "risk",
                "title": "Infrastructure configuration scan reported failed checks.",
                "evidence": f"checkov_failed_checks={checkov_summary['failed_checks']}",
                "impact": "IaC hygiene is relevant for infra-heavy and production-facing repos.",
            }
        )
    hadolint_summary = _summarize_hadolint(hadolint_data)
    if hadolint_summary["issue_count"]:
        findings.append(
            {
                "severity": "warn",
                "category": "risk",
                "title": "Dockerfile lint scan reported issues.",
                "evidence": f"hadolint_issues={hadolint_summary['issue_count']}",
                "impact": "Container hygiene is relevant for deployable service repos.",
            }
        )
    interrogate_summary = _summarize_interrogate(interrogate_data)
    if interrogate_summary["docstring_coverage"] is not None:
        findings.append(
            {
                "severity": "info",
                "category": "quality",
                "title": "Python docstring coverage scan was available.",
                "evidence": f"docstring_coverage={interrogate_summary['docstring_coverage']}",
                "impact": (
                    "Documentation coverage is a useful maintainability signal for Python repos."
                ),
            }
        )
    return findings


def _summarize_scc(data: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
    if not isinstance(data, list):
        return {"code_lines": None, "comment_ratio": None}
    code = 0
    comments = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        code += int(item.get("Code") or 0)
        comments += int(item.get("Comment") or 0)
    return {
        "code_lines": code,
        "comment_ratio": round(comments / code, 4) if code else 0.0,
    }


def _summarize_pip_audit(data: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
    if data is None:
        return {"vulnerability_count": None}
    if isinstance(data, list):
        count = sum(len(item.get("vulns") or []) for item in data if isinstance(item, dict))
        return {"vulnerability_count": count}
    if isinstance(data, dict):
        vulns = data.get("dependencies")
        if isinstance(vulns, list):
            count = sum(len(item.get("vulns") or []) for item in vulns if isinstance(item, dict))
            return {"vulnerability_count": count}
    return {"vulnerability_count": 0}


def _summarize_npm_audit(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"vulnerability_count": None}
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        vulns = metadata.get("vulnerabilities")
        if isinstance(vulns, dict):
            return {"vulnerability_count": sum(int(value or 0) for value in vulns.values())}
    return {"vulnerability_count": 0}


def _summarize_trivy(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"vulnerability_count": None}
    count = 0
    for result in data.get("Results", []) or []:
        vulns = result.get("Vulnerabilities") or []
        if isinstance(vulns, list):
            count += len(vulns)
    return {"vulnerability_count": count}


def _summarize_semgrep(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"result_count": None}
    results = data.get("results")
    return {"result_count": len(results) if isinstance(results, list) else 0}


def _summarize_checkov(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"failed_checks": None}
    failed = (
        data.get("summary", {}).get("failed") if isinstance(data.get("summary"), dict) else None
    )
    return {"failed_checks": failed if isinstance(failed, int) else 0}


def _summarize_hadolint(data: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
    if data is None:
        return {"issue_count": None}
    if isinstance(data, list):
        return {"issue_count": len(data)}
    return {"issue_count": 0}


def _summarize_interrogate(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"docstring_coverage": None}
    percentage = data.get("percentage")
    if isinstance(percentage, (int, float)):
        return {"docstring_coverage": round(float(percentage), 2)}
    return {"docstring_coverage": None}
