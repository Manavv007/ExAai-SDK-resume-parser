"""Detect project stacks and safe command plans for sandbox evaluation."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from agent.sandbox.base import SandboxCommand


def _exists(repo_dir: Path, *names: str) -> bool:
    return any((repo_dir / name).exists() for name in names)


def _has_glob(repo_dir: Path, *patterns: str) -> bool:
    return any(next(repo_dir.glob(pattern), None) is not None for pattern in patterns)


def _read_package_json(repo_dir: Path) -> dict[str, Any]:
    package_file = repo_dir / "package.json"
    if not package_file.exists():
        return {}
    try:
        data = json.loads(package_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _non_comment_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _list_dirs(repo_dir: Path, names: tuple[str, ...]) -> list[str]:
    return sorted(name for name in names if (repo_dir / name).is_dir())


def _list_files(repo_dir: Path, names: tuple[str, ...]) -> list[str]:
    return sorted(name for name in names if (repo_dir / name).exists())


def _detect_framework_markers(repo_dir: Path, package_data: dict[str, Any]) -> list[str]:
    markers: set[str] = set()
    requirements_text = _read_text_if_exists(repo_dir / "requirements.txt").lower()
    pyproject_text = _read_text_if_exists(repo_dir / "pyproject.toml").lower()
    package_text = json.dumps(package_data).lower() if package_data else ""

    python_markers = {
        "streamlit": "streamlit",
        "fastapi": "fastapi",
        "django": "django",
        "flask": "flask",
        "pytest": "pytest",
        "langchain": "langchain",
    }
    for needle, marker in python_markers.items():
        if needle in requirements_text or needle in pyproject_text:
            markers.add(marker)

    node_markers = {
        "react": "react",
        "next": "nextjs",
        "vite": "vite",
        "express": "express",
        "vitest": "vitest",
    }
    for needle, marker in node_markers.items():
        if needle in package_text:
            markers.add(marker)

    if (repo_dir / "Dockerfile").exists():
        markers.add("docker")
    if (repo_dir / ".github" / "workflows").exists():
        markers.add("github-actions")

    return sorted(markers)


def _parse_python_dependencies(repo_dir: Path) -> list[str]:
    requirements = repo_dir / "requirements.txt"
    if requirements.exists():
        return _non_comment_lines(_read_text_if_exists(requirements))

    pyproject_text = _read_text_if_exists(repo_dir / "pyproject.toml")
    match = re.search(r"dependencies\s*=\s*\[(.*?)\]", pyproject_text, re.DOTALL)
    if not match:
        return []

    items = re.findall(r'"([^"]+)"|\'([^\']+)\'', match.group(1))
    dependencies: list[str] = []
    for first, second in items:
        value = first or second
        if value:
            dependencies.append(value.strip())
    return dependencies


def _python_dependencies_pinned(dependencies: list[str]) -> bool:
    if not dependencies:
        return False
    exact_pattern = re.compile(r"^[A-Za-z0-9_.\-]+(\[[^\]]+\])?==[^=].+$")
    return all(exact_pattern.match(dep) for dep in dependencies)


def _dependency_health(repo_dir: Path, package_data: dict[str, Any]) -> dict[str, Any]:
    if (repo_dir / "package.json").exists():
        dependency_map: dict[str, str] = {}
        for key in ("dependencies", "devDependencies", "peerDependencies"):
            value = package_data.get(key)
            if isinstance(value, dict):
                dependency_map.update({str(name): str(spec) for name, spec in value.items()})

        pinned = bool(dependency_map) and all(
            not spec.startswith(("^", "~", ">", "<", "*"))
            and "x" not in spec.lower()
            and spec.lower() != "latest"
            for spec in dependency_map.values()
        )
        return {
            "dependency_count": len(dependency_map),
            "pinned_versions": pinned,
            "outdated_dependencies": None,
        }

    python_dependencies = _parse_python_dependencies(repo_dir)
    if python_dependencies:
        return {
            "dependency_count": len(python_dependencies),
            "pinned_versions": _python_dependencies_pinned(python_dependencies),
            "outdated_dependencies": None,
        }

    return {
        "dependency_count": 0,
        "pinned_versions": False,
        "outdated_dependencies": None,
    }


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
    combined_names = lower_dirs | lower_files

    for layer, names in directory_candidates.items():
        if any(name in combined_names for name in names):
            layers.append(layer)

    if "streamlit" in framework_markers and "ui" not in layers:
        layers.append("ui")
    if (
        "fastapi" in framework_markers or "flask" in framework_markers
    ) and "services" not in layers:
        layers.append("services")

    separation_of_concerns = bool(source_dirs) and bool(test_dirs) and len(layers) >= 2
    return {
        "layers": sorted(layers),
        "separation_of_concerns": separation_of_concerns,
    }


def _infer_project_shape(repo_dir: Path, package_data: dict[str, Any], stack: list[str]) -> str:
    top_dirs = (
        [path.name for path in repo_dir.iterdir() if path.is_dir()] if repo_dir.exists() else []
    )
    markers = _detect_framework_markers(repo_dir, package_data)
    if {"packages", "apps"} & set(top_dirs):
        return "monorepo"
    if "streamlit" in markers:
        return "interactive_app"
    if "fastapi" in markers or "flask" in markers:
        return "service"
    if "django" in markers:
        return "web_app"
    if "node" in stack and "react" in markers:
        return "frontend_app"
    if "python" in stack and _exists(repo_dir, "setup.py", "pyproject.toml"):
        return "python_project"
    if len(stack) == 1 and stack[0] == "go":
        return "go_project"
    return "application" if stack else "unknown"


def _find_entrypoints(repo_dir: Path, package_data: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for name in (
        "main.py",
        "app.py",
        "manage.py",
        "server.py",
        "index.js",
        "main.js",
        "src/main.ts",
    ):
        if (repo_dir / name).exists():
            candidates.append(name)

    package_name = package_data.get("main")
    if isinstance(package_name, str) and package_name:
        candidates.append(package_name)

    bin_field = package_data.get("bin")
    if isinstance(bin_field, str) and bin_field:
        candidates.append(bin_field)
    elif isinstance(bin_field, dict):
        candidates.extend(str(value) for value in bin_field.values() if isinstance(value, str))

    return sorted(dict.fromkeys(candidates))


def _node_install_command(repo_dir: Path) -> str:
    if (repo_dir / "pnpm-lock.yaml").exists():
        return "corepack enable && pnpm install --frozen-lockfile"
    if (repo_dir / "yarn.lock").exists():
        return "corepack enable && yarn install --frozen-lockfile"
    if (repo_dir / "package-lock.json").exists():
        return "npm ci"
    return "npm install"


def detect_project(
    repo_dir: Path,
) -> tuple[
    list[str],
    dict[str, bool | int | str | list[str]],
    list[str],
    dict[str, Any],
    list[dict[str, str]],
]:
    """Return stack names, quality signals, and risk flags for a repository."""
    stack: list[str] = []
    risk_flags: list[str] = []
    dependency_files: list[str] = []

    if _exists(repo_dir, "pyproject.toml", "requirements.txt", "setup.py"):
        stack.append("python")
    if _exists(repo_dir, "package.json"):
        stack.append("node")
    if _exists(repo_dir, "go.mod"):
        stack.append("go")
    if _exists(repo_dir, "Cargo.toml"):
        stack.append("rust")
    if _exists(repo_dir, "pom.xml", "build.gradle", "build.gradle.kts"):
        stack.append("java")

    for name in [
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
    ]:
        if (repo_dir / name).exists():
            dependency_files.append(name)

    package_data = _read_package_json(repo_dir)
    scripts = package_data.get("scripts") if isinstance(package_data.get("scripts"), dict) else {}
    if scripts and any(name in scripts for name in ("preinstall", "install", "postinstall")):
        risk_flags.append("package.json install lifecycle script present")

    has_tests = _has_glob(
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
    has_ci = _has_glob(repo_dir, ".github/workflows/*") or _exists(
        repo_dir,
        ".gitlab-ci.yml",
        ".travis.yml",
        "circle.yml",
    )
    has_docs = _exists(repo_dir, "README.md", "README.rst", "README.txt") or _has_glob(
        repo_dir,
        "docs/**",
    )
    has_docker = _exists(repo_dir, "Dockerfile", "docker-compose.yml", "docker-compose.yaml")
    file_count = sum(1 for path in repo_dir.rglob("*") if path.is_file())

    framework_markers = _detect_framework_markers(repo_dir, package_data)
    source_dirs = _list_dirs(
        repo_dir,
        ("src", "app", "apps", "lib", "pkg", "server", "backend", "frontend"),
    )
    test_dirs = _list_dirs(repo_dir, ("tests", "test", "__tests__"))
    dependency_health = _dependency_health(repo_dir, package_data)
    architecture = _architecture_profile(
        repo_dir,
        source_dirs,
        test_dirs,
        framework_markers,
    )
    config_files = _list_files(
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
        ),
    )
    git_metrics = _calculate_git_metrics(repo_dir)
    code_metrics = _calculate_code_metrics(repo_dir)
    secrets_hits = _calculate_secrets(repo_dir)
    sample_files = _collect_sample_files(repo_dir)
    has_dockerfile = (repo_dir / "Dockerfile").exists()

    react_doctor_score = None
    react_findings = []

    is_react = "react" in framework_markers
    if is_react:
        react_data = _run_react_doctor(repo_dir)
        if react_data:
            score_obj = react_data.get("score")
            if isinstance(score_obj, dict):
                react_doctor_score = score_obj.get("value")
            elif isinstance(score_obj, (int, float)):
                react_doctor_score = score_obj

            react_findings = _parse_react_doctor_diagnostics(react_data)

            react_diags = react_data.get("diagnostics", [])
            react_violations_count = len(react_diags)
            total_loc = sum(
                len(_read_text_if_exists(f).splitlines())
                for f in _collect_files_to_analyze(repo_dir)
            )
            code_metrics["lint_violations_per_kloc"] = (
                round((react_violations_count / total_loc) * 1000, 2) if total_loc > 0 else 0.0
            )

    repo_profile = {
        "project_shape": _infer_project_shape(repo_dir, package_data, stack),
        "framework_markers": framework_markers,
        "entrypoints": _find_entrypoints(repo_dir, package_data),
        "source_dirs": source_dirs,
        "test_dirs": test_dirs,
        "config_files": config_files,
        "dependency_health": dependency_health,
        "architecture": architecture,
        "commit_count": git_metrics["commit_count"],
        "unique_authors": git_metrics["unique_authors"],
        "days_since_last_commit": git_metrics["days_since_last_commit"],
        "has_ci": has_ci,
        "has_tests": has_tests,
        "has_docs": has_docs,
        "has_dockerfile": has_dockerfile,
        "top_author_commit_share": git_metrics["top_author_commit_share"],
        "sole_author": git_metrics["sole_author"],
        "avg_cyclomatic_complexity": code_metrics["avg_cyclomatic_complexity"],
        "type_annotation_ratio": code_metrics["type_annotation_ratio"],
        "error_handling_density": code_metrics["error_handling_density"],
        "todo_fixme_density": code_metrics["todo_fixme_density"],
        "lint_violations_per_kloc": code_metrics["lint_violations_per_kloc"],
        "secret_pattern_hits": secrets_hits,
        "sample_files": sample_files,
    }
    if react_doctor_score is not None:
        repo_profile["react_doctor_score"] = react_doctor_score

    findings: list[dict[str, str]] = []
    if has_tests:
        test_evidence = ", ".join(test_dirs) or "pattern match only"
        findings.append(
            {
                "severity": "info",
                "category": "tests",
                "title": "Repository includes an automated test surface.",
                "evidence": f"Detected test directories/files: {test_evidence}",
                "impact": (
                    "Gives us a direct way to validate candidate code behavior in the sandbox."
                ),
            }
        )
    else:
        findings.append(
            {
                "severity": "warn",
                "category": "tests",
                "title": "Repository does not expose an obvious automated test suite.",
                "evidence": "No common test directories or file patterns were detected.",
                "impact": (
                    "Reduces confidence in correctness and makes dynamic evaluation shallower."
                ),
            }
        )
    if has_ci:
        findings.append(
            {
                "severity": "info",
                "category": "quality",
                "title": "Repository includes CI configuration.",
                "evidence": "Detected workflow or CI config files in the repository root.",
                "impact": "Suggests the candidate has some repeatable validation workflow.",
            }
        )
    if framework_markers:
        findings.append(
            {
                "severity": "info",
                "category": "structure",
                "title": "Project stack markers were identified.",
                "evidence": f"Framework/tool markers: {', '.join(framework_markers)}",
                "impact": (
                    "Helps us judge whether the repo looks like a serious "
                    "application versus a toy snapshot."
                ),
            }
        )
    if dependency_health["dependency_count"]:
        pinned_text = (
            "uses pinned dependency versions"
            if dependency_health["pinned_versions"]
            else "does not pin all dependency versions"
        )
        findings.append(
            {
                "severity": "info" if dependency_health["pinned_versions"] else "warn",
                "category": "dependencies",
                "title": "Dependency hygiene was inferred from manifest files.",
                "evidence": (
                    f"{dependency_health['dependency_count']} dependencies detected; "
                    f"repo {pinned_text}."
                ),
                "impact": (
                    "Version pinning improves reproducibility, while loose ranges can "
                    "make builds drift over time."
                ),
            }
        )
    if architecture["layers"]:
        findings.append(
            {
                "severity": "info",
                "category": "structure",
                "title": "Repository shows identifiable architecture layers.",
                "evidence": f"Detected layers: {', '.join(architecture['layers'])}",
                "impact": (
                    "Clear layering usually makes the codebase easier to reason about and maintain."
                ),
            }
        )

    findings.extend(react_findings)
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
    """Build a deterministic install/build/test command plan."""
    commands: list[SandboxCommand] = []
    package_data = _read_package_json(repo_dir)
    scripts = package_data.get("scripts") if isinstance(package_data.get("scripts"), dict) else {}

    if _exists(repo_dir, "pyproject.toml", "requirements.txt", "setup.py"):
        if (repo_dir / "pyproject.toml").exists():
            commands.append(
                SandboxCommand(step="install", command='python -m pip install -e ".[dev]"')
            )
        elif (repo_dir / "requirements.txt").exists():
            commands.append(
                SandboxCommand(step="install", command="python -m pip install -r requirements.txt")
            )
        commands.append(SandboxCommand(step="build", command="python -m compileall -q ."))
        if _has_glob(repo_dir, "tests/**", "test/**", "**/*_test.py"):
            commands.append(SandboxCommand(step="test", command="python -m pytest -q"))

    if (repo_dir / "package.json").exists():
        commands.append(SandboxCommand(step="install", command=_node_install_command(repo_dir)))
        if "build" in scripts:
            commands.append(SandboxCommand(step="build", command="npm run build"))
        if "test" in scripts:
            commands.append(SandboxCommand(step="test", command="npm test -- --watch=false"))

    if (repo_dir / "go.mod").exists():
        commands.append(SandboxCommand(step="test", command="go test ./..."))

    if (repo_dir / "Cargo.toml").exists():
        commands.append(SandboxCommand(step="test", command="cargo test --locked"))

    if (repo_dir / "pom.xml").exists():
        commands.append(SandboxCommand(step="test", command="mvn test -q"))
    elif (repo_dir / "build.gradle").exists() or (repo_dir / "build.gradle.kts").exists():
        commands.append(SandboxCommand(step="test", command="./gradlew test --no-daemon"))

    if not commands:
        commands.append(
            SandboxCommand(
                step="inspect",
                command="find . -maxdepth 2 -type f | sort | head -200",
            )
        )

    return commands


class PythonASTAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.complexity_nodes = 0
        self.total_functions = 0
        self.annotated_functions = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.total_functions += 1
        has_annotation = bool(node.returns)
        if not has_annotation:
            for arg in node.args.args:
                if arg.annotation:
                    has_annotation = True
                    break
            if not has_annotation and hasattr(node.args, "kwonlyargs"):
                for arg in node.args.kwonlyargs:
                    if arg.annotation:
                        has_annotation = True
                        break
            if not has_annotation and node.args.vararg and node.args.vararg.annotation:
                has_annotation = True
            if not has_annotation and node.args.kwarg and node.args.kwarg.annotation:
                has_annotation = True

        if has_annotation:
            self.annotated_functions += 1

        self.complexity_nodes += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)

    def visit_If(self, node):
        self.complexity_nodes += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.complexity_nodes += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.complexity_nodes += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.complexity_nodes += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.complexity_nodes += len(node.values) - 1
        self.generic_visit(node)

    def visit_ListComp(self, node):
        self.complexity_nodes += len(node.generators)
        self.generic_visit(node)

    def visit_DictComp(self, node):
        self.complexity_nodes += len(node.generators)
        self.generic_visit(node)

    def visit_SetComp(self, node):
        self.complexity_nodes += len(node.generators)
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        self.complexity_nodes += len(node.generators)
        self.generic_visit(node)


def _collect_files_to_analyze(repo_dir: Path, max_files: int = 150) -> list[Path]:
    extensions = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".go",
        ".java",
        ".rs",
        ".cpp",
        ".c",
        ".h",
        ".cs",
        ".rb",
        ".php",
    }
    exclude_dirs = {
        ".git",
        "node_modules",
        "venv",
        ".venv",
        "env",
        "build",
        "dist",
        "__pycache__",
        "target",
        "tests",
        "test",
        ".github",
        ".idea",
        ".vscode",
    }

    files = []
    for root, dirs, filenames in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for name in filenames:
            if "test" in name.lower() or "spec" in name.lower():
                continue
            path = Path(root) / name
            if path.suffix.lower() in extensions:
                files.append(path)
                if len(files) >= max_files:
                    return files
    return files


def _estimate_non_python_complexity_and_annotations(
    content: str, suffix: str
) -> tuple[int, int, int, int]:
    func_pattern = re.compile(
        r"\b(function|func|fn)\b|"
        r"\b(public|private|protected|static)?\s*[\w\<\>\[\]]+\s+\w+\s*\([^\)]*\)\s*\{"
    )
    functions = len(func_pattern.findall(content))

    complexity_keywords = re.compile(r"\b(if|for|while|catch|case)\b|&&|\|\|")
    complexity_points = len(complexity_keywords.findall(content))

    complexity_points += functions

    annotated_functions = 0
    if suffix in {".ts", ".tsx", ".go", ".rs", ".java", ".cpp", ".c", ".h", ".cs"}:
        annotated_functions = functions

    return functions, complexity_points, functions, annotated_functions


def _calculate_git_metrics(repo_dir: Path) -> dict[str, Any]:
    metrics = {
        "commit_count": 0,
        "unique_authors": 0,
        "days_since_last_commit": 0,
        "top_author_commit_share": 0.0,
        "sole_author": False,
    }

    if not (repo_dir / ".git").exists():
        return metrics

    try:
        res_count = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if res_count.returncode == 0:
            metrics["commit_count"] = int(res_count.stdout.strip())

        res_authors = subprocess.run(
            ["git", "log", "--format=%an"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if res_authors.returncode == 0:
            authors = [line.strip() for line in res_authors.stdout.splitlines() if line.strip()]
            if authors:
                metrics["unique_authors"] = len(set(authors))
                metrics["sole_author"] = metrics["unique_authors"] == 1

                counts = Counter(authors)
                top_author_commits = counts.most_common(1)[0][1]
                metrics["top_author_commit_share"] = round(top_author_commits / len(authors), 2)

        res_time = subprocess.run(
            ["git", "log", "-1", "--format=%at"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if res_time.returncode == 0 and res_time.stdout.strip():
            commit_time = int(res_time.stdout.strip())
            elapsed = time.time() - commit_time
            metrics["days_since_last_commit"] = max(0, int(elapsed / 86400))
    except Exception:
        pass

    return metrics


def _calculate_code_metrics(repo_dir: Path) -> dict[str, Any]:
    files = _collect_files_to_analyze(repo_dir)

    total_loc = 0
    total_complexity_points = 0
    total_functions = 0

    total_annotated_functions = 0
    total_functions_for_annotations = 0

    error_handling_count = 0
    todo_fixme_count = 0
    lint_violations_count = 0

    error_pattern = re.compile(r"\b(except|catch|recover|finally)\b|err\s*!=\s*nil|\.catch\b")
    todo_pattern = re.compile(r"\b(TODO|FIXME)\b", re.IGNORECASE)

    ts_files = 0
    js_ts_files = 0

    for path in files:
        content = _read_text_if_exists(path)
        if not content:
            continue

        lines = content.splitlines()
        loc = len(lines)
        total_loc += loc

        error_handling_count += len(error_pattern.findall(content))
        todo_fixme_count += len(todo_pattern.findall(content))

        violations = 0
        consecutive_empty = 0
        for line in lines:
            if len(line) > 120:
                violations += 1
            clean_line = line.replace("\n", "").replace("\r", "")
            if clean_line and clean_line != clean_line.rstrip():
                violations += 1
            if not line.strip():
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    violations += 1
            else:
                consecutive_empty = 0
        lint_violations_count += violations

        suffix = path.suffix.lower()
        if suffix == ".py":
            try:
                tree = ast.parse(content, filename=str(path))
                analyzer = PythonASTAnalyzer()
                analyzer.visit(tree)

                total_functions += analyzer.total_functions
                total_complexity_points += analyzer.complexity_nodes

                total_functions_for_annotations += analyzer.total_functions
                total_annotated_functions += analyzer.annotated_functions
            except Exception:
                fns, comp, tot_m, ann_m = _estimate_non_python_complexity_and_annotations(
                    content, ".py"
                )
                total_functions += fns
                total_complexity_points += comp
                total_functions_for_annotations += tot_m
                total_annotated_functions += ann_m
        else:
            if suffix in {".js", ".jsx", ".ts", ".tsx"}:
                js_ts_files += 1
                if suffix in {".ts", ".tsx"}:
                    ts_files += 1

            fns, comp, tot_m, ann_m = _estimate_non_python_complexity_and_annotations(
                content, suffix
            )
            total_functions += fns
            total_complexity_points += comp

            if suffix not in {".js", ".jsx", ".ts", ".tsx"}:
                total_functions_for_annotations += tot_m
                total_annotated_functions += ann_m

    avg_complexity = 1.0
    if total_functions > 0:
        avg_complexity = round(total_complexity_points / total_functions, 2)

    type_annotation_ratio = 0.0
    if js_ts_files > 0:
        type_annotation_ratio = round(ts_files / js_ts_files, 2)
    elif total_functions_for_annotations > 0:
        type_annotation_ratio = round(
            total_annotated_functions / total_functions_for_annotations, 2
        )

    error_handling_density = 0.0
    if total_loc > 0:
        error_handling_density = round(error_handling_count / total_loc, 4)

    todo_fixme_density = 0.0
    if total_loc > 0:
        todo_fixme_density = round(todo_fixme_count / total_loc, 4)

    lint_violations_per_kloc = 0.0
    if total_loc > 0:
        lint_violations_per_kloc = round((lint_violations_count / total_loc) * 1000, 2)

    return {
        "avg_cyclomatic_complexity": avg_complexity,
        "type_annotation_ratio": type_annotation_ratio,
        "error_handling_density": error_handling_density,
        "todo_fixme_density": todo_fixme_density,
        "lint_violations_per_kloc": lint_violations_per_kloc,
    }


def _calculate_secrets(repo_dir: Path) -> int:
    patterns = [
        re.compile(
            r"([^A-Z0-9]|^)(AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}([^A-Z0-9]|$)",
            re.IGNORECASE,
        ),
        re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),
        re.compile(
            r"https://hooks\.slack\.(?:com|example)/services/T[A-Z0-9_]{8}/B[A-Z0-9_]{8}/[A-Za-z0-9_]{24}",
            re.IGNORECASE,
        ),
        re.compile(r"amqps?://[a-zA-Z0-9_.-]+:[a-zA-Z0-9_.-]+@[a-zA-Z0-9_.-]+"),
        re.compile(
            r"\b[a-z0-9_]*(password|passwd|secret|api_key|apikey|token|private_key|auth_token)[a-z0-9_]*\s*=\s*['\"]?[a-zA-Z0-9_\-\.\+]{16,}['\"]?",
            re.IGNORECASE,
        ),
    ]

    hits = 0
    files = _collect_files_to_analyze(repo_dir)
    for env_file in repo_dir.rglob("*.env*"):
        if env_file.is_file() and ".git" not in env_file.parts:
            files.append(env_file)

    seen = set()
    files = [x for x in files if not (x in seen or seen.add(x))]

    for path in files[:150]:
        content = _read_text_if_exists(path)
        if not content:
            continue
        for pattern in patterns:
            hits += len(pattern.findall(content))

    return hits


def _collect_sample_files(repo_dir: Path) -> list[dict[str, Any]]:
    files = _collect_files_to_analyze(repo_dir)
    if not files:
        return []

    files_with_size = []
    for f in files:
        try:
            files_with_size.append((f, f.stat().st_size))
        except OSError:
            continue

    files_with_size.sort(key=lambda x: x[1], reverse=True)
    top_files = [f for f, _ in files_with_size[:3]]

    sample_files = []
    for path in top_files:
        content = _read_text_if_exists(path)
        lines = content.splitlines()
        preview = "\n".join(lines[:30])
        try:
            rel_path = str(path.relative_to(repo_dir)).replace("\\", "/")
        except ValueError:
            rel_path = str(path).replace("\\", "/")

        sample_files.append(
            {
                "path": rel_path,
                "lines": len(lines),
                "content_preview": preview,
            }
        )
    return sample_files


def _run_react_doctor(repo_dir: Path) -> dict[str, Any] | None:
    if "PYTEST_CURRENT_TEST" in os.environ:
        if getattr(subprocess.run, "__module__", "") == "subprocess":
            return None
    try:
        npx_cmd = shutil.which("npx")
        if not npx_cmd:
            npx_cmd = "npx"

        completed = subprocess.run(
            [npx_cmd, "react-doctor@latest", "--json", "--no-telemetry"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            shell=True,
            timeout=45,
        )

        stdout = completed.stdout.strip()
        start_idx = stdout.find("{")
        end_idx = stdout.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = stdout[start_idx : end_idx + 1]
            data = json.loads(json_str)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return None


def _parse_react_doctor_diagnostics(react_data: dict[str, Any]) -> list[dict[str, str]]:
    findings = []
    diagnostics = react_data.get("diagnostics", [])
    if not isinstance(diagnostics, list):
        return findings

    for diag in diagnostics:
        if not isinstance(diag, dict):
            continue

        file_path = diag.get("filePath") or diag.get("file") or diag.get("path") or "unknown"
        line = diag.get("line") or diag.get("loc") or ""
        message = diag.get("message") or diag.get("description") or "React code issue"
        severity_raw = str(diag.get("severity") or "").lower()
        rule = diag.get("ruleId") or diag.get("rule") or "react-doctor"

        severity = "info"
        if "error" in severity_raw or "critical" in severity_raw:
            severity = "high"
        elif "warning" in severity_raw or "warn" in severity_raw:
            severity = "warn"

        evidence = f"File: {file_path}"
        if line:
            evidence += f" (Line {line})"
        evidence += f" - Rule: {rule}"

        findings.append(
            {
                "severity": severity,
                "category": "quality",
                "title": f"React Doctor: {message}",
                "evidence": evidence,
                "impact": (
                    "React Doctor identified this code structure as an "
                    "anti-pattern or potential bottleneck."
                ),
            }
        )
    return findings
