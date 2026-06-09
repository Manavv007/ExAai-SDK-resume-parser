"""Fast repository filesystem and manifest scanning helpers."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

MAX_SOURCE_FILES = 150
MAX_SAMPLE_FILES = 5
MAX_SAMPLE_PREVIEW_LINES = 30
MAX_SAMPLE_FILE_BYTES = 128_000
SOURCE_EXTENSIONS = {
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
EXCLUDE_DIRS = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "build",
    "dist",
    "__pycache__",
    "target",
    ".idea",
    ".vscode",
}


README_BASENAMES = frozenset({"readme.md", "readme.rst", "readme.txt", "readme"})


def exists(repo_dir: Path, *names: str) -> bool:
    return any((repo_dir / name).exists() for name in names)


def find_readme_path(repo_dir: Path) -> Path | None:
    """Return the repo README path, matching common names case-insensitively."""
    if not repo_dir.is_dir():
        return None

    for name in ("README.md", "README.rst", "README.txt", "README"):
        path = repo_dir / name
        if path.is_file():
            return path

    try:
        for entry in repo_dir.iterdir():
            if entry.is_file() and entry.name.lower() in README_BASENAMES:
                return entry
    except OSError:
        return None

    return None


def has_readme(repo_dir: Path) -> bool:
    return find_readme_path(repo_dir) is not None


def has_dockerfile(repo_dir: Path) -> bool:
    return (repo_dir / "Dockerfile").is_file()


def has_docker_compose(repo_dir: Path) -> bool:
    return exists(repo_dir, "docker-compose.yml", "docker-compose.yaml")


def has_docker_config(repo_dir: Path) -> bool:
    return has_dockerfile(repo_dir) or has_docker_compose(repo_dir)


def has_glob(repo_dir: Path, *patterns: str) -> bool:
    return any(next(repo_dir.glob(pattern), None) is not None for pattern in patterns)


def read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def read_package_json(repo_dir: Path) -> dict[str, Any]:
    package_file = repo_dir / "package.json"
    if not package_file.exists():
        return {}
    try:
        data = json.loads(package_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def non_comment_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def list_dirs(repo_dir: Path, names: tuple[str, ...]) -> list[str]:
    return sorted(name for name in names if (repo_dir / name).is_dir())


def list_files(repo_dir: Path, names: tuple[str, ...]) -> list[str]:
    return sorted(name for name in names if (repo_dir / name).exists())


def collect_source_files(repo_dir: Path, *, include_tests: bool = False) -> list[Path]:
    files: list[Path] = []
    for root, dirs, filenames in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for name in filenames:
            if not include_tests and ("test" in name.lower() or "spec" in name.lower()):
                continue
            path = Path(root) / name
            if path.suffix.lower() not in SOURCE_EXTENSIONS:
                continue
            try:
                if path.stat().st_size > MAX_SAMPLE_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append(path)
            if len(files) >= MAX_SOURCE_FILES:
                return files
    return files


def collect_sample_files(repo_dir: Path) -> list[dict[str, Any]]:
    files = collect_source_files(repo_dir)
    if not files:
        return []

    files_with_size: list[tuple[Path, int]] = []
    for path in files:
        try:
            files_with_size.append((path, path.stat().st_size))
        except OSError:
            continue

    files_with_size.sort(key=lambda item: item[1], reverse=True)
    sample_files: list[dict[str, Any]] = []
    for path, _size in files_with_size[:MAX_SAMPLE_FILES]:
        content = read_text_if_exists(path)
        lines = content.splitlines()
        try:
            rel_path = str(path.relative_to(repo_dir)).replace("\\", "/")
        except ValueError:
            rel_path = str(path).replace("\\", "/")
        sample_files.append(
            {
                "path": rel_path,
                "lines": len(lines),
                "content_preview": "\n".join(lines[:MAX_SAMPLE_PREVIEW_LINES]),
            }
        )
    return sample_files


def parse_python_dependencies(repo_dir: Path) -> list[str]:
    requirements = repo_dir / "requirements.txt"
    if requirements.exists():
        return non_comment_lines(read_text_if_exists(requirements))

    pyproject_text = read_text_if_exists(repo_dir / "pyproject.toml")
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


def python_dependencies_pinned(dependencies: list[str]) -> bool:
    if not dependencies:
        return False
    exact_pattern = re.compile(r"^[A-Za-z0-9_.\-]+(\[[^\]]+\])?==[^=].+$")
    return all(exact_pattern.match(dep) for dep in dependencies)


def dependency_health(repo_dir: Path, package_data: dict[str, Any]) -> dict[str, Any]:
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

    python_dependencies = parse_python_dependencies(repo_dir)
    if python_dependencies:
        return {
            "dependency_count": len(python_dependencies),
            "pinned_versions": python_dependencies_pinned(python_dependencies),
            "outdated_dependencies": None,
        }

    return {
        "dependency_count": 0,
        "pinned_versions": False,
        "outdated_dependencies": None,
    }


def detect_framework_markers(repo_dir: Path, package_data: dict[str, Any]) -> list[str]:
    markers: set[str] = set()
    requirements_text = read_text_if_exists(repo_dir / "requirements.txt").lower()
    pyproject_text = read_text_if_exists(repo_dir / "pyproject.toml").lower()
    package_text = json.dumps(package_data).lower() if package_data else ""

    python_markers = {
        "streamlit": "streamlit",
        "fastapi": "fastapi",
        "django": "django",
        "flask": "flask",
        "pytest": "pytest",
        "langchain": "langchain",
    }
    node_markers = {
        "react": "react",
        "next": "nextjs",
        "vite": "vite",
        "express": "express",
        "vitest": "vitest",
    }

    for needle, marker in python_markers.items():
        if needle in requirements_text or needle in pyproject_text:
            markers.add(marker)
    for needle, marker in node_markers.items():
        if needle in package_text:
            markers.add(marker)

    if (repo_dir / "Dockerfile").exists():
        markers.add("docker")
    if (repo_dir / ".github" / "workflows").exists():
        markers.add("github-actions")

    return sorted(markers)


def infer_project_shape(repo_dir: Path, package_data: dict[str, Any], stack: list[str]) -> str:
    top_dirs = (
        [path.name for path in repo_dir.iterdir() if path.is_dir()]
        if repo_dir.exists()
        else []
    )
    markers = detect_framework_markers(repo_dir, package_data)
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
    if "python" in stack and exists(repo_dir, "setup.py", "pyproject.toml"):
        return "python_project"
    if len(stack) == 1 and stack[0] == "go":
        return "go_project"
    return "application" if stack else "unknown"


def find_entrypoints(repo_dir: Path, package_data: dict[str, Any]) -> list[str]:
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

    package_main = package_data.get("main")
    if isinstance(package_main, str) and package_main:
        candidates.append(package_main)

    bin_field = package_data.get("bin")
    if isinstance(bin_field, str) and bin_field:
        candidates.append(bin_field)
    elif isinstance(bin_field, dict):
        candidates.extend(str(value) for value in bin_field.values() if isinstance(value, str))

    return sorted(dict.fromkeys(candidates))


def infer_repo_type_tags(
    *,
    stack: list[str],
    framework_markers: list[str],
    architecture_layers: list[str],
    project_shape: str,
    package_data: dict[str, Any],
    repo_dir: Path,
) -> list[str]:
    tags: set[str] = set()
    if project_shape == "monorepo":
        tags.add("monorepo")
    if (
        "react" in framework_markers
        or "nextjs" in framework_markers
        or "vite" in framework_markers
    ):
        tags.add("frontend_app")
    if (
        "fastapi" in framework_markers
        or "flask" in framework_markers
        or "express" in framework_markers
    ):
        tags.add("backend_service")
    if "frontend_app" in tags and "backend_service" in tags:
        tags.add("fullstack_app")
    if "langchain" in framework_markers:
        tags.add("ai_agent_project")
    if "guardrails" in architecture_layers:
        tags.add("rag_pipeline")
    if "pipeline" in architecture_layers and "data" in architecture_layers:
        tags.add("data_pipeline")
    if exists(repo_dir, "terraform", "pulumi", "helm", "k8s"):
        tags.add("infra_iac_repo")
    if exists(repo_dir, "setup.py", "pyproject.toml", "Cargo.toml", "go.mod") and project_shape in {
        "python_project",
        "go_project",
        "application",
    }:
        tags.add("sdk_or_library")
    scripts = package_data.get("scripts") if isinstance(package_data.get("scripts"), dict) else {}
    if (
        scripts
        and any(name in scripts for name in ("start", "dev", "build"))
        and "fullstack_app" not in tags
    ):
        tags.add("automation_tool")
    if not tags:
        tags.add("research_prototype" if len(stack) <= 1 else "portfolio_project")
    return sorted(tags)
