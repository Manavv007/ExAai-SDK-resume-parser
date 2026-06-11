"""Role-aware repository file focus, classification, and content quality helpers."""

from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Any

RepoRole = str  # aligned | adjacent | peripheral | orthogonal
ContentStatus = str  # ok | stub | vague | empty | missing
REPO_CLASSIFICATIONS = frozenset({"aligned", "adjacent", "peripheral", "orthogonal"})
RISK_ONLY_EVALUATION_MODE = "risk_only"


def is_risk_only_evaluation(focus_spec: dict[str, Any] | None) -> bool:
    """True when sandbox should run security scans without file excerpts."""
    return (
        isinstance(focus_spec, dict)
        and str(focus_spec.get("evaluation_mode") or "").strip().lower()
        == RISK_ONLY_EVALUATION_MODE
    )


def build_risk_only_focus_spec(
    *,
    repo_role: RepoRole,
    candidate_tags: list[str] | None,
    file_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Focus spec for vuln/secret pre-pass (no sample_files or top_files)."""
    return {
        "evaluation_mode": RISK_ONLY_EVALUATION_MODE,
        "repo_role": repo_role,
        "candidate_tags": list(candidate_tags or []),
        "file_paths": list(file_paths or []),
        "focus_paths": [],
        "max_files": 0,
        "top_files_count": 0,
        "pick_mode": "risk_only",
    }


MANIFEST_BASENAMES = frozenset(
    {
        "package.json",
        "package-lock.json",
        "pyproject.toml",
        "requirements.txt",
        "setup.py",
        "go.mod",
        "cargo.toml",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
    }
)

ENTRYPOINT_CANDIDATES = (
    "main.py",
    "app.py",
    "manage.py",
    "server.py",
    "index.js",
    "main.js",
    "src/main.ts",
    "src/index.ts",
)

BACKEND_PATH_HINTS = (
    "api",
    "apis",
    "backend",
    "server",
    "services",
    "service",
    "routes",
    "controllers",
    "handlers",
    "models",
    "domain",
    "internal",
    "pkg",
    "migrations",
)

FRONTEND_PATH_HINTS = ("static", "assets", "public", "dist", "styles", "css", "frontend", "ui")
AI_PATH_HINTS = ("agent", "agents", "llm", "rag", "prompt", "chain", "workflow", "n8n")

STUB_MARKERS = (
    "todo",
    "fixme",
    "notimplemented",
    "pass  #",
    "raise notimplementederror",
    "placeholder",
    "lorem ipsum",
    "coming soon",
    "implement me",
)

ORTHOGONAL_REPO_TAGS = frozenset({"frontend_app"})
BACKEND_CANDIDATE_TAGS = frozenset(
    {"backend_engineer", "fullstack_engineer", "general_software_engineer"}
)
AI_CANDIDATE_TAGS = frozenset({"ai_engineer", "ml_engineer"})


def classify_repo_role(
    *,
    repo_type_tags: list[str] | None,
    candidate_tags: list[str] | None,
    file_paths: list[str] | None = None,
) -> RepoRole:
    """Classify how relevant a repo is to the candidate profile."""
    tags = {str(tag).strip() for tag in (repo_type_tags or []) if str(tag).strip()}
    profile = {str(tag).strip() for tag in (candidate_tags or []) if str(tag).strip()}
    paths = [str(path) for path in (file_paths or []) if path]

    css_html_ratio = _css_html_ratio(paths)
    if css_html_ratio >= 0.65 and "backend_service" not in tags:
        return "orthogonal"

    if profile & AI_CANDIDATE_TAGS:
        if tags & {"ai_agent_project", "rag_pipeline", "data_pipeline"}:
            return "aligned"
        if "automation_tool" in tags or any("n8n" in path.lower() for path in paths):
            return "adjacent"

    if profile & BACKEND_CANDIDATE_TAGS:
        if tags & {"backend_service", "fullstack_app", "sdk_or_library"}:
            return "aligned"
        if tags & {"frontend_app"} and "fullstack_app" not in tags:
            return "orthogonal"
        if tags & {"frontend_app", "fullstack_app"}:
            return "adjacent"

    if tags & {"research_prototype", "portfolio_project"}:
        return "peripheral"
    if tags & {"backend_service", "fullstack_app", "ai_agent_project"}:
        return "aligned"
    if tags & {"frontend_app"}:
        return "adjacent"
    return "peripheral"


def classify_content_quality(content: str) -> ContentStatus:
    """Estimate whether file content is substantive or hollow."""
    text = (content or "").strip()
    if not text:
        return "empty"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "empty"

    non_comment = [line for line in lines if not line.startswith(("#", "//", "/*", "*", "--"))]
    if not non_comment:
        return "vague"

    joined = "\n".join(non_comment).lower()
    if len(non_comment) <= 3 and any(marker in joined for marker in STUB_MARKERS):
        return "stub"

    code_like = sum(
        1
        for line in non_comment
        if re.search(r"\b(def|class|function|interface|public |private |import |return )\b", line)
    )
    if code_like == 0 and len(non_comment) <= 8:
        return "vague"
    if any(marker in joined for marker in STUB_MARKERS) and code_like <= 1:
        return "stub"
    return "ok"


def build_mandatory_focus_paths(file_paths: list[str]) -> list[dict[str, Any]]:
    """Always sample manifests, README, entrypoints, and one test file when present."""
    paths = sorted({str(path).replace("\\", "/") for path in file_paths if path})
    path_set = set(paths)
    selected: list[dict[str, Any]] = []

    def add(path: str, *, max_lines: int = 80) -> None:
        normalized = path.replace("\\", "/")
        if normalized in path_set and not any(item["path"] == normalized for item in selected):
            selected.append({"path": normalized, "max_lines": max_lines, "source": "mandatory"})

    readme = next((path for path in paths if path.lower().endswith("readme.md")), None)
    if readme:
        add(readme, max_lines=80)
    for name in MANIFEST_BASENAMES:
        match = next((path for path in paths if path.lower().endswith(name)), None)
        if match:
            add(match, max_lines=120)
    for candidate in ENTRYPOINT_CANDIDATES:
        add(candidate, max_lines=120)
    test_file = next(
        (
            path
            for path in paths
            if "/test" in path.lower()
            or path.lower().startswith("test")
            or path.endswith(("_test.py", ".test.ts", ".test.js", "Test.java"))
        ),
        None,
    )
    if test_file:
        add(test_file, max_lines=100)
    return selected[:8]


def resolve_focus_path(requested: str, file_paths: list[str]) -> tuple[str | None, bool]:
    """Resolve a requested path against a repo tree; return substitute flag."""
    normalized = requested.replace("\\", "/").lstrip("./")
    path_set = {path.replace("\\", "/") for path in file_paths}
    if normalized in path_set:
        return normalized, False

    basename = normalized.rsplit("/", 1)[-1]
    basename_matches = [
        path for path in file_paths if path.endswith("/" + basename) or path == basename
    ]
    if basename_matches:
        return sorted(basename_matches, key=len)[0].replace("\\", "/"), True

    close = get_close_matches(normalized, sorted(path_set), n=1, cutoff=0.72)
    if close:
        return close[0], True
    return None, False


def rank_paths_for_profile(
    file_paths: list[str],
    *,
    candidate_tags: list[str] | None,
    repo_role: RepoRole,
) -> list[str]:
    """Rank repo paths for role-aware heuristic sampling."""
    profile = {str(tag) for tag in (candidate_tags or [])}
    ranked: list[tuple[int, str]] = []

    for raw_path in file_paths:
        path = raw_path.replace("\\", "/")
        lower = path.lower()
        if any(part in lower for part in ("/node_modules/", "/dist/", "/build/", "/.git/")):
            continue
        if lower.endswith((".min.js", ".min.css", ".map", ".lock", ".png", ".jpg", ".svg")):
            continue
        score = 0
        if lower.endswith((".py", ".java", ".go", ".rs", ".ts", ".js", ".cs", ".rb", ".php")):
            score += 2
        if any(hint in lower for hint in BACKEND_PATH_HINTS):
            score += 4
        if profile & AI_CANDIDATE_TAGS and any(hint in lower for hint in AI_PATH_HINTS):
            score += 5
        if repo_role == "orthogonal" and any(hint in lower for hint in FRONTEND_PATH_HINTS):
            score += 1
        elif any(hint in lower for hint in FRONTEND_PATH_HINTS):
            score -= 2
        if "/test" in lower or lower.endswith(("_test.py", ".test.ts", ".test.js")):
            score += 2
        ranked.append((score, path))

    ranked.sort(key=lambda item: (-item[0], len(item[1])))
    return [path for score, path in ranked if score > 0] or [path for _, path in ranked]


def validate_orchestrated_sandbox_repo_spec(
    *,
    repo_url: str,
    classification: Any,
    structure_classification: str | None,
    focus_paths: list[Any] | None,
    require_agent_focus: bool,
) -> list[str]:
    """
    Validate agent sandbox repo_specs when evidence orchestration is enabled.

    Requires non-empty focus_paths and a classification copied from
    ``get_github_repo_structures`` (must match the structure tool output).
    """
    if not require_agent_focus:
        return []

    errors: list[str] = []
    if not focus_paths:
        errors.append(
            f"{repo_url}: focus_paths is required; pick 1-5 JD-aligned code paths from "
            "get_github_repo_structures (suggested_focus_paths or the file tree)"
        )
        return errors

    cls = str(classification or "").strip().lower()
    if not cls:
        errors.append(
            f"{repo_url}: classification is required; copy the classification field from "
            "get_github_repo_structures for this repo"
        )
    elif cls not in REPO_CLASSIFICATIONS:
        errors.append(
            f"{repo_url}: classification must be one of: " + ", ".join(sorted(REPO_CLASSIFICATIONS))
        )

    expected = str(structure_classification or "").strip().lower()
    if cls and expected and cls != expected:
        errors.append(
            f"{repo_url}: classification '{cls}' does not match get_github_repo_structures "
            f"result '{expected}' — do not invent classifications"
        )
    return errors


def validate_repo_focus_paths(
    *,
    repo_url: str,
    focus_paths: list[Any] | None,
    file_paths: list[str],
    max_paths: int,
) -> list[str]:
    """
    Validate agent ``focus_paths`` for ``run_sandbox_analysis``.

    Returns human-readable errors; empty list means the request is acceptable.
    """
    if not focus_paths:
        return []
    if not isinstance(focus_paths, list):
        return [f"{repo_url}: focus_paths must be a list of {{path, max_lines?}} objects"]

    limit = max(1, int(max_paths))
    if len(focus_paths) > limit:
        return [
            f"{repo_url}: focus_paths has {len(focus_paths)} entries; "
            f"maximum is {limit} per repository"
        ]

    errors: list[str] = []
    for index, raw in enumerate(focus_paths):
        if not isinstance(raw, dict):
            errors.append(f"{repo_url}: focus_paths[{index}] must be an object with a path field")
            continue
        requested = str(raw.get("path") or "").strip()
        if not requested:
            errors.append(f"{repo_url}: focus_paths[{index}] is missing path")
            continue
        resolved, _substituted = resolve_focus_path(requested, file_paths)
        if not resolved:
            errors.append(f"{repo_url}: focus_paths path not found in repo tree: {requested}")
    return errors


def _resolve_agent_focus_items(
    agent_focus_paths: list[dict[str, Any]] | None,
    file_paths: list[str],
    *,
    max_files: int,
) -> list[dict[str, Any]]:
    """Resolve agent focus_paths to concrete repo-relative paths (deduped, capped)."""
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw in agent_focus_paths or []:
        if not isinstance(raw, dict):
            continue
        requested = str(raw.get("path") or "").strip()
        if not requested:
            continue
        path, substituted = resolve_focus_path(requested, file_paths)
        if not path or path in seen:
            continue
        seen.add(path)
        resolved.append(
            {
                "path": path,
                "max_lines": max(20, min(400, int(raw.get("max_lines") or 120))),
                "source": "agent",
                "requested_path": requested,
            }
        )
        if len(resolved) >= max_files:
            break
    return resolved


def merge_repo_focus_spec(
    *,
    file_paths: list[str],
    candidate_tags: list[str] | None,
    repo_role: RepoRole,
    agent_focus_paths: list[dict[str, Any]] | None = None,
    max_files: int = 12,
) -> dict[str, Any]:
    """
    Build sandbox file-focus spec.

    When the agent supplies at least one valid path, only those agent picks are used
    (no mandatory README/manifest padding and no heuristic rank). Otherwise falls back
    to mandatory + heuristic sampling for programmatic sandbox runs.
    """
    cap = max(1, int(max_files))
    agent_items = _resolve_agent_focus_items(
        agent_focus_paths,
        file_paths,
        max_files=cap,
    )

    if agent_items:
        return {
            "repo_role": repo_role,
            "max_files": cap,
            "pick_mode": "agent_only",
            "focus_paths": agent_items,
            "agent_focus_paths": agent_items,
            "file_paths": file_paths,
            "candidate_tags": list(candidate_tags or []),
        }

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_item(
        path: str, *, max_lines: int, source: str, requested_path: str | None = None
    ) -> None:
        if path in seen or len(merged) >= cap:
            return
        seen.add(path)
        merged.append(
            {
                "path": path,
                "max_lines": max(20, min(400, int(max_lines))),
                "source": source,
                "requested_path": requested_path or path,
            }
        )

    for item in build_mandatory_focus_paths(file_paths):
        add_item(item["path"], max_lines=item["max_lines"], source="mandatory")

    if repo_role != "orthogonal":
        for path in rank_paths_for_profile(
            file_paths, candidate_tags=candidate_tags, repo_role=repo_role
        ):
            add_item(path, max_lines=100, source="heuristic")
            if len(merged) >= cap:
                break

    return {
        "repo_role": repo_role,
        "max_files": cap,
        "pick_mode": "legacy",
        "focus_paths": merged,
        "agent_focus_paths": [],
        "file_paths": file_paths,
        "candidate_tags": list(candidate_tags or []),
    }


def select_evaluation_paths(
    file_paths: list[str],
    *,
    candidate_tags: list[str] | None,
    repo_role: RepoRole,
    agent_focus_paths: list[dict[str, Any]] | None = None,
    max_files: int = 5,
) -> list[str]:
    """
    Choose up to ``max_files`` code paths for top-file evaluation (no git-history ranking).

    Priority: agent-selected paths, then JD/role heuristic rank.
    """
    selected: list[str] = []
    seen: set[str] = set()

    for raw in agent_focus_paths or []:
        if not isinstance(raw, dict):
            continue
        requested = str(raw.get("path") or "").strip()
        if not requested:
            continue
        resolved, _substituted = resolve_focus_path(requested, file_paths)
        if resolved and resolved not in seen:
            seen.add(resolved)
            selected.append(resolved)
        if len(selected) >= max_files:
            return selected[:max_files]

    if selected:
        return selected[:max_files]

    for path in rank_paths_for_profile(
        file_paths,
        candidate_tags=candidate_tags,
        repo_role=repo_role,
    ):
        if path in seen:
            continue
        seen.add(path)
        selected.append(path)
        if len(selected) >= max_files:
            break

    return selected[:max_files]


def build_repo_structure_summary(
    *,
    repo_url: str,
    repo_name: str,
    file_paths: list[str],
    languages: dict[str, Any] | None,
    repo_type_tags: list[str] | None,
    candidate_tags: list[str] | None,
) -> dict[str, Any]:
    """Compact repo structure payload for the agent."""
    role = classify_repo_role(
        repo_type_tags=repo_type_tags,
        candidate_tags=candidate_tags,
        file_paths=file_paths,
    )
    top_dirs = sorted(
        {path.split("/", 1)[0] for path in file_paths if "/" in path and not path.startswith(".")}
    )[:20]
    code_paths = [
        path
        for path in file_paths
        if path.lower().endswith((".py", ".java", ".go", ".ts", ".js", ".rs", ".cs", ".rb", ".php"))
    ]
    return {
        "repo_url": repo_url,
        "repo_name": repo_name,
        "classification": role,
        "repo_type_tags": list(repo_type_tags or []),
        "languages": languages or {},
        "top_level_dirs": top_dirs,
        "file_count": len(file_paths),
        "code_file_count": len(code_paths),
        "mandatory_focus_paths": build_mandatory_focus_paths(file_paths),
        "suggested_focus_paths": rank_paths_for_profile(
            code_paths,
            candidate_tags=candidate_tags,
            repo_role=role,
        )[:8],
    }


def _css_html_ratio(paths: list[str]) -> float:
    if not paths:
        return 0.0
    code_paths = [
        path
        for path in paths
        if path.lower().endswith(
            (".py", ".java", ".go", ".ts", ".js", ".rs", ".cs", ".html", ".css")
        )
    ]
    if not code_paths:
        return 0.0
    html_css = sum(
        1 for path in code_paths if path.lower().endswith((".html", ".css", ".scss", ".sass"))
    )
    return html_css / len(code_paths)
