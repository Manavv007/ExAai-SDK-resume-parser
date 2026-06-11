"""Select and compact top evaluation files from a cloned repository."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.sandbox.evaluator.file_compactor import DEFAULT_CHAR_CAP, compact_file_content
from agent.sandbox.evaluator.filesystem_scanner import MAX_SAMPLE_FILE_BYTES, read_text_if_exists
from agent.tools.repo_focus import classify_content_quality, select_evaluation_paths


def collect_top_files(
    repo_dir: Path,
    focus_spec: dict[str, Any] | None = None,
    *,
    max_files: int = 5,
    char_cap: int = DEFAULT_CHAR_CAP,
) -> list[dict[str, Any]]:
    """
    Pick up to ``max_files`` role-relevant paths and return compacted content per file.

    Path priority (no git-history / yek ranking):
    1. Agent-provided focus_paths (source=agent)
    2. JD/role heuristic rank (candidate_tags + repo_role)
    """
    spec = focus_spec if isinstance(focus_spec, dict) else {}
    file_paths = list(spec.get("file_paths") or [])
    if not file_paths:
        file_paths = _list_repo_relative_paths(repo_dir)

    agent_focus = [
        item
        for item in (spec.get("focus_paths") or [])
        if isinstance(item, dict) and str(item.get("source") or "") == "agent"
    ]
    if not agent_focus:
        agent_focus = spec.get("agent_focus_paths") if isinstance(spec.get("agent_focus_paths"), list) else []

    limit = int(spec.get("top_files_count") or max_files or 5)
    selected_paths = select_evaluation_paths(
        file_paths,
        candidate_tags=list(spec.get("candidate_tags") or []),
        repo_role=str(spec.get("repo_role") or "peripheral"),
        agent_focus_paths=agent_focus,
        max_files=limit,
    )

    top_files: list[dict[str, Any]] = []
    for rank, rel_path in enumerate(selected_paths, start=1):
        path = repo_dir / rel_path
        if not path.is_file():
            top_files.append(
                {
                    "path": rel_path,
                    "importance_rank": rank,
                    "language": _language_label(rel_path),
                    "total_lines": 0,
                    "sent_lines": 0,
                    "compaction_tier": "missing",
                    "content_status": "missing",
                    "truncated": False,
                    "content": "",
                }
            )
            continue

        try:
            if path.stat().st_size > MAX_SAMPLE_FILE_BYTES:
                top_files.append(
                    {
                        "path": rel_path,
                        "importance_rank": rank,
                        "language": _language_label(rel_path),
                        "total_lines": 0,
                        "sent_lines": 0,
                        "compaction_tier": "skipped",
                        "content_status": "too_large",
                        "truncated": False,
                        "content": "",
                    }
                )
                continue
        except OSError:
            continue

        raw = read_text_if_exists(path)
        compact = compact_file_content(raw, rel_path, char_cap=char_cap)
        top_files.append(
            {
                "path": rel_path,
                "importance_rank": rank,
                "language": _language_label(rel_path),
                "total_lines": compact["total_lines"],
                "sent_lines": compact["sent_lines"],
                "compaction_tier": compact["compaction_tier"],
                "content_status": classify_content_quality(raw),
                "truncated": compact["truncated"],
                "content": compact["content"],
            }
        )
    return top_files


def _language_label(path: str) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix or "unknown"


def _list_repo_relative_paths(repo_dir: Path) -> list[str]:
    paths: list[str] = []
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(repo_dir)).replace("\\", "/")
        if any(part in rel for part in ("/node_modules/", "/.git/", "/dist/", "/build/")):
            continue
        paths.append(rel)
    return paths
