"""Merge sandbox top_files into final screening top_file_evaluation entries."""

from __future__ import annotations

from typing import Any

MATCH_SIGNALS = frozenset({"positive", "neutral", "negative"})


def iter_sandbox_top_files(github_repo_analyses: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Flatten top_files from all sandbox reports with repo metadata."""
    if not isinstance(github_repo_analyses, dict):
        return []

    reports = github_repo_analyses.get("sandbox_reports")
    if not isinstance(reports, list):
        return []

    flattened: list[dict[str, Any]] = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        profile = report.get("repo_profile")
        if not isinstance(profile, dict):
            continue
        top_files = profile.get("top_files")
        if not isinstance(top_files, list):
            continue

        repo = str(report.get("repo") or "")
        repo_url = str(report.get("url") or "")
        classification = report.get("classification") or profile.get("repo_role")

        for item in top_files:
            if not isinstance(item, dict) or not item.get("path"):
                continue
            flattened.append(
                {
                    **item,
                    "repo": repo,
                    "repo_url": repo_url,
                    "classification": classification,
                }
            )
    return flattened


def _entry_key(repo: str, repo_url: str, path: str) -> tuple[str, str]:
    return (repo_url or repo, str(path))


def _sanitize_jd_criteria(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text[:120])
    return out[:5]


def _normalize_match_signal(raw: Any, sandbox_item: dict[str, Any]) -> str:
    signal = str(raw or "").strip().lower()
    if signal in MATCH_SIGNALS:
        return signal

    status = str(sandbox_item.get("content_status") or "").lower()
    if status in ("stub", "empty", "vague", "missing", "too_large"):
        return "negative"
    if status == "ok":
        return "positive"
    return "neutral"


def _normalize_assessment(raw: Any, sandbox_item: dict[str, Any]) -> str:
    text = str(raw or "").strip()
    if text:
        return text[:500]

    path = str(sandbox_item.get("path") or "file")
    status = str(sandbox_item.get("content_status") or "unknown")
    tier = str(sandbox_item.get("compaction_tier") or "unknown")
    sent = sandbox_item.get("sent_lines")
    total = sandbox_item.get("total_lines")

    if status == "missing":
        return f"Focused path {path} was not available in the sandbox clone."
    if status == "too_large":
        return f"{path} exceeded the sandbox size limit; content was not inlined."
    if status in ("stub", "empty", "vague"):
        return (
            f"Focused review of {path} ({tier}) shows {status} content — "
            "limited evidence for role-aligned depth."
        )
    return (
        f"Reviewed {path} using {tier} compaction "
        f"({sent}/{total} lines shown) for JD-aligned signals."
    )[:500]


def _evidence_snippet(sandbox_item: dict[str, Any]) -> str:
    snippet = str(sandbox_item.get("content") or "").strip().replace("\n", " ")
    if not snippet:
        return "(empty)"
    return snippet[:200]


def merge_top_file_evaluation(
    raw_items: Any,
    github_repo_analyses: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    Build top_file_evaluation from sandbox top_files, enriched by optional agent submit rows.

    Sandbox file metadata and snippets are authoritative; agent supplies jd_criteria,
    match_signal, and assessment when provided.
    """
    sandbox_items = iter_sandbox_top_files(github_repo_analyses)
    if not sandbox_items:
        return []

    agent_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "").strip()
            if not path:
                continue
            key = _entry_key(
                str(item.get("repo") or ""),
                str(item.get("repo_url") or ""),
                path,
            )
            agent_by_key[key] = item

    merged: list[dict[str, Any]] = []
    for sandbox_item in sandbox_items:
        path = str(sandbox_item.get("path") or "")
        repo = str(sandbox_item.get("repo") or "")
        repo_url = str(sandbox_item.get("repo_url") or "")
        key = _entry_key(repo, repo_url, path)
        agent_item = agent_by_key.get(key) or agent_by_key.get(("", path))

        merged.append(
            {
                "repo": repo,
                "repo_url": repo_url,
                "path": path,
                "importance_rank": int(sandbox_item.get("importance_rank") or len(merged) + 1),
                "classification": str(sandbox_item.get("classification") or "unknown"),
                "language": str(sandbox_item.get("language") or "unknown"),
                "compaction_tier": str(sandbox_item.get("compaction_tier") or "unknown"),
                "total_lines": int(sandbox_item.get("total_lines") or 0),
                "sent_lines": int(sandbox_item.get("sent_lines") or 0),
                "content_status": str(sandbox_item.get("content_status") or "unknown"),
                "jd_criteria": _sanitize_jd_criteria(
                    agent_item.get("jd_criteria") if agent_item else []
                ),
                "match_signal": _normalize_match_signal(
                    agent_item.get("match_signal") if agent_item else None,
                    sandbox_item,
                ),
                "assessment": _normalize_assessment(
                    agent_item.get("assessment") if agent_item else None,
                    sandbox_item,
                ),
                "evidence_snippet": _evidence_snippet(sandbox_item),
            }
        )

    merged.sort(key=lambda item: int(item.get("importance_rank") or 0))
    return merged
