"""In-process prep state registry for ADK tool calls.

ADK tool_context.state may omit large prep fields (rubric, jd_structured, jd_raw).
Submit and post-agent repair merge the original prep snapshot by application_id.
"""

from __future__ import annotations

from typing import Any

_PREP_BY_APPLICATION: dict[str, dict[str, Any]] = {}


def session_state_to_dict(state: Any) -> dict[str, Any]:
    """Convert ADK session/tool state objects into a plain dict."""
    if isinstance(state, dict):
        return dict(state)

    inner = getattr(state, "_value", None)
    if isinstance(inner, dict):
        return dict(inner)

    to_dict = getattr(state, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, dict):
            return dict(converted)

    get = getattr(state, "get", None)
    if callable(get):
        keys = getattr(state, "keys", None)
        if callable(keys):
            try:
                return {key: get(key) for key in keys()}
            except Exception:
                pass

    return {}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if value == [] or value == {}:
        return True
    return False


def merge_github_repo_analyses(
    prep_github: dict[str, Any] | None,
    session_github: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge GitHub analysis blobs, preferring the richer sandbox_reports list."""
    if not isinstance(prep_github, dict) and not isinstance(session_github, dict):
        return None
    merged: dict[str, Any] = dict(prep_github or {})
    if isinstance(session_github, dict):
        merged.update(session_github)

    prep_reports = (
        prep_github.get("sandbox_reports")
        if isinstance(prep_github, dict) and isinstance(prep_github.get("sandbox_reports"), list)
        else []
    )
    session_reports = (
        session_github.get("sandbox_reports")
        if isinstance(session_github, dict)
        and isinstance(session_github.get("sandbox_reports"), list)
        else []
    )
    if len(session_reports) >= len(prep_reports) and session_reports:
        merged["sandbox_reports"] = session_reports
    elif prep_reports:
        merged["sandbox_reports"] = prep_reports

    prep_discovered = (
        prep_github.get("discovered_github_repo_urls")
        if isinstance(prep_github, dict)
        and isinstance(prep_github.get("discovered_github_repo_urls"), list)
        else []
    )
    session_discovered = (
        session_github.get("discovered_github_repo_urls")
        if isinstance(session_github, dict)
        and isinstance(session_github.get("discovered_github_repo_urls"), list)
        else []
    )
    if session_discovered or prep_discovered:
        seen: set[str] = set()
        merged_discovered: list[str] = []
        for raw in prep_discovered + session_discovered:
            url = str(raw or "").strip()
            if not url:
                continue
            lowered = url.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            merged_discovered.append(url)
        merged["discovered_github_repo_urls"] = merged_discovered
    return merged


def register_prep_state(state: Any) -> None:
    state_dict = session_state_to_dict(state)
    application_id = str(state_dict.get("application_id") or "").strip()
    if application_id:
        _PREP_BY_APPLICATION[application_id] = state_dict


def get_prep_state(application_id: str) -> dict[str, Any] | None:
    application_id = application_id.strip()
    if not application_id:
        return None
    prep = _PREP_BY_APPLICATION.get(application_id)
    return dict(prep) if prep else None


def clear_prep_state(application_id: str) -> None:
    _PREP_BY_APPLICATION.pop(application_id.strip(), None)


def merge_with_prep_state(state: Any) -> dict[str, Any]:
    """Overlay tool/session state on the prep snapshot, keeping prep JD/rubric when missing."""
    state_dict = session_state_to_dict(state)
    application_id = str(state_dict.get("application_id") or "")
    prep = get_prep_state(application_id)
    if not prep:
        return state_dict

    merged = dict(prep)
    for key, value in state_dict.items():
        if key == "screening_result":
            continue
        if key == "github_repo_analyses":
            continue
        if _is_missing(value) and not _is_missing(merged.get(key)):
            continue
        merged[key] = value

    merged_github = merge_github_repo_analyses(
        prep.get("github_repo_analyses")
        if isinstance(prep.get("github_repo_analyses"), dict)
        else None,
        state_dict.get("github_repo_analyses")
        if isinstance(state_dict.get("github_repo_analyses"), dict)
        else None,
    )
    if merged_github is not None:
        merged["github_repo_analyses"] = merged_github
    return merged
