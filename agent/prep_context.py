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


def register_prep_state(state: dict[str, Any]) -> None:
    application_id = str(state.get("application_id") or "").strip()
    if application_id:
        _PREP_BY_APPLICATION[application_id] = dict(state)


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
        if _is_missing(value) and not _is_missing(merged.get(key)):
            continue
        merged[key] = value
    return merged
