"""Deterministic GitHub repo portfolio and code-quality scoring for final evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from agent.tools.rubric_builder import derive_overall_score_from_matches
from agent.tools.sandbox_scoring import (
    _classification_weight,
    _repo_classification,
    compute_sandbox_score_ceiling,
    compute_sandbox_score_penalty,
)

_CLASSIFICATION_PORTFOLIO_WEIGHT = {
    "aligned": 1.0,
    "adjacent": 0.6,
    "peripheral": 0.2,
    "orthogonal": 0.0,
    "unknown": 0.4,
}

_CODE_QUALITY_WEIGHTS: dict[str, int] = {
    "type_annotations": 10,
    "error_handling": 15,
    "secrets": 20,
    "complexity": 15,
    "lint": 10,
    "merge_hygiene": 10,
    "ci": 10,
    "file_substance": 10,
}

_CONTENT_STATUS_SCORES = {
    "ok": 1.0,
    "stub": 0.3,
    "vague": 0.3,
    "empty": 0.0,
    "missing": 0.0,
}

_JD_FIT_WEIGHT = 0.55
_PORTFOLIO_WEIGHT = 0.20
_CODE_QUALITY_WEIGHT = 0.25


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_or_zero(value: Any) -> int:
    parsed = _float_or_none(value)
    return int(parsed) if parsed is not None else 0


def recency_score(days_since_last_commit: int) -> int:
    if days_since_last_commit < 30:
        return 100
    if days_since_last_commit < 180:
        return 60
    if days_since_last_commit < 365:
        return 30
    return 10


def _days_active_estimate(
    *,
    commit_count: int,
    days_since_last_commit: int,
    github_metadata: dict[str, Any] | None,
) -> int:
    meta = github_metadata or {}
    created = meta.get("created_at")
    pushed = meta.get("last_push_at") or meta.get("pushed_at")
    for label, raw in (("created", created), ("pushed", pushed)):
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            normalized = raw.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            if label == "created" and isinstance(pushed, str) and pushed.strip():
                pushed_dt = datetime.fromisoformat(pushed.replace("Z", "+00:00"))
                if pushed_dt.tzinfo is None:
                    pushed_dt = pushed_dt.replace(tzinfo=UTC)
                return max(1, (pushed_dt - dt).days)
            if label == "created":
                return max(1, (datetime.now(UTC) - dt).days)
        except ValueError:
            continue
    if commit_count > 0:
        return min(365, max(30, commit_count * 7))
    return max(1, min(365, 365 - days_since_last_commit))


def compute_ownership_confidence(
    *,
    git_profile: dict[str, Any],
    is_fork: bool,
    candidate_commit_ratio: float | None = None,
) -> float:
    ratio = candidate_commit_ratio
    if ratio is None:
        ratio = _float_or_none(git_profile.get("top_author_commit_share")) or 0.0
    solo_bonus = 10.0 if ratio > 0.85 else 0.0
    confidence = (ratio * 0.80) + ((solo_bonus / 10.0) * 0.20)
    if git_profile.get("history_is_shallow"):
        confidence *= 0.85
    confidence = _clamp(confidence, 0.0, 1.0)
    if is_fork:
        confidence = min(confidence, 0.30)
    return round(confidence, 4)


def compute_activity_score(
    *,
    git_profile: dict[str, Any],
    github_metadata: dict[str, Any] | None = None,
) -> int:
    commit_count = _int_or_zero(git_profile.get("commit_count"))
    days_since = _int_or_zero(git_profile.get("days_since_last_commit"))
    days_active = _days_active_estimate(
        commit_count=commit_count,
        days_since_last_commit=days_since,
        github_metadata=github_metadata,
    )
    commit_component = min(commit_count / 30.0, 1.0) * 40.0
    active_component = min(days_active / 365.0, 1.0) * 30.0
    recency_component = recency_score(days_since) * 0.30
    score = commit_component + active_component + recency_component
    if commit_count < 15 and recency_score(days_since) >= 60:
        score = max(score, 40.0)
    return int(round(_clamp(score)))


def _documentation_content_quality(documentation_profile: dict[str, Any]) -> int:
    if not documentation_profile.get("readme_present"):
        return 0
    points = 0
    if documentation_profile.get("has_setup_instructions"):
        points += 25
    if documentation_profile.get("has_architecture_section"):
        points += 25
    if documentation_profile.get("has_docs_dir"):
        points += 10
    readme_bytes = _int_or_zero(documentation_profile.get("readme_bytes"))
    if readme_bytes > 3000:
        points += 20
    elif readme_bytes > 1000:
        points += 12
    elif readme_bytes > 200:
        points += 6
    return min(70, points)


def compute_documentation_score(
    *,
    documentation_profile: dict[str, Any],
    has_license: bool = False,
) -> int:
    readme_points = 25 if documentation_profile.get("readme_present") else 0
    content_quality = _documentation_content_quality(documentation_profile)
    license_points = 5 if has_license else 0
    return int(round(_clamp(readme_points + content_quality + license_points)))


def compute_collaborators_score(
    *,
    unique_collaborators: int | None,
    git_profile: dict[str, Any] | None = None,
) -> int:
    count = unique_collaborators
    if count is None and git_profile:
        authors = _int_or_zero(git_profile.get("unique_authors"))
        count = max(0, authors - 1)
    count = count or 0
    if count <= 1:
        return 50
    return int(round(_clamp(min(count / 3.0, 1.0) * 100.0)))


def compute_repo_portfolio_raw_score(
    *,
    activity_score: int,
    documentation_score: int,
    collaborators_score: int,
) -> int:
    raw = (
        activity_score * 0.35
        + documentation_score * 0.40
        + collaborators_score * 0.25
    )
    return int(round(_clamp(raw)))


def _normalize_type_annotations(ratio: float | None) -> float | None:
    if ratio is None:
        return None
    return _clamp(ratio, 0.0, 1.0)


def _normalize_error_handling(density: float | None) -> float | None:
    if density is None:
        return None
    return _clamp(min(density / 0.05, 1.0), 0.0, 1.0)


def _normalize_secrets(secret_hits: int) -> float:
    if secret_hits <= 0:
        return 1.0
    return max(0.0, 1.0 - (secret_hits * 0.2))


def _normalize_complexity(avg_cc: float | None) -> float | None:
    if avg_cc is None:
        return None
    return max(0.0, 1.0 - ((avg_cc - 1.0) / 9.0))


def _normalize_lint(violations_per_kloc: float | None) -> float | None:
    if violations_per_kloc is None:
        return None
    return max(0.0, 1.0 - (violations_per_kloc / 10.0))


def _normalize_merge_ratio(ratio: float | None) -> float | None:
    if ratio is None:
        return None
    return _clamp(min(ratio / 0.2, 1.0), 0.0, 1.0)


def _file_substance_score(top_files: list[Any]) -> float | None:
    if not top_files:
        return None
    scores: list[float] = []
    for item in top_files:
        if not isinstance(item, dict):
            continue
        status = str(item.get("content_status") or "unknown").strip().lower()
        scores.append(_CONTENT_STATUS_SCORES.get(status, 0.5))
    if not scores:
        return None
    return sum(scores) / len(scores)


def compute_code_quality_score(
    repo_profile: dict[str, Any],
    *,
    top_files: list[Any] | None = None,
) -> dict[str, Any]:
    """Domain-agnostic code quality score when sandbox code metrics exist."""
    code_metrics = (
        repo_profile.get("code_metrics")
        if isinstance(repo_profile.get("code_metrics"), dict)
        else {}
    )
    security = (
        repo_profile.get("security_profile")
        if isinstance(repo_profile.get("security_profile"), dict)
        else {}
    )
    git_profile = (
        repo_profile.get("git_profile")
        if isinstance(repo_profile.get("git_profile"), dict)
        else {}
    )
    files = top_files
    if files is None:
        raw_top = repo_profile.get("top_files")
        files = raw_top if isinstance(raw_top, list) else []

    secret_hits = _int_or_zero(
        security.get("secret_pattern_hits") or repo_profile.get("secret_pattern_hits")
    )

    components: dict[str, float | None] = {
        "type_annotations": _normalize_type_annotations(
            _float_or_none(code_metrics.get("type_annotation_ratio"))
        ),
        "error_handling": _normalize_error_handling(
            _float_or_none(code_metrics.get("error_handling_density"))
        ),
        "secrets": _normalize_secrets(secret_hits),
        "complexity": _normalize_complexity(
            _float_or_none(
                code_metrics.get("avg_cyclomatic_complexity")
                or repo_profile.get("avg_cyclomatic_complexity")
            )
        ),
        "lint": _normalize_lint(
            _float_or_none(
                code_metrics.get("lint_violations_per_kloc")
                or repo_profile.get("lint_violations_per_kloc")
            )
        ),
        "merge_hygiene": _normalize_merge_ratio(
            _float_or_none(git_profile.get("merge_to_commit_ratio"))
        ),
        "ci": 1.0 if repo_profile.get("has_ci") else 0.0,
        "file_substance": _file_substance_score(files or []),
    }

    available_weight = 0
    weighted_sum = 0.0
    component_scores: dict[str, float | None] = {}
    for key, weight in _CODE_QUALITY_WEIGHTS.items():
        normalized = components.get(key)
        if normalized is None:
            component_scores[key] = None
            continue
        component_scores[key] = round(normalized * 100.0, 2)
        weighted_sum += normalized * weight
        available_weight += weight

    bonus = 0
    classification = str(repo_profile.get("repo_role") or "unknown").lower()
    if classification in ("aligned", "adjacent"):
        if repo_profile.get("has_tests"):
            bonus += 3
        if repo_profile.get("has_docker"):
            bonus += 2

    if available_weight <= 0:
        return {
            "code_quality_score": None,
            "components": component_scores,
            "bonus": bonus,
            "available_weight": 0,
        }

    base = (weighted_sum / available_weight) * 100.0
    score = int(round(_clamp(base + bonus)))
    return {
        "code_quality_score": score,
        "components": component_scores,
        "bonus": bonus,
        "available_weight": available_weight,
    }


def _repo_github_context(
    report: dict[str, Any],
    github_repo_analyses: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(github_repo_analyses, dict):
        return {}
    url = str(report.get("url") or "").rstrip("/").lower()
    repo_name = str(report.get("repo") or "").lower()
    for item in github_repo_analyses.get("repo_analyses") or []:
        if not isinstance(item, dict):
            continue
        item_url = str(item.get("url") or "").rstrip("/").lower()
        if item_url == url or item_url.endswith(f"/{repo_name}"):
            return item
    return {}


def score_sandbox_report(
    report: dict[str, Any],
    *,
    github_repo_analyses: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not report.get("clone_ok"):
        return None

    profile = report.get("repo_profile") if isinstance(report.get("repo_profile"), dict) else {}
    git_profile = profile.get("git_profile") if isinstance(profile.get("git_profile"), dict) else {}
    documentation_profile = (
        profile.get("documentation_profile")
        if isinstance(profile.get("documentation_profile"), dict)
        else {}
    )
    github_ctx = _repo_github_context(report, github_repo_analyses)
    github_meta = (
        github_ctx.get("github_metadata")
        if isinstance(github_ctx.get("github_metadata"), dict)
        else {}
    )
    is_fork = bool(github_ctx.get("is_fork"))
    has_license = bool(github_meta.get("license_present"))
    if not has_license:
        has_license = bool(profile.get("has_license"))

    contributors = github_meta.get("contributors_count")
    unique_collaborators = (
        int(contributors) if isinstance(contributors, int) and contributors >= 0 else None
    )

    ownership_confidence = compute_ownership_confidence(
        git_profile=git_profile,
        is_fork=is_fork,
    )
    activity_score = compute_activity_score(
        git_profile=git_profile,
        github_metadata=github_meta,
    )
    documentation_score = compute_documentation_score(
        documentation_profile=documentation_profile,
        has_license=has_license,
    )
    collaborators_score = compute_collaborators_score(
        unique_collaborators=unique_collaborators,
        git_profile=git_profile,
    )
    repo_raw_score = compute_repo_portfolio_raw_score(
        activity_score=activity_score,
        documentation_score=documentation_score,
        collaborators_score=collaborators_score,
    )
    repo_final_score = int(round(repo_raw_score * ownership_confidence))

    role = _repo_classification(report)
    profile_with_role = dict(profile)
    profile_with_role["repo_role"] = role
    code_quality = compute_code_quality_score(
        profile_with_role,
        top_files=profile.get("top_files") if isinstance(profile.get("top_files"), list) else [],
    )

    return {
        "url": report.get("url") or report.get("repo"),
        "repo": report.get("repo"),
        "classification": role,
        "ownership_confidence": ownership_confidence,
        "activity_score": activity_score,
        "documentation_score": documentation_score,
        "collaborators_score": collaborators_score,
        "repo_raw_score": repo_raw_score,
        "repo_final_score": repo_final_score,
        "code_quality_score": code_quality.get("code_quality_score"),
        "code_quality_components": code_quality.get("components"),
        "code_quality_bonus": code_quality.get("bonus"),
        "is_fork": is_fork,
    }


def _portfolio_aggregate(repo_scores: list[dict[str, Any]]) -> int | None:
    weighted: list[tuple[float, float]] = []
    for item in repo_scores:
        role = str(item.get("classification") or "unknown").lower()
        weight = _CLASSIFICATION_PORTFOLIO_WEIGHT.get(
            role,
            _classification_weight(role),
        )
        if weight <= 0:
            continue
        final = item.get("repo_final_score")
        if not isinstance(final, int):
            continue
        weighted.append((float(final), weight))
    if not weighted:
        return None
    total_weight = sum(weight for _, weight in weighted)
    if total_weight <= 0:
        return None
    value = sum(score * weight for score, weight in weighted) / total_weight
    return int(round(_clamp(value)))


def _code_quality_aggregate(repo_scores: list[dict[str, Any]]) -> int | None:
    weighted: list[tuple[float, float]] = []
    for item in repo_scores:
        role = str(item.get("classification") or "unknown").lower()
        if role not in ("aligned", "adjacent"):
            continue
        weight = _CLASSIFICATION_PORTFOLIO_WEIGHT.get(role, 0.0)
        score = item.get("code_quality_score")
        if not isinstance(score, int):
            continue
        weighted.append((float(score), weight))
    if not weighted:
        return None
    total_weight = sum(weight for _, weight in weighted)
    if total_weight <= 0:
        return None
    value = sum(score * weight for score, weight in weighted) / total_weight
    return int(round(_clamp(value)))


def build_evaluation_breakdown(
    *,
    requirement_matches: list[dict[str, Any]],
    rubric: list[Any],
    github_repo_analyses: dict[str, Any] | None,
    jd_fit_score: int | None = None,
) -> dict[str, Any] | None:
    """Build portfolio + code-quality breakdown and composite score."""
    if jd_fit_score is None:
        jd_fit_score = derive_overall_score_from_matches(requirement_matches, rubric)
    if jd_fit_score <= 0:
        return None

    reports: list[dict[str, Any]] = []
    if isinstance(github_repo_analyses, dict):
        raw = github_repo_analyses.get("sandbox_reports")
        if isinstance(raw, list):
            reports = [item for item in raw if isinstance(item, dict)]

    repo_scores = [
        scored
        for report in reports
        if (scored := score_sandbox_report(report, github_repo_analyses=github_repo_analyses))
    ]

    portfolio_score = _portfolio_aggregate(repo_scores)
    code_quality_score = _code_quality_aggregate(repo_scores)
    sandbox_penalty = compute_sandbox_score_penalty(reports) if reports else 0

    if portfolio_score is None and code_quality_score is None and not repo_scores:
        return {
            "jd_fit_score": jd_fit_score,
            "repo_portfolio_score": None,
            "code_quality_score": None,
            "sandbox_penalty": sandbox_penalty,
            "composite_score": jd_fit_score,
            "repos": [],
        }

    portfolio_component = float(portfolio_score if portfolio_score is not None else jd_fit_score)
    code_component = float(
        code_quality_score if code_quality_score is not None else portfolio_component
    )
    composite_base = (
        jd_fit_score * _JD_FIT_WEIGHT
        + portfolio_component * _PORTFOLIO_WEIGHT
        + code_component * _CODE_QUALITY_WEIGHT
    )
    composite_after_penalty = max(0.0, composite_base - float(sandbox_penalty))
    ceiling = compute_sandbox_score_ceiling(reports) if reports else None
    if ceiling is not None:
        composite_after_penalty = min(composite_after_penalty, float(ceiling))

    ownership_values = [
        float(item["ownership_confidence"])
        for item in repo_scores
        if isinstance(item.get("ownership_confidence"), (int, float))
    ]
    ownership_avg = (
        round(sum(ownership_values) / len(ownership_values), 4) if ownership_values else None
    )

    return {
        "jd_fit_score": jd_fit_score,
        "repo_portfolio_score": portfolio_score,
        "code_quality_score": code_quality_score,
        "sandbox_penalty": sandbox_penalty,
        "risk_ceiling": ceiling,
        "ownership_multiplier_avg": ownership_avg,
        "composite_score": int(round(_clamp(composite_after_penalty))),
        "blend_weights": {
            "jd_fit": _JD_FIT_WEIGHT,
            "repo_portfolio": _PORTFOLIO_WEIGHT,
            "code_quality": _CODE_QUALITY_WEIGHT,
        },
        "repos": repo_scores,
    }


def resolve_score_from_evaluation_breakdown(
    llm_or_derived_score: int,
    breakdown: dict[str, Any] | None,
) -> tuple[int, str]:
    """
    Prefer composite score when repo metrics exist; otherwise keep LLM/rubric score.

    Sandbox penalty and ceiling are already folded into composite_score.
    """
    if not breakdown or not isinstance(breakdown.get("composite_score"), int):
        return llm_or_derived_score, "llm_or_rubric"

    repos = breakdown.get("repos")
    if not isinstance(repos, list) or not repos:
        return llm_or_derived_score, "llm_or_rubric"

    return int(breakdown["composite_score"]), "evaluation_composite"
