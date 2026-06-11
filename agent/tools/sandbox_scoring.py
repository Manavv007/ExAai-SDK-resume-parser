"""Deterministic score adjustments from sandbox repository reports."""

from __future__ import annotations

from typing import Any


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _repo_classification(report: dict[str, Any]) -> str:
    profile = report.get("repo_profile") if isinstance(report.get("repo_profile"), dict) else {}
    raw = str(report.get("classification") or profile.get("repo_role") or "unknown").strip().lower()
    if raw in ("aligned", "adjacent", "peripheral", "orthogonal"):
        return raw
    return "unknown"


def _classification_weight(role: str) -> float:
    return {
        "aligned": 1.0,
        "adjacent": 0.6,
        "peripheral": 0.2,
        "orthogonal": 0.0,
        "unknown": 0.4,
    }.get(role, 0.4)


def _vulnerability_total(report: dict[str, Any]) -> int:
    profile = report.get("repo_profile") if isinstance(report.get("repo_profile"), dict) else {}
    external = (
        profile.get("external_tool_signals")
        if isinstance(profile.get("external_tool_signals"), dict)
        else {}
    )
    total = 0
    for tool in ("pip_audit", "trivy", "npm_audit"):
        tool_summary = external.get(tool) if isinstance(external.get(tool), dict) else {}
        count = _int_or_none(tool_summary.get("vulnerability_count"))
        if count and count > 0:
            total += count
    return total


def repo_sandbox_risk_penalty(report: dict[str, Any]) -> int:
    """Estimate how many points to subtract for one sandboxed repository."""
    if not report.get("clone_ok"):
        return 0

    role = _repo_classification(report)
    weight = _classification_weight(role)
    if weight <= 0:
        return 0

    penalty = 0
    profile = report.get("repo_profile") if isinstance(report.get("repo_profile"), dict) else {}
    security = (
        profile.get("security_profile")
        if isinstance(profile.get("security_profile"), dict)
        else {}
    )
    findings = report.get("findings") if isinstance(report.get("findings"), list) else []

    if security.get("secret_hygiene") == "weak":
        penalty += 10

    secret_hits = _int_or_none(security.get("secret_pattern_hits"))
    if secret_hits is None:
        secret_hits = _int_or_none(profile.get("secret_pattern_hits"))
    if secret_hits and secret_hits > 0:
        penalty += min(15, secret_hits * 3)

    vuln_total = _vulnerability_total(report)
    if vuln_total >= 100:
        penalty += 30
    elif vuln_total >= 50:
        penalty += 20
    elif vuln_total >= 20:
        penalty += 12
    elif vuln_total >= 5:
        penalty += 6

    high_findings = sum(
        1 for item in findings if isinstance(item, dict) and item.get("severity") == "high"
    )
    penalty += high_findings * 6

    weighted = int(round(penalty * weight))
    per_repo_cap = 35 if weight >= 0.6 else 20
    return min(weighted, per_repo_cap)


def repo_sandbox_score_ceiling(report: dict[str, Any]) -> int | None:
    """Maximum overall candidate score when this repo shows severe aligned risk."""
    if not report.get("clone_ok"):
        return None

    role = _repo_classification(report)
    weight = _classification_weight(role)
    if weight < 0.5:
        return None

    profile = report.get("repo_profile") if isinstance(report.get("repo_profile"), dict) else {}
    security = (
        profile.get("security_profile")
        if isinstance(profile.get("security_profile"), dict)
        else {}
    )
    findings = report.get("findings") if isinstance(report.get("findings"), list) else []
    vuln_total = _vulnerability_total(report)
    high_findings = sum(
        1 for item in findings if isinstance(item, dict) and item.get("severity") == "high"
    )
    weak_secrets = security.get("secret_hygiene") == "weak"

    if role == "aligned":
        if vuln_total >= 100 or (weak_secrets and vuln_total >= 20):
            return 65
        if vuln_total >= 50:
            return 68
        if vuln_total >= 20:
            return 72
        if high_findings >= 1 and weak_secrets:
            return 68
    elif role == "adjacent":
        if vuln_total >= 100:
            return 68
        if vuln_total >= 50:
            return 72

    return None


def compute_sandbox_score_penalty(sandbox_reports: list[dict[str, Any]]) -> int:
    """Blend per-repo penalties, weighting the worst repo most heavily."""
    penalties = [repo_sandbox_risk_penalty(report) for report in sandbox_reports]
    penalties = [value for value in penalties if value > 0]
    if not penalties:
        return 0

    worst = max(penalties)
    average = sum(penalties) / len(penalties)
    return max(0, min(40, int(round((0.7 * worst) + (0.3 * average)))))


def compute_sandbox_score_ceiling(sandbox_reports: list[dict[str, Any]]) -> int | None:
    """Lowest per-repo ceiling across aligned/adjacent repos (most restrictive wins)."""
    ceilings = [
        ceiling
        for report in sandbox_reports
        if (ceiling := repo_sandbox_score_ceiling(report)) is not None
    ]
    return min(ceilings) if ceilings else None


def apply_sandbox_score_penalty(
    score: int,
    github_repo_analyses: dict[str, Any] | None,
) -> tuple[int, int]:
    """Return adjusted score and total points removed (penalty + ceiling)."""
    if not isinstance(github_repo_analyses, dict):
        return score, 0

    reports = github_repo_analyses.get("sandbox_reports")
    if not isinstance(reports, list) or not reports:
        return score, 0

    penalty = compute_sandbox_score_penalty(reports)
    ceiling = compute_sandbox_score_ceiling(reports)

    adjusted = max(0, score - penalty) if penalty > 0 else score
    if ceiling is not None:
        adjusted = min(adjusted, ceiling)

    total_reduction = max(0, score - adjusted)
    if total_reduction <= 0:
        return score, 0

    return adjusted, total_reduction


_SANDBOX_PENALTY_NOTE = "Sandbox repo review reduced the score by"


def reconcile_sandbox_penalty_in_result(
    result: dict[str, Any],
    github_repo_analyses: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply sandbox penalty when reports exist but normalize missed them (stale state merge)."""
    if not isinstance(result, dict):
        return result
    if not isinstance(github_repo_analyses, dict):
        return result

    reports = github_repo_analyses.get("sandbox_reports")
    if not isinstance(reports, list) or not reports:
        return result

    similarity = result.get("resume_similarity_score")
    if not isinstance(similarity, dict):
        return result

    reasoning = str(similarity.get("reasoning") or "")
    if _SANDBOX_PENALTY_NOTE in reasoning:
        return result

    score = similarity.get("score")
    if not isinstance(score, int):
        return result

    adjusted, reduction = apply_sandbox_score_penalty(score, github_repo_analyses)
    if reduction <= 0:
        return result

    ceiling = compute_sandbox_score_ceiling(reports)
    penalty_note = (
        f" {_SANDBOX_PENALTY_NOTE} {reduction} points "
        "due to engineering-risk signals (secrets, vulnerabilities, or high-severity findings)."
    )
    if ceiling is not None and score - reduction < ceiling and adjusted == ceiling:
        penalty_note += f" Score capped at {ceiling} for severe aligned-repo risk."

    similarity["score"] = adjusted
    similarity["reasoning"] = (reasoning + penalty_note).strip()[:500]

    if adjusted >= 75 and result.get("recommendation") == "hold":
        result["recommendation"] = "advance"
    if adjusted < 60 and result.get("recommendation") == "advance":
        result["recommendation"] = "hold"

    return result
