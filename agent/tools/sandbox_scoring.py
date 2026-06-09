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


def repo_sandbox_risk_penalty(report: dict[str, Any]) -> int:
    """Estimate how many points to subtract for one sandboxed repository."""
    if not report.get("clone_ok"):
        return 0

    penalty = 0
    profile = report.get("repo_profile") if isinstance(report.get("repo_profile"), dict) else {}
    security = (
        profile.get("security_profile")
        if isinstance(profile.get("security_profile"), dict)
        else {}
    )
    quality_raw = report.get("quality_signals")
    quality = quality_raw if isinstance(quality_raw, dict) else {}
    findings = report.get("findings") if isinstance(report.get("findings"), list) else []
    external = (
        profile.get("external_tool_signals")
        if isinstance(profile.get("external_tool_signals"), dict)
        else {}
    )

    if security.get("secret_hygiene") == "weak":
        penalty += 8

    secret_hits = _int_or_none(security.get("secret_pattern_hits"))
    if secret_hits is None:
        secret_hits = _int_or_none(profile.get("secret_pattern_hits"))
    if secret_hits and secret_hits > 0:
        penalty += min(12, secret_hits * 3)

    vuln_total = 0
    for tool in ("pip_audit", "trivy", "npm_audit"):
        tool_summary = external.get(tool) if isinstance(external.get(tool), dict) else {}
        count = _int_or_none(tool_summary.get("vulnerability_count"))
        if count and count > 0:
            vuln_total += count

    if vuln_total >= 50:
        penalty += 15
    elif vuln_total >= 20:
        penalty += 10
    elif vuln_total >= 5:
        penalty += 5

    if not quality.get("has_tests"):
        penalty += 3
    if not quality.get("has_ci"):
        penalty += 2

    high_findings = sum(
        1 for item in findings if isinstance(item, dict) and item.get("severity") == "high"
    )
    penalty += high_findings * 5

    return min(penalty, 25)


def compute_sandbox_score_penalty(sandbox_reports: list[dict[str, Any]]) -> int:
    """Blend per-repo penalties, weighting the worst repo most heavily."""
    penalties = [repo_sandbox_risk_penalty(report) for report in sandbox_reports]
    penalties = [value for value in penalties if value > 0]
    if not penalties:
        return 0

    worst = max(penalties)
    average = sum(penalties) / len(penalties)
    return max(0, min(30, int(round((0.7 * worst) + (0.3 * average)))))


def apply_sandbox_score_penalty(
    score: int,
    github_repo_analyses: dict[str, Any] | None,
) -> tuple[int, int]:
    """Return adjusted score and penalty points applied."""
    if not isinstance(github_repo_analyses, dict):
        return score, 0

    reports = github_repo_analyses.get("sandbox_reports")
    if not isinstance(reports, list) or not reports:
        return score, 0

    penalty = compute_sandbox_score_penalty(reports)
    if penalty <= 0:
        return score, 0

    return max(0, score - penalty), penalty
