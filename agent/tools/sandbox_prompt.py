"""Compact sandbox report summaries for LLM scoring prompts."""

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


def _vulnerability_count(report: dict[str, Any]) -> int:
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


def format_sandbox_reports_for_prompt(
    reports: list[dict[str, Any]],
    *,
    max_chars: int | None = None,
) -> str:
    """Human-readable sandbox digest for agent/scorer prompts."""
    if not reports:
        return "(none)"

    lines: list[str] = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        profile = report.get("repo_profile") if isinstance(report.get("repo_profile"), dict) else {}
        security = (
            profile.get("security_profile")
            if isinstance(profile.get("security_profile"), dict)
            else {}
        )
        findings = report.get("findings") if isinstance(report.get("findings"), list) else []
        high_findings = [
            item.get("title")
            for item in findings
            if isinstance(item, dict) and item.get("severity") == "high"
        ]
        warn_findings = [
            item.get("title")
            for item in findings
            if isinstance(item, dict) and item.get("severity") == "warn"
        ]

        role = report.get("classification") or profile.get("repo_role") or "unknown"
        vuln_count = _vulnerability_count(report)
        risk_tier = _risk_tier(role, vuln_count, security, len(high_findings))
        sample_lines = _format_sample_files(profile.get("sample_files"))
        top_file_lines = _format_top_files(profile.get("top_files"))
        eval_mode = report.get("evaluation_mode") or (
            (report.get("repo_profile") or {}).get("evaluation_mode")
        )
        mode_note = " [risk-only: vulns/secrets, no file excerpts]" if eval_mode == "risk_only" else ""
        lines.append(
            f"- {report.get('repo') or report.get('url')}{mode_note}\n"
            f"  classification={role}, risk_tier={risk_tier}, clone_ok={report.get('clone_ok')}, "
            f"secret_hygiene={security.get('secret_hygiene')}, "
            f"secret_hits={security.get('secret_pattern_hits') or profile.get('secret_pattern_hits') or 0}, "
            f"combined_vulnerabilities={vuln_count}, "
            f"high_findings={len(high_findings)}, "
            f"warnings={len(warn_findings)}\n"
            f"  assessment: {report.get('overall_assessment') or report.get('summary') or 'n/a'}\n"
            f"  notable_high: {high_findings[:3] or 'none'}\n"
            f"  notable_warn: {warn_findings[:3] or 'none'}\n"
            f"{sample_lines}\n"
            f"{top_file_lines}"
        )
    text = "\n".join(lines) if lines else "(none)"
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars - 20].rstrip() + "\n...(truncated)"
    return text


def _risk_tier(
    role: str,
    vuln_count: int,
    security: dict[str, Any],
    high_findings: int,
) -> str:
    role_l = str(role or "unknown").lower()
    weak_secrets = security.get("secret_hygiene") == "weak"
    if role_l in ("aligned", "adjacent"):
        if vuln_count >= 100 or (weak_secrets and vuln_count >= 20):
            return "CRITICAL"
        if vuln_count >= 50 or (weak_secrets and high_findings >= 1):
            return "SEVERE"
        if vuln_count >= 20 or high_findings >= 1:
            return "ELEVATED"
    if vuln_count >= 5:
        return "MODERATE"
    return "LOW"


def _format_top_files(top_files: Any) -> str:
    if not isinstance(top_files, list) or not top_files:
        return "  top_files: none"
    lines = ["  top_files:"]
    for item in top_files[:5]:
        if not isinstance(item, dict):
            continue
        preview = str(item.get("content") or "").strip().replace("\n", " ")[:160]
        lines.append(
            f"    - #{item.get('importance_rank')} {item.get('path')} "
            f"[{item.get('compaction_tier')}, {item.get('content_status')}, "
            f"{item.get('sent_lines')}/{item.get('total_lines')} lines]: "
            f"{preview or '(empty)'}"
        )
    return "\n".join(lines)


def _format_sample_files(sample_files: Any) -> str:
    if not isinstance(sample_files, list) or not sample_files:
        return "  focused_files: none"
    lines = ["  focused_files:"]
    for item in sample_files[:8]:
        if not isinstance(item, dict):
            continue
        status = item.get("content_status") or "unknown"
        source = item.get("source") or "sample"
        preview = str(item.get("content_preview") or "").strip().replace("\n", " ")[:180]
        lines.append(
            f"    - {item.get('path')} [{source}, {status}]: {preview or '(empty)'}"
        )
    return "\n".join(lines)


SANDBOX_LLM_SCORING_RULES = """\
Sandbox scoring rules (when SANDBOX REPORTS are present):
- Read every sandbox repo summary, risk_tier, and focused file excerpts before submit_screening_result.
- Repo classifications: aligned (strict), adjacent (moderate), peripheral (low weight), orthogonal (exclude role depth).
- Do NOT lower scores only because a repo lacks tests or CI (common for coursework, DSA, or HDL repos).
- Lower scores when aligned repos show hollow/stub/empty focused files where substance was expected.
- Treat aligned/adjacent repos with risk_tier SEVERE or CRITICAL as major negatives — not offset by other repos.
- Score bands for aligned repos (resume_similarity_score guidance before system caps):
  * CRITICAL (50+ combined vulns, or weak secret hygiene with 20+ vulns): overall score should be 60-65.
  * SEVERE (weak secrets + high findings, or 50+ vulns): overall score should be 62-68.
  * ELEVATED (20+ vulns or high-severity findings): overall score should be 68-75.
- Orthogonal repos (e.g. coursework DBMS) must not inflate scores; missing tests/CI there is neutral.
- Penalize resume claim vs repo evidence gaps when the resume oversells a repo.
- Cite sandbox evidence in requirement_matches or resume_similarity_score.reasoning when it affects judgment.
- For each sandbox top_files path, add a top_file_evaluation row at submit with jd_criteria,
  match_signal (positive|neutral|negative), and assessment tied to the JD. The server fills
  compaction metadata and evidence_snippet; only include rows for paths present in top_files.
- The platform applies mandatory risk caps for severe aligned-repo signals; your score should reflect that risk.
"""
