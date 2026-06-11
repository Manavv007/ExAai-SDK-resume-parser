from agent.tools.sandbox_prompt import format_sandbox_reports_for_prompt


def test_format_sandbox_reports_truncates_when_max_chars_set() -> None:
    long_preview = "y" * 400
    reports = [
        {
            "repo": "owner/big-repo",
            "clone_ok": True,
            "repo_profile": {
                "security_profile": {"secret_hygiene": "poor"},
                "external_tool_signals": {"trivy": {"vulnerability_count": 99}},
                "top_files": [
                    {
                        "path": f"src/file_{i}.py",
                        "content": long_preview,
                        "importance_rank": i + 1,
                        "compaction_tier": "full",
                        "content_status": "ok",
                        "sent_lines": 10,
                        "total_lines": 10,
                    }
                    for i in range(8)
                ],
                "sample_files": [
                    {
                        "path": f"lib/sample_{i}.py",
                        "content_preview": long_preview,
                        "content_status": "ok",
                        "source": "focus",
                    }
                    for i in range(8)
                ],
            },
            "findings": [
                {"severity": "high", "title": f"finding-{i}-" + ("x" * 80)}
                for i in range(20)
            ],
        }
    ]
    full = format_sandbox_reports_for_prompt(reports)
    compact = format_sandbox_reports_for_prompt(reports, max_chars=500)

    assert len(full) > 500
    assert len(compact) <= 500
    assert compact.endswith("...(truncated)")
