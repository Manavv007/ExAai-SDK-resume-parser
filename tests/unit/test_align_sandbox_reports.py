from agent.tools.github_analyzer import align_sandbox_reports_with_urls


def test_align_sandbox_reports_normalizes_trailing_git_and_www() -> None:
    reports = [
        {
            "repo": "DevKansara97/chaos-repo",
            "url": "https://github.com/DevKansara97/chaos-repo.git",
            "clone_ok": True,
        }
    ]
    aligned = align_sandbox_reports_with_urls(
        ["https://www.github.com/DevKansara97/chaos-repo"],
        reports,
    )
    assert len(aligned) == 1
    assert aligned[0]["repo"] == "DevKansara97/chaos-repo"


def test_align_sandbox_reports_falls_back_to_reports_when_urls_miss() -> None:
    reports = [
        {"repo": "owner/a", "url": "https://github.com/owner/a", "clone_ok": True},
        {"repo": "owner/b", "url": "https://github.com/owner/b", "clone_ok": True},
    ]
    aligned = align_sandbox_reports_with_urls(
        ["https://github.com/owner/missing"],
        reports,
    )
    assert aligned == reports
