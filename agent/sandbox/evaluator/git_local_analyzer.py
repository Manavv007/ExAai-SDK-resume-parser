"""Local git-history metrics for cloned repositories."""

from __future__ import annotations

import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any


def calculate_git_metrics(repo_dir: Path) -> dict[str, Any]:
    metrics = {
        "commit_count": 0,
        "unique_authors": 0,
        "days_since_last_commit": 0,
        "top_author_commit_share": 0.0,
        "sole_author": False,
        "merge_to_commit_ratio": 0.0,
        "history_is_shallow": bool((repo_dir / ".git" / "shallow").exists()),
    }
    if not (repo_dir / ".git").exists():
        return metrics

    try:
        res_count = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if res_count.returncode == 0 and res_count.stdout.strip():
            metrics["commit_count"] = int(res_count.stdout.strip())

        res_authors = subprocess.run(
            ["git", "log", "--format=%an"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if res_authors.returncode == 0:
            authors = [line.strip() for line in res_authors.stdout.splitlines() if line.strip()]
            if authors:
                metrics["unique_authors"] = len(set(authors))
                metrics["sole_author"] = metrics["unique_authors"] == 1
                top_author_commits = Counter(authors).most_common(1)[0][1]
                metrics["top_author_commit_share"] = round(top_author_commits / len(authors), 2)

        res_time = subprocess.run(
            ["git", "log", "-1", "--format=%at"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if res_time.returncode == 0 and res_time.stdout.strip():
            commit_time = int(res_time.stdout.strip())
            metrics["days_since_last_commit"] = max(0, int((time.time() - commit_time) / 86400))

        if metrics["commit_count"] > 0:
            res_merges = subprocess.run(
                ["git", "log", "--oneline", "--merges"],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            merge_count = 0
            if res_merges.returncode == 0:
                merge_count = sum(1 for line in res_merges.stdout.splitlines() if line.strip())
            metrics["merge_to_commit_ratio"] = round(
                merge_count / metrics["commit_count"],
                4,
            )
    except Exception:
        pass

    return metrics
