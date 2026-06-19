"""Local git-history metrics for cloned repositories."""

from __future__ import annotations

import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

_GITHUB_NOREPLY_RE = re.compile(
    r"^(?:\d+\+)?([^@+]+)@users\.noreply\.github\.com$",
    re.IGNORECASE,
)


def _author_identity(name: str, email: str) -> str:
    """Stable contributor key: email beats display name (names vary across machines)."""
    email_l = (email or "").strip().lower()
    if email_l:
        noreply = _GITHUB_NOREPLY_RE.match(email_l)
        if noreply:
            return f"github:{noreply.group(1).lower()}"
        return f"email:{email_l}"
    collapsed = re.sub(r"[^a-z0-9]", "", (name or "").strip().lower())
    return f"name:{collapsed}" if collapsed else "unknown"


def _summarize_authorship(identities: list[str]) -> tuple[int, float, bool]:
    """
    Return (unique_authors, top_author_commit_share, sole_author).

    Students often commit as "Manav", "Manav Bhavsar", and "Manavv007" with a mix of
    Gmail and GitHub noreply addresses. When one identity clearly dominates and at most
    one personal email plus one GitHub noreply slug appear, treat as sole authorship.
    """
    if not identities:
        return 0, 0.0, False

    counts = Counter(identities)
    total = sum(counts.values())
    unique = len(counts)
    top_commits = counts.most_common(1)[0][1]
    share = round(top_commits / total, 4) if total else 0.0

    sole = unique == 1
    if not sole and unique <= 3 and share >= 0.85:
        github_slugs = {key.split(":", 1)[1] for key in counts if key.startswith("github:")}
        email_keys = [key for key in counts if key.startswith("email:")]
        if len(github_slugs) <= 1 and len(email_keys) <= 1:
            sole = True
            share = 1.0
            unique = 1

    return unique, float(share), sole


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
            ["git", "log", "--format=%an%x00%ae"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if res_authors.returncode == 0:
            identities: list[str] = []
            for line in res_authors.stdout.splitlines():
                if not line.strip():
                    continue
                name, _, email = line.partition("\0")
                identities.append(_author_identity(name.strip(), email.strip()))
            if identities:
                unique, share, sole = _summarize_authorship(identities)
                metrics["unique_authors"] = unique
                metrics["top_author_commit_share"] = round(share, 2)
                metrics["sole_author"] = sole

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
