"""Rule-based secret hygiene checks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agent.sandbox.evaluator.external_tools import run_gitleaks
from agent.sandbox.evaluator.filesystem_scanner import collect_source_files, read_text_if_exists

_ENV_EXAMPLE_NAMES = frozenset({".env.example", ".env.sample", ".env.template"})

SECRET_PATTERNS = [
    re.compile(
        r"([^A-Z0-9]|^)(AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}([^A-Z0-9]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN)\s*=\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),
    re.compile(
        r"https://hooks\.slack\.(?:com|example)/services/T[A-Z0-9_]{8}/B[A-Z0-9_]{8}/[A-Za-z0-9_]{24}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b[a-z0-9_]*(password|passwd|secret|api_key|apikey|token|private_key|auth_token)"
        r"[a-z0-9_]*\s*=\s*['\"]?[a-zA-Z0-9_\-\.\+/=]{16,}['\"]?",
        re.IGNORECASE,
    ),
]


def find_committed_env_files(repo_dir: Path) -> list[Path]:
    """Return committed .env-style files, excluding common safe templates."""
    found: list[Path] = []
    for path in repo_dir.rglob(".env*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.name.lower() in _ENV_EXAMPLE_NAMES:
            continue
        if path.name.startswith(".env"):
            found.append(path)
    return sorted(found)


def _collect_secret_scan_files(repo_dir: Path) -> list[Path]:
    files = collect_source_files(repo_dir, include_tests=True)
    files.extend(find_committed_env_files(repo_dir))
    seen: set[Path] = set()
    return [path for path in files if not (path in seen or seen.add(path))]


def _regex_secret_hits(repo_dir: Path) -> int:
    hits = 0
    for path in _collect_secret_scan_files(repo_dir)[:150]:
        content = read_text_if_exists(path)
        if not content:
            continue
        for pattern in SECRET_PATTERNS:
            hits += len(pattern.findall(content))
    return hits


def calculate_secret_hits(repo_dir: Path) -> int:
    regex_hits = _regex_secret_hits(repo_dir)
    gitleaks_hits = run_gitleaks(repo_dir)
    if gitleaks_hits is None:
        return regex_hits
    return max(regex_hits, gitleaks_hits)


def build_security_profile(repo_dir: Path, secret_hits: int) -> dict[str, Any]:
    env_files = find_committed_env_files(repo_dir)
    env_example = repo_dir / ".env.example"
    gitignore = read_text_if_exists(repo_dir / ".gitignore")
    covers_env = any(
        entry in gitignore
        for entry in (".env", ".env.*", "*.pem", "*.key", "secrets.json")
    )
    has_env_file = bool(env_files)
    return {
        "secret_pattern_hits": secret_hits,
        "has_env_file": has_env_file,
        "has_env_example": env_example.exists(),
        "gitignore_covers_env": covers_env,
        "secret_hygiene": (
            "weak"
            if secret_hits > 0 or has_env_file
            else "good" if env_example.exists() and covers_env else "mixed"
        ),
    }
