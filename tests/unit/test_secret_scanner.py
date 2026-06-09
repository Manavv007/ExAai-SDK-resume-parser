"""Unit tests for secret scanning and env-file detection."""

from __future__ import annotations

from pathlib import Path

from agent.sandbox.evaluator.secret_scanner import (
    build_security_profile,
    calculate_secret_hits,
    find_committed_env_files,
)


def test_aws_example_keys_in_env_are_detected(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
        "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n",
        encoding="utf-8",
    )

    hits = calculate_secret_hits(tmp_path)
    assert hits >= 2


def test_regex_scan_runs_when_gitleaks_reports_zero(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "agent.sandbox.evaluator.secret_scanner.run_gitleaks",
        lambda _repo_dir: 0,
    )

    assert calculate_secret_hits(tmp_path) >= 1


def test_find_committed_env_files_skips_templates(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("API_KEY=real-key-value-here-12345\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("API_KEY=\n", encoding="utf-8")
    nested = tmp_path / "config"
    nested.mkdir()
    (nested / ".env.local").write_text("TOKEN=another-secret-token-value\n", encoding="utf-8")

    env_files = find_committed_env_files(tmp_path)
    paths = {path.relative_to(tmp_path).as_posix() for path in env_files}

    assert ".env" in paths
    assert "config/.env.local" in paths
    assert ".env.example" not in paths


def test_build_security_profile_flags_committed_env_files(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    env_content = "DB_PASSWORD=super-secret-password-12345\n"
    (tmp_path / "config" / ".env").write_text(env_content, encoding="utf-8")

    profile = build_security_profile(tmp_path, secret_hits=0)

    assert profile["has_env_file"] is True
    assert profile["secret_hygiene"] == "weak"
