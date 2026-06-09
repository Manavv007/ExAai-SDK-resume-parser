"""README, docs, TODO, and generic text-signal extraction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agent.sandbox.evaluator.filesystem_scanner import find_readme_path, read_text_if_exists


def build_documentation_profile(repo_dir: Path) -> dict[str, Any]:
    readme_path = find_readme_path(repo_dir)
    readme_text = read_text_if_exists(readme_path) if readme_path else ""
    lowered = readme_text.lower()
    has_badges = "[![" in readme_text or "shields.io" in lowered
    has_setup = any(word in lowered for word in ("install", "setup", "getting started"))
    has_architecture = "architecture" in lowered or "design" in lowered
    has_docs_dir = (repo_dir / "docs").exists()
    return {
        "readme_present": bool(readme_path),
        "readme_bytes": len(readme_text.encode("utf-8")) if readme_text else 0,
        "has_badges": has_badges,
        "has_setup_instructions": has_setup,
        "has_architecture_section": has_architecture,
        "has_docs_dir": has_docs_dir,
        "has_changelog": (repo_dir / "CHANGELOG.md").exists(),
        "has_contributing": (repo_dir / "CONTRIBUTING.md").exists(),
        "has_security_md": (repo_dir / "SECURITY.md").exists(),
        "has_adr": (repo_dir / "docs" / "adr").exists() or (repo_dir / "adr").exists(),
    }


def count_todo_fixme_density(contents: list[str], total_loc: int) -> float:
    if total_loc <= 0:
        return 0.0
    todo_pattern = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b", re.IGNORECASE)
    count = sum(len(todo_pattern.findall(content)) for content in contents)
    return round(count / total_loc, 4)
