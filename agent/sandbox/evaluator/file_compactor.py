"""Tiered file compaction for sandbox top-file evaluation payloads."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

RAW_LINE_LIMIT = 200
STRIP_LINE_LIMIT = 800
DEFAULT_CHAR_CAP = 12_000

_COMMENT_LINE = re.compile(r"^\s*(#|//).*")
_BLANK_LINE = re.compile(r"^\s*$")


def _language_from_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".go": "go",
        ".java": "java",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
    }.get(suffix, "unknown")


def strip_noise_lines(content: str, path: str) -> str:
    """Remove blank lines and line comments (lightweight, language-aware)."""
    language = _language_from_path(path)
    lines = content.splitlines()
    if language == "python":
        return _strip_python_noise(content)
    kept: list[str] = []
    for line in lines:
        if _BLANK_LINE.match(line):
            continue
        if language in (
            "javascript",
            "typescript",
            "go",
            "java",
            "rust",
            "csharp",
        ) and _COMMENT_LINE.match(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def _strip_python_noise(content: str) -> str:
    try:
        import ast

        tree = ast.parse(content)
    except SyntaxError:
        return "\n".join(line for line in content.splitlines() if not _COMMENT_LINE.match(line))

    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(getattr(node.body[0], "value", None), ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                end = getattr(node.body[0], "end_lineno", node.body[0].lineno)
                for line_no in range(node.lineno, (end or node.lineno) + 1):
                    docstring_lines.add(line_no)

    kept: list[str] = []
    for index, line in enumerate(content.splitlines(), start=1):
        if index in docstring_lines:
            continue
        if _BLANK_LINE.match(line) or _COMMENT_LINE.match(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def extract_python_skeleton(content: str) -> str:
    """Extract signatures and the first few body lines per function/class."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return strip_noise_lines(content, "file.py")[:DEFAULT_CHAR_CAP]

    lines = content.splitlines()
    blocks: list[str] = []
    seen_lines: set[int] = set()

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = node.lineno - 1
        if start in seen_lines:
            continue
        seen_lines.add(start)
        end = min(start + 4, len(lines))
        block = "\n".join(lines[start:end])
        blocks.append(block)
        blocks.append("    ...")

    if not blocks:
        return "\n".join(lines[: min(120, len(lines))])
    return "\n\n".join(blocks)


def extract_generic_skeleton(content: str) -> str:
    """Heuristic skeleton for non-Python source files."""
    lines = content.splitlines()
    picked: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("import ", "from ", "export ", "package ", "using ", "namespace ")):
            picked.append(line)
        elif re.match(
            r"^(class |interface |enum |struct |func |function |def |async function )", stripped
        ):
            picked.append(line)
            picked.append("    ...")
        if len(picked) >= 80:
            break
    return "\n".join(picked) if picked else "\n".join(lines[: min(120, len(lines))])


def compact_file_content(
    content: str,
    path: str,
    *,
    char_cap: int = DEFAULT_CHAR_CAP,
) -> dict[str, Any]:
    """Apply tiered compaction based on file length."""
    lines = content.splitlines()
    total_lines = len(lines)

    if total_lines <= RAW_LINE_LIMIT:
        tier = "raw"
        compacted = content
    elif total_lines <= STRIP_LINE_LIMIT:
        tier = "stripped"
        compacted = strip_noise_lines(content, path)
    elif _language_from_path(path) == "python":
        tier = "skeleton"
        compacted = extract_python_skeleton(content)
    else:
        tier = "skeleton"
        compacted = extract_generic_skeleton(content)

    truncated = len(compacted) > char_cap
    if truncated:
        compacted = compacted[:char_cap]

    sent_lines = len(compacted.splitlines()) if compacted else 0
    return {
        "compaction_tier": tier,
        "total_lines": total_lines,
        "sent_lines": sent_lines,
        "content": compacted,
        "truncated": truncated,
    }
