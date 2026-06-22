"""Best-effort Tree-sitter metrics for non-Python source files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agent.sandbox.evaluator.filesystem_scanner import collect_source_files, read_text_if_exists

TREE_SITTER_LANGUAGE_KEYS = {
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
}


def analyze_non_python_code(repo_dir: Path) -> dict[str, Any]:
    files = [path for path in collect_source_files(repo_dir) if path.suffix.lower() != ".py"]
    if not files:
        return {
            "avg_cyclomatic_complexity": None,
            "avg_function_length": None,
            "type_annotation_ratio": None,
            "error_handling_density": None,
            "lint_violations_per_kloc": None,
        }

    try:
        from tree_sitter import Parser
        from tree_sitter_language_pack import get_language
    except Exception:
        return {
            "avg_cyclomatic_complexity": None,
            "avg_function_length": None,
            "type_annotation_ratio": None,
            "error_handling_density": None,
            "lint_violations_per_kloc": None,
        }

    total_functions = 0
    total_function_lines = 0
    total_complexity = 0
    total_typed_functions = 0
    error_handling_points = 0
    total_loc = 0
    parser = Parser()

    for path in files:
        key = TREE_SITTER_LANGUAGE_KEYS.get(path.suffix.lower())
        if not key:
            continue
        try:
            parser.language = get_language(key)
        except Exception:
            continue

        content = read_text_if_exists(path)
        if not content:
            continue
        total_loc += len(content.splitlines())
        tree = parser.parse(content.encode("utf-8"))
        root = tree.root_node
        function_nodes = list(_iter_function_nodes(root))
        total_functions += len(function_nodes)
        for node in function_nodes:
            total_function_lines += max(1, node.end_point[0] - node.start_point[0] + 1)
            snippet = content.encode("utf-8")[node.start_byte : node.end_byte].decode(
                "utf-8", errors="ignore"
            )
            total_complexity += 1 + len(
                re.findall(r"\b(if|for|while|catch|case|switch)\b|&&|\|\|", snippet)
            )
            if _looks_typed(snippet, path.suffix.lower()):
                total_typed_functions += 1
            error_handling_points += len(re.findall(r"\b(catch|throw|panic)\b", snippet))

    if total_functions == 0:
        return {
            "avg_cyclomatic_complexity": None,
            "avg_function_length": None,
            "type_annotation_ratio": None,
            "error_handling_density": None,
            "lint_violations_per_kloc": None,
        }

    return {
        "avg_cyclomatic_complexity": round(total_complexity / total_functions, 2),
        "avg_function_length": round(total_function_lines / total_functions, 2),
        "type_annotation_ratio": round(total_typed_functions / total_functions, 2),
        "error_handling_density": round(error_handling_points / total_loc, 4) if total_loc else 0.0,
        "lint_violations_per_kloc": None,
    }


def _iter_function_nodes(node: Any):
    function_types = {
        "function_declaration",
        "function_definition",
        "method_definition",
        "method_declaration",
        "arrow_function",
    }
    if getattr(node, "type", "") in function_types:
        yield node
    for child in getattr(node, "children", []):
        yield from _iter_function_nodes(child)


def _looks_typed(snippet: str, suffix: str) -> bool:
    if suffix in {".ts", ".tsx", ".go", ".java", ".rs", ".cpp", ".c", ".h"}:
        return True
    return ":" in snippet and "=>" not in snippet
