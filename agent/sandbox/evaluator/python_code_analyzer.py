"""Python AST-based code metrics."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from agent.sandbox.evaluator.filesystem_scanner import collect_source_files, read_text_if_exists


class PythonASTAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.total_functions = 0
        self.annotated_functions = 0
        self.total_complexity_points = 0
        self.total_function_lines = 0
        self.error_handling_points = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.total_functions += 1
        self.total_complexity_points += 1
        self.total_function_lines += max(1, (node.end_lineno or node.lineno) - node.lineno + 1)
        if self._has_annotations(node):
            self.annotated_functions += 1
        self.generic_visit(node)

    @staticmethod
    def _has_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        if node.returns:
            return True
        args = list(node.args.args) + list(getattr(node.args, "kwonlyargs", []))
        if node.args.vararg:
            args.append(node.args.vararg)
        if node.args.kwarg:
            args.append(node.args.kwarg)
        return any(arg.annotation for arg in args)

    def visit_If(self, node: ast.If) -> None:
        self.total_complexity_points += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.total_complexity_points += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.total_complexity_points += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.total_complexity_points += 1
        self.error_handling_points += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.error_handling_points += 1
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        self.error_handling_points += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.total_complexity_points += max(0, len(node.values) - 1)
        self.generic_visit(node)


def analyze_python_code(repo_dir: Path) -> dict[str, Any]:
    files = [path for path in collect_source_files(repo_dir) if path.suffix.lower() == ".py"]
    if not files:
        return {
            "avg_cyclomatic_complexity": None,
            "avg_function_length": None,
            "type_annotation_ratio": None,
            "error_handling_density": None,
        }

    analyzer = PythonASTAnalyzer()
    total_loc = 0
    for path in files:
        content = read_text_if_exists(path)
        if not content:
            continue
        total_loc += len(content.splitlines())
        try:
            analyzer.visit(ast.parse(content, filename=str(path)))
        except SyntaxError:
            continue

    if analyzer.total_functions == 0:
        return {
            "avg_cyclomatic_complexity": 1.0,
            "avg_function_length": 0.0,
            "type_annotation_ratio": 0.0,
            "error_handling_density": round(analyzer.error_handling_points / total_loc, 4)
            if total_loc
            else 0.0,
        }

    return {
        "avg_cyclomatic_complexity": round(
            analyzer.total_complexity_points / analyzer.total_functions, 2
        ),
        "avg_function_length": round(analyzer.total_function_lines / analyzer.total_functions, 2),
        "type_annotation_ratio": round(
            analyzer.annotated_functions / analyzer.total_functions, 2
        ),
        "error_handling_density": round(analyzer.error_handling_points / total_loc, 4)
        if total_loc
        else 0.0,
    }
