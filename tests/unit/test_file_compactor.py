"""Tests for sandbox file compaction tiers."""

from __future__ import annotations

from agent.sandbox.evaluator.file_compactor import compact_file_content
from agent.sandbox.evaluator.top_files import collect_top_files
from agent.tools.repo_focus import select_evaluation_paths


def test_compact_small_file_is_raw() -> None:
    source = "def add(a, b):\n    return a + b\n"
    result = compact_file_content(source, "app/math.py")
    assert result["compaction_tier"] == "raw"
    assert "return a + b" in result["content"]


def test_compact_medium_file_is_stripped() -> None:
    source = "\n".join(
        ["# header", "", "def f(x):", '    """doc"""', "    return x"] + ["x = 1"] * 250
    )
    result = compact_file_content(source, "app/medium.py")
    assert result["compaction_tier"] == "stripped"
    assert "# header" not in result["content"]


def test_compact_large_python_file_is_skeleton() -> None:
    body = "\n".join(f"def f{n}(x):\n    return x + {n}" for n in range(450))
    result = compact_file_content(body, "app/large.py")
    assert result["compaction_tier"] == "skeleton"
    assert "def f0" in result["content"]
    assert "return x + 299" not in result["content"]


def test_select_evaluation_paths_agent_only_when_agent_paths_present() -> None:
    paths = [
        "README.md",
        "src/api/routes.py",
        "src/services/order_service.py",
        "tests/test_routes.py",
    ]
    selected = select_evaluation_paths(
        paths,
        candidate_tags=["backend", "python"],
        repo_role="aligned",
        agent_focus_paths=[{"path": "src/services/order_service.py"}],
        max_files=5,
    )
    assert selected == ["src/services/order_service.py"]


def test_select_evaluation_paths_heuristic_when_no_agent_paths() -> None:
    paths = [
        "README.md",
        "src/api/routes.py",
        "src/services/order_service.py",
    ]
    selected = select_evaluation_paths(
        paths,
        candidate_tags=["backend", "python"],
        repo_role="aligned",
        agent_focus_paths=None,
        max_files=2,
    )
    assert len(selected) == 2
    assert "src/api/routes.py" in selected


def test_collect_top_files_from_repo(tmp_path) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "main.py").write_text("def main():\n    return 42\n", encoding="utf-8")
    (repo / "README.md").write_text("# demo", encoding="utf-8")

    top = collect_top_files(
        repo,
        {
            "repo_role": "aligned",
            "candidate_tags": ["python"],
            "file_paths": ["README.md", "src/main.py"],
            "top_files_count": 2,
        },
    )
    assert len(top) >= 1
    assert any(item["path"] == "src/main.py" for item in top)
    assert top[0]["content"]
