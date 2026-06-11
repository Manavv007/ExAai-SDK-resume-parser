"""Tests for focused sandbox file sampling."""

from __future__ import annotations

from pathlib import Path

from agent.sandbox.evaluator.filesystem_scanner import collect_focused_sample_files


def test_collect_focused_sample_files_prefers_focus_paths(tmp_path: Path) -> None:
    (tmp_path / "small.py").write_text("print('small')", encoding="utf-8")
    (tmp_path / "large.css").write_text("body { color: red; }\n" * 200, encoding="utf-8")
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "service.py").write_text(
        "def process():\n    return {'ok': True}\n",
        encoding="utf-8",
    )

    samples = collect_focused_sample_files(
        tmp_path,
        {
            "max_files": 3,
            "focus_paths": [
                {"path": "app/service.py", "max_lines": 50, "source": "agent"},
            ],
        },
    )

    assert samples
    assert samples[0]["path"] == "app/service.py"
    assert samples[0]["source"] == "agent"
    assert samples[0]["content_status"] == "ok"
