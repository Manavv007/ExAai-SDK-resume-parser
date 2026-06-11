"""Tests for Cloud Run FILE_FOCUS_JSON compaction."""

from __future__ import annotations

import json

from agent.sandbox.focus_transport import (
    compact_file_focus_for_cloud_run_job,
    file_focus_json_for_cloud_run_job,
)


def test_compact_file_focus_drops_repo_tree() -> None:
    huge_paths = [f"src/module_{index}/file_{index}.py" for index in range(1200)]
    spec = {
        "repo_role": "aligned",
        "max_files": 5,
        "focus_paths": [{"path": "app/main.py", "source": "agent", "max_lines": 120}],
        "file_paths": huge_paths,
        "agent_focus_paths": [{"path": "app/main.py", "source": "agent"}],
        "candidate_tags": ["backend_engineer"],
        "top_files_count": 5,
    }
    compact = compact_file_focus_for_cloud_run_job(spec)
    assert compact is not None
    assert "file_paths" not in compact
    assert "agent_focus_paths" not in compact
    assert compact["focus_paths"][0]["path"] == "app/main.py"

    encoded = file_focus_json_for_cloud_run_job(spec)
    assert len(encoded.encode("utf-8")) < 4000
    assert json.loads(encoded)["repo_role"] == "aligned"
