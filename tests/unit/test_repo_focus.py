"""Tests for role-aware repository file focus helpers."""

from __future__ import annotations

from agent.tools.repo_focus import (
    build_mandatory_focus_paths,
    classify_content_quality,
    classify_repo_role,
    infer_repo_type_tags_from_signals,
    merge_repo_focus_spec,
    reconcile_sandbox_report_classification,
    resolve_focus_path,
    validate_orchestrated_sandbox_repo_spec,
    validate_repo_focus_paths,
)


def test_classify_content_quality_detects_stub() -> None:
    assert classify_content_quality("class OrderService:\n    pass  # TODO implement") == "stub"
    assert classify_content_quality("def handler():\n    return db.query(User).all()\n") == "ok"


def test_classify_repo_role_backend_vs_theme() -> None:
    role = classify_repo_role(
        repo_type_tags=["frontend_app"],
        candidate_tags=["backend_engineer"],
        file_paths=["static/style.css", "assets/index.html", "public/app.css"],
    )
    assert role == "orthogonal"

    aligned = classify_repo_role(
        repo_type_tags=["backend_service"],
        candidate_tags=["backend_engineer"],
        file_paths=["app/api/routes.py", "app/services/order.py"],
    )
    assert aligned == "aligned"


def test_resolve_focus_path_substitutes_close_match() -> None:
    paths = ["src/api/routes.py", "README.md", "requirements.txt"]
    resolved, substituted = resolve_focus_path("api/routes.py", paths)
    assert resolved == "src/api/routes.py"
    assert substituted is True


def test_merge_repo_focus_spec_agent_only_when_agent_paths_present() -> None:
    file_paths = [
        "README.md",
        "requirements.txt",
        "app/api/routes.py",
        "app/services/order_service.py",
        "static/style.css",
    ]
    spec = merge_repo_focus_spec(
        file_paths=file_paths,
        candidate_tags=["backend_engineer"],
        repo_role="aligned",
        agent_focus_paths=[{"path": "app/services/order_service.py", "max_lines": 150}],
        max_files=5,
    )
    paths = [item["path"] for item in spec["focus_paths"]]
    assert spec["pick_mode"] == "agent_only"
    assert paths == ["app/services/order_service.py"]
    assert all(item["source"] == "agent" for item in spec["focus_paths"])


def test_merge_repo_focus_spec_legacy_when_no_agent_paths() -> None:
    file_paths = [
        "README.md",
        "requirements.txt",
        "app/api/routes.py",
        "app/services/order_service.py",
    ]
    spec = merge_repo_focus_spec(
        file_paths=file_paths,
        candidate_tags=["backend_engineer"],
        repo_role="aligned",
        agent_focus_paths=None,
        max_files=8,
    )
    paths = [item["path"] for item in spec["focus_paths"]]
    assert spec["pick_mode"] == "legacy"
    assert "README.md" in paths
    assert "app/api/routes.py" in paths


def test_merge_repo_focus_spec_caps_agent_paths_at_max_files() -> None:
    file_paths = [f"src/module_{index}.py" for index in range(10)]
    agent_focus = [{"path": path} for path in file_paths]
    spec = merge_repo_focus_spec(
        file_paths=file_paths,
        candidate_tags=["backend_engineer"],
        repo_role="aligned",
        agent_focus_paths=agent_focus,
        max_files=3,
    )
    assert spec["pick_mode"] == "agent_only"
    assert len(spec["focus_paths"]) == 3


def test_validate_orchestrated_spec_requires_focus_paths_and_matching_classification() -> None:
    errors = validate_orchestrated_sandbox_repo_spec(
        repo_url="https://github.com/dev/service",
        classification="aligned",
        structure_classification="orthogonal",
        focus_paths=[{"path": "app/main.py"}],
        require_agent_focus=True,
    )
    assert any("does not match" in err for err in errors)

    conservative = validate_orchestrated_sandbox_repo_spec(
        repo_url="https://github.com/dev/service",
        classification="peripheral",
        structure_classification="aligned",
        focus_paths=[{"path": "app/main.py"}],
        require_agent_focus=True,
    )
    assert not any("does not match" in err for err in conservative)

    missing_focus = validate_orchestrated_sandbox_repo_spec(
        repo_url="https://github.com/dev/service",
        classification="aligned",
        structure_classification="aligned",
        focus_paths=[],
        require_agent_focus=True,
    )
    assert any("focus_paths is required" in err for err in missing_focus)


def test_validate_repo_focus_paths_rejects_over_max() -> None:
    file_paths = [f"src/file_{index}.py" for index in range(10)]
    focus_paths = [{"path": path} for path in file_paths[:6]]
    errors = validate_repo_focus_paths(
        repo_url="https://github.com/dev/service",
        focus_paths=focus_paths,
        file_paths=file_paths,
        max_paths=5,
    )
    assert len(errors) == 1
    assert "maximum is 5" in errors[0]


def test_validate_repo_focus_paths_rejects_missing_path() -> None:
    errors = validate_repo_focus_paths(
        repo_url="https://github.com/dev/service",
        focus_paths=[{"path": "src/missing.py"}],
        file_paths=["README.md", "app/main.py"],
        max_paths=5,
    )
    assert len(errors) == 1
    assert "not found" in errors[0]


def test_validate_repo_focus_paths_accepts_resolved_paths() -> None:
    errors = validate_repo_focus_paths(
        repo_url="https://github.com/dev/service",
        focus_paths=[{"path": "api/routes.py"}],
        file_paths=["src/api/routes.py"],
        max_paths=5,
    )
    assert errors == []


def test_build_mandatory_focus_paths_includes_manifests() -> None:
    paths = build_mandatory_focus_paths(
        ["README.md", "requirements.txt", "main.py", "tests/test_app.py"]
    )
    selected = {item["path"] for item in paths}
    assert "README.md" in selected
    assert "requirements.txt" in selected
    assert "main.py" in selected


def test_infer_repo_type_tags_detects_rag_pipeline_from_paths() -> None:
    tags = infer_repo_type_tags_from_signals(
        file_paths=[
            "backend/app/services/rag_retriever.py",
            "backend/app/api/v1/chat.py",
            "requirements.txt",
        ],
        dependencies={"langchain", "pinecone-client", "fastapi"},
        framework_markers=["fastapi", "langchain"],
        architecture_layers=["pipeline", "data", "services"],
        base_tags=["backend_service"],
    )
    assert "ai_agent_project" in tags
    assert "data_pipeline" in tags
    assert "rag_pipeline" in tags


def test_classify_saral_like_repo_aligned_for_ai_jd() -> None:
    jd_keywords = {"rag", "embedding", "llm", "fastapi", "python", "pipeline"}
    role = classify_repo_role(
        repo_type_tags=[
            "backend_service",
            "ai_agent_project",
            "data_pipeline",
            "rag_pipeline",
        ],
        candidate_tags=["ai_engineer", "backend_engineer"],
        file_paths=[
            "backend/app/services/rag_retriever.py",
            "backend/app/services/llm_engine.py",
        ],
        jd_keywords=jd_keywords,
        framework_markers=["fastapi", "langchain"],
    )
    assert role == "aligned"


def test_reconcile_sandbox_report_upgrades_classification() -> None:
    report = {
        "url": "https://github.com/user/saral",
        "classification": "peripheral",
        "repo_profile": {
            "framework_markers": ["fastapi", "langchain"],
            "architecture": {"layers": ["pipeline", "data", "services"]},
            "repo_type_tags": ["backend_service", "ai_agent_project", "data_pipeline"],
            "project_shape": "interactive_app",
        },
    }
    role = reconcile_sandbox_report_classification(
        report,
        candidate_tags=["ai_engineer"],
        jd_keywords={"rag", "embedding", "llm"},
        file_paths=["backend/app/services/rag_retriever.py"],
    )
    assert role == "aligned"
    assert report["classification"] == "aligned"
