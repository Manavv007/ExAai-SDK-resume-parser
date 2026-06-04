from pathlib import Path

from agent.prep import prepare_screening_state

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_prepare_screening_state_redacts_and_extracts_links() -> None:
    resume = (FIXTURES / "sample_resume.txt").read_bytes()
    jd = (FIXTURES / "sample_jd.txt").read_text(encoding="utf-8")

    state = prepare_screening_state(
        application_id="11111111-1111-4111-8111-111111111111",
        job_id="22222222-2222-4222-8222-222222222222",
        resume_bytes=resume,
        resume_filename="resume.txt",
        jd_text=jd,
    )

    assert "jane.doe@example.com" not in state["resume_text"]
    assert state["application_id"].startswith("11111111")
    assert len(state["profile_urls"]) >= 1
    assert state["jd_structured"]["domain"] == "technical"
    assert state["redaction_count"] >= 1
    assert len(state["rubric"]) >= 1
    assert "Evaluate only technical skills" in state["rubric_preamble"]
