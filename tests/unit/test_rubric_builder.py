from agent.tools.parser import JdStructured
from agent.tools.rubric_builder import (
    BIAS_AVOIDANCE_PREAMBLE,
    MUST_HAVE_SCORE_CAP,
    build_rubric,
    build_rubric_bundle,
    enforce_must_have_score_cap,
    infer_requirement_type,
)


def test_build_rubric_must_and_nice() -> None:
    jd = JdStructured(
        job_title="Backend Engineer",
        domain="technical",
        must_have=["5+ years Python", "PostgreSQL"],
        nice_to_have=["Kubernetes"],
    )
    rubric = build_rubric(jd)

    assert len(rubric) == 3
    assert rubric[0].weight == "must_have"
    assert rubric[0].requirement_type == "technical_skill"
    assert rubric[-1].weight == "nice_to_have"


def test_build_rubric_bundle_includes_preamble() -> None:
    bundle = build_rubric_bundle(
        {"must_have": ["API design"], "nice_to_have": [], "domain": "technical"}
    )
    assert bundle["rubric"][0]["criterion"] == "API design"
    assert BIAS_AVOIDANCE_PREAMBLE in bundle["rubric_preamble"]
    assert str(MUST_HAVE_SCORE_CAP) in bundle["rubric_preamble"]


def test_infer_requirement_type_education() -> None:
    assert infer_requirement_type("Bachelor degree in CS") == "education"


def test_enforce_must_have_score_cap() -> None:
    rubric = build_rubric(
        JdStructured(must_have=["Python"], nice_to_have=[], domain="technical")
    )
    low_match = {
        "requirement": "Python",
        "match_score": 20,
        "requirement_type": "technical_skill",
        "evidence": "x",
    }
    capped = enforce_must_have_score_cap(85, [low_match], rubric)
    assert capped == MUST_HAVE_SCORE_CAP


def test_enforce_must_have_no_cap_when_passing() -> None:
    rubric = build_rubric(JdStructured(must_have=["Python"], nice_to_have=[], domain="technical"))
    high_match = {
        "requirement": "Python",
        "match_score": 80,
        "requirement_type": "technical_skill",
        "evidence": "x",
    }
    score = enforce_must_have_score_cap(85, [high_match], rubric)
    assert score == 85
