import json
from unittest.mock import MagicMock, patch

from agent.tools.parser import parse_jd_structured
from agent.tools.rubric_builder import build_rubric


@patch("google.genai.Client")
def test_gemini_assigns_requirement_types(mock_client_cls, test_settings) -> None:
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    payload = {
        "job_title": "Registered Nurse",
        "domain": "healthcare",
        "requirements": [
            {
                "text": "Valid RN license",
                "weight": "must_have",
                "requirement_type": "education",
            },
            {
                "text": "3+ years acute care",
                "weight": "must_have",
                "requirement_type": "experience",
            },
            {
                "text": "Patient communication skills",
                "weight": "nice_to_have",
                "requirement_type": "soft_skill",
            },
        ],
        "must_have": ["Valid RN license", "3+ years acute care"],
        "nice_to_have": ["Patient communication skills"],
    }
    mock_response = MagicMock()
    mock_response.text = json.dumps(payload)
    mock_client.models.generate_content.return_value = mock_response

    jd = parse_jd_structured(
        "Registered Nurse. Must have RN license and 3+ years acute care.",
        use_llm=True,
    )

    assert jd.requirements[0].requirement_type == "education"
    rubric = build_rubric(jd)
    assert rubric[0].requirement_type == "education"
    assert rubric[1].requirement_type == "experience"
    assert rubric[2].requirement_type == "soft_skill"


def test_heuristic_still_falls_back_without_llm_types() -> None:
    jd = parse_jd_structured(
        "Nurse role. Requirements:\n- Valid RN license\n- 3+ years experience",
        use_llm=False,
    )
    rubric = build_rubric(jd)
    assert rubric[0].requirement_type in {
        "education",
        "experience",
        "technical_skill",
        "soft_skill",
        "responsibility",
    }
