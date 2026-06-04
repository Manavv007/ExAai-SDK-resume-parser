from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app

FIXTURE_APP_ID = "11111111-1111-4111-8111-111111111111"
FIXTURE_JOB_ID = "22222222-2222-4222-8222-222222222222"


@pytest.fixture
def client(test_settings) -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_screen_requires_auth(client: TestClient) -> None:
    response = client.post(
        "/screen",
        data={"application_id": FIXTURE_APP_ID, "job_id": FIXTURE_JOB_ID},
        files={"resume": ("resume.txt", b"Engineer", "text/plain")},
    )
    assert response.status_code == 401


@patch("api.routes.run_screening_async", new_callable=AsyncMock)
def test_screen_success_with_form_api_key(mock_run, client: TestClient, test_settings) -> None:
    mock_run.return_value = {
        "application_id": FIXTURE_APP_ID,
        "job_id": FIXTURE_JOB_ID,
        "resume_screening_status": "completed",
        "resume_similarity_score": {"score": 70, "reasoning": "Good fit."},
        "requirement_matches": [],
        "recommendation": "advance",
        "recommendation_reasoning": "Meets requirements.",
        "red_flags": [],
        "sources_crawled": [],
        "metadata": {
            "schema_version": "1.0",
            "model_version": "exaai-adk/test",
            "processed_at": "2026-06-04T12:00:00Z",
            "processing_time_ms": 100,
            "resume_text_chars": 10,
            "agent_version": "0.1.0",
        },
        "errors": [],
    }

    response = client.post(
        "/screen",
        data={
            "application_id": FIXTURE_APP_ID,
            "job_id": FIXTURE_JOB_ID,
            "jd_text": "Need Python experience.",
            "api_key": "test-key",
        },
        files={"resume": ("resume.txt", b"Python engineer resume text.", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["resume_screening_status"] == "completed"


@patch("api.routes.run_screening_async", new_callable=AsyncMock)
def test_screen_success(mock_run, client: TestClient, test_settings) -> None:
    mock_run.return_value = {
        "application_id": FIXTURE_APP_ID,
        "job_id": FIXTURE_JOB_ID,
        "resume_screening_status": "completed",
        "resume_similarity_score": {"score": 70, "reasoning": "Good fit."},
        "requirement_matches": [],
        "recommendation": "advance",
        "recommendation_reasoning": "Meets requirements.",
        "red_flags": [],
        "sources_crawled": [],
        "metadata": {
            "schema_version": "1.0",
            "model_version": "exaai-adk/test",
            "processed_at": "2026-06-04T12:00:00Z",
            "processing_time_ms": 100,
            "resume_text_chars": 10,
            "agent_version": "0.1.0",
        },
        "errors": [],
    }

    response = client.post(
        "/screen",
        headers={"Authorization": "Bearer test-key"},
        data={
            "application_id": FIXTURE_APP_ID,
            "job_id": FIXTURE_JOB_ID,
            "jd_text": "Need Python experience.",
        },
        files={"resume": ("resume.txt", b"Python engineer resume text.", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["resume_screening_status"] == "completed"
    assert "X-Request-ID" in response.headers


def test_screen_accepts_authorization_without_bearer_prefix(
    client: TestClient, test_settings
) -> None:
    with patch("api.routes.run_screening_async", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {
            "application_id": FIXTURE_APP_ID,
            "job_id": FIXTURE_JOB_ID,
            "resume_screening_status": "failed",
            "errors": [{"code": "X", "message": "y"}],
        }
        response = client.post(
            "/screen",
            headers={"Authorization": "test-key"},
            data={
                "application_id": FIXTURE_APP_ID,
                "job_id": FIXTURE_JOB_ID,
                "jd_text": "JD",
            },
            files={"resume": ("resume.txt", b"text", "text/plain")},
        )
    assert response.status_code != 401


def test_screen_invalid_uuid(client: TestClient, test_settings) -> None:
    response = client.post(
        "/screen",
        headers={"Authorization": "Bearer test-key"},
        data={"application_id": "bad", "job_id": FIXTURE_JOB_ID, "jd_text": "JD"},
        files={"resume": ("resume.txt", b"text", "text/plain")},
    )
    assert response.status_code == 400
