from fastapi.testclient import TestClient

from api.main import app

FIXTURE_APP_ID = "11111111-1111-4111-8111-111111111111"
FIXTURE_JOB_ID = "22222222-2222-4222-8222-222222222222"

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "agent_version" in body
    assert "model" in body


def test_screen_requires_auth(test_settings) -> None:
    response = client.post(
        "/screen",
        data={"application_id": FIXTURE_APP_ID, "job_id": FIXTURE_JOB_ID},
        files={"resume": ("resume.txt", b"Engineer", "text/plain")},
    )
    assert response.status_code == 401


def test_screen_rejects_invalid_key(test_settings) -> None:
    response = client.post(
        "/screen",
        headers={"Authorization": "Bearer wrong-key"},
        data={"application_id": FIXTURE_APP_ID, "job_id": FIXTURE_JOB_ID},
        files={"resume": ("resume.txt", b"Engineer", "text/plain")},
    )
    assert response.status_code == 401
