from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "agent_version" in body
    assert "model" in body


def test_screen_requires_auth(test_settings) -> None:
    response = client.post("/screen")
    assert response.status_code == 401


def test_screen_rejects_invalid_key(test_settings) -> None:
    response = client.post(
        "/screen",
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert response.status_code == 401
