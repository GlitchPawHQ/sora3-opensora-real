from fastapi.testclient import TestClient

import app


client = TestClient(app.app)


def test_health_reports_generation_disabled_by_default():
    assert app.GENERATION_ENABLED is False
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["generation_enabled"] is False


def test_generation_endpoint_does_not_return_mock_job_when_disabled():
    response = client.post("/api/jobs", json={"prompt": "A red fox runs through a forest."})
    assert response.status_code == 503
    assert "disabled" in response.json()["detail"].lower()


def test_unknown_job_is_not_found():
    response = client.get("/api/jobs/not-a-real-job")
    assert response.status_code == 404
