# backend/tests/test_health.py
import os
import pytest

@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-DUMMY")  # just to allow import
    # Keep other defaults

def test_health_endpoint():
    from fastapi.testclient import TestClient
    import main  # imports config -> requires OPENAI_API_KEY
    client = TestClient(main.app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}
