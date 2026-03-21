"""Tests for the Atlas-0 API server."""

from fastapi.testclient import TestClient

from atlas.api.server import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_spatial_query_empty():
    response = client.post("/query", json={"query": "where is the cup?"})
    assert response.status_code == 200
    assert response.json() == []
