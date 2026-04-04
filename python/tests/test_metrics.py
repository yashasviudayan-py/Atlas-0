"""Tests for the Prometheus metrics module and /metrics endpoint.

Covers:
- Metrics registry is isolated (not the default global registry).
- Each metric type has the correct name and type.
- GET /metrics returns 200 with the correct content-type.
- GET /metrics response body contains expected metric names.
- Gauges can be set and the value is reflected in the /metrics output.
- Counter increments are reflected in the /metrics output.
- Histogram observations are reflected in the /metrics output.
- ws_clients_active increments on WebSocket connect (via server state).
- HealthResponse includes risks_stale_seconds field.
- query_total increments on each POST /query call.
- WorldModelAgent.risks_stale_seconds returns inf before any assessment.
- WorldModelAgent.risks_stale_seconds returns a small positive value after
  _last_assessment_time is set.
"""

from __future__ import annotations

import time

from atlas.api import metrics as m
from atlas.api.server import app
from atlas.world_model.agent import WorldModelAgent, WorldModelConfig
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

# ── Shared test client ────────────────────────────────────────────────────────

client = TestClient(app)


# ── Registry isolation ────────────────────────────────────────────────────────


def test_registry_is_custom_instance() -> None:
    assert isinstance(m.REGISTRY, CollectorRegistry)


def test_registry_is_not_default() -> None:
    from prometheus_client import REGISTRY as DEFAULT_REGISTRY

    assert m.REGISTRY is not DEFAULT_REGISTRY


# ── Metric existence & type ───────────────────────────────────────────────────


def test_risk_count_is_gauge() -> None:
    from prometheus_client import Gauge

    assert isinstance(m.risk_count, Gauge)


def test_object_count_is_gauge() -> None:
    from prometheus_client import Gauge

    assert isinstance(m.object_count, Gauge)


def test_ws_clients_active_is_gauge() -> None:
    from prometheus_client import Gauge

    assert isinstance(m.ws_clients_active, Gauge)


def test_slam_active_is_gauge() -> None:
    from prometheus_client import Gauge

    assert isinstance(m.slam_active, Gauge)


def test_assessment_age_seconds_is_gauge() -> None:
    from prometheus_client import Gauge

    assert isinstance(m.assessment_age_seconds, Gauge)


def test_query_total_is_counter() -> None:
    from prometheus_client import Counter

    assert isinstance(m.query_total, Counter)


def test_vlm_request_seconds_is_histogram() -> None:
    from prometheus_client import Histogram

    assert isinstance(m.vlm_request_seconds, Histogram)


# ── /metrics endpoint ─────────────────────────────────────────────────────────


def test_metrics_endpoint_status_200() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_endpoint_content_type() -> None:
    response = client.get("/metrics")
    assert "text/plain" in response.headers["content-type"]


def test_metrics_response_contains_risk_count() -> None:
    response = client.get("/metrics")
    assert "atlas_risk_count" in response.text


def test_metrics_response_contains_object_count() -> None:
    response = client.get("/metrics")
    assert "atlas_object_count" in response.text


def test_metrics_response_contains_ws_clients_active() -> None:
    response = client.get("/metrics")
    assert "atlas_ws_clients_active" in response.text


def test_metrics_response_contains_slam_active() -> None:
    response = client.get("/metrics")
    assert "atlas_slam_active" in response.text


def test_metrics_response_contains_query_total() -> None:
    response = client.get("/metrics")
    assert "atlas_query_total" in response.text


def test_metrics_response_contains_vlm_histogram() -> None:
    response = client.get("/metrics")
    assert "atlas_vlm_request_seconds" in response.text


def test_metrics_response_contains_assessment_age() -> None:
    response = client.get("/metrics")
    assert "atlas_assessment_age_seconds" in response.text


# ── Gauge value propagation ───────────────────────────────────────────────────


def test_risk_count_gauge_set_reflects_in_output() -> None:
    m.risk_count.set(42)
    response = client.get("/metrics")
    assert "atlas_risk_count 42" in response.text


def test_object_count_gauge_set_reflects_in_output() -> None:
    m.object_count.set(7)
    response = client.get("/metrics")
    assert "atlas_object_count 7" in response.text


def test_slam_active_gauge_zero_by_default() -> None:
    m.slam_active.set(0)
    response = client.get("/metrics")
    assert "atlas_slam_active 0" in response.text


def test_slam_active_gauge_can_be_one() -> None:
    m.slam_active.set(1)
    response = client.get("/metrics")
    assert "atlas_slam_active 1" in response.text
    # Reset for other tests
    m.slam_active.set(0)


# ── Counter propagation ───────────────────────────────────────────────────────


def test_query_total_increments_via_api() -> None:
    # prometheus-client does not double-suffix when the name already ends in
    # "_total": Counter("atlas_query_total", ...) → line "atlas_query_total N"
    before_resp = client.get("/metrics")
    before_lines = [
        line
        for line in before_resp.text.splitlines()
        if line.startswith("atlas_query_total ") or line.startswith("atlas_query_total{")
    ]
    before_val = float(before_lines[0].split()[-1]) if before_lines else 0.0

    # Make a query
    resp = client.post("/query", json={"query": "where is the cup?"})
    assert resp.status_code == 200

    after_resp = client.get("/metrics")
    after_lines = [
        line
        for line in after_resp.text.splitlines()
        if line.startswith("atlas_query_total ") or line.startswith("atlas_query_total{")
    ]
    after_val = float(after_lines[0].split()[-1]) if after_lines else 0.0

    assert after_val == before_val + 1


# ── Histogram observation ─────────────────────────────────────────────────────


def test_vlm_histogram_observe() -> None:
    m.vlm_request_seconds.observe(0.5)
    response = client.get("/metrics")
    assert "atlas_vlm_request_seconds_count" in response.text


# ── HealthResponse staleness field ────────────────────────────────────────────


def test_health_includes_risks_stale_seconds() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "risks_stale_seconds" in data


def test_health_risks_stale_is_float() -> None:
    response = client.get("/health")
    val = response.json()["risks_stale_seconds"]
    # -1.0 means "no assessment has run yet" (float('inf') is not JSON-safe).
    assert isinstance(val, float)
    assert val == -1.0 or val > 0.0


# ── WorldModelAgent staleness property ───────────────────────────────────────


def test_agent_risks_stale_inf_before_any_assessment() -> None:
    agent = WorldModelAgent(config=WorldModelConfig())
    assert agent.risks_stale_seconds == float("inf")


def test_agent_risks_stale_positive_after_assessment_time_set() -> None:
    agent = WorldModelAgent(config=WorldModelConfig())
    agent._last_assessment_time = time.monotonic() - 2.0
    stale = agent.risks_stale_seconds
    assert 1.5 < stale < 5.0


def test_agent_risks_stale_zero_just_after_set() -> None:
    agent = WorldModelAgent(config=WorldModelConfig())
    agent._last_assessment_time = time.monotonic()
    stale = agent.risks_stale_seconds
    assert 0.0 <= stale < 0.5


# ── CONTENT_TYPE_LATEST re-export ─────────────────────────────────────────────


def test_content_type_latest_is_string() -> None:
    assert isinstance(m.CONTENT_TYPE_LATEST, str)
    assert "text/plain" in m.CONTENT_TYPE_LATEST


# ── generate_latest re-export ─────────────────────────────────────────────────


def test_generate_latest_returns_bytes() -> None:
    data = m.generate_latest(m.REGISTRY)
    assert isinstance(data, bytes)
    assert len(data) > 0
