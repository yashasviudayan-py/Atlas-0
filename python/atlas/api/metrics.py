"""Prometheus metrics registry for Atlas-0.

All metrics use a dedicated :data:`REGISTRY` so the default
``prometheus_client`` global registry is not polluted during testing.

Exposed metrics
---------------
- ``atlas_risk_count`` (Gauge) — current number of active risks.
- ``atlas_object_count`` (Gauge) — current number of labeled objects.
- ``atlas_query_total`` (Counter) — total spatial queries processed since
  startup.
- ``atlas_ws_clients_active`` (Gauge) — currently connected WebSocket clients.
- ``atlas_slam_active`` (Gauge) — 1 if the Rust SLAM pipeline is connected,
  0 otherwise.
- ``atlas_assessment_age_seconds`` (Gauge) — seconds since the last successful
  world-model assessment cycle (staleness indicator).
- ``atlas_vlm_request_seconds`` (Histogram) — VLM inference latency buckets.

Usage::

    from atlas.api.metrics import risk_count, query_total

    risk_count.set(len(risks))
    query_total.inc()
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST as _CONTENT_TYPE_LATEST
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

__all__ = [
    "CONTENT_TYPE_LATEST",
    "REGISTRY",
    "assessment_age_seconds",
    "generate_latest",
    "object_count",
    "query_total",
    "risk_count",
    "slam_active",
    "vlm_request_seconds",
    "ws_clients_active",
]

#: Isolated registry — never touches the default process-wide registry.
REGISTRY: CollectorRegistry = CollectorRegistry()

#: Re-exported content-type constant for the /metrics route.
CONTENT_TYPE_LATEST: str = _CONTENT_TYPE_LATEST

# ── Gauges ────────────────────────────────────────────────────────────────────

risk_count: Gauge = Gauge(
    "atlas_risk_count",
    "Current number of active risks in the scene",
    registry=REGISTRY,
)

object_count: Gauge = Gauge(
    "atlas_object_count",
    "Current number of labeled semantic objects",
    registry=REGISTRY,
)

ws_clients_active: Gauge = Gauge(
    "atlas_ws_clients_active",
    "Number of currently connected WebSocket clients",
    registry=REGISTRY,
)

slam_active: Gauge = Gauge(
    "atlas_slam_active",
    "1 if the Rust SLAM pipeline is connected, 0 otherwise",
    registry=REGISTRY,
)

assessment_age_seconds: Gauge = Gauge(
    "atlas_assessment_age_seconds",
    "Seconds since the last successful world-model assessment cycle",
    registry=REGISTRY,
)

# ── Counter ───────────────────────────────────────────────────────────────────

query_total: Counter = Counter(
    "atlas_query_total",
    "Total number of spatial queries processed since startup",
    registry=REGISTRY,
)

# ── Histogram ─────────────────────────────────────────────────────────────────

vlm_request_seconds: Histogram = Histogram(
    "atlas_vlm_request_seconds",
    "VLM inference request latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=REGISTRY,
)
