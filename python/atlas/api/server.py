"""FastAPI app assembly for Atlas-0.

This module wires the application together: lifespan, production middleware
(security headers, tracing, request metrics, and a fixed-window rate limit),
the CORS policy, the static frontend mount, and the concern-specific routers.

The endpoint handlers and business logic live in dedicated modules:

- :mod:`atlas.api.state` — shared state, config, and singletons.
- :mod:`atlas.api.jobs` — job persistence and derived report fields.
- :mod:`atlas.api.analytics` — startup checks, descriptors, access control,
  and operator analytics.
- :mod:`atlas.api.pipeline` — the upload ingestion pipeline and sample report.
- :mod:`atlas.api.routers` — the HTTP routers grouped by concern.

A set of names is re-exported at the bottom for backwards compatibility with
callers and tests that import internals from ``atlas.api.server``.
"""

from __future__ import annotations

import contextlib
import json
import time
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from atlas.api.analytics import (
    _compare_job_to_benchmark,  # noqa: F401 — re-exported for tests/back-compat
    _load_eval_corpus_entries,  # noqa: F401 — re-exported for tests/back-compat
    _request_host,
    _run_startup_checks,
    _service_started_at,
)
from atlas.api.helpers import (
    _PUBLIC_PRODUCT_EVENTS,  # noqa: F401 — re-exported for tests/back-compat
    _safe_request_id,
    _trace_id_from_traceparent,
    _traceparent,
)
from atlas.api.jobs import (
    _ensure_job_derived_fields,  # noqa: F401 — re-exported for tests/back-compat
    _refresh_operational_metrics,
)
from atlas.api.metrics import http_requests_total, rate_limited_total
from atlas.api.models import UploadJobStatus
from atlas.api.pipeline import (
    _build_sample_report,  # noqa: F401 — re-exported for tests/back-compat
    _enqueue_upload_job,  # noqa: F401 — re-exported for tests/back-compat
    _process_upload,  # noqa: F401 — re-exported for tests/back-compat
    _resume_pending_upload_jobs,
    _start_upload_workers,
    _stop_upload_workers,
    run_detached_upload_worker,
)
from atlas.api.routers import core as core_router
from atlas.api.routers import operator as operator_router
from atlas.api.routers import product as product_router
from atlas.api.routers import uploads as uploads_router
from atlas.api.routers.core import prometheus_metrics
from atlas.api.state import (
    _FRONTEND_DIR,
    _api_cfg,
    _get_agent,
    _get_aggregator,
    _get_overlay_builder,
    _set_agent,
    _state,
    _upload_cfg,
    _upload_jobs,
    _upload_store,  # noqa: F401 — re-exported for tests/back-compat
)
from atlas.world_model.relationships import RelationType

logger = structlog.get_logger(__name__)


@contextlib.asynccontextmanager
async def _lifespan(application: FastAPI):  # type: ignore[type-arg]
    """Start the world-model agent on server startup; stop it on shutdown."""
    _service_started_at()
    startup = _run_startup_checks()
    if _upload_cfg.strict_startup_checks and not startup.get("ready"):
        raise RuntimeError(str(startup.get("summary")))
    agent = _get_agent()
    await agent.start()
    await _start_upload_workers()
    await _resume_pending_upload_jobs()
    _refresh_operational_metrics()
    logger.info("world_model_agent_started_via_lifespan")
    yield
    await _stop_upload_workers()
    await agent.stop()
    logger.info("world_model_agent_stopped_via_lifespan")


app = FastAPI(
    title="Atlas-0",
    description="Spatial Reasoning & Physical World-Model Engine API",
    version="0.1.0",
    lifespan=_lifespan,
)

_BASE_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'; "
        "form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(self), microphone=(), geolocation=()",
}
_PRIVATE_CACHE_PREFIXES = ("/upload", "/jobs", "/reports")


def _rate_limit_scope(request: Request) -> tuple[str, int] | None:
    """Return the configured rate-limit scope and quota for a request."""
    path = request.url.path
    method = request.method.upper()
    if method == "POST" and path == "/upload":
        return ("upload", int(_api_cfg.rate_limit_upload_requests or 0))
    if method == "POST" and path in {"/product/events", "/product/waitlist"}:
        return ("product_write", int(_api_cfg.rate_limit_public_requests or 0))
    return None


def _rate_limit_key(request: Request, scope: str) -> str:
    """Return a privacy-light per-client rate-limit key."""
    host = _request_host(request) or "unknown"
    return f"{scope}:{host}"


def _prune_rate_limit_buckets(
    buckets: dict[str, dict[str, float | int]],
    *,
    now: float,
    incoming_key: str,
) -> None:
    """Drop expired or oldest rate-limit buckets before adding a new client."""
    expired_keys = [
        key for key, bucket in buckets.items() if now >= float(bucket.get("reset_at", 0.0))
    ]
    for key in expired_keys:
        buckets.pop(key, None)

    max_buckets = int(_api_cfg.rate_limit_max_buckets or 0)
    if max_buckets <= 0 or incoming_key in buckets:
        return

    overflow = len(buckets) - max_buckets + 1
    if overflow <= 0:
        return

    oldest_keys = sorted(
        buckets,
        key=lambda key: float(buckets[key].get("reset_at", 0.0)),
    )
    for key in oldest_keys[:overflow]:
        buckets.pop(key, None)


def _check_rate_limit(request: Request) -> tuple[bool, str | None, int | None]:
    """Apply a fixed-window in-memory rate limit for public write endpoints."""
    scoped = _rate_limit_scope(request)
    if scoped is None:
        return (True, None, None)
    scope, limit = scoped
    if limit <= 0:
        return (True, scope, None)

    now = time.monotonic()
    window_seconds = float(_api_cfg.rate_limit_window_seconds or 60.0)
    key = _rate_limit_key(request, scope)
    buckets = _state.setdefault("rate_limit_buckets", {})
    _prune_rate_limit_buckets(buckets, now=now, incoming_key=key)
    bucket = buckets.get(key)
    if not bucket or now >= float(bucket.get("reset_at", 0.0)):
        buckets[key] = {"count": 1, "reset_at": now + window_seconds}
        return (True, scope, int(window_seconds))

    bucket["count"] = int(bucket.get("count", 0)) + 1
    retry_after = max(1, int(float(bucket.get("reset_at", now)) - now))
    return (bucket["count"] <= limit, scope, retry_after)


def _metric_path(request: Request) -> str:
    """Return a low-cardinality route path for request metrics."""
    route = request.scope.get("route")
    return str(getattr(route, "path", None) or request.url.path)


@app.middleware("http")
async def _add_production_headers(request: Request, call_next: Any) -> Response:
    """Apply browser hardening, trace, metric, and rate-limit controls."""
    request_id = _safe_request_id(request.headers.get("x-request-id"))
    trace_id = _trace_id_from_traceparent(request.headers.get("traceparent"))
    allowed, scope, retry_after = _check_rate_limit(request)
    if not allowed:
        rate_limited_total.labels(scope=scope or "unknown").inc()
        response = Response(
            content=json.dumps({"detail": "Too many requests. Please retry shortly."}),
            media_type="application/json",
            status_code=429,
            headers={"Retry-After": str(retry_after or 1)},
        )
    else:
        response = await call_next(request)

    for header, value in _BASE_SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    response.headers.setdefault("X-Request-ID", request_id)
    response.headers.setdefault("traceparent", _traceparent(trace_id))

    path = request.url.path
    if path in _PRIVATE_CACHE_PREFIXES or path.startswith(("/jobs/", "/reports/")):
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("Pragma", "no-cache")
    http_requests_total.labels(
        method=request.method.upper(),
        path=_metric_path(request),
        status=str(response.status_code),
    ).inc()
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=_api_cfg.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Audience-Mode",
        "X-Filename",
        "X-Room-Label",
        "X-Request-ID",
        "traceparent",
    ],
    expose_headers=["X-Request-ID", "traceparent", "Retry-After"],
)

# The mount is optional: if the directory doesn't exist (e.g. stripped Docker
# layer) the API still starts normally.
if _FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")

    @app.get("/", include_in_schema=False)
    async def _root_redirect() -> RedirectResponse:
        """Send visitors who hit the bare host straight to the web app."""
        return RedirectResponse(url="/app/", status_code=307)

    @app.get("/favicon.ico", include_in_schema=False)
    async def _favicon_redirect() -> RedirectResponse:
        """Serve the app icon for browsers requesting a root favicon."""
        return RedirectResponse(url="/app/atlas-icon.svg", status_code=307)

app.include_router(core_router.router)
app.include_router(product_router.router)
app.include_router(operator_router.router)
app.include_router(uploads_router.router)


# Re-export for convenience (e.g. tests that patch the agent).
__all__ = [
    "RelationType",
    "UploadJobStatus",
    "_get_agent",
    "_get_aggregator",
    "_get_overlay_builder",
    "_set_agent",
    "_state",
    "_upload_jobs",
    "app",
    "prometheus_metrics",
    "run_detached_upload_worker",
]
