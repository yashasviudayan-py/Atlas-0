"""FastAPI server exposing Atlas-0 spatial queries and AR overlay data.

Provides endpoints for:
- Querying the semantic 3D map with natural language.
- Listing all labeled objects and their physical properties.
- Returning the full current scene state.
- Streaming risk assessments via WebSocket.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hmac
import io
import ipaddress
import json
import pathlib
import uuid
from datetime import UTC, datetime
from email.parser import BytesParser
from email.policy import default
from typing import Annotated, Any

import structlog
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from atlas.api.metrics import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    assessment_age_seconds,
    generate_latest,
    object_count,
    query_total,
    risk_count,
    slam_active,
    ws_clients_active,
)
from atlas.api.overlay import CameraParams, OverlayBuilder
from atlas.api.upload_analysis import (
    analyze_uploaded_image,
    analyze_uploaded_video,
    build_finding_replays,
)
from atlas.api.upload_store import UploadStore
from atlas.utils.config import load_config
from atlas.utils.video import probe_video_metadata
from atlas.vlm.inference import SemanticLabel, VLMConfig, VLMEngine
from atlas.world_model.agent import RiskEntry, WorldModelAgent
from atlas.world_model.hazards import (
    audience_mode_label,
    build_room_wins,
    build_weekend_fix_list,
    normalize_audience_mode,
)
from atlas.world_model.query_parser import QueryParser, QueryType
from atlas.world_model.relationships import RelationType, SemanticObject
from atlas.world_model.risk_aggregator import RiskAggregator

logger = structlog.get_logger(__name__)


@contextlib.asynccontextmanager
async def _lifespan(application: FastAPI):  # type: ignore[type-arg]
    """Start the world-model agent on server startup; stop it on shutdown."""
    agent = _get_agent()
    await agent.start()
    await _start_upload_workers()
    await _resume_pending_upload_jobs()
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

# ── Static frontend ───────────────────────────────────────────────────────────
# Serve the Three.js AR overlay frontend from the repo's frontend/ directory.
# The mount is optional: if the directory doesn't exist (e.g. stripped Docker
# layer) the API still starts normally.
_FRONTEND_DIR = pathlib.Path(__file__).parents[3] / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")

# ── Application state ─────────────────────────────────────────────────────────
# Stored in a dict to avoid `global` statements (PLW0603).

_state: dict[str, Any] = {}
_query_parser = QueryParser()
_runtime_cfg = load_config()
_api_cfg = _runtime_cfg.api
_upload_cfg = _runtime_cfg.uploads
_evaluation_cfg = _runtime_cfg.evaluation
_upload_store = UploadStore(
    pathlib.Path(_upload_cfg.storage_dir),
    save_original_uploads=_upload_cfg.save_original_uploads,
    max_persisted_jobs=_upload_cfg.max_persisted_jobs,
    retention_days=_upload_cfg.retention_days,
    max_storage_bytes=_upload_cfg.max_storage_bytes,
)


def _get_agent() -> WorldModelAgent:
    """FastAPI dependency: lazily create and return the singleton agent."""
    if "agent" not in _state:
        _state["agent"] = WorldModelAgent()
    return _state["agent"]


def _set_agent(agent: WorldModelAgent) -> None:
    """Replace the singleton agent — used by tests to inject a mock."""
    _state["agent"] = agent


def _get_aggregator() -> RiskAggregator:
    """Return the singleton :class:`~atlas.world_model.risk_aggregator.RiskAggregator`."""
    if "aggregator" not in _state:
        _state["aggregator"] = RiskAggregator()
    return _state["aggregator"]  # type: ignore[return-value]


def _get_overlay_builder() -> OverlayBuilder:
    """Return the singleton :class:`~atlas.api.overlay.OverlayBuilder`."""
    if "overlay_builder" not in _state:
        _state["overlay_builder"] = OverlayBuilder(camera=CameraParams())
    return _state["overlay_builder"]  # type: ignore[return-value]


def _build_runtime_vlm_config() -> VLMConfig:
    """Build a :class:`VLMConfig` from the active Atlas runtime config."""
    vlm = load_config().vlm
    return VLMConfig(
        provider=vlm.provider,
        fallback_provider=vlm.fallback_provider,
        model_name=vlm.model_name,
        ollama_host=vlm.ollama_host,
        claude_model=vlm.claude_model,
        openai_model=vlm.openai_model,
        max_tokens=vlm.max_tokens,
        temperature=vlm.temperature,
        timeout_seconds=vlm.timeout_seconds,
    )


def _current_objects(agent: WorldModelAgent) -> list[SemanticObject]:
    """Return the best available object list for API responses."""
    objects = agent.get_objects_sync()
    return objects or agent.build_objects_from_store()


# ── Pydantic models ───────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Response model for GET /health."""

    status: str
    slam_active: bool
    vlm_active: bool
    frame_count: int
    object_count: int
    risk_count: int
    risks_stale_seconds: float


class SpatialQuery(BaseModel):
    """Request body for POST /query."""

    query: str
    max_results: int = 5


class SpatialQueryResult(BaseModel):
    """A single result returned by POST /query."""

    object_label: str
    position: list[float]
    confidence: float
    risk_level: float
    description: str


class ObjectInfo(BaseModel):
    """Physical and spatial metadata for one labeled object."""

    object_id: int
    label: str
    material: str
    mass_kg: float
    fragility: float
    friction: float
    confidence: float
    position: list[float]
    relationships: list[str]


class RiskInfo(BaseModel):
    """Summary of one risk entry."""

    object_id: int
    object_label: str
    position: list[float]
    risk_score: float
    description: str


class SceneState(BaseModel):
    """Full snapshot of the current scene."""

    object_count: int
    objects: list[ObjectInfo]
    risk_count: int
    risks: list[RiskInfo]
    point_cloud: list[list[float]] = []
    """Pseudo-depth point cloud from uploaded images — each entry is [x, y, z, r, g, b]
    where rgb is normalised 0-1.  Used by the 3DGS frontend to render upload-derived
    structure in world space."""


class OperatorAccessResponse(BaseModel):
    """Public-facing access policy used by the hosted frontend."""

    requires_token: bool
    allow_unauthenticated_loopback: bool
    enable_job_listing: bool
    mode: str


class OperatorSettingsResponse(BaseModel):
    """Protected operator diagnostics for hosted beta deployments."""

    access: dict[str, Any]
    uploads: dict[str, Any]
    queue: dict[str, Any]
    storage: dict[str, Any]
    providers: dict[str, Any]
    evaluation: dict[str, Any]
    product: dict[str, Any]


class PrivacyPolicyResponse(BaseModel):
    """Public privacy posture exposed to the hosted frontend."""

    retention_days: int
    save_original_uploads: bool
    delete_supported: bool
    text_redaction_enabled: bool
    summary: str
    details: list[str]


class OverlayRiskEntry(BaseModel):
    """Richer risk payload sent over the WebSocket delta stream.

    Carries physics + heuristic merged scores, trajectory data, and
    pre-built overlay primitives ready for the Three.js renderer.
    """

    object_id: int
    object_label: str
    position: list[float] | None
    combined_score: float
    physics_score: float
    heuristic_score: float
    risk_type: str
    impact_point: list[float] | None
    trajectory_points: list[list[float]] | None
    description: str
    overlay: dict[str, Any]


class RiskDeltaMessage(BaseModel):
    """Delta update message pushed over ``/ws/risks``.

    Only items that changed since the previous tick are included.
    """

    added: list[OverlayRiskEntry]
    updated: list[OverlayRiskEntry]
    removed: list[int]


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check(
    agent: Annotated[WorldModelAgent, Depends(_get_agent)],
) -> HealthResponse:
    """Check system health and component status.

    Returns counts of currently labeled objects and active risks.
    ``slam_active`` reflects whether a live Rust SLAM pipeline is connected
    (always ``False`` until Phase 1/2 integration in Part 8).
    """
    objects = _current_objects(agent)
    risks = await agent.get_risks()
    snapshot = agent.get_latest_snapshot_sync()
    frame_count = int(getattr(snapshot, "frame_id", 0)) if snapshot is not None else 0
    slam_is_live = snapshot is not None and (
        frame_count > 0 or len(getattr(snapshot, "gaussians", ())) > 0
    )
    stale = agent.risks_stale_seconds
    # -1.0 means "no assessment has completed yet" (JSON can't encode inf).
    stale_json = stale if stale != float("inf") else -1.0
    # Update Prometheus gauges on every health poll.
    object_count.set(len(objects))
    risk_count.set(len(risks))
    slam_active.set(int(slam_is_live))
    assessment_age_seconds.set(stale_json)
    return HealthResponse(
        status="ok",
        slam_active=slam_is_live,
        vlm_active=agent.vlm_active,
        frame_count=frame_count,
        object_count=len(objects),
        risk_count=len(risks),
        risks_stale_seconds=stale_json,
    )


@app.post("/query", response_model=list[SpatialQueryResult])
async def spatial_query(
    query: SpatialQuery,
    agent: Annotated[WorldModelAgent, Depends(_get_agent)],
) -> list[SpatialQueryResult]:
    """Query the semantic 3D map with natural language.

    Supported query types (auto-detected):

    - **RISK**: "What is the most unstable object?" → ranked by risk score.
    - **LOCATION**: "Where is the glass?" → object positions.
    - **PROPERTY**: "What is the laptop made of?" → material / mass / fragility.
    - **SPATIAL_RELATION**: "What is on top of the table?" → relationship lookup.

    Returns up to ``max_results`` matching entries.

    Example::

        POST /query
        {"query": "Where is the most unstable object?", "max_results": 3}
    """
    parsed = _query_parser.parse(query.query)
    objects = _current_objects(agent)
    risks = await agent.get_risks()

    results: list[SpatialQueryResult] = []

    if parsed.query_type == QueryType.RISK:
        for risk in risks[: query.max_results]:
            results.append(
                SpatialQueryResult(
                    object_label=risk.object_label,
                    position=list(risk.position),
                    confidence=risk.risk_score,
                    risk_level=risk.risk_score,
                    description=risk.description,
                )
            )

    elif parsed.query_type == QueryType.LOCATION:
        subject = parsed.subject.lower()
        for obj in objects:
            if subject and subject not in obj.label.lower():
                continue
            results.append(
                SpatialQueryResult(
                    object_label=obj.label,
                    position=list(obj.position),
                    confidence=obj.confidence,
                    risk_level=_risk_score_for(obj.object_id, risks),
                    description=f"{obj.label} at {_fmt_pos(obj.position)}",
                )
            )
            if len(results) >= query.max_results:
                break

    elif parsed.query_type == QueryType.PROPERTY:
        subject = parsed.subject.lower()
        for obj in objects:
            if subject and subject not in obj.label.lower():
                continue
            value = getattr(obj, parsed.predicate, None)
            desc = (
                f"{obj.label}: {parsed.predicate} = {value}"
                if value is not None
                else f"{obj.label}: {parsed.predicate} unknown"
            )
            results.append(
                SpatialQueryResult(
                    object_label=obj.label,
                    position=list(obj.position),
                    confidence=obj.confidence,
                    risk_level=_risk_score_for(obj.object_id, risks),
                    description=desc,
                )
            )
            if len(results) >= query.max_results:
                break

    elif parsed.query_type == QueryType.SPATIAL_RELATION:
        ref = parsed.reference_object.lower()
        relation = parsed.relation
        obj_by_id = {o.object_id: o for o in objects}
        found = 0
        for obj in objects:
            if found >= query.max_results:
                break
            for rel in obj.relationships:
                if relation and relation not in rel.relation.value:
                    continue
                target = obj_by_id.get(rel.target_object_id)
                if ref and (target is None or ref not in target.label.lower()):
                    continue
                target_label = target.label if target is not None else "unknown"
                results.append(
                    SpatialQueryResult(
                        object_label=obj.label,
                        position=list(obj.position),
                        confidence=rel.confidence,
                        risk_level=_risk_score_for(obj.object_id, risks),
                        description=f"{obj.label} is {rel.relation.value} {target_label}",
                    )
                )
                found += 1
                if found >= query.max_results:
                    break

    else:
        # UNKNOWN — fuzzy label search
        subject = parsed.subject.lower()
        for obj in objects:
            if subject and subject not in obj.label.lower():
                continue
            results.append(
                SpatialQueryResult(
                    object_label=obj.label,
                    position=list(obj.position),
                    confidence=obj.confidence,
                    risk_level=_risk_score_for(obj.object_id, risks),
                    description=f"{obj.label} at {_fmt_pos(obj.position)}",
                )
            )
            if len(results) >= query.max_results:
                break

    query_total.inc()
    logger.info(
        "spatial_query_resolved",
        query=query.query,
        query_type=parsed.query_type.value,
        results=len(results),
    )
    return results


@app.get("/objects", response_model=list[ObjectInfo])
async def list_objects(
    agent: Annotated[WorldModelAgent, Depends(_get_agent)],
) -> list[ObjectInfo]:
    """List all labeled objects with their physical properties and relationships.

    Returns an empty list when no map snapshot has been processed yet.
    """
    objects = _current_objects(agent)
    risks = await agent.get_risks()
    return [_to_object_info(obj, risks) for obj in objects]


@app.get("/scene", response_model=SceneState)
async def get_scene(
    agent: Annotated[WorldModelAgent, Depends(_get_agent)],
) -> SceneState:
    """Return the full current scene state including all objects and risks.

    Useful for initialising an AR frontend or running a one-shot scene dump.
    """
    objects = _current_objects(agent)
    risks = await agent.get_risks()

    # Merge point clouds from all completed upload jobs
    all_points: list[list[float]] = []
    for job in _upload_jobs.values():
        if job.get("status") == "complete":
            all_points.extend(job.get("point_cloud") or [])

    return SceneState(
        object_count=len(objects),
        objects=[_to_object_info(obj, risks) for obj in objects],
        risk_count=len(risks),
        risks=[_to_risk_info(r) for r in risks],
        point_cloud=all_points,
    )


@app.get("/operator/access", response_model=OperatorAccessResponse)
def operator_access() -> OperatorAccessResponse:
    """Expose the minimal hosted-access policy needed by the frontend."""
    return OperatorAccessResponse(**_operator_access_descriptor())


@app.get("/product/privacy", response_model=PrivacyPolicyResponse)
def product_privacy() -> PrivacyPolicyResponse:
    """Expose user-visible privacy and deletion controls."""
    return PrivacyPolicyResponse(**_public_privacy_descriptor())


@app.get("/operator/settings", response_model=OperatorSettingsResponse)
def operator_settings(request: Request) -> OperatorSettingsResponse:
    """Return protected operator diagnostics and upload-policy visibility."""
    _require_private_access(request)
    counts = _job_status_counts()
    return OperatorSettingsResponse(
        access=_operator_access_descriptor(),
        uploads={
            "save_original_uploads": _upload_cfg.save_original_uploads,
            "retention_days": _upload_cfg.retention_days,
            "max_upload_bytes": _upload_cfg.max_upload_bytes,
            "max_video_duration_seconds": _upload_cfg.max_video_duration_seconds,
            "max_concurrent_jobs": _upload_cfg.max_concurrent_jobs,
            "max_queue_depth": _upload_cfg.max_queue_depth,
            "max_job_attempts": _upload_cfg.max_job_attempts,
            "job_timeout_seconds": _upload_cfg.job_timeout_seconds,
            "max_storage_bytes": _upload_cfg.max_storage_bytes,
        },
        queue={
            "queued_jobs": counts["queued"],
            "processing_jobs": counts["processing"],
            "completed_jobs": counts["complete"],
            "failed_jobs": counts["error"],
            "worker_count": _upload_cfg.max_concurrent_jobs,
        },
        storage=_upload_store.storage_summary(),
        providers=_provider_runtime_summary(),
        evaluation=_aggregate_evaluation_metrics(),
        product=_aggregate_product_metrics(),
    )


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    """Expose Prometheus metrics for scraping.

    Returns metrics in the standard Prometheus text exposition format.
    Compatible with any Prometheus-compatible scraper (Prometheus, Grafana
    Agent, VictoriaMetrics, etc.).

    The endpoint is excluded from the OpenAPI schema to avoid cluttering
    the Swagger UI.

    Returns:
        Plain-text Prometheus metrics response.
    """
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.websocket("/ws/risks")
async def risk_stream(
    websocket: WebSocket,
    agent: Annotated[WorldModelAgent, Depends(_get_agent)],
) -> None:
    """Stream real-time risk delta updates to the AR overlay.

    Each message is a :class:`RiskDeltaMessage` JSON object containing only
    the entries that changed since the previous tick:

    .. code-block:: json

        {
          "added":   [{ ...OverlayRiskEntry... }],
          "updated": [{ ...OverlayRiskEntry... }],
          "removed": [42, 7]
        }

    Unchanged entries are omitted to minimise bandwidth.  The frontend should
    maintain its own risk map keyed by ``object_id`` and apply the delta.

    The stream includes pre-built overlay primitives (risk zones, trajectory
    arcs, impact zones, alert text) for direct use by the Three.js renderer.
    """
    await websocket.accept()
    ws_clients_active.inc()
    aggregator = _get_aggregator()
    overlay_builder = _get_overlay_builder()
    # Track previous scores to detect changes; map object_id → combined_score.
    prev_scores: dict[int, float] = {}

    try:
        while True:
            # Sync heuristic risks from the agent into the aggregator.
            heuristic_risks = await agent.get_risks()
            aggregator.update_heuristic_risks(heuristic_risks)

            current = aggregator.get_top_risks()
            current_ids = {r.object_id for r in current}

            removed = [oid for oid in prev_scores if oid not in current_ids]

            added: list[OverlayRiskEntry] = []
            updated: list[OverlayRiskEntry] = []

            for risk in current:
                overlay_primitives = overlay_builder.build_from_risk(risk)
                overlay_payload = {
                    "risk_zone": overlay_primitives["risk_zone"].to_dict(),
                    "trajectory_arc": (
                        overlay_primitives["trajectory_arc"].to_dict()
                        if overlay_primitives["trajectory_arc"] is not None
                        else None
                    ),
                    "impact_zone": (
                        overlay_primitives["impact_zone"].to_dict()
                        if overlay_primitives["impact_zone"] is not None
                        else None
                    ),
                    "alert": overlay_primitives["alert"].to_dict(),
                }

                traj_arc = overlay_primitives["trajectory_arc"]
                entry = OverlayRiskEntry(
                    object_id=risk.object_id,
                    object_label=risk.object_label,
                    position=list(risk.position) if risk.position is not None else None,
                    combined_score=risk.combined_score,
                    physics_score=risk.physics_score,
                    heuristic_score=risk.heuristic_score,
                    risk_type=risk.risk_type,
                    impact_point=(
                        list(risk.impact_point) if risk.impact_point is not None else None
                    ),
                    trajectory_points=(
                        [list(p) for p in traj_arc.points] if traj_arc is not None else None
                    ),
                    description=risk.description,
                    overlay=overlay_payload,
                )

                if risk.object_id not in prev_scores:
                    added.append(entry)
                elif abs(risk.combined_score - prev_scores[risk.object_id]) > 0.01:
                    updated.append(entry)

            prev_scores = {r.object_id: r.combined_score for r in current}

            if added or updated or removed:
                msg = RiskDeltaMessage(added=added, updated=updated, removed=removed)
                await websocket.send_json(msg.model_dump())

            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("ws_risks_error", error=str(exc))
    finally:
        ws_clients_active.dec()
        await websocket.close()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _risk_score_for(object_id: int, risks: list[RiskEntry]) -> float:
    """Look up the risk score for *object_id*; return 0.0 if not in the list.

    Args:
        object_id: Object to look up.
        risks: Current risk list.

    Returns:
        Risk score in [0, 1].
    """
    for risk in risks:
        if risk.object_id == object_id:
            return risk.risk_score
    return 0.0


def _fmt_pos(pos: tuple[float, float, float]) -> str:
    """Format a 3D position as ``(x.xx, y.yy, z.zz)``."""
    return f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"


def _to_object_info(obj: SemanticObject, risks: list[RiskEntry]) -> ObjectInfo:
    """Convert *obj* to an :class:`ObjectInfo` response model.

    Args:
        obj: Source semantic object.
        risks: Current risk list (used to look up risk_level).

    Returns:
        :class:`ObjectInfo` response.
    """
    return ObjectInfo(
        object_id=obj.object_id,
        label=obj.label,
        material=obj.material,
        mass_kg=obj.mass_kg,
        fragility=obj.fragility,
        friction=obj.friction,
        confidence=obj.confidence,
        position=list(obj.position),
        relationships=[f"{r.relation.value}:{r.target_object_id}" for r in obj.relationships],
    )


def _to_risk_info(risk: RiskEntry) -> RiskInfo:
    """Convert a :class:`RiskEntry` to a :class:`RiskInfo` response model.

    Args:
        risk: Source risk entry.

    Returns:
        :class:`RiskInfo` response.
    """
    return RiskInfo(
        object_id=risk.object_id,
        object_label=risk.object_label,
        position=list(risk.position),
        risk_score=risk.risk_score,
        description=risk.description,
    )


# ── Upload job models ─────────────────────────────────────────────────────────

_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"})
_VIDEO_TYPES = frozenset({"video/mp4", "video/quicktime", "video/webm", "video/x-msvideo"})

# In-memory job store (keyed by job_id).
_upload_jobs: dict[str, dict[str, Any]] = _upload_store.load_jobs()


def _save_job(job: dict[str, Any]) -> None:
    """Persist the current in-memory job manifest to disk."""
    _upload_store.save_job(job)


def _update_job(job: dict[str, Any], **fields: Any) -> None:
    """Update a job dict and persist the new state."""
    job.update(fields)
    _ensure_job_derived_fields(job)
    _save_job(job)


def _job_artifacts(job: dict[str, Any]) -> dict[str, Any]:
    """Return the mutable artifact map stored on one job."""
    artifacts = job.get("artifacts")
    if isinstance(artifacts, dict):
        return artifacts
    fresh: dict[str, Any] = {}
    job["artifacts"] = fresh
    return fresh


def _set_job_artifact(job: dict[str, Any], name: str, pointer: dict[str, Any] | None) -> None:
    """Attach or remove one artifact pointer on a job."""
    artifacts = _job_artifacts(job)
    if pointer is None:
        artifacts.pop(name, None)
    else:
        artifacts[name] = pointer


def _refresh_job_artifacts(job: dict[str, Any]) -> None:
    """Rebuild artifact pointers for one persisted job from files on disk."""
    job_id = str(job.get("job_id", ""))
    if not job_id:
        return

    _job_artifacts(job)

    if _upload_store.has_job_input(job_id):
        queued_path = next(_upload_store.job_dir(job_id).glob("queued-input.*"), None)
        if queued_path is not None:
            _set_job_artifact(
                job,
                "queued_input",
                _upload_store.artifact_pointer(
                    job_id,
                    queued_path.relative_to(_upload_store.job_dir(job_id)),
                    kind="queued_input",
                    media_type=str(job.get("content_type") or "application/octet-stream"),
                ),
            )
    else:
        _set_job_artifact(job, "queued_input", None)

    upload_path = next(_upload_store.job_dir(job_id).glob("upload.*"), None)
    if upload_path is not None:
        _set_job_artifact(
            job,
            "original_upload",
            _upload_store.artifact_pointer(
                job_id,
                upload_path.relative_to(_upload_store.job_dir(job_id)),
                kind="original_upload",
                media_type=str(job.get("content_type") or "application/octet-stream"),
            ),
        )
    else:
        _set_job_artifact(job, "original_upload", None)

    report_path = _upload_store.report_path(job_id)
    if report_path.exists():
        _set_job_artifact(
            job,
            "report_pdf",
            _upload_store.artifact_pointer(
                job_id,
                report_path.relative_to(_upload_store.job_dir(job_id)),
                kind="report_pdf",
                media_type="application/pdf",
                url=str(job.get("report_url") or f"/reports/{job_id}.pdf"),
            ),
        )
    else:
        _set_job_artifact(job, "report_pdf", None)

    evidence_artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(_upload_store.evidence_dir(job_id).glob("*")):
        if not path.is_file():
            continue
        evidence_id = path.stem
        media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        evidence_artifacts[evidence_id] = _upload_store.artifact_pointer(
            job_id,
            path.relative_to(_upload_store.job_dir(job_id)),
            kind="evidence_image",
            media_type=media_type,
            url=f"/jobs/{job_id}/evidence/{evidence_id}",
        )
    _set_job_artifact(job, "evidence", evidence_artifacts or None)

    for frame in job.get("evidence_frames") or []:
        evidence_id = str(frame.get("evidence_id", ""))
        if evidence_id and evidence_id in evidence_artifacts:
            frame["artifact"] = evidence_artifacts[evidence_id]

    replay_artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(_upload_store.replay_dir(job_id).glob("*.gif")):
        if not path.is_file():
            continue
        replay_id = path.stem
        replay_artifacts[replay_id] = _upload_store.artifact_pointer(
            job_id,
            path.relative_to(_upload_store.job_dir(job_id)),
            kind="finding_replay",
            media_type="image/gif",
            url=f"/jobs/{job_id}/replays/{replay_id}",
        )
    _set_job_artifact(job, "finding_replays", replay_artifacts or None)

    for risk in job.get("risks") or []:
        replay = risk.get("replay")
        if not isinstance(replay, dict):
            continue
        replay_id = str(replay.get("replay_id", ""))
        if replay_id and replay_id in replay_artifacts:
            replay["image_url"] = replay_artifacts[replay_id]["url"]
            replay["media_type"] = replay_artifacts[replay_id]["media_type"]
            replay["artifact"] = replay_artifacts[replay_id]


for _job in _upload_jobs.values():
    _refresh_job_artifacts(_job)


def _feedback_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    """Return aggregate verdict counts for stored feedback events."""
    counts = {"useful": 0, "wrong": 0, "duplicate": 0}
    for event in events:
        verdict = str(event.get("verdict", "")).lower()
        if verdict in counts:
            counts[verdict] += 1
    return counts


def _eval_corpus_dir() -> pathlib.Path:
    """Return the directory holding seeded evaluation fixtures."""
    return pathlib.Path(__file__).parents[3] / "data" / "eval_corpus"


def _load_eval_corpus_entries() -> list[dict[str, Any]]:
    """Load all seeded evaluation corpus entries from disk."""
    entries: list[dict[str, Any]] = []
    corpus_dir = _eval_corpus_dir()
    if not corpus_dir.exists():
        return entries

    for path in sorted(corpus_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("label"):
            entries.append(payload)
    return entries


def _load_expected_benchmark(label: str) -> dict[str, Any] | None:
    """Load a supported benchmark fixture definition for comparison."""
    for entry in _load_eval_corpus_entries():
        if str(entry.get("label")) == label:
            return entry
    return None


def _auto_benchmark_label(job: dict[str, Any]) -> str | None:
    """Infer a benchmark label for known regression fixtures."""
    filename = str((job.get("summary") or {}).get("filename") or job.get("filename") or "")
    for entry in _load_eval_corpus_entries():
        if filename and filename == str(entry.get("fixture_name", "")):
            return str(entry.get("label"))
    return None


def _compare_job_to_benchmark(job: dict[str, Any], label: str) -> dict[str, Any] | None:
    """Compare one completed job against a supported benchmark fixture."""
    expected = _load_expected_benchmark(label)
    if expected is None:
        return None

    risks = list(job.get("risks") or [])
    summary = dict(job.get("summary") or {})
    hazard_codes = [str(risk.get("hazard_code", "")) for risk in risks]
    missing_codes = [
        code for code in expected.get("required_hazard_codes", []) if code not in hazard_codes
    ]
    matched = (
        str(job.get("scene_source", "")) == str(expected.get("scene_source", ""))
        and int(summary.get("object_count", 0) or 0)
        >= int(expected.get("min_object_count", 0) or 0)
        and int(summary.get("hazard_count", 0) or 0)
        >= int(expected.get("min_hazard_count", 0) or 0)
        and (hazard_codes[0] if hazard_codes else None) == expected.get("top_hazard_code")
        and not missing_codes
    )
    return {
        "label": label,
        "matched": matched,
        "missing_hazard_codes": missing_codes,
        "top_hazard_code": hazard_codes[0] if hazard_codes else None,
        "expected_top_hazard_code": expected.get("top_hazard_code"),
        "expected_min_object_count": expected.get("min_object_count"),
        "expected_min_hazard_count": expected.get("min_hazard_count"),
    }


def _aggregate_evaluation_metrics() -> dict[str, Any]:
    """Aggregate evaluation and benchmark signals across persisted jobs."""
    complete_jobs = [job for job in _upload_jobs.values() if str(job.get("status")) == "complete"]
    total_complete = len(complete_jobs)
    reviewed_jobs = 0
    benchmarked_jobs = 0
    benchmark_matches = 0
    jobs_with_missed_hazards = 0
    jobs_needing_review = 0
    disputed_jobs = 0
    coverage_sum = 0.0

    for job in complete_jobs:
        summary = dict(job.get("evaluation_summary") or {})
        human = dict(job.get("human_evaluation") or {})
        if human:
            reviewed_jobs += 1
        if bool(summary.get("benchmark_label")):
            benchmarked_jobs += 1
        if bool(summary.get("benchmark_match")):
            benchmark_matches += 1
        if int(summary.get("missed_hazard_count", 0) or 0) > 0:
            jobs_with_missed_hazards += 1
        if bool(summary.get("needs_review")):
            jobs_needing_review += 1
        if int(summary.get("disputed_findings", 0) or 0) > 0:
            disputed_jobs += 1
        coverage_sum += float(summary.get("review_coverage", 0.0) or 0.0)

    avg_review_coverage = round(coverage_sum / total_complete, 2) if total_complete else 0.0
    benchmark_match_rate = (
        round(benchmark_matches / benchmarked_jobs, 2) if benchmarked_jobs else 0.0
    )
    false_positive_job_rate = round(disputed_jobs / total_complete, 2) if total_complete else 0.0
    missed_hazard_rate = (
        round(jobs_with_missed_hazards / total_complete, 2) if total_complete else 0.0
    )
    corpus_entries = _load_eval_corpus_entries()
    release_gates = _evaluation_release_gates(
        reviewed_jobs=reviewed_jobs,
        benchmarked_jobs=benchmarked_jobs,
        benchmark_match_rate=benchmark_match_rate,
        false_positive_job_rate=false_positive_job_rate,
        missed_hazard_rate=missed_hazard_rate,
        avg_review_coverage=avg_review_coverage,
        seed_fixture_count=len(corpus_entries),
    )

    return {
        "completed_jobs": total_complete,
        "reviewed_jobs": reviewed_jobs,
        "benchmarked_jobs": benchmarked_jobs,
        "benchmark_match_rate": benchmark_match_rate,
        "jobs_with_missed_hazards": jobs_with_missed_hazards,
        "missed_hazard_rate": missed_hazard_rate,
        "jobs_needing_review": jobs_needing_review,
        "false_positive_job_rate": false_positive_job_rate,
        "avg_review_coverage": avg_review_coverage,
        "seed_fixture_count": len(corpus_entries),
        "target_corpus_size": _evaluation_cfg.target_corpus_size,
        "release_gates": release_gates,
    }


def _evaluation_release_gates(
    *,
    reviewed_jobs: int,
    benchmarked_jobs: int,
    benchmark_match_rate: float,
    false_positive_job_rate: float,
    missed_hazard_rate: float,
    avg_review_coverage: float,
    seed_fixture_count: int,
) -> dict[str, Any]:
    """Build a release-gate summary for operator review."""
    gates = [
        {
            "id": "reviewed_jobs",
            "label": "Reviewed jobs",
            "actual": reviewed_jobs,
            "target": _evaluation_cfg.min_reviewed_jobs,
            "passed": reviewed_jobs >= _evaluation_cfg.min_reviewed_jobs,
        },
        {
            "id": "benchmark_match_rate",
            "label": "Benchmark match rate",
            "actual": benchmark_match_rate,
            "target": _evaluation_cfg.min_benchmark_match_rate,
            "passed": benchmarked_jobs > 0
            and benchmark_match_rate >= _evaluation_cfg.min_benchmark_match_rate,
        },
        {
            "id": "false_positive_job_rate",
            "label": "False-positive job rate",
            "actual": false_positive_job_rate,
            "target": _evaluation_cfg.max_false_positive_job_rate,
            "passed": false_positive_job_rate <= _evaluation_cfg.max_false_positive_job_rate,
        },
        {
            "id": "missed_hazard_rate",
            "label": "Missed-hazard job rate",
            "actual": missed_hazard_rate,
            "target": _evaluation_cfg.max_missed_hazard_rate,
            "passed": missed_hazard_rate <= _evaluation_cfg.max_missed_hazard_rate,
        },
        {
            "id": "avg_review_coverage",
            "label": "Average review coverage",
            "actual": avg_review_coverage,
            "target": _evaluation_cfg.min_avg_review_coverage,
            "passed": avg_review_coverage >= _evaluation_cfg.min_avg_review_coverage,
        },
        {
            "id": "seed_corpus_progress",
            "label": "Seed eval corpus",
            "actual": seed_fixture_count,
            "target": _evaluation_cfg.target_corpus_size,
            "passed": seed_fixture_count >= _evaluation_cfg.target_corpus_size,
        },
    ]
    ready_for_beta = all(bool(gate["passed"]) for gate in gates[:5])
    return {
        "ready_for_beta": ready_for_beta,
        "summary": (
            "Release gates passed for broader beta."
            if ready_for_beta
            else "Release gates are still open. Keep growing the eval corpus and review coverage."
        ),
        "gates": gates,
    }


def _aggregate_product_metrics() -> dict[str, Any]:
    """Aggregate product-loop metrics used for beta operations."""
    jobs = list(_upload_jobs.values())
    terminal_jobs = [job for job in jobs if str(job.get("status")) in {"complete", "error"}]
    completed_jobs = [job for job in jobs if str(job.get("status")) == "complete"]
    success_rate = round(len(completed_jobs) / len(terminal_jobs), 2) if terminal_jobs else 0.0
    rescan_recommended = sum(
        1 for job in completed_jobs if bool((job.get("summary") or {}).get("rescan_recommended"))
    )
    useful_events = 0
    total_feedback_events = 0
    total_duration_seconds = 0.0
    completed_with_duration = 0
    room_label_counts: dict[str, int] = {}

    for job in completed_jobs:
        feedback = dict(job.get("feedback_summary") or {})
        useful_events += int(feedback.get("useful", 0) or 0)
        total_feedback_events += sum(int(feedback.get(key, 0) or 0) for key in feedback)
        room_label = _normalize_room_label(
            job.get("room_label") or (job.get("summary") or {}).get("room_label")
        )
        if room_label:
            room_label_counts[room_label] = room_label_counts.get(room_label, 0) + 1
        started_at = job.get("started_at")
        completed_at = job.get("completed_at")
        if started_at and completed_at:
            try:
                start_dt = datetime.fromisoformat(str(started_at))
                done_dt = datetime.fromisoformat(str(completed_at))
                total_duration_seconds += max(0.0, (done_dt - start_dt).total_seconds())
                completed_with_duration += 1
            except ValueError:
                pass

    usefulness_rate = (
        round(useful_events / total_feedback_events, 2) if total_feedback_events else 0.0
    )
    avg_report_seconds = (
        round(total_duration_seconds / completed_with_duration, 1)
        if completed_with_duration
        else 0.0
    )

    return {
        "terminal_jobs": len(terminal_jobs),
        "completed_jobs": len(completed_jobs),
        "repeat_scan_rooms": sum(1 for count in room_label_counts.values() if count > 1),
        "labeled_rooms": len(room_label_counts),
        "upload_success_rate": success_rate,
        "rescan_recommended_rate": (
            round(rescan_recommended / len(completed_jobs), 2) if completed_jobs else 0.0
        ),
        "report_usefulness_rate": usefulness_rate,
        "avg_report_seconds": avg_report_seconds,
    }


def _provider_runtime_summary() -> dict[str, Any]:
    """Summarize the active VLM routing strategy for operator diagnostics."""
    vlm = load_config().vlm
    chain = [vlm.provider]
    if vlm.fallback_provider and vlm.fallback_provider != vlm.provider:
        chain.append(vlm.fallback_provider)
    return {
        "primary_provider": vlm.provider,
        "fallback_provider": vlm.fallback_provider,
        "provider_chain": chain,
        "routing_mode": "fallback" if len(chain) > 1 else "single",
        "primary_model": vlm.model_name
        if vlm.provider == "ollama"
        else (vlm.claude_model if vlm.provider == "claude" else vlm.openai_model),
    }


def _build_evaluation_summary(job: dict[str, Any]) -> dict[str, Any]:
    """Summarize report review coverage and correction signals."""
    risks = list(job.get("risks") or [])
    events = list(job.get("finding_feedback") or [])
    human_evaluation = dict(job.get("human_evaluation") or {})
    event_counts = _feedback_counts(events)
    total_findings = len(risks)
    reviewed_findings = 0
    disputed_findings = 0
    helpful_findings = 0
    high_priority_pending = 0

    for risk in risks:
        counts = dict(risk.get("feedback_summary") or {})
        useful = int(counts.get("useful", 0) or 0)
        wrong = int(counts.get("wrong", 0) or 0)
        duplicate = int(counts.get("duplicate", 0) or 0)
        total = useful + wrong + duplicate
        if total > 0:
            reviewed_findings += 1
        if wrong > 0 or duplicate > 0:
            disputed_findings += 1
        if useful > 0:
            helpful_findings += 1
        if total == 0 and str(risk.get("severity", "")).lower() in {"critical", "high"}:
            high_priority_pending += 1

    pending_findings = max(0, total_findings - reviewed_findings)
    review_coverage = round(reviewed_findings / total_findings, 2) if total_findings else 0.0
    missed_hazard_count = len(human_evaluation.get("missed_hazards") or [])
    benchmark = dict(human_evaluation.get("benchmark") or {})
    benchmark_label = benchmark.get("label") or human_evaluation.get("benchmark_label")
    benchmark_match = benchmark.get("matched")
    precision_proxy = (
        round(event_counts["useful"] / max(1, sum(event_counts.values())), 2)
        if sum(event_counts.values()) > 0
        else 0.0
    )
    recall_proxy = (
        round(total_findings / (total_findings + missed_hazard_count), 2)
        if total_findings or missed_hazard_count
        else 0.0
    )

    if total_findings == 0:
        summary = "No findings to review yet."
    elif reviewed_findings == 0:
        summary = (
            f"No findings have feedback yet. {high_priority_pending} higher-priority"
            " finding(s) still need human review."
        )
    else:
        summary = (
            f"{reviewed_findings} of {total_findings} finding"
            f"{'' if total_findings == 1 else 's'} reviewed."
            f" {disputed_findings} marked wrong or duplicate so far."
        )

    human_status = str(human_evaluation.get("status", "")).strip() or None
    if human_status == "missed_hazard":
        summary += f" Human review flagged {missed_hazard_count} missed hazard(s)."
    elif human_status == "needs_review":
        summary += " Human review says the report still needs follow-up."
    elif human_status == "confirmed":
        summary += " Human review marked the current report as directionally sound."
    if benchmark_label:
        summary += (
            f" Benchmark {benchmark_label} " f"{'matched' if benchmark_match else 'did not match'}."
        )

    return {
        "total_findings": total_findings,
        "reviewed_findings": reviewed_findings,
        "pending_findings": pending_findings,
        "review_coverage": review_coverage,
        "useful_events": event_counts["useful"],
        "wrong_events": event_counts["wrong"],
        "duplicate_events": event_counts["duplicate"],
        "helpful_findings": helpful_findings,
        "disputed_findings": disputed_findings,
        "high_priority_pending": high_priority_pending,
        "needs_review": high_priority_pending > 0
        or pending_findings > 0
        or human_status in {"needs_review", "missed_hazard"},
        "human_status": human_status,
        "missed_hazard_count": missed_hazard_count,
        "benchmark_label": benchmark_label,
        "benchmark_match": benchmark_match,
        "benchmark_summary": benchmark,
        "precision_proxy": precision_proxy,
        "recall_proxy": recall_proxy,
        "summary": summary,
    }


def _iso_sort_key(job: dict[str, Any]) -> str:
    """Return a stable descending sort key for job recency."""
    return str(job.get("completed_at") or job.get("queued_at") or job.get("job_id") or "")


def _normalize_room_label(value: str | None) -> str | None:
    """Normalize a user-facing room label for storage and comparison."""
    label = " ".join(str(value or "").strip().split())
    if not label:
        return None
    return label


def _share_path_for_job(job_id: str) -> str:
    """Return the frontend deep link for one upload report."""
    return f"/app?view=report&job={job_id}"


def _share_summary(job: dict[str, Any]) -> str:
    """Create a short share-safe summary for one completed report."""
    summary = dict(job.get("summary") or {})
    audience_label = str(
        summary.get("audience_label") or audience_mode_label(job.get("audience_mode"))
    )
    room_label = _normalize_room_label(job.get("room_label") or summary.get("room_label"))
    headline = str(summary.get("headline") or "ATLAS-0 room scan")
    room_score = summary.get("room_score")
    score_text = f" · {room_score}/100 room score" if isinstance(room_score, int | float) else ""
    room_text = f"{room_label} · " if room_label else ""
    return f"{room_text}{audience_label} · {headline}{score_text}"


def _room_score_payload(job: dict[str, Any]) -> dict[str, Any] | None:
    """Compute a lightweight room safety score for repeat-use comparisons."""
    if str(job.get("status")) != "complete":
        return None

    summary = dict(job.get("summary") or {})
    scan_quality = dict(job.get("scan_quality") or {})
    risks = list(job.get("risks") or [])
    weights = [30, 18, 10, 6, 4]
    hazard_penalty = 0.0
    for index, risk in enumerate(
        sorted(
            risks,
            key=lambda item: float(item.get("priority_score", item.get("risk_score", 0.0))),
            reverse=True,
        )[: len(weights)]
    ):
        hazard_penalty += weights[index] * float(
            risk.get("priority_score", risk.get("risk_score", 0.0)) or 0.0
        )

    quality_penalty = 0.0
    status = str(scan_quality.get("status", "unknown")).lower()
    if status == "fair":
        quality_penalty += 6.0
    elif status == "poor":
        quality_penalty += 14.0
    if bool(scan_quality.get("rescan_recommended")):
        quality_penalty += 8.0
    if str(summary.get("analysis_outcome", "accepted")).lower() == "rejected":
        quality_penalty += 20.0

    score = max(0, min(100, int(round(100.0 - min(78.0, hazard_penalty) - quality_penalty))))
    if score >= 82:
        band = "safer now"
        summary_text = (
            "Lower apparent risk in the current scan, though this is still a screening result."
        )
    elif score >= 65:
        band = "needs fixes"
        summary_text = (
            "A few concentrated risks still deserve follow-up before calling the room calm."
        )
    else:
        band = "high attention"
        summary_text = (
            "ATLAS-0 sees enough concentrated risk here that a follow-up pass"
            " should start with the top actions."
        )

    return {
        "room_score": score,
        "room_score_band": band,
        "room_score_summary": summary_text,
    }


def _room_history(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Return recent completed scans for the same labeled room."""
    room_label = _normalize_room_label(
        job.get("room_label") or (job.get("summary") or {}).get("room_label")
    )
    if room_label is None:
        return []
    audience_mode = normalize_audience_mode(job.get("audience_mode"))

    siblings = [
        candidate
        for candidate in _upload_jobs.values()
        if str(candidate.get("status")) == "complete"
        and candidate.get("job_id") != job.get("job_id")
        and normalize_audience_mode(candidate.get("audience_mode")) == audience_mode
        and _normalize_room_label(
            candidate.get("room_label") or (candidate.get("summary") or {}).get("room_label")
        )
        == room_label
    ]
    siblings.sort(key=_iso_sort_key, reverse=True)
    history: list[dict[str, Any]] = []
    for candidate in siblings[:4]:
        candidate_summary = dict(candidate.get("summary") or {})
        score_payload = _room_score_payload(candidate)
        if score_payload:
            candidate_summary.update(score_payload)
        history.append(
            {
                "job_id": candidate.get("job_id"),
                "filename": candidate.get("filename"),
                "completed_at": candidate.get("completed_at"),
                "room_score": candidate_summary.get("room_score"),
                "hazard_count": candidate_summary.get("hazard_count"),
                "top_severity": candidate_summary.get("top_severity"),
                "audience_mode": normalize_audience_mode(candidate.get("audience_mode")),
            }
        )
    return history


def _room_comparison(job: dict[str, Any]) -> dict[str, Any] | None:
    """Compare the current room scan to the most recent previous scan."""
    history = _room_history(job)
    if not history:
        return None

    current_summary = dict(job.get("summary") or {})
    previous = history[0]
    current_score = current_summary.get("room_score")
    previous_score = previous.get("room_score")
    if current_score is None or previous_score is None:
        return None

    delta = int(current_score) - int(previous_score)
    hazard_delta = int(current_summary.get("hazard_count", 0) or 0) - int(
        previous.get("hazard_count", 0) or 0
    )
    trend = "improved" if delta > 0 else "worse" if delta < 0 else "flat"
    return {
        "previous_job_id": previous.get("job_id"),
        "previous_filename": previous.get("filename"),
        "previous_completed_at": previous.get("completed_at"),
        "previous_room_score": previous_score,
        "current_room_score": current_score,
        "score_delta": delta,
        "hazard_delta": hazard_delta,
        "trend": trend,
        "summary": (
            "This room looks safer than the last saved scan."
            if trend == "improved"
            else "This room looks riskier than the last saved scan."
            if trend == "worse"
            else "This room looks broadly similar to the last saved scan."
        ),
    }


def _ensure_job_derived_fields(job: dict[str, Any]) -> dict[str, Any]:
    """Populate lightweight derived fields for older or manually inserted jobs."""
    if job.get("risks") is not None and not isinstance(job.get("evaluation_summary"), dict):
        job["evaluation_summary"] = _build_evaluation_summary(job)
    summary = dict(job.get("summary") or {})
    audience_mode = normalize_audience_mode(job.get("audience_mode"))
    job["audience_mode"] = audience_mode
    job["share_url"] = _share_path_for_job(str(job.get("job_id", "")))
    summary["audience_mode"] = audience_mode
    summary["audience_label"] = audience_mode_label(audience_mode)
    room_label = _normalize_room_label(job.get("room_label") or summary.get("room_label"))
    if room_label:
        job["room_label"] = room_label
        summary["room_label"] = room_label
    if summary:
        score_payload = _room_score_payload(job)
        if score_payload:
            summary.update(score_payload)
        job["summary"] = summary
        job["room_history"] = _room_history(job)
        job["room_comparison"] = _room_comparison(job)
        job["weekend_fix_list"] = build_weekend_fix_list(
            list(job.get("risks") or []),
            audience_mode=audience_mode,
        )
        job["room_wins"] = build_room_wins(
            list(job.get("risks") or []),
            dict(job.get("scan_quality") or {}),
            comparison_summary=job.get("room_comparison"),
            audience_mode=audience_mode,
        )
        summary["share_summary"] = _share_summary(job)
    return job


for _job in _upload_jobs.values():
    _ensure_job_derived_fields(_job)


def _finding_key(hazard: dict[str, Any]) -> str:
    """Build a stable identifier for one report finding."""
    return (
        f"{hazard.get('object_id') or hazard.get('object_label') or 'finding'}:"
        f"{hazard.get('hazard_code') or 'unknown'}"
    )


def _request_host(request: Request) -> str:
    """Return the peer host string for one request."""
    return str(request.client.host if request.client else "")


def _is_loopback_request(request: Request) -> bool:
    """Return True when the request originates from loopback/testclient."""
    host = _request_host(request)
    if host in {"localhost", "testclient"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _extract_access_token(request: Request) -> str | None:
    """Extract an API token from Authorization or X-Atlas-Key headers."""
    authorization = request.headers.get("authorization", "").strip()
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        return token or None
    query_token = request.query_params.get("access_token", "").strip()
    if query_token:
        return query_token
    header_token = request.headers.get("x-atlas-key", "").strip()
    return header_token or None


def _require_private_access(request: Request) -> None:
    """Enforce private-beta access for upload/report endpoints."""
    configured_token = _api_cfg.access_token
    if configured_token:
        provided_token = _extract_access_token(request)
        if not provided_token or not hmac.compare_digest(provided_token, configured_token):
            raise HTTPException(status_code=401, detail="Missing or invalid Atlas API token.")
        return

    if _api_cfg.allow_unauthenticated_loopback and _is_loopback_request(request):
        return

    raise HTTPException(
        status_code=403,
        detail=(
            "Atlas-0 upload/report access is restricted to loopback unless"
            " api.access_token is configured."
        ),
    )


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def _upload_queue() -> asyncio.Queue[str | None] | None:
    """Return the background upload queue when workers are running."""
    queue = _state.get("upload_queue")
    return queue if isinstance(queue, asyncio.Queue) else None


def _upload_queue_ids() -> set[str]:
    """Return the set of queued job IDs."""
    queued = _state.setdefault("upload_queue_ids", set())
    return queued if isinstance(queued, set) else set()


def _upload_cancelled_jobs() -> set[str]:
    """Return the set of deleted jobs that workers should ignore."""
    cancelled = _state.setdefault("upload_cancelled_jobs", set())
    return cancelled if isinstance(cancelled, set) else set()


def _is_upload_cancelled(job_id: str) -> bool:
    """Return True when one job has been deleted/cancelled."""
    return job_id in _upload_cancelled_jobs()


def _job_status_counts() -> dict[str, int]:
    """Aggregate upload-job status counts for operator diagnostics."""
    counts = {"queued": 0, "processing": 0, "complete": 0, "error": 0}
    for job in _upload_jobs.values():
        status = str(job.get("status", "")).lower()
        if status in counts:
            counts[status] += 1
    return counts


def _operator_access_descriptor() -> dict[str, Any]:
    """Return the effective access policy for upload/report features."""
    requires_token = bool(_api_cfg.access_token)
    mode = (
        "token"
        if requires_token
        else "loopback"
        if _api_cfg.allow_unauthenticated_loopback
        else "restricted"
    )
    return {
        "requires_token": requires_token,
        "allow_unauthenticated_loopback": _api_cfg.allow_unauthenticated_loopback,
        "enable_job_listing": _api_cfg.enable_job_listing,
        "mode": mode,
    }


def _public_privacy_descriptor() -> dict[str, Any]:
    """Return user-visible privacy defaults for upload/report flows."""
    details = [
        f"Uploads and artifacts are retained for up to {_upload_cfg.retention_days} day(s).",
        (
            "Original uploads are kept."
            if _upload_cfg.save_original_uploads
            else "Original uploads are not persisted by default after processing."
        ),
        "Text-heavy crops are blurred before model analysis."
        if _upload_cfg.redact_text_heavy_regions
        else "Text-heavy crop redaction is disabled in this environment.",
        "You can delete a scan and its artifacts from the report view at any time.",
    ]
    return {
        "retention_days": _upload_cfg.retention_days,
        "save_original_uploads": _upload_cfg.save_original_uploads,
        "delete_supported": True,
        "text_redaction_enabled": _upload_cfg.redact_text_heavy_regions,
        "summary": (
            "ATLAS-0 keeps upload artifacts on a time-limited retention window for reports,"
            " review, and debugging, and it exposes delete controls in the product."
        ),
        "details": details,
    }


async def _start_upload_workers() -> None:
    """Start the persistent upload queue and worker tasks once."""
    if _upload_queue() is not None:
        return

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    _state["upload_queue"] = queue
    _state["upload_queue_ids"] = set()
    _state["upload_cancelled_jobs"] = set()
    _state["upload_worker_tasks"] = [
        asyncio.create_task(_upload_worker_loop(index))
        for index in range(_upload_cfg.max_concurrent_jobs)
    ]


async def _stop_upload_workers() -> None:
    """Stop background upload workers and clear queue state."""
    queue = _upload_queue()
    tasks = _state.get("upload_worker_tasks")
    if queue is None or not isinstance(tasks, list):
        return

    for _ in tasks:
        await queue.put(None)
    await asyncio.gather(*tasks, return_exceptions=True)

    _state.pop("upload_worker_tasks", None)
    _state.pop("upload_queue", None)
    _state.pop("upload_queue_ids", None)
    _state.pop("upload_cancelled_jobs", None)


async def _enqueue_upload_job(job_id: str) -> None:
    """Queue one upload job for background processing exactly once."""
    await _start_upload_workers()
    queue = _upload_queue()
    if queue is None:
        return

    queued_ids = _upload_queue_ids()
    if job_id in queued_ids:
        return

    queued_ids.add(job_id)
    await queue.put(job_id)


async def _resume_pending_upload_jobs() -> None:
    """Requeue persisted jobs that were interrupted before finishing."""
    for job_id, job in _upload_jobs.items():
        status = str(job.get("status", "")).lower()
        if status not in {"queued", "processing"}:
            continue

        if not _upload_store.has_job_input(job_id):
            _update_job(
                job,
                status="error",
                stage="complete",
                progress=1.0,
                error=(
                    "Queued upload could not be resumed because its source file is no longer"
                    " available on disk."
                ),
                completed_at=_utc_now_iso(),
            )
            continue

        _update_job(
            job,
            status="queued",
            stage="upload",
            progress=min(float(job.get("progress") or 0.0), 0.1),
            error=None,
            queued_at=_utc_now_iso(),
        )
        await _enqueue_upload_job(job_id)


async def _upload_worker_loop(worker_index: int) -> None:
    """Worker task that pulls queued upload jobs from the persistent queue."""
    queue = _upload_queue()
    if queue is None:
        return

    while True:
        job_id = await queue.get()
        if job_id is None:
            queue.task_done()
            break

        _upload_queue_ids().discard(job_id)
        try:
            await _process_upload(job_id)
        except Exception as exc:  # pragma: no cover - safety net for worker loop
            logger.exception(
                "upload_worker_failed",
                worker_index=worker_index,
                job_id=job_id,
                error=str(exc),
            )
        finally:
            queue.task_done()


def _active_upload_job_count() -> int:
    """Return the number of queued or processing upload jobs."""
    return sum(
        1 for job in _upload_jobs.values() if str(job.get("status")) in {"queued", "processing"}
    )


class UploadJobStatus(BaseModel):
    """Status of a media upload and analysis job."""

    job_id: str
    filename: str
    room_label: str | None = None
    audience_mode: str | None = None
    status: str  # queued | processing | complete | error
    stage: str  # upload | ingest | vlm | risk | complete
    progress: float
    objects: list[dict[str, Any]] | None = None
    risks: list[dict[str, Any]] | None = None
    fix_first: list[dict[str, Any]] | None = None
    weekend_fix_list: list[dict[str, Any]] | None = None
    summary: dict[str, Any] | None = None
    recommendations: list[dict[str, Any]] | None = None
    evidence_frames: list[dict[str, Any]] | None = None
    scan_quality: dict[str, Any] | None = None
    trust_notes: list[str] | None = None
    scene_source: str | None = None
    finding_feedback: list[dict[str, Any]] | None = None
    feedback_summary: dict[str, int] | None = None
    evaluation_summary: dict[str, Any] | None = None
    human_evaluation: dict[str, Any] | None = None
    room_history: list[dict[str, Any]] | None = None
    room_comparison: dict[str, Any] | None = None
    room_wins: list[dict[str, Any]] | None = None
    report_url: str | None = None
    share_url: str | None = None
    error: str | None = None
    artifacts: dict[str, Any] | None = None
    attempt_count: int = 0
    queued_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class FindingFeedbackRequest(BaseModel):
    """One user feedback event for a specific report finding."""

    hazard_code: str
    verdict: str
    object_id: str | None = None
    note: str | None = None


class JobEvaluationRequest(BaseModel):
    """One human review verdict for a completed report."""

    status: str
    benchmark_label: str | None = None
    missed_hazards: list[str] | None = None
    note: str | None = None


async def _get_or_init_upload_vlm() -> VLMEngine:
    """Return a cached, initialised :class:`VLMEngine` for upload analysis."""
    if "upload_vlm" not in _state:
        engine = VLMEngine(_build_runtime_vlm_config())
        await engine.initialize()
        _state["upload_vlm"] = engine
    return _state["upload_vlm"]  # type: ignore[return-value]


async def _label_upload_region(content: bytes, region_hint: str) -> SemanticLabel:
    """Label one upload region, falling back to image heuristics when needed."""
    engine = await _get_or_init_upload_vlm()
    label = await engine.label_region(content, region_hint=region_hint)
    if label.label in ("unknown", "") or label.confidence < 0.35:
        label = _analyze_image_heuristic(content)
    return label


def _generate_depth_pointcloud(
    content: bytes,
    n_points: int = 600,
) -> list[list[float]]:
    """Sample a pseudo-3D point cloud from a 2D image.

    Uses luminance-as-depth (bright = near) plus edge-weighted sampling so
    interesting areas (objects, contours) are denser than flat backgrounds.
    Returns a list of ``[x, y, z, r, g, b]`` rows in world-space metres,
    normalised colours 0-1.

    Coordinate convention (right-handed, Y-up):
      * Image left→right  maps to world X  ∈ [-2.5, 2.5]
      * Image top→bottom  maps to world Y  ∈ [ 2.0, 0.0]
      * Luminance depth   maps to world Z  ∈ [ 0.0, 3.0]  (bright=near=0)

    Args:
        content:  Raw image bytes (JPEG / PNG / WEBP).
        n_points: Number of points to sample.

    Returns:
        List of ``[x, y, z, r, g, b]`` rows.
    """
    try:
        import numpy as np  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        img = Image.open(io.BytesIO(content)).convert("RGB")
        # Resize to a manageable grid while keeping aspect ratio
        max_side = 160
        w, h = img.size
        scale = min(max_side / w, max_side / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img_small = img.resize((nw, nh), Image.LANCZOS)
        arr = np.array(img_small, dtype=np.float32)  # (nh, nw, 3)

        r_ch, g_ch, b_ch = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        lum = (0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch) / 255.0  # (nh, nw)

        # Edge strength → sampling weight (concentrate on object boundaries)
        gy = np.gradient(lum, axis=0)
        gx = np.gradient(lum, axis=1)
        weight = np.sqrt(gx**2 + gy**2) + 0.08  # floor so flat areas still sampled
        weight_flat = weight.flatten()
        probs = weight_flat / weight_flat.sum()

        n_actual = min(n_points, nw * nh)
        indices = np.random.choice(nw * nh, size=n_actual, replace=False, p=probs)

        rows: list[list[float]] = []
        for idx in indices:
            iy, ix = divmod(int(idx), nw)

            # Normalised image coords in [-1, 1]
            u = (ix / (nw - 1)) * 2.0 - 1.0  # left → right → world +X
            v = 1.0 - (iy / (nh - 1)) * 2.0  # top → bottom → world +Y (inverted)

            depth_01 = 1.0 - float(lum[iy, ix])  # bright=0 (near), dark=1 (far)

            x = float(u) * 2.5
            y = float(v) * 1.0 + 1.0  # world Y 0..2 centred at 1
            z = float(depth_01) * 3.0  # Z 0..3 m

            nr = float(r_ch[iy, ix]) / 255.0
            ng = float(g_ch[iy, ix]) / 255.0
            nb = float(b_ch[iy, ix]) / 255.0

            rows.append(
                [round(x, 3), round(y, 3), round(z, 3), round(nr, 3), round(ng, 3), round(nb, 3)]
            )

        return rows

    except Exception as exc:
        logger.warning("depth_pointcloud_failed", error=str(exc))
        return []


def _analyze_image_heuristic(content: bytes) -> SemanticLabel:
    """Derive a :class:`SemanticLabel` from image pixel statistics alone.

    Used when the VLM is offline.  Analyses colour, brightness, saturation,
    and texture variance to produce a plausible label + physics properties.

    Args:
        content: Raw image bytes (JPEG / PNG / WEBP).

    Returns:
        A :class:`SemanticLabel` inferred from visual statistics.
    """
    try:
        import numpy as np  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        img = Image.open(io.BytesIO(content)).convert("RGB")
        img.thumbnail((128, 128))
        arr = np.array(img, dtype=np.float32)

        mean_r = float(arr[:, :, 0].mean())
        mean_g = float(arr[:, :, 1].mean())
        mean_b = float(arr[:, :, 2].mean())
        brightness = (mean_r + mean_g + mean_b) / 3.0 / 255.0

        # Saturation (normalised range 0-1)
        mx = max(mean_r, mean_g, mean_b)
        mn = min(mean_r, mean_g, mean_b)
        saturation = (mx - mn) / (mx + 1.0)

        # Texture complexity: stddev of greyscale
        gray = arr.mean(axis=2)
        texture = float(gray.std()) / 255.0

        # Dominant-hue bucket (0-5 like HSV hue / 60)
        if mx == mn:
            hue_bucket = -1  # achromatic
        elif mx == mean_r:
            hue_bucket = 0  # red/orange
        elif mx == mean_g:
            hue_bucket = 2  # green
        else:
            hue_bucket = 4  # blue

        # ── Material + label heuristics ────────────────────────────────────
        if saturation < 0.12:
            # Near-greyscale → metal or stone
            if brightness > 0.7:
                label, material = "Polished Surface", "Metal"
                mass_kg, fragility, friction = 4.0, 0.25, 0.40
            elif brightness > 0.35:
                label, material = "Stone Object", "Stone"
                mass_kg, fragility, friction = 8.0, 0.18, 0.60
            else:
                label, material = "Dark Object", "Carbon"
                mass_kg, fragility, friction = 2.0, 0.30, 0.45
        elif hue_bucket == 0 and brightness < 0.55:
            # Warm browns → wood
            label, material = "Wooden Object", "Wood"
            mass_kg, fragility, friction = 6.0, 0.20, 0.65
        elif hue_bucket == 0 and brightness >= 0.55:
            # Warm light → ceramic / terracotta
            label, material = "Ceramic Object", "Ceramic"
            mass_kg, fragility, friction = 1.2, 0.72, 0.55
        elif hue_bucket == 2:
            # Greens → organic / plant
            label, material = "Organic Object", "Plant"
            mass_kg, fragility, friction = 0.8, 0.55, 0.40
        elif hue_bucket == 4 and brightness > 0.55:
            # Blue-bright → glass or water
            label, material = "Glass Object", "Glass"
            mass_kg, fragility, friction = 0.6, 0.90, 0.20
        elif brightness > 0.75:
            # Very bright → plastic or paper
            label, material = "Plastic Object", "Plastic"
            mass_kg, fragility, friction = 0.5, 0.40, 0.50
        else:
            label, material = "Composite Object", "Mixed"
            mass_kg, fragility, friction = 2.0, 0.50, 0.50

        # Texture raises fragility slightly (complex surface = more facets to break)
        fragility = min(1.0, fragility + texture * 0.2)

        # Confidence scales with how clear the heuristic signal is
        confidence = min(0.82, 0.45 + saturation * 0.3 + (1.0 - abs(brightness - 0.5)) * 0.15)

        logger.info(
            "image_heuristic_analysis",
            label=label,
            material=material,
            brightness=round(brightness, 2),
            saturation=round(saturation, 2),
            texture=round(texture, 2),
        )
        return SemanticLabel(
            label=label,
            material=material,
            mass_kg=round(mass_kg, 2),
            fragility=round(fragility, 2),
            friction=round(friction, 2),
            confidence=round(confidence, 2),
        )

    except Exception as exc:
        logger.warning("image_heuristic_failed", error=str(exc))
        return SemanticLabel(
            label="Unknown Object",
            material="Unknown",
            mass_kg=1.0,
            fragility=0.5,
            friction=0.5,
            confidence=0.30,
        )


def _point_cloud_centroid(points: list[list[float]]) -> tuple[float, float, float]:
    """Estimate a stable object position from an upload-derived point cloud."""
    if not points:
        return (0.0, 0.8, 1.5)

    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    zs = [float(p[2]) for p in points]
    return (
        round(sum(xs) / len(xs), 3),
        round(sum(ys) / len(ys), 3),
        round(sum(zs) / len(zs), 3),
    )


def _encode_data_url(content: bytes, mime_type: str) -> str:
    """Encode raw media bytes as a data URL for inline evidence previews."""
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _risk_severity(score: float) -> str:
    """Convert a numeric score into a user-facing severity bucket."""
    if score >= 0.78:
        return "critical"
    if score >= 0.58:
        return "high"
    if score >= 0.35:
        return "moderate"
    return "low"


def _location_label(position: tuple[float, float, float]) -> str:
    """Describe an approximate room zone from an estimated object position."""
    x, _y, z = position
    horizontal = "center"
    depth = "middle"

    if x < -0.8:
        horizontal = "left"
    elif x > 0.8:
        horizontal = "right"

    if z < 0.8:
        depth = "front"
    elif z > 1.8:
        depth = "back"

    if horizontal == "center" and depth == "middle":
        return "center area"
    return f"{depth}-{horizontal}".replace("-center", "")


def _build_trust_notes(scene_source: str) -> list[str]:
    """Return explicit honesty notes for the current scene estimation mode."""
    if scene_source == "heuristic_estimate":
        return [
            "This scan uses upload-side heuristic grounding rather than full SLAM reconstruction.",
            (
                "Object locations are approximate and should be treated as"
                " directional, not survey-grade."
            ),
            (
                "Use the evidence frames and recommendations as the primary"
                " output, not the point cloud alone."
            ),
        ]
    return ["This report is based on measured scene data."]


def _recommendation_for(
    obj: dict[str, Any],
    risk: dict[str, Any],
) -> dict[str, Any]:
    """Generate a deterministic action card for one risky object."""
    label = str(obj.get("label", "Object"))
    material = str(obj.get("material", "Unknown"))
    risk_score = float(risk.get("risk_score", 0.0))
    fragility = float(obj.get("fragility", 0.0))
    mass_kg = float(obj.get("mass_kg", 0.0))
    location = str(obj.get("location_label", "scan area"))
    label_lower = label.lower()

    action = "Reposition this item to a more stable location."
    why = f"{label} is one of the higher-risk items in the scan."
    priority = _risk_severity(risk_score)

    if (
        any(word in label_lower for word in ("glass", "vase", "cup", "mug", "bottle"))
        or fragility > 0.72
    ):
        action = "Move it farther from edges and lower it onto a wider, more stable surface."
        why = f"It appears fragile ({material}) and likely to break if tipped or dropped."
    elif any(word in label_lower for word in ("lamp", "shelf", "bookcase", "rack")):
        action = "Stabilize or anchor it, and clear the surrounding fall zone."
        why = "It looks tall or top-heavy, which raises tipping risk."
    elif mass_kg > 5.0:
        action = "Lower it and keep heavy weight below waist height if possible."
        why = "Heavier objects create more impact risk if they shift or fall."
    elif material.lower() in {"plant", "plastic", "mixed"}:
        action = "Reduce clutter around it and move it away from walk paths or edges."
        why = "Its current placement appears more risky than the object itself."

    return {
        "title": f"Stabilize {label}",
        "priority": priority,
        "location": location,
        "action": action,
        "why": why,
    }


def _build_recommendations(
    objects: list[dict[str, Any]],
    risks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build ordered recommendation cards from the current risk list."""
    objects_by_label = {str(obj.get("label", "")).lower(): obj for obj in objects}
    recommendations: list[dict[str, Any]] = []

    ranked_risks = sorted(
        risks,
        key=lambda entry: float(entry.get("risk_score", 0.0)),
        reverse=True,
    )

    for risk in ranked_risks[:5]:
        key = str(risk.get("object_label", "")).lower()
        obj = objects_by_label.get(key)
        if obj is None:
            continue
        recommendations.append(_recommendation_for(obj, risk))

    return recommendations


def _build_summary(
    filename: str,
    objects: list[dict[str, Any]],
    risks: list[dict[str, Any]],
    scene_source: str,
) -> dict[str, Any]:
    """Build a compact report summary for the frontend."""
    top_risk = max(risks, key=lambda entry: float(entry.get("risk_score", 0.0)), default=None)
    return {
        "filename": filename,
        "object_count": len(objects),
        "hazard_count": len(risks),
        "top_severity": (
            _risk_severity(float(top_risk.get("risk_score", 0.0))) if top_risk else "none"
        ),
        "top_hazard_label": top_risk.get("object_label") if top_risk else None,
        "scene_source": scene_source,
        "confidence_label": (
            "Approximate spatial grounding"
            if scene_source == "heuristic_estimate"
            else "Measured scene grounding"
        ),
    }


def _build_pdf_report(job: dict[str, Any]) -> bytes:
    """Generate a compact PDF report without extra runtime dependencies."""
    summary = job.get("summary") or {}
    risks = job.get("risks") or []
    fix_first = job.get("fix_first") or []
    weekend_fix_list = job.get("weekend_fix_list") or []
    recommendations = job.get("recommendations") or []
    scan_quality = job.get("scan_quality") or {}
    trust_notes = job.get("trust_notes") or []
    evaluation_summary = job.get("evaluation_summary") or {}
    room_wins = job.get("room_wins") or []

    lines = [
        "ATLAS-0 Room Safety Report",
        f"Scan file: {summary.get('filename', 'unknown')}",
        (
            "Audience mode: "
            f"{summary.get('audience_label', audience_mode_label(job.get('audience_mode')))}"
        ),
        f"Hazards found: {summary.get('hazard_count', 0)}",
        f"Objects detected: {summary.get('object_count', 0)}",
        f"Scene source: {summary.get('scene_source', 'unknown')}",
        f"Report posture: {summary.get('report_posture', 'screening')}",
        (
            "Scan quality: "
            f"{str(scan_quality.get('status', 'unknown')).upper()} "
            f"({int(float(scan_quality.get('score', 0.0)) * 100)} / 100)"
        ),
        f"Coverage: {summary.get('coverage_label', 'Unknown')}",
        summary.get(
            "screening_statement",
            (
                "This report flags likely hazards from the uploaded scan."
                " It does not certify that the room is safe."
            ),
        ),
        "",
        "Fix first:",
    ]

    if fix_first:
        for action in fix_first[:3]:
            lines.append(f"- {action.get('title', 'Action')}: {action.get('action', '')}")
    else:
        lines.append("- No high-priority actions were generated.")

    lines.extend(["", "Weekend fix list:"])
    if weekend_fix_list:
        for item in weekend_fix_list[:3]:
            lines.append(
                f"- {item.get('title', 'Weekend fix')} ({item.get('effort', '20-30 minutes')}): "
                f"{item.get('task', '')}"
            )
    else:
        lines.append("- No weekend fix list was generated.")

    lines.extend(
        [
            "",
            "Top hazards:",
        ]
    )

    if risks:
        for risk in risks[:5]:
            lines.append(
                f"- {risk.get('hazard_title', risk.get('object_label', 'Object'))} "
                f"({str(risk.get('severity', 'low')).upper()}): "
                f"{risk.get('what', risk.get('description', ''))}"
            )
    else:
        lines.append(
            "- No high-confidence hazards were detected in this scan."
            " This is not a safety clearance."
        )

    lines.extend(["", "Recommended actions:"])
    if recommendations:
        for rec in recommendations[:5]:
            lines.append(f"- {rec.get('title', 'Action')}: {rec.get('action', '')}")
    else:
        lines.append("- No follow-up actions were generated.")

    if scan_quality.get("warnings"):
        lines.extend(["", "Scan quality warnings:"])
        for warning in scan_quality["warnings"][:3]:
            lines.append(f"- {warning}")

    if trust_notes:
        lines.extend(["", "Trust notes:"])
        for note in trust_notes[:3]:
            lines.append(f"- {note}")

    if room_wins:
        lines.extend(["", "Positive signs in this scan:"])
        for win in room_wins[:3]:
            lines.append(f"- {win.get('title', 'Positive sign')}: {win.get('detail', '')}")

    if evaluation_summary:
        lines.extend(
            [
                "",
                "Review loop:",
                f"- {evaluation_summary.get('summary', 'No review summary available.')}",
                (
                    f"- Precision proxy: "
                    f"{int(float(evaluation_summary.get('precision_proxy', 0.0)) * 100)} / 100"
                ),
                (
                    f"- Recall proxy: "
                    f"{int(float(evaluation_summary.get('recall_proxy', 0.0)) * 100)} / 100"
                ),
            ]
        )

    max_lines = 34
    visible_lines = lines[:max_lines]

    def _pdf_escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    y = 790
    content_lines: list[str] = []
    for index, line in enumerate(visible_lines):
        font_size = 18 if index == 0 else 11
        content_lines.append(f"BT /F1 {font_size} Tf 48 {y} Td ({_pdf_escape(line)}) Tj ET")
        y -= 24 if index == 0 else 17

    stream = "\n".join(content_lines).encode("latin-1", "replace")
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        (
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
        ),
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        (
            f"5 0 obj << /Length {len(stream)} >> stream\n".encode("ascii")
            + stream
            + b"\nendstream endobj\n"
        ),
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf)


async def _process_video_upload(job: dict[str, Any], content: bytes) -> None:
    """Extract frames from a video and run the full analysis pipeline on each.

    Frames are sampled evenly across the video duration. Each frame is analyzed
    by the VLM and contributes to the point cloud. Results are merged: objects
    are deduplicated by label, risks take the max score across frames.

    Args:
        job: The in-memory job dict to update with status and results.
        content: Raw video file bytes (MP4, MOV, WEBM, AVI).
    """
    from atlas.utils.video import extract_frames, is_video_available

    if not is_video_available():
        job.update(
            {
                "status": "error",
                "stage": "complete",
                "progress": 1.0,
                "objects": [],
                "risks": [],
                "error": (
                    "Video support requires the 'av' package. "
                    'Install it with: pip install "atlas-0[video]"'
                ),
            }
        )
        return

    job.update({"stage": "ingest", "progress": 0.1})

    frames = extract_frames(content, max_frames=8)
    if not frames:
        job.update(
            {
                "status": "error",
                "stage": "complete",
                "progress": 1.0,
                "objects": [],
                "risks": [],
                "error": "Could not extract frames from the video file. "
                "Ensure the file is a valid MP4, MOV, WEBM, or AVI.",
            }
        )
        return

    logger.info("video_upload_frames_extracted", frame_count=len(frames))
    engine = await _get_or_init_upload_vlm()
    evidence_frames: list[dict[str, Any]] = []

    # Maps label -> best SemanticLabel seen across frames (by confidence).
    best_labels: dict[str, Any] = {}
    all_points: list[list[float]] = []

    for i, frame_bytes in enumerate(frames):
        progress = 0.15 + (i / len(frames)) * 0.70
        job.update({"stage": "vlm", "progress": round(progress, 2)})

        label = await engine.label_region(frame_bytes, region_hint=f"video frame {i + 1}")
        if label.label in ("unknown", "") or label.confidence < 0.35:
            label = _analyze_image_heuristic(frame_bytes)

        if len(evidence_frames) < 4:
            evidence_frames.append(
                {
                    "caption": f"Evidence frame {i + 1}",
                    "kind": "video_frame",
                    "confidence": round(label.confidence, 2),
                    "image_url": _encode_data_url(frame_bytes, "image/jpeg"),
                }
            )

        # Contribute this frame's point cloud (offset along Z by frame index
        # so multi-frame clouds have spatial spread rather than all stacking).
        frame_points = _generate_depth_pointcloud(frame_bytes, n_points=400)
        z_offset = i * 0.4  # metres apart per frame
        for pt in frame_points:
            all_points.append([pt[0], pt[1], pt[2] + z_offset, pt[3], pt[4], pt[5]])
        frame_position = _point_cloud_centroid(
            [[pt[0], pt[1], pt[2] + z_offset] for pt in frame_points]
        )

        existing = best_labels.get(label.label)
        if existing is None:
            best_labels[label.label] = {
                "label": label.label,
                "material": label.material,
                "mass_kg": label.mass_kg,
                "fragility": label.fragility,
                "friction": label.friction,
                "confidence": label.confidence,
                "positions": [frame_position],
                "observations": 1,
            }
            continue

        existing["positions"].append(frame_position)
        existing["observations"] += 1
        if label.confidence > existing["confidence"]:
            existing.update(
                {
                    "label": label.label,
                    "material": label.material,
                    "mass_kg": label.mass_kg,
                    "fragility": label.fragility,
                    "friction": label.friction,
                    "confidence": label.confidence,
                }
            )

    job.update({"stage": "risk", "progress": 0.90})

    objects = []
    for obj_data in best_labels.values():
        position = _point_cloud_centroid(
            [[x, y, z] for x, y, z in obj_data.get("positions", [(0.0, 0.8, 1.5)])]
        )
        obj_data["position"] = [position[0], position[1], position[2]]
        obj_data["location_label"] = _location_label(position)
        objects.append(obj_data)

    risks: list[dict[str, Any]] = []
    agent = _get_agent()
    scene_source = "heuristic_estimate"
    trust_notes = _build_trust_notes(scene_source)

    for obj_data in objects:
        mass_factor = min(obj_data["mass_kg"] / 20.0, 0.3) * 0.3
        risk_score = min(1.0, obj_data["fragility"] * 0.4 + mass_factor + 0.1)
        if risk_score > 0.25:
            risks.append(
                {
                    "object_label": obj_data["label"],
                    "risk_score": round(risk_score, 3),
                    "severity": _risk_severity(risk_score),
                    "location_label": obj_data["location_label"],
                    "description": (
                        f"{obj_data['label']} — fragility {obj_data['fragility']:.2f}, "
                        f"mass {obj_data['mass_kg']:.1f} kg, approx. {obj_data['location_label']}"
                    ),
                }
            )
        # Inject into the live world model so /objects and /scene update.
        from atlas.vlm.inference import SemanticLabel

        sl = SemanticLabel(
            label=obj_data["label"],
            material=obj_data["material"],
            mass_kg=obj_data["mass_kg"],
            fragility=obj_data["fragility"],
            friction=obj_data["friction"],
            confidence=obj_data["confidence"],
        )
        position = tuple(obj_data["position"])
        await agent.ingest_from_upload(sl, position=position)

    recommendations = _build_recommendations(objects, risks)
    summary = _build_summary(job["filename"], objects, risks, scene_source)

    job.update(
        {
            "status": "complete",
            "stage": "complete",
            "progress": 1.0,
            "objects": objects,
            "risks": risks,
            "point_cloud": all_points,
            "summary": summary,
            "recommendations": recommendations,
            "evidence_frames": evidence_frames,
            "trust_notes": trust_notes,
            "scene_source": scene_source,
            "report_url": f"/reports/{job['job_id']}.pdf",
        }
    )
    logger.info(
        "video_upload_complete",
        objects=len(objects),
        risks=len(risks),
        points=len(all_points),
    )


async def _process_upload(job_id: str) -> None:
    """Worker task: load one queued upload from disk and process it."""
    job = _upload_jobs.get(job_id)
    if job is None or _is_upload_cancelled(job_id):
        _upload_cancelled_jobs().discard(job_id)
        return

    content = _upload_store.load_job_input(job_id)
    if content is None:
        _update_job(
            job,
            status="error",
            stage="complete",
            progress=1.0,
            error="Queued upload input is missing from disk.",
            completed_at=_utc_now_iso(),
        )
        return

    content_type = str(job.get("content_type") or "application/octet-stream")
    attempt_count = int(job.get("attempt_count") or 0) + 1

    try:
        _update_job(
            job,
            status="processing",
            stage="ingest",
            progress=0.15,
            attempt_count=attempt_count,
            started_at=_utc_now_iso(),
            error=None,
        )

        async with asyncio.timeout(_upload_cfg.job_timeout_seconds):
            await asyncio.sleep(0.3)

            is_image = content_type in _IMAGE_TYPES or content_type.startswith("image/")
            is_video = content_type in _VIDEO_TYPES or content_type.startswith("video/")

            if is_image:
                _update_job(job, stage="vlm", progress=0.4)
                await asyncio.sleep(0.2)
                result = await analyze_uploaded_image(
                    content,
                    filename=job["filename"],
                    content_type=content_type,
                    labeler=_label_upload_region,
                    audience_mode=str(job.get("audience_mode") or "general"),
                )
            elif is_video:
                _update_job(job, stage="vlm", progress=0.35)
                result = await analyze_uploaded_video(
                    content,
                    filename=job["filename"],
                    labeler=_label_upload_region,
                    audience_mode=str(job.get("audience_mode") or "general"),
                )
            else:
                msg = f"Unsupported file type: {content_type!r}"
                raise ValueError(msg)

        if _is_upload_cancelled(job_id) or job_id not in _upload_jobs:
            return

        _update_job(job, stage="risk", progress=0.9)

        risks = [dict(risk) for risk in result.risks]
        evidence_frames = [dict(frame) for frame in result.evidence_frames]
        evidence_artifacts: dict[str, dict[str, Any]] = {}
        for frame in evidence_frames:
            evidence_id = str(frame.get("evidence_id", ""))
            content_bytes = result.evidence_artifacts.get(evidence_id)
            if not evidence_id or content_bytes is None:
                continue
            evidence_path = _upload_store.save_evidence_image(
                job_id,
                evidence_id,
                content_bytes,
                suffix=".jpg",
            )
            frame["image_url"] = f"/jobs/{job_id}/evidence/{evidence_id}"
            pointer = _upload_store.artifact_pointer(
                job_id,
                evidence_path.relative_to(_upload_store.job_dir(job_id)),
                kind="evidence_image",
                media_type="image/jpeg",
                url=frame["image_url"],
            )
            frame["artifact"] = pointer
            evidence_artifacts[evidence_id] = pointer

        replay_descriptors, replay_payloads = build_finding_replays(
            risks,
            result.evidence_artifacts,
        )
        finding_replays: dict[str, dict[str, Any]] = {}
        finding_replays_by_key: dict[str, dict[str, Any]] = {}
        for descriptor in replay_descriptors:
            replay_id = str(descriptor.get("replay_id", ""))
            replay_bytes = replay_payloads.get(replay_id)
            if not replay_id or not replay_bytes:
                continue
            replay_path = _upload_store.save_replay_gif(job_id, replay_id, replay_bytes)
            image_url = f"/jobs/{job_id}/replays/{replay_id}"
            pointer = _upload_store.artifact_pointer(
                job_id,
                replay_path.relative_to(_upload_store.job_dir(job_id)),
                kind="finding_replay",
                media_type="image/gif",
                url=image_url,
            )
            replay = dict(descriptor)
            replay["image_url"] = image_url
            replay["artifact"] = pointer
            finding_replays[replay_id] = pointer
            finding_key = (
                f"{replay.get('object_id') or 'finding'}:"
                f"{replay.get('hazard_code') or 'unknown'}"
            )
            finding_replays_by_key[finding_key] = replay

        for risk in risks:
            replay = finding_replays_by_key.get(_finding_key(risk))
            if replay is not None:
                risk["replay"] = replay

        _update_job(
            job,
            status="complete",
            stage="complete",
            progress=1.0,
            objects=result.objects,
            risks=risks,
            point_cloud=result.point_cloud,
            fix_first=result.fix_first,
            summary=result.summary,
            recommendations=result.recommendations,
            evidence_frames=evidence_frames,
            scan_quality=result.scan_quality,
            trust_notes=result.trust_notes,
            scene_source=result.scene_source,
            finding_feedback=[],
            feedback_summary={"useful": 0, "wrong": 0, "duplicate": 0},
            human_evaluation=None,
            report_url=f"/reports/{job['job_id']}.pdf",
            completed_at=_utc_now_iso(),
        )

        pdf_bytes = _build_pdf_report(job)
        report_path = _upload_store.save_report_pdf(job_id, pdf_bytes)
        artifacts = _job_artifacts(job)
        artifacts["report_pdf"] = _upload_store.artifact_pointer(
            job_id,
            report_path.relative_to(_upload_store.job_dir(job_id)),
            kind="report_pdf",
            media_type="application/pdf",
            url=f"/reports/{job['job_id']}.pdf",
        )
        if evidence_artifacts:
            artifacts["evidence"] = evidence_artifacts
        if finding_replays:
            artifacts["finding_replays"] = finding_replays

        agent = _get_agent()
        for obj in result.objects:
            semantic_label = SemanticLabel(
                label=str(obj["label"]),
                material=str(obj["material"]),
                mass_kg=float(obj["mass_kg"]),
                fragility=float(obj["fragility"]),
                friction=float(obj["friction"]),
                confidence=float(obj["confidence"]),
            )
            position = tuple(float(v) for v in obj.get("position", [0.0, 0.8, 1.5]))
            await agent.ingest_from_upload(semantic_label, position=position)

        _upload_store.remove_job_input(job_id)
        artifacts.pop("queued_input", None)
        _save_job(job)
    except asyncio.CancelledError:
        logger.info("upload_processing_cancelled", job_id=job_id)
        raise
    except TimeoutError:
        logger.warning("upload_processing_timed_out", job_id=job_id)
        if job_id not in _upload_jobs or _is_upload_cancelled(job_id):
            return
        if attempt_count < _upload_cfg.max_job_attempts:
            _update_job(
                job,
                status="queued",
                stage="upload",
                progress=0.05,
                error="Upload analysis timed out. Retrying automatically.",
                queued_at=_utc_now_iso(),
            )
            await _enqueue_upload_job(job_id)
            return

        _update_job(
            job,
            status="error",
            stage="complete",
            progress=1.0,
            error="Upload analysis timed out.",
            completed_at=_utc_now_iso(),
        )
        _upload_store.remove_job_input(job_id)
        _set_job_artifact(job, "queued_input", None)
    except Exception as exc:
        logger.warning("upload_processing_failed", job_id=job_id, error=str(exc))
        if job_id not in _upload_jobs or _is_upload_cancelled(job_id):
            return
        if attempt_count < _upload_cfg.max_job_attempts:
            _update_job(
                job,
                status="queued",
                stage="upload",
                progress=0.05,
                error=f"Retrying after failure: {exc}",
                queued_at=_utc_now_iso(),
            )
            await _enqueue_upload_job(job_id)
            return

        _update_job(
            job,
            status="error",
            stage="complete",
            progress=1.0,
            error=str(exc),
            completed_at=_utc_now_iso(),
        )
        _upload_store.remove_job_input(job_id)
        _set_job_artifact(job, "queued_input", None)
    finally:
        _upload_cancelled_jobs().discard(job_id)


# ── Upload endpoints ──────────────────────────────────────────────────────────


def _extract_single_file_from_multipart(
    body: bytes,
    content_type: str,
) -> tuple[str, str, bytes]:
    """Extract the ``file`` form part without requiring ``python-multipart``."""
    envelope = (f"Content-Type: {content_type}\r\n" "MIME-Version: 1.0\r\n\r\n").encode() + body
    message = BytesParser(policy=default).parsebytes(envelope)

    if not message.is_multipart():
        raise ValueError("Expected multipart/form-data payload.")

    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        if part.get_param("name", header="content-disposition") != "file":
            continue

        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename() or "upload.bin"
        part_content_type = part.get_content_type() or "application/octet-stream"
        return filename, part_content_type, payload

    raise ValueError("Multipart upload is missing the 'file' field.")


async def _read_upload_request(request: Request) -> tuple[str, str, bytes]:
    """Read an upload request in raw or multipart form."""
    content_type = request.headers.get("content-type", "")
    body = await _read_request_body_limited(request, _upload_cfg.max_upload_bytes)
    if not body:
        raise HTTPException(status_code=400, detail="Upload body is empty.")

    if content_type.startswith("multipart/form-data"):
        try:
            return _extract_single_file_from_multipart(body, content_type)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    filename = request.headers.get("x-filename", "upload.bin")
    fallback_type = content_type or "application/octet-stream"
    return filename, fallback_type, body


def _request_room_label(request: Request) -> str | None:
    """Extract an optional room label supplied by the frontend."""
    label = _normalize_room_label(request.headers.get("x-room-label"))
    if label and len(label) > 80:
        raise HTTPException(status_code=400, detail="Room label must be 80 characters or fewer.")
    return label


def _request_audience_mode(request: Request) -> str:
    """Extract the upload audience mode from request headers."""
    raw_mode = request.headers.get("x-audience-mode")
    mode = normalize_audience_mode(raw_mode)
    if raw_mode and normalize_audience_mode(raw_mode) == "general":
        cleaned = str(raw_mode).strip().lower().replace("-", "_").replace(" ", "_")
        if cleaned and cleaned != "general":
            raise HTTPException(
                status_code=400,
                detail="Audience mode must be general, toddler, pet, or renter.",
            )
    return mode


async def _read_request_body_limited(request: Request, max_bytes: int) -> bytes:
    """Read one request body while enforcing a maximum byte size."""
    content_length = request.headers.get("content-length")
    if content_length:
        with contextlib.suppress(ValueError):
            if int(content_length) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds the {max_bytes} byte limit.",
                )

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds the {max_bytes} byte limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_upload_constraints(content_type: str, content: bytes) -> None:
    """Reject uploads that violate configured size/type/duration constraints."""
    if len(content) > _upload_cfg.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {_upload_cfg.max_upload_bytes} byte limit.",
        )

    is_video = content_type in _VIDEO_TYPES or content_type.startswith("video/")
    if not is_video:
        return

    metadata = probe_video_metadata(content)
    if metadata is None:
        return
    if metadata.duration_s > _upload_cfg.max_video_duration_seconds:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Video duration {metadata.duration_s:.1f}s exceeds the "
                f"{_upload_cfg.max_video_duration_seconds:.1f}s limit."
            ),
        )


@app.post("/upload", response_model=UploadJobStatus)
async def upload_media(request: Request) -> UploadJobStatus:
    """Accept an image or video file and run it through the analysis pipeline.

    Returns immediately with a ``job_id``. Poll ``GET /jobs/{job_id}`` for
    status updates as the file moves through the pipeline stages:
    ``upload → ingest → vlm → risk → complete``.

    Supported image formats: JPEG, PNG, WEBP, GIF.
    Accepts both ``multipart/form-data`` browser uploads and raw file bodies.
    """
    _require_private_access(request)

    if _active_upload_job_count() >= _upload_cfg.max_queue_depth:
        raise HTTPException(
            status_code=429,
            detail=(
                "Atlas-0 is already holding the maximum number of queued and"
                " active uploads. Please retry shortly."
            ),
        )

    filename, content_type, content = await _read_upload_request(request)
    room_label = _request_room_label(request)
    audience_mode = _request_audience_mode(request)
    _validate_upload_constraints(content_type, content)

    job_id = uuid.uuid4().hex[:8]
    now = _utc_now_iso()

    job: dict[str, Any] = {
        "job_id": job_id,
        "filename": filename,
        "room_label": room_label,
        "audience_mode": audience_mode,
        "content_type": content_type,
        "status": "queued",
        "stage": "upload",
        "progress": 0.0,
        "objects": None,
        "risks": None,
        "fix_first": None,
        "weekend_fix_list": None,
        "summary": None,
        "recommendations": None,
        "evidence_frames": None,
        "scan_quality": None,
        "trust_notes": None,
        "scene_source": None,
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "human_evaluation": None,
        "room_wins": None,
        "report_url": None,
        "share_url": _share_path_for_job(job_id),
        "error": None,
        "artifacts": {},
        "attempt_count": 0,
        "queued_at": now,
        "started_at": None,
        "completed_at": None,
    }
    _upload_jobs[job_id] = job
    _upload_store.create_job(job)
    queued_input_path = _upload_store.save_job_input(job_id, filename, content)
    _set_job_artifact(
        job,
        "queued_input",
        _upload_store.artifact_pointer(
            job_id,
            queued_input_path.relative_to(_upload_store.job_dir(job_id)),
            kind="queued_input",
            media_type=content_type,
        ),
    )
    original_upload_path = _upload_store.save_original_upload(job_id, filename, content)
    if original_upload_path is not None:
        _set_job_artifact(
            job,
            "original_upload",
            _upload_store.artifact_pointer(
                job_id,
                original_upload_path.relative_to(_upload_store.job_dir(job_id)),
                kind="original_upload",
                media_type=content_type,
            ),
        )

    _update_job(job, stage="upload", progress=0.1)
    await _enqueue_upload_job(job_id)

    logger.info("upload_accepted", job_id=job_id, filename=filename, content_type=content_type)
    return UploadJobStatus(**job)


@app.get("/jobs", response_model=list[UploadJobStatus])
def list_upload_jobs(request: Request) -> list[UploadJobStatus]:
    """List all upload jobs and their current status."""
    _require_private_access(request)
    if not _api_cfg.enable_job_listing:
        raise HTTPException(
            status_code=403,
            detail="Job listing is disabled. Query a known job ID directly instead.",
        )
    return [UploadJobStatus(**_ensure_job_derived_fields(j)) for j in _upload_jobs.values()]


@app.get("/jobs/{job_id}", response_model=UploadJobStatus)
def get_upload_job(job_id: str, request: Request) -> UploadJobStatus:
    """Return the current status of a specific upload job.

    Args:
        job_id: The 8-character hex job ID returned by ``POST /upload``.

    Raises:
        HTTPException: 404 if the job is not found.
    """
    _require_private_access(request)
    if job_id not in _upload_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return UploadJobStatus(**_ensure_job_derived_fields(_upload_jobs[job_id]))


@app.post("/jobs/{job_id}/feedback", response_model=UploadJobStatus)
def record_job_feedback(
    job_id: str,
    payload: FindingFeedbackRequest,
    request: Request,
) -> UploadJobStatus:
    """Store user feedback for one finding in a completed upload report."""
    _require_private_access(request)
    job = _upload_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    if job.get("status") != "complete":
        raise HTTPException(status_code=409, detail="Feedback is only accepted for completed jobs.")

    verdict = payload.verdict.lower().strip()
    if verdict not in {"useful", "wrong", "duplicate"}:
        raise HTTPException(
            status_code=400,
            detail="Feedback verdict must be useful, wrong, or duplicate.",
        )
    note = (payload.note or "").strip()
    if len(note) > 500:
        raise HTTPException(
            status_code=400,
            detail="Feedback note must be 500 characters or fewer.",
        )

    risks = job.get("risks") or []
    target = next(
        (
            risk
            for risk in risks
            if str(risk.get("hazard_code", "")) == payload.hazard_code
            and (payload.object_id is None or str(risk.get("object_id", "")) == payload.object_id)
        ),
        None,
    )
    if target is None:
        raise HTTPException(status_code=404, detail="Matching finding was not found in the report.")

    event = {
        "hazard_code": payload.hazard_code,
        "object_id": payload.object_id or target.get("object_id"),
        "verdict": verdict,
        "note": note or None,
        "created_at": datetime.now(UTC).isoformat(),
        "finding_key": _finding_key(target),
    }

    events = list(job.get("finding_feedback") or [])
    events.append(event)
    counts = dict(target.get("feedback_summary") or {"useful": 0, "wrong": 0, "duplicate": 0})
    counts[verdict] = counts.get(verdict, 0) + 1
    target["feedback_summary"] = counts
    target["latest_feedback"] = verdict

    _update_job(
        job,
        risks=risks,
        finding_feedback=events,
        feedback_summary=_feedback_counts(events),
    )
    return UploadJobStatus(**job)


@app.post("/jobs/{job_id}/evaluation", response_model=UploadJobStatus)
def record_job_evaluation(
    job_id: str,
    payload: JobEvaluationRequest,
    request: Request,
) -> UploadJobStatus:
    """Store one human review verdict for a completed report."""
    _require_private_access(request)
    job = _upload_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    if job.get("status") != "complete":
        raise HTTPException(
            status_code=409,
            detail="Evaluation is only accepted for completed jobs.",
        )

    status = payload.status.strip().lower()
    if status not in {"confirmed", "needs_review", "missed_hazard"}:
        raise HTTPException(
            status_code=400,
            detail="Evaluation status must be confirmed, needs_review, or missed_hazard.",
        )

    note = (payload.note or "").strip()
    if len(note) > 500:
        raise HTTPException(
            status_code=400,
            detail="Evaluation note must be 500 characters or fewer.",
        )

    missed_hazards = [item.strip() for item in (payload.missed_hazards or []) if item.strip()]
    if len(missed_hazards) > 8:
        raise HTTPException(status_code=400, detail="At most 8 missed hazards may be submitted.")

    benchmark_label = (payload.benchmark_label or "").strip() or _auto_benchmark_label(job)
    benchmark = _compare_job_to_benchmark(job, benchmark_label) if benchmark_label else None

    _update_job(
        job,
        human_evaluation={
            "status": status,
            "benchmark_label": benchmark_label,
            "benchmark": benchmark,
            "missed_hazards": missed_hazards,
            "note": note or None,
            "reviewed_at": datetime.now(UTC).isoformat(),
        },
    )
    return UploadJobStatus(**job)


@app.get("/jobs/{job_id}/evidence/{evidence_id}")
def download_evidence(job_id: str, evidence_id: str, request: Request) -> Response:
    """Download one persisted evidence crop for a completed report."""
    _require_private_access(request)
    if job_id not in _upload_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")

    artifact = _upload_store.load_evidence_image(job_id, evidence_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Evidence artifact not found.")

    content, media_type = artifact
    return Response(content=content, media_type=media_type)


@app.get("/jobs/{job_id}/replays/{replay_id}")
def download_finding_replay(job_id: str, replay_id: str, request: Request) -> Response:
    """Download one persisted finding replay for a completed report."""
    _require_private_access(request)
    if job_id not in _upload_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")

    artifact = _upload_store.load_replay_gif(job_id, replay_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Finding replay not found.")

    content, media_type = artifact
    return Response(content=content, media_type=media_type)


@app.delete("/jobs/{job_id}", status_code=204)
def delete_upload_job(job_id: str, request: Request) -> Response:
    """Delete one persisted upload job and its artifacts."""
    _require_private_access(request)
    cancelled = _upload_cancelled_jobs()
    cancelled.add(job_id)
    existed = job_id in _upload_jobs
    removed = _upload_store.delete_job(job_id)
    _upload_jobs.pop(job_id, None)
    if not existed and not removed:
        cancelled.discard(job_id)
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return Response(status_code=204)


@app.get("/reports/{job_id}.pdf")
def download_report(job_id: str, request: Request) -> Response:
    """Download a generated PDF report for a completed scan job."""
    _require_private_access(request)
    job = _upload_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    if job.get("status") != "complete":
        raise HTTPException(status_code=409, detail="Report is not ready yet.")

    pdf_bytes = _upload_store.load_report_pdf(job_id)
    if pdf_bytes is None:
        pdf_bytes = _build_pdf_report(job)
        report_path = _upload_store.save_report_pdf(job_id, pdf_bytes)
        _set_job_artifact(
            job,
            "report_pdf",
            _upload_store.artifact_pointer(
                job_id,
                report_path.relative_to(_upload_store.job_dir(job_id)),
                kind="report_pdf",
                media_type="application/pdf",
                url=f"/reports/{job_id}.pdf",
            ),
        )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="atlas0-report-{job_id}.pdf"'},
    )


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
]
