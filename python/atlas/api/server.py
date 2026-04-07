"""FastAPI server exposing Atlas-0 spatial queries and AR overlay data.

Provides endpoints for:
- Querying the semantic 3D map with natural language.
- Listing all labeled objects and their physical properties.
- Returning the full current scene state.
- Streaming risk assessments via WebSocket.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import pathlib
import uuid
from typing import Annotated, Any

import structlog
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
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
from atlas.vlm.inference import SemanticLabel, VLMConfig, VLMEngine
from atlas.world_model.agent import RiskEntry, WorldModelAgent, WorldModelConfig
from atlas.world_model.query_parser import QueryParser, QueryType
from atlas.world_model.relationships import RelationType, SemanticObject
from atlas.world_model.risk_aggregator import RiskAggregator

logger = structlog.get_logger(__name__)


@contextlib.asynccontextmanager
async def _lifespan(application: FastAPI):  # type: ignore[type-arg]
    """Start the world-model agent on server startup; stop it on shutdown."""
    agent = _get_agent()
    await agent.start()
    logger.info("world_model_agent_started_via_lifespan")
    yield
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


def _get_agent() -> WorldModelAgent:
    """FastAPI dependency: lazily create and return the singleton agent."""
    if "agent" not in _state:
        _state["agent"] = WorldModelAgent(config=WorldModelConfig())
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
    objects = agent.get_objects_sync()
    risks = await agent.get_risks()
    stale = agent.risks_stale_seconds
    # -1.0 means "no assessment has completed yet" (JSON can't encode inf).
    stale_json = stale if stale != float("inf") else -1.0
    # Update Prometheus gauges on every health poll.
    object_count.set(len(objects))
    risk_count.set(len(risks))
    slam_active.set(0)
    assessment_age_seconds.set(stale_json)
    return HealthResponse(
        status="ok",
        slam_active=False,  # Phase 1/2 integration — Part 8
        vlm_active=agent.vlm_active,
        frame_count=0,
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
    objects = agent.get_objects_sync()
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
    objects = agent.get_objects_sync()
    risks = await agent.get_risks()
    return [_to_object_info(obj, risks) for obj in objects]


@app.get("/scene", response_model=SceneState)
async def get_scene(
    agent: Annotated[WorldModelAgent, Depends(_get_agent)],
) -> SceneState:
    """Return the full current scene state including all objects and risks.

    Useful for initialising an AR frontend or running a one-shot scene dump.
    """
    objects = agent.get_objects_sync()
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
_upload_jobs: dict[str, dict[str, Any]] = {}


class UploadJobStatus(BaseModel):
    """Status of a media upload and analysis job."""

    job_id: str
    filename: str
    status: str  # queued | processing | complete | error
    stage: str  # upload | ingest | vlm | risk | complete
    progress: float
    objects: list[dict[str, Any]] | None = None
    risks: list[dict[str, Any]] | None = None
    error: str | None = None


async def _get_or_init_upload_vlm() -> VLMEngine:
    """Return a cached, initialised :class:`VLMEngine` for upload analysis."""
    if "upload_vlm" not in _state:
        engine = VLMEngine(VLMConfig())
        await engine.initialize()
        _state["upload_vlm"] = engine
    return _state["upload_vlm"]  # type: ignore[return-value]


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


# Spread successive uploads across a plausible room layout so objects don't
# all land at the origin.
_UPLOAD_POSITIONS: list[tuple[float, float, float]] = [
    (0.5, 0.82, 0.3),  # table surface, right
    (-0.5, 0.82, -0.3),  # table surface, left
    (1.4, 0.82, 0.1),  # table surface, far right
    (-2.2, 1.10, 0.8),  # floor lamp height
    (0.0, 0.42, 0.0),  # table centre (lower)
    (2.0, 0.82, 1.0),  # shelf
    (-1.2, 0.82, -0.8),  # sideboard
    (0.8, 1.50, 0.5),  # upper shelf
]
_upload_pos_index: int = 0


def _next_upload_position() -> tuple[float, float, float]:
    global _upload_pos_index
    pos = _UPLOAD_POSITIONS[_upload_pos_index % len(_UPLOAD_POSITIONS)]
    _upload_pos_index += 1
    return pos


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

    # Maps label -> best SemanticLabel seen across frames (by confidence).
    best_labels: dict[str, Any] = {}
    all_points: list[list[float]] = []

    for i, frame_bytes in enumerate(frames):
        progress = 0.15 + (i / len(frames)) * 0.70
        job.update({"stage": "vlm", "progress": round(progress, 2)})

        label = await engine.label_region(frame_bytes, region_hint=f"video frame {i + 1}")
        if label.label in ("unknown", "") or label.confidence < 0.35:
            label = _analyze_image_heuristic(frame_bytes)

        # Keep the highest-confidence label for each object type.
        existing = best_labels.get(label.label)
        if existing is None or label.confidence > existing["confidence"]:
            best_labels[label.label] = {
                "label": label.label,
                "material": label.material,
                "mass_kg": label.mass_kg,
                "fragility": label.fragility,
                "friction": label.friction,
                "confidence": label.confidence,
            }

        # Contribute this frame's point cloud (offset along Z by frame index
        # so multi-frame clouds have spatial spread rather than all stacking).
        frame_points = _generate_depth_pointcloud(frame_bytes, n_points=400)
        z_offset = i * 0.4  # metres apart per frame
        for pt in frame_points:
            all_points.append([pt[0], pt[1], pt[2] + z_offset, pt[3], pt[4], pt[5]])

    job.update({"stage": "risk", "progress": 0.90})

    objects = list(best_labels.values())
    risks: list[dict[str, Any]] = []
    agent = _get_agent()

    for obj_data in objects:
        mass_factor = min(obj_data["mass_kg"] / 20.0, 0.3) * 0.3
        risk_score = min(1.0, obj_data["fragility"] * 0.4 + mass_factor + 0.1)
        if risk_score > 0.25:
            risks.append(
                {
                    "object_label": obj_data["label"],
                    "risk_score": round(risk_score, 3),
                    "description": (
                        f"{obj_data['label']} — fragility {obj_data['fragility']:.2f}, "
                        f"mass {obj_data['mass_kg']:.1f} kg"
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
        position = _next_upload_position()
        await agent.ingest_from_upload(sl, position=position)

    job.update(
        {
            "status": "complete",
            "stage": "complete",
            "progress": 1.0,
            "objects": objects,
            "risks": risks,
            "point_cloud": all_points,
        }
    )
    logger.info(
        "video_upload_complete",
        objects=len(objects),
        risks=len(risks),
        points=len(all_points),
    )


async def _process_upload(job_id: str, content: bytes, content_type: str) -> None:
    """Background task: run the file through the analysis pipeline."""
    job = _upload_jobs[job_id]

    try:
        job.update({"status": "processing", "stage": "ingest", "progress": 0.15})
        await asyncio.sleep(0.3)

        is_image = content_type in _IMAGE_TYPES or content_type.startswith("image/")
        is_video = content_type in _VIDEO_TYPES or content_type.startswith("video/")

        if is_image:
            job.update({"stage": "vlm", "progress": 0.4})
            engine = await _get_or_init_upload_vlm()
            label = await engine.label_region(content, region_hint="uploaded scene image")

            # If VLM is offline the fallback returns "unknown" — run pixel-level
            # heuristic analysis on the actual image bytes instead.
            if label.label in ("unknown", "") or label.confidence < 0.35:
                label = _analyze_image_heuristic(content)

            job.update({"stage": "risk", "progress": 0.75})
            await asyncio.sleep(0.2)

            mass_factor = min(label.mass_kg / 20.0, 0.3) * 0.3
            risk_score = min(1.0, label.fragility * 0.4 + mass_factor + 0.1)
            objects = [
                {
                    "label": label.label,
                    "material": label.material,
                    "mass_kg": label.mass_kg,
                    "fragility": label.fragility,
                    "friction": label.friction,
                    "confidence": label.confidence,
                }
            ]
            risks = (
                [
                    {
                        "object_label": label.label,
                        "risk_score": round(risk_score, 3),
                        "description": (
                            f"{label.label} — fragility {label.fragility:.2f}, "
                            f"mass {label.mass_kg:.1f} kg"
                        ),
                    }
                ]
                if risk_score > 0.25
                else []
            )
            # Generate pseudo-depth point cloud from actual image pixels
            point_cloud = _generate_depth_pointcloud(content)

            job.update(
                {
                    "status": "complete",
                    "stage": "complete",
                    "progress": 1.0,
                    "objects": objects,
                    "risks": risks,
                    "point_cloud": point_cloud,
                }
            )

            # ── Inject into the live world model so /objects and /scene update ──
            agent = _get_agent()
            position = _next_upload_position()
            await agent.ingest_from_upload(label, position=position)

        elif is_video:
            await _process_video_upload(job, content)
        else:
            msg = f"Unsupported file type: {content_type!r}"
            raise ValueError(msg)

    except Exception as exc:
        logger.warning("upload_processing_failed", job_id=job_id, error=str(exc))
        job.update({"status": "error", "progress": 1.0, "error": str(exc)})


# ── Upload endpoints ──────────────────────────────────────────────────────────


@app.post("/upload", response_model=UploadJobStatus)
async def upload_media(file: Annotated[UploadFile, File()]) -> UploadJobStatus:
    """Accept an image or video file and run it through the analysis pipeline.

    Returns immediately with a ``job_id``. Poll ``GET /jobs/{job_id}`` for
    status updates as the file moves through the pipeline stages:
    ``upload → ingest → vlm → risk → complete``.

    Supported image formats: JPEG, PNG, WEBP, GIF.
    Video formats (MP4, MOV, WEBM) are accepted but require the Rust SLAM
    pipeline for full 3-D reconstruction.
    """
    job_id = uuid.uuid4().hex[:8]
    filename = file.filename or "unknown"
    content_type = file.content_type or ""

    job: dict[str, Any] = {
        "job_id": job_id,
        "filename": filename,
        "status": "queued",
        "stage": "upload",
        "progress": 0.0,
        "objects": None,
        "risks": None,
        "error": None,
    }
    _upload_jobs[job_id] = job

    content = await file.read()
    job.update({"stage": "upload", "progress": 0.1})

    _task = asyncio.create_task(_process_upload(job_id, content, content_type))
    _state.setdefault("_tasks", []).append(_task)

    logger.info("upload_accepted", job_id=job_id, filename=filename, content_type=content_type)
    return UploadJobStatus(**job)


@app.get("/jobs", response_model=list[UploadJobStatus])
def list_upload_jobs() -> list[UploadJobStatus]:
    """List all upload jobs and their current status."""
    return [UploadJobStatus(**j) for j in _upload_jobs.values()]


@app.get("/jobs/{job_id}", response_model=UploadJobStatus)
def get_upload_job(job_id: str) -> UploadJobStatus:
    """Return the current status of a specific upload job.

    Args:
        job_id: The 8-character hex job ID returned by ``POST /upload``.

    Raises:
        HTTPException: 404 if the job is not found.
    """
    if job_id not in _upload_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return UploadJobStatus(**_upload_jobs[job_id])


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
