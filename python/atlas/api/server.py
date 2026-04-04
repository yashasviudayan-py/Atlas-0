"""FastAPI server exposing Atlas-0 spatial queries and AR overlay data.

Provides endpoints for:
- Querying the semantic 3D map with natural language.
- Listing all labeled objects and their physical properties.
- Returning the full current scene state.
- Streaming risk assessments via WebSocket.
"""

from __future__ import annotations

import asyncio
import pathlib
from typing import Annotated, Any

import structlog
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
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
from atlas.world_model.agent import RiskEntry, WorldModelAgent, WorldModelConfig
from atlas.world_model.query_parser import QueryParser, QueryType
from atlas.world_model.relationships import RelationType, SemanticObject
from atlas.world_model.risk_aggregator import RiskAggregator

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="Atlas-0",
    description="Spatial Reasoning & Physical World-Model Engine API",
    version="0.1.0",
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
    return SceneState(
        object_count=len(objects),
        objects=[_to_object_info(obj, risks) for obj in objects],
        risk_count=len(risks),
        risks=[_to_risk_info(r) for r in risks],
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


# Re-export for convenience (e.g. tests that patch the agent).
__all__ = [
    "RelationType",
    "_get_agent",
    "_get_aggregator",
    "_get_overlay_builder",
    "_set_agent",
    "_state",
    "app",
    "prometheus_metrics",
]
