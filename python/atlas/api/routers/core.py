"""Core spatial-reasoning API: health, query, objects, scene, metrics, ws."""

from __future__ import annotations

import asyncio
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Response, WebSocket, WebSocketDisconnect

from atlas.api.analytics import _startup_check_summary
from atlas.api.helpers import _fmt_pos
from atlas.api.jobs import _refresh_operational_metrics
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
from atlas.api.models import (
    HealthResponse,
    ObjectInfo,
    OverlayRiskEntry,
    RiskDeltaMessage,
    RiskInfo,
    SceneState,
    SpatialQuery,
    SpatialQueryResult,
)
from atlas.api.state import (
    _current_objects,
    _get_agent,
    _get_aggregator,
    _get_overlay_builder,
    _query_parser,
    _upload_cfg,
    _upload_jobs,
)
from atlas.world_model.agent import RiskEntry, WorldModelAgent
from atlas.world_model.query_parser import QueryType
from atlas.world_model.relationships import SemanticObject

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
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
    startup = _startup_check_summary()
    warnings = [
        check["detail"]
        for check in startup.get("checks", [])
        if str(check.get("status")) != "pass" and isinstance(check.get("detail"), str)
    ]
    _refresh_operational_metrics()
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
        deployment_ready=bool(startup.get("ready")),
        worker_mode=_upload_cfg.worker_mode,
        warnings=warnings[:4],
    )


@router.post("/query", response_model=list[SpatialQueryResult])
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


@router.get("/objects", response_model=list[ObjectInfo])
async def list_objects(
    agent: Annotated[WorldModelAgent, Depends(_get_agent)],
) -> list[ObjectInfo]:
    """List all labeled objects with their physical properties and relationships.

    Returns an empty list when no map snapshot has been processed yet.
    """
    objects = _current_objects(agent)
    risks = await agent.get_risks()
    return [_to_object_info(obj, risks) for obj in objects]


@router.get("/scene", response_model=SceneState)
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


@router.get("/metrics", include_in_schema=False)
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


@router.websocket("/ws/risks")
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
