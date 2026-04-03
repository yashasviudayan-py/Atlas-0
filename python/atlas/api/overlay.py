"""Overlay primitives for AR visualisation of risk assessments.

Converts :class:`~atlas.world_model.risk_aggregator.AggregatedRisk` objects
into renderable overlay primitives for the Three.js frontend.  Supports
perspective projection from 3D world space to 2D screen space using a
pinhole camera model and an optional world-to-camera pose matrix.

Primitives
----------
:class:`RiskZone`
    Semi-transparent sphere highlighting an at-risk object.
:class:`TrajectoryArc`
    Predicted fall path as a sampled list of 3D world-space points.
:class:`ImpactZone`
    Circular polygon on the predicted impact surface.
:class:`Alert`
    Text badge with severity and optional projected screen position.

Example::

    from atlas.api.overlay import CameraParams, OverlayBuilder

    builder = OverlayBuilder(camera=CameraParams(fx=600, fy=600, cx=320, cy=240))
    payload = builder.build_overlay_payload(aggregated_risks)
    # payload is a JSON-serialisable dict ready for the WebSocket stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import structlog

from atlas.world_model.risk_aggregator import AggregatedRisk

logger = structlog.get_logger(__name__)

# ── Camera parameters ─────────────────────────────────────────────────────────


@dataclass
class CameraParams:
    """Pinhole camera intrinsics + optional pose for 3D→2D projection.

    Args:
        fx: Focal length in x (pixels).
        fy: Focal length in y (pixels).
        cx: Principal point x (pixels).
        cy: Principal point y (pixels).
        width: Image width in pixels.
        height: Image height in pixels.
        pose_matrix: Flattened 4x4 row-major **world-to-camera** transform
            (16 floats).  If ``None``, :func:`project_3d_to_2d` always returns
            ``None``.

    Example::

        cam = CameraParams(fx=525.0, fy=525.0, cx=320.0, cy=240.0)
    """

    fx: float = 525.0
    fy: float = 525.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480
    pose_matrix: list[float] | None = None  # 16 floats, row-major 4x4


# ── Projection ────────────────────────────────────────────────────────────────


def project_3d_to_2d(
    point: tuple[float, float, float],
    camera: CameraParams,
) -> tuple[float, float] | None:
    """Project a 3D world-space point to 2D pixel coordinates.

    Applies the world-to-camera pose matrix then the standard pinhole
    projection model: ``px = fx * (Xc / Zc) + cx``.

    Args:
        point: World-space ``(x, y, z)`` coordinate.
        camera: Camera intrinsics and optional world-to-camera pose.

    Returns:
        ``(px, py)`` pixel coordinates, or ``None`` if the point is behind
        the camera (``Zc ≤ 0``) or *camera.pose_matrix* is ``None`` / invalid.

    Example::

        cam = CameraParams(
            fx=525.0, fy=525.0, cx=320.0, cy=240.0,
            pose_matrix=[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
        )
        px = project_3d_to_2d((0.0, 0.0, 1.0), cam)
        assert px == (320.0, 240.0)
    """
    if camera.pose_matrix is None:
        return None

    m = camera.pose_matrix
    if len(m) != 16:
        logger.warning("invalid_pose_matrix_length", got=len(m), expected=16)
        return None

    wx, wy, wz = point
    # Row-major 4x4 multiply; ignore the homogeneous w output.
    xc = m[0] * wx + m[1] * wy + m[2] * wz + m[3]
    yc = m[4] * wx + m[5] * wy + m[6] * wz + m[7]
    zc = m[8] * wx + m[9] * wy + m[10] * wz + m[11]

    if zc <= 0.0:
        return None  # behind camera

    px = camera.fx * (xc / zc) + camera.cx
    py = camera.fy * (yc / zc) + camera.cy
    return (px, py)


# ── Overlay primitives ────────────────────────────────────────────────────────


@dataclass
class RiskZone:
    """Semi-transparent sphere zone highlighting an at-risk object.

    Args:
        center: World-space centre ``(x, y, z)`` in metres.
        radius: Zone radius in metres; scaled by combined risk score.
        color: RGB tuple with each component in ``[0, 255]``.
        opacity: Zone opacity in ``[0.0, 1.0]``.
        object_id: Source object identifier.
        risk_type: Category string (e.g. ``"Fall"``, ``"Spill"``).

    Example::

        zone = RiskZone(
            center=(1.0, 1.5, 0.5), radius=0.6,
            color=(255, 80, 0), opacity=0.4,
            object_id=3, risk_type="Fall",
        )
        d = zone.to_dict()
        assert d["type"] == "risk_zone"
    """

    center: tuple[float, float, float]
    radius: float
    color: tuple[int, int, int]
    opacity: float
    object_id: int
    risk_type: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        Returns:
            Dict with keys ``type``, ``object_id``, ``risk_type``,
            ``center``, ``radius``, ``color``, ``opacity``.
        """
        return {
            "type": "risk_zone",
            "object_id": self.object_id,
            "risk_type": self.risk_type,
            "center": list(self.center),
            "radius": self.radius,
            "color": list(self.color),
            "opacity": self.opacity,
        }


@dataclass
class TrajectoryArc:
    """Predicted fall trajectory as an ordered list of 3D world-space points.

    Args:
        points: Sequence of ``(x, y, z)`` positions sampled along the arc.
        object_id: Source object identifier.

    Example::

        arc = TrajectoryArc(
            points=[(0.0, 2.0, 0.0), (0.1, 1.0, 0.0), (0.2, 0.0, 0.0)],
            object_id=1,
        )
        assert arc.to_dict()["type"] == "trajectory_arc"
    """

    points: list[tuple[float, float, float]]
    object_id: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        Returns:
            Dict with keys ``type``, ``object_id``, ``points``.
        """
        return {
            "type": "trajectory_arc",
            "object_id": self.object_id,
            "points": [list(p) for p in self.points],
        }


@dataclass
class ImpactZone:
    """Circular polygon on the predicted impact surface.

    Args:
        polygon: List of ``(x, y, z)`` positions forming the polygon boundary.
        center: Impact point centre ``(x, y, z)``.
        object_id: Source object identifier.

    Example::

        zone = ImpactZone(
            polygon=[(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
            center=(0.5, 0.0, 0.5),
            object_id=2,
        )
    """

    polygon: list[tuple[float, float, float]]
    center: tuple[float, float, float]
    object_id: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        Returns:
            Dict with keys ``type``, ``object_id``, ``center``, ``polygon``.
        """
        return {
            "type": "impact_zone",
            "object_id": self.object_id,
            "center": list(self.center),
            "polygon": [list(p) for p in self.polygon],
        }


@dataclass
class Alert:
    """Text badge overlaid near a world-space anchor point.

    Args:
        text: Message to display (truncated to 100 chars by the builder).
        severity: Severity in ``[0, 1]`` (drives visual styling in the frontend).
        world_position: World-space anchor ``(x, y, z)``.
        screen_position: Projected pixel coordinates ``(px, py)``, or ``None``
            if the anchor is off-screen or no camera pose was provided.
        object_id: Source object identifier.

    Example::

        alert = Alert(
            text="Vase may fall", severity=0.8,
            world_position=(1.0, 1.5, 0.5),
            screen_position=(312.0, 190.0),
            object_id=3,
        )
    """

    text: str
    severity: float
    world_position: tuple[float, float, float]
    screen_position: tuple[float, float] | None
    object_id: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        Returns:
            Dict with keys ``type``, ``object_id``, ``text``, ``severity``,
            ``world_position``, ``screen_position`` (list or null).
        """
        return {
            "type": "alert",
            "object_id": self.object_id,
            "text": self.text,
            "severity": self.severity,
            "world_position": list(self.world_position),
            "screen_position": list(self.screen_position) if self.screen_position else None,
        }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _risk_color(score: float) -> tuple[int, int, int]:
    """Map *score* ∈ [0, 1] to an RGB colour (yellow → orange → red).

    Args:
        score: Risk score in ``[0, 1]``.

    Returns:
        ``(r, g, b)`` with each component in ``[0, 255]``.
    """
    score = max(0.0, min(1.0, score))
    r = min(255, int(128 + 127 * score))
    g = max(0, int(255 - 255 * score))
    return (r, g, 0)


def _build_trajectory_arc(
    start: tuple[float, float, float],
    impact: tuple[float, float, float],
    n_points: int = 16,
) -> list[tuple[float, float, float]]:
    """Generate a parabolic arc from *start* to *impact* under gravity.

    The horizontal components (x, z) interpolate linearly.  The vertical
    component (y) follows a parabola that peaks halfway and lands at *impact*,
    giving a natural ballistic appearance.

    Args:
        start: World-space start position (object centre before falling).
        impact: Predicted impact point.
        n_points: Number of arc segments; returned list has ``n_points + 1``
            entries.

    Returns:
        List of ``(x, y, z)`` positions along the arc.
    """
    sx, sy, sz = start
    ix, iy, iz = impact
    arc: list[tuple[float, float, float]] = []
    for i in range(n_points + 1):
        t = i / n_points
        x = sx + (ix - sx) * t
        # Parabola: starts at sy, peaks midway, ends at iy.
        vertical_drop = (iy - sy) * t
        parabolic_lift = 0.5 * abs(sy - iy) * t * (1.0 - t) * 4.0
        y = sy + vertical_drop + parabolic_lift * (1.0 - t * 2.0) if sy > iy else sy + vertical_drop
        z = sz + (iz - sz) * t
        arc.append((round(x, 4), round(y, 4), round(z, 4)))
    return arc


def _build_impact_polygon(
    center: tuple[float, float, float],
    radius: float = 0.25,
    n_verts: int = 12,
) -> list[tuple[float, float, float]]:
    """Build a horizontal circular polygon centred at *center*.

    All polygon vertices share the y-coordinate of *center* (flat on the
    impact surface).

    Args:
        center: Centre of the impact zone.
        radius: Polygon radius in metres.
        n_verts: Number of polygon vertices.

    Returns:
        List of ``(x, y, z)`` positions forming the closed polygon.
    """
    cx, cy, cz = center
    polygon: list[tuple[float, float, float]] = []
    for i in range(n_verts):
        angle = 2.0 * math.pi * i / n_verts
        polygon.append(
            (
                round(cx + radius * math.cos(angle), 4),
                cy,
                round(cz + radius * math.sin(angle), 4),
            )
        )
    return polygon


# ── OverlayBuilder ────────────────────────────────────────────────────────────


class OverlayBuilder:
    """Converts :class:`~atlas.world_model.risk_aggregator.AggregatedRisk`
    entries into a complete overlay payload for the Three.js frontend.

    For each risk the builder produces:

    - A :class:`RiskZone` sphere (always present).
    - A :class:`TrajectoryArc` parabola (only when *impact_point* is known).
    - An :class:`ImpactZone` polygon (only when *impact_point* is known).
    - An :class:`Alert` text badge (always present).

    Args:
        camera: Camera intrinsics + pose used for 3D→2D projection.  Defaults
            to a 640x480 camera with no pose (projection returns ``None``).
        risk_zone_base_radius: Baseline zone radius in metres before score
            scaling.
        trajectory_points: Number of arc segments in generated trajectories.
        impact_zone_radius: Radius of the impact polygon in metres.

    Example::

        builder = OverlayBuilder(risk_zone_base_radius=0.4)
        builder.update_camera(CameraParams(pose_matrix=[...]))
        payload = builder.build_overlay_payload(risks)
    """

    def __init__(
        self,
        camera: CameraParams | None = None,
        risk_zone_base_radius: float = 0.5,
        trajectory_points: int = 16,
        impact_zone_radius: float = 0.25,
    ) -> None:
        self._camera = camera if camera is not None else CameraParams()
        self._base_radius = risk_zone_base_radius
        self._traj_points = trajectory_points
        self._impact_radius = impact_zone_radius

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def camera(self) -> CameraParams:
        """Current camera parameters."""
        return self._camera

    # ── Mutation ──────────────────────────────────────────────────────────────

    def update_camera(self, camera: CameraParams) -> None:
        """Replace the camera parameters (e.g. after a new SLAM keyframe).

        Args:
            camera: Updated intrinsics and world-to-camera pose.
        """
        self._camera = camera

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_from_risk(
        self,
        risk: AggregatedRisk,
    ) -> dict[str, Any]:
        """Convert one :class:`AggregatedRisk` to a set of overlay primitives.

        The object *position* is used as the risk zone centre when available;
        otherwise the *impact_point* is used.  If neither is available the
        zone is placed at the world origin and a warning is logged.

        Args:
            risk: Aggregated risk entry to visualise.

        Returns:
            Dict with keys:

            - ``"risk_zone"`` — :class:`RiskZone` instance.
            - ``"trajectory_arc"`` — :class:`TrajectoryArc` or ``None``.
            - ``"impact_zone"`` — :class:`ImpactZone` or ``None``.
            - ``"alert"`` — :class:`Alert` instance.
        """
        # Determine the object anchor (centre of risk zone / alert).
        if risk.position is not None:
            anchor = risk.position
        elif risk.impact_point is not None:
            # Lift slightly above impact point as a proxy for object position.
            anchor = (risk.impact_point[0], risk.impact_point[1] + 1.5, risk.impact_point[2])
        else:
            logger.debug("no_position_for_risk", object_id=risk.object_id)
            anchor = (0.0, 0.0, 0.0)

        color = _risk_color(risk.combined_score)
        opacity = round(min(0.85, 0.3 + 0.55 * risk.combined_score), 3)
        radius = round(self._base_radius * (1.0 + risk.combined_score), 3)

        risk_zone = RiskZone(
            center=anchor,
            radius=radius,
            color=color,
            opacity=opacity,
            object_id=risk.object_id,
            risk_type=risk.risk_type,
        )

        trajectory_arc: TrajectoryArc | None = None
        impact_zone: ImpactZone | None = None

        if risk.impact_point is not None:
            arc_pts = _build_trajectory_arc(anchor, risk.impact_point, self._traj_points)
            trajectory_arc = TrajectoryArc(points=arc_pts, object_id=risk.object_id)

            poly = _build_impact_polygon(risk.impact_point, self._impact_radius)
            impact_zone = ImpactZone(
                polygon=poly,
                center=risk.impact_point,
                object_id=risk.object_id,
            )

        screen_pos = project_3d_to_2d(anchor, self._camera)
        label = risk.object_label
        desc = risk.description[:80] if risk.description else ""
        alert_text = f"{label}: {desc}" if desc else label

        alert = Alert(
            text=alert_text,
            severity=round(risk.combined_score, 3),
            world_position=anchor,
            screen_position=screen_pos,
            object_id=risk.object_id,
        )

        return {
            "risk_zone": risk_zone,
            "trajectory_arc": trajectory_arc,
            "impact_zone": impact_zone,
            "alert": alert,
        }

    def build_overlay_payload(
        self,
        risks: list[AggregatedRisk],
    ) -> dict[str, Any]:
        """Build a complete overlay payload from a ranked list of risks.

        Args:
            risks: Ordered list from
                :meth:`~atlas.world_model.risk_aggregator.RiskAggregator.get_top_risks`.

        Returns:
            JSON-serialisable dict with keys:

            - ``"risk_zones"`` — list of risk zone dicts.
            - ``"trajectory_arcs"`` — list of arc dicts (``None`` entries excluded).
            - ``"impact_zones"`` — list of zone dicts (``None`` entries excluded).
            - ``"alerts"`` — list of alert dicts.
        """
        risk_zones: list[dict[str, Any]] = []
        trajectory_arcs: list[dict[str, Any]] = []
        impact_zones: list[dict[str, Any]] = []
        alerts: list[dict[str, Any]] = []

        for risk in risks:
            primitives = self.build_from_risk(risk)
            risk_zones.append(primitives["risk_zone"].to_dict())
            if primitives["trajectory_arc"] is not None:
                trajectory_arcs.append(primitives["trajectory_arc"].to_dict())
            if primitives["impact_zone"] is not None:
                impact_zones.append(primitives["impact_zone"].to_dict())
            alerts.append(primitives["alert"].to_dict())

        return {
            "risk_zones": risk_zones,
            "trajectory_arcs": trajectory_arcs,
            "impact_zones": impact_zones,
            "alerts": alerts,
        }


__all__ = [
    "Alert",
    "CameraParams",
    "ImpactZone",
    "OverlayBuilder",
    "RiskZone",
    "TrajectoryArc",
    "project_3d_to_2d",
]
