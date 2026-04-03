"""Tests for the overlay module — Phase 3 Part 11.

Covers:
- CameraParams defaults and custom values.
- project_3d_to_2d: identity pose, off-screen, behind-camera, no-pose cases.
- _risk_color: low/mid/high scores, clamping.
- _build_trajectory_arc: point count, start/end correctness, monotonicity.
- _build_impact_polygon: vertex count, planar, roughly circular.
- RiskZone.to_dict: all fields present, correct types.
- TrajectoryArc.to_dict: all fields present.
- ImpactZone.to_dict: all fields present.
- Alert.to_dict: with/without screen_position.
- OverlayBuilder.build_from_risk: no position, no impact_point, both, only impact.
- OverlayBuilder.build_overlay_payload: structure, empty input.
- OverlayBuilder.update_camera: camera replaced.
- AggregatedRisk.position propagation from heuristic source (regression).
"""

from __future__ import annotations

import math

import pytest
from atlas.api.overlay import (
    Alert,
    CameraParams,
    ImpactZone,
    OverlayBuilder,
    RiskZone,
    TrajectoryArc,
    _build_impact_polygon,
    _build_trajectory_arc,
    _risk_color,
    project_3d_to_2d,
)
from atlas.world_model.agent import RiskEntry
from atlas.world_model.risk_aggregator import AggregatedRisk, RiskAggregator

# ── Helpers ───────────────────────────────────────────────────────────────────


def _identity_cam() -> CameraParams:
    """Camera with an identity world-to-camera transform."""
    return CameraParams(
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        pose_matrix=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    )


def _make_risk(
    oid: int = 1,
    combined: float = 0.7,
    risk_type: str = "Fall",
    impact_point: tuple[float, float, float] | None = None,
    position: tuple[float, float, float] | None = None,
) -> AggregatedRisk:
    return AggregatedRisk(
        object_id=oid,
        object_label="vase",
        physics_score=combined,
        heuristic_score=combined,
        combined_score=combined,
        risk_type=risk_type,
        impact_point=impact_point,
        description="test risk",
        sources=["physics"],
        position=position,
    )


# ── CameraParams ──────────────────────────────────────────────────────────────


class TestCameraParams:
    def test_defaults(self) -> None:
        cam = CameraParams()
        assert cam.fx == 525.0
        assert cam.fy == 525.0
        assert cam.width == 640
        assert cam.height == 480
        assert cam.pose_matrix is None

    def test_custom_values(self) -> None:
        cam = CameraParams(fx=600.0, fy=600.0, cx=400.0, cy=300.0)
        assert cam.fx == 600.0
        assert cam.cx == 400.0


# ── project_3d_to_2d ─────────────────────────────────────────────────────────


class TestProject3dTo2d:
    def test_no_pose_returns_none(self) -> None:
        cam = CameraParams()
        assert project_3d_to_2d((1.0, 2.0, 3.0), cam) is None

    def test_identity_pose_on_axis(self) -> None:
        cam = _identity_cam()
        result = project_3d_to_2d((0.0, 0.0, 1.0), cam)
        assert result is not None
        px, py = result
        assert abs(px - 320.0) < 1e-6  # on-axis → principal point x
        assert abs(py - 240.0) < 1e-6  # on-axis → principal point y

    def test_off_centre_point(self) -> None:
        cam = _identity_cam()
        result = project_3d_to_2d((1.0, 0.0, 1.0), cam)
        assert result is not None
        px, _ = result
        assert px > 320.0  # positive X shifts right

    def test_behind_camera_returns_none(self) -> None:
        cam = _identity_cam()
        # Z = -1 is behind the camera.
        assert project_3d_to_2d((0.0, 0.0, -1.0), cam) is None

    def test_invalid_pose_length_returns_none(self) -> None:
        cam = CameraParams(pose_matrix=[1.0, 0.0, 0.0])
        assert project_3d_to_2d((0.0, 0.0, 1.0), cam) is None

    def test_translation_in_pose(self) -> None:
        # Pose adds +1 to Zc, so world origin projects to Zc=1 (in front).
        cam = CameraParams(
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            pose_matrix=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        )
        result = project_3d_to_2d((0.0, 0.0, 0.0), cam)
        assert result is not None
        assert abs(result[0] - 320.0) < 1e-6


# ── _risk_color ───────────────────────────────────────────────────────────────


class TestRiskColor:
    def test_zero_score_is_yellow_ish(self) -> None:
        r, g, b = _risk_color(0.0)
        assert b == 0
        assert g > r  # mostly green component

    def test_full_score_is_red(self) -> None:
        r, g, b = _risk_color(1.0)
        assert r == 255
        assert g == 0
        assert b == 0

    def test_mid_score_has_both_r_and_g(self) -> None:
        r, g, b = _risk_color(0.5)
        assert r > 0
        assert g > 0
        assert b == 0

    def test_clamped_above_one(self) -> None:
        c1 = _risk_color(1.0)
        c2 = _risk_color(2.0)
        assert c1 == c2

    def test_clamped_below_zero(self) -> None:
        c1 = _risk_color(0.0)
        c2 = _risk_color(-0.5)
        assert c1 == c2


# ── _build_trajectory_arc ─────────────────────────────────────────────────────


class TestBuildTrajectoryArc:
    def test_default_n_points(self) -> None:
        arc = _build_trajectory_arc((0.0, 2.0, 0.0), (1.0, 0.0, 0.0))
        assert len(arc) == 17  # n_points + 1

    def test_custom_n_points(self) -> None:
        arc = _build_trajectory_arc((0.0, 2.0, 0.0), (1.0, 0.0, 0.0), n_points=8)
        assert len(arc) == 9

    def test_first_point_is_start(self) -> None:
        arc = _build_trajectory_arc((1.0, 3.0, 2.0), (4.0, 0.0, 5.0))
        assert arc[0] == pytest.approx((1.0, 3.0, 2.0), abs=1e-3)

    def test_last_point_is_impact(self) -> None:
        arc = _build_trajectory_arc((0.0, 2.0, 0.0), (3.0, 0.0, 1.0))
        assert arc[-1] == pytest.approx((3.0, 0.0, 1.0), abs=1e-3)

    def test_x_is_monotone(self) -> None:
        arc = _build_trajectory_arc((0.0, 2.0, 0.0), (3.0, 0.0, 0.0))
        xs = [p[0] for p in arc]
        # X should increase monotonically from 0 to 3.
        assert all(xs[i] <= xs[i + 1] + 1e-6 for i in range(len(xs) - 1))


# ── _build_impact_polygon ─────────────────────────────────────────────────────


class TestBuildImpactPolygon:
    def test_default_vertex_count(self) -> None:
        poly = _build_impact_polygon((0.0, 0.0, 0.0))
        assert len(poly) == 12

    def test_custom_vertex_count(self) -> None:
        poly = _build_impact_polygon((0.0, 0.0, 0.0), n_verts=8)
        assert len(poly) == 8

    def test_all_vertices_at_same_y(self) -> None:
        poly = _build_impact_polygon((1.0, 0.5, 2.0), radius=0.3)
        for _, y, _ in poly:
            assert abs(y - 0.5) < 1e-6

    def test_vertices_are_roughly_on_circle(self) -> None:
        cx, cy, cz = 1.0, 0.0, 2.0
        r = 0.4
        poly = _build_impact_polygon((cx, cy, cz), radius=r)
        for vx, _, vz in poly:
            dist = math.sqrt((vx - cx) ** 2 + (vz - cz) ** 2)
            assert abs(dist - r) < 1e-3


# ── Primitive to_dict ─────────────────────────────────────────────────────────


class TestPrimitiveSerialisation:
    def test_risk_zone_dict_keys(self) -> None:
        zone = RiskZone(
            center=(1.0, 2.0, 3.0),
            radius=0.5,
            color=(255, 0, 0),
            opacity=0.4,
            object_id=1,
            risk_type="Fall",
        )
        d = zone.to_dict()
        for key in ("type", "object_id", "risk_type", "center", "radius", "color", "opacity"):
            assert key in d
        assert d["type"] == "risk_zone"
        assert d["center"] == [1.0, 2.0, 3.0]

    def test_trajectory_arc_dict_keys(self) -> None:
        arc = TrajectoryArc(points=[(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)], object_id=2)
        d = arc.to_dict()
        assert d["type"] == "trajectory_arc"
        assert len(d["points"]) == 2
        assert d["points"][0] == [0.0, 1.0, 0.0]

    def test_impact_zone_dict_keys(self) -> None:
        zone = ImpactZone(
            polygon=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            center=(0.5, 0.0, 0.0),
            object_id=3,
        )
        d = zone.to_dict()
        assert d["type"] == "impact_zone"
        assert d["center"] == [0.5, 0.0, 0.0]

    def test_alert_dict_with_screen_position(self) -> None:
        alert = Alert(
            text="vase may fall",
            severity=0.8,
            world_position=(1.0, 1.5, 0.5),
            screen_position=(312.0, 190.0),
            object_id=4,
        )
        d = alert.to_dict()
        assert d["type"] == "alert"
        assert d["screen_position"] == [312.0, 190.0]

    def test_alert_dict_no_screen_position(self) -> None:
        alert = Alert(
            text="cable trip hazard",
            severity=0.4,
            world_position=(0.5, 0.0, 1.0),
            screen_position=None,
            object_id=5,
        )
        d = alert.to_dict()
        assert d["screen_position"] is None


# ── OverlayBuilder ────────────────────────────────────────────────────────────


class TestOverlayBuilder:
    def test_default_camera_set(self) -> None:
        b = OverlayBuilder()
        assert b.camera.fx == 525.0

    def test_update_camera(self) -> None:
        b = OverlayBuilder()
        new_cam = CameraParams(fx=800.0)
        b.update_camera(new_cam)
        assert b.camera.fx == 800.0

    def test_build_from_risk_always_has_risk_zone(self) -> None:
        b = OverlayBuilder()
        result = b.build_from_risk(_make_risk())
        assert result["risk_zone"] is not None
        assert isinstance(result["risk_zone"], RiskZone)

    def test_build_from_risk_always_has_alert(self) -> None:
        b = OverlayBuilder()
        result = b.build_from_risk(_make_risk())
        assert result["alert"] is not None
        assert isinstance(result["alert"], Alert)

    def test_build_from_risk_no_impact_point(self) -> None:
        b = OverlayBuilder()
        result = b.build_from_risk(_make_risk(impact_point=None))
        assert result["trajectory_arc"] is None
        assert result["impact_zone"] is None

    def test_build_from_risk_with_impact_point(self) -> None:
        b = OverlayBuilder()
        result = b.build_from_risk(_make_risk(impact_point=(1.0, 0.0, 2.0)))
        assert result["trajectory_arc"] is not None
        assert result["impact_zone"] is not None

    def test_build_from_risk_uses_position_as_anchor(self) -> None:
        b = OverlayBuilder()
        risk = _make_risk(position=(3.0, 1.5, 4.0))
        result = b.build_from_risk(risk)
        assert result["risk_zone"].center == (3.0, 1.5, 4.0)

    def test_build_from_risk_falls_back_to_impact_if_no_position(self) -> None:
        b = OverlayBuilder()
        risk = _make_risk(impact_point=(1.0, 0.0, 2.0), position=None)
        result = b.build_from_risk(risk)
        # Anchor should be impact_point lifted by 1.5m.
        zone = result["risk_zone"]
        assert abs(zone.center[1] - 1.5) < 1e-6  # impact y=0 + 1.5

    def test_risk_zone_radius_scales_with_score(self) -> None:
        b = OverlayBuilder()
        low = b.build_from_risk(_make_risk(combined=0.1))["risk_zone"]
        high = b.build_from_risk(_make_risk(combined=0.9))["risk_zone"]
        assert high.radius > low.radius

    def test_alert_screen_position_none_without_pose(self) -> None:
        b = OverlayBuilder(camera=CameraParams())  # no pose_matrix
        result = b.build_from_risk(_make_risk(position=(1.0, 1.0, 1.0)))
        assert result["alert"].screen_position is None

    def test_alert_screen_position_set_with_pose(self) -> None:
        b = OverlayBuilder(camera=_identity_cam())
        result = b.build_from_risk(_make_risk(position=(0.0, 0.0, 1.0)))
        assert result["alert"].screen_position is not None

    def test_build_overlay_payload_structure(self) -> None:
        b = OverlayBuilder()
        risks = [_make_risk(oid=1, impact_point=(0.0, 0.0, 0.0)), _make_risk(oid=2)]
        payload = b.build_overlay_payload(risks)
        for key in ("risk_zones", "trajectory_arcs", "impact_zones", "alerts"):
            assert key in payload
        assert len(payload["risk_zones"]) == 2
        assert len(payload["alerts"]) == 2
        # Only first risk has impact_point → 1 arc + 1 zone.
        assert len(payload["trajectory_arcs"]) == 1
        assert len(payload["impact_zones"]) == 1

    def test_build_overlay_payload_empty_input(self) -> None:
        b = OverlayBuilder()
        payload = b.build_overlay_payload([])
        assert payload["risk_zones"] == []
        assert payload["alerts"] == []

    def test_trajectory_arc_end_matches_impact_point(self) -> None:
        b = OverlayBuilder()
        impact = (2.0, 0.0, 3.0)
        result = b.build_from_risk(_make_risk(position=(0.0, 2.0, 0.0), impact_point=impact))
        arc = result["trajectory_arc"]
        assert arc is not None
        last = arc.points[-1]
        assert last == pytest.approx(list(impact), abs=1e-3)


# ── AggregatedRisk.position propagation (regression) ─────────────────────────


class TestAggregatedRiskPosition:
    def test_position_propagated_from_heuristic(self) -> None:
        agg = RiskAggregator()
        agg.update_heuristic_risks(
            [
                RiskEntry(
                    object_id=1,
                    object_label="bottle",
                    position=(0.5, 1.2, 0.8),
                    risk_score=0.6,
                    fragility=0.7,
                    mass_kg=0.5,
                    description="near edge",
                )
            ]
        )
        risks = agg.get_top_risks()
        assert len(risks) == 1
        assert risks[0].position == pytest.approx((0.5, 1.2, 0.8))

    def test_position_none_for_physics_only(self) -> None:
        agg = RiskAggregator()
        agg.update_physics_risks(
            [{"object_id": 5, "risk_type": "Fall", "probability": 0.7, "description": "fall"}]
        )
        risks = agg.get_top_risks()
        assert risks[0].position is None
