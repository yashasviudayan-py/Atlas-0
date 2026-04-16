"""Tests for the upload hazard ontology."""

from __future__ import annotations

from atlas.world_model.hazards import (
    build_fix_first_actions,
    build_recommendations_from_hazards,
    evaluate_upload_hazards,
)


def test_evaluate_upload_hazards_returns_expected_codes() -> None:
    obj = {
        "object_id": "track-01",
        "label": "Glass Object",
        "material": "Glass",
        "mass_kg": 0.7,
        "fragility": 0.92,
        "confidence": 0.81,
        "grounding_confidence": 0.74,
        "position": [1.3, 1.1, 1.2],
        "location_label": "front-right",
        "observation_count": 3,
        "position_variance": 0.14,
        "estimated_height_m": 1.4,
        "estimated_width_m": 0.42,
        "edge_proximity": 0.84,
        "path_clutter_score": 0.22,
        "evidence_ids": ["e1", "e2"],
    }

    hazards = evaluate_upload_hazards([obj])
    codes = {hazard["hazard_code"] for hazard in hazards}

    assert "fragile_breakable" in codes
    assert "edge_placement" in codes
    assert "liquid_spill" in codes
    assert all("evidence" in hazard for hazard in hazards)
    assert any(hazard["reasoning"]["rule_hits"] for hazard in hazards)
    assert all("object_snapshot" in hazard["reasoning"] for hazard in hazards)


def test_build_recommendations_from_hazards_deduplicates() -> None:
    hazards = [
        {
            "object_id": "track-01",
            "hazard_code": "fragile_breakable",
            "hazard_title": "Fragile breakable item",
            "object_label": "Glass Object",
            "severity": "critical",
            "location_label": "front-right",
            "recommendation": "Move it away from the edge.",
            "why": "It is fragile and near the edge.",
            "priority_score": 0.91,
            "confidence_label": "strong",
        },
        {
            "object_id": "track-01",
            "hazard_code": "edge_placement",
            "hazard_title": "Object placed near an edge",
            "object_label": "Glass Object",
            "severity": "high",
            "location_label": "front-right",
            "recommendation": "Move it away from the edge.",
            "why": "It is fragile and near the edge.",
            "priority_score": 0.73,
            "confidence_label": "approximate",
        },
    ]

    recommendations = build_recommendations_from_hazards(hazards)

    assert len(recommendations) == 1
    assert recommendations[0]["title"] == "Fragile breakable item"


def test_build_fix_first_actions_prefers_highest_priority_per_object() -> None:
    hazards = [
        {
            "object_id": "track-01",
            "hazard_code": "fragile_breakable",
            "hazard_title": "Fragile breakable item",
            "severity": "critical",
            "location_label": "front-right",
            "what_to_do_next": "Move it away from the edge.",
            "why_it_matters": "It is fragile and near the edge.",
            "confidence": 0.82,
            "confidence_label": "strong",
            "priority_score": 0.91,
        },
        {
            "object_id": "track-01",
            "hazard_code": "edge_placement",
            "hazard_title": "Object placed near an edge",
            "severity": "high",
            "location_label": "front-right",
            "what_to_do_next": "Pull it farther back.",
            "why_it_matters": "It may fall if bumped.",
            "confidence": 0.68,
            "confidence_label": "approximate",
            "priority_score": 0.71,
        },
        {
            "object_id": "track-02",
            "hazard_code": "walkway_clutter",
            "hazard_title": "Walkway clutter",
            "severity": "moderate",
            "location_label": "front-center",
            "what_to_do_next": "Clear the walking path.",
            "why_it_matters": "It may create a trip hazard.",
            "confidence": 0.65,
            "confidence_label": "approximate",
            "priority_score": 0.66,
        },
    ]

    actions = build_fix_first_actions(hazards)

    assert len(actions) == 2
    assert actions[0]["hazard_code"] == "fragile_breakable"
    assert actions[1]["hazard_code"] == "walkway_clutter"
