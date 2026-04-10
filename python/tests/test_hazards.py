"""Tests for the upload hazard ontology."""

from __future__ import annotations

from atlas.world_model.hazards import (
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


def test_build_recommendations_from_hazards_deduplicates() -> None:
    hazards = [
        {
            "hazard_code": "fragile_breakable",
            "hazard_title": "Fragile breakable item",
            "object_label": "Glass Object",
            "severity": "critical",
            "location_label": "front-right",
            "recommendation": "Move it away from the edge.",
            "why": "It is fragile and near the edge.",
        },
        {
            "hazard_code": "fragile_breakable",
            "hazard_title": "Fragile breakable item",
            "object_label": "Glass Object",
            "severity": "critical",
            "location_label": "front-right",
            "recommendation": "Move it away from the edge.",
            "why": "It is fragile and near the edge.",
        },
    ]

    recommendations = build_recommendations_from_hazards(hazards)

    assert len(recommendations) == 1
    assert recommendations[0]["title"] == "Fragile breakable item"
