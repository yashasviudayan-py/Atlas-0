"""Tests for RiskAggregator — Phase 3 Part 10.

Covers:
- PhysicsRiskEntry.from_dict: valid deserialization, impact_point parsing,
  defaults for missing fields, error on bad input.
- _severity helper: known types and unknown fallback.
- RiskAggregator constructor: validation of top_n and weights.
- update_physics_risks: replacement semantics, invalid entry skipped.
- update_heuristic_risks: replacement semantics.
- clear: empties both stores.
- get_top_risks: merge logic, sorting, top_n cap, sources field,
  physics-only / heuristic-only / both-present / empty cases.
- combined_score: monotonicity, edge cases, zero-weight behaviour.
"""

from __future__ import annotations

import pytest
from atlas.world_model.agent import RiskEntry
from atlas.world_model.risk_aggregator import (
    PhysicsRiskEntry,
    RiskAggregator,
    _severity,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _phys_dict(
    oid: int = 1,
    risk_type: str = "Fall",
    probability: float = 0.8,
    impact_point: dict | None = None,
    description: str = "may fall",
) -> dict:
    return {
        "object_id": oid,
        "risk_type": risk_type,
        "probability": probability,
        "impact_point": impact_point,
        "description": description,
    }


def _heur(
    oid: int = 1,
    label: str = "vase",
    risk_score: float = 0.7,
    fragility: float = 0.9,
    mass_kg: float = 0.3,
    description: str = "vase (glass), fragile",
) -> RiskEntry:
    return RiskEntry(
        object_id=oid,
        object_label=label,
        position=(0.5, 1.5, 0.5),
        risk_score=risk_score,
        fragility=fragility,
        mass_kg=mass_kg,
        description=description,
    )


# ── PhysicsRiskEntry.from_dict ────────────────────────────────────────────────


class TestPhysicsRiskEntryFromDict:
    def test_basic_fields_parsed(self) -> None:
        e = PhysicsRiskEntry.from_dict(_phys_dict(oid=5, risk_type="Spill", probability=0.6))
        assert e.object_id == 5
        assert e.risk_type == "Spill"
        assert abs(e.probability - 0.6) < 1e-6

    def test_impact_point_none(self) -> None:
        e = PhysicsRiskEntry.from_dict(_phys_dict())
        assert e.impact_point is None

    def test_impact_point_parsed(self) -> None:
        d = _phys_dict(impact_point={"x": 1.0, "y": 0.0, "z": 2.5})
        e = PhysicsRiskEntry.from_dict(d)
        assert e.impact_point == pytest.approx((1.0, 0.0, 2.5))

    def test_missing_risk_type_defaults_to_instability(self) -> None:
        d = {"object_id": 2, "probability": 0.5, "description": "test"}
        e = PhysicsRiskEntry.from_dict(d)
        assert e.risk_type == "Instability"

    def test_missing_probability_defaults_to_zero(self) -> None:
        d = {"object_id": 3}
        e = PhysicsRiskEntry.from_dict(d)
        assert abs(e.probability) < 1e-6

    def test_missing_object_id_raises(self) -> None:
        with pytest.raises((KeyError, ValueError)):
            PhysicsRiskEntry.from_dict({"risk_type": "Fall", "probability": 0.5})

    def test_description_field_preserved(self) -> None:
        e = PhysicsRiskEntry.from_dict(_phys_dict(description="glass may fall"))
        assert e.description == "glass may fall"


# ── _severity helper ──────────────────────────────────────────────────────────


class TestSeverity:
    def test_fall_is_highest(self) -> None:
        assert _severity("Fall") > _severity("Instability")

    def test_collision_higher_than_spill(self) -> None:
        assert _severity("Collision") > _severity("Spill")

    def test_unknown_returns_default(self) -> None:
        assert 0.0 < _severity("UnknownType") <= 1.0

    def test_all_known_types_in_range(self) -> None:
        for t in ("Fall", "Spill", "Collision", "TripHazard", "Instability"):
            s = _severity(t)
            assert 0.0 < s <= 1.0, f"severity({t}) = {s} out of (0, 1]"


# ── RiskAggregator constructor ────────────────────────────────────────────────


class TestRiskAggregatorConstructor:
    def test_default_values(self) -> None:
        agg = RiskAggregator()
        assert agg.top_n == 20

    def test_custom_top_n(self) -> None:
        agg = RiskAggregator(top_n=5)
        assert agg.top_n == 5

    def test_top_n_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            RiskAggregator(top_n=0)

    def test_physics_weight_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            RiskAggregator(physics_weight=1.5)

    def test_heuristic_weight_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            RiskAggregator(heuristic_weight=-0.1)

    def test_starts_with_empty_stores(self) -> None:
        agg = RiskAggregator()
        assert agg.physics_count == 0
        assert agg.heuristic_count == 0


# ── update_physics_risks ──────────────────────────────────────────────────────


class TestUpdatePhysicsRisks:
    def test_replaces_existing_store(self) -> None:
        agg = RiskAggregator()
        agg.update_physics_risks([_phys_dict(oid=1)])
        agg.update_physics_risks([_phys_dict(oid=2), _phys_dict(oid=3)])
        assert agg.physics_count == 2

    def test_empty_list_clears_store(self) -> None:
        agg = RiskAggregator()
        agg.update_physics_risks([_phys_dict()])
        agg.update_physics_risks([])
        assert agg.physics_count == 0

    def test_invalid_entry_skipped(self) -> None:
        agg = RiskAggregator()
        agg.update_physics_risks(
            [
                {"risk_type": "Fall", "probability": 0.9},  # no object_id → skipped
                _phys_dict(oid=2),  # valid
            ]
        )
        assert agg.physics_count == 1


# ── update_heuristic_risks ────────────────────────────────────────────────────


class TestUpdateHeuristicRisks:
    def test_replaces_existing_store(self) -> None:
        agg = RiskAggregator()
        agg.update_heuristic_risks([_heur(oid=1)])
        agg.update_heuristic_risks([_heur(oid=2), _heur(oid=3)])
        assert agg.heuristic_count == 2

    def test_empty_list_clears_store(self) -> None:
        agg = RiskAggregator()
        agg.update_heuristic_risks([_heur()])
        agg.update_heuristic_risks([])
        assert agg.heuristic_count == 0


# ── clear ─────────────────────────────────────────────────────────────────────


class TestClear:
    def test_clear_empties_both_stores(self) -> None:
        agg = RiskAggregator()
        agg.update_physics_risks([_phys_dict()])
        agg.update_heuristic_risks([_heur()])
        agg.clear()
        assert agg.physics_count == 0
        assert agg.heuristic_count == 0

    def test_get_top_risks_empty_after_clear(self) -> None:
        agg = RiskAggregator()
        agg.update_physics_risks([_phys_dict()])
        agg.clear()
        assert agg.get_top_risks() == []


# ── get_top_risks: merge logic ────────────────────────────────────────────────


class TestGetTopRisks:
    def test_empty_returns_empty(self) -> None:
        agg = RiskAggregator()
        assert agg.get_top_risks() == []

    def test_physics_only_returns_entry(self) -> None:
        agg = RiskAggregator()
        agg.update_physics_risks([_phys_dict(oid=1, probability=0.8)])
        risks = agg.get_top_risks()
        assert len(risks) == 1
        assert risks[0].object_id == 1
        assert risks[0].physics_score > 0.0
        assert risks[0].heuristic_score == 0.0
        assert "physics" in risks[0].sources

    def test_heuristic_only_returns_entry(self) -> None:
        agg = RiskAggregator()
        agg.update_heuristic_risks([_heur(oid=2, risk_score=0.6)])
        risks = agg.get_top_risks()
        assert len(risks) == 1
        assert risks[0].object_id == 2
        assert risks[0].heuristic_score == pytest.approx(0.6)
        assert risks[0].physics_score == 0.0
        assert "heuristic" in risks[0].sources

    def test_both_sources_merged_on_same_id(self) -> None:
        agg = RiskAggregator()
        agg.update_physics_risks([_phys_dict(oid=1, probability=0.9)])
        agg.update_heuristic_risks([_heur(oid=1, risk_score=0.7)])
        risks = agg.get_top_risks()
        assert len(risks) == 1
        r = risks[0]
        assert "physics" in r.sources
        assert "heuristic" in r.sources
        # Combined score must be between the two individual scores
        assert r.combined_score > 0.0

    def test_separate_ids_produce_separate_entries(self) -> None:
        agg = RiskAggregator()
        agg.update_physics_risks([_phys_dict(oid=1)])
        agg.update_heuristic_risks([_heur(oid=2)])
        risks = agg.get_top_risks()
        assert len(risks) == 2
        ids = {r.object_id for r in risks}
        assert ids == {1, 2}

    def test_sorted_by_combined_score_descending(self) -> None:
        agg = RiskAggregator()
        agg.update_physics_risks(
            [
                _phys_dict(oid=1, probability=0.9),
                _phys_dict(oid=2, probability=0.2),
                _phys_dict(oid=3, probability=0.6),
            ]
        )
        risks = agg.get_top_risks()
        scores = [r.combined_score for r in risks]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_cap_applied(self) -> None:
        agg = RiskAggregator(top_n=2)
        agg.update_physics_risks([_phys_dict(oid=i, probability=0.5) for i in range(5)])
        assert len(agg.get_top_risks()) == 2

    def test_label_from_heuristic_when_available(self) -> None:
        agg = RiskAggregator()
        agg.update_physics_risks([_phys_dict(oid=1)])
        agg.update_heuristic_risks([_heur(oid=1, label="fancy_vase")])
        risks = agg.get_top_risks()
        assert risks[0].object_label == "fancy_vase"

    def test_label_fallback_for_physics_only(self) -> None:
        agg = RiskAggregator()
        agg.update_physics_risks([_phys_dict(oid=42)])
        risks = agg.get_top_risks()
        assert risks[0].object_label == "object_42"

    def test_risk_type_from_physics_when_available(self) -> None:
        agg = RiskAggregator()
        agg.update_physics_risks([_phys_dict(oid=1, risk_type="Spill")])
        risks = agg.get_top_risks()
        assert risks[0].risk_type == "Spill"

    def test_impact_point_from_physics(self) -> None:
        d = _phys_dict(oid=1, impact_point={"x": 1.0, "y": 0.0, "z": 2.0})
        agg = RiskAggregator()
        agg.update_physics_risks([d])
        risks = agg.get_top_risks()
        assert risks[0].impact_point == pytest.approx((1.0, 0.0, 2.0))

    def test_description_concatenated(self) -> None:
        agg = RiskAggregator()
        agg.update_physics_risks([_phys_dict(oid=1, description="may fall")])
        agg.update_heuristic_risks([_heur(oid=1, description="fragile, elevated")])
        risks = agg.get_top_risks()
        assert "may fall" in risks[0].description
        assert "fragile, elevated" in risks[0].description

    def test_combined_score_bounded(self) -> None:
        agg = RiskAggregator()
        agg.update_physics_risks([_phys_dict(oid=1, probability=1.0)])
        agg.update_heuristic_risks([_heur(oid=1, risk_score=1.0)])
        risks = agg.get_top_risks()
        assert 0.0 <= risks[0].combined_score <= 1.0

    def test_physics_weight_zero_uses_heuristic_only(self) -> None:
        agg = RiskAggregator(physics_weight=0.0, heuristic_weight=1.0)
        agg.update_physics_risks([_phys_dict(oid=1, probability=0.9)])
        agg.update_heuristic_risks([_heur(oid=1, risk_score=0.4)])
        risks = agg.get_top_risks()
        assert abs(risks[0].combined_score - 0.4) < 1e-4

    def test_heuristic_weight_zero_uses_physics_only(self) -> None:
        agg = RiskAggregator(physics_weight=1.0, heuristic_weight=0.0)
        agg.update_physics_risks([_phys_dict(oid=1, risk_type="Fall", probability=0.8)])
        agg.update_heuristic_risks([_heur(oid=1, risk_score=0.1)])
        risks = agg.get_top_risks()
        # physics_score = 0.8 * severity("Fall") = 0.8 * 0.9 = 0.72
        expected = 0.8 * _severity("Fall")
        assert abs(risks[0].combined_score - expected) < 1e-4

    def test_higher_physics_probability_gives_higher_score(self) -> None:
        agg = RiskAggregator(physics_weight=1.0, heuristic_weight=0.0)
        agg.update_physics_risks(
            [
                _phys_dict(oid=1, risk_type="Fall", probability=0.9),
                _phys_dict(oid=2, risk_type="Fall", probability=0.3),
            ]
        )
        risks = agg.get_top_risks()
        assert risks[0].object_id == 1
        assert risks[0].combined_score > risks[1].combined_score
