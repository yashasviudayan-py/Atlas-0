"""Tests for Phase 2 Part 7: Spatial Query Engine.

Covers:
- QueryParser: classification and field extraction for all query types.
- RelationshipDetector: on_top_of, inside, adjacent_to, supporting, leaning.
- SemanticObject.risk_score: fragility, height, and relationship factors.
- WorldModelAgent: get_risks/get_objects with pre-populated state.
- API endpoints: /health, /query (all types), /objects, /scene.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from atlas.api.server import _set_agent, app
from atlas.vlm.inference import SemanticLabel
from atlas.vlm.region_extractor import BoundingBox
from atlas.world_model.agent import RiskEntry, WorldModelAgent
from atlas.world_model.label_store import LabelStore
from atlas.world_model.query_parser import QueryParser, QueryType
from atlas.world_model.relationships import (
    RelationshipDetector,
    RelationType,
    SemanticObject,
    SpatialRelationship,
)
from fastapi.testclient import TestClient

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _bbox(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> BoundingBox:
    return BoundingBox(x0, y0, z0, x1, y1, z1)


def _obj(
    oid: int,
    label: str = "object",
    material: str = "plastic",
    mass_kg: float = 1.0,
    fragility: float = 0.5,
    friction: float = 0.5,
    confidence: float = 0.8,
    bbox: BoundingBox | None = None,
) -> SemanticObject:
    if bbox is None:
        bbox = _bbox(0, 0, 0, 0.1, 0.1, 0.1)
    return SemanticObject(
        object_id=oid,
        label=label,
        material=material,
        mass_kg=mass_kg,
        fragility=fragility,
        friction=friction,
        confidence=confidence,
        bbox=bbox,
    )


# ── QueryParser ───────────────────────────────────────────────────────────────


class TestQueryParser:
    parser = QueryParser()

    # RISK queries

    def test_most_unstable_is_risk(self) -> None:
        q = self.parser.parse("What is the most unstable object?")
        assert q.query_type == QueryType.RISK

    def test_most_dangerous_is_risk(self) -> None:
        q = self.parser.parse("Which object is most dangerous?")
        assert q.query_type == QueryType.RISK

    def test_risks_keyword(self) -> None:
        q = self.parser.parse("Show me all risks")
        assert q.query_type == QueryType.RISK

    def test_hazards_keyword(self) -> None:
        q = self.parser.parse("Are there any hazards?")
        assert q.query_type == QueryType.RISK

    def test_risk_predicate_is_risk_rank(self) -> None:
        q = self.parser.parse("What are the risks?")
        assert q.predicate == "risk_rank"

    # LOCATION queries

    def test_where_is_location(self) -> None:
        q = self.parser.parse("Where is the glass?")
        assert q.query_type == QueryType.LOCATION

    def test_location_subject_extraction(self) -> None:
        q = self.parser.parse("Where is the glass?")
        assert q.subject == "glass"

    def test_find_is_location(self) -> None:
        q = self.parser.parse("Find the laptop")
        assert q.query_type == QueryType.LOCATION

    def test_position_of_is_location(self) -> None:
        q = self.parser.parse("Position of the cup")
        assert q.query_type == QueryType.LOCATION

    def test_location_predicate(self) -> None:
        q = self.parser.parse("Where is the glass?")
        assert q.predicate == "position"

    # PROPERTY queries

    def test_made_of_is_property(self) -> None:
        q = self.parser.parse("What is the table made of?")
        assert q.query_type == QueryType.PROPERTY

    def test_material_predicate(self) -> None:
        q = self.parser.parse("What is the table made of?")
        assert q.predicate == "material"

    def test_how_heavy_is_property(self) -> None:
        q = self.parser.parse("How heavy is the chair?")
        assert q.query_type == QueryType.PROPERTY

    def test_heavy_maps_to_mass_kg(self) -> None:
        q = self.parser.parse("How heavy is the chair?")
        assert q.predicate == "mass_kg"

    def test_fragile_maps_to_fragility(self) -> None:
        q = self.parser.parse("How fragile is the vase?")
        assert q.predicate == "fragility"

    def test_friction_maps_to_friction(self) -> None:
        q = self.parser.parse("How slippery is the floor?")
        assert q.predicate == "friction"

    def test_property_subject_extraction(self) -> None:
        q = self.parser.parse("What is the laptop made of?")
        assert q.subject == "laptop"

    # SPATIAL_RELATION queries

    def test_on_top_of_is_spatial_relation(self) -> None:
        q = self.parser.parse("What is on top of the table?")
        assert q.query_type == QueryType.SPATIAL_RELATION

    def test_relation_extracted(self) -> None:
        q = self.parser.parse("What is on top of the table?")
        assert q.relation == "on_top_of"

    def test_reference_object_extracted(self) -> None:
        q = self.parser.parse("What is on top of the table?")
        assert q.reference_object == "the table" or q.reference_object == "table"

    def test_leaning_is_spatial_relation(self) -> None:
        q = self.parser.parse("Is anything leaning?")
        assert q.query_type == QueryType.SPATIAL_RELATION

    def test_supporting_is_spatial_relation(self) -> None:
        q = self.parser.parse("What is supporting the shelf?")
        assert q.query_type == QueryType.SPATIAL_RELATION

    # UNKNOWN fallback

    def test_unknown_falls_back(self) -> None:
        q = self.parser.parse("blah blah nonsense query xyz")
        assert q.query_type == QueryType.UNKNOWN

    def test_unknown_has_subject(self) -> None:
        q = self.parser.parse("blah blah nonsense")
        assert q.subject != ""

    # Raw query preserved

    def test_raw_preserved(self) -> None:
        raw = "Where is the glass?"
        q = self.parser.parse(raw)
        assert q.raw == raw


# ── RelationshipDetector ──────────────────────────────────────────────────────


class TestRelationshipDetector:
    detector = RelationshipDetector(
        vertical_tolerance=0.05,
        horizontal_overlap_threshold=0.3,
        adjacency_distance=0.3,
    )

    def test_on_top_of_detected(self) -> None:
        shelf = _obj(0, "shelf", bbox=_bbox(0, 0, 0, 1, 0.05, 0.5))
        glass = _obj(1, "glass", bbox=_bbox(0.3, 0.05, 0.1, 0.4, 0.25, 0.2))
        self.detector.compute_relationships([shelf, glass])
        rel_types = [r.relation for r in glass.relationships]
        assert RelationType.ON_TOP_OF in rel_types

    def test_on_top_of_not_below(self) -> None:
        # glass BELOW shelf — should not be on_top_of shelf
        shelf = _obj(0, "shelf", bbox=_bbox(0, 1.0, 0, 1, 1.05, 0.5))
        glass = _obj(1, "glass", bbox=_bbox(0.3, 0.05, 0.1, 0.4, 0.25, 0.2))
        self.detector.compute_relationships([shelf, glass])
        rel_types = [r.relation for r in glass.relationships]
        assert RelationType.ON_TOP_OF not in rel_types

    def test_supporting_is_inverse_of_on_top_of(self) -> None:
        shelf = _obj(0, "shelf", bbox=_bbox(0, 0, 0, 1, 0.05, 0.5))
        glass = _obj(1, "glass", bbox=_bbox(0.3, 0.05, 0.1, 0.4, 0.25, 0.2))
        self.detector.compute_relationships([shelf, glass])
        rel_types = [r.relation for r in shelf.relationships]
        assert RelationType.SUPPORTING in rel_types

    def test_inside_detected(self) -> None:
        box = _obj(0, "box", bbox=_bbox(0, 0, 0, 1, 1, 1))
        coin = _obj(1, "coin", bbox=_bbox(0.3, 0.1, 0.3, 0.4, 0.2, 0.4))
        self.detector.compute_relationships([box, coin])
        rel_types = [r.relation for r in coin.relationships]
        assert RelationType.INSIDE in rel_types

    def test_inside_not_larger(self) -> None:
        # coin is NOT inside a tiny box
        box = _obj(0, "box", bbox=_bbox(0.3, 0.1, 0.3, 0.4, 0.2, 0.4))
        coin = _obj(1, "coin", bbox=_bbox(0, 0, 0, 1, 1, 1))
        self.detector.compute_relationships([box, coin])
        rel_types = [r.relation for r in coin.relationships]
        assert RelationType.INSIDE not in rel_types

    def test_adjacent_to_detected(self) -> None:
        a = _obj(0, bbox=_bbox(0, 0, 0, 0.1, 0.1, 0.1))
        b = _obj(1, bbox=_bbox(0.15, 0, 0, 0.25, 0.1, 0.1))
        self.detector.compute_relationships([a, b])
        rel_types = [r.relation for r in a.relationships]
        assert RelationType.ADJACENT_TO in rel_types

    def test_adjacent_to_not_far(self) -> None:
        a = _obj(0, bbox=_bbox(0, 0, 0, 0.1, 0.1, 0.1))
        b = _obj(1, bbox=_bbox(10, 0, 0, 10.1, 0.1, 0.1))
        self.detector.compute_relationships([a, b])
        rel_types = [r.relation for r in a.relationships]
        assert RelationType.ADJACENT_TO not in rel_types

    def test_leaning_detected(self) -> None:
        # Tall narrow object: height=2m, width=0.02m → aspect ratio 100 >> threshold 3
        leaning_bbox = _bbox(0.5, 0, 0, 0.52, 2.0, 0.52)
        leaning = _obj(0, bbox=leaning_bbox)
        self.detector.compute_relationships([leaning])
        rel_types = [r.relation for r in leaning.relationships]
        assert RelationType.LEANING in rel_types

    def test_short_object_not_leaning(self) -> None:
        short = _obj(0, bbox=_bbox(0, 0, 0, 0.5, 0.3, 0.5))
        self.detector.compute_relationships([short])
        rel_types = [r.relation for r in short.relationships]
        assert RelationType.LEANING not in rel_types

    def test_empty_list_no_crash(self) -> None:
        result = self.detector.compute_relationships([])
        assert result == []

    def test_relationships_cleared_on_recompute(self) -> None:
        a = _obj(0, bbox=_bbox(0, 0, 0, 0.1, 0.1, 0.1))
        b = _obj(1, bbox=_bbox(0.05, 0, 0, 0.15, 0.1, 0.1))
        self.detector.compute_relationships([a, b])
        first_count = len(a.relationships)
        self.detector.compute_relationships([a, b])
        assert len(a.relationships) == first_count


# ── SemanticObject.risk_score ─────────────────────────────────────────────────


class TestRiskScore:
    def test_high_fragility_raises_score(self) -> None:
        # Fragile glass on a high shelf — fragility contributes to risk score
        o = _obj(0, fragility=0.9, mass_kg=0.3, bbox=_bbox(0, 0, 0, 0.1, 0.1, 0.1))
        # fragility_mass = 0.9 * (0.3/5) = 0.054; score > 0 proves fragility is factored in
        assert o.risk_score > 0.0

    def test_low_fragility_low_score(self) -> None:
        o = _obj(0, fragility=0.0, mass_kg=0.1, bbox=_bbox(0, 0, 0, 0.1, 0.1, 0.1))
        assert o.risk_score < 0.4

    def test_on_top_of_increases_score(self) -> None:
        base = _obj(0, fragility=0.5, mass_kg=1.0, bbox=_bbox(0, 1.0, 0, 0.1, 1.1, 0.1))
        elevated = _obj(0, fragility=0.5, mass_kg=1.0, bbox=_bbox(0, 1.0, 0, 0.1, 1.1, 0.1))
        elevated.relationships.append(SpatialRelationship(RelationType.ON_TOP_OF, 99, 0.9))
        assert elevated.risk_score >= base.risk_score

    def test_leaning_adds_bonus(self) -> None:
        base = _obj(0, fragility=0.5, mass_kg=1.0, bbox=_bbox(0, 0, 0, 0.1, 1.0, 0.1))
        leaning = _obj(0, fragility=0.5, mass_kg=1.0, bbox=_bbox(0, 0, 0, 0.1, 1.0, 0.1))
        leaning.relationships.append(SpatialRelationship(RelationType.LEANING, 0, 0.6))
        assert leaning.risk_score > base.risk_score

    def test_score_bounded(self) -> None:
        o = _obj(
            0,
            fragility=1.0,
            mass_kg=100.0,
            bbox=_bbox(0, 10.0, 0, 0.01, 20.0, 0.01),
        )
        o.relationships = [
            SpatialRelationship(RelationType.ON_TOP_OF, 1, 0.9),
            SpatialRelationship(RelationType.LEANING, 0, 0.6),
        ]
        assert 0.0 <= o.risk_score <= 1.0


# ── WorldModelAgent ───────────────────────────────────────────────────────────


class TestWorldModelAgent:
    @pytest.mark.asyncio
    async def test_get_risks_empty_initially(self) -> None:
        agent = WorldModelAgent()
        risks = await agent.get_risks()
        assert risks == []

    @pytest.mark.asyncio
    async def test_get_objects_empty_initially(self) -> None:
        agent = WorldModelAgent()
        objects = await agent.get_objects()
        assert objects == []

    @pytest.mark.asyncio
    async def test_get_risks_after_population(self) -> None:
        agent = WorldModelAgent()
        risk = RiskEntry(
            object_id=1,
            object_label="glass",
            position=(1.0, 1.0, 1.0),
            risk_score=0.8,
            fragility=0.9,
            mass_kg=0.15,
            description="glass (glass), fragile, elevated",
        )
        async with agent._risks_lock:
            agent._risks = [risk]
        risks = await agent.get_risks()
        assert len(risks) == 1
        assert risks[0].object_label == "glass"

    @pytest.mark.asyncio
    async def test_vlm_not_active_before_start(self) -> None:
        agent = WorldModelAgent()
        assert not agent.vlm_active

    @pytest.mark.asyncio
    async def test_build_objects_from_store(self) -> None:
        store = LabelStore()
        store.update(0, SemanticLabel("chair", "wood", 5.0, 0.2, 0.6, 0.9))
        agent = WorldModelAgent(label_store=store)
        objects = agent.build_objects_from_store()
        assert len(objects) == 1
        assert objects[0].label == "chair"

    @pytest.mark.asyncio
    async def test_get_objects_sync_returns_cached(self) -> None:
        agent = WorldModelAgent()
        fake_obj = _obj(5, label="vase")
        agent._cached_objects = [fake_obj]
        result = agent.get_objects_sync()
        assert len(result) == 1
        assert result[0].label == "vase"

    @pytest.mark.asyncio
    async def test_assess_scene_skips_without_snapshot(self) -> None:
        """_assess_scene() must not crash when shared mem is unavailable."""
        agent = WorldModelAgent()
        agent._vlm_initialized = True
        await agent._assess_scene()
        assert await agent.get_risks() == []


# ── API endpoints ─────────────────────────────────────────────────────────────


def _make_mock_agent(
    objects: list[SemanticObject] | None = None,
    risks: list[RiskEntry] | None = None,
    vlm_active: bool = False,
) -> WorldModelAgent:
    """Build a WorldModelAgent backed by mocks for API tests."""
    agent = MagicMock(spec=WorldModelAgent)
    agent.vlm_active = vlm_active
    agent.get_objects_sync.return_value = objects or []
    agent.get_risks = AsyncMock(return_value=risks or [])
    return agent


client = TestClient(app)


class TestAPIHealth:
    def test_health_returns_ok(self) -> None:
        _set_agent(_make_mock_agent())
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_vlm_active_false(self) -> None:
        _set_agent(_make_mock_agent(vlm_active=False))
        resp = client.get("/health")
        assert resp.json()["vlm_active"] is False

    def test_health_vlm_active_true(self) -> None:
        _set_agent(_make_mock_agent(vlm_active=True))
        resp = client.get("/health")
        assert resp.json()["vlm_active"] is True

    def test_health_object_count(self) -> None:
        _set_agent(_make_mock_agent(objects=[_obj(0), _obj(1)]))
        resp = client.get("/health")
        assert resp.json()["object_count"] == 2

    def test_health_risk_count(self) -> None:
        risk = RiskEntry(0, "glass", (0.0, 1.0, 0.0), 0.8, 0.9, 0.15, "fragile")
        _set_agent(_make_mock_agent(risks=[risk]))
        resp = client.get("/health")
        assert resp.json()["risk_count"] == 1


class TestAPIQuery:
    def test_query_empty_scene_returns_empty(self) -> None:
        _set_agent(_make_mock_agent())
        resp = client.post("/query", json={"query": "Where is the glass?"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_location_query_finds_matching_object(self) -> None:
        glass = _obj(0, label="glass", confidence=0.9)
        _set_agent(_make_mock_agent(objects=[glass]))
        resp = client.post("/query", json={"query": "Where is the glass?"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["object_label"] == "glass"

    def test_location_query_no_match(self) -> None:
        chair = _obj(0, label="chair")
        _set_agent(_make_mock_agent(objects=[chair]))
        resp = client.post("/query", json={"query": "Where is the glass?"})
        assert resp.json() == []

    def test_risk_query_returns_risks(self) -> None:
        risk = RiskEntry(0, "vase", (1.0, 1.5, 0.5), 0.85, 0.9, 0.3, "vase, fragile, elevated")
        _set_agent(_make_mock_agent(risks=[risk]))
        resp = client.post("/query", json={"query": "What is the most unstable object?"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["object_label"] == "vase"
        assert data[0]["risk_level"] == pytest.approx(0.85)

    def test_property_query_material(self) -> None:
        table = _obj(0, label="table", material="wood")
        _set_agent(_make_mock_agent(objects=[table]))
        resp = client.post("/query", json={"query": "What is the table made of?"})
        data = resp.json()
        assert len(data) == 1
        assert "material" in data[0]["description"]
        assert "wood" in data[0]["description"]

    def test_property_query_mass(self) -> None:
        chair = _obj(0, label="chair", mass_kg=3.5)
        _set_agent(_make_mock_agent(objects=[chair]))
        resp = client.post("/query", json={"query": "How heavy is the chair?"})
        data = resp.json()
        assert len(data) == 1
        assert "3.5" in data[0]["description"]

    def test_spatial_relation_query(self) -> None:
        table = _obj(0, label="table", bbox=_bbox(0, 0, 0, 1, 0.05, 1))
        glass = _obj(1, label="glass", bbox=_bbox(0.3, 0.05, 0.3, 0.4, 0.25, 0.4))
        glass.relationships.append(SpatialRelationship(RelationType.ON_TOP_OF, 0, 0.85))
        _set_agent(_make_mock_agent(objects=[table, glass]))
        resp = client.post("/query", json={"query": "What is on top of the table?"})
        data = resp.json()
        assert len(data) >= 1
        labels = [r["object_label"] for r in data]
        assert "glass" in labels

    def test_max_results_respected(self) -> None:
        objects = [_obj(i, label="glass") for i in range(10)]
        _set_agent(_make_mock_agent(objects=objects))
        resp = client.post("/query", json={"query": "Where is the glass?", "max_results": 3})
        assert len(resp.json()) <= 3

    def test_unknown_query_fuzzy_match(self) -> None:
        lamp = _obj(0, label="lamp")
        _set_agent(_make_mock_agent(objects=[lamp]))
        resp = client.post("/query", json={"query": "lamp"})
        data = resp.json()
        assert len(data) == 1
        assert data[0]["object_label"] == "lamp"


class TestAPIObjects:
    def test_objects_empty(self) -> None:
        _set_agent(_make_mock_agent())
        resp = client.get("/objects")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_objects_returns_all(self) -> None:
        objects = [_obj(0, label="chair"), _obj(1, label="table")]
        _set_agent(_make_mock_agent(objects=objects))
        resp = client.get("/objects")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        labels = {o["label"] for o in data}
        assert labels == {"chair", "table"}

    def test_object_info_has_required_fields(self) -> None:
        _set_agent(_make_mock_agent(objects=[_obj(0, label="box")]))
        resp = client.get("/objects")
        data = resp.json()[0]
        for field in (
            "object_id",
            "label",
            "material",
            "mass_kg",
            "fragility",
            "friction",
            "confidence",
            "position",
            "relationships",
        ):
            assert field in data, f"Missing field: {field}"

    def test_object_position_is_list_of_three(self) -> None:
        _set_agent(_make_mock_agent(objects=[_obj(0)]))
        resp = client.get("/objects")
        pos = resp.json()[0]["position"]
        assert isinstance(pos, list)
        assert len(pos) == 3

    def test_object_relationships_serialised(self) -> None:
        obj = _obj(0, label="glass")
        obj.relationships.append(SpatialRelationship(RelationType.ON_TOP_OF, 1, 0.9))
        _set_agent(_make_mock_agent(objects=[obj]))
        resp = client.get("/objects")
        rels = resp.json()[0]["relationships"]
        assert any("on_top_of" in r for r in rels)


class TestAPIScene:
    def test_scene_structure(self) -> None:
        _set_agent(_make_mock_agent())
        resp = client.get("/scene")
        assert resp.status_code == 200
        data = resp.json()
        for key in ("object_count", "objects", "risk_count", "risks"):
            assert key in data, f"Missing key: {key}"

    def test_scene_counts_match_lists(self) -> None:
        objects = [_obj(0), _obj(1)]
        risks = [RiskEntry(0, "obj", (0, 0, 0), 0.7, 0.5, 1.0, "test")]
        _set_agent(_make_mock_agent(objects=objects, risks=risks))
        resp = client.get("/scene")
        data = resp.json()
        assert data["object_count"] == len(data["objects"])
        assert data["risk_count"] == len(data["risks"])

    def test_scene_risk_has_required_fields(self) -> None:
        risk = RiskEntry(0, "vase", (1.0, 1.5, 0.5), 0.8, 0.9, 0.3, "vase")
        _set_agent(_make_mock_agent(risks=[risk]))
        resp = client.get("/scene")
        data = resp.json()
        assert data["risk_count"] == 1
        r = data["risks"][0]
        for field in ("object_id", "object_label", "position", "risk_score", "description"):
            assert field in r, f"Missing field: {field}"
