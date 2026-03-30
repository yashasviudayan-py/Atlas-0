"""Spatial relationship detection between labeled 3D objects.

Given a list of :class:`SemanticObject` instances (each carrying a
:class:`~atlas.vlm.region_extractor.BoundingBox`), computes pairwise
directed relationships such as *on_top_of*, *inside*, *adjacent_to*,
*supporting*, and *leaning*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import structlog

from atlas.vlm.region_extractor import BoundingBox

logger = structlog.get_logger(__name__)


class RelationType(Enum):
    """Type of directed spatial relationship from object A toward object B."""

    ON_TOP_OF = "on_top_of"
    """A is resting on B's upper surface."""

    INSIDE = "inside"
    """A is spatially contained within B."""

    ADJACENT_TO = "adjacent_to"
    """A and B are within a short distance of each other."""

    SUPPORTING = "supporting"
    """A physically supports B — B would fall if A were removed."""

    LEANING = "leaning"
    """A is tall and narrow with its centre of mass near an edge (self-relation)."""


@dataclass
class SpatialRelationship:
    """A directed relationship from one object to another.

    Args:
        relation: The type of spatial relationship.
        target_object_id: ID of the related object.
        confidence: Score in [0, 1] for this relationship.

    Example::

        rel = SpatialRelationship(RelationType.ON_TOP_OF, target_object_id=7, confidence=0.9)
    """

    relation: RelationType
    target_object_id: int
    confidence: float


@dataclass
class SemanticObject:
    """A labeled 3D object with physical properties and spatial relationships.

    Args:
        object_id: Unique integer identifier (DBSCAN cluster ID).
        label: Human-readable object name (e.g. ``"wine glass"``).
        material: Dominant material (e.g. ``"glass"``).
        mass_kg: Estimated mass in kilograms.
        fragility: Fragility score 0-1 (0 = indestructible, 1 = extremely fragile).
        friction: Surface friction coefficient 0-1.
        confidence: VLM labeling confidence 0-1.
        bbox: Axis-aligned bounding box in world space.
        relationships: Computed spatial relationships; populated by
            :meth:`~RelationshipDetector.compute_relationships`.

    Example::

        from atlas.vlm.region_extractor import BoundingBox
        obj = SemanticObject(
            object_id=0, label="glass", material="glass",
            mass_kg=0.15, fragility=0.9, friction=0.3, confidence=0.85,
            bbox=BoundingBox(0, 0, 0, 0.1, 0.2, 0.1),
        )
        assert obj.position == (0.05, 0.1, 0.05)
    """

    object_id: int
    label: str
    material: str
    mass_kg: float
    fragility: float
    friction: float
    confidence: float
    bbox: BoundingBox
    relationships: list[SpatialRelationship] = field(default_factory=list)

    @property
    def position(self) -> tuple[float, float, float]:
        """Geometric centre of the object's bounding box."""
        return self.bbox.center

    @property
    def risk_score(self) -> float:
        """Heuristic risk score combining fragility, mass, height, and pose.

        Combines:
        - *fragility x clamped-mass* -- fragile heavy objects are riskier.
        - *height factor* — objects higher up are more dangerous if they fall.
        - *elevation count* — each ON_TOP_OF relationship adds risk.
        - *leaning bonus* — leaning objects are scored higher.

        Returns:
            Float in [0, 1] — higher means more at risk.
        """
        # Fragility combined with mass (capped at 5 kg for normalisation)
        fragility_mass = self.fragility * min(1.0, self.mass_kg / 5.0)

        # Height above ground (Y axis; capped at 2 m for normalisation)
        height_factor = min(1.0, max(0.0, self.bbox.center[1]) / 2.0)

        on_top_count = sum(1 for r in self.relationships if r.relation == RelationType.ON_TOP_OF)
        elevation_bonus = min(0.3, on_top_count * 0.15)

        leaning_bonus = (
            0.2 if any(r.relation == RelationType.LEANING for r in self.relationships) else 0.0
        )

        return min(
            1.0,
            fragility_mass * 0.4 + height_factor * 0.3 + elevation_bonus + leaning_bonus,
        )


# ── Detector ──────────────────────────────────────────────────────────────────


class RelationshipDetector:
    """Detects pairwise spatial relationships between labeled scene objects.

    All geometry is computed analytically from bounding boxes — no ML
    inference is required.

    Args:
        vertical_tolerance: Maximum vertical gap (m) between A's bottom and
            B's top for an *on_top_of* relationship.
        horizontal_overlap_threshold: Minimum fraction of A's XZ footprint
            that must overlap B's for *on_top_of* / *supporting*.
        adjacency_distance: Maximum edge-to-edge XZ distance (m) for
            *adjacent_to*.
        containment_margin: Tolerance margin (m) applied when testing whether
            A is fully contained within B.

    Example::

        from atlas.vlm.region_extractor import BoundingBox

        detector = RelationshipDetector()
        shelf = SemanticObject(0, "shelf", "wood", 5.0, 0.1, 0.6, 0.9,
                               BoundingBox(0, 0, 0, 1, 0.05, 0.5))
        glass = SemanticObject(1, "glass", "glass", 0.15, 0.9, 0.3, 0.85,
                               BoundingBox(0.3, 0.05, 0.1, 0.4, 0.25, 0.2))
        result = detector.compute_relationships([shelf, glass])
        labels = [r.relation for r in result[1].relationships]
        assert RelationType.ON_TOP_OF in labels
    """

    def __init__(
        self,
        vertical_tolerance: float = 0.05,
        horizontal_overlap_threshold: float = 0.3,
        adjacency_distance: float = 0.3,
        containment_margin: float = 0.05,
    ) -> None:
        self._vert_tol = vertical_tolerance
        self._horiz_overlap = horizontal_overlap_threshold
        self._adj_dist = adjacency_distance
        self._contain_margin = containment_margin

    def compute_relationships(self, objects: list[SemanticObject]) -> list[SemanticObject]:
        """Compute all pairwise spatial relationships and attach them to objects.

        The :attr:`~SemanticObject.relationships` list of every object is
        cleared and rebuilt in-place.

        Args:
            objects: List of :class:`SemanticObject` instances.

        Returns:
            The same list with :attr:`~SemanticObject.relationships` populated.
        """
        for obj in objects:
            obj.relationships.clear()

        for i, obj_a in enumerate(objects):
            for j, obj_b in enumerate(objects):
                if i == j:
                    continue
                rels = self._relationships_from(obj_a, obj_b)
                obj_a.relationships.extend(rels)

            # Leaning is a unary property checked per object
            if self._is_leaning(obj_a):
                obj_a.relationships.append(
                    SpatialRelationship(
                        relation=RelationType.LEANING,
                        target_object_id=obj_a.object_id,
                        confidence=0.6,
                    )
                )

        total = sum(len(o.relationships) for o in objects)
        logger.debug(
            "relationships_computed",
            object_count=len(objects),
            total_relationships=total,
        )
        return objects

    # ── Pair-level checks ─────────────────────────────────────────────────────

    def _relationships_from(
        self, obj_a: SemanticObject, obj_b: SemanticObject
    ) -> list[SpatialRelationship]:
        """Return relationships directed *from* obj_a *toward* obj_b.

        Args:
            obj_a: Subject object.
            obj_b: Reference object.

        Returns:
            List of :class:`SpatialRelationship` instances (may be empty).
        """
        rels: list[SpatialRelationship] = []
        a, b = obj_a.bbox, obj_b.bbox

        if self._is_on_top_of(a, b):
            rels.append(SpatialRelationship(RelationType.ON_TOP_OF, obj_b.object_id, 0.85))

        if self._is_inside(a, b):
            rels.append(SpatialRelationship(RelationType.INSIDE, obj_b.object_id, 0.85))

        if self._is_adjacent_to(a, b):
            rels.append(SpatialRelationship(RelationType.ADJACENT_TO, obj_b.object_id, 0.9))

        if self._is_supporting(a, b):
            rels.append(SpatialRelationship(RelationType.SUPPORTING, obj_b.object_id, 0.8))

        return rels

    # ── Geometric predicates ──────────────────────────────────────────────────

    def _horizontal_overlap_fraction(self, bbox_a: BoundingBox, bbox_b: BoundingBox) -> float:
        """Fraction of A's XZ footprint area that overlaps with B's.

        Args:
            bbox_a: First bounding box.
            bbox_b: Second bounding box.

        Returns:
            Overlap fraction in [0, 1].  Returns 0.0 when there is no overlap
            or A has zero area.
        """
        overlap_x = max(0.0, min(bbox_a.x_max, bbox_b.x_max) - max(bbox_a.x_min, bbox_b.x_min))
        overlap_z = max(0.0, min(bbox_a.z_max, bbox_b.z_max) - max(bbox_a.z_min, bbox_b.z_min))
        overlap_area = overlap_x * overlap_z
        area_a = (bbox_a.x_max - bbox_a.x_min) * (bbox_a.z_max - bbox_a.z_min)
        if area_a < 1e-9:
            return 0.0
        return overlap_area / area_a

    def _xz_edge_distance(self, bbox_a: BoundingBox, bbox_b: BoundingBox) -> float:
        """Minimum edge-to-edge distance between A and B in the XZ plane.

        Returns 0.0 when the footprints overlap.

        Args:
            bbox_a: First bounding box.
            bbox_b: Second bounding box.

        Returns:
            Non-negative distance in metres.
        """
        dx = max(0.0, max(bbox_a.x_min, bbox_b.x_min) - min(bbox_a.x_max, bbox_b.x_max))
        dz = max(0.0, max(bbox_a.z_min, bbox_b.z_min) - min(bbox_a.z_max, bbox_b.z_max))
        return math.hypot(dx, dz)

    def _is_on_top_of(self, bbox_a: BoundingBox, bbox_b: BoundingBox) -> bool:
        """True when A is resting on top of B.

        Conditions:
        - A's bottom (y_min) is near B's top (y_max) within *vertical_tolerance*.
        - A is above B (A.y_min ≥ B.y_min).
        - A's XZ footprint overlaps B's by at least *horizontal_overlap_threshold*.

        Args:
            bbox_a: Candidate resting object.
            bbox_b: Candidate surface object.
        """
        if bbox_a.y_min < bbox_b.y_min:
            return False
        if abs(bbox_a.y_min - bbox_b.y_max) > self._vert_tol:
            return False
        return self._horizontal_overlap_fraction(bbox_a, bbox_b) >= self._horiz_overlap

    def _is_inside(self, bbox_a: BoundingBox, bbox_b: BoundingBox) -> bool:
        """True when A is fully contained within B (with margin tolerance).

        Args:
            bbox_a: Potential inner object.
            bbox_b: Potential container object.
        """
        m = self._contain_margin
        return (
            bbox_a.x_min >= bbox_b.x_min - m
            and bbox_a.x_max <= bbox_b.x_max + m
            and bbox_a.y_min >= bbox_b.y_min - m
            and bbox_a.y_max <= bbox_b.y_max + m
            and bbox_a.z_min >= bbox_b.z_min - m
            and bbox_a.z_max <= bbox_b.z_max + m
        )

    def _is_adjacent_to(self, bbox_a: BoundingBox, bbox_b: BoundingBox) -> bool:
        """True when A and B are within *adjacency_distance* of each other in XZ.

        Args:
            bbox_a: First bounding box.
            bbox_b: Second bounding box.
        """
        return self._xz_edge_distance(bbox_a, bbox_b) <= self._adj_dist

    def _is_supporting(self, bbox_a: BoundingBox, bbox_b: BoundingBox) -> bool:
        """True when A is below B and B's XZ footprint overlaps A's.

        This is the inverse perspective of *on_top_of*: B is on top of A,
        therefore A is supporting B.

        Args:
            bbox_a: Candidate support surface.
            bbox_b: Candidate supported object.
        """
        if bbox_a.y_max > bbox_b.y_max:
            return False
        if abs(bbox_b.y_min - bbox_a.y_max) > self._vert_tol:
            return False
        return self._horizontal_overlap_fraction(bbox_b, bbox_a) >= self._horiz_overlap

    def _is_leaning(self, obj: SemanticObject, aspect_threshold: float = 3.0) -> bool:
        """True when the object is tall and narrow — a physics proxy for tip risk.

        A bounding box alone cannot reveal the true lean angle.  We use the
        height-to-minimum-footprint-dimension ratio as a proxy: objects with a
        high aspect ratio (e.g. bottles, floor lamps, stacked boxes) have their
        centre of mass well above their support polygon and are more likely to
        tip over when perturbed.

        Args:
            obj: Object to test.
            aspect_threshold: Minimum height / min(width, depth) ratio to be
                considered leaning.  Default 3.0 means three times taller than
                its narrowest horizontal dimension.
        """
        bbox = obj.bbox
        width = bbox.x_max - bbox.x_min
        depth = bbox.z_max - bbox.z_min
        height = bbox.y_max - bbox.y_min

        if width < 1e-6 or depth < 1e-6 or height < 1e-6:
            return False

        return height / min(width, depth) >= aspect_threshold
