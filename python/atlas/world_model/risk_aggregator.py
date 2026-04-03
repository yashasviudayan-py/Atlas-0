"""Risk aggregator: merges physics-engine risks with VLM heuristic risks.

The aggregator maintains two independent risk sources:

- **Physics risks** — ``RiskAssessment`` records produced by the Rust physics
  engine (``atlas_physics::simulator``) and delivered here as plain dicts
  (JSON-deserialized from the risk channel / shared memory).
- **Heuristic risks** — :class:`~atlas.world_model.agent.RiskEntry` objects
  produced by the world-model agent's VLM-based assessment loop.

Both sources are merged by object ID.  When both agree on the same object the
scores are combined via a configurable weighted average.  The final ranked list
is sorted by ``combined_score`` and capped at ``top_n`` entries.

Example::

    aggregator = RiskAggregator(top_n=10)

    # Receive a new batch of physics risks from the Rust engine:
    aggregator.update_physics_risks([
        {"object_id": 1, "risk_type": "Fall",
         "probability": 0.85, "impact_point": None, "description": "may fall"},
    ])

    # Receive the latest heuristic risks from the world-model agent:
    aggregator.update_heuristic_risks([
        RiskEntry(object_id=1, object_label="vase", position=(0.5, 1.5, 0.5),
                  risk_score=0.7, fragility=0.9, mass_kg=0.3,
                  description="vase (glass), fragile, elevated"),
    ])

    top = aggregator.get_top_risks()
    print(top[0].combined_score)  # merged score
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from atlas.world_model.agent import RiskEntry

logger = structlog.get_logger(__name__)

# ─── Severity lookup ──────────────────────────────────────────────────────────

#: Per-type severity multiplier applied to the physics probability score.
_RISK_TYPE_SEVERITY: dict[str, float] = {
    "Fall": 0.9,
    "Spill": 0.6,
    "Collision": 0.8,
    "TripHazard": 0.7,
    "Instability": 0.5,
}


def _severity(risk_type: str) -> float:
    """Return the severity multiplier for *risk_type* (default 0.5)."""
    return _RISK_TYPE_SEVERITY.get(risk_type, 0.5)


# ─── PhysicsRiskEntry ─────────────────────────────────────────────────────────


@dataclass
class PhysicsRiskEntry:
    """A risk assessment from the Rust physics engine.

    Mirrors ``atlas_core::semantic::RiskAssessment`` after JSON
    deserialization.

    Args:
        object_id: Unique object identifier (matches ``SemanticObject.id``).
        risk_type: Category string: ``"Fall"``, ``"Spill"``, ``"Collision"``,
            ``"TripHazard"``, or ``"Instability"``.
        probability: Physics-derived probability in ``[0, 1]``.
        impact_point: Predicted impact position ``(x, y, z)`` or ``None``.
        description: Human-readable summary from the simulator.
    """

    object_id: int
    risk_type: str
    probability: float
    impact_point: tuple[float, float, float] | None
    description: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhysicsRiskEntry:
        """Deserialize from a Rust ``RiskAssessment`` JSON dict.

        Args:
            data: Mapping with keys ``object_id``, ``risk_type``,
                ``probability``, ``impact_point`` (nullable), ``description``.

        Returns:
            A :class:`PhysicsRiskEntry` instance.

        Raises:
            KeyError: If a required field is missing.
            ValueError: If a field cannot be cast to its expected type.
        """
        raw_ip = data.get("impact_point")
        impact_point: tuple[float, float, float] | None = None
        if raw_ip is not None:
            impact_point = (
                float(raw_ip["x"]),
                float(raw_ip["y"]),
                float(raw_ip["z"]),
            )
        return cls(
            object_id=int(data["object_id"]),
            risk_type=str(data.get("risk_type", "Instability")),
            probability=float(data.get("probability", 0.0)),
            impact_point=impact_point,
            description=str(data.get("description", "")),
        )


# ─── AggregatedRisk ───────────────────────────────────────────────────────────


@dataclass
class AggregatedRisk:
    """A merged risk entry combining physics and heuristic assessments.

    Args:
        object_id: Unique object identifier.
        object_label: Human-readable label (from heuristic side when available,
            else ``"object_{object_id}"``).
        physics_score: Physics probability * severity (0 if no physics data).
        heuristic_score: VLM heuristic risk score (0 if no heuristic data).
        combined_score: Final merged score in ``[0, 1]``.
        risk_type: Dominant risk type (from physics engine when available).
        impact_point: Predicted impact location or ``None``.
        description: Pipe-separated descriptions from contributing sources.
        sources: Which sources contributed, e.g. ``["physics", "heuristic"]``.
        position: World-space object centre ``(x, y, z)`` from the heuristic
            source, or ``None`` when only physics data is available.
    """

    object_id: int
    object_label: str
    physics_score: float
    heuristic_score: float
    combined_score: float
    risk_type: str
    impact_point: tuple[float, float, float] | None
    description: str
    sources: list[str]
    position: tuple[float, float, float] | None = None


# ─── RiskAggregator ───────────────────────────────────────────────────────────


class RiskAggregator:
    """Aggregates physics and heuristic risks into a unified ranked list.

    Holds two risk stores (physics and heuristic) and merges them on demand.
    Objects that appear in both stores have their scores combined; objects that
    appear in only one store carry a zero score from the other.

    Ranking formula::

        combined = (physics_weight * physics_score
                    + heuristic_weight * heuristic_score)
                   / (physics_weight + heuristic_weight)

    where ``physics_score = probability * severity(risk_type)``.

    Example::

        agg = RiskAggregator(top_n=5, physics_weight=0.7, heuristic_weight=0.3)
        agg.update_physics_risks([...])
        agg.update_heuristic_risks([...])
        top = agg.get_top_risks()

    Args:
        top_n: Maximum number of entries returned by :meth:`get_top_risks`.
        physics_weight: Weight applied to the physics score component.
        heuristic_weight: Weight applied to the heuristic score component.

    Raises:
        ValueError: If *top_n* < 1 or either weight is outside ``[0, 1]``.
    """

    def __init__(
        self,
        top_n: int = 20,
        physics_weight: float = 0.6,
        heuristic_weight: float = 0.4,
    ) -> None:
        if top_n < 1:
            raise ValueError(f"top_n must be at least 1, got {top_n}")
        if not (0.0 <= physics_weight <= 1.0):
            raise ValueError(f"physics_weight must be in [0, 1], got {physics_weight}")
        if not (0.0 <= heuristic_weight <= 1.0):
            raise ValueError(f"heuristic_weight must be in [0, 1], got {heuristic_weight}")

        self._top_n = top_n
        self._physics_weight = physics_weight
        self._heuristic_weight = heuristic_weight
        self._physics_risks: dict[int, PhysicsRiskEntry] = {}
        self._heuristic_risks: dict[int, RiskEntry] = {}

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def top_n(self) -> int:
        """Maximum number of risks returned by :meth:`get_top_risks`."""
        return self._top_n

    @property
    def physics_count(self) -> int:
        """Number of objects currently tracked via the physics source."""
        return len(self._physics_risks)

    @property
    def heuristic_count(self) -> int:
        """Number of objects currently tracked via the heuristic source."""
        return len(self._heuristic_risks)

    # ── Mutation ──────────────────────────────────────────────────────────────

    def update_physics_risks(self, risks: list[dict[str, Any]]) -> None:
        """Replace the physics risk store with a new batch.

        Invalid entries are skipped with a warning log.

        Args:
            risks: List of dicts matching the ``RiskAssessment`` JSON schema.
        """
        self._physics_risks = {}
        for raw in risks:
            try:
                entry = PhysicsRiskEntry.from_dict(raw)
                self._physics_risks[entry.object_id] = entry
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("invalid_physics_risk_skipped", error=str(exc), data=raw)

    def update_heuristic_risks(self, risks: list[RiskEntry]) -> None:
        """Replace the heuristic risk store with a new batch.

        Args:
            risks: List of :class:`~atlas.world_model.agent.RiskEntry` objects
                from the world-model agent.
        """
        self._heuristic_risks = {r.object_id: r for r in risks}

    def clear(self) -> None:
        """Remove all stored risks from both sources."""
        self._physics_risks.clear()
        self._heuristic_risks.clear()

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_top_risks(self) -> list[AggregatedRisk]:
        """Return the top-``top_n`` aggregated risks, sorted by combined score.

        Merges physics and heuristic risks by object ID.  Objects with no
        corresponding entry in one source receive a score of 0.0 from that
        source.

        Returns:
            List of :class:`AggregatedRisk` sorted descending by
            ``combined_score``, capped at ``top_n`` entries.
        """
        merged = self._merge()
        merged.sort(key=lambda r: r.combined_score, reverse=True)
        return merged[: self._top_n]

    # ── Private ───────────────────────────────────────────────────────────────

    def _merge(self) -> list[AggregatedRisk]:
        all_ids = set(self._physics_risks) | set(self._heuristic_risks)
        result: list[AggregatedRisk] = []

        for oid in all_ids:
            phys = self._physics_risks.get(oid)
            heur = self._heuristic_risks.get(oid)

            physics_score = phys.probability * _severity(phys.risk_type) if phys else 0.0
            heuristic_score = heur.risk_score if heur else 0.0
            combined = self._combine(physics_score, heuristic_score)

            label = heur.object_label if heur else f"object_{oid}"
            risk_type = phys.risk_type if phys else "Instability"
            impact_point = phys.impact_point if phys else None
            position = heur.position if heur else None

            desc_parts: list[str] = []
            if phys and phys.description:
                desc_parts.append(phys.description)
            if heur and heur.description:
                desc_parts.append(heur.description)
            description = " | ".join(desc_parts)

            sources: list[str] = []
            if phys:
                sources.append("physics")
            if heur:
                sources.append("heuristic")

            result.append(
                AggregatedRisk(
                    object_id=oid,
                    object_label=label,
                    physics_score=round(physics_score, 4),
                    heuristic_score=round(heuristic_score, 4),
                    combined_score=round(combined, 4),
                    risk_type=risk_type,
                    impact_point=impact_point,
                    description=description,
                    sources=sources,
                    position=position,
                )
            )

        return result

    def _combine(self, physics_score: float, heuristic_score: float) -> float:
        """Weighted combination of *physics_score* and *heuristic_score*.

        Args:
            physics_score: Pre-computed ``probability * severity`` value.
            heuristic_score: VLM heuristic risk score.

        Returns:
            Combined score in ``[0, 1]``.
        """
        total_weight = self._physics_weight + self._heuristic_weight
        if total_weight < 1e-9:
            return 0.0
        combined = (
            self._physics_weight * physics_score + self._heuristic_weight * heuristic_score
        ) / total_weight
        return min(combined, 1.0)
