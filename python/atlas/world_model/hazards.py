"""Hazard ontology and deterministic reasoning for upload reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


def risk_severity(score: float) -> str:
    """Convert a numeric hazard score into a user-facing severity bucket."""
    if score >= 0.78:
        return "critical"
    if score >= 0.58:
        return "high"
    if score >= 0.35:
        return "moderate"
    return "low"


def confidence_bucket(score: float) -> str:
    """Convert a numeric confidence into a user-facing evidence label."""
    if score >= 0.78:
        return "strong"
    if score >= 0.56:
        return "approximate"
    return "weak"


def severity_rank(label: str) -> int:
    """Return a sortable rank for severity strings."""
    return {
        "critical": 4,
        "high": 3,
        "moderate": 2,
        "low": 1,
    }.get(str(label).lower(), 0)


@dataclass(frozen=True)
class HazardDefinition:
    """One hazard type in the upload report ontology."""

    code: str
    category: str
    title: str
    default_action: str


HAZARD_ONTOLOGY: tuple[HazardDefinition, ...] = (
    HazardDefinition(
        code="fragile_breakable",
        category="breakage",
        title="Fragile breakable item",
        default_action="Move it to a wider, more stable surface away from edges.",
    ),
    HazardDefinition(
        code="top_heavy_tipping",
        category="tipping",
        title="Top-heavy tipping risk",
        default_action="Stabilize or anchor the item and clear the surrounding fall zone.",
    ),
    HazardDefinition(
        code="heavy_elevated",
        category="falling",
        title="Heavy elevated object",
        default_action="Lower the object and keep heavy weight below waist height.",
    ),
    HazardDefinition(
        code="edge_placement",
        category="falling",
        title="Object placed near an edge",
        default_action="Pull it farther back from the edge or relocate it.",
    ),
    HazardDefinition(
        code="walkway_clutter",
        category="trip",
        title="Walkway clutter",
        default_action="Clear the item away from the walking path.",
    ),
    HazardDefinition(
        code="unstable_stack",
        category="stacking",
        title="Unstable stack",
        default_action="Flatten, separate, or secure the stacked items.",
    ),
    HazardDefinition(
        code="liquid_spill",
        category="spill",
        title="Potential spill source",
        default_action="Move the container away from edges and traffic areas.",
    ),
    HazardDefinition(
        code="unsupported_tall_item",
        category="stability",
        title="Tall unsupported item",
        default_action="Add support, widen the base, or place it against a stable surface.",
    ),
)

_ONTOLOGY_BY_CODE = {entry.code: entry for entry in HAZARD_ONTOLOGY}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _format_signal(signal: str) -> str:
    key, _sep, value = signal.partition("=")
    return f"{key.replace('_', ' ')}: {value}" if value else key.replace("_", " ")


def _signal_rule_hits(signals: list[str]) -> list[str]:
    """Translate raw signal keys into a clearer reasoning trace."""
    labels: list[str] = []
    for signal in signals:
        key, _sep, value = signal.partition("=")
        cleaned = key.replace("_", " ")
        if value:
            labels.append(f"{cleaned} triggered at {value}")
        else:
            labels.append(f"{cleaned} triggered")
    return labels


def _build_hazard(
    obj: dict[str, Any],
    code: str,
    score: float,
    *,
    why: str,
    signals: list[str],
    recommendation: str | None = None,
) -> dict[str, Any]:
    """Construct a normalized hazard finding payload."""
    definition = _ONTOLOGY_BY_CODE[code]
    grounded_confidence = float(obj.get("grounding_confidence", obj.get("confidence", 0.4)))
    report_confidence = _clamp((float(obj.get("confidence", 0.4)) + grounded_confidence) / 2.0)
    observation_count = int(obj.get("observation_count", 1))
    evidence_ids = list(obj.get("evidence_ids", []))[:3]
    priority_score = _clamp(score * 0.58 + report_confidence * 0.27 + 0.15)
    recommendation_text = recommendation or definition.default_action
    evidence_label = confidence_bucket(report_confidence)

    return {
        "hazard_code": definition.code,
        "hazard_family": definition.category,
        "hazard_title": definition.title,
        "object_id": obj.get("object_id"),
        "object_label": obj.get("label", "Object"),
        "risk_score": round(_clamp(score), 3),
        "severity": risk_severity(score),
        "location_label": obj.get("location_label", "scan area"),
        "description": (
            f"{definition.title} involving {obj.get('label', 'an object')} "
            f"in the {obj.get('location_label', 'scan area')}."
        ),
        "what": (
            f"{definition.title} involving {obj.get('label', 'an object')} "
            f"in the {obj.get('location_label', 'scan area')}."
        ),
        "why": why,
        "why_it_matters": why,
        "recommendation": recommendation_text,
        "what_to_do_next": recommendation_text,
        "confidence": round(report_confidence, 2),
        "confidence_label": evidence_label,
        "priority_score": round(priority_score, 3),
        "evidence": {
            "observation_count": observation_count,
            "grounding_confidence": round(grounded_confidence, 2),
            "signals": signals,
            "evidence_ids": evidence_ids,
        },
        "reasoning": {
            "signals": [_format_signal(signal) for signal in signals],
            "rule_hits": _signal_rule_hits(signals),
            "support_summary": (
                f"{observation_count} supporting observation"
                f"{'' if observation_count == 1 else 's'} across the scan"
            ),
            "grounding_confidence": round(grounded_confidence, 2),
            "grounding_confidence_label": confidence_bucket(grounded_confidence),
            "evidence_ids": evidence_ids,
            "object_snapshot": {
                "material": obj.get("material", "unknown"),
                "estimated_height_m": round(float(obj.get("estimated_height_m", 0.0)), 2),
                "estimated_width_m": round(float(obj.get("estimated_width_m", 0.0)), 2),
                "observation_count": observation_count,
                "location_label": obj.get("location_label", "scan area"),
                "text_redacted_observation_count": int(
                    obj.get("text_redacted_observation_count", 0) or 0
                ),
            },
        },
        "feedback_summary": {
            "useful": 0,
            "wrong": 0,
            "duplicate": 0,
        },
        "latest_feedback": None,
    }


def evaluate_upload_hazards(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate the upload hazard ontology against localized objects."""
    findings: list[dict[str, Any]] = []

    for obj in objects:
        label = str(obj.get("label", "")).lower()
        material = str(obj.get("material", "")).lower()
        fragility = float(obj.get("fragility", 0.0))
        mass_kg = float(obj.get("mass_kg", 0.0))
        height_m = float(obj.get("estimated_height_m", 0.0))
        width_m = float(obj.get("estimated_width_m", 0.0))
        edge_proximity = float(obj.get("edge_proximity", 0.0))
        path_clutter = float(obj.get("path_clutter_score", 0.0))
        observation_count = int(obj.get("observation_count", 1))
        variance = float(obj.get("position_variance", 0.0))
        slenderness = height_m / max(width_m, 0.2)
        front_zone = "front" in str(obj.get("location_label", "")).lower()

        if fragility >= 0.72 or material in {"glass", "ceramic"}:
            score = _clamp(0.52 + fragility * 0.35 + edge_proximity * 0.2)
            findings.append(
                _build_hazard(
                    obj,
                    "fragile_breakable",
                    score,
                    why=(
                        f"{obj.get('label', 'This item')} appears fragile and likely to break if it"
                        " tips, slips, or is bumped."
                    ),
                    signals=[
                        f"fragility={fragility:.2f}",
                        f"material={material or 'unknown'}",
                        f"edge_proximity={edge_proximity:.2f}",
                    ],
                )
            )

        if slenderness >= 1.8 and (mass_kg >= 2.0 or "lamp" in label or "shelf" in label):
            score = _clamp(0.46 + min(0.3, slenderness / 5.0) + min(0.2, variance))
            findings.append(
                _build_hazard(
                    obj,
                    "top_heavy_tipping",
                    score,
                    why=(
                        f"{obj.get('label', 'This item')} looks tall relative to its base, which"
                        " increases tipping risk."
                    ),
                    signals=[
                        f"slenderness={slenderness:.2f}",
                        f"mass_kg={mass_kg:.1f}",
                        f"position_variance={variance:.2f}",
                    ],
                )
            )

        if mass_kg >= 5.0 and float(obj.get("position", [0.0, 0.0, 0.0])[1]) >= 1.0:
            score = _clamp(0.42 + min(0.3, mass_kg / 15.0) + min(0.2, height_m / 3.0))
            findings.append(
                _build_hazard(
                    obj,
                    "heavy_elevated",
                    score,
                    why=(
                        "A heavier object above floor level creates more impact risk if it shifts"
                        " or falls."
                    ),
                    signals=[
                        f"mass_kg={mass_kg:.1f}",
                        f"height_m={height_m:.2f}",
                    ],
                )
            )

        if edge_proximity >= 0.68:
            score = _clamp(0.38 + edge_proximity * 0.42)
            findings.append(
                _build_hazard(
                    obj,
                    "edge_placement",
                    score,
                    why=(
                        "The object repeatedly appears close to the visible"
                        " edge of its support area."
                    ),
                    signals=[
                        f"edge_proximity={edge_proximity:.2f}",
                        f"observations={observation_count}",
                    ],
                )
            )

        if front_zone and path_clutter >= 0.55:
            score = _clamp(0.34 + path_clutter * 0.4 + min(0.12, width_m / 3.0))
            findings.append(
                _build_hazard(
                    obj,
                    "walkway_clutter",
                    score,
                    why=(
                        "The object appears in a front/traffic zone and may"
                        " interrupt the walking path."
                    ),
                    signals=[
                        f"path_clutter_score={path_clutter:.2f}",
                        f"width_m={width_m:.2f}",
                    ],
                )
            )

        if "stack" in label or ("book" in label and slenderness >= 1.2):
            score = _clamp(0.36 + min(0.24, variance * 1.6) + min(0.18, fragility * 0.2))
            findings.append(
                _build_hazard(
                    obj,
                    "unstable_stack",
                    score,
                    why=(
                        "Stacked or leaning items are more likely to slide or"
                        " topple when disturbed."
                    ),
                    signals=[
                        f"position_variance={variance:.2f}",
                        f"slenderness={slenderness:.2f}",
                    ],
                )
            )

        if any(word in label for word in ("cup", "mug", "bottle", "vase", "glass")):
            score = _clamp(0.3 + fragility * 0.28 + edge_proximity * 0.22)
            findings.append(
                _build_hazard(
                    obj,
                    "liquid_spill",
                    score,
                    why=(
                        "Containers and vessels create spill risk when they"
                        " sit near edges or active areas."
                    ),
                    signals=[
                        f"fragility={fragility:.2f}",
                        f"edge_proximity={edge_proximity:.2f}",
                    ],
                )
            )

        if height_m >= 1.15 and observation_count >= 2 and variance >= 0.08:
            score = _clamp(0.35 + min(0.28, height_m / 4.0) + min(0.22, variance))
            findings.append(
                _build_hazard(
                    obj,
                    "unsupported_tall_item",
                    score,
                    why=(
                        "A tall item observed from multiple viewpoints with"
                        " unstable placement deserves follow-up."
                    ),
                    signals=[
                        f"height_m={height_m:.2f}",
                        f"observations={observation_count}",
                        f"position_variance={variance:.2f}",
                    ],
                )
            )

    findings.sort(key=lambda entry: float(entry["risk_score"]), reverse=True)
    return findings[:10]


def build_fix_first_actions(hazards: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    """Return the top actions a user should take first."""
    best_by_object: dict[str, dict[str, Any]] = {}

    for hazard in hazards:
        object_key = str(
            hazard.get("object_id") or hazard.get("object_label") or hazard.get("hazard_code")
        )
        existing = best_by_object.get(object_key)
        current_score = float(hazard.get("priority_score", hazard.get("risk_score", 0.0)))
        if existing is None or current_score > float(
            existing.get("priority_score", existing.get("risk_score", 0.0))
        ):
            best_by_object[object_key] = hazard

    ranked = sorted(
        best_by_object.values(),
        key=lambda hazard: (
            float(hazard.get("priority_score", hazard.get("risk_score", 0.0))),
            severity_rank(str(hazard.get("severity", "low"))),
        ),
        reverse=True,
    )

    actions: list[dict[str, Any]] = []
    generated_at = datetime.now(UTC).isoformat()
    for hazard in ranked[:limit]:
        actions.append(
            {
                "title": str(hazard.get("hazard_title", "Fix first")),
                "action": str(hazard.get("what_to_do_next", hazard.get("recommendation", ""))),
                "why": str(hazard.get("why_it_matters", hazard.get("why", ""))),
                "location": str(hazard.get("location_label", "scan area")),
                "severity": str(hazard.get("severity", "low")),
                "confidence": float(hazard.get("confidence", 0.0)),
                "confidence_label": str(hazard.get("confidence_label", "weak")),
                "hazard_code": str(hazard.get("hazard_code", "")),
                "object_id": str(hazard.get("object_id", "")),
                "priority_score": round(
                    float(hazard.get("priority_score", hazard.get("risk_score", 0.0))),
                    3,
                ),
                "generated_at": generated_at,
            }
        )

    return actions


def build_recommendations_from_hazards(hazards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert hazard findings into action cards for the report UI."""
    recommendations: list[dict[str, Any]] = []
    best_by_object: dict[str, dict[str, Any]] = {}

    for hazard in hazards:
        object_key = str(
            hazard.get("object_id") or hazard.get("object_label") or hazard.get("hazard_code")
        )
        current_score = float(hazard.get("priority_score", hazard.get("risk_score", 0.0)))
        existing = best_by_object.get(object_key)
        if existing is None or current_score > float(
            existing.get("priority_score", existing.get("risk_score", 0.0))
        ):
            best_by_object[object_key] = hazard

    ranked = sorted(
        best_by_object.values(),
        key=lambda hazard: (
            float(hazard.get("priority_score", hazard.get("risk_score", 0.0))),
            severity_rank(str(hazard.get("severity", "low"))),
        ),
        reverse=True,
    )

    for hazard in ranked:
        recommendations.append(
            {
                "title": str(hazard.get("hazard_title", "Recommendation")),
                "priority": str(hazard.get("severity", "low")),
                "location": str(hazard.get("location_label", "scan area")),
                "action": str(hazard.get("what_to_do_next", hazard.get("recommendation", ""))),
                "why": str(hazard.get("why_it_matters", hazard.get("why", ""))),
                "hazard_code": str(hazard.get("hazard_code", "")),
                "confidence_label": str(hazard.get("confidence_label", "weak")),
                "priority_score": round(
                    float(hazard.get("priority_score", hazard.get("risk_score", 0.0))),
                    3,
                ),
            }
        )

    return recommendations[:6]
