"""Hazard ontology and deterministic reasoning for upload reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

AUDIENCE_MODE_LABELS: dict[str, str] = {
    "general": "General home safety",
    "toddler": "Toddler mode",
    "pet": "Pet mode",
    "renter": "Move-in / renter mode",
}

_AUDIENCE_PRIORITY_BOOSTS: dict[str, dict[str, float]] = {
    "general": {},
    "toddler": {
        "fragile_breakable": 0.12,
        "top_heavy_tipping": 0.14,
        "heavy_elevated": 0.1,
        "edge_placement": 0.14,
        "walkway_clutter": 0.08,
        "liquid_spill": 0.04,
        "unsupported_tall_item": 0.12,
    },
    "pet": {
        "fragile_breakable": 0.1,
        "edge_placement": 0.08,
        "walkway_clutter": 0.16,
        "unstable_stack": 0.08,
        "liquid_spill": 0.14,
    },
    "renter": {
        "top_heavy_tipping": 0.08,
        "heavy_elevated": 0.1,
        "edge_placement": 0.06,
        "walkway_clutter": 0.1,
        "unstable_stack": 0.12,
        "unsupported_tall_item": 0.1,
    },
}

_AUDIENCE_FOCUS_COPY: dict[str, dict[str, str]] = {
    "toddler": {
        "fragile_breakable": (
            "Raised in toddler mode because reachable breakables matter more around "
            "young children."
        ),
        "top_heavy_tipping": (
            "Raised in toddler mode because pull-down and bump risk matter more "
            "around young children."
        ),
        "heavy_elevated": (
            "Raised in toddler mode because heavier overhead items matter more in "
            "child-focused rooms."
        ),
        "edge_placement": (
            "Raised in toddler mode because reachable edges matter more around " "young children."
        ),
        "walkway_clutter": (
            "Raised in toddler mode because floor-level trip hazards matter more "
            "during child movement."
        ),
        "unsupported_tall_item": (
            "Raised in toddler mode because tall unstable items deserve extra "
            "caution in child spaces."
        ),
    },
    "pet": {
        "fragile_breakable": (
            "Raised in pet mode because tail-height and curious-pet bump risks " "matter more here."
        ),
        "edge_placement": (
            "Raised in pet mode because objects near edges are easier for pets to " "bump down."
        ),
        "walkway_clutter": (
            "Raised in pet mode because floor-level clutter matters more in pet " "traffic paths."
        ),
        "liquid_spill": (
            "Raised in pet mode because open containers and spill sources matter "
            "more around pets."
        ),
        "unstable_stack": (
            "Raised in pet mode because unstable stacks can be disturbed more " "easily by pets."
        ),
    },
    "renter": {
        "top_heavy_tipping": (
            "Raised in renter mode because quick, non-invasive stabilizations are "
            "a strong move-in priority."
        ),
        "heavy_elevated": (
            "Raised in renter mode because shelves and elevated storage are "
            "common move-in follow-ups."
        ),
        "walkway_clutter": (
            "Raised in renter mode because circulation issues make a space feel "
            "less settled and usable."
        ),
        "unstable_stack": (
            "Raised in renter mode because temporary move-in stacks deserve early " "cleanup."
        ),
        "unsupported_tall_item": (
            "Raised in renter mode because tall freestanding items often need "
            "early stabilization."
        ),
    },
}


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


def normalize_audience_mode(value: str | None) -> str:
    """Return one supported audience mode, defaulting to general."""
    mode = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return mode if mode in AUDIENCE_MODE_LABELS else "general"


def audience_mode_label(mode: str | None) -> str:
    """Return the user-facing label for one audience mode."""
    return AUDIENCE_MODE_LABELS[normalize_audience_mode(mode)]


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
    audience_mode: str = "general",
    recommendation: str | None = None,
) -> dict[str, Any]:
    """Construct a normalized hazard finding payload."""
    definition = _ONTOLOGY_BY_CODE[code]
    mode = normalize_audience_mode(audience_mode)
    grounded_confidence = float(obj.get("grounding_confidence", obj.get("confidence", 0.4)))
    report_confidence = _clamp((float(obj.get("confidence", 0.4)) + grounded_confidence) / 2.0)
    observation_count = int(obj.get("observation_count", 1))
    frame_span = int(obj.get("frame_span", observation_count) or observation_count)
    evidence_ids = list(obj.get("evidence_ids", []))[:3]
    mode_priority_bonus = float(_AUDIENCE_PRIORITY_BOOSTS.get(mode, {}).get(code, 0.0))
    priority_score = _clamp(score * 0.58 + report_confidence * 0.27 + 0.15 + mode_priority_bonus)
    recommendation_text = recommendation or definition.default_action
    evidence_label = confidence_bucket(report_confidence)
    mode_focus = _AUDIENCE_FOCUS_COPY.get(mode, {}).get(code)

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
        "audience_mode": mode,
        "audience_label": audience_mode_label(mode),
        "mode_priority_bonus": round(mode_priority_bonus, 3),
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
                obj.get("support_summary")
                or f"{observation_count} supporting observation"
                f"{'' if observation_count == 1 else 's'} across the scan"
            ),
            "grounding_confidence": round(grounded_confidence, 2),
            "grounding_confidence_label": confidence_bucket(grounded_confidence),
            "evidence_ids": evidence_ids,
            "localization_method": obj.get("localization_method", "single_frame_estimate"),
            "multi_frame_support": bool(obj.get("multi_frame_support", False)),
            "audience_mode": mode,
            "audience_label": audience_mode_label(mode),
            "mode_focus": mode_focus,
            "object_snapshot": {
                "material": obj.get("material", "unknown"),
                "estimated_height_m": round(float(obj.get("estimated_height_m", 0.0)), 2),
                "estimated_width_m": round(float(obj.get("estimated_width_m", 0.0)), 2),
                "observation_count": observation_count,
                "frame_span": frame_span,
                "position_variance": round(float(obj.get("position_variance", 0.0)), 3),
                "bbox_stability": round(float(obj.get("bbox_stability", 0.0)), 3),
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
        "follow_up_status": None,
        "follow_up_updated_at": None,
        "follow_up_note": None,
    }


def evaluate_upload_hazards(
    objects: list[dict[str, Any]],
    *,
    audience_mode: str = "general",
) -> list[dict[str, Any]]:
    """Evaluate the upload hazard ontology against localized objects."""
    mode = normalize_audience_mode(audience_mode)
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
                    audience_mode=mode,
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
                    audience_mode=mode,
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
                    audience_mode=mode,
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
                    audience_mode=mode,
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
                    audience_mode=mode,
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
                    audience_mode=mode,
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
                    audience_mode=mode,
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
                    audience_mode=mode,
                    signals=[
                        f"height_m={height_m:.2f}",
                        f"observations={observation_count}",
                        f"position_variance={variance:.2f}",
                    ],
                )
            )

    findings.sort(key=lambda entry: float(entry["risk_score"]), reverse=True)
    return findings[:10]


def build_fix_first_actions(
    hazards: list[dict[str, Any]],
    limit: int = 3,
    *,
    audience_mode: str = "general",
) -> list[dict[str, Any]]:
    """Return the top actions a user should take first."""
    mode = normalize_audience_mode(audience_mode)
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
                "audience_mode": mode,
                "audience_label": audience_mode_label(mode),
                "generated_at": generated_at,
            }
        )

    return actions


def build_recommendations_from_hazards(
    hazards: list[dict[str, Any]],
    *,
    audience_mode: str = "general",
) -> list[dict[str, Any]]:
    """Convert hazard findings into action cards for the report UI."""
    mode = normalize_audience_mode(audience_mode)
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
                "audience_mode": mode,
                "audience_label": audience_mode_label(mode),
            }
        )

    return recommendations[:6]


def build_weekend_fix_list(
    hazards: list[dict[str, Any]],
    *,
    audience_mode: str = "general",
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Turn prioritized findings into a short, approachable weekend fix list."""
    mode = normalize_audience_mode(audience_mode)
    effort_by_code = {
        "fragile_breakable": "10 minutes",
        "edge_placement": "10 minutes",
        "walkway_clutter": "15 minutes",
        "liquid_spill": "10 minutes",
        "unstable_stack": "20 minutes",
        "top_heavy_tipping": "30-45 minutes",
        "heavy_elevated": "30-45 minutes",
        "unsupported_tall_item": "30-45 minutes",
    }
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
        key=lambda hazard: float(hazard.get("priority_score", hazard.get("risk_score", 0.0))),
        reverse=True,
    )

    return [
        {
            "title": str(hazard.get("hazard_title", "Weekend fix")),
            "task": str(hazard.get("what_to_do_next", hazard.get("recommendation", ""))),
            "benefit": str(hazard.get("why_it_matters", hazard.get("why", ""))),
            "location": str(hazard.get("location_label", "scan area")),
            "effort": effort_by_code.get(str(hazard.get("hazard_code", "")), "20-30 minutes"),
            "priority_score": round(
                float(hazard.get("priority_score", hazard.get("risk_score", 0.0))),
                3,
            ),
            "audience_mode": mode,
            "audience_label": audience_mode_label(mode),
        }
        for hazard in ranked[:limit]
    ]


def build_room_wins(
    hazards: list[dict[str, Any]],
    scan_quality: dict[str, Any],
    *,
    comparison_summary: dict[str, Any] | None = None,
    audience_mode: str = "general",
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Highlight calm signals without overstating certainty."""
    mode = normalize_audience_mode(audience_mode)
    codes = {str(hazard.get("hazard_code", "")) for hazard in hazards}
    wins: list[dict[str, str]] = []

    if comparison_summary and comparison_summary.get("trend") == "improved":
        wins.append(
            {
                "title": "Safer than the last saved scan",
                "detail": str(comparison_summary.get("summary", "")),
            }
        )

    if float(scan_quality.get("score", 0.0) or 0.0) >= 0.72:
        wins.append(
            {
                "title": "Broad scan coverage",
                "detail": (
                    "This upload gave ATLAS-0 enough view coverage to ground the "
                    "strongest findings more cleanly."
                ),
            }
        )

    if "walkway_clutter" not in codes:
        wins.append(
            {
                "title": (
                    "Floor-level path looked calmer for pets."
                    if mode == "pet"
                    else "Walking path looked mostly open"
                ),
                "detail": (
                    "ATLAS-0 did not surface a strong front-path clutter signal " "in this scan."
                ),
            }
        )

    if {"top_heavy_tipping", "unsupported_tall_item"}.isdisjoint(codes):
        wins.append(
            {
                "title": (
                    "No obvious pull-down hotspot surfaced"
                    if mode == "toddler"
                    else "No obvious tall-item tipping hotspot surfaced"
                ),
                "detail": "No strong tipping or tall unsupported item dominated this scan.",
            }
        )

    if "heavy_elevated" not in codes:
        wins.append(
            {
                "title": (
                    "Shelves looked calmer for a move-in check"
                    if mode == "renter"
                    else "Heavier items mostly stayed low"
                ),
                "detail": "ATLAS-0 did not flag a strong heavy-overhead risk in the current view.",
            }
        )

    if {"edge_placement", "fragile_breakable"}.isdisjoint(codes):
        wins.append(
            {
                "title": (
                    "Reachable surfaces looked calmer for young kids."
                    if mode == "toddler"
                    else "Visible surfaces looked calmer around edges"
                ),
                "detail": (
                    "This scan did not surface a strong edge-placement or "
                    "breakable-at-edge pattern."
                ),
            }
        )

    deduped: list[dict[str, str]] = []
    seen_titles: set[str] = set()
    for win in wins:
        title = str(win.get("title", ""))
        if title and title not in seen_titles:
            seen_titles.add(title)
            deduped.append(win)

    return deduped[:limit]
