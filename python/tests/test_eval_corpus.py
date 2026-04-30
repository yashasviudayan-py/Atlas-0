"""Tests for seeded ATLAS-0 evaluation corpus definitions."""

from __future__ import annotations

import json
from pathlib import Path

from atlas.api import server

_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "eval_corpus"


def test_eval_corpus_has_broad_seed_coverage() -> None:
    entries = server._load_eval_corpus_entries()
    labels = {entry["label"] for entry in entries}
    audience_modes = {entry.get("audience_mode", "general") for entry in entries}

    assert len(entries) >= 8
    assert {"sample_walkthrough", "toddler_reach_test", "pet_cord_sweep"} <= labels
    assert {"general", "toddler", "pet", "renter"} <= audience_modes


def test_eval_corpus_entries_define_grounding_expectations() -> None:
    for path in sorted(_CORPUS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["label"]
        assert payload["fixture_name"]
        assert payload["scene_source"] in {"estimated_multiview", "single_view_estimate"}
        assert payload["min_object_count"] >= 1
        assert payload["min_hazard_count"] >= 0
        assert isinstance(payload["required_hazard_codes"], list)
        assert payload["top_hazard_code"] or payload["min_hazard_count"] == 0
        assert payload.get("min_multiframe_supported_objects", 0) >= 0
        assert 0.0 <= payload.get("min_avg_grounding_confidence", 0.0) <= 1.0


def test_benchmark_comparison_checks_multiframe_grounding() -> None:
    job = {
        "scene_source": "estimated_multiview",
        "objects": [
            {"multi_frame_support": True, "grounding_confidence": 0.68},
            {"multi_frame_support": True, "grounding_confidence": 0.62},
            {"multi_frame_support": False, "grounding_confidence": 0.51},
        ],
        "summary": {"object_count": 3, "hazard_count": 3},
        "risks": [
            {"hazard_code": "unsupported_tall_item"},
            {"hazard_code": "edge_placement"},
            {"hazard_code": "blocked_path"},
        ],
    }

    comparison = server._compare_job_to_benchmark(job, "toddler_reach_test")

    assert comparison is not None
    assert comparison["matched"] is True
    assert comparison["multiframe_supported_objects"] == 2
    assert comparison["avg_grounding_confidence"] >= 0.55
