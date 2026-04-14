"""Regression tests for the reusable upload analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from atlas.api.upload_analysis import analyze_frame_samples, analyze_image_heuristic
from atlas.utils.video import ExtractedFrame


async def _heuristic_labeler(content: bytes, _hint: str):
    return analyze_image_heuristic(content)


async def test_sample_walkthrough_matches_expected_report() -> None:
    fixture_root = Path(__file__).parent.parent.parent / "data" / "sample_walkthrough"
    expected = json.loads((fixture_root / "expected_report.json").read_text(encoding="utf-8"))

    frames = [
        ExtractedFrame(index=index, timestamp_s=index * 0.6, image_bytes=path.read_bytes())
        for index, path in enumerate(sorted((fixture_root / "frames").glob("*.jpg")))
    ]

    result = await analyze_frame_samples(
        frames,
        filename=expected["fixture_name"],
        source_content_type="image/jpeg",
        labeler=_heuristic_labeler,
    )

    hazard_codes = [risk["hazard_code"] for risk in result.risks]

    assert result.scene_source == expected["scene_source"]
    assert len(result.objects) >= expected["min_object_count"]
    assert len(result.risks) >= expected["min_hazard_count"]
    assert len(result.fix_first) >= 1
    assert result.scan_quality["status"] in {"good", "fair", "poor"}
    assert "scan_quality_label" in result.summary
    assert "coverage_label" in result.summary
    assert "screening_statement" in result.summary
    assert result.summary["report_posture"] in {
        "actionable screening",
        "limited screening",
        "preliminary screening",
    }
    assert any("not a certification" in note for note in result.trust_notes)
    assert hazard_codes[0] == expected["top_hazard_code"]
    for code in expected["required_hazard_codes"]:
        assert code in hazard_codes
