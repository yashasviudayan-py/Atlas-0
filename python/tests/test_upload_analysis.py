"""Tests for upload analysis helpers."""

from __future__ import annotations

import io

from atlas.api.upload_analysis import build_finding_replays
from PIL import Image


def test_build_finding_replays_generates_gif_for_top_findings() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (18, 18), color="#2c8f84").save(buf, format="JPEG")
    tiny_jpeg = buf.getvalue()
    risks = [
        {
            "hazard_code": "edge_placement",
            "hazard_title": "Object placed near an edge",
            "location_label": "front-right shelf",
            "object_id": "track-01",
            "priority_score": 0.84,
            "risk_score": 0.73,
            "reasoning": {"evidence_ids": ["e-01", "e-02"]},
        }
    ]

    descriptors, artifacts = build_finding_replays(
        risks,
        {"e-01": tiny_jpeg, "e-02": tiny_jpeg},
    )

    assert len(descriptors) == 1
    assert descriptors[0]["replay_id"] == "finding-01"
    assert descriptors[0]["frame_count"] == 2
    assert artifacts["finding-01"][:6] in {b"GIF87a", b"GIF89a"}
