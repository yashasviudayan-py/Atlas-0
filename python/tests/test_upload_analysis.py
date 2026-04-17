"""Tests for upload analysis helpers."""

from __future__ import annotations

import io

from atlas.api.upload_analysis import (
    _calibrate_risks_for_scan,
    _sanitize_region_crop,
    build_finding_replays,
)
from atlas.utils.config import UploadsConfig
from PIL import Image, ImageDraw


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


def test_sanitize_region_crop_blurs_text_heavy_images() -> None:
    image = Image.new("RGB", (260, 160), color="white")
    draw = ImageDraw.Draw(image)
    for row, y in enumerate(range(18, 138, 12)):
        for col, x in enumerate(range(18, 238, 18)):
            if (row + col) % 2 == 0:
                draw.rectangle((x, y, x + 10, y + 4), fill="black")

    buf = io.BytesIO()
    image.save(buf, format="JPEG")

    sanitized, safety = _sanitize_region_crop(
        buf.getvalue(),
        area_ratio=0.18,
        aspect_ratio=1.62,
        frame_redactions=0,
        upload_cfg=UploadsConfig(text_density_threshold=0.0),
    )

    assert safety["text_heavy"] is True
    assert safety["redacted"] is True
    assert sanitized != buf.getvalue()


def test_calibrate_risks_for_scan_downgrades_weak_support() -> None:
    risks = [
        {
            "object_id": "track-01",
            "confidence": 0.74,
            "confidence_label": "strong",
            "reasoning": {
                "grounding_confidence": 0.48,
                "object_snapshot": {"observation_count": 1},
            },
        }
    ]
    objects = [
        {
            "object_id": "track-01",
            "observation_count": 1,
            "text_redacted_observation_count": 1,
        }
    ]
    scan_quality = {
        "status": "poor",
        "score": 0.31,
        "metrics": {
            "frame_count": 1,
            "motion_coverage": 0.12,
            "saliency_coverage": 0.26,
        },
    }

    _calibrate_risks_for_scan(risks, objects, scan_quality)

    assert risks[0]["confidence"] < 0.74
    assert risks[0]["confidence_label"] in {"approximate", "weak"}
    assert risks[0]["reasoning"]["confidence_reasons"]
