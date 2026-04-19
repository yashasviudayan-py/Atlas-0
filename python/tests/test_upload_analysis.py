"""Tests for upload analysis helpers."""

from __future__ import annotations

import asyncio
import io

from atlas.api.upload_analysis import (
    _apply_scan_acceptance_policy,
    _calibrate_risks_for_scan,
    _sanitize_region_crop,
    analyze_frame_samples,
    build_finding_replays,
)
from atlas.utils.config import UploadsConfig
from atlas.utils.video import ExtractedFrame
from atlas.vlm.inference import SemanticLabel
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


def test_apply_scan_acceptance_policy_rejects_weak_walkthrough() -> None:
    scan_quality = {
        "status": "poor",
        "score": 0.31,
        "usable": True,
        "rescan_recommended": True,
        "warnings": [],
        "retry_guidance": [],
        "metrics": {
            "frame_count": 2,
            "motion_coverage": 0.08,
            "saliency_coverage": 0.12,
        },
    }

    risks, fix_first, recommendations = _apply_scan_acceptance_policy(
        scan_quality,
        scan_kind="video",
        risks=[{"hazard_code": "edge_placement"}],
        fix_first=[{"title": "Fix"}],
        recommendations=[{"title": "Rec"}],
    )

    assert risks == []
    assert fix_first == []
    assert recommendations == []
    assert scan_quality["reportability"] == "rejected"
    assert scan_quality["hard_reject"] is True
    assert scan_quality["rejection_reasons"]


def test_apply_scan_acceptance_policy_downgrades_single_image() -> None:
    scan_quality = {
        "status": "good",
        "score": 0.78,
        "usable": True,
        "rescan_recommended": False,
        "warnings": [],
        "retry_guidance": [],
        "metrics": {
            "frame_count": 1,
            "motion_coverage": 1.0,
            "saliency_coverage": 0.66,
        },
    }

    risks, _, _ = _apply_scan_acceptance_policy(
        scan_quality,
        scan_kind="image",
        risks=[{"hazard_code": "edge_placement"}],
        fix_first=[],
        recommendations=[],
    )

    assert risks == [{"hazard_code": "edge_placement"}]
    assert scan_quality["reportability"] == "downgraded"
    assert scan_quality["hard_reject"] is False


def test_analyze_frame_samples_carries_audience_mode_into_summary() -> None:
    buf = io.BytesIO()
    image = Image.new("RGB", (220, 180), color="#f6efe6")
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 28, 130, 148), outline="black", width=6)
    draw.rectangle((142, 18, 196, 164), fill="#70503d")
    image.save(buf, format="JPEG")

    async def fake_labeler(_content: bytes, _hint: str) -> SemanticLabel:
        return SemanticLabel(
            label="Shelf",
            material="Wood",
            mass_kg=4.5,
            fragility=0.26,
            friction=0.64,
            confidence=0.81,
        )

    result = asyncio.run(
        analyze_frame_samples(
            [ExtractedFrame(index=0, timestamp_s=0.0, image_bytes=buf.getvalue())],
            filename="nursery.jpg",
            scan_kind="image",
            labeler=fake_labeler,
            audience_mode="toddler",
        )
    )

    assert result.summary["audience_mode"] == "toddler"
    assert result.summary["audience_label"] == "Toddler mode"
