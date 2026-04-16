"""Reusable upload analysis pipeline for images and walkthrough frames."""

from __future__ import annotations

import io
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from atlas.utils.video import ExtractedFrame, extract_frame_samples
from atlas.vlm.inference import SemanticLabel
from atlas.world_model.hazards import (
    build_fix_first_actions,
    build_recommendations_from_hazards,
    confidence_bucket,
    evaluate_upload_hazards,
)

Labeler = Callable[[bytes, str], Awaitable[SemanticLabel]]


@dataclass(frozen=True)
class RegionCandidate:
    """One salient image region proposed for labeling/tracking."""

    bbox_norm: tuple[float, float, float, float]
    crop_bytes: bytes
    mean_rgb: tuple[float, float, float]
    area_ratio: float
    aspect_ratio: float
    edge_proximity: float
    path_clutter_score: float


@dataclass
class Observation:
    """A labeled observation of one region in one frame."""

    observation_id: str
    frame_index: int
    timestamp_s: float
    label: SemanticLabel
    bbox_norm: tuple[float, float, float, float]
    area_ratio: float
    aspect_ratio: float
    edge_proximity: float
    path_clutter_score: float
    mean_rgb: tuple[float, float, float]
    estimated_position: tuple[float, float, float]
    crop_bytes: bytes


@dataclass
class UploadAnalysisResult:
    """Normalized upload report payload consumed by the API/frontend."""

    objects: list[dict[str, Any]]
    risks: list[dict[str, Any]]
    fix_first: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    evidence_frames: list[dict[str, Any]]
    evidence_artifacts: dict[str, bytes]
    scan_quality: dict[str, Any]
    trust_notes: list[str]
    summary: dict[str, Any]
    scene_source: str
    point_cloud: list[list[float]]


def build_finding_replays(
    risks: list[dict[str, Any]],
    evidence_artifacts: dict[str, bytes],
    *,
    max_replays: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Build short GIF replays for the highest-priority findings."""
    descriptors: list[dict[str, Any]] = []
    replay_artifacts: dict[str, bytes] = {}

    ranked_risks = sorted(
        risks,
        key=lambda risk: (
            float(risk.get("priority_score", 0.0)),
            float(risk.get("risk_score", 0.0)),
        ),
        reverse=True,
    )
    for index, risk in enumerate(ranked_risks[:max_replays], start=1):
        evidence_ids = list(risk.get("reasoning", {}).get("evidence_ids") or [])
        frame_bytes = [
            evidence_artifacts[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in evidence_artifacts
        ]
        if not frame_bytes:
            continue

        replay_id = f"finding-{index:02d}"
        replay_artifacts[replay_id] = _build_replay_gif(
            frame_bytes,
            title=str(risk.get("hazard_title", "Finding")),
            subtitle=str(risk.get("location_label", "scan area")),
        )
        descriptors.append(
            {
                "replay_id": replay_id,
                "hazard_code": risk.get("hazard_code"),
                "object_id": risk.get("object_id"),
                "caption": f"{risk.get('hazard_title', 'Finding')} replay",
                "frame_count": len(frame_bytes),
                "media_type": "image/gif",
                "image_url": None,
            }
        )

    return descriptors, replay_artifacts


def analyze_image_heuristic(content: bytes) -> SemanticLabel:
    """Derive a semantic label from image pixels when the VLM is unavailable."""
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img.thumbnail((128, 128))
        arr = np.array(img, dtype=np.float32)

        mean_r = float(arr[:, :, 0].mean())
        mean_g = float(arr[:, :, 1].mean())
        mean_b = float(arr[:, :, 2].mean())
        brightness = (mean_r + mean_g + mean_b) / 3.0 / 255.0

        mx = max(mean_r, mean_g, mean_b)
        mn = min(mean_r, mean_g, mean_b)
        saturation = (mx - mn) / (mx + 1.0)
        gray = arr.mean(axis=2)
        texture = float(gray.std()) / 255.0

        if mx == mn:
            hue_bucket = -1
        elif mx == mean_r:
            hue_bucket = 0
        elif mx == mean_g:
            hue_bucket = 2
        else:
            hue_bucket = 4

        if saturation < 0.12:
            if brightness > 0.7:
                label, material = "Polished Surface", "Metal"
                mass_kg, fragility, friction = 4.0, 0.25, 0.40
            elif brightness > 0.35:
                label, material = "Stone Object", "Stone"
                mass_kg, fragility, friction = 8.0, 0.18, 0.60
            else:
                label, material = "Dark Object", "Carbon"
                mass_kg, fragility, friction = 2.0, 0.30, 0.45
        elif hue_bucket == 0 and brightness < 0.55:
            label, material = "Wooden Object", "Wood"
            mass_kg, fragility, friction = 6.0, 0.20, 0.65
        elif hue_bucket == 0 and brightness >= 0.55:
            label, material = "Ceramic Object", "Ceramic"
            mass_kg, fragility, friction = 1.2, 0.72, 0.55
        elif hue_bucket == 2:
            label, material = "Organic Object", "Plant"
            mass_kg, fragility, friction = 0.8, 0.55, 0.40
        elif hue_bucket == 4 and brightness > 0.55:
            label, material = "Glass Object", "Glass"
            mass_kg, fragility, friction = 0.6, 0.90, 0.20
        elif brightness > 0.75:
            label, material = "Plastic Object", "Plastic"
            mass_kg, fragility, friction = 0.5, 0.40, 0.50
        else:
            label, material = "Composite Object", "Mixed"
            mass_kg, fragility, friction = 2.0, 0.50, 0.50

        fragility = min(1.0, fragility + texture * 0.2)
        confidence = min(0.82, 0.45 + saturation * 0.3 + (1.0 - abs(brightness - 0.5)) * 0.15)
        return SemanticLabel(
            label=label,
            material=material,
            mass_kg=round(mass_kg, 2),
            fragility=round(fragility, 2),
            friction=round(friction, 2),
            confidence=round(confidence, 2),
        )
    except Exception:
        return SemanticLabel(
            label="Unknown Object",
            material="Unknown",
            mass_kg=1.0,
            fragility=0.5,
            friction=0.5,
            confidence=0.30,
        )


def generate_depth_pointcloud(content: bytes, n_points: int = 320) -> list[list[float]]:
    """Sample a pseudo-depth point cloud from image luminance."""
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        max_side = 160
        w, h = img.size
        scale = min(max_side / max(w, 1), max_side / max(h, 1))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img_small = img.resize((nw, nh), Image.LANCZOS)
        arr = np.array(img_small, dtype=np.float32)

        r_ch, g_ch, b_ch = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        lum = (0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch) / 255.0
        gy = np.gradient(lum, axis=0)
        gx = np.gradient(lum, axis=1)
        weight = np.sqrt(gx**2 + gy**2) + 0.08
        probs = weight.flatten()
        probs = probs / probs.sum()

        n_actual = min(n_points, nw * nh)
        indices = np.random.choice(nw * nh, size=n_actual, replace=False, p=probs)

        rows: list[list[float]] = []
        for idx in indices:
            iy, ix = divmod(int(idx), nw)
            u = (ix / max(nw - 1, 1)) * 2.0 - 1.0
            v = 1.0 - (iy / max(nh - 1, 1)) * 2.0
            depth_01 = 1.0 - float(lum[iy, ix])

            rows.append(
                [
                    round(float(u) * 1.6, 3),
                    round(float(v) * 0.9 + 0.9, 3),
                    round(float(depth_01) * 2.4, 3),
                    round(float(r_ch[iy, ix]) / 255.0, 3),
                    round(float(g_ch[iy, ix]) / 255.0, 3),
                    round(float(b_ch[iy, ix]) / 255.0, 3),
                ]
            )

        return rows
    except Exception:
        return []


def point_cloud_centroid(points: list[list[float]]) -> tuple[float, float, float]:
    """Return the centroid of an upload point cloud."""
    if not points:
        return (0.0, 0.8, 1.5)
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    zs = [float(p[2]) for p in points]
    return (
        round(sum(xs) / len(xs), 3),
        round(sum(ys) / len(ys), 3),
        round(sum(zs) / len(zs), 3),
    )


def location_label(position: tuple[float, float, float]) -> str:
    """Describe an approximate room zone from a 3D position."""
    x, _y, z = position
    horizontal = "center"
    depth = "middle"

    if x < -0.8:
        horizontal = "left"
    elif x > 0.8:
        horizontal = "right"

    if z < 1.0:
        depth = "front"
    elif z > 2.4:
        depth = "back"

    if horizontal == "center" and depth == "middle":
        return "center area"
    return f"{depth}-{horizontal}".replace("-center", "")


def build_trust_notes(scene_source: str, scan_quality: dict[str, Any] | None = None) -> list[str]:
    """Return explicit honesty notes for the current grounding mode."""
    notes: list[str]
    if scene_source == "estimated_multiview":
        notes = [
            (
                "Locations are estimated from repeated frame observations,"
                " not survey-grade reconstruction."
            ),
            (
                "Use hazard evidence and recommendations as the primary"
                " output when positions look approximate."
            ),
            (
                "Objects seen in more frames have better grounding confidence"
                " than one-off detections."
            ),
        ]
    elif scene_source == "single_view_estimate":
        notes = [
            (
                "This report is based on a single uploaded image, so location"
                " and depth are estimated from one view."
            ),
            (
                "Treat object positions as approximate directional hints"
                " rather than precise measurements."
            ),
        ]
    else:
        notes = ["This report is based on heuristic scene estimates."]

    notes.append(
        "ATLAS-0 is a screening tool for likely room hazards,"
        " not a certification that the room is safe."
    )

    if scan_quality and scan_quality.get("warnings"):
        notes.append(
            "Capture quality limited some conclusions. Review the scan quality"
            " panel before treating weak findings as high confidence."
        )
    return notes


async def analyze_uploaded_image(
    content: bytes,
    *,
    filename: str,
    content_type: str,
    labeler: Labeler,
) -> UploadAnalysisResult:
    """Analyze a single uploaded image via the shared frame pipeline."""
    frame = ExtractedFrame(index=0, timestamp_s=0.0, image_bytes=content)
    return await analyze_frame_samples(
        [frame],
        filename=filename,
        source_content_type=content_type or "image/jpeg",
        labeler=labeler,
    )


async def analyze_uploaded_video(
    content: bytes,
    *,
    filename: str,
    labeler: Labeler,
    max_frames: int = 6,
) -> UploadAnalysisResult:
    """Analyze a video upload by sampling frames and localizing tracked regions."""
    frame_samples = extract_frame_samples(content, max_frames=max_frames)
    if not frame_samples:
        raise ValueError(
            "Could not extract frames from the video file. Ensure it is a"
            " valid MP4, MOV, WEBM, or AVI."
        )
    return await analyze_frame_samples(
        frame_samples,
        filename=filename,
        source_content_type="image/jpeg",
        labeler=labeler,
    )


async def analyze_frame_samples(
    frame_samples: list[ExtractedFrame],
    *,
    filename: str,
    source_content_type: str,
    labeler: Labeler,
) -> UploadAnalysisResult:
    """Analyze pre-sampled walkthrough frames into a localized hazard report."""
    if not frame_samples:
        raise ValueError("At least one frame sample is required.")

    _ = source_content_type

    decoded_frames = [_decode_rgb(sample.image_bytes) for sample in frame_samples]
    camera_positions = _estimate_camera_path(decoded_frames)
    regions_per_frame = [_extract_salient_regions(frame_arr) for frame_arr in decoded_frames]
    scan_quality = _assess_scan_quality(decoded_frames, camera_positions, regions_per_frame)

    observations: list[Observation] = []
    evidence_frames: list[dict[str, Any]] = []
    evidence_artifacts: dict[str, bytes] = {}

    for sample, _frame_arr, camera_position, frame_regions in zip(
        frame_samples,
        decoded_frames,
        camera_positions,
        regions_per_frame,
        strict=True,
    ):
        for region_idx, region in enumerate(frame_regions, start=1):
            hint = f"frame {sample.index + 1} region {region_idx}"
            label = await labeler(region.crop_bytes, hint)

            observation_id = f"f{sample.index:02d}-r{region_idx:02d}"
            position = _estimate_observation_position(region, camera_position)
            observations.append(
                Observation(
                    observation_id=observation_id,
                    frame_index=sample.index,
                    timestamp_s=sample.timestamp_s,
                    label=label,
                    bbox_norm=region.bbox_norm,
                    area_ratio=region.area_ratio,
                    aspect_ratio=region.aspect_ratio,
                    edge_proximity=region.edge_proximity,
                    path_clutter_score=region.path_clutter_score,
                    mean_rgb=region.mean_rgb,
                    estimated_position=position,
                    crop_bytes=region.crop_bytes,
                )
            )
            evidence_artifacts[observation_id] = region.crop_bytes
            if len(evidence_frames) < 6:
                evidence_frames.append(
                    {
                        "evidence_id": observation_id,
                        "caption": f"{label.label} observation",
                        "kind": "region_crop",
                        "confidence": round(label.confidence, 2),
                        "image_url": None,
                        "frame_index": sample.index,
                        "timestamp_s": round(sample.timestamp_s, 2),
                        "object_label": label.label,
                        "media_type": "image/jpeg",
                    }
                )

    tracks = _track_observations(observations)
    objects = _build_objects_from_tracks(tracks)
    scene_source = "estimated_multiview" if len(frame_samples) > 1 else "single_view_estimate"
    risks = evaluate_upload_hazards(objects)
    fix_first = build_fix_first_actions(risks)
    recommendations = build_recommendations_from_hazards(risks)
    summary = _build_summary(filename, objects, risks, scene_source, scan_quality)
    point_cloud = _build_scene_point_cloud(tracks)

    return UploadAnalysisResult(
        objects=objects,
        risks=risks,
        fix_first=fix_first,
        recommendations=recommendations,
        evidence_frames=evidence_frames,
        evidence_artifacts=evidence_artifacts,
        scan_quality=scan_quality,
        trust_notes=build_trust_notes(scene_source, scan_quality),
        summary=summary,
        scene_source=scene_source,
        point_cloud=point_cloud,
    )


def _decode_rgb(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image, dtype=np.uint8)


def _extract_salient_regions(frame_arr: np.ndarray, max_regions: int = 3) -> list[RegionCandidate]:
    """Propose a few salient regions from one frame."""
    image = Image.fromarray(frame_arr, mode="RGB")
    orig_h, orig_w = frame_arr.shape[0], frame_arr.shape[1]

    scale = 192 / max(orig_w, orig_h)
    scaled_w = max(48, int(orig_w * scale))
    scaled_h = max(48, int(orig_h * scale))
    small = image.resize((scaled_w, scaled_h), Image.LANCZOS)
    arr = np.array(small, dtype=np.float32)

    lum = arr.mean(axis=2) / 255.0
    sat = (arr.max(axis=2) - arr.min(axis=2)) / np.maximum(arr.max(axis=2), 1.0)
    border_pixels = np.concatenate(
        [
            arr[0, :, :],
            arr[-1, :, :],
            arr[:, 0, :],
            arr[:, -1, :],
        ],
        axis=0,
    )
    background_color = np.median(border_pixels, axis=0)
    color_distance = np.linalg.norm(arr - background_color, axis=2) / 255.0
    grad_x = np.abs(np.gradient(lum, axis=1))
    grad_y = np.abs(np.gradient(lum, axis=0))
    saliency = color_distance * 1.15 + grad_x * 0.7 + grad_y * 0.7 + sat * 0.55

    threshold = float(np.quantile(saliency, 0.78))
    mask = saliency >= max(threshold, 0.12)
    mask &= color_distance >= max(float(np.quantile(color_distance, 0.72)), 0.08)
    components = _connected_components(mask)

    candidates: list[RegionCandidate] = []
    total_area = float(mask.shape[0] * mask.shape[1])
    for rows, cols in components:
        area = len(rows)
        area_ratio = area / max(total_area, 1.0)
        if area_ratio < 0.015 or area_ratio > 0.55:
            continue

        y0, y1 = min(rows), max(rows) + 1
        x0, x1 = min(cols), max(cols) + 1
        if (x1 - x0) < 8 or (y1 - y0) < 8:
            continue

        bbox_norm = (
            x0 / scaled_w,
            y0 / scaled_h,
            x1 / scaled_w,
            y1 / scaled_h,
        )
        crop = _crop_norm(frame_arr, bbox_norm)
        crop_h, crop_w = crop.shape[0], crop.shape[1]
        aspect_ratio = crop_h / max(crop_w, 1)
        mean_rgb = tuple(float(v) for v in crop.mean(axis=(0, 1)))
        crop_center_x = (bbox_norm[0] + bbox_norm[2]) / 2.0
        crop_center_y = (bbox_norm[1] + bbox_norm[3]) / 2.0
        edge_proximity = max(
            0.0,
            1.0
            - min(
                crop_center_x,
                1.0 - crop_center_x,
                crop_center_y,
                1.0 - crop_center_y,
            )
            * 2.0,
        )
        path_clutter_score = (1.0 - bbox_norm[1]) * (0.5 + area_ratio)

        candidates.append(
            RegionCandidate(
                bbox_norm=bbox_norm,
                crop_bytes=_encode_jpeg(crop),
                mean_rgb=mean_rgb,
                area_ratio=round(area_ratio, 4),
                aspect_ratio=round(aspect_ratio, 4),
                edge_proximity=round(edge_proximity, 4),
                path_clutter_score=round(min(1.0, path_clutter_score), 4),
            )
        )

    if not candidates:
        fallback_bbox = (0.22, 0.18, 0.78, 0.88)
        crop = _crop_norm(frame_arr, fallback_bbox)
        candidates.append(
            RegionCandidate(
                bbox_norm=fallback_bbox,
                crop_bytes=_encode_jpeg(crop),
                mean_rgb=tuple(float(v) for v in crop.mean(axis=(0, 1))),
                area_ratio=round(((0.78 - 0.22) * (0.88 - 0.18)), 4),
                aspect_ratio=round(crop.shape[0] / max(crop.shape[1], 1), 4),
                edge_proximity=0.2,
                path_clutter_score=0.3,
            )
        )

    candidates.sort(
        key=lambda candidate: candidate.area_ratio * (1.0 + candidate.edge_proximity),
        reverse=True,
    )
    return candidates[:max_regions]


def _connected_components(mask: np.ndarray) -> list[tuple[list[int], list[int]]]:
    """Return connected components for a binary image."""
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components: list[tuple[list[int], list[int]]] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            rows: list[int] = []
            cols: list[int] = []
            visited[y, x] = True

            while stack:
                cy, cx = stack.pop()
                rows.append(cy)
                cols.append(cx)
                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            components.append((rows, cols))

    return components


def _crop_norm(frame_arr: np.ndarray, bbox_norm: tuple[float, float, float, float]) -> np.ndarray:
    height, width = frame_arr.shape[0], frame_arr.shape[1]
    x0 = max(0, int(bbox_norm[0] * width))
    y0 = max(0, int(bbox_norm[1] * height))
    x1 = min(width, int(bbox_norm[2] * width))
    y1 = min(height, int(bbox_norm[3] * height))
    crop = frame_arr[y0:y1, x0:x1]
    if crop.size == 0:
        return frame_arr
    return crop


def _encode_jpeg(arr: np.ndarray) -> bytes:
    image = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88, optimize=True)
    return buf.getvalue()


def _build_replay_gif(frame_bytes: list[bytes], *, title: str, subtitle: str) -> bytes:
    """Compose a simple animated GIF for one finding replay."""
    frames: list[Image.Image] = []
    for raw in frame_bytes[:4]:
        crop = Image.open(io.BytesIO(raw)).convert("RGB")
        panel = Image.new("RGB", (420, 256), color="#171616")
        fitted = ImageOps.contain(crop, (388, 188))
        panel.paste(fitted, ((420 - fitted.width) // 2, 22))
        draw = ImageDraw.Draw(panel)
        draw.rounded_rectangle((18, 214, 402, 242), radius=14, fill="#221D1A")
        draw.text((30, 221), title[:34], fill="#F8F5F0")
        draw.text((30, 237), subtitle[:34], fill="#BDAEA2")
        frames.append(panel)

    if not frames:
        return b""

    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=650,
        loop=0,
        optimize=False,
    )
    return buf.getvalue()


def _estimate_camera_path(frames: list[np.ndarray]) -> list[tuple[float, float]]:
    """Estimate a simple camera path from frame-to-frame image shifts."""
    if not frames:
        return []

    downscaled = [
        np.array(
            Image.fromarray(frame, mode="RGB").convert("L").resize((48, 48), Image.BILINEAR),
            dtype=np.float32,
        )
        for frame in frames
    ]

    positions: list[tuple[float, float]] = [(0.0, 0.0)]
    current_x = 0.0
    current_z = 0.8

    for previous, current in pairwise(downscaled):
        shift_x, shift_y = _best_shift(previous, current, max_shift=6)
        current_x += -shift_x / 48.0 * 0.8
        current_z += 0.42 + abs(shift_y) / 48.0 * 0.08
        positions.append((round(current_x, 3), round(current_z, 3)))

    return positions


def _best_shift(previous: np.ndarray, current: np.ndarray, max_shift: int = 6) -> tuple[int, int]:
    best_score = float("inf")
    best = (0, 0)

    for shift_y in range(-max_shift, max_shift + 1):
        for shift_x in range(-max_shift, max_shift + 1):
            y0_prev = max(0, shift_y)
            y1_prev = min(previous.shape[0], previous.shape[0] + shift_y)
            x0_prev = max(0, shift_x)
            x1_prev = min(previous.shape[1], previous.shape[1] + shift_x)

            y0_cur = max(0, -shift_y)
            y1_cur = min(current.shape[0], current.shape[0] - shift_y)
            x0_cur = max(0, -shift_x)
            x1_cur = min(current.shape[1], current.shape[1] - shift_x)

            if y1_prev - y0_prev < 12 or x1_prev - x0_prev < 12:
                continue

            prev_window = previous[y0_prev:y1_prev, x0_prev:x1_prev]
            cur_window = current[y0_cur:y1_cur, x0_cur:x1_cur]
            score = float(np.mean((prev_window - cur_window) ** 2))

            if score < best_score:
                best_score = score
                best = (shift_x, shift_y)

    return best


def _estimate_observation_position(
    region: RegionCandidate,
    camera_position: tuple[float, float],
) -> tuple[float, float, float]:
    cam_x, cam_z = camera_position
    cx = (region.bbox_norm[0] + region.bbox_norm[2]) / 2.0
    cy = (region.bbox_norm[1] + region.bbox_norm[3]) / 2.0

    view_x = (cx - 0.5) * 2.0
    view_y = 0.5 - cy
    estimated_depth = max(0.7, min(4.0, 4.4 - math.sqrt(region.area_ratio) * 6.0))

    x = cam_x + view_x * estimated_depth * 0.9
    y = 1.0 + view_y * 2.2
    z = cam_z + estimated_depth
    return (round(x, 3), round(y, 3), round(z, 3))


def _track_observations(observations: list[Observation]) -> list[list[Observation]]:
    tracks: list[list[Observation]] = []

    for obs in sorted(observations, key=lambda item: (item.frame_index, item.observation_id)):
        best_index = None
        best_score = 0.0

        for index, track in enumerate(tracks):
            score = _match_score(track, obs)
            if score > best_score:
                best_score = score
                best_index = index

        if best_index is not None and best_score >= 0.55:
            tracks[best_index].append(obs)
        else:
            tracks.append([obs])

    return tracks


def _match_score(track: list[Observation], obs: Observation) -> float:
    last = track[-1]
    if last.frame_index == obs.frame_index:
        return 0.0

    label_match = 0.45 if last.label.label == obs.label.label else 0.0
    material_match = 0.15 if last.label.material == obs.label.material else 0.0

    color_distance = (
        math.sqrt(sum((a - b) ** 2 for a, b in zip(last.mean_rgb, obs.mean_rgb, strict=True)))
        / 441.6729559300637
    )
    color_score = max(0.0, 0.2 - color_distance * 0.22)

    position_distance = math.dist(last.estimated_position, obs.estimated_position)
    position_score = max(0.0, 0.15 - min(position_distance, 2.0) * 0.08)

    area_score = max(0.0, 0.1 - abs(last.area_ratio - obs.area_ratio) * 0.6)
    gap_penalty = max(0.0, 0.08 - abs(last.frame_index - obs.frame_index - 1) * 0.04)
    return label_match + material_match + color_score + position_score + area_score + gap_penalty


def _build_objects_from_tracks(tracks: list[list[Observation]]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []

    for index, track in enumerate(tracks, start=1):
        representative = max(track, key=lambda item: item.label.confidence)
        weights = [max(obs.label.confidence, 0.15) for obs in track]
        positions = np.array([obs.estimated_position for obs in track], dtype=np.float32)
        centroid = np.average(positions, axis=0, weights=weights)
        variance = (
            float(np.mean(np.linalg.norm(positions - centroid, axis=1))) if len(track) > 1 else 0.0
        )
        height_m = float(
            np.mean(
                [
                    max(
                        0.15,
                        (obs.bbox_norm[3] - obs.bbox_norm[1]) * max(obs.estimated_position[2], 0.8),
                    )
                    for obs in track
                ]
            )
        )
        width_m = float(
            np.mean(
                [
                    max(
                        0.12,
                        (obs.bbox_norm[2] - obs.bbox_norm[0]) * max(obs.estimated_position[2], 0.8),
                    )
                    for obs in track
                ]
            )
        )
        grounding_confidence = max(0.25, min(0.94, 0.42 + len(track) * 0.11 - variance * 0.2))
        evidence_ids = [obs.observation_id for obs in track[:4]]

        objects.append(
            {
                "object_id": f"track-{index:02d}",
                "label": representative.label.label,
                "material": representative.label.material,
                "mass_kg": representative.label.mass_kg,
                "fragility": representative.label.fragility,
                "friction": representative.label.friction,
                "confidence": round(
                    float(np.mean([obs.label.confidence for obs in track])),
                    2,
                ),
                "position": [round(float(v), 3) for v in centroid.tolist()],
                "location_label": location_label(
                    (
                        round(float(centroid[0]), 3),
                        round(float(centroid[1]), 3),
                        round(float(centroid[2]), 3),
                    )
                ),
                "observation_count": len(track),
                "grounding_confidence": round(grounding_confidence, 2),
                "position_variance": round(variance, 3),
                "estimated_height_m": round(height_m, 3),
                "estimated_width_m": round(width_m, 3),
                "edge_proximity": round(max(obs.edge_proximity for obs in track), 3),
                "path_clutter_score": round(max(obs.path_clutter_score for obs in track), 3),
                "evidence_ids": evidence_ids,
            }
        )

    return objects


def _build_scene_point_cloud(tracks: list[list[Observation]]) -> list[list[float]]:
    """Build a scene point cloud anchored to tracked object observations."""
    scene_points: list[list[float]] = []

    for track in tracks:
        for obs in track[:2]:
            local_points = generate_depth_pointcloud(obs.crop_bytes, n_points=80)
            local_centroid = point_cloud_centroid(local_points)
            for point in local_points:
                scene_points.append(
                    [
                        round(point[0] - local_centroid[0] + obs.estimated_position[0], 3),
                        round(point[1] - local_centroid[1] + obs.estimated_position[1], 3),
                        round(point[2] - local_centroid[2] + obs.estimated_position[2], 3),
                        point[3],
                        point[4],
                        point[5],
                    ]
                )

    return scene_points[:1200]


def _assess_scan_quality(
    decoded_frames: list[np.ndarray],
    camera_positions: list[tuple[float, float]],
    regions_per_frame: list[list[RegionCandidate]],
) -> dict[str, Any]:
    """Estimate whether the upload is visually usable for hazard reporting."""
    if not decoded_frames:
        return {
            "status": "poor",
            "score": 0.0,
            "usable": False,
            "capture_summary": (
                "No usable frames were available, so the scan cannot support" " a report."
            ),
            "rescan_recommended": True,
            "warnings": ["No frames were available to assess scan quality."],
            "retry_guidance": ["Upload a valid image or a 20-60 second walkthrough video."],
            "metrics": {},
        }

    brightness_values: list[float] = []
    dark_clip_values: list[float] = []
    bright_clip_values: list[float] = []
    sharpness_values: list[float] = []
    saliency_coverages: list[float] = []

    for frame_arr, regions in zip(decoded_frames, regions_per_frame, strict=True):
        gray = frame_arr.astype(np.float32).mean(axis=2) / 255.0
        brightness_values.append(float(np.mean(gray)))
        dark_clip_values.append(float(np.mean(gray < 0.15)))
        bright_clip_values.append(float(np.mean(gray > 0.9)))
        grad_y, grad_x = np.gradient(gray)
        sharpness_values.append(float(np.mean(np.sqrt(grad_x**2 + grad_y**2))))
        saliency_coverages.append(min(1.0, sum(region.area_ratio for region in regions) / 0.28))

    frame_count = len(decoded_frames)
    total_motion = (
        sum(math.dist(previous, current) for previous, current in pairwise(camera_positions))
        if len(camera_positions) > 1
        else 0.0
    )
    mean_brightness = float(np.mean(brightness_values))
    mean_dark_clip = float(np.mean(dark_clip_values))
    mean_bright_clip = float(np.mean(bright_clip_values))
    mean_sharpness = float(np.mean(sharpness_values))
    mean_saliency = float(np.mean(saliency_coverages))

    brightness_score = max(0.0, 1.0 - abs(mean_brightness - 0.5) / 0.35)
    exposure_score = max(0.0, 1.0 - min(1.0, (mean_dark_clip + mean_bright_clip) * 1.8))
    sharpness_score = min(1.0, mean_sharpness / 0.09)
    motion_score = (
        1.0 if frame_count == 1 else min(1.0, total_motion / max(0.9, (frame_count - 1) * 0.28))
    )
    saliency_score = min(1.0, mean_saliency)

    score = (
        brightness_score * 0.2
        + exposure_score * 0.2
        + sharpness_score * 0.24
        + motion_score * 0.18
        + saliency_score * 0.18
    )

    warnings: list[str] = []
    guidance: list[str] = []

    if sharpness_score < 0.42:
        warnings.append("The scan looks blurry, which weakens object labeling and localization.")
        guidance.append(
            "Hold the phone steadier and pause briefly on shelves, tables," " and tall items."
        )
    if brightness_score < 0.42:
        if mean_brightness < 0.35:
            warnings.append("The scan is darker than ideal, so some hazards may be missed.")
            guidance.append("Turn on room lights or rescan during brighter conditions.")
        else:
            warnings.append("Parts of the scan are overexposed, which can hide object edges.")
            guidance.append("Avoid pointing directly at windows or bright lamps while scanning.")
    if frame_count > 1 and motion_score < 0.32:
        warnings.append("The walkthrough covers too little motion to ground objects confidently.")
        guidance.append(
            "Move slowly across the room so objects appear from more than" " one viewpoint."
        )
    if saliency_score < 0.34:
        warnings.append("The frames contain limited clear object coverage for the report.")
        guidance.append("Keep the main surfaces and objects centered in frame for a little longer.")
    if frame_count > 1 and frame_count < 4:
        warnings.append("The walkthrough is short, so spatial coverage is limited.")
        guidance.append("Aim for a 20-60 second scan that sweeps the full room once.")

    status = "good" if score >= 0.72 else "fair" if score >= 0.48 else "poor"
    capture_summary = (
        "Broad enough for a first-pass hazard screen."
        if status == "good"
        else "Usable, but some weaker findings may need a rescan."
        if status == "fair"
        else "Too limited to trust smaller details without rescanning."
    )
    rescan_recommended = status == "poor" or len(warnings) >= 2

    return {
        "status": status,
        "score": round(score, 2),
        "usable": score >= 0.48,
        "capture_summary": capture_summary,
        "rescan_recommended": rescan_recommended,
        "warnings": warnings[:4],
        "retry_guidance": guidance[:4],
        "metrics": {
            "frame_count": frame_count,
            "brightness": round(mean_brightness, 2),
            "dark_clip": round(mean_dark_clip, 2),
            "bright_clip": round(mean_bright_clip, 2),
            "sharpness": round(mean_sharpness, 3),
            "motion_coverage": round(motion_score, 2),
            "saliency_coverage": round(mean_saliency, 2),
        },
    }


def _coverage_label(objects: list[dict[str, Any]], scan_quality: dict[str, Any]) -> str:
    """Return a user-facing coverage band for the uploaded scan."""
    metrics = scan_quality.get("metrics") or {}
    frame_count = int(metrics.get("frame_count", 0) or 0)
    motion = float(metrics.get("motion_coverage", 0.0) or 0.0)
    saliency = float(metrics.get("saliency_coverage", 0.0) or 0.0)
    object_count = len(objects)
    score = (
        float(scan_quality.get("score", 0.0)) * 0.45
        + motion * 0.25
        + saliency * 0.2
        + min(1.0, object_count / 6.0) * 0.1
    )
    if frame_count >= 4 and score >= 0.72:
        return "broad"
    if frame_count >= 2 and score >= 0.48:
        return "partial"
    return "limited"


def _coverage_summary(
    coverage_label_value: str,
    scan_quality: dict[str, Any],
    *,
    object_count: int,
) -> str:
    """Explain how complete the current upload coverage looks."""
    metrics = scan_quality.get("metrics") or {}
    frame_count = int(metrics.get("frame_count", 0) or 0)
    motion = int(round(float(metrics.get("motion_coverage", 0.0) or 0.0) * 100))
    saliency = int(round(float(metrics.get("saliency_coverage", 0.0) or 0.0) * 100))
    if coverage_label_value == "broad":
        return (
            f"{frame_count} sampled frames captured broad room coverage with"
            f" {motion}% motion coverage and {saliency}% object coverage."
        )
    if coverage_label_value == "partial":
        return (
            f"{frame_count} sampled frames produced a usable but incomplete screen."
            f" ATLAS-0 tracked {object_count} object(s), so weaker findings should"
            " still be reviewed with caution."
        )
    return (
        f"{frame_count} sampled frames gave limited coverage with {motion}% motion"
        f" coverage and {saliency}% object coverage. Treat the result as a narrow"
        " screen and rescan before relying on smaller details."
    )


def _build_summary(
    filename: str,
    objects: list[dict[str, Any]],
    risks: list[dict[str, Any]],
    scene_source: str,
    scan_quality: dict[str, Any],
) -> dict[str, Any]:
    top_risk = max(risks, key=lambda entry: float(entry.get("risk_score", 0.0)), default=None)
    confidence_label = (
        "Estimated multi-view grounding"
        if scene_source == "estimated_multiview"
        else "Single-view estimate"
    )
    low_confidence_count = sum(1 for risk in risks if float(risk.get("confidence", 0.0)) < 0.6)
    high_confidence_count = sum(
        1
        for risk in risks
        if float(risk.get("confidence", 0.0)) >= 0.6
        and float(risk.get("reasoning", {}).get("grounding_confidence", 0.0)) >= 0.55
    )
    coverage_label_value = _coverage_label(objects, scan_quality)
    rescan_recommended = bool(scan_quality.get("rescan_recommended")) or (
        coverage_label_value == "limited"
    )
    screening_statement = (
        "This report flags likely hazards from the uploaded scan."
        " It does not certify that the room is safe."
    )
    if top_risk:
        headline = f"Start with {top_risk.get('hazard_title', 'the top hazard')}"
        overview = (
            f"ATLAS-0 flagged {len(risks)} likely hazard"
            f"{'' if len(risks) == 1 else 's'} in this scan."
            f" Begin with {top_risk.get('hazard_title', 'the top hazard')} and use"
            " the evidence frames before treating smaller findings as certain."
        )
        report_posture = "actionable screening"
    elif rescan_recommended:
        headline = "Rescan recommended before trusting smaller details"
        overview = (
            "No high-confidence hazards were detected, but scan coverage was limited."
            " Treat this as an incomplete screen rather than a clean bill of health."
        )
        report_posture = "limited screening"
    else:
        headline = "No high-confidence hazards detected"
        overview = (
            "ATLAS-0 did not surface a strong hazard in this upload, but this is still"
            " a screening result, not proof that the room is safe."
        )
        report_posture = "preliminary screening"
    return {
        "filename": filename,
        "object_count": len(objects),
        "hazard_count": len(risks),
        "top_severity": top_risk.get("severity", "none") if top_risk else "none",
        "top_hazard_label": top_risk.get("hazard_title") if top_risk else None,
        "scene_source": scene_source,
        "confidence_label": confidence_label,
        "scan_quality_label": str(scan_quality.get("status", "unknown")).capitalize(),
        "scan_quality_score": float(scan_quality.get("score", 0.0)),
        "warning_count": len(scan_quality.get("warnings", [])),
        "low_confidence_count": low_confidence_count,
        "high_confidence_hazard_count": high_confidence_count,
        "top_confidence_label": confidence_bucket(float(top_risk.get("confidence", 0.0)))
        if top_risk
        else "weak",
        "coverage_label": coverage_label_value.capitalize(),
        "coverage_summary": _coverage_summary(
            coverage_label_value,
            scan_quality,
            object_count=len(objects),
        ),
        "report_posture": report_posture,
        "rescan_recommended": rescan_recommended,
        "headline": headline,
        "overview": overview,
        "screening_statement": screening_statement,
    }
