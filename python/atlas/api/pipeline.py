"""Upload ingestion pipeline: queue, workers, processing, and sample report.

Owns the background/detached upload workers, per-job processing, VLM labeling,
multipart parsing and validation, pseudo-depth point clouds, the offline image
heuristic, and the cached public sample report. Patched-in-tests collaborators
(``analyze_uploaded_image``, ``_build_pdf_report``) are called via their module
namespaces so monkeypatching keeps working.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import pathlib
import socket
import uuid
from datetime import UTC, datetime
from email.parser import BytesParser
from email.policy import default
from typing import Any

import structlog
from fastapi import HTTPException, Request

from atlas.api import reports, upload_analysis
from atlas.api.helpers import _finding_key, _normalize_room_label, _utc_now_iso
from atlas.api.jobs import (
    _ensure_job_derived_fields,
    _job_artifacts,
    _record_job_failure,
    _refresh_job_artifacts,
    _refresh_operational_metrics,
    _refresh_upload_jobs_from_disk,
    _save_job,
    _set_job_artifact,
    _set_worker_activity,
    _update_job,
)
from atlas.api.metrics import upload_job_seconds, upload_total
from atlas.api.state import (
    _IMAGE_TYPES,
    _VIDEO_TYPES,
    _build_runtime_vlm_config,
    _get_agent,
    _state,
    _upload_cfg,
    _upload_jobs,
    _upload_store,
)
from atlas.api.upload_analysis import analyze_frame_samples, build_finding_replays
from atlas.utils.video import ExtractedFrame, probe_video_metadata
from atlas.vlm.inference import SemanticLabel, VLMEngine
from atlas.world_model.hazards import normalize_audience_mode

logger = structlog.get_logger(__name__)


def _sample_report_frames_dir() -> pathlib.Path:
    """Return the directory holding the built-in sample walkthrough frames."""
    return pathlib.Path(__file__).parents[3] / "data" / "sample_walkthrough" / "frames"


def _sample_report_cache() -> dict[str, Any] | None:
    """Return the in-memory sample report cache when available."""
    cache = _state.get("sample_report_cache")
    return cache if isinstance(cache, dict) else None


async def _build_sample_report() -> dict[str, Any]:
    """Build and cache the public sample walkthrough report."""
    cached = _sample_report_cache()
    if cached is not None:
        return cached

    lock = _state.get("sample_report_lock")
    if not isinstance(lock, asyncio.Lock):
        lock = asyncio.Lock()
        _state["sample_report_lock"] = lock

    async with lock:
        cached = _sample_report_cache()
        if cached is not None:
            return cached

        frames_dir = _sample_report_frames_dir()
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if not frame_paths:
            raise HTTPException(status_code=500, detail="Built-in sample frames are unavailable.")

        frame_samples = [
            ExtractedFrame(
                index=index,
                timestamp_s=round(index * 1.25, 2),
                image_bytes=path.read_bytes(),
            )
            for index, path in enumerate(frame_paths)
        ]
        result = await analyze_frame_samples(
            frame_samples,
            filename="sample_walkthrough",
            scan_kind="video",
            labeler=_label_upload_region,
            audience_mode="general",
        )

        job_id = "sample-walkthrough"
        risks = [dict(risk) for risk in result.risks]
        evidence_frames = [dict(frame) for frame in result.evidence_frames]
        for frame in evidence_frames:
            evidence_id = str(frame.get("evidence_id", ""))
            if evidence_id:
                frame["image_url"] = f"/sample-report/evidence/{evidence_id}"

        replay_descriptors, replay_payloads = build_finding_replays(
            risks,
            result.evidence_artifacts,
        )
        replay_by_key: dict[str, dict[str, Any]] = {}
        for descriptor in replay_descriptors:
            replay = dict(descriptor)
            replay_id = str(replay.get("replay_id", ""))
            if not replay_id:
                continue
            replay["image_url"] = f"/sample-report/replays/{replay_id}"
            replay_key = (
                f"{replay.get('object_id') or 'finding'}:"
                f"{replay.get('hazard_code') or 'unknown'}"
            )
            replay_by_key[replay_key] = replay

        for risk in risks:
            replay = replay_by_key.get(_finding_key(risk))
            if replay is not None:
                risk["replay"] = replay

        now = _utc_now_iso()
        job: dict[str, Any] = {
            "job_id": job_id,
            "filename": "sample_walkthrough",
            "room_label": "Sample living room",
            "is_sample": True,
            "sample_key": "walkthrough",
            "audience_mode": "general",
            "content_type": "video/mp4",
            "status": "complete",
            "stage": "complete",
            "progress": 1.0,
            "objects": result.objects,
            "risks": risks,
            "point_cloud": result.point_cloud,
            "fix_first": result.fix_first,
            "weekend_fix_list": None,
            "summary": result.summary,
            "recommendations": result.recommendations,
            "evidence_frames": evidence_frames,
            "scan_quality": result.scan_quality,
            "trust_notes": [
                "This is ATLAS-0's built-in sample walkthrough.",
                *result.trust_notes,
            ],
            "scene_source": result.scene_source,
            "finding_feedback": [],
            "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
            "human_evaluation": None,
            "finding_follow_up": [],
            "resolution_summary": None,
            "room_history": [],
            "room_comparison": None,
            "room_wins": None,
            "report_url": "/sample-report/report.pdf",
            "share_url": "/app?view=report&sample=walkthrough",
            "error": None,
            "artifacts": {
                "report_pdf": {
                    "kind": "report_pdf",
                    "storage_backend": "memory",
                    "storage_key": "sample-report/report.pdf",
                    "media_type": "application/pdf",
                    "url": "/sample-report/report.pdf",
                },
            },
            "attempt_count": 0,
            "queued_at": now,
            "started_at": now,
            "completed_at": now,
        }
        if job.get("summary") is None:
            job["summary"] = {}
        job["summary"]["share_summary"] = "Built-in sample walkthrough report"
        pdf_bytes = reports._build_pdf_report(_ensure_job_derived_fields(job))
        cache = {
            "job": job,
            "evidence_artifacts": dict(result.evidence_artifacts),
            "replay_artifacts": dict(replay_payloads),
            "report_pdf": pdf_bytes,
        }
        _state["sample_report_cache"] = cache
        return cache


def _upload_queue() -> asyncio.Queue[str | None] | None:
    """Return the background upload queue when workers are running."""
    queue = _state.get("upload_queue")
    return queue if isinstance(queue, asyncio.Queue) else None


def _upload_queue_ids() -> set[str]:
    """Return the set of queued job IDs."""
    queued = _state.setdefault("upload_queue_ids", set())
    return queued if isinstance(queued, set) else set()


def _upload_cancelled_jobs() -> set[str]:
    """Return the set of deleted jobs that workers should ignore."""
    cancelled = _state.setdefault("upload_cancelled_jobs", set())
    return cancelled if isinstance(cancelled, set) else set()


def _is_upload_cancelled(job_id: str) -> bool:
    """Return True when one job has been deleted/cancelled."""
    return job_id in _upload_cancelled_jobs()


async def _start_upload_workers() -> None:
    """Start the persistent upload queue and worker tasks once."""
    if _upload_cfg.worker_mode != "in_process":
        _state["upload_worker_tasks"] = []
        _state["active_upload_workers"] = 0
        _refresh_operational_metrics()
        return

    if _upload_queue() is not None:
        return

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    _state["upload_queue"] = queue
    _state["upload_queue_ids"] = set()
    _state["upload_cancelled_jobs"] = set()
    _state["active_upload_workers"] = 0
    _state["upload_worker_tasks"] = [
        asyncio.create_task(_upload_worker_loop(index))
        for index in range(_upload_cfg.max_concurrent_jobs)
    ]
    _refresh_operational_metrics()


async def _stop_upload_workers() -> None:
    """Stop background upload workers and clear queue state."""
    queue = _upload_queue()
    tasks = _state.get("upload_worker_tasks")
    if queue is None or not isinstance(tasks, list):
        return

    for _ in tasks:
        await queue.put(None)
    await asyncio.gather(*tasks, return_exceptions=True)

    _state.pop("upload_worker_tasks", None)
    _state.pop("upload_queue", None)
    _state.pop("upload_queue_ids", None)
    _state.pop("upload_cancelled_jobs", None)
    _state.pop("active_upload_workers", None)
    _refresh_operational_metrics()


async def _enqueue_upload_job(job_id: str) -> None:
    """Queue one upload job for background processing exactly once."""
    await _start_upload_workers()
    if _upload_cfg.worker_mode != "in_process":
        _refresh_operational_metrics()
        return

    queue = _upload_queue()
    if queue is None:
        return

    queued_ids = _upload_queue_ids()
    if job_id in queued_ids:
        return

    queued_ids.add(job_id)
    await queue.put(job_id)
    _refresh_operational_metrics()


async def _resume_pending_upload_jobs() -> None:
    """Requeue persisted jobs that were interrupted before finishing."""
    _state["last_resume_scan_at"] = _utc_now_iso()
    _refresh_upload_jobs_from_disk()
    for job_id, job in _upload_jobs.items():
        status = str(job.get("status", "")).lower()
        if status not in {"queued", "processing"}:
            continue

        if not _upload_store.has_job_input(job_id):
            _update_job(
                job,
                status="error",
                stage="complete",
                progress=1.0,
                error=(
                    "Queued upload could not be resumed because its source file is no longer"
                    " available on disk."
                ),
                completed_at=_utc_now_iso(),
            )
            continue

        _update_job(
            job,
            status="queued",
            stage="upload",
            progress=min(float(job.get("progress") or 0.0), 0.1),
            error=None,
            queued_at=_utc_now_iso(),
        )
        await _enqueue_upload_job(job_id)


async def _upload_worker_loop(worker_index: int) -> None:
    """Worker task that pulls queued upload jobs from the persistent queue."""
    queue = _upload_queue()
    if queue is None:
        return

    while True:
        job_id = await queue.get()
        if job_id is None:
            queue.task_done()
            break

        _upload_queue_ids().discard(job_id)
        _set_worker_activity(int(_state.get("active_upload_workers", 0) or 0) + 1)
        _refresh_operational_metrics()
        try:
            await _process_upload(job_id)
        except Exception as exc:  # pragma: no cover - safety net for worker loop
            logger.exception(
                "upload_worker_failed",
                worker_index=worker_index,
                job_id=job_id,
                error=str(exc),
            )
        finally:
            _set_worker_activity(max(0, int(_state.get("active_upload_workers", 0) or 0) - 1))
            _refresh_operational_metrics()
            queue.task_done()


def _next_detached_worker_id(prefix: str = "atlas-worker") -> str:
    """Build a stable-ish worker identifier for detached execution."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _write_detached_worker_heartbeat(
    worker_id: str,
    *,
    state: str,
    started_at: str,
    claimed_job_id: str | None = None,
) -> None:
    """Persist one durable detached-worker heartbeat record."""
    now = datetime.now(UTC)
    _upload_store.save_worker_record(
        worker_id,
        {
            "worker_id": worker_id,
            "worker_mode": "external",
            "state": state,
            "claimed_job_id": claimed_job_id,
            "started_at": started_at,
            "heartbeat_at": now.isoformat(),
            "heartbeat_unix": now.timestamp(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        },
    )


def _claim_next_external_job(worker_id: str) -> str | None:
    """Claim the next queued job from disk for detached worker execution."""
    claimed = _upload_store.claim_next_job(
        worker_id,
        lease_seconds=_upload_cfg.worker_claim_ttl_seconds,
    )
    if claimed is None:
        return None
    job_id, manifest = claimed
    _upload_jobs[job_id] = _ensure_job_derived_fields(manifest)
    _refresh_job_artifacts(_upload_jobs[job_id])
    return job_id


async def run_detached_upload_worker(
    *,
    worker_id: str | None = None,
    once: bool = False,
) -> bool:
    """Run one external upload worker loop against durable queued jobs."""
    active_worker_id = worker_id or _next_detached_worker_id()
    started_at = _utc_now_iso()
    heartbeat_interval = max(1.0, float(_upload_cfg.worker_heartbeat_seconds or 10.0))
    next_heartbeat = 0.0
    try:
        while True:
            loop_now = asyncio.get_running_loop().time()
            if loop_now >= next_heartbeat:
                _write_detached_worker_heartbeat(
                    active_worker_id,
                    state="idle",
                    started_at=started_at,
                )
                next_heartbeat = loop_now + heartbeat_interval
                _refresh_operational_metrics()

            _refresh_upload_jobs_from_disk()
            job_id = _claim_next_external_job(active_worker_id)
            if job_id is None:
                if once:
                    return False
                await asyncio.sleep(_upload_cfg.worker_poll_seconds)
                continue

            _set_worker_activity(int(_state.get("active_upload_workers", 0) or 0) + 1)
            _write_detached_worker_heartbeat(
                active_worker_id,
                state="processing",
                started_at=started_at,
                claimed_job_id=job_id,
            )
            _refresh_operational_metrics()
            try:
                await _process_upload(job_id, worker_id=active_worker_id)
            finally:
                _set_worker_activity(max(0, int(_state.get("active_upload_workers", 0) or 0) - 1))
                _write_detached_worker_heartbeat(
                    active_worker_id,
                    state="idle",
                    started_at=started_at,
                )
                _refresh_operational_metrics()
            if once:
                return True
    finally:
        _upload_store.delete_worker_record(active_worker_id)
        _refresh_operational_metrics()


def _active_upload_job_count() -> int:
    """Return the number of queued or processing upload jobs."""
    _refresh_upload_jobs_from_disk()
    return sum(
        1 for job in _upload_jobs.values() if str(job.get("status")) in {"queued", "processing"}
    )


async def _get_or_init_upload_vlm() -> VLMEngine:
    """Return a cached, initialised :class:`VLMEngine` for upload analysis."""
    if "upload_vlm" not in _state:
        engine = VLMEngine(_build_runtime_vlm_config())
        await engine.initialize()
        _state["upload_vlm"] = engine
    return _state["upload_vlm"]  # type: ignore[return-value]


async def _label_upload_region(content: bytes, region_hint: str) -> SemanticLabel:
    """Label one upload region, falling back to image heuristics when needed."""
    engine = await _get_or_init_upload_vlm()
    label = await engine.label_region(content, region_hint=region_hint)
    if label.label in ("unknown", "") or label.confidence < 0.35:
        label = _analyze_image_heuristic(content)
    return label


def _generate_depth_pointcloud(
    content: bytes,
    n_points: int = 600,
) -> list[list[float]]:
    """Sample a pseudo-3D point cloud from a 2D image.

    Uses luminance-as-depth (bright = near) plus edge-weighted sampling so
    interesting areas (objects, contours) are denser than flat backgrounds.
    Returns a list of ``[x, y, z, r, g, b]`` rows in world-space metres,
    normalised colours 0-1.

    Coordinate convention (right-handed, Y-up):
      * Image left→right  maps to world X  ∈ [-2.5, 2.5]
      * Image top→bottom  maps to world Y  ∈ [ 2.0, 0.0]
      * Luminance depth   maps to world Z  ∈ [ 0.0, 3.0]  (bright=near=0)

    Args:
        content:  Raw image bytes (JPEG / PNG / WEBP).
        n_points: Number of points to sample.

    Returns:
        List of ``[x, y, z, r, g, b]`` rows.
    """
    try:
        import numpy as np  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        img = Image.open(io.BytesIO(content)).convert("RGB")
        # Resize to a manageable grid while keeping aspect ratio
        max_side = 160
        w, h = img.size
        scale = min(max_side / w, max_side / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img_small = img.resize((nw, nh), Image.LANCZOS)
        arr = np.array(img_small, dtype=np.float32)  # (nh, nw, 3)

        r_ch, g_ch, b_ch = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        lum = (0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch) / 255.0  # (nh, nw)

        # Edge strength → sampling weight (concentrate on object boundaries)
        gy = np.gradient(lum, axis=0)
        gx = np.gradient(lum, axis=1)
        weight = np.sqrt(gx**2 + gy**2) + 0.08  # floor so flat areas still sampled
        weight_flat = weight.flatten()
        probs = weight_flat / weight_flat.sum()

        n_actual = min(n_points, nw * nh)
        indices = np.random.choice(nw * nh, size=n_actual, replace=False, p=probs)

        rows: list[list[float]] = []
        for idx in indices:
            iy, ix = divmod(int(idx), nw)

            # Normalised image coords in [-1, 1]
            u = (ix / (nw - 1)) * 2.0 - 1.0  # left → right → world +X
            v = 1.0 - (iy / (nh - 1)) * 2.0  # top → bottom → world +Y (inverted)

            depth_01 = 1.0 - float(lum[iy, ix])  # bright=0 (near), dark=1 (far)

            x = float(u) * 2.5
            y = float(v) * 1.0 + 1.0  # world Y 0..2 centred at 1
            z = float(depth_01) * 3.0  # Z 0..3 m

            nr = float(r_ch[iy, ix]) / 255.0
            ng = float(g_ch[iy, ix]) / 255.0
            nb = float(b_ch[iy, ix]) / 255.0

            rows.append(
                [round(x, 3), round(y, 3), round(z, 3), round(nr, 3), round(ng, 3), round(nb, 3)]
            )

        return rows

    except Exception as exc:
        logger.warning("depth_pointcloud_failed", error=str(exc))
        return []


def _analyze_image_heuristic(content: bytes) -> SemanticLabel:
    """Derive a :class:`SemanticLabel` from image pixel statistics alone.

    Used when the VLM is offline.  Analyses colour, brightness, saturation,
    and texture variance to produce a plausible label + physics properties.

    Args:
        content: Raw image bytes (JPEG / PNG / WEBP).

    Returns:
        A :class:`SemanticLabel` inferred from visual statistics.
    """
    try:
        import numpy as np  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        img = Image.open(io.BytesIO(content)).convert("RGB")
        img.thumbnail((128, 128))
        arr = np.array(img, dtype=np.float32)

        mean_r = float(arr[:, :, 0].mean())
        mean_g = float(arr[:, :, 1].mean())
        mean_b = float(arr[:, :, 2].mean())
        brightness = (mean_r + mean_g + mean_b) / 3.0 / 255.0

        # Saturation (normalised range 0-1)
        mx = max(mean_r, mean_g, mean_b)
        mn = min(mean_r, mean_g, mean_b)
        saturation = (mx - mn) / (mx + 1.0)

        # Texture complexity: stddev of greyscale
        gray = arr.mean(axis=2)
        texture = float(gray.std()) / 255.0

        # Dominant-hue bucket (0-5 like HSV hue / 60)
        if mx == mn:
            hue_bucket = -1  # achromatic
        elif mx == mean_r:
            hue_bucket = 0  # red/orange
        elif mx == mean_g:
            hue_bucket = 2  # green
        else:
            hue_bucket = 4  # blue

        # ── Material + label heuristics ────────────────────────────────────
        if saturation < 0.12:
            # Near-greyscale → metal or stone
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
            # Warm browns → wood
            label, material = "Wooden Object", "Wood"
            mass_kg, fragility, friction = 6.0, 0.20, 0.65
        elif hue_bucket == 0 and brightness >= 0.55:
            # Warm light → ceramic / terracotta
            label, material = "Ceramic Object", "Ceramic"
            mass_kg, fragility, friction = 1.2, 0.72, 0.55
        elif hue_bucket == 2:
            # Greens → organic / plant
            label, material = "Organic Object", "Plant"
            mass_kg, fragility, friction = 0.8, 0.55, 0.40
        elif hue_bucket == 4 and brightness > 0.55:
            # Blue-bright → glass or water
            label, material = "Glass Object", "Glass"
            mass_kg, fragility, friction = 0.6, 0.90, 0.20
        elif brightness > 0.75:
            # Very bright → plastic or paper
            label, material = "Plastic Object", "Plastic"
            mass_kg, fragility, friction = 0.5, 0.40, 0.50
        else:
            label, material = "Composite Object", "Mixed"
            mass_kg, fragility, friction = 2.0, 0.50, 0.50

        # Texture raises fragility slightly (complex surface = more facets to break)
        fragility = min(1.0, fragility + texture * 0.2)

        # Confidence scales with how clear the heuristic signal is
        confidence = min(0.82, 0.45 + saturation * 0.3 + (1.0 - abs(brightness - 0.5)) * 0.15)

        logger.info(
            "image_heuristic_analysis",
            label=label,
            material=material,
            brightness=round(brightness, 2),
            saturation=round(saturation, 2),
            texture=round(texture, 2),
        )
        return SemanticLabel(
            label=label,
            material=material,
            mass_kg=round(mass_kg, 2),
            fragility=round(fragility, 2),
            friction=round(friction, 2),
            confidence=round(confidence, 2),
        )

    except Exception as exc:
        logger.warning("image_heuristic_failed", error=str(exc))
        return SemanticLabel(
            label="Unknown Object",
            material="Unknown",
            mass_kg=1.0,
            fragility=0.5,
            friction=0.5,
            confidence=0.30,
        )


async def _process_upload(job_id: str, *, worker_id: str | None = None) -> None:
    """Worker task: load one queued upload from disk and process it."""
    job = _upload_jobs.get(job_id)
    if job is None or _is_upload_cancelled(job_id):
        _upload_cancelled_jobs().discard(job_id)
        return

    content = _upload_store.load_job_input(job_id)
    if content is None:
        _update_job(
            job,
            status="error",
            stage="complete",
            progress=1.0,
            error="Queued upload input is missing from disk.",
            completed_at=_utc_now_iso(),
        )
        return

    content_type = str(job.get("content_type") or "application/octet-stream")
    attempt_count = int(job.get("attempt_count") or 0) + 1
    started_wall = datetime.now(UTC)

    try:
        _update_job(
            job,
            status="processing",
            stage="ingest",
            progress=0.15,
            attempt_count=attempt_count,
            started_at=_utc_now_iso(),
            error=None,
        )

        async with asyncio.timeout(_upload_cfg.job_timeout_seconds):
            await asyncio.sleep(0.3)

            is_image = content_type in _IMAGE_TYPES or content_type.startswith("image/")
            is_video = content_type in _VIDEO_TYPES or content_type.startswith("video/")

            if is_image:
                _update_job(job, stage="vlm", progress=0.4)
                await asyncio.sleep(0.2)
                result = await upload_analysis.analyze_uploaded_image(
                    content,
                    filename=job["filename"],
                    content_type=content_type,
                    labeler=_label_upload_region,
                    audience_mode=str(job.get("audience_mode") or "general"),
                )
            elif is_video:
                _update_job(job, stage="vlm", progress=0.35)
                result = await upload_analysis.analyze_uploaded_video(
                    content,
                    filename=job["filename"],
                    labeler=_label_upload_region,
                    audience_mode=str(job.get("audience_mode") or "general"),
                )
            else:
                msg = f"Unsupported file type: {content_type!r}"
                raise ValueError(msg)

            if worker_id:
                _upload_store.refresh_job_claim(
                    job_id,
                    worker_id,
                    lease_seconds=_upload_cfg.worker_claim_ttl_seconds,
                )

        if _is_upload_cancelled(job_id) or job_id not in _upload_jobs:
            return

        _update_job(job, stage="risk", progress=0.9)

        risks = [dict(risk) for risk in result.risks]
        evidence_frames = [dict(frame) for frame in result.evidence_frames]
        evidence_artifacts: dict[str, dict[str, Any]] = {}
        for frame in evidence_frames:
            evidence_id = str(frame.get("evidence_id", ""))
            content_bytes = result.evidence_artifacts.get(evidence_id)
            if not evidence_id or content_bytes is None:
                continue
            evidence_path = _upload_store.save_evidence_image(
                job_id,
                evidence_id,
                content_bytes,
                suffix=".jpg",
            )
            frame["image_url"] = f"/jobs/{job_id}/evidence/{evidence_id}"
            pointer = _upload_store.artifact_pointer(
                job_id,
                _upload_store.job_relative_path(job_id, evidence_path),
                kind="evidence_image",
                media_type="image/jpeg",
                url=frame["image_url"],
            )
            frame["artifact"] = pointer
            evidence_artifacts[evidence_id] = pointer

        replay_descriptors, replay_payloads = build_finding_replays(
            risks,
            result.evidence_artifacts,
        )
        finding_replays: dict[str, dict[str, Any]] = {}
        finding_replays_by_key: dict[str, dict[str, Any]] = {}
        for descriptor in replay_descriptors:
            replay_id = str(descriptor.get("replay_id", ""))
            replay_bytes = replay_payloads.get(replay_id)
            if not replay_id or not replay_bytes:
                continue
            replay_path = _upload_store.save_replay_gif(job_id, replay_id, replay_bytes)
            image_url = f"/jobs/{job_id}/replays/{replay_id}"
            pointer = _upload_store.artifact_pointer(
                job_id,
                _upload_store.job_relative_path(job_id, replay_path),
                kind="finding_replay",
                media_type="image/gif",
                url=image_url,
            )
            replay = dict(descriptor)
            replay["image_url"] = image_url
            replay["artifact"] = pointer
            finding_replays[replay_id] = pointer
            finding_key = (
                f"{replay.get('object_id') or 'finding'}:"
                f"{replay.get('hazard_code') or 'unknown'}"
            )
            finding_replays_by_key[finding_key] = replay

        for risk in risks:
            replay = finding_replays_by_key.get(_finding_key(risk))
            if replay is not None:
                risk["replay"] = replay

        _update_job(
            job,
            status="complete",
            stage="complete",
            progress=1.0,
            is_sample=False,
            sample_key=None,
            objects=result.objects,
            risks=risks,
            point_cloud=result.point_cloud,
            fix_first=result.fix_first,
            summary=result.summary,
            recommendations=result.recommendations,
            evidence_frames=evidence_frames,
            scan_quality=result.scan_quality,
            trust_notes=result.trust_notes,
            scene_source=result.scene_source,
            finding_feedback=[],
            feedback_summary={"useful": 0, "wrong": 0, "duplicate": 0},
            human_evaluation=None,
            finding_follow_up=[],
            resolution_summary=None,
            report_url=f"/reports/{job['job_id']}.pdf",
            completed_at=_utc_now_iso(),
        )
        _upload_store.append_product_event(
            {
                "event_name": "upload_completed",
                "surface": "upload_pipeline",
                "job_id": job_id,
                "sample_key": None,
                "audience_mode": normalize_audience_mode(job.get("audience_mode")),
                "room_labeled": bool(job.get("room_label")),
                "host": "server",
                "created_at": str(job.get("completed_at") or _utc_now_iso()),
            }
        )
        upload_total.labels(outcome="completed").inc()

        pdf_bytes = reports._build_pdf_report(job)
        report_path = _upload_store.save_report_pdf(job_id, pdf_bytes)
        artifacts = _job_artifacts(job)
        artifacts["report_pdf"] = _upload_store.artifact_pointer(
            job_id,
            _upload_store.job_relative_path(job_id, report_path),
            kind="report_pdf",
            media_type="application/pdf",
            url=f"/reports/{job['job_id']}.pdf",
        )
        if evidence_artifacts:
            artifacts["evidence"] = evidence_artifacts
        if finding_replays:
            artifacts["finding_replays"] = finding_replays

        agent = _get_agent()
        for obj in result.objects:
            semantic_label = SemanticLabel(
                label=str(obj["label"]),
                material=str(obj["material"]),
                mass_kg=float(obj["mass_kg"]),
                fragility=float(obj["fragility"]),
                friction=float(obj["friction"]),
                confidence=float(obj["confidence"]),
            )
            position = tuple(float(v) for v in obj.get("position", [0.0, 0.8, 1.5]))
            await agent.ingest_from_upload(semantic_label, position=position)

        _upload_store.remove_job_input(job_id)
        artifacts.pop("queued_input", None)
        _save_job(job)
        upload_job_seconds.labels(outcome="completed").observe(
            max(0.0, (datetime.now(UTC) - started_wall).total_seconds())
        )
        _refresh_operational_metrics()
    except asyncio.CancelledError:
        logger.info("upload_processing_cancelled", job_id=job_id)
        raise
    except TimeoutError:
        logger.warning("upload_processing_timed_out", job_id=job_id)
        if job_id not in _upload_jobs or _is_upload_cancelled(job_id):
            return
        if attempt_count < _upload_cfg.max_job_attempts:
            _record_job_failure(
                job_id=job_id,
                stage=str(job.get("stage") or "processing"),
                error="Upload analysis timed out.",
                attempt_count=attempt_count,
                will_retry=True,
            )
            _update_job(
                job,
                status="queued",
                stage="upload",
                progress=0.05,
                error="Upload analysis timed out. Retrying automatically.",
                queued_at=_utc_now_iso(),
            )
            upload_total.labels(outcome="retried").inc()
            await _enqueue_upload_job(job_id)
            return

        _record_job_failure(
            job_id=job_id,
            stage=str(job.get("stage") or "processing"),
            error="Upload analysis timed out.",
            attempt_count=attempt_count,
            will_retry=False,
        )
        _update_job(
            job,
            status="error",
            stage="complete",
            progress=1.0,
            error="Upload analysis timed out.",
            completed_at=_utc_now_iso(),
        )
        _upload_store.remove_job_input(job_id)
        _set_job_artifact(job, "queued_input", None)
        upload_total.labels(outcome="timed_out").inc()
        upload_job_seconds.labels(outcome="timed_out").observe(
            max(0.0, (datetime.now(UTC) - started_wall).total_seconds())
        )
        _refresh_operational_metrics()
    except Exception as exc:
        logger.warning("upload_processing_failed", job_id=job_id, error=str(exc))
        if job_id not in _upload_jobs or _is_upload_cancelled(job_id):
            return
        if attempt_count < _upload_cfg.max_job_attempts:
            _record_job_failure(
                job_id=job_id,
                stage=str(job.get("stage") or "processing"),
                error=str(exc),
                attempt_count=attempt_count,
                will_retry=True,
            )
            _update_job(
                job,
                status="queued",
                stage="upload",
                progress=0.05,
                error=f"Retrying after failure: {exc}",
                queued_at=_utc_now_iso(),
            )
            upload_total.labels(outcome="retried").inc()
            await _enqueue_upload_job(job_id)
            return

        _record_job_failure(
            job_id=job_id,
            stage=str(job.get("stage") or "processing"),
            error=str(exc),
            attempt_count=attempt_count,
            will_retry=False,
        )
        _update_job(
            job,
            status="error",
            stage="complete",
            progress=1.0,
            error=str(exc),
            completed_at=_utc_now_iso(),
        )
        _upload_store.remove_job_input(job_id)
        _set_job_artifact(job, "queued_input", None)
        upload_total.labels(outcome="failed").inc()
        upload_job_seconds.labels(outcome="failed").observe(
            max(0.0, (datetime.now(UTC) - started_wall).total_seconds())
        )
        _refresh_operational_metrics()
    finally:
        if worker_id:
            _upload_store.release_job_claim(job_id, worker_id)
        _upload_cancelled_jobs().discard(job_id)


def _extract_single_file_from_multipart(
    body: bytes,
    content_type: str,
) -> tuple[str, str, bytes]:
    """Extract the ``file`` form part without requiring ``python-multipart``."""
    envelope = (f"Content-Type: {content_type}\r\n" "MIME-Version: 1.0\r\n\r\n").encode() + body
    message = BytesParser(policy=default).parsebytes(envelope)

    if not message.is_multipart():
        raise ValueError("Expected multipart/form-data payload.")

    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        if part.get_param("name", header="content-disposition") != "file":
            continue

        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename() or "upload.bin"
        part_content_type = part.get_content_type() or "application/octet-stream"
        return filename, part_content_type, payload

    raise ValueError("Multipart upload is missing the 'file' field.")


async def _read_upload_request(request: Request) -> tuple[str, str, bytes]:
    """Read an upload request in raw or multipart form."""
    content_type = request.headers.get("content-type", "")
    body = await _read_request_body_limited(request, _upload_cfg.max_upload_bytes)
    if not body:
        raise HTTPException(status_code=400, detail="Upload body is empty.")

    if content_type.startswith("multipart/form-data"):
        try:
            return _extract_single_file_from_multipart(body, content_type)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    filename = request.headers.get("x-filename", "upload.bin")
    fallback_type = content_type or "application/octet-stream"
    return filename, fallback_type, body


def _request_room_label(request: Request) -> str | None:
    """Extract an optional room label supplied by the frontend."""
    label = _normalize_room_label(request.headers.get("x-room-label"))
    if label and len(label) > 80:
        raise HTTPException(status_code=400, detail="Room label must be 80 characters or fewer.")
    return label


def _request_audience_mode(request: Request) -> str:
    """Extract the upload audience mode from request headers."""
    raw_mode = request.headers.get("x-audience-mode")
    mode = normalize_audience_mode(raw_mode)
    if raw_mode and normalize_audience_mode(raw_mode) == "general":
        cleaned = str(raw_mode).strip().lower().replace("-", "_").replace(" ", "_")
        if cleaned and cleaned != "general":
            raise HTTPException(
                status_code=400,
                detail="Audience mode must be general, toddler, pet, or renter.",
            )
    return mode


async def _read_request_body_limited(request: Request, max_bytes: int) -> bytes:
    """Read one request body while enforcing a maximum byte size."""
    content_length = request.headers.get("content-length")
    if content_length:
        with contextlib.suppress(ValueError):
            if int(content_length) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds the {max_bytes} byte limit.",
                )

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds the {max_bytes} byte limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_upload_constraints(content_type: str, content: bytes) -> None:
    """Reject uploads that violate configured size/type/duration constraints."""
    if len(content) > _upload_cfg.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {_upload_cfg.max_upload_bytes} byte limit.",
        )

    is_video = content_type in _VIDEO_TYPES or content_type.startswith("video/")
    if not is_video:
        return

    metadata = probe_video_metadata(content)
    if metadata is None:
        return
    if metadata.duration_s > _upload_cfg.max_video_duration_seconds:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Video duration {metadata.duration_s:.1f}s exceeds the "
                f"{_upload_cfg.max_video_duration_seconds:.1f}s limit."
            ),
        )

    max_video_pixels = int(getattr(_upload_cfg, "max_video_pixels", 0) or 0)
    if (
        max_video_pixels > 0
        and metadata.width > 0
        and metadata.height > 0
        and metadata.width * metadata.height > max_video_pixels
    ):
        raise HTTPException(
            status_code=413,
            detail=(
                f"Video resolution {metadata.width}x{metadata.height} exceeds the "
                f"{max_video_pixels}-pixel limit."
            ),
        )
