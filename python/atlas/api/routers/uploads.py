"""Upload, job lifecycle, evidence/replay, and report-download endpoints."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from atlas.api import pipeline, reports
from atlas.api.analytics import (
    _auto_benchmark_label,
    _compare_job_to_benchmark,
    _request_host,
    _require_demo_access,
    _require_private_access,
    _review_ready_for_eval,
    _storage_route_id,
)
from atlas.api.helpers import _finding_key, _normalize_follow_up_status, _utc_now_iso
from atlas.api.jobs import (
    _ensure_job_derived_fields,
    _feedback_counts,
    _refresh_operational_metrics,
    _refresh_upload_jobs_from_disk,
    _set_job_artifact,
    _share_path_for_job,
    _update_job,
)
from atlas.api.metrics import job_delete_total, report_download_total, upload_total
from atlas.api.models import (
    EvalCandidateRequest,
    FindingFeedbackRequest,
    FindingFollowUpRequest,
    JobEvaluationRequest,
    UploadJobStatus,
)
from atlas.api.pipeline import (
    _active_upload_job_count,
    _read_upload_request,
    _request_audience_mode,
    _request_room_label,
    _upload_cancelled_jobs,
    _validate_upload_constraints,
)
from atlas.api.state import _api_cfg, _state, _upload_cfg, _upload_jobs, _upload_store
from atlas.world_model.hazards import normalize_audience_mode

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/upload", response_model=UploadJobStatus)
async def upload_media(request: Request) -> UploadJobStatus:
    """Accept an image or video file and run it through the analysis pipeline.

    Returns immediately with a ``job_id``. Poll ``GET /jobs/{job_id}`` for
    status updates as the file moves through the pipeline stages:
    ``upload → ingest → vlm → risk → complete``.

    Supported image formats: JPEG, PNG, WEBP, GIF.
    Accepts both ``multipart/form-data`` browser uploads and raw file bodies.
    """
    _require_demo_access(request)

    if _active_upload_job_count() >= _upload_cfg.max_queue_depth:
        raise HTTPException(
            status_code=429,
            detail=(
                "Atlas-0 is already holding the maximum number of queued and"
                " active uploads. Please retry shortly."
            ),
        )

    filename, content_type, content = await _read_upload_request(request)
    room_label = _request_room_label(request)
    audience_mode = _request_audience_mode(request)
    _validate_upload_constraints(content_type, content)

    job_id = uuid.uuid4().hex[:8]
    now = _utc_now_iso()

    job: dict[str, Any] = {
        "job_id": job_id,
        "filename": filename,
        "room_label": room_label,
        "is_sample": False,
        "sample_key": None,
        "audience_mode": audience_mode,
        "content_type": content_type,
        "status": "queued",
        "stage": "upload",
        "progress": 0.0,
        "objects": None,
        "risks": None,
        "fix_first": None,
        "weekend_fix_list": None,
        "summary": None,
        "recommendations": None,
        "evidence_frames": None,
        "scan_quality": None,
        "trust_notes": None,
        "scene_source": None,
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "human_evaluation": None,
        "finding_follow_up": [],
        "resolution_summary": None,
        "room_wins": None,
        "report_url": None,
        "share_url": _share_path_for_job(job_id),
        "error": None,
        "artifacts": {},
        "attempt_count": 0,
        "queued_at": now,
        "started_at": None,
        "completed_at": None,
    }
    _upload_jobs[job_id] = job
    _upload_store.create_job(job)
    queued_input_path = _upload_store.save_job_input(job_id, filename, content)
    _set_job_artifact(
        job,
        "queued_input",
        _upload_store.artifact_pointer(
            job_id,
            _upload_store.job_relative_path(job_id, queued_input_path),
            kind="queued_input",
            media_type=content_type,
        ),
    )
    original_upload_path = _upload_store.save_original_upload(job_id, filename, content)
    if original_upload_path is not None:
        _set_job_artifact(
            job,
            "original_upload",
            _upload_store.artifact_pointer(
                job_id,
                _upload_store.job_relative_path(job_id, original_upload_path),
                kind="original_upload",
                media_type=content_type,
            ),
        )

    _update_job(job, stage="upload", progress=0.1)
    _upload_store.append_product_event(
        {
            "event_name": "upload_started",
            "surface": "upload_form",
            "job_id": job_id,
            "sample_key": None,
            "audience_mode": audience_mode,
            "room_labeled": bool(room_label),
            "host": _request_host(request),
            "created_at": now,
        }
    )
    upload_total.labels(outcome="accepted").inc()
    _refresh_operational_metrics()
    await pipeline._enqueue_upload_job(job_id)

    logger.info("upload_accepted", job_id=job_id, filename=filename, content_type=content_type)
    return UploadJobStatus(**job)


@router.get("/sample-report", response_model=UploadJobStatus)
async def get_sample_report() -> UploadJobStatus:
    """Return the built-in public sample walkthrough report."""
    cache = await pipeline._build_sample_report()
    _upload_store.append_product_event(
        {
            "event_name": "sample_report_opened",
            "surface": "sample_report",
            "job_id": cache["job"].get("job_id"),
            "sample_key": cache["job"].get("sample_key"),
            "audience_mode": normalize_audience_mode(cache["job"].get("audience_mode")),
            "room_labeled": bool(cache["job"].get("room_label")),
            "host": "server",
            "created_at": _utc_now_iso(),
        }
    )
    _refresh_operational_metrics()
    return UploadJobStatus(**cache["job"])


@router.get("/jobs", response_model=list[UploadJobStatus])
def list_upload_jobs(request: Request) -> list[UploadJobStatus]:
    """List all upload jobs and their current status."""
    _require_private_access(request)
    _refresh_upload_jobs_from_disk()
    if not _api_cfg.enable_job_listing:
        raise HTTPException(
            status_code=403,
            detail="Job listing is disabled. Query a known job ID directly instead.",
        )
    return [UploadJobStatus(**_ensure_job_derived_fields(j)) for j in _upload_jobs.values()]


@router.get("/jobs/{job_id}", response_model=UploadJobStatus)
def get_upload_job(job_id: str, request: Request) -> UploadJobStatus:
    """Return the current status of a specific upload job.

    Args:
        job_id: The 8-character hex job ID returned by ``POST /upload``.

    Raises:
        HTTPException: 404 if the job is not found.
    """
    _require_demo_access(request)
    job_id = _storage_route_id(job_id, "job_id")
    _refresh_upload_jobs_from_disk()
    if job_id not in _upload_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return UploadJobStatus(**_ensure_job_derived_fields(_upload_jobs[job_id]))


@router.post("/jobs/{job_id}/follow-up", response_model=UploadJobStatus)
def record_job_follow_up(
    job_id: str,
    payload: FindingFollowUpRequest,
    request: Request,
) -> UploadJobStatus:
    """Store a persistent follow-up state for one completed report finding."""
    _require_demo_access(request)
    job_id = _storage_route_id(job_id, "job_id")
    _refresh_upload_jobs_from_disk()
    job = _upload_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    if job.get("status") != "complete":
        raise HTTPException(
            status_code=409,
            detail="Follow-up is only accepted for completed jobs.",
        )

    normalized_status = str(payload.status or "").strip().lower()
    if normalized_status not in {"resolved", "monitor", "ignored", "open"}:
        raise HTTPException(
            status_code=400,
            detail="Follow-up status must be resolved, monitor, ignored, or open.",
        )
    note = str(payload.note or "").strip()
    if len(note) > 280:
        raise HTTPException(
            status_code=400,
            detail="Follow-up note must be 280 characters or fewer.",
        )

    risks = list(job.get("risks") or [])
    target = next(
        (
            risk
            for risk in risks
            if str(risk.get("hazard_code", "")) == payload.hazard_code
            and (payload.object_id is None or str(risk.get("object_id", "")) == payload.object_id)
        ),
        None,
    )
    if target is None:
        raise HTTPException(status_code=404, detail="Matching finding was not found in the report.")

    status = _normalize_follow_up_status(normalized_status)
    created_at = _utc_now_iso() if status else None
    target["follow_up_status"] = status
    target["follow_up_updated_at"] = created_at
    target["follow_up_note"] = (note or None) if status else None
    event = {
        "hazard_code": payload.hazard_code,
        "object_id": payload.object_id or target.get("object_id"),
        "status": normalized_status,
        "note": note or None,
        "created_at": created_at or _utc_now_iso(),
        "finding_key": _finding_key(target),
    }
    events = list(job.get("finding_follow_up") or [])
    events.append(event)
    _update_job(job, risks=risks, finding_follow_up=events)
    return UploadJobStatus(**job)


@router.post("/jobs/{job_id}/feedback", response_model=UploadJobStatus)
def record_job_feedback(
    job_id: str,
    payload: FindingFeedbackRequest,
    request: Request,
) -> UploadJobStatus:
    """Store user feedback for one finding in a completed upload report."""
    _require_demo_access(request)
    job_id = _storage_route_id(job_id, "job_id")
    _refresh_upload_jobs_from_disk()
    job = _upload_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    if job.get("status") != "complete":
        raise HTTPException(status_code=409, detail="Feedback is only accepted for completed jobs.")

    verdict = payload.verdict.lower().strip()
    if verdict not in {"useful", "wrong", "duplicate"}:
        raise HTTPException(
            status_code=400,
            detail="Feedback verdict must be useful, wrong, or duplicate.",
        )
    note = (payload.note or "").strip()
    if len(note) > 500:
        raise HTTPException(
            status_code=400,
            detail="Feedback note must be 500 characters or fewer.",
        )

    risks = job.get("risks") or []
    target = next(
        (
            risk
            for risk in risks
            if str(risk.get("hazard_code", "")) == payload.hazard_code
            and (payload.object_id is None or str(risk.get("object_id", "")) == payload.object_id)
        ),
        None,
    )
    if target is None:
        raise HTTPException(status_code=404, detail="Matching finding was not found in the report.")

    event = {
        "hazard_code": payload.hazard_code,
        "object_id": payload.object_id or target.get("object_id"),
        "verdict": verdict,
        "note": note or None,
        "created_at": datetime.now(UTC).isoformat(),
        "finding_key": _finding_key(target),
    }

    events = list(job.get("finding_feedback") or [])
    events.append(event)
    counts = dict(target.get("feedback_summary") or {"useful": 0, "wrong": 0, "duplicate": 0})
    counts[verdict] = counts.get(verdict, 0) + 1
    target["feedback_summary"] = counts
    target["latest_feedback"] = verdict

    _update_job(
        job,
        risks=risks,
        finding_feedback=events,
        feedback_summary=_feedback_counts(events),
    )
    return UploadJobStatus(**job)


@router.post("/jobs/{job_id}/evaluation", response_model=UploadJobStatus)
def record_job_evaluation(
    job_id: str,
    payload: JobEvaluationRequest,
    request: Request,
) -> UploadJobStatus:
    """Store one human review verdict for a completed report."""
    _require_private_access(request)
    job_id = _storage_route_id(job_id, "job_id")
    _refresh_upload_jobs_from_disk()
    job = _upload_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    if job.get("status") != "complete":
        raise HTTPException(
            status_code=409,
            detail="Evaluation is only accepted for completed jobs.",
        )

    status = payload.status.strip().lower()
    if status not in {"confirmed", "needs_review", "missed_hazard"}:
        raise HTTPException(
            status_code=400,
            detail="Evaluation status must be confirmed, needs_review, or missed_hazard.",
        )

    note = (payload.note or "").strip()
    if len(note) > 500:
        raise HTTPException(
            status_code=400,
            detail="Evaluation note must be 500 characters or fewer.",
        )

    missed_hazards = [item.strip() for item in (payload.missed_hazards or []) if item.strip()]
    if len(missed_hazards) > 8:
        raise HTTPException(status_code=400, detail="At most 8 missed hazards may be submitted.")

    benchmark_label = (payload.benchmark_label or "").strip() or _auto_benchmark_label(job)
    benchmark = _compare_job_to_benchmark(job, benchmark_label) if benchmark_label else None

    _update_job(
        job,
        human_evaluation={
            "status": status,
            "benchmark_label": benchmark_label,
            "benchmark": benchmark,
            "missed_hazards": missed_hazards,
            "note": note or None,
            "reviewed_at": datetime.now(UTC).isoformat(),
        },
    )
    return UploadJobStatus(**job)


@router.post("/jobs/{job_id}/eval-candidate", response_model=UploadJobStatus)
def export_eval_candidate(
    job_id: str,
    payload: EvalCandidateRequest,
    request: Request,
) -> UploadJobStatus:
    """Export one reviewed completed report into the persisted eval-candidate store."""
    _require_private_access(request)
    job_id = _storage_route_id(job_id, "job_id")
    _refresh_upload_jobs_from_disk()
    job = _upload_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    if job.get("status") != "complete":
        raise HTTPException(
            status_code=409,
            detail="Eval candidates can only be exported from completed jobs.",
        )
    if not _review_ready_for_eval(job):
        raise HTTPException(
            status_code=409,
            detail="Add human review or finding feedback before exporting this eval candidate.",
        )

    note = str(payload.note or "").strip()
    if len(note) > 500:
        raise HTTPException(
            status_code=400,
            detail="Eval candidate note must be 500 characters or fewer.",
        )
    requested_label = str(payload.label or "").strip().lower().replace(" ", "_").replace("-", "_")
    label = requested_label or str(job.get("job_id"))
    if len(label) > 80:
        raise HTTPException(
            status_code=400,
            detail="Eval candidate label must be 80 characters or fewer.",
        )
    label = _storage_route_id(label, "candidate_id")

    summary = dict(job.get("summary") or {})
    evaluation = dict(job.get("evaluation_summary") or {})
    human = dict(job.get("human_evaluation") or {})
    hazard_codes = [str(risk.get("hazard_code", "")) for risk in list(job.get("risks") or [])]
    candidate = {
        "candidate_id": label,
        "job_id": job_id,
        "filename": job.get("filename"),
        "room_label": job.get("room_label"),
        "audience_mode": normalize_audience_mode(job.get("audience_mode")),
        "scene_source": job.get("scene_source"),
        "top_hazard_code": hazard_codes[0] if hazard_codes else None,
        "hazard_codes": hazard_codes,
        "hazard_count": int(summary.get("hazard_count", 0) or 0),
        "object_count": int(summary.get("object_count", 0) or 0),
        "review_ready": True,
        "evaluation_summary": evaluation,
        "human_evaluation": human or None,
        "benchmark_label": evaluation.get("benchmark_label"),
        "export_note": note or None,
        "exported_at": _utc_now_iso(),
        "source": "operator_export",
    }
    _upload_store.save_eval_candidate(label, candidate)
    _update_job(
        job,
        human_evaluation={
            **human,
            "eval_candidate_label": label,
            "eval_candidate_exported_at": candidate["exported_at"],
        },
    )
    return UploadJobStatus(**job)


@router.get("/jobs/{job_id}/evidence/{evidence_id}")
def download_evidence(job_id: str, evidence_id: str, request: Request) -> Response:
    """Download one persisted evidence crop for a completed report."""
    _require_demo_access(request)
    job_id = _storage_route_id(job_id, "job_id")
    evidence_id = _storage_route_id(evidence_id, "evidence_id")
    _refresh_upload_jobs_from_disk()
    if job_id not in _upload_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")

    artifact = _upload_store.load_evidence_image(job_id, evidence_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Evidence artifact not found.")

    content, media_type = artifact
    return Response(content=content, media_type=media_type)


@router.get("/sample-report/evidence/{evidence_id}")
async def download_sample_evidence(evidence_id: str) -> Response:
    """Download one evidence crop from the built-in sample report."""
    cache = await pipeline._build_sample_report()
    payload = cache["evidence_artifacts"].get(evidence_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Evidence {evidence_id!r} not found")
    return Response(content=payload, media_type="image/jpeg")


@router.get("/jobs/{job_id}/replays/{replay_id}")
def download_finding_replay(job_id: str, replay_id: str, request: Request) -> Response:
    """Download one persisted finding replay for a completed report."""
    _require_demo_access(request)
    job_id = _storage_route_id(job_id, "job_id")
    replay_id = _storage_route_id(replay_id, "replay_id")
    _refresh_upload_jobs_from_disk()
    if job_id not in _upload_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")

    artifact = _upload_store.load_replay_gif(job_id, replay_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Finding replay not found.")

    content, media_type = artifact
    return Response(content=content, media_type=media_type)


@router.get("/sample-report/replays/{replay_id}")
async def download_sample_replay(replay_id: str) -> Response:
    """Download one replay artifact from the built-in sample report."""
    cache = await pipeline._build_sample_report()
    payload = cache["replay_artifacts"].get(replay_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Replay {replay_id!r} not found")
    return Response(content=payload, media_type="image/gif")


@router.delete("/jobs/{job_id}", status_code=204)
def delete_upload_job(job_id: str, request: Request) -> Response:
    """Delete one persisted upload job and its artifacts."""
    _require_private_access(request)
    job_id = _storage_route_id(job_id, "job_id")
    _refresh_upload_jobs_from_disk()
    cancelled = _upload_cancelled_jobs()
    cancelled.add(job_id)
    existed = job_id in _upload_jobs
    removed = _upload_store.delete_job(job_id)
    _upload_jobs.pop(job_id, None)
    if not existed and not removed:
        cancelled.discard(job_id)
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    job_delete_total.inc()
    _state["last_prune_at"] = _utc_now_iso()
    _refresh_operational_metrics()
    return Response(status_code=204)


@router.get("/reports/{job_id}.pdf")
def download_report(job_id: str, request: Request) -> Response:
    """Download a generated PDF report for a completed scan job."""
    _require_demo_access(request)
    job_id = _storage_route_id(job_id, "job_id")
    _refresh_upload_jobs_from_disk()
    job = _upload_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    if job.get("status") != "complete":
        raise HTTPException(status_code=409, detail="Report is not ready yet.")

    pdf_bytes = _upload_store.load_report_pdf(job_id)
    if pdf_bytes is None:
        pdf_bytes = reports._build_pdf_report(job)
        report_path = _upload_store.save_report_pdf(job_id, pdf_bytes)
        _set_job_artifact(
            job,
            "report_pdf",
            _upload_store.artifact_pointer(
                job_id,
                _upload_store.job_relative_path(job_id, report_path),
                kind="report_pdf",
                media_type="application/pdf",
                url=f"/reports/{job_id}.pdf",
            ),
        )
    _upload_store.append_product_event(
        {
            "event_name": "report_pdf_downloaded",
            "surface": "report_pdf",
            "job_id": job_id,
            "sample_key": None,
            "audience_mode": normalize_audience_mode(job.get("audience_mode")),
            "room_labeled": bool(job.get("room_label")),
            "host": _request_host(request),
            "created_at": _utc_now_iso(),
        }
    )
    report_download_total.labels(source="job").inc()
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="atlas0-report-{job_id}.pdf"'},
    )


@router.get("/sample-report/report.pdf")
async def download_sample_report() -> Response:
    """Download the built-in sample walkthrough PDF."""
    cache = await pipeline._build_sample_report()
    _upload_store.append_product_event(
        {
            "event_name": "report_pdf_downloaded",
            "surface": "sample_report_pdf",
            "job_id": cache["job"].get("job_id"),
            "sample_key": cache["job"].get("sample_key"),
            "audience_mode": normalize_audience_mode(cache["job"].get("audience_mode")),
            "room_labeled": bool(cache["job"].get("room_label")),
            "host": "server",
            "created_at": _utc_now_iso(),
        }
    )
    report_download_total.labels(source="sample").inc()
    return Response(
        content=cache["report_pdf"],
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="atlas0-sample-report.pdf"'},
    )
