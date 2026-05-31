"""Upload-job persistence, derived report fields, and operational metrics.

Owns reading/writing job manifests, computing derived job-view fields
(resolution, evaluation, room score/history/comparison, share links), artifact
pointer rebuilding, and the gauges derived from queue/storage state. Depends
only on :mod:`atlas.api.state` and leaf modules to keep the graph acyclic.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import structlog

from atlas.api.helpers import (
    _iso_sort_key,
    _normalize_follow_up_status,
    _normalize_room_label,
    _utc_now_iso,
)
from atlas.api.metrics import (
    storage_bytes,
    upload_queue_depth,
    upload_workers_active,
)
from atlas.api.state import (
    _state,
    _upload_cfg,
    _upload_jobs,
    _upload_store,
)
from atlas.world_model.hazards import (
    audience_mode_label,
    build_room_wins,
    build_weekend_fix_list,
    normalize_audience_mode,
)

logger = structlog.get_logger(__name__)


def _recent_job_failures() -> list[dict[str, Any]]:
    """Return the bounded in-memory job failure log."""
    items = _state.setdefault("recent_job_failures", [])
    return items if isinstance(items, list) else []


def _record_job_failure(
    *,
    job_id: str,
    stage: str,
    error: str,
    attempt_count: int,
    will_retry: bool,
) -> None:
    """Append one failure event to the bounded operator failure log."""
    failures = _recent_job_failures()
    failures.append(
        {
            "job_id": job_id,
            "stage": stage,
            "error": error[:240],
            "attempt_count": attempt_count,
            "will_retry": will_retry,
            "created_at": _utc_now_iso(),
        }
    )
    limit = max(1, int(_upload_cfg.job_failure_log_limit or 20))
    if len(failures) > limit:
        del failures[:-limit]


def _set_worker_activity(active_workers: int) -> None:
    """Store and export the number of active upload workers."""
    _state["active_upload_workers"] = max(0, active_workers)
    upload_workers_active.set(int(_state["active_upload_workers"]))


def _active_external_worker_records() -> list[dict[str, Any]]:
    """Return detached workers whose durable heartbeat is still fresh."""
    return _upload_store.active_worker_records(
        stale_after_seconds=_upload_cfg.worker_stale_after_seconds,
    )


def _effective_active_worker_count() -> int:
    """Return the best operator-facing active worker count for the current mode."""
    if _upload_cfg.worker_mode == "external":
        return len(_active_external_worker_records())
    return int(_state.get("active_upload_workers", 0) or 0)


def _refresh_operational_metrics() -> None:
    """Refresh gauges derived from queue depth and persisted storage."""
    queue_depth = _job_status_counts().get("queued", 0)
    upload_queue_depth.set(queue_depth)
    upload_workers_active.set(_effective_active_worker_count())
    storage = _upload_store.storage_summary()
    storage_bytes.set(int(storage.get("bytes_used", 0) or 0))


def _refresh_upload_jobs_from_disk() -> None:
    """Reload persisted upload jobs so API and detached workers share job state."""
    persisted = _upload_store.load_jobs()
    for job_id, payload in persisted.items():
        _upload_jobs[job_id] = payload
    for job in _upload_jobs.values():
        _ensure_job_derived_fields(job)
        if _upload_store.manifest_path(str(job.get("job_id") or "")).exists():
            _refresh_job_artifacts(job)


def _save_job(job: dict[str, Any]) -> None:
    """Persist the current in-memory job manifest to disk."""
    _upload_store.save_job(job)
    _refresh_operational_metrics()


def _update_job(job: dict[str, Any], **fields: Any) -> None:
    """Update a job dict and persist the new state."""
    job.update(fields)
    _ensure_job_derived_fields(job)
    _save_job(job)


def _job_artifacts(job: dict[str, Any]) -> dict[str, Any]:
    """Return the mutable artifact map stored on one job."""
    artifacts = job.get("artifacts")
    if isinstance(artifacts, dict):
        return artifacts
    fresh: dict[str, Any] = {}
    job["artifacts"] = fresh
    return fresh


def _set_job_artifact(job: dict[str, Any], name: str, pointer: dict[str, Any] | None) -> None:
    """Attach or remove one artifact pointer on a job."""
    artifacts = _job_artifacts(job)
    if pointer is None:
        artifacts.pop(name, None)
    else:
        artifacts[name] = pointer


def _refresh_job_artifacts(job: dict[str, Any]) -> None:
    """Rebuild artifact pointers for one persisted job from files on disk."""
    job_id = str(job.get("job_id", ""))
    if not job_id:
        return

    _job_artifacts(job)

    if _upload_store.has_job_input(job_id):
        queued_path = next(_upload_store.job_dir(job_id).glob("queued-input.*"), None)
        if queued_path is not None:
            _set_job_artifact(
                job,
                "queued_input",
                _upload_store.artifact_pointer(
                    job_id,
                    _upload_store.job_relative_path(job_id, queued_path),
                    kind="queued_input",
                    media_type=str(job.get("content_type") or "application/octet-stream"),
                ),
            )
    else:
        _set_job_artifact(job, "queued_input", None)

    upload_path = next(_upload_store.artifact_job_dir(job_id).glob("upload.*"), None)
    if upload_path is not None:
        _set_job_artifact(
            job,
            "original_upload",
            _upload_store.artifact_pointer(
                job_id,
                _upload_store.job_relative_path(job_id, upload_path),
                kind="original_upload",
                media_type=str(job.get("content_type") or "application/octet-stream"),
            ),
        )
    else:
        _set_job_artifact(job, "original_upload", None)

    report_path = _upload_store.report_path(job_id)
    if report_path.exists():
        _set_job_artifact(
            job,
            "report_pdf",
            _upload_store.artifact_pointer(
                job_id,
                _upload_store.job_relative_path(job_id, report_path),
                kind="report_pdf",
                media_type="application/pdf",
                url=str(job.get("report_url") or f"/reports/{job_id}.pdf"),
            ),
        )
    else:
        _set_job_artifact(job, "report_pdf", None)

    evidence_artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(_upload_store.evidence_dir(job_id).glob("*")):
        if not path.is_file():
            continue
        evidence_id = path.stem
        media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        evidence_artifacts[evidence_id] = _upload_store.artifact_pointer(
            job_id,
            _upload_store.job_relative_path(job_id, path),
            kind="evidence_image",
            media_type=media_type,
            url=f"/jobs/{job_id}/evidence/{evidence_id}",
        )
    _set_job_artifact(job, "evidence", evidence_artifacts or None)

    for frame in job.get("evidence_frames") or []:
        evidence_id = str(frame.get("evidence_id", ""))
        if evidence_id and evidence_id in evidence_artifacts:
            frame["artifact"] = evidence_artifacts[evidence_id]

    replay_artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(_upload_store.replay_dir(job_id).glob("*.gif")):
        if not path.is_file():
            continue
        replay_id = path.stem
        replay_artifacts[replay_id] = _upload_store.artifact_pointer(
            job_id,
            _upload_store.job_relative_path(job_id, path),
            kind="finding_replay",
            media_type="image/gif",
            url=f"/jobs/{job_id}/replays/{replay_id}",
        )
    _set_job_artifact(job, "finding_replays", replay_artifacts or None)

    for risk in job.get("risks") or []:
        replay = risk.get("replay")
        if not isinstance(replay, dict):
            continue
        replay_id = str(replay.get("replay_id", ""))
        if replay_id and replay_id in replay_artifacts:
            replay["image_url"] = replay_artifacts[replay_id]["url"]
            replay["media_type"] = replay_artifacts[replay_id]["media_type"]
            replay["artifact"] = replay_artifacts[replay_id]


def _feedback_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    """Return aggregate verdict counts for stored feedback events."""
    counts = {"useful": 0, "wrong": 0, "duplicate": 0}
    for event in events:
        verdict = str(event.get("verdict", "")).lower()
        if verdict in counts:
            counts[verdict] += 1
    return counts


def _apply_follow_up_events(job: dict[str, Any]) -> None:
    """Project persisted follow-up events onto the current risk list."""
    latest_by_key: dict[str, dict[str, Any]] = {}
    for event in list(job.get("finding_follow_up") or []):
        finding_key = str(event.get("finding_key", "")).strip()
        if finding_key:
            latest_by_key[finding_key] = event

    for risk in list(job.get("risks") or []):
        current_status = _normalize_follow_up_status(risk.get("follow_up_status"))
        finding_key = (
            f"{risk.get('object_id') or risk.get('object_label') or 'finding'}:"
            f"{risk.get('hazard_code') or 'unknown'}"
        )
        event = latest_by_key.get(finding_key)
        if event is None:
            risk["follow_up_status"] = current_status
            continue

        status = _normalize_follow_up_status(event.get("status"))
        risk["follow_up_status"] = status
        risk["follow_up_updated_at"] = event.get("created_at") if status else None
        note = str(event.get("note") or "").strip()
        risk["follow_up_note"] = (note or None) if status else None


def _build_resolution_summary(job: dict[str, Any]) -> dict[str, Any]:
    """Summarize the current follow-up state for one completed report."""
    risks = list(job.get("risks") or [])
    counts = {"resolved": 0, "monitor": 0, "ignored": 0, "open": 0}

    for risk in risks:
        status = _normalize_follow_up_status(risk.get("follow_up_status"))
        if status is None:
            counts["open"] += 1
        else:
            counts[status] += 1

    total_findings = len(risks)
    acted_on_count = counts["resolved"] + counts["monitor"] + counts["ignored"]
    progress_ratio = round(acted_on_count / total_findings, 2) if total_findings else 0.0
    if total_findings == 0:
        summary = "No findings need follow-up yet."
    else:
        summary = (
            f"{counts['resolved']} resolved, {counts['monitor']} being monitored,"
            f" {counts['ignored']} intentionally ignored, and {counts['open']}"
            " still open."
        )

    return {
        "total_findings": total_findings,
        "resolved_count": counts["resolved"],
        "monitor_count": counts["monitor"],
        "ignored_count": counts["ignored"],
        "open_count": counts["open"],
        "acted_on_count": acted_on_count,
        "progress_ratio": progress_ratio,
        "summary": summary,
    }


def _build_evaluation_summary(job: dict[str, Any]) -> dict[str, Any]:
    """Summarize report review coverage and correction signals."""
    risks = list(job.get("risks") or [])
    events = list(job.get("finding_feedback") or [])
    human_evaluation = dict(job.get("human_evaluation") or {})
    event_counts = _feedback_counts(events)
    total_findings = len(risks)
    reviewed_findings = 0
    disputed_findings = 0
    helpful_findings = 0
    high_priority_pending = 0

    for risk in risks:
        counts = dict(risk.get("feedback_summary") or {})
        useful = int(counts.get("useful", 0) or 0)
        wrong = int(counts.get("wrong", 0) or 0)
        duplicate = int(counts.get("duplicate", 0) or 0)
        total = useful + wrong + duplicate
        if total > 0:
            reviewed_findings += 1
        if wrong > 0 or duplicate > 0:
            disputed_findings += 1
        if useful > 0:
            helpful_findings += 1
        if total == 0 and str(risk.get("severity", "")).lower() in {"critical", "high"}:
            high_priority_pending += 1

    pending_findings = max(0, total_findings - reviewed_findings)
    review_coverage = round(reviewed_findings / total_findings, 2) if total_findings else 0.0
    missed_hazard_count = len(human_evaluation.get("missed_hazards") or [])
    benchmark = dict(human_evaluation.get("benchmark") or {})
    benchmark_label = benchmark.get("label") or human_evaluation.get("benchmark_label")
    benchmark_match = benchmark.get("matched")
    precision_proxy = (
        round(event_counts["useful"] / max(1, sum(event_counts.values())), 2)
        if sum(event_counts.values()) > 0
        else 0.0
    )
    recall_proxy = (
        round(total_findings / (total_findings + missed_hazard_count), 2)
        if total_findings or missed_hazard_count
        else 0.0
    )

    if total_findings == 0:
        summary = "No findings to review yet."
    elif reviewed_findings == 0:
        summary = (
            f"No findings have feedback yet. {high_priority_pending} higher-priority"
            " finding(s) still need human review."
        )
    else:
        summary = (
            f"{reviewed_findings} of {total_findings} finding"
            f"{'' if total_findings == 1 else 's'} reviewed."
            f" {disputed_findings} marked wrong or duplicate so far."
        )

    human_status = str(human_evaluation.get("status", "")).strip() or None
    if human_status == "missed_hazard":
        summary += f" Human review flagged {missed_hazard_count} missed hazard(s)."
    elif human_status == "needs_review":
        summary += " Human review says the report still needs follow-up."
    elif human_status == "confirmed":
        summary += " Human review marked the current report as directionally sound."
    if benchmark_label:
        summary += (
            f" Benchmark {benchmark_label} " f"{'matched' if benchmark_match else 'did not match'}."
        )

    eval_actions: list[str] = []
    if high_priority_pending > 0:
        eval_actions.append("Review high-priority findings before using this as a release signal.")
    if missed_hazard_count > 0:
        eval_actions.append("Add the missed hazard note to the labeled eval corpus.")
    if benchmark_match is False:
        eval_actions.append("Compare against the benchmark fixture and label the mismatch reason.")
    if disputed_findings > 0:
        eval_actions.append(
            "Tag wrong or duplicate findings so future eval slices can track false positives."
        )
    if not eval_actions:
        eval_actions.append("Export as a clean candidate after one human review pass.")
    if missed_hazard_count > 0 or benchmark_match is False:
        eval_priority = "high"
    elif high_priority_pending > 0 or pending_findings > 0 or disputed_findings > 0:
        eval_priority = "medium"
    else:
        eval_priority = "ready"

    return {
        "total_findings": total_findings,
        "reviewed_findings": reviewed_findings,
        "pending_findings": pending_findings,
        "review_coverage": review_coverage,
        "useful_events": event_counts["useful"],
        "wrong_events": event_counts["wrong"],
        "duplicate_events": event_counts["duplicate"],
        "helpful_findings": helpful_findings,
        "disputed_findings": disputed_findings,
        "high_priority_pending": high_priority_pending,
        "needs_review": high_priority_pending > 0
        or pending_findings > 0
        or human_status in {"needs_review", "missed_hazard"},
        "human_status": human_status,
        "missed_hazard_count": missed_hazard_count,
        "benchmark_label": benchmark_label,
        "benchmark_match": benchmark_match,
        "benchmark_summary": benchmark,
        "precision_proxy": precision_proxy,
        "recall_proxy": recall_proxy,
        "eval_priority": eval_priority,
        "eval_corpus_actions": eval_actions[:4],
        "eval_candidate_reason": (
            "missed hazard or benchmark mismatch"
            if eval_priority == "high"
            else "needs review coverage"
            if eval_priority == "medium"
            else "ready after human confirmation"
        ),
        "summary": summary,
    }


def _share_path_for_job(job_id: str) -> str:
    """Return the frontend deep link for one upload report."""
    return f"/app?view=report&job={job_id}"


def _share_summary(job: dict[str, Any]) -> str:
    """Create a short share-safe summary for one completed report."""
    summary = dict(job.get("summary") or {})
    audience_label = str(
        summary.get("audience_label") or audience_mode_label(job.get("audience_mode"))
    )
    room_label = _normalize_room_label(job.get("room_label") or summary.get("room_label"))
    headline = str(summary.get("headline") or "ATLAS-0 room scan")
    room_score = summary.get("room_score")
    score_text = f" · {room_score}/100 room score" if isinstance(room_score, int | float) else ""
    room_text = f"{room_label} · " if room_label else ""
    return f"{room_text}{audience_label} · {headline}{score_text}"


def _job_expiry_iso(job: dict[str, Any]) -> str | None:
    """Return the expected artifact expiry timestamp for one persisted job."""
    retention_days = int(_upload_cfg.retention_days or 0)
    if retention_days <= 0:
        return None
    anchor = str(job.get("completed_at") or job.get("queued_at") or "").strip()
    if not anchor:
        return None
    try:
        expires = datetime.fromisoformat(anchor) + timedelta(days=retention_days)
    except ValueError:
        return None
    return expires.isoformat()


def _room_score_payload(job: dict[str, Any]) -> dict[str, Any] | None:
    """Compute a lightweight room safety score for repeat-use comparisons."""
    if str(job.get("status")) != "complete":
        return None

    summary = dict(job.get("summary") or {})
    scan_quality = dict(job.get("scan_quality") or {})
    risks = list(job.get("risks") or [])
    weights = [30, 18, 10, 6, 4]
    hazard_penalty = 0.0
    for index, risk in enumerate(
        sorted(
            risks,
            key=lambda item: float(item.get("priority_score", item.get("risk_score", 0.0))),
            reverse=True,
        )[: len(weights)]
    ):
        hazard_penalty += weights[index] * float(
            risk.get("priority_score", risk.get("risk_score", 0.0)) or 0.0
        )

    quality_penalty = 0.0
    status = str(scan_quality.get("status", "unknown")).lower()
    if status == "fair":
        quality_penalty += 6.0
    elif status == "poor":
        quality_penalty += 14.0
    if bool(scan_quality.get("rescan_recommended")):
        quality_penalty += 8.0
    if str(summary.get("analysis_outcome", "accepted")).lower() == "rejected":
        quality_penalty += 20.0

    score = max(0, min(100, int(round(100.0 - min(78.0, hazard_penalty) - quality_penalty))))
    if score >= 82:
        band = "safer now"
        summary_text = (
            "Lower apparent risk in the current scan, though this is still a screening result."
        )
    elif score >= 65:
        band = "needs fixes"
        summary_text = (
            "A few concentrated risks still deserve follow-up before calling the room calm."
        )
    else:
        band = "high attention"
        summary_text = (
            "ATLAS-0 sees enough concentrated risk here that a follow-up pass"
            " should start with the top actions."
        )

    return {
        "room_score": score,
        "room_score_band": band,
        "room_score_summary": summary_text,
    }


def _room_history(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Return recent completed scans for the same labeled room."""
    room_label = _normalize_room_label(
        job.get("room_label") or (job.get("summary") or {}).get("room_label")
    )
    if room_label is None:
        return []
    audience_mode = normalize_audience_mode(job.get("audience_mode"))

    siblings = [
        candidate
        for candidate in _upload_jobs.values()
        if str(candidate.get("status")) == "complete"
        and candidate.get("job_id") != job.get("job_id")
        and normalize_audience_mode(candidate.get("audience_mode")) == audience_mode
        and _normalize_room_label(
            candidate.get("room_label") or (candidate.get("summary") or {}).get("room_label")
        )
        == room_label
    ]
    siblings.sort(key=_iso_sort_key, reverse=True)
    history: list[dict[str, Any]] = []
    for candidate in siblings[:4]:
        candidate_summary = dict(candidate.get("summary") or {})
        score_payload = _room_score_payload(candidate)
        if score_payload:
            candidate_summary.update(score_payload)
        history.append(
            {
                "job_id": candidate.get("job_id"),
                "filename": candidate.get("filename"),
                "completed_at": candidate.get("completed_at"),
                "room_score": candidate_summary.get("room_score"),
                "hazard_count": candidate_summary.get("hazard_count"),
                "top_severity": candidate_summary.get("top_severity"),
                "audience_mode": normalize_audience_mode(candidate.get("audience_mode")),
            }
        )
    return history


def _comparison_evidence_snapshot(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a compact evidence snapshot for before/after UI comparison."""
    frames = list(job.get("evidence_frames") or [])
    snapshot: list[dict[str, Any]] = []
    for index, frame in enumerate(frames[:3]):
        if not isinstance(frame, dict):
            continue
        snapshot.append(
            {
                "evidence_id": frame.get("evidence_id") or f"frame-{index + 1}",
                "caption": frame.get("caption")
                or frame.get("object_label")
                or frame.get("hazard_title")
                or "Evidence frame",
                "image_url": frame.get("image_url"),
                "confidence": frame.get("confidence"),
                "redacted": bool(frame.get("redacted")),
            }
        )
    return snapshot


def _room_comparison(job: dict[str, Any]) -> dict[str, Any] | None:
    """Compare the current room scan to the most recent previous scan."""
    history = _room_history(job)
    if not history:
        return None

    current_summary = dict(job.get("summary") or {})
    previous = history[0]
    current_score = current_summary.get("room_score")
    previous_score = previous.get("room_score")
    if current_score is None or previous_score is None:
        return None

    previous_job = _upload_jobs.get(str(previous.get("job_id") or ""))
    delta = int(current_score) - int(previous_score)
    hazard_delta = int(current_summary.get("hazard_count", 0) or 0) - int(
        previous.get("hazard_count", 0) or 0
    )
    trend = "improved" if delta > 0 else "worse" if delta < 0 else "flat"
    return {
        "previous_job_id": previous.get("job_id"),
        "previous_filename": previous.get("filename"),
        "previous_completed_at": previous.get("completed_at"),
        "previous_room_score": previous_score,
        "current_room_score": current_score,
        "score_delta": delta,
        "hazard_delta": hazard_delta,
        "trend": trend,
        "previous_evidence": _comparison_evidence_snapshot(previous_job or {}),
        "current_evidence": _comparison_evidence_snapshot(job),
        "summary": (
            "This room looks safer than the last saved scan."
            if trend == "improved"
            else "This room looks riskier than the last saved scan."
            if trend == "worse"
            else "This room looks broadly similar to the last saved scan."
        ),
    }


def _ensure_job_derived_fields(job: dict[str, Any]) -> dict[str, Any]:
    """Populate lightweight derived fields for older or manually inserted jobs."""
    if job.get("risks") is not None:
        _apply_follow_up_events(job)
        job["resolution_summary"] = _build_resolution_summary(job)
        job["evaluation_summary"] = _build_evaluation_summary(job)
    summary = dict(job.get("summary") or {})
    audience_mode = normalize_audience_mode(job.get("audience_mode"))
    job["audience_mode"] = audience_mode
    sample_key = str(job.get("sample_key") or "").strip()
    job["share_url"] = (
        f"/app?view=report&sample={sample_key}"
        if sample_key
        else _share_path_for_job(str(job.get("job_id", "")))
    )
    summary["audience_mode"] = audience_mode
    summary["audience_label"] = audience_mode_label(audience_mode)
    room_label = _normalize_room_label(job.get("room_label") or summary.get("room_label"))
    if room_label:
        job["room_label"] = room_label
        summary["room_label"] = room_label
    if summary:
        score_payload = _room_score_payload(job)
        if score_payload:
            summary.update(score_payload)
        summary["artifact_expires_at"] = _job_expiry_iso(job)
        summary["retention_days"] = _upload_cfg.retention_days
        job["summary"] = summary
        job["room_history"] = _room_history(job)
        job["room_comparison"] = _room_comparison(job)
        job["weekend_fix_list"] = build_weekend_fix_list(
            list(job.get("risks") or []),
            audience_mode=audience_mode,
        )
        job["room_wins"] = build_room_wins(
            list(job.get("risks") or []),
            dict(job.get("scan_quality") or {}),
            comparison_summary=job.get("room_comparison"),
            audience_mode=audience_mode,
        )
        if isinstance(job.get("resolution_summary"), dict):
            summary["resolution_summary"] = job["resolution_summary"]
        summary["share_summary"] = _share_summary(job)
    job["expires_at"] = _job_expiry_iso(job)
    return job


def _job_status_counts() -> dict[str, int]:
    """Aggregate upload-job status counts for operator diagnostics."""
    _refresh_upload_jobs_from_disk()
    counts = {"queued": 0, "processing": 0, "complete": 0, "error": 0}
    for job in _upload_jobs.values():
        status = str(job.get("status", "")).lower()
        if status in counts:
            counts[status] += 1
    return counts


# Rebuild artifact pointers and derived fields for jobs loaded from disk.
for _job in _upload_jobs.values():
    _refresh_job_artifacts(_job)
for _job in _upload_jobs.values():
    _ensure_job_derived_fields(_job)
