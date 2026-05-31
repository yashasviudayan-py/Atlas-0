"""Operator analytics, startup checks, descriptors, and access control.

Aggregates evaluation/product metrics, builds the beta operator inbox, runs
production-readiness startup checks, exposes public product descriptors, and
enforces private-beta access. Depends on :mod:`atlas.api.state` and
:mod:`atlas.api.jobs`.
"""

from __future__ import annotations

import contextlib
import hmac
import ipaddress
import json
import os
import pathlib
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import HTTPException, Request

from atlas.api.helpers import (
    _mask_waitlist_email,
    _normalize_public_event_name,
    _normalize_room_label,
    _utc_now_iso,
)
from atlas.api.jobs import (
    _active_external_worker_records,
    _effective_active_worker_count,
    _job_status_counts,
    _recent_job_failures,
)
from atlas.api.state import (
    _FRONTEND_DIR,
    _api_cfg,
    _evaluation_cfg,
    _state,
    _upload_cfg,
    _upload_jobs,
    _upload_store,
)
from atlas.api.upload_store import validate_storage_id
from atlas.utils.config import load_config

logger = structlog.get_logger(__name__)


def _service_started_at() -> str:
    """Return the service start timestamp, creating it on first access."""
    started_at = _state.get("service_started_at")
    if isinstance(started_at, str) and started_at:
        return started_at
    started_at = _utc_now_iso()
    _state["service_started_at"] = started_at
    return started_at


def _startup_check_summary() -> dict[str, Any]:
    """Return the cached startup-check summary."""
    summary = _state.get("startup_checks")
    if isinstance(summary, dict):
        return summary
    return {"ready": False, "checks": [], "summary": "Startup checks have not run yet."}


def _provider_env_ready(provider: str | None) -> bool:
    """Return True when the configured provider has the expected env wiring."""
    token = str(provider or "").strip().lower()
    if not token or token == "ollama":
        return True
    if token == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if token == "claude":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return False


def _run_startup_checks() -> dict[str, Any]:
    """Validate storage/provider/runtime assumptions needed for production use."""
    checks: list[dict[str, str]] = []

    storage_root = pathlib.Path(_upload_cfg.storage_dir)
    try:
        storage_root.mkdir(parents=True, exist_ok=True)
        probe = storage_root / ".atlas-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        checks.append(
            {
                "name": "storage_root",
                "status": "pass",
                "detail": f"Storage root {storage_root} is writable.",
            }
        )
    except OSError as exc:
        checks.append(
            {
                "name": "storage_root",
                "status": "fail",
                "detail": f"Storage root {storage_root} is not writable: {exc}.",
            }
        )

    if _upload_cfg.max_queue_depth >= _upload_cfg.max_concurrent_jobs:
        checks.append(
            {
                "name": "queue_capacity",
                "status": "pass",
                "detail": "Queue depth is at least as large as worker concurrency.",
            }
        )
    else:
        checks.append(
            {
                "name": "queue_capacity",
                "status": "fail",
                "detail": "Queue depth must be greater than or equal to worker concurrency.",
            }
        )

    if _upload_cfg.max_storage_bytes >= _upload_cfg.max_upload_bytes:
        checks.append(
            {
                "name": "storage_budget",
                "status": "pass",
                "detail": "Storage budget can hold at least one max-size upload safely.",
            }
        )
    else:
        checks.append(
            {
                "name": "storage_budget",
                "status": "warn",
                "detail": (
                    "Storage budget is smaller than one max-size upload and will churn quickly."
                ),
            }
        )

    if _upload_cfg.artifact_backend == "local_fs":
        checks.append(
            {
                "name": "artifact_backend",
                "status": "pass",
                "detail": "Artifact backend is local_fs with job-local artifact storage.",
            }
        )
    elif _upload_cfg.artifact_backend == "object_store_fs":
        checks.append(
            {
                "name": "artifact_backend",
                "status": "pass",
                "detail": (
                    "Artifact backend is object_store_fs with a filesystem-backed object"
                    f" root at {_upload_store.object_dir()}."
                ),
            }
        )
    else:
        checks.append(
            {
                "name": "artifact_backend",
                "status": "fail",
                "detail": f"Unsupported artifact backend {_upload_cfg.artifact_backend!r}.",
            }
        )

    rate_limits_enabled = (
        _api_cfg.rate_limit_public_requests > 0 and _api_cfg.rate_limit_upload_requests > 0
    )
    checks.append(
        {
            "name": "api_rate_limits",
            "status": "pass" if rate_limits_enabled else "warn",
            "detail": (
                "Public write and upload rate limits are enabled."
                if rate_limits_enabled
                else (
                    "One or more API write rate limits are disabled; "
                    "enable them before production."
                )
            ),
        }
    )

    cors_allows_any_origin = "*" in {origin.strip() for origin in _api_cfg.cors_origins}
    checks.append(
        {
            "name": "cors_origins",
            "status": "warn" if cors_allows_any_origin else "pass",
            "detail": (
                "CORS currently allows any origin; restrict this for hosted production."
                if cors_allows_any_origin
                else "CORS origins are explicitly configured."
            ),
        }
    )

    checks.append(
        {
            "name": "worker_mode",
            "status": "pass" if _upload_cfg.worker_mode in {"in_process", "external"} else "fail",
            "detail": f"Worker mode is {_upload_cfg.worker_mode}.",
        }
    )

    if _upload_cfg.worker_mode == "external":
        detached = _active_external_worker_records()
        checks.append(
            {
                "name": "external_workers",
                "status": "pass" if detached else "warn",
                "detail": (
                    f"{len(detached)} detached upload worker(s) are advertising heartbeats."
                    if detached
                    else "No detached upload workers are currently advertising heartbeats."
                ),
            }
        )

    providers = [load_config().vlm.provider, load_config().vlm.fallback_provider]
    for provider in providers:
        token = str(provider or "").strip().lower()
        if not token:
            continue
        checks.append(
            {
                "name": f"provider_{token}",
                "status": "pass" if _provider_env_ready(token) else "warn",
                "detail": (
                    f"{token} provider credentials look available."
                    if _provider_env_ready(token)
                    else f"{token} provider is configured without the expected API credentials."
                ),
            }
        )

    if _FRONTEND_DIR.exists():
        checks.append(
            {
                "name": "frontend_bundle",
                "status": "pass",
                "detail": "Frontend bundle is present for the hosted product path.",
            }
        )
    else:
        checks.append(
            {
                "name": "frontend_bundle",
                "status": "warn",
                "detail": "Frontend bundle is missing; API-only mode will still boot.",
            }
        )

    ready = all(check["status"] == "pass" for check in checks if check["name"] != "frontend_bundle")
    summary = {
        "ready": ready,
        "checked_at": _utc_now_iso(),
        "summary": (
            "Startup checks passed for the current deployment profile."
            if ready
            else "Startup checks found production-readiness gaps that operators should fix."
        ),
        "checks": checks,
    }
    _state["startup_checks"] = summary
    return summary


def _operator_system_summary() -> dict[str, Any]:
    """Return deployment and worker diagnostics for operator review."""
    startup = _startup_check_summary()
    started_at = _service_started_at()
    detached_workers = _active_external_worker_records()
    uptime_seconds = 0.0
    with contextlib.suppress(ValueError):
        uptime_seconds = max(
            0.0,
            (datetime.now(UTC) - datetime.fromisoformat(started_at)).total_seconds(),
        )

    return {
        "worker_mode": _upload_cfg.worker_mode,
        "deployment_ready": bool(startup.get("ready")),
        "startup_summary": startup.get("summary"),
        "startup_checks": startup.get("checks", []),
        "service_started_at": started_at,
        "uptime_seconds": round(uptime_seconds, 1),
        "storage_root": str(pathlib.Path(_upload_cfg.storage_dir)),
        "artifact_backend": _upload_cfg.artifact_backend,
        "artifact_base_url": _upload_cfg.artifact_base_url,
        "artifact_object_dir": str(_upload_store.object_dir()),
        "recent_failures": list(reversed(_recent_job_failures()[-5:])),
        "active_workers": _effective_active_worker_count(),
        "detached_workers": detached_workers[:5],
        "queue_depth": _job_status_counts().get("queued", 0),
        "last_resume_scan_at": _state.get("last_resume_scan_at"),
        "last_prune_at": _state.get("last_prune_at"),
    }


def _load_eval_candidate_entries() -> list[dict[str, Any]]:
    """Load saved operator-exported evaluation candidates from disk."""
    return _upload_store.load_eval_candidates()


def _review_ready_for_eval(job: dict[str, Any]) -> bool:
    """Return True when a completed job is mature enough for eval export."""
    evaluation = dict(job.get("evaluation_summary") or {})
    human = dict(job.get("human_evaluation") or {})
    return bool(human) or int(evaluation.get("reviewed_findings", 0) or 0) > 0


def _eval_corpus_dir() -> pathlib.Path:
    """Return the directory holding seeded evaluation fixtures."""
    return pathlib.Path(__file__).parents[3] / "data" / "eval_corpus"


def _load_eval_corpus_entries() -> list[dict[str, Any]]:
    """Load all seeded evaluation corpus entries from disk."""
    entries: list[dict[str, Any]] = []
    corpus_dir = _eval_corpus_dir()
    if not corpus_dir.exists():
        return entries

    for path in sorted(corpus_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("label"):
            entries.append(payload)
    return entries


def _load_expected_benchmark(label: str) -> dict[str, Any] | None:
    """Load a supported benchmark fixture definition for comparison."""
    for entry in _load_eval_corpus_entries():
        if str(entry.get("label")) == label:
            return entry
    return None


def _auto_benchmark_label(job: dict[str, Any]) -> str | None:
    """Infer a benchmark label for known regression fixtures."""
    filename = str((job.get("summary") or {}).get("filename") or job.get("filename") or "")
    for entry in _load_eval_corpus_entries():
        if filename and filename == str(entry.get("fixture_name", "")):
            return str(entry.get("label"))
    return None


def _compare_job_to_benchmark(job: dict[str, Any], label: str) -> dict[str, Any] | None:
    """Compare one completed job against a supported benchmark fixture."""
    expected = _load_expected_benchmark(label)
    if expected is None:
        return None

    risks = list(job.get("risks") or [])
    objects = list(job.get("objects") or [])
    summary = dict(job.get("summary") or {})
    hazard_codes = [str(risk.get("hazard_code", "")) for risk in risks]
    missing_codes = [
        code for code in expected.get("required_hazard_codes", []) if code not in hazard_codes
    ]
    multiframe_supported = sum(1 for obj in objects if bool(obj.get("multi_frame_support")))
    grounding_values = [
        float(obj.get("grounding_confidence", 0.0) or 0.0)
        for obj in objects
        if obj.get("grounding_confidence") is not None
    ]
    avg_grounding_confidence = (
        round(sum(grounding_values) / len(grounding_values), 2) if grounding_values else 0.0
    )
    min_multiframe_supported = int(expected.get("min_multiframe_supported_objects", 0) or 0)
    min_avg_grounding = float(expected.get("min_avg_grounding_confidence", 0.0) or 0.0)
    matched = (
        str(job.get("scene_source", "")) == str(expected.get("scene_source", ""))
        and int(summary.get("object_count", 0) or 0)
        >= int(expected.get("min_object_count", 0) or 0)
        and int(summary.get("hazard_count", 0) or 0)
        >= int(expected.get("min_hazard_count", 0) or 0)
        and (hazard_codes[0] if hazard_codes else None) == expected.get("top_hazard_code")
        and multiframe_supported >= min_multiframe_supported
        and avg_grounding_confidence >= min_avg_grounding
        and not missing_codes
    )
    return {
        "label": label,
        "matched": matched,
        "missing_hazard_codes": missing_codes,
        "top_hazard_code": hazard_codes[0] if hazard_codes else None,
        "expected_top_hazard_code": expected.get("top_hazard_code"),
        "expected_min_object_count": expected.get("min_object_count"),
        "expected_min_hazard_count": expected.get("min_hazard_count"),
        "multiframe_supported_objects": multiframe_supported,
        "expected_min_multiframe_supported_objects": min_multiframe_supported,
        "avg_grounding_confidence": avg_grounding_confidence,
        "expected_min_avg_grounding_confidence": min_avg_grounding,
    }


def _aggregate_evaluation_metrics() -> dict[str, Any]:
    """Aggregate evaluation and benchmark signals across persisted jobs."""
    complete_jobs = [job for job in _upload_jobs.values() if str(job.get("status")) == "complete"]
    total_complete = len(complete_jobs)
    reviewed_jobs = 0
    benchmarked_jobs = 0
    benchmark_matches = 0
    jobs_with_missed_hazards = 0
    jobs_needing_review = 0
    disputed_jobs = 0
    coverage_sum = 0.0

    for job in complete_jobs:
        summary = dict(job.get("evaluation_summary") or {})
        human = dict(job.get("human_evaluation") or {})
        if human:
            reviewed_jobs += 1
        if bool(summary.get("benchmark_label")):
            benchmarked_jobs += 1
        if bool(summary.get("benchmark_match")):
            benchmark_matches += 1
        if int(summary.get("missed_hazard_count", 0) or 0) > 0:
            jobs_with_missed_hazards += 1
        if bool(summary.get("needs_review")):
            jobs_needing_review += 1
        if int(summary.get("disputed_findings", 0) or 0) > 0:
            disputed_jobs += 1
        coverage_sum += float(summary.get("review_coverage", 0.0) or 0.0)

    avg_review_coverage = round(coverage_sum / total_complete, 2) if total_complete else 0.0
    benchmark_match_rate = (
        round(benchmark_matches / benchmarked_jobs, 2) if benchmarked_jobs else 0.0
    )
    false_positive_job_rate = round(disputed_jobs / total_complete, 2) if total_complete else 0.0
    missed_hazard_rate = (
        round(jobs_with_missed_hazards / total_complete, 2) if total_complete else 0.0
    )
    corpus_entries = _load_eval_corpus_entries()
    eval_candidates = _load_eval_candidate_entries()
    review_ready_candidates = sum(
        1 for candidate in eval_candidates if bool(candidate.get("review_ready"))
    )
    available_eval_cases = len(corpus_entries) + review_ready_candidates
    release_gates = _evaluation_release_gates(
        reviewed_jobs=reviewed_jobs,
        benchmarked_jobs=benchmarked_jobs,
        benchmark_match_rate=benchmark_match_rate,
        false_positive_job_rate=false_positive_job_rate,
        missed_hazard_rate=missed_hazard_rate,
        avg_review_coverage=avg_review_coverage,
        seed_fixture_count=available_eval_cases,
    )

    return {
        "completed_jobs": total_complete,
        "reviewed_jobs": reviewed_jobs,
        "benchmarked_jobs": benchmarked_jobs,
        "benchmark_match_rate": benchmark_match_rate,
        "jobs_with_missed_hazards": jobs_with_missed_hazards,
        "missed_hazard_rate": missed_hazard_rate,
        "jobs_needing_review": jobs_needing_review,
        "false_positive_job_rate": false_positive_job_rate,
        "avg_review_coverage": avg_review_coverage,
        "seed_fixture_count": len(corpus_entries),
        "saved_eval_candidates": len(eval_candidates),
        "review_ready_eval_candidates": review_ready_candidates,
        "available_eval_cases": available_eval_cases,
        "candidate_gap": max(0, _evaluation_cfg.target_corpus_size - available_eval_cases),
        "target_corpus_size": _evaluation_cfg.target_corpus_size,
        "release_gates": release_gates,
    }


def _evaluation_release_gates(
    *,
    reviewed_jobs: int,
    benchmarked_jobs: int,
    benchmark_match_rate: float,
    false_positive_job_rate: float,
    missed_hazard_rate: float,
    avg_review_coverage: float,
    seed_fixture_count: int,
) -> dict[str, Any]:
    """Build a release-gate summary for operator review."""
    gates = [
        {
            "id": "reviewed_jobs",
            "label": "Reviewed jobs",
            "actual": reviewed_jobs,
            "target": _evaluation_cfg.min_reviewed_jobs,
            "passed": reviewed_jobs >= _evaluation_cfg.min_reviewed_jobs,
        },
        {
            "id": "benchmark_match_rate",
            "label": "Benchmark match rate",
            "actual": benchmark_match_rate,
            "target": _evaluation_cfg.min_benchmark_match_rate,
            "passed": benchmarked_jobs > 0
            and benchmark_match_rate >= _evaluation_cfg.min_benchmark_match_rate,
        },
        {
            "id": "false_positive_job_rate",
            "label": "False-positive job rate",
            "actual": false_positive_job_rate,
            "target": _evaluation_cfg.max_false_positive_job_rate,
            "passed": false_positive_job_rate <= _evaluation_cfg.max_false_positive_job_rate,
        },
        {
            "id": "missed_hazard_rate",
            "label": "Missed-hazard job rate",
            "actual": missed_hazard_rate,
            "target": _evaluation_cfg.max_missed_hazard_rate,
            "passed": missed_hazard_rate <= _evaluation_cfg.max_missed_hazard_rate,
        },
        {
            "id": "avg_review_coverage",
            "label": "Average review coverage",
            "actual": avg_review_coverage,
            "target": _evaluation_cfg.min_avg_review_coverage,
            "passed": avg_review_coverage >= _evaluation_cfg.min_avg_review_coverage,
        },
        {
            "id": "seed_corpus_progress",
            "label": "Seed eval corpus",
            "actual": seed_fixture_count,
            "target": _evaluation_cfg.target_corpus_size,
            "passed": seed_fixture_count >= _evaluation_cfg.target_corpus_size,
        },
    ]
    ready_for_beta = all(bool(gate["passed"]) for gate in gates[:5])
    return {
        "ready_for_beta": ready_for_beta,
        "summary": (
            "Release gates passed for broader beta."
            if ready_for_beta
            else "Release gates are still open. Keep growing the eval corpus and review coverage."
        ),
        "gates": gates,
    }


def _aggregate_product_metrics() -> dict[str, Any]:
    """Aggregate product-loop metrics used for beta operations."""
    jobs = list(_upload_jobs.values())
    terminal_jobs = [job for job in jobs if str(job.get("status")) in {"complete", "error"}]
    completed_jobs = [job for job in jobs if str(job.get("status")) == "complete"]
    success_rate = round(len(completed_jobs) / len(terminal_jobs), 2) if terminal_jobs else 0.0
    rescan_recommended = sum(
        1 for job in completed_jobs if bool((job.get("summary") or {}).get("rescan_recommended"))
    )
    useful_events = 0
    total_feedback_events = 0
    total_duration_seconds = 0.0
    completed_with_duration = 0
    room_label_counts: dict[str, int] = {}

    for job in completed_jobs:
        feedback = dict(job.get("feedback_summary") or {})
        useful_events += int(feedback.get("useful", 0) or 0)
        total_feedback_events += sum(int(feedback.get(key, 0) or 0) for key in feedback)
        room_label = _normalize_room_label(
            job.get("room_label") or (job.get("summary") or {}).get("room_label")
        )
        if room_label:
            room_label_counts[room_label] = room_label_counts.get(room_label, 0) + 1
        started_at = job.get("started_at")
        completed_at = job.get("completed_at")
        if started_at and completed_at:
            try:
                start_dt = datetime.fromisoformat(str(started_at))
                done_dt = datetime.fromisoformat(str(completed_at))
                total_duration_seconds += max(0.0, (done_dt - start_dt).total_seconds())
                completed_with_duration += 1
            except ValueError:
                pass

    usefulness_rate = (
        round(useful_events / total_feedback_events, 2) if total_feedback_events else 0.0
    )
    avg_report_seconds = (
        round(total_duration_seconds / completed_with_duration, 1)
        if completed_with_duration
        else 0.0
    )
    product_events = _upload_store.load_product_events()
    waitlist_entries = _upload_store.load_waitlist_entries()
    event_counts: dict[str, int] = {}
    for event in product_events:
        name = _normalize_public_event_name(event.get("event_name"))
        if name is None:
            continue
        event_counts[name] = event_counts.get(name, 0) + 1

    return {
        "terminal_jobs": len(terminal_jobs),
        "completed_jobs": len(completed_jobs),
        "repeat_scan_rooms": sum(1 for count in room_label_counts.values() if count > 1),
        "labeled_rooms": len(room_label_counts),
        "upload_success_rate": success_rate,
        "rescan_recommended_rate": (
            round(rescan_recommended / len(completed_jobs), 2) if completed_jobs else 0.0
        ),
        "report_usefulness_rate": usefulness_rate,
        "avg_report_seconds": avg_report_seconds,
        "product_event_count": len(product_events),
        "waitlist_signups": len(waitlist_entries),
        "beta_onboarding_events": event_counts.get("beta_onboarding_started", 0),
        "sample_report_opens": event_counts.get("sample_report_opened", 0),
        "sample_journey_events": event_counts.get("sample_journey_opened", 0),
        "sample_cta_events": event_counts.get("sample_cta_clicked", 0),
        "landing_section_events": event_counts.get("landing_section_viewed", 0),
        "share_events": event_counts.get("report_share_copied", 0),
        "share_card_events": event_counts.get("report_share_card_copied", 0),
        "report_theme_events": event_counts.get("report_theme_changed", 0),
        "beta_invite_events": event_counts.get("beta_invite_copied", 0),
        "room_win_card_shared_events": event_counts.get("room_win_card_shared", 0),
        "room_win_events": event_counts.get("room_win_copied", 0),
        "post_report_feedback_events": event_counts.get("post_report_feedback_submitted", 0),
        "first_run_events": event_counts.get("first_run_started", 0),
        "confidence_inspector_events": event_counts.get("confidence_inspector_opened", 0),
        "fix_plan_events": event_counts.get("fix_plan_copied", 0),
        "fix_today_events": event_counts.get("fix_today_copied", 0),
        "fix_quest_events": event_counts.get("fix_quest_completed", 0),
        "fix_library_events": event_counts.get("fix_library_opened", 0),
        "fix_guide_events": event_counts.get("fix_guide_opened", 0),
        "one_thing_started_events": event_counts.get("one_thing_today_started", 0),
        "one_thing_completed_events": event_counts.get("one_thing_today_completed", 0),
        "room_care_calendar_events": event_counts.get("room_care_calendar_opened", 0),
        "room_care_completed_events": event_counts.get("room_care_task_completed", 0),
        "room_care_regenerated_events": event_counts.get("room_care_week_regenerated", 0),
        "room_health_timeline_events": event_counts.get("room_health_timeline_opened", 0),
        "settings_daily_value_events": event_counts.get("settings_daily_value_changed", 0),
        "before_after_card_events": event_counts.get("before_after_card_copied", 0),
        "trust_dashboard_events": event_counts.get("trust_dashboard_opened", 0),
        "live_capture_events": event_counts.get("live_capture_coach_started", 0),
        "live_capture_quality_events": event_counts.get("live_capture_quality_checked", 0),
        "report_question_events": event_counts.get("report_question_asked", 0),
        "report_answer_copy_events": event_counts.get("report_answer_copied", 0),
        "privacy_receipt_events": event_counts.get("privacy_receipt_opened", 0),
        "privacy_receipt_copy_events": event_counts.get("privacy_receipt_copied", 0),
        "evidence_privacy_toggle_events": event_counts.get("evidence_privacy_toggled", 0),
        "pwa_offline_ready_events": event_counts.get("pwa_offline_ready", 0),
        "daily_mission_events": event_counts.get("daily_mission_started", 0),
        "daily_mission_completed_events": event_counts.get("daily_mission_completed", 0),
        "mystery_mode_events": event_counts.get("mystery_mode_started", 0),
        "personal_mode_events": event_counts.get("personal_mode_selected", 0),
        "sample_gallery_events": event_counts.get("sample_gallery_opened", 0),
        "field_note_events": event_counts.get("field_note_expanded", 0),
        "evidence_story_events": event_counts.get("evidence_story_opened", 0),
        "room_map_preview_events": event_counts.get("room_map_preview_opened", 0),
        "room_compare_events": event_counts.get("room_compare_opened", 0),
        "room_passport_events": event_counts.get("room_passport_opened", 0),
        "room_personality_events": event_counts.get("room_personality_viewed", 0),
        "room_playbook_events": event_counts.get("room_playbook_started", 0),
        "fix_verification_events": event_counts.get("fix_verification_started", 0),
        "fix_verification_copy_events": event_counts.get("fix_verification_copied", 0),
        "evidence_frame_focus_events": event_counts.get("evidence_frame_focused", 0),
        "share_card_studio_events": event_counts.get("share_card_studio_copied", 0),
        "confidence_explainer_events": event_counts.get("confidence_explainer_opened", 0),
        "welcome_tour_events": event_counts.get("welcome_tour_completed", 0),
        "home_pulse_events": event_counts.get("home_pulse_opened", 0),
        "weekly_recap_events": event_counts.get("weekly_recap_copied", 0),
        "weekly_challenge_events": event_counts.get("weekly_challenge_completed", 0),
        "home_bingo_events": event_counts.get("home_bingo_task_completed", 0),
        "room_ritual_events": event_counts.get("room_ritual_started", 0),
        "room_ritual_completed_events": event_counts.get("room_ritual_completed", 0),
        "home_journal_events": event_counts.get("home_journal_opened", 0),
        "room_reminder_events": event_counts.get("room_reminder_clicked", 0),
        "seasonal_pack_events": event_counts.get("seasonal_pack_started", 0),
        "seasonal_pack_selected_events": event_counts.get("seasonal_pack_selected", 0),
        "smart_rescan_coach_events": event_counts.get("smart_rescan_coach_opened", 0),
        "capture_coach_events": event_counts.get("capture_coach_checked", 0),
        "same_room_rescan_events": event_counts.get("same_room_rescan_started", 0),
        "rescan_prompt_events": event_counts.get("rescan_prompt_clicked", 0),
        "pdf_download_events": event_counts.get("report_pdf_downloaded", 0),
        "pdf_export_click_events": event_counts.get("pdf_export_clicked", 0),
        "scan_preflight_failed_events": event_counts.get("scan_preflight_failed", 0),
        "upload_start_events": event_counts.get("upload_started", 0),
        "upload_completed_events": event_counts.get("upload_completed", 0),
        "cta_start_scan_events": event_counts.get("cta_start_scan", 0),
    }


def _provider_runtime_summary() -> dict[str, Any]:
    """Summarize the active VLM routing strategy for operator diagnostics."""
    vlm = load_config().vlm
    chain = [vlm.provider]
    if vlm.fallback_provider and vlm.fallback_provider != vlm.provider:
        chain.append(vlm.fallback_provider)
    return {
        "primary_provider": vlm.provider,
        "fallback_provider": vlm.fallback_provider,
        "provider_chain": chain,
        "routing_mode": "fallback" if len(chain) > 1 else "single",
        "primary_model": vlm.model_name
        if vlm.provider == "ollama"
        else (vlm.claude_model if vlm.provider == "claude" else vlm.openai_model),
    }


def _job_started_seconds(job: dict[str, Any]) -> float | None:
    """Return elapsed seconds between a job start and terminal timestamp."""
    started_at = job.get("started_at") or job.get("created_at")
    ended_at = job.get("completed_at") or job.get("failed_at") or job.get("updated_at")
    if not started_at or not ended_at:
        return None
    try:
        start_dt = datetime.fromisoformat(str(started_at))
        end_dt = datetime.fromisoformat(str(ended_at))
    except ValueError:
        return None
    return max(0.0, (end_dt - start_dt).total_seconds())


def _build_beta_inbox() -> dict[str, Any]:
    """Build the protected operator inbox for beta growth and learning loops."""
    jobs = list(_upload_jobs.values())
    product_events = _upload_store.load_product_events()
    waitlist_entries = _upload_store.load_waitlist_entries()
    event_counts: dict[str, int] = {}
    for event in product_events:
        name = _normalize_public_event_name(event.get("event_name"))
        if name:
            event_counts[name] = event_counts.get(name, 0) + 1

    terminal_jobs = [job for job in jobs if str(job.get("status")) in {"complete", "error"}]
    completed_jobs = [job for job in jobs if str(job.get("status")) == "complete"]
    failed_jobs = [job for job in jobs if str(job.get("status")) == "error"]
    negative_reports: list[dict[str, Any]] = []
    review_needed: list[dict[str, Any]] = []
    missed_hazard_notes: list[dict[str, Any]] = []

    for job in completed_jobs:
        feedback = dict(job.get("feedback_summary") or {})
        wrong = int(feedback.get("wrong", 0) or 0)
        duplicate = int(feedback.get("duplicate", 0) or 0)
        room_label = job.get("room_label") or (job.get("summary") or {}).get("room_label")
        if wrong or duplicate:
            negative_reports.append(
                {
                    "job_id": job.get("job_id"),
                    "filename": job.get("filename"),
                    "wrong": wrong,
                    "duplicate": duplicate,
                    "room_label": room_label,
                }
            )

        evaluation = dict(job.get("evaluation_summary") or {})
        if bool(evaluation.get("needs_review")) or _review_ready_for_eval(job):
            review_needed.append(
                {
                    "job_id": job.get("job_id"),
                    "filename": job.get("filename"),
                    "summary": evaluation.get("summary"),
                    "review_ready_for_eval": _review_ready_for_eval(job),
                }
            )

        human = dict(job.get("human_evaluation") or {})
        for missed in list(human.get("missed_hazards") or [])[:3]:
            missed_hazard_notes.append(
                {
                    "job_id": job.get("job_id"),
                    "note": str(missed)[:180],
                    "room_label": room_label,
                }
            )

    durations = [
        value for value in (_job_started_seconds(job) for job in terminal_jobs) if value is not None
    ]
    recent_waitlist = sorted(
        waitlist_entries,
        key=lambda entry: str(entry.get("created_at") or ""),
        reverse=True,
    )[:8]
    recent_failures = sorted(
        failed_jobs,
        key=lambda job: str(
            job.get("failed_at") or job.get("updated_at") or job.get("created_at") or ""
        ),
        reverse=True,
    )[:6]

    return {
        "summary": (
            "Review failed uploads, negative feedback, waitlist demand, and eval-ready"
            " reports before inviting the next beta batch."
        ),
        "funnel": {
            "cta_start_scan": event_counts.get("cta_start_scan", 0),
            "upload_started": event_counts.get("upload_started", 0),
            "upload_completed": event_counts.get("upload_completed", 0),
            "report_viewed": event_counts.get("report_viewed", 0),
            "report_theme_changed": event_counts.get("report_theme_changed", 0),
            "first_run_started": event_counts.get("first_run_started", 0),
            "beta_onboarding_started": event_counts.get("beta_onboarding_started", 0),
            "sample_journey_opened": event_counts.get("sample_journey_opened", 0),
            "mystery_mode_started": event_counts.get("mystery_mode_started", 0),
            "personal_mode_selected": event_counts.get("personal_mode_selected", 0),
            "sample_gallery_opened": event_counts.get("sample_gallery_opened", 0),
            "before_after_card_copied": event_counts.get("before_after_card_copied", 0),
            "field_note_expanded": event_counts.get("field_note_expanded", 0),
            "evidence_story_opened": event_counts.get("evidence_story_opened", 0),
            "room_map_preview_opened": event_counts.get("room_map_preview_opened", 0),
            "room_compare_opened": event_counts.get("room_compare_opened", 0),
            "room_passport_opened": event_counts.get("room_passport_opened", 0),
            "room_personality_viewed": event_counts.get("room_personality_viewed", 0),
            "room_playbook_started": event_counts.get("room_playbook_started", 0),
            "fix_verification_started": event_counts.get("fix_verification_started", 0),
            "fix_quest_completed": event_counts.get("fix_quest_completed", 0),
            "fix_library_opened": event_counts.get("fix_library_opened", 0),
            "fix_guide_opened": event_counts.get("fix_guide_opened", 0),
            "one_thing_today_started": event_counts.get("one_thing_today_started", 0),
            "one_thing_today_completed": event_counts.get("one_thing_today_completed", 0),
            "room_care_calendar_opened": event_counts.get("room_care_calendar_opened", 0),
            "room_care_task_completed": event_counts.get("room_care_task_completed", 0),
            "room_care_week_regenerated": event_counts.get("room_care_week_regenerated", 0),
            "room_health_timeline_opened": event_counts.get("room_health_timeline_opened", 0),
            "settings_daily_value_changed": event_counts.get("settings_daily_value_changed", 0),
            "share_card_studio_copied": event_counts.get("share_card_studio_copied", 0),
            "confidence_explainer_opened": event_counts.get("confidence_explainer_opened", 0),
            "welcome_tour_completed": event_counts.get("welcome_tour_completed", 0),
            "home_pulse_opened": event_counts.get("home_pulse_opened", 0),
            "weekly_recap_copied": event_counts.get("weekly_recap_copied", 0),
            "home_bingo_task_completed": event_counts.get("home_bingo_task_completed", 0),
            "room_ritual_started": event_counts.get("room_ritual_started", 0),
            "room_ritual_completed": event_counts.get("room_ritual_completed", 0),
            "seasonal_pack_selected": event_counts.get("seasonal_pack_selected", 0),
            "smart_rescan_coach_opened": event_counts.get("smart_rescan_coach_opened", 0),
            "home_journal_opened": event_counts.get("home_journal_opened", 0),
            "fix_today_copied": event_counts.get("fix_today_copied", 0),
            "pdf_downloads": event_counts.get("report_pdf_downloaded", 0),
            "share_events": event_counts.get("report_share_copied", 0),
            "share_card_copies": event_counts.get("report_share_card_copied", 0),
            "room_win_card_shared": event_counts.get("room_win_card_shared", 0),
            "weekly_challenge_completed": event_counts.get("weekly_challenge_completed", 0),
            "post_report_feedback_submitted": event_counts.get("post_report_feedback_submitted", 0),
            "confidence_inspector_opened": event_counts.get("confidence_inspector_opened", 0),
            "scan_preflight_failed": event_counts.get("scan_preflight_failed", 0),
            "rescan_prompt_clicked": event_counts.get("rescan_prompt_clicked", 0),
            "waitlist_submitted": event_counts.get("waitlist_submitted", 0),
            "completion_rate": round(len(completed_jobs) / len(terminal_jobs), 2)
            if terminal_jobs
            else 0.0,
            "avg_terminal_seconds": round(sum(durations) / len(durations), 1) if durations else 0.0,
        },
        "recent_waitlist": [
            {
                "email": _mask_waitlist_email(entry.get("email")),
                "use_case": entry.get("use_case"),
                "source": entry.get("source"),
                "audience_mode": entry.get("audience_mode"),
                "created_at": entry.get("created_at"),
            }
            for entry in recent_waitlist
        ],
        "failed_uploads": [
            {
                "job_id": job.get("job_id"),
                "filename": job.get("filename"),
                "stage": job.get("stage"),
                "error": str(job.get("error") or job.get("message") or "Unknown failure")[:180],
            }
            for job in recent_failures
        ],
        "negative_feedback_reports": negative_reports[:8],
        "review_needed_reports": review_needed[:10],
        "missed_hazard_notes": missed_hazard_notes[:10],
        "eval_candidate_readiness": {
            "review_ready_reports": sum(1 for job in completed_jobs if _review_ready_for_eval(job)),
            "saved_eval_candidates": len(_load_eval_candidate_entries()),
            "target_corpus_size": _evaluation_cfg.target_corpus_size,
        },
    }


def _request_host(request: Request) -> str:
    """Return the peer host string for one request."""
    return str(request.client.host if request.client else "")


def _is_loopback_request(request: Request) -> bool:
    """Return True when the request originates from loopback/testclient."""
    host = _request_host(request)
    if host in {"localhost", "testclient"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _extract_access_token(request: Request) -> str | None:
    """Extract an API token from Authorization or X-Atlas-Key headers."""
    authorization = request.headers.get("authorization", "").strip()
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        return token or None
    query_token = request.query_params.get("access_token", "").strip()
    if query_token:
        return query_token
    header_token = request.headers.get("x-atlas-key", "").strip()
    return header_token or None


def _storage_route_id(value: str, label: str) -> str:
    """Validate a route parameter before using it as a storage identifier."""
    try:
        return validate_storage_id(value, label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _require_private_access(request: Request) -> None:
    """Enforce private-beta access for upload/report endpoints."""
    configured_token = _api_cfg.access_token
    if configured_token:
        provided_token = _extract_access_token(request)
        if not provided_token or not hmac.compare_digest(provided_token, configured_token):
            raise HTTPException(status_code=401, detail="Missing or invalid Atlas API token.")
        return

    if _api_cfg.allow_unauthenticated_loopback and _is_loopback_request(request):
        return

    raise HTTPException(
        status_code=403,
        detail=(
            "Atlas-0 upload/report access is restricted to loopback unless"
            " api.access_token is configured."
        ),
    )


def _operator_access_descriptor() -> dict[str, Any]:
    """Return the effective access policy for upload/report features."""
    requires_token = bool(_api_cfg.access_token)
    mode = (
        "token"
        if requires_token
        else "loopback"
        if _api_cfg.allow_unauthenticated_loopback
        else "restricted"
    )
    return {
        "requires_token": requires_token,
        "allow_unauthenticated_loopback": _api_cfg.allow_unauthenticated_loopback,
        "enable_job_listing": _api_cfg.enable_job_listing,
        "mode": mode,
    }


def _public_privacy_descriptor() -> dict[str, Any]:
    """Return user-visible privacy defaults for upload/report flows."""
    details = [
        f"Uploads and artifacts are retained for up to {_upload_cfg.retention_days} day(s).",
        (
            "Original uploads are kept."
            if _upload_cfg.save_original_uploads
            else "Original uploads are not persisted by default after processing."
        ),
        "Text-heavy crops are blurred before model analysis."
        if _upload_cfg.redact_text_heavy_regions
        else "Text-heavy crop redaction is disabled in this environment.",
        "You can delete a scan and its artifacts from the report view at any time.",
    ]
    return {
        "retention_days": _upload_cfg.retention_days,
        "save_original_uploads": _upload_cfg.save_original_uploads,
        "delete_supported": True,
        "text_redaction_enabled": _upload_cfg.redact_text_heavy_regions,
        "artifact_backend": _upload_cfg.artifact_backend,
        "summary": (
            "ATLAS-0 keeps upload artifacts on a time-limited retention window for reports,"
            " review, and debugging, and it exposes delete controls in the product."
        ),
        "details": details,
    }


def _public_trust_proof_descriptor() -> dict[str, Any]:
    """Return aggregate-only product trust signals without report identifiers."""
    completed_jobs = [job for job in _upload_jobs.values() if str(job.get("status")) == "complete"]
    rejected_or_downgraded = 0
    evidence_backed = 0
    useful_feedback = 0
    negative_feedback = 0
    eval_ready = 0

    for job in completed_jobs:
        scan_quality = dict(job.get("scan_quality") or {})
        reportability = str(scan_quality.get("reportability") or "").lower()
        if (
            reportability in {"rejected", "downgraded"}
            or bool(scan_quality.get("hard_reject"))
            or bool(scan_quality.get("rescan_recommended"))
        ):
            rejected_or_downgraded += 1

        if list(job.get("evidence_frames") or []):
            evidence_backed += 1

        feedback = dict(job.get("feedback_summary") or {})
        useful_feedback += int(feedback.get("useful", 0) or 0)
        negative_feedback += int(feedback.get("wrong", 0) or 0) + int(
            feedback.get("duplicate", 0) or 0
        )

        if _review_ready_for_eval(job):
            eval_ready += 1

    return {
        "completed_scans": len(completed_jobs),
        "rejected_or_downgraded_scans": rejected_or_downgraded,
        "evidence_backed_reports": evidence_backed,
        "useful_feedback_count": useful_feedback,
        "negative_feedback_count": negative_feedback,
        "eval_ready_reports": eval_ready,
        "sample_report_available": True,
        "known_limits": [
            "Upload-side room geometry is approximate and should be treated as screening support.",
            "Low-light, fast motion, mirrors, and sparse coverage can weaken findings.",
            "ATLAS-0 prioritizes room hazard decision support; it is not a safety certification.",
        ],
        "proof_points": [
            {
                "label": "Evidence-linked findings",
                "value": str(evidence_backed),
            },
            {
                "label": "Scans downgraded or rejected when weak",
                "value": str(rejected_or_downgraded),
            },
            {
                "label": "Reports ready for eval review",
                "value": str(eval_ready),
            },
        ],
        "privacy_notes": [
            f"Artifacts follow a {_upload_cfg.retention_days} day retention window.",
            "The public proof dashboard exposes aggregate counts only.",
            "Private report IDs, filenames, emails, notes, and evidence images are never "
            "returned here.",
        ],
    }


def _public_upload_guidance_descriptor() -> dict[str, Any]:
    """Return public upload constraints and capture coaching for first-run UX."""
    return {
        "max_upload_bytes": _upload_cfg.max_upload_bytes,
        "max_video_duration_seconds": _upload_cfg.max_video_duration_seconds,
        "recommended_duration_seconds": {"min": 20, "max": 60},
        "accepted_extensions": [".mp4", ".mov", ".webm", ".jpg", ".jpeg", ".png", ".heic"],
        "accepted_media_prefixes": ["video/", "image/"],
        "one_room_only": True,
        "checklist": [
            "Scan one room only.",
            "Move slowly enough that shelves, tables, and corners stay in frame.",
            "Keep the room bright and avoid pointing directly at windows or lamps.",
            "Avoid scanning documents, screens, or private photos when possible.",
        ],
        "retry_guidance": [
            "Rescan if the video is dark, blurry, very short, or mostly pointed at the floor.",
            "Use the same room label for before/after comparisons.",
        ],
    }
