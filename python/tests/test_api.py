"""Tests for the Atlas-0 API server."""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from atlas.api import server as server_mod
from atlas.api.server import _api_cfg, _state, _upload_cfg, _upload_jobs, _upload_store, app
from atlas.utils.video import VideoMetadata
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_upload_jobs(tmp_path: Path):
    """Keep upload job state isolated across tests."""
    snapshot = dict(_upload_jobs)
    state_snapshot = {
        "upload_queue": _state.get("upload_queue"),
        "upload_queue_ids": _state.get("upload_queue_ids"),
        "upload_cancelled_jobs": _state.get("upload_cancelled_jobs"),
        "upload_worker_tasks": _state.get("upload_worker_tasks"),
        "active_upload_workers": _state.get("active_upload_workers"),
        "startup_checks": _state.get("startup_checks"),
        "service_started_at": _state.get("service_started_at"),
        "recent_job_failures": _state.get("recent_job_failures"),
        "last_resume_scan_at": _state.get("last_resume_scan_at"),
        "last_prune_at": _state.get("last_prune_at"),
    }
    api_snapshot = {
        "access_token": _api_cfg.access_token,
        "allow_unauthenticated_loopback": _api_cfg.allow_unauthenticated_loopback,
        "enable_job_listing": _api_cfg.enable_job_listing,
    }
    upload_snapshot = {
        "worker_mode": _upload_cfg.worker_mode,
        "worker_poll_seconds": _upload_cfg.worker_poll_seconds,
        "worker_claim_ttl_seconds": _upload_cfg.worker_claim_ttl_seconds,
        "worker_heartbeat_seconds": _upload_cfg.worker_heartbeat_seconds,
        "worker_stale_after_seconds": _upload_cfg.worker_stale_after_seconds,
        "max_upload_bytes": _upload_cfg.max_upload_bytes,
        "max_video_duration_seconds": _upload_cfg.max_video_duration_seconds,
        "max_concurrent_jobs": _upload_cfg.max_concurrent_jobs,
        "max_queue_depth": _upload_cfg.max_queue_depth,
        "max_job_attempts": _upload_cfg.max_job_attempts,
        "job_timeout_seconds": _upload_cfg.job_timeout_seconds,
        "artifact_backend": _upload_cfg.artifact_backend,
        "artifact_base_url": _upload_cfg.artifact_base_url,
        "artifact_object_dir": _upload_cfg.artifact_object_dir,
        "strict_startup_checks": _upload_cfg.strict_startup_checks,
        "job_failure_log_limit": _upload_cfg.job_failure_log_limit,
    }
    root_snapshot = _upload_store.root_dir
    artifact_backend_snapshot = _upload_store._artifact_backend
    artifact_base_url_snapshot = _upload_store._artifact_base_url
    artifact_object_dir_snapshot = _upload_store._artifact_object_dir
    _upload_jobs.clear()
    _upload_store.root_dir = tmp_path
    _upload_store._artifact_object_dir = tmp_path / "_objects"
    _state["upload_queue"] = asyncio.Queue()
    _state["upload_queue_ids"] = set()
    _state["upload_cancelled_jobs"] = set()
    _state["upload_worker_tasks"] = []
    _state["active_upload_workers"] = 0
    _state["startup_checks"] = {"ready": True, "checks": [], "summary": "ok"}
    _state["service_started_at"] = "2026-04-21T00:00:00+00:00"
    _state["recent_job_failures"] = []
    yield
    _upload_jobs.clear()
    _upload_jobs.update(snapshot)
    _upload_store.root_dir = root_snapshot
    for key, value in state_snapshot.items():
        if value is None:
            _state.pop(key, None)
        else:
            _state[key] = value
    _api_cfg.access_token = api_snapshot["access_token"]
    _api_cfg.allow_unauthenticated_loopback = api_snapshot["allow_unauthenticated_loopback"]
    _api_cfg.enable_job_listing = api_snapshot["enable_job_listing"]
    _upload_cfg.worker_mode = upload_snapshot["worker_mode"]
    _upload_cfg.worker_poll_seconds = upload_snapshot["worker_poll_seconds"]
    _upload_cfg.worker_claim_ttl_seconds = upload_snapshot["worker_claim_ttl_seconds"]
    _upload_cfg.worker_heartbeat_seconds = upload_snapshot["worker_heartbeat_seconds"]
    _upload_cfg.worker_stale_after_seconds = upload_snapshot["worker_stale_after_seconds"]
    _upload_cfg.max_upload_bytes = upload_snapshot["max_upload_bytes"]
    _upload_cfg.max_video_duration_seconds = upload_snapshot["max_video_duration_seconds"]
    _upload_cfg.max_concurrent_jobs = upload_snapshot["max_concurrent_jobs"]
    _upload_cfg.max_queue_depth = upload_snapshot["max_queue_depth"]
    _upload_cfg.max_job_attempts = upload_snapshot["max_job_attempts"]
    _upload_cfg.job_timeout_seconds = upload_snapshot["job_timeout_seconds"]
    _upload_cfg.artifact_backend = upload_snapshot["artifact_backend"]
    _upload_cfg.artifact_base_url = upload_snapshot["artifact_base_url"]
    _upload_cfg.artifact_object_dir = upload_snapshot["artifact_object_dir"]
    _upload_cfg.strict_startup_checks = upload_snapshot["strict_startup_checks"]
    _upload_cfg.job_failure_log_limit = upload_snapshot["job_failure_log_limit"]
    _upload_store._artifact_backend = artifact_backend_snapshot
    _upload_store._artifact_base_url = artifact_base_url_snapshot
    _upload_store._artifact_object_dir = artifact_object_dir_snapshot


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "deployment_ready" in data
    assert "worker_mode" in data


def test_security_headers_are_set() -> None:
    response = client.get("/health")

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["permissions-policy"] == "camera=(), microphone=(), geolocation=()"


def test_private_routes_disable_browser_caching() -> None:
    response = client.get("/jobs/missing")

    assert response.status_code == 404
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"


def test_spatial_query_empty():
    response = client.post("/query", json={"query": "where is the cup?"})
    assert response.status_code == 200
    assert response.json() == []


def test_job_listing_disabled_by_default() -> None:
    response = client.get("/jobs")
    assert response.status_code == 403


def test_operator_access_is_public() -> None:
    response = client.get("/operator/access")

    assert response.status_code == 200
    assert response.json()["mode"] in {"loopback", "token"}


def test_product_privacy_is_public() -> None:
    response = client.get("/product/privacy")

    assert response.status_code == 200
    data = response.json()
    assert data["delete_supported"] is True
    assert "retention" in data["summary"].lower()
    assert data["artifact_backend"] == _upload_cfg.artifact_backend


def test_product_upload_guidance_is_public() -> None:
    response = client.get("/product/upload-guidance")

    assert response.status_code == 200
    data = response.json()
    assert data["max_upload_bytes"] == _upload_cfg.max_upload_bytes
    assert data["max_video_duration_seconds"] == _upload_cfg.max_video_duration_seconds
    assert data["recommended_duration_seconds"] == {"min": 20, "max": 60}
    assert "video/" in data["accepted_media_prefixes"]
    assert ".mp4" in data["accepted_extensions"]
    assert data["one_room_only"] is True
    assert data["checklist"]
    assert data["retry_guidance"]


def test_product_waitlist_is_public() -> None:
    response = client.post(
        "/product/waitlist",
        json={"email": "beta@example.com", "use_case": "toddler-proofing"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["accepted"] is True
    assert data["waitlist_count"] == 1
    assert _upload_store.load_waitlist_entries()[0]["email"] == "beta@example.com"


def test_product_waitlist_deduplicates_email() -> None:
    first = client.post(
        "/product/waitlist",
        json={
            "email": "Beta@Example.com",
            "use_case": "renter move-in",
            "source": "hero_waitlist",
            "audience_mode": "renter",
        },
    )
    second = client.post(
        "/product/waitlist",
        json={
            "email": "beta@example.com",
            "use_case": "pet safety",
            "source": "settings",
            "audience_mode": "pet",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["waitlist_count"] == 1
    assert "already" in second.json()["message"].lower()
    entries = _upload_store.load_waitlist_entries()
    assert len(entries) == 1
    assert entries[0]["email"] == "beta@example.com"
    assert entries[0]["source"] == "hero_waitlist"
    assert entries[0]["audience_mode"] == "renter"


def test_product_event_is_public() -> None:
    response = client.post(
        "/product/events",
        json={"event_name": "cta_start_scan", "surface": "hero"},
    )

    assert response.status_code == 204
    events = _upload_store.load_product_events()
    assert events[0]["event_name"] == "cta_start_scan"
    assert events[0]["surface"] == "hero"


def test_product_event_accepts_attribution_fields() -> None:
    response = client.post(
        "/product/events",
        json={
            "event_name": "cta_start_scan",
            "surface": "hero",
            "session_id": "session-123",
            "client_ts": "2026-05-01T00:00:00Z",
            "referrer": "https://example.com/post",
            "utm_source": "friend",
            "utm_campaign": "beta",
            "mission_id": "cord-safari",
            "challenge_id": "cord-safari",
            "audience_mode": "pet",
            "room_label": "Living room",
            "room_labeled": True,
        },
    )

    assert response.status_code == 204
    event = _upload_store.load_product_events()[0]
    assert event["session_id"] == "session-123"
    assert event["utm_source"] == "friend"
    assert event["utm_campaign"] == "beta"
    assert event["mission_id"] == "cord-safari"
    assert event["challenge_id"] == "cord-safari"
    assert event["audience_mode"] == "pet"
    assert event["room_label"] == "Living room"
    assert event["room_labeled"] is True


def test_product_event_rejects_unknown_event_name() -> None:
    response = client.post(
        "/product/events",
        json={"event_name": "surprise_growth_hack", "surface": "hero"},
    )

    assert response.status_code == 400
    assert _upload_store.load_product_events() == []


def test_product_event_accepts_beta_loop_events() -> None:
    response = client.post(
        "/product/events",
        json={"event_name": "fix_plan_copied", "surface": "report_action_loop"},
    )

    assert response.status_code == 204
    assert _upload_store.load_product_events()[0]["event_name"] == "fix_plan_copied"


def test_product_event_accepts_warm_trust_design_events() -> None:
    for event_name in (
        "landing_section_viewed",
        "sample_cta_clicked",
        "first_run_started",
        "scan_preflight_failed",
        "confidence_inspector_opened",
        "report_share_card_copied",
        "pdf_export_clicked",
        "rescan_prompt_clicked",
    ):
        response = client.post(
            "/product/events",
            json={"event_name": event_name, "surface": "warm_trust_ui"},
        )

        assert response.status_code == 204

    events = _upload_store.load_product_events()
    assert [event["event_name"] for event in events] == [
        "landing_section_viewed",
        "sample_cta_clicked",
        "first_run_started",
        "scan_preflight_failed",
        "confidence_inspector_opened",
        "report_share_card_copied",
        "pdf_export_clicked",
        "rescan_prompt_clicked",
    ]


def test_product_event_persists_preflight_failure_metadata() -> None:
    response = client.post(
        "/product/events",
        json={
            "event_name": "scan_preflight_failed",
            "surface": "guided_scan_wizard",
            "file_type": "application/zip",
            "file_size": 12345,
            "reason": "Unsupported file type",
        },
    )

    assert response.status_code == 204
    event = _upload_store.load_product_events()[0]
    assert event["file_type"] == "application/zip"
    assert event["file_size"] == 12345
    assert event["reason"] == "Unsupported file type"


def test_job_listing_requires_token_when_configured() -> None:
    _api_cfg.enable_job_listing = True
    _api_cfg.access_token = "secret-token"

    unauthenticated = client.get("/jobs")
    authenticated = client.get("/jobs", headers={"Authorization": "Bearer secret-token"})

    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 200


def test_operator_settings_require_token_when_configured() -> None:
    _api_cfg.access_token = "secret-token"

    unauthenticated = client.get("/operator/settings")
    authenticated = client.get(
        "/operator/settings",
        headers={"Authorization": "Bearer secret-token"},
    )

    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 200
    assert authenticated.json()["uploads"]["worker_mode"] == _upload_cfg.worker_mode
    assert authenticated.json()["uploads"]["max_queue_depth"] == _upload_cfg.max_queue_depth
    assert authenticated.json()["uploads"]["max_storage_bytes"] == _upload_cfg.max_storage_bytes
    assert authenticated.json()["uploads"]["artifact_backend"] == _upload_cfg.artifact_backend
    assert authenticated.json()["providers"]["primary_provider"] in {"ollama", "claude", "openai"}
    assert "upload_success_rate" in authenticated.json()["product"]
    assert "waitlist_signups" in authenticated.json()["product"]
    assert "beta_invite_events" in authenticated.json()["product"]
    assert "capture_coach_events" in authenticated.json()["product"]
    assert "same_room_rescan_events" in authenticated.json()["product"]
    assert "reviewed_jobs" in authenticated.json()["evaluation"]
    assert "saved_eval_candidates" in authenticated.json()["evaluation"]
    assert "release_gates" in authenticated.json()["evaluation"]
    assert "beta_inbox" in authenticated.json()
    assert "funnel" in authenticated.json()["beta_inbox"]
    assert "deployment_ready" in authenticated.json()["system"]
    assert "startup_checks" in authenticated.json()["system"]


def test_operator_beta_inbox_summarizes_learning_loop_without_raw_email() -> None:
    _api_cfg.access_token = "secret-token"
    _upload_store.append_waitlist_entry(
        {
            "email": "person@example.com",
            "use_case": "pet safety",
            "source": "hero_waitlist",
            "audience_mode": "pet",
            "created_at": "2026-05-01T00:00:00+00:00",
        }
    )
    _upload_store.append_product_event(
        {
            "event_name": "cta_start_scan",
            "surface": "hero",
            "created_at": "2026-05-01T00:00:00+00:00",
        }
    )
    _upload_jobs["jobwrong"] = {
        "job_id": "jobwrong",
        "filename": "room.mp4",
        "status": "complete",
        "created_at": "2026-05-01T00:00:00+00:00",
        "completed_at": "2026-05-01T00:00:10+00:00",
        "feedback_summary": {"useful": 0, "wrong": 1, "duplicate": 0},
        "evaluation_summary": {"needs_review": True, "summary": "1 finding needs review."},
        "human_evaluation": {"missed_hazards": ["Missed a cord near the door."]},
        "summary": {"room_label": "Living room"},
    }
    _upload_jobs["jobfail"] = {
        "job_id": "jobfail",
        "filename": "bad.mov",
        "status": "error",
        "stage": "analyze",
        "error": "video decode failed",
        "created_at": "2026-05-01T00:00:00+00:00",
        "failed_at": "2026-05-01T00:00:03+00:00",
    }

    response = client.get(
        "/operator/settings",
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 200
    inbox = response.json()["beta_inbox"]
    assert inbox["funnel"]["cta_start_scan"] == 1
    assert inbox["recent_waitlist"][0]["email"] == "pe***@example.com"
    assert "person@example.com" not in str(inbox)
    assert inbox["failed_uploads"][0]["job_id"] == "jobfail"
    assert inbox["negative_feedback_reports"][0]["job_id"] == "jobwrong"
    assert inbox["missed_hazard_notes"][0]["note"] == "Missed a cord near the door."


def test_operator_storage_prune_requires_token_when_configured() -> None:
    _api_cfg.access_token = "secret-token"

    unauthenticated = client.post("/operator/storage/prune")
    authenticated = client.post(
        "/operator/storage/prune",
        headers={"Authorization": "Bearer secret-token"},
    )

    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 200
    assert "deleted_jobs" in authenticated.json()


def test_sample_report_is_public(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sample_report() -> dict[str, object]:
        job = {
            "job_id": "sample-walkthrough",
            "filename": "sample_walkthrough",
            "room_label": "Sample living room",
            "is_sample": True,
            "sample_key": "walkthrough",
            "audience_mode": "general",
            "status": "complete",
            "stage": "complete",
            "progress": 1.0,
            "objects": [],
            "risks": [],
            "fix_first": [],
            "summary": {"filename": "sample_walkthrough", "hazard_count": 0, "object_count": 0},
            "recommendations": [],
            "evidence_frames": [],
            "scan_quality": {"status": "good", "score": 0.74, "usable": True, "warnings": []},
            "trust_notes": ["This is a built-in sample walkthrough."],
            "scene_source": "estimated_multiview",
            "finding_feedback": [],
            "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
            "human_evaluation": None,
            "finding_follow_up": [],
            "resolution_summary": None,
            "report_url": "/sample-report/report.pdf",
            "share_url": "/app?view=report&sample=walkthrough",
            "error": None,
            "artifacts": {},
            "attempt_count": 0,
            "queued_at": "2026-04-20T00:00:00+00:00",
            "started_at": "2026-04-20T00:00:00+00:00",
            "completed_at": "2026-04-20T00:00:00+00:00",
        }
        return {
            "job": server_mod._ensure_job_derived_fields(job),
            "evidence_artifacts": {},
            "replay_artifacts": {},
            "report_pdf": b"pdf",
        }

    monkeypatch.setattr(server_mod, "_build_sample_report", fake_sample_report)

    response = client.get("/sample-report")

    assert response.status_code == 200
    data = response.json()
    assert data["is_sample"] is True
    assert data["sample_key"] == "walkthrough"
    assert data["share_url"] == "/app?view=report&sample=walkthrough"


def test_upload_job_status_exposes_report_fields():
    _upload_jobs["job12345"] = {
        "job_id": "job12345",
        "filename": "living-room.mp4",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [
            {
                "object_id": "obj-1",
                "hazard_code": "top_heavy_tipping",
                "object_label": "Floor lamp",
                "risk_score": 0.82,
                "description": "Tall lamp leaning near a walkway.",
                "what": "Tall lamp leaning near a walkway.",
                "why_it_matters": "Tall items can tip when bumped.",
                "what_to_do_next": "Move it away from the walkway or secure the base.",
                "severity": "high",
                "location_label": "front-right zone",
                "confidence": 0.79,
                "confidence_label": "strong",
                "reasoning": {
                    "support_summary": "3 supporting observations across the scan",
                    "signals": ["slenderness: 2.1"],
                    "grounding_confidence": 0.76,
                    "rule_hits": ["slenderness triggered at 2.1"],
                    "evidence_ids": ["f00-r01", "f01-r01"],
                    "object_snapshot": {
                        "material": "Metal",
                        "estimated_height_m": 1.82,
                        "estimated_width_m": 0.38,
                        "observation_count": 3,
                        "location_label": "front-right zone",
                    },
                },
                "replay": {
                    "replay_id": "finding-01",
                    "hazard_code": "top_heavy_tipping",
                    "object_id": "obj-1",
                    "caption": "Floor lamp replay",
                    "frame_count": 2,
                    "media_type": "image/gif",
                    "image_url": "/jobs/job12345/replays/finding-01",
                },
                "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
                "latest_feedback": None,
            }
        ],
        "fix_first": [
            {
                "title": "Top-heavy tipping risk",
                "action": "Move it away from the walkway or secure the base.",
                "why": "Tall items can tip when bumped.",
                "location": "front-right zone",
                "severity": "high",
                "confidence_label": "strong",
            }
        ],
        "summary": {
            "filename": "living-room.mp4",
            "hazard_count": 1,
            "object_count": 1,
            "scene_source": "heuristic_estimate",
            "confidence_label": "Approximate spatial grounding",
            "scan_quality_label": "Fair",
            "scan_quality_score": 0.62,
            "top_hazard_label": "Floor lamp",
            "top_severity": "high",
        },
        "recommendations": [
            {
                "title": "Stabilize tall lamp",
                "action": "Move it away from the walkway or secure the base.",
                "why": "Tall items can tip when bumped.",
                "location": "front-right zone",
                "priority": "high",
            }
        ],
        "evidence_frames": [
            {
                "image_url": "data:image/jpeg;base64,ZmFrZQ==",
                "caption": "Lamp near walkway",
                "confidence": 0.82,
            }
        ],
        "scan_quality": {
            "status": "fair",
            "score": 0.62,
            "usable": True,
            "warnings": ["The walkthrough covers too little motion to ground objects confidently."],
            "retry_guidance": [
                "Move slowly across the room so objects appear from more than one viewpoint."
            ],
            "metrics": {"frame_count": 5},
        },
        "trust_notes": [
            "Locations are approximate because upload-side geometry is still heuristic."
        ],
        "scene_source": "heuristic_estimate",
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": "/reports/job12345.pdf",
        "artifacts": {
            "report_pdf": {
                "kind": "report_pdf",
                "storage_backend": "local_fs",
                "storage_key": "jobs/job12345/report.pdf",
                "relative_path": "report.pdf",
                "media_type": "application/pdf",
                "size_bytes": 12,
                "url": "/reports/job12345.pdf",
            }
        },
        "error": None,
    }

    response = client.get("/jobs/job12345")
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["top_hazard_label"] == "Floor lamp"
    assert data["fix_first"][0]["title"] == "Top-heavy tipping risk"
    assert data["recommendations"][0]["priority"] == "high"
    assert data["scan_quality"]["status"] == "fair"
    assert data["evidence_frames"][0]["caption"] == "Lamp near walkway"
    assert data["report_url"] == "/reports/job12345.pdf"
    assert data["share_url"] == "/app?view=report&job=job12345"
    assert data["audience_mode"] == "general"
    assert data["artifacts"]["report_pdf"]["storage_key"] == "jobs/job12345/report.pdf"
    assert data["scene_source"] == "heuristic_estimate"
    assert data["evaluation_summary"]["total_findings"] == 1
    assert data["evaluation_summary"]["pending_findings"] == 1
    assert isinstance(data["weekend_fix_list"], list)
    assert isinstance(data["room_wins"], list)


def test_download_evidence_returns_file_backed_artifact(tmp_path: Path) -> None:
    _upload_jobs["jobev001"] = {
        "job_id": "jobev001",
        "filename": "kitchen.mp4",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [],
        "fix_first": [],
        "summary": {"filename": "kitchen.mp4", "hazard_count": 0, "object_count": 0},
        "recommendations": [],
        "evidence_frames": [
            {
                "evidence_id": "f00-r01",
                "caption": "Cup observation",
                "image_url": "/jobs/jobev001/evidence/f00-r01",
            }
        ],
        "scan_quality": {"status": "good", "score": 0.8, "usable": True, "warnings": []},
        "trust_notes": [],
        "scene_source": "estimated_multiview",
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": "/reports/jobev001.pdf",
        "error": None,
    }
    _upload_store.save_evidence_image("jobev001", "f00-r01", b"jpeg-evidence")

    response = client.get("/jobs/jobev001/evidence/f00-r01")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    assert response.content == b"jpeg-evidence"


def test_download_replay_returns_file_backed_artifact(tmp_path: Path) -> None:
    _upload_jobs["jobreplay1"] = {
        "job_id": "jobreplay1",
        "filename": "kitchen.mp4",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [
            {
                "object_id": "track-01",
                "hazard_code": "edge_placement",
                "hazard_title": "Object placed near an edge",
                "replay": {
                    "replay_id": "finding-01",
                    "image_url": "/jobs/jobreplay1/replays/finding-01",
                },
            }
        ],
        "fix_first": [],
        "summary": {"filename": "kitchen.mp4", "hazard_count": 1, "object_count": 1},
        "recommendations": [],
        "evidence_frames": [],
        "scan_quality": {"status": "good", "score": 0.8, "usable": True, "warnings": []},
        "trust_notes": [],
        "scene_source": "estimated_multiview",
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": "/reports/jobreplay1.pdf",
        "error": None,
    }
    _upload_store.save_replay_gif("jobreplay1", "finding-01", b"GIF89a-fixture")

    response = client.get("/jobs/jobreplay1/replays/finding-01")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/gif")
    assert response.content == b"GIF89a-fixture"


def test_report_pdf_missing_job_returns_404():
    response = client.get("/reports/missing.pdf")
    assert response.status_code == 404


def test_report_pdf_download_for_completed_job():
    _upload_jobs["jobpdf01"] = {
        "job_id": "jobpdf01",
        "filename": "nursery.mp4",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [
            {
                "object_id": "obj-2",
                "hazard_code": "unstable_stack",
                "object_label": "Book stack",
                "risk_score": 0.61,
                "description": "Stack is leaning near the crib.",
                "what": "Stack is leaning near the crib.",
                "why_it_matters": "Loose stacks can slide or fall.",
                "what_to_do_next": "Move the heaviest books lower or flatten the pile.",
                "severity": "medium",
                "location_label": "left wall",
                "confidence": 0.71,
                "confidence_label": "approximate",
                "reasoning": {
                    "support_summary": "2 supporting observations across the scan",
                    "signals": ["position variance: 0.14"],
                    "grounding_confidence": 0.67,
                },
                "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
                "latest_feedback": None,
            }
        ],
        "fix_first": [
            {
                "title": "Unstable stack",
                "action": "Move the heaviest books lower or flatten the pile.",
                "why": "Loose stacks can slide or fall.",
                "location": "left wall",
                "severity": "medium",
                "confidence_label": "approximate",
            }
        ],
        "summary": {
            "filename": "nursery.mp4",
            "hazard_count": 1,
            "object_count": 2,
            "scene_source": "heuristic_estimate",
            "confidence_label": "Approximate spatial grounding",
            "scan_quality_label": "Good",
            "scan_quality_score": 0.79,
            "top_hazard_label": "Book stack",
            "top_severity": "medium",
        },
        "recommendations": [
            {
                "title": "Lower the stack",
                "action": "Move the heaviest books lower or flatten the pile.",
                "why": "Loose stacks can slide or fall.",
                "location": "left wall",
                "priority": "medium",
            }
        ],
        "evidence_frames": [],
        "scan_quality": {
            "status": "good",
            "score": 0.79,
            "usable": True,
            "warnings": [],
            "retry_guidance": [],
        },
        "trust_notes": ["Treat scene positions as approximate."],
        "scene_source": "heuristic_estimate",
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": "/reports/jobpdf01.pdf",
        "error": None,
    }

    response = client.get("/reports/jobpdf01.pdf")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert "atlas0-report-jobpdf01.pdf" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF-")
    assert b"Report posture" in response.content


def test_report_pdf_download_accepts_query_token() -> None:
    _api_cfg.access_token = "secret-token"
    _upload_jobs["jobpdf02"] = {
        "job_id": "jobpdf02",
        "filename": "study.mp4",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [],
        "fix_first": [],
        "summary": {"filename": "study.mp4", "hazard_count": 0, "object_count": 0},
        "recommendations": [],
        "evidence_frames": [],
        "scan_quality": {"status": "good", "score": 0.8, "usable": True, "warnings": []},
        "trust_notes": [],
        "scene_source": "estimated_multiview",
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": "/reports/jobpdf02.pdf",
        "error": None,
    }
    _upload_store.create_job(_upload_jobs["jobpdf02"])
    _upload_store.save_report_pdf("jobpdf02", b"%PDF-1.4\nfixture\n")

    response = client.get("/reports/jobpdf02.pdf?access_token=secret-token")

    assert response.status_code == 200


def test_report_pdf_download_works_with_object_store_backend() -> None:
    _upload_cfg.artifact_backend = "object_store_fs"
    _upload_cfg.artifact_object_dir = str(_upload_store.root_dir / "_objects")
    _upload_store._artifact_backend = "object_store_fs"
    _upload_store._artifact_object_dir = _upload_store.root_dir / "_objects"
    _upload_jobs["jobpdf03"] = {
        "job_id": "jobpdf03",
        "filename": "kitchen.mp4",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [],
        "fix_first": [],
        "summary": {"filename": "kitchen.mp4", "hazard_count": 0, "object_count": 0},
        "recommendations": [],
        "evidence_frames": [],
        "scan_quality": {"status": "good", "score": 0.8, "usable": True, "warnings": []},
        "trust_notes": [],
        "scene_source": "estimated_multiview",
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": "/reports/jobpdf03.pdf",
        "error": None,
    }
    _upload_store.create_job(_upload_jobs["jobpdf03"])
    _upload_store.save_report_pdf("jobpdf03", b"%PDF-1.4\nfixture\n")

    response = client.get("/reports/jobpdf03.pdf")

    assert response.status_code == 200
    assert response.content.startswith(b"%PDF-")


def test_record_job_feedback_updates_job_and_finding() -> None:
    _upload_jobs["jobfeed1"] = {
        "job_id": "jobfeed1",
        "filename": "hallway.mp4",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [
            {
                "object_id": "track-01",
                "hazard_code": "walkway_clutter",
                "hazard_title": "Walkway clutter",
                "object_label": "Box",
                "risk_score": 0.58,
                "severity": "moderate",
                "location_label": "front-center",
                "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
                "latest_feedback": None,
            }
        ],
        "fix_first": [],
        "summary": {"filename": "hallway.mp4", "hazard_count": 1, "object_count": 1},
        "recommendations": [],
        "evidence_frames": [],
        "scan_quality": {"status": "fair", "score": 0.61, "usable": True, "warnings": []},
        "trust_notes": [],
        "scene_source": "estimated_multiview",
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": "/reports/jobfeed1.pdf",
        "error": None,
    }

    response = client.post(
        "/jobs/jobfeed1/feedback",
        json={
            "hazard_code": "walkway_clutter",
            "object_id": "track-01",
            "verdict": "useful",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["feedback_summary"]["useful"] == 1
    assert data["finding_feedback"][0]["verdict"] == "useful"
    assert data["risks"][0]["latest_feedback"] == "useful"
    assert data["risks"][0]["feedback_summary"]["useful"] == 1
    assert data["evaluation_summary"]["reviewed_findings"] == 1
    assert data["evaluation_summary"]["pending_findings"] == 0
    assert data["evaluation_summary"]["useful_events"] == 1


def test_record_job_follow_up_updates_resolution_summary() -> None:
    _upload_jobs["jobfollow1"] = {
        "job_id": "jobfollow1",
        "filename": "hallway.mp4",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [
            {
                "object_id": "track-01",
                "hazard_code": "walkway_clutter",
                "hazard_title": "Walkway clutter",
                "object_label": "Box",
                "risk_score": 0.58,
                "severity": "moderate",
                "location_label": "front-center",
                "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
                "latest_feedback": None,
                "follow_up_status": None,
            }
        ],
        "fix_first": [],
        "summary": {"filename": "hallway.mp4", "hazard_count": 1, "object_count": 1},
        "recommendations": [],
        "evidence_frames": [],
        "scan_quality": {"status": "fair", "score": 0.61, "usable": True, "warnings": []},
        "trust_notes": [],
        "scene_source": "estimated_multiview",
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "human_evaluation": None,
        "finding_follow_up": [],
        "report_url": "/reports/jobfollow1.pdf",
        "error": None,
    }

    response = client.post(
        "/jobs/jobfollow1/follow-up",
        json={
            "hazard_code": "walkway_clutter",
            "object_id": "track-01",
            "status": "resolved",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["finding_follow_up"][0]["status"] == "resolved"
    assert data["risks"][0]["follow_up_status"] == "resolved"
    assert data["resolution_summary"]["resolved_count"] == 1
    assert data["resolution_summary"]["open_count"] == 0


def test_record_job_evaluation_updates_job_summary() -> None:
    _upload_jobs["jobeval1"] = {
        "job_id": "jobeval1",
        "filename": "hallway.mp4",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [
            {
                "object_id": "track-01",
                "hazard_code": "walkway_clutter",
                "hazard_title": "Walkway clutter",
                "object_label": "Box",
                "risk_score": 0.58,
                "severity": "moderate",
                "location_label": "front-center",
                "feedback_summary": {"useful": 1, "wrong": 0, "duplicate": 0},
                "latest_feedback": "useful",
            }
        ],
        "fix_first": [],
        "summary": {"filename": "hallway.mp4", "hazard_count": 1, "object_count": 1},
        "recommendations": [],
        "evidence_frames": [],
        "scan_quality": {"status": "fair", "score": 0.61, "usable": True, "warnings": []},
        "trust_notes": [],
        "scene_source": "estimated_multiview",
        "finding_feedback": [
            {
                "hazard_code": "walkway_clutter",
                "object_id": "track-01",
                "verdict": "useful",
                "finding_key": "track-01:walkway_clutter",
            }
        ],
        "feedback_summary": {"useful": 1, "wrong": 0, "duplicate": 0},
        "report_url": "/reports/jobeval1.pdf",
        "error": None,
    }

    response = client.post(
        "/jobs/jobeval1/evaluation",
        json={
            "status": "missed_hazard",
            "missed_hazards": ["loose cable by doorway"],
            "note": "Caught one missed trip hazard during review.",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["human_evaluation"]["status"] == "missed_hazard"
    assert data["evaluation_summary"]["missed_hazard_count"] == 1
    assert data["evaluation_summary"]["human_status"] == "missed_hazard"
    assert data["evaluation_summary"]["needs_review"] is True


def test_export_eval_candidate_persists_review_ready_case() -> None:
    _upload_jobs["jobevalexport"] = {
        "job_id": "jobevalexport",
        "filename": "kitchen.mp4",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [
            {
                "object_id": "track-01",
                "hazard_code": "edge_placement",
                "hazard_title": "Edge placement",
                "object_label": "Glass bowl",
                "risk_score": 0.73,
                "severity": "high",
                "location_label": "counter edge",
                "feedback_summary": {"useful": 1, "wrong": 0, "duplicate": 0},
                "latest_feedback": "useful",
            }
        ],
        "fix_first": [],
        "summary": {"filename": "kitchen.mp4", "hazard_count": 1, "object_count": 1},
        "recommendations": [],
        "evidence_frames": [],
        "scan_quality": {"status": "good", "score": 0.73, "usable": True, "warnings": []},
        "trust_notes": [],
        "scene_source": "estimated_multiview",
        "finding_feedback": [
            {
                "hazard_code": "edge_placement",
                "object_id": "track-01",
                "verdict": "useful",
                "finding_key": "track-01:edge_placement",
            }
        ],
        "feedback_summary": {"useful": 1, "wrong": 0, "duplicate": 0},
        "report_url": "/reports/jobevalexport.pdf",
        "error": None,
    }
    server_mod._ensure_job_derived_fields(_upload_jobs["jobevalexport"])

    response = client.post(
        "/jobs/jobevalexport/eval-candidate",
        json={"label": "kitchen_case_01"},
    )

    assert response.status_code == 200
    candidates = _upload_store.load_eval_candidates()
    assert candidates[0]["candidate_id"] == "kitchen_case_01"
    assert candidates[0]["review_ready"] is True
    assert candidates[0]["top_hazard_code"] == "edge_placement"


def test_export_eval_candidate_rejects_unsafe_label() -> None:
    _upload_jobs["jobevalbad"] = {
        "job_id": "jobevalbad",
        "filename": "office.mp4",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [
            {
                "hazard_code": "edge_placement",
                "object_id": "track-01",
                "object_label": "box",
                "severity": "medium",
                "confidence": 0.8,
            }
        ],
        "fix_first": [],
        "summary": {"filename": "office.mp4", "hazard_count": 1, "object_count": 1},
        "recommendations": [],
        "evidence_frames": [],
        "scan_quality": {"status": "good", "score": 0.73, "usable": True, "warnings": []},
        "trust_notes": [],
        "scene_source": "estimated_multiview",
        "finding_feedback": [{"hazard_code": "edge_placement", "verdict": "useful"}],
        "feedback_summary": {"useful": 1, "wrong": 0, "duplicate": 0},
        "evaluation_summary": {"reviewed_findings": 1},
        "human_evaluation": {"status": "confirmed"},
        "report_url": "/reports/jobevalbad.pdf",
        "error": None,
    }
    server_mod._ensure_job_derived_fields(_upload_jobs["jobevalbad"])

    response = client.post(
        "/jobs/jobevalbad/eval-candidate",
        json={"label": "../escape"},
    )

    assert response.status_code == 400
    assert "candidate_id" in response.json()["detail"]


def test_delete_upload_job_removes_persisted_artifacts() -> None:
    _upload_jobs["jobdel01"] = {
        "job_id": "jobdel01",
        "filename": "office.mp4",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [],
        "fix_first": [],
        "summary": {"filename": "office.mp4", "hazard_count": 0, "object_count": 0},
        "recommendations": [],
        "evidence_frames": [],
        "scan_quality": {"status": "good", "score": 0.73, "usable": True, "warnings": []},
        "trust_notes": [],
        "scene_source": "estimated_multiview",
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": "/reports/jobdel01.pdf",
        "error": None,
    }
    _upload_store.create_job(_upload_jobs["jobdel01"])
    _upload_store.save_report_pdf("jobdel01", b"pdf")

    response = client.delete("/jobs/jobdel01")

    assert response.status_code == 204
    assert "jobdel01" not in _upload_jobs
    assert not _upload_store.job_dir("jobdel01").exists()


def test_delete_upload_job_rejects_unsafe_job_id() -> None:
    response = client.delete("/jobs/%2E%2E")

    assert response.status_code == 400
    assert "job_id" in response.json()["detail"]


def test_upload_rejects_when_queue_depth_is_full() -> None:
    _upload_cfg.max_queue_depth = 1
    _upload_jobs["busyjob1"] = {
        "job_id": "busyjob1",
        "filename": "busy.mp4",
        "status": "queued",
        "stage": "upload",
        "progress": 0.1,
        "objects": None,
        "risks": None,
        "fix_first": None,
        "summary": None,
        "recommendations": None,
        "evidence_frames": None,
        "scan_quality": None,
        "trust_notes": None,
        "scene_source": None,
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": None,
        "error": None,
    }

    response = client.post(
        "/upload",
        headers={"content-type": "application/octet-stream", "x-filename": "room.bin"},
        content=b"1234",
    )

    assert response.status_code == 429


def test_upload_rejects_request_over_size_limit() -> None:
    _upload_cfg.max_upload_bytes = 4

    response = client.post(
        "/upload",
        headers={"content-type": "application/octet-stream", "x-filename": "big.bin"},
        content=b"12345",
    )

    assert response.status_code == 413


def test_upload_rejects_video_over_duration_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _upload_cfg.max_video_duration_seconds = 10.0
    monkeypatch.setattr(
        server_mod,
        "probe_video_metadata",
        lambda _content: VideoMetadata(duration_s=12.5, frame_count=42),
    )

    response = client.post(
        "/upload",
        headers={"content-type": "video/mp4", "x-filename": "long.mp4"},
        content=b"fake-video",
    )

    assert response.status_code == 413


def test_upload_persists_job_input_and_enqueues_work(monkeypatch: pytest.MonkeyPatch) -> None:
    queued: list[str] = []

    async def fake_enqueue(job_id: str) -> None:
        queued.append(job_id)

    monkeypatch.setattr(server_mod, "_enqueue_upload_job", fake_enqueue)

    response = client.post(
        "/upload",
        headers={"content-type": "application/octet-stream", "x-filename": "room.bin"},
        content=b"1234",
    )

    assert response.status_code == 200
    data = response.json()
    job_id = data["job_id"]
    assert queued == [job_id]
    assert _upload_store.has_job_input(job_id) is True
    assert data["artifacts"]["queued_input"]["storage_key"] == f"jobs/{job_id}/queued-input.bin"


def test_upload_accepts_room_label_header(monkeypatch: pytest.MonkeyPatch) -> None:
    queued: list[str] = []

    async def fake_enqueue(job_id: str) -> None:
        queued.append(job_id)

    monkeypatch.setattr(server_mod, "_enqueue_upload_job", fake_enqueue)

    response = client.post(
        "/upload",
        headers={
            "content-type": "application/octet-stream",
            "x-filename": "room.bin",
            "x-room-label": "Nursery",
        },
        content=b"1234",
    )

    assert response.status_code == 200
    data = response.json()
    assert data["room_label"] == "Nursery"
    assert queued == [data["job_id"]]


def test_upload_accepts_audience_mode_header(monkeypatch: pytest.MonkeyPatch) -> None:
    queued: list[str] = []

    async def fake_enqueue(job_id: str) -> None:
        queued.append(job_id)

    monkeypatch.setattr(server_mod, "_enqueue_upload_job", fake_enqueue)

    response = client.post(
        "/upload",
        headers={
            "content-type": "application/octet-stream",
            "x-filename": "room.bin",
            "x-audience-mode": "pet",
        },
        content=b"1234",
    )

    assert response.status_code == 200
    data = response.json()
    assert data["audience_mode"] == "pet"
    assert queued == [data["job_id"]]


def test_upload_rejects_unknown_audience_mode() -> None:
    response = client.post(
        "/upload",
        headers={
            "content-type": "application/octet-stream",
            "x-filename": "room.bin",
            "x-audience-mode": "grandma-mode",
        },
        content=b"1234",
    )

    assert response.status_code == 400
    assert "Audience mode" in response.json()["detail"]


def test_completed_job_exposes_room_score_and_comparison() -> None:
    _upload_jobs["oldroom1"] = {
        "job_id": "oldroom1",
        "filename": "nursery-before.mp4",
        "room_label": "Nursery",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [{"hazard_code": "edge_placement", "risk_score": 0.82, "priority_score": 0.84}],
        "fix_first": [],
        "summary": {"filename": "nursery-before.mp4", "hazard_count": 1, "object_count": 2},
        "recommendations": [],
        "evidence_frames": [],
        "scan_quality": {"status": "fair", "score": 0.59, "usable": True, "warnings": []},
        "trust_notes": [],
        "scene_source": "estimated_multiview",
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": "/reports/oldroom1.pdf",
        "completed_at": "2026-04-17T10:00:00+00:00",
        "error": None,
    }
    _upload_jobs["newroom1"] = {
        "job_id": "newroom1",
        "filename": "nursery-after.mp4",
        "room_label": "Nursery",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [],
        "fix_first": [],
        "summary": {"filename": "nursery-after.mp4", "hazard_count": 0, "object_count": 2},
        "recommendations": [],
        "evidence_frames": [],
        "scan_quality": {"status": "good", "score": 0.81, "usable": True, "warnings": []},
        "trust_notes": [],
        "scene_source": "estimated_multiview",
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": "/reports/newroom1.pdf",
        "completed_at": "2026-04-18T10:00:00+00:00",
        "error": None,
    }

    response = client.get("/jobs/newroom1")

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["room_score"] >= 0
    assert data["room_comparison"]["previous_job_id"] == "oldroom1"
    assert data["room_history"][0]["job_id"] == "oldroom1"


def test_room_history_filters_to_same_audience_mode() -> None:
    _upload_jobs["old-general"] = {
        "job_id": "old-general",
        "filename": "nursery-general.mp4",
        "room_label": "Nursery",
        "audience_mode": "general",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [],
        "fix_first": [],
        "summary": {"filename": "nursery-general.mp4", "hazard_count": 0, "object_count": 2},
        "recommendations": [],
        "evidence_frames": [],
        "scan_quality": {"status": "good", "score": 0.73, "usable": True, "warnings": []},
        "trust_notes": [],
        "scene_source": "estimated_multiview",
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": "/reports/old-general.pdf",
        "completed_at": "2026-04-17T10:00:00+00:00",
        "error": None,
    }
    _upload_jobs["old-toddler"] = {
        "job_id": "old-toddler",
        "filename": "nursery-toddler-before.mp4",
        "room_label": "Nursery",
        "audience_mode": "toddler",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [],
        "fix_first": [],
        "summary": {"filename": "nursery-toddler-before.mp4", "hazard_count": 0, "object_count": 2},
        "recommendations": [],
        "evidence_frames": [],
        "scan_quality": {"status": "fair", "score": 0.61, "usable": True, "warnings": []},
        "trust_notes": [],
        "scene_source": "estimated_multiview",
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": "/reports/old-toddler.pdf",
        "completed_at": "2026-04-18T10:00:00+00:00",
        "error": None,
    }
    _upload_jobs["new-toddler"] = {
        "job_id": "new-toddler",
        "filename": "nursery-toddler-after.mp4",
        "room_label": "Nursery",
        "audience_mode": "toddler",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [],
        "fix_first": [],
        "summary": {"filename": "nursery-toddler-after.mp4", "hazard_count": 0, "object_count": 2},
        "recommendations": [],
        "evidence_frames": [],
        "scan_quality": {"status": "good", "score": 0.82, "usable": True, "warnings": []},
        "trust_notes": [],
        "scene_source": "estimated_multiview",
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": "/reports/new-toddler.pdf",
        "completed_at": "2026-04-19T10:00:00+00:00",
        "error": None,
    }

    response = client.get("/jobs/new-toddler")

    assert response.status_code == 200
    data = response.json()
    assert data["room_history"][0]["job_id"] == "old-toddler"
    assert all(entry["job_id"] != "old-general" for entry in data["room_history"])


async def test_resume_pending_upload_jobs_requeues_processing_jobs() -> None:
    _upload_jobs["jobresume"] = {
        "job_id": "jobresume",
        "filename": "resume.mp4",
        "content_type": "video/mp4",
        "status": "processing",
        "stage": "vlm",
        "progress": 0.5,
        "objects": None,
        "risks": None,
        "fix_first": None,
        "summary": None,
        "recommendations": None,
        "evidence_frames": None,
        "scan_quality": None,
        "trust_notes": None,
        "scene_source": None,
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": None,
        "error": None,
    }
    _upload_store.create_job(_upload_jobs["jobresume"])
    _upload_store.save_job_input("jobresume", "resume.mp4", b"video")

    await server_mod._resume_pending_upload_jobs()

    queue = _state["upload_queue"]
    assert _upload_jobs["jobresume"]["status"] == "queued"
    assert await queue.get() == "jobresume"


async def test_detached_upload_worker_claims_and_completes_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _upload_cfg.worker_mode = "external"
    job = {
        "job_id": "jobexternal",
        "filename": "room.png",
        "content_type": "image/png",
        "status": "queued",
        "stage": "upload",
        "progress": 0.0,
        "objects": None,
        "risks": None,
        "fix_first": None,
        "summary": None,
        "recommendations": None,
        "evidence_frames": None,
        "scan_quality": None,
        "trust_notes": None,
        "scene_source": None,
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": None,
        "error": None,
        "attempt_count": 0,
    }
    _upload_jobs["jobexternal"] = job
    _upload_store.create_job(job)
    _upload_store.save_job_input("jobexternal", "room.png", b"image")

    async def fake_analyze(*_args, **_kwargs):
        return SimpleNamespace(
            objects=[],
            risks=[],
            point_cloud=[],
            fix_first=[],
            summary={"filename": "room.png", "hazard_count": 0, "object_count": 0},
            recommendations=[],
            evidence_frames=[],
            evidence_artifacts={},
            scan_quality={"status": "good", "score": 0.8, "usable": True, "warnings": []},
            trust_notes=[],
            scene_source="estimated_multiview",
        )

    async def fake_ingest(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server_mod, "analyze_uploaded_image", fake_analyze)
    monkeypatch.setattr(server_mod, "_build_pdf_report", lambda _job: b"%PDF-1.4\nfixture\n")
    monkeypatch.setattr(
        server_mod,
        "_get_agent",
        lambda: SimpleNamespace(ingest_from_upload=fake_ingest),
    )

    processed = await server_mod.run_detached_upload_worker(
        worker_id="worker-test",
        once=True,
    )

    assert processed is True
    assert _upload_jobs["jobexternal"]["status"] == "complete"
    assert _upload_store.load_job_claim("jobexternal") is None


def test_operator_settings_reports_active_detached_workers() -> None:
    _api_cfg.access_token = "secret-token"
    _upload_cfg.worker_mode = "external"
    _upload_store.save_worker_record(
        "worker-a",
        {
            "worker_id": "worker-a",
            "state": "idle",
            "heartbeat_unix": time.time(),
            "heartbeat_at": "2026-04-22T00:00:00+00:00",
        },
    )

    response = client.get(
        "/operator/settings",
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["queue"]["worker_count"] == 1
    assert data["storage"]["worker_records"] == 1
    assert data["system"]["active_workers"] == 1


async def test_process_upload_retries_then_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    job = {
        "job_id": "jobproc1",
        "filename": "room.png",
        "content_type": "image/png",
        "audience_mode": "pet",
        "status": "queued",
        "stage": "upload",
        "progress": 0.1,
        "objects": None,
        "risks": None,
        "fix_first": None,
        "summary": None,
        "recommendations": None,
        "evidence_frames": None,
        "scan_quality": None,
        "trust_notes": None,
        "scene_source": None,
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": None,
        "error": None,
        "attempt_count": 0,
    }
    _upload_jobs["jobproc1"] = job
    _upload_store.create_job(job)
    _upload_store.save_job_input("jobproc1", "room.png", b"image")
    _upload_cfg.max_job_attempts = 2

    calls = {"count": 0}

    async def fake_analyze(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary failure")
        assert _kwargs["audience_mode"] == "pet"
        return SimpleNamespace(
            objects=[],
            risks=[],
            point_cloud=[],
            fix_first=[],
            summary={"filename": "room.png", "hazard_count": 0, "object_count": 0},
            recommendations=[],
            evidence_frames=[],
            evidence_artifacts={},
            scan_quality={"status": "good", "score": 0.8, "usable": True, "warnings": []},
            trust_notes=[],
            scene_source="estimated_multiview",
        )

    async def fake_ingest(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server_mod, "analyze_uploaded_image", fake_analyze)
    monkeypatch.setattr(server_mod, "_build_pdf_report", lambda _job: b"%PDF-1.4\nfixture\n")
    monkeypatch.setattr(
        server_mod,
        "_get_agent",
        lambda: SimpleNamespace(ingest_from_upload=fake_ingest),
    )

    await server_mod._process_upload("jobproc1")

    assert _upload_jobs["jobproc1"]["status"] == "queued"
    assert _upload_jobs["jobproc1"]["attempt_count"] == 1

    await server_mod._process_upload("jobproc1")

    assert _upload_jobs["jobproc1"]["status"] == "complete"
    assert _upload_jobs["jobproc1"]["attempt_count"] == 2
    assert _upload_store.has_job_input("jobproc1") is False


async def test_process_upload_persists_finding_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    job = {
        "job_id": "jobreplayproc",
        "filename": "room.png",
        "content_type": "image/png",
        "status": "queued",
        "stage": "upload",
        "progress": 0.0,
        "objects": None,
        "risks": None,
        "point_cloud": None,
        "fix_first": None,
        "summary": None,
        "recommendations": None,
        "evidence_frames": None,
        "scan_quality": None,
        "trust_notes": None,
        "scene_source": None,
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": None,
        "error": None,
        "attempt_count": 0,
    }
    _upload_jobs["jobreplayproc"] = job
    _upload_store.create_job(job)
    _upload_store.save_job_input("jobreplayproc", "room.png", b"image")

    replay_gif = b"GIF89a-generated"

    async def fake_analyze(*_args, **_kwargs):
        return SimpleNamespace(
            objects=[],
            risks=[
                {
                    "object_id": "track-01",
                    "hazard_code": "edge_placement",
                    "hazard_title": "Object placed near an edge",
                    "risk_score": 0.77,
                    "priority_score": 0.82,
                    "location_label": "front shelf",
                    "reasoning": {"evidence_ids": ["f00-r01"]},
                }
            ],
            point_cloud=[],
            fix_first=[],
            summary={"filename": "room.png", "hazard_count": 1, "object_count": 1},
            recommendations=[],
            evidence_frames=[
                {
                    "evidence_id": "f00-r01",
                    "caption": "Shelf crop",
                    "confidence": 0.81,
                    "image_url": None,
                }
            ],
            evidence_artifacts={"f00-r01": b"jpeg-evidence"},
            scan_quality={"status": "good", "score": 0.8, "usable": True, "warnings": []},
            trust_notes=[],
            scene_source="estimated_multiview",
        )

    async def fake_ingest(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server_mod, "analyze_uploaded_image", fake_analyze)
    monkeypatch.setattr(server_mod, "_build_pdf_report", lambda _job: b"%PDF-1.4\nfixture\n")
    monkeypatch.setattr(
        server_mod,
        "build_finding_replays",
        lambda *_args, **_kwargs: (
            [
                {
                    "replay_id": "finding-01",
                    "hazard_code": "edge_placement",
                    "object_id": "track-01",
                    "caption": "Edge placement replay",
                    "frame_count": 1,
                    "media_type": "image/gif",
                    "image_url": None,
                }
            ],
            {"finding-01": replay_gif},
        ),
    )
    monkeypatch.setattr(
        server_mod,
        "_get_agent",
        lambda: SimpleNamespace(ingest_from_upload=fake_ingest),
    )

    await server_mod._process_upload("jobreplayproc")

    completed = _upload_jobs["jobreplayproc"]
    assert completed["status"] == "complete"
    assert completed["artifacts"]["finding_replays"]["finding-01"]["storage_key"] == (
        "jobs/jobreplayproc/replays/finding-01.gif"
    )
    assert completed["risks"][0]["replay"]["image_url"] == "/jobs/jobreplayproc/replays/finding-01"
    assert _upload_store.load_replay_gif("jobreplayproc", "finding-01") == (replay_gif, "image/gif")
