"""Tests for disk-backed upload job persistence."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from atlas.api.upload_store import UploadStore


def test_upload_store_round_trips_job_manifest(tmp_path: Path) -> None:
    store = UploadStore(tmp_path)
    job = {
        "job_id": "job-001",
        "filename": "room.mp4",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "summary": {"hazard_count": 2},
        "report_url": "/reports/job-001.pdf",
    }

    store.create_job(job)
    loaded = store.load_jobs()

    assert loaded["job-001"]["filename"] == "room.mp4"
    assert loaded["job-001"]["summary"]["hazard_count"] == 2


def test_upload_store_saves_original_upload_when_enabled(tmp_path: Path) -> None:
    store = UploadStore(tmp_path, save_original_uploads=True)
    store.create_job({"job_id": "job-002"})

    saved = store.save_original_upload("job-002", "scan.mov", b"movie-bytes")

    assert saved is not None
    assert saved.name == "upload.mov"
    assert saved.read_bytes() == b"movie-bytes"


def test_upload_store_persists_report_pdf(tmp_path: Path) -> None:
    store = UploadStore(tmp_path)
    store.create_job({"job_id": "job-003"})
    pdf_bytes = b"%PDF-1.4\nfixture\n"

    store.save_report_pdf("job-003", pdf_bytes)

    assert store.load_report_pdf("job-003") == pdf_bytes


def test_upload_store_persists_report_pdf_in_object_store_fs(tmp_path: Path) -> None:
    store = UploadStore(
        tmp_path,
        artifact_backend="object_store_fs",
        artifact_object_dir=tmp_path / "objects",
    )
    store.create_job({"job_id": "job-003b"})

    saved = store.save_report_pdf("job-003b", b"%PDF-1.4\nfixture\n")

    assert store.load_report_pdf("job-003b") == b"%PDF-1.4\nfixture\n"
    assert "objects/jobs/job-003b/report.pdf" in str(saved)


def test_upload_store_manifest_is_json(tmp_path: Path) -> None:
    store = UploadStore(tmp_path)
    store.create_job({"job_id": "job-004", "status": "queued"})

    manifest = tmp_path / "job-004" / "job.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))

    assert data["job_id"] == "job-004"
    assert data["status"] == "queued"


def test_upload_store_persists_evidence_image(tmp_path: Path) -> None:
    store = UploadStore(tmp_path)
    store.create_job({"job_id": "job-005"})

    store.save_evidence_image("job-005", "evidence-01", b"jpeg-bytes")
    loaded = store.load_evidence_image("job-005", "evidence-01")

    assert loaded is not None
    assert loaded[0] == b"jpeg-bytes"
    assert loaded[1] == "image/jpeg"


def test_upload_store_delete_job_removes_artifacts(tmp_path: Path) -> None:
    store = UploadStore(tmp_path, save_original_uploads=True)
    store.create_job({"job_id": "job-006"})
    store.save_original_upload("job-006", "scan.mov", b"movie")
    store.save_report_pdf("job-006", b"pdf")
    store.save_evidence_image("job-006", "e-01", b"jpeg")

    assert store.delete_job("job-006") is True
    assert not (tmp_path / "job-006").exists()


def test_upload_store_delete_job_removes_object_store_artifacts(tmp_path: Path) -> None:
    store = UploadStore(
        tmp_path,
        artifact_backend="object_store_fs",
        artifact_object_dir=tmp_path / "objects",
        save_original_uploads=True,
    )
    store.create_job({"job_id": "job-006b"})
    store.save_original_upload("job-006b", "scan.mov", b"movie")
    store.save_report_pdf("job-006b", b"pdf")

    assert store.delete_job("job-006b") is True
    assert not (tmp_path / "job-006b").exists()
    assert not (tmp_path / "objects" / "jobs" / "job-006b").exists()


def test_upload_store_persists_and_removes_job_input(tmp_path: Path) -> None:
    store = UploadStore(tmp_path)
    store.create_job({"job_id": "job-007"})

    store.save_job_input("job-007", "scan.mp4", b"video-bytes")

    assert store.has_job_input("job-007") is True
    assert store.load_job_input("job-007") == b"video-bytes"

    store.remove_job_input("job-007")

    assert store.has_job_input("job-007") is False
    assert store.load_job_input("job-007") is None


def test_upload_store_persists_replay_gif(tmp_path: Path) -> None:
    store = UploadStore(tmp_path)
    store.create_job({"job_id": "job-007b"})

    store.save_replay_gif("job-007b", "finding-01", b"gif-bytes")
    loaded = store.load_replay_gif("job-007b", "finding-01")

    assert loaded is not None
    assert loaded[0] == b"gif-bytes"
    assert loaded[1] == "image/gif"


def test_upload_store_storage_summary_counts_files(tmp_path: Path) -> None:
    store = UploadStore(tmp_path, save_original_uploads=True)
    store.create_job({"job_id": "job-008"})
    store.save_job_input("job-008", "scan.mov", b"queue")
    store.save_original_upload("job-008", "scan.mov", b"original")
    store.save_report_pdf("job-008", b"pdf")
    store.save_evidence_image("job-008", "e-01", b"jpeg")
    store.save_replay_gif("job-008", "finding-01", b"gif")

    summary = store.storage_summary()

    assert summary["persisted_jobs"] == 1
    assert summary["byte_budget"] == 1_500_000_000
    assert summary["queued_inputs"] == 1
    assert summary["original_uploads"] == 1
    assert summary["reports"] == 1
    assert summary["evidence_files"] == 1
    assert summary["replay_files"] == 1
    assert summary["bytes_used"] >= (
        len(b"queue") + len(b"original") + len(b"pdf") + len(b"jpeg") + len(b"gif")
    )


def test_upload_store_artifact_pointer_uses_storage_keys(tmp_path: Path) -> None:
    store = UploadStore(tmp_path)
    store.create_job({"job_id": "job-009"})
    store.save_report_pdf("job-009", b"pdf")

    pointer = store.artifact_pointer(
        "job-009",
        "report.pdf",
        kind="report_pdf",
        media_type="application/pdf",
        url="/reports/job-009.pdf",
    )

    assert pointer["storage_backend"] == "local_fs"
    assert pointer["storage_key"] == "jobs/job-009/report.pdf"
    assert pointer["url"] == "/reports/job-009.pdf"


def test_upload_store_artifact_pointer_uses_object_store_size(tmp_path: Path) -> None:
    store = UploadStore(
        tmp_path,
        artifact_backend="object_store_fs",
        artifact_object_dir=tmp_path / "objects",
    )
    store.create_job({"job_id": "job-009b"})
    report_path = store.save_report_pdf("job-009b", b"pdf-bytes")

    pointer = store.artifact_pointer(
        "job-009b",
        store.job_relative_path("job-009b", report_path),
        kind="report_pdf",
        media_type="application/pdf",
    )

    assert pointer["storage_backend"] == "object_store_fs"
    assert pointer["size_bytes"] == len(b"pdf-bytes")


def test_upload_store_prunes_oldest_jobs_when_byte_budget_exceeded(tmp_path: Path) -> None:
    store = UploadStore(tmp_path, max_storage_bytes=500)
    store.create_job({"job_id": "job-010"})
    store.save_report_pdf("job-010", b"1" * 220)
    os.utime(store.job_dir("job-010"), (1, 1))
    store.create_job({"job_id": "job-011"})
    store.save_report_pdf("job-011", b"2" * 220)

    store.create_job({"job_id": "job-012"})

    assert store.job_dir("job-010").exists() is False
    assert store.job_dir("job-011").exists() is True


def test_upload_store_claims_next_queued_job(tmp_path: Path) -> None:
    store = UploadStore(tmp_path)
    store.create_job({"job_id": "job-013", "status": "queued"})
    store.save_job_input("job-013", "scan.mp4", b"video")

    claimed = store.claim_next_job("worker-a", lease_seconds=60.0)

    assert claimed is not None
    assert claimed[0] == "job-013"
    assert store.load_job_claim("job-013")["worker_id"] == "worker-a"


def test_upload_store_reclaims_stale_job_claim(tmp_path: Path) -> None:
    store = UploadStore(tmp_path)
    store.create_job({"job_id": "job-014", "status": "queued"})
    store.save_job_input("job-014", "scan.mp4", b"video")
    assert store.claim_next_job("worker-a", lease_seconds=0.01) is not None

    import time

    time.sleep(0.02)
    claimed = store.claim_next_job("worker-b", lease_seconds=60.0)

    assert claimed is not None
    assert claimed[0] == "job-014"
    assert store.load_job_claim("job-014")["worker_id"] == "worker-b"


def test_upload_store_tracks_active_worker_records(tmp_path: Path) -> None:
    store = UploadStore(tmp_path)
    store.save_worker_record(
        "worker-a",
        {
            "worker_id": "worker-a",
            "state": "idle",
            "heartbeat_unix": time.time(),
        },
    )

    active = store.active_worker_records(stale_after_seconds=30.0)

    assert len(active) == 1
    assert active[0]["worker_id"] == "worker-a"
