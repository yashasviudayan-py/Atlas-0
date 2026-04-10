"""Tests for disk-backed upload job persistence."""

from __future__ import annotations

import json
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


def test_upload_store_manifest_is_json(tmp_path: Path) -> None:
    store = UploadStore(tmp_path)
    store.create_job({"job_id": "job-004", "status": "queued"})

    manifest = tmp_path / "job-004" / "job.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))

    assert data["job_id"] == "job-004"
    assert data["status"] == "queued"
