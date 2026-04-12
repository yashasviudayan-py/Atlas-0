"""Disk-backed persistence for upload jobs and report artifacts."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class UploadStore:
    """Persist upload jobs and generated report artifacts to disk."""

    def __init__(
        self,
        root_dir: Path,
        *,
        save_original_uploads: bool = False,
        max_persisted_jobs: int = 200,
        retention_days: int = 14,
    ) -> None:
        self.root_dir = root_dir
        self._save_original_uploads = save_original_uploads
        self._max_persisted_jobs = max_persisted_jobs
        self._retention_days = retention_days
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        """Return the directory holding all files for one upload job."""
        return self.root_dir / job_id

    def manifest_path(self, job_id: str) -> Path:
        """Return the JSON manifest path for one upload job."""
        return self.job_dir(job_id) / "job.json"

    def report_path(self, job_id: str) -> Path:
        """Return the persisted PDF path for one upload job."""
        return self.job_dir(job_id) / "report.pdf"

    def evidence_dir(self, job_id: str) -> Path:
        """Return the directory holding evidence crops for one job."""
        return self.job_dir(job_id) / "evidence"

    def evidence_path(self, job_id: str, evidence_id: str, suffix: str = ".jpg") -> Path:
        """Return the persisted path for one evidence artifact."""
        return self.evidence_dir(job_id) / f"{evidence_id}{suffix}"

    def create_job(self, job: dict[str, Any]) -> None:
        """Create the job directory and write its initial manifest."""
        self.job_dir(str(job["job_id"])).mkdir(parents=True, exist_ok=True)
        self.save_job(job)
        self._prune_old_jobs()

    def save_job(self, job: dict[str, Any]) -> None:
        """Persist the current job manifest atomically."""
        job_id = str(job["job_id"])
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        serializable = deepcopy(job)
        tmp_path = self.manifest_path(job_id).with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(serializable, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.manifest_path(job_id))

    def load_jobs(self) -> dict[str, dict[str, Any]]:
        """Load all persisted jobs from disk."""
        jobs: dict[str, dict[str, Any]] = {}
        for manifest in sorted(self.root_dir.glob("*/job.json")):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("upload_store_manifest_skipped", path=str(manifest), error=str(exc))
                continue

            job_id = str(data.get("job_id", manifest.parent.name))
            jobs[job_id] = data
        return jobs

    def save_original_upload(self, job_id: str, filename: str, content: bytes) -> Path | None:
        """Persist the original upload bytes when enabled."""
        if not self._save_original_uploads:
            return None

        suffix = Path(filename).suffix or ".bin"
        upload_path = self.job_dir(job_id) / f"upload{suffix}"
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        upload_path.write_bytes(content)
        return upload_path

    def save_report_pdf(self, job_id: str, pdf_bytes: bytes) -> Path:
        """Persist a generated PDF report for a job."""
        report_path = self.report_path(job_id)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(pdf_bytes)
        return report_path

    def load_report_pdf(self, job_id: str) -> bytes | None:
        """Return the persisted PDF bytes for a job, if present."""
        report_path = self.report_path(job_id)
        if not report_path.exists():
            return None
        return report_path.read_bytes()

    def save_evidence_image(
        self,
        job_id: str,
        evidence_id: str,
        content: bytes,
        *,
        suffix: str = ".jpg",
    ) -> Path:
        """Persist one evidence image for a job."""
        evidence_path = self.evidence_path(job_id, evidence_id, suffix=suffix)
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_bytes(content)
        return evidence_path

    def load_evidence_image(self, job_id: str, evidence_id: str) -> tuple[bytes, str] | None:
        """Load one persisted evidence image and its content type."""
        for suffix, media_type in (
            (".jpg", "image/jpeg"),
            (".jpeg", "image/jpeg"),
            (".png", "image/png"),
        ):
            evidence_path = self.evidence_path(job_id, evidence_id, suffix=suffix)
            if evidence_path.exists():
                return (evidence_path.read_bytes(), media_type)
        return None

    def delete_job(self, job_id: str) -> bool:
        """Delete all persisted artifacts for a job."""
        job_dir = self.job_dir(job_id)
        if not job_dir.exists():
            return False
        self._delete_tree(job_dir)
        return True

    def _prune_old_jobs(self) -> None:
        """Best-effort pruning to keep persisted job storage bounded."""
        job_dirs = sorted(
            (path for path in self.root_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if self._retention_days > 0:
            cutoff = time.time() - self._retention_days * 86_400
            for path in job_dirs:
                try:
                    if path.stat().st_mtime < cutoff:
                        self._delete_tree(path)
                except OSError as exc:
                    logger.warning("upload_store_prune_failed", path=str(path), error=str(exc))

            job_dirs = sorted(
                (path for path in self.root_dir.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )

        for path in job_dirs[self._max_persisted_jobs :]:
            try:
                self._delete_tree(path)
            except OSError as exc:
                logger.warning("upload_store_prune_failed", path=str(path), error=str(exc))

    def _delete_tree(self, path: Path) -> None:
        """Delete a directory tree without requiring shutil."""
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                child.rmdir()
        path.rmdir()
