"""Disk-backed persistence for upload jobs and report artifacts."""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)
_ARTIFACT_BACKEND = "local_fs"


class UploadStore:
    """Persist upload jobs and generated report artifacts to disk."""

    def __init__(
        self,
        root_dir: Path,
        *,
        save_original_uploads: bool = False,
        max_persisted_jobs: int = 200,
        retention_days: int = 14,
        max_storage_bytes: int = 1_500_000_000,
    ) -> None:
        self.root_dir = root_dir
        self._save_original_uploads = save_original_uploads
        self._max_persisted_jobs = max_persisted_jobs
        self._retention_days = retention_days
        self._max_storage_bytes = max_storage_bytes
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def meta_dir(self) -> Path:
        """Return the directory holding non-job metadata files."""
        path = self.root_dir / "_meta"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def job_dir(self, job_id: str) -> Path:
        """Return the directory holding all files for one upload job."""
        return self.root_dir / job_id

    def manifest_path(self, job_id: str) -> Path:
        """Return the JSON manifest path for one upload job."""
        return self.job_dir(job_id) / "job.json"

    def report_path(self, job_id: str) -> Path:
        """Return the persisted PDF path for one upload job."""
        return self.job_dir(job_id) / "report.pdf"

    def upload_path(self, job_id: str, suffix: str = ".bin") -> Path:
        """Return the persisted original-upload path for one job."""
        return self.job_dir(job_id) / f"upload{suffix}"

    def queued_input_path(self, job_id: str, suffix: str = ".bin") -> Path:
        """Return the persisted worker-input path for one upload job."""
        return self.job_dir(job_id) / f"queued-input{suffix}"

    def evidence_dir(self, job_id: str) -> Path:
        """Return the directory holding evidence crops for one job."""
        return self.job_dir(job_id) / "evidence"

    def replay_dir(self, job_id: str) -> Path:
        """Return the directory holding finding replays for one job."""
        return self.job_dir(job_id) / "replays"

    def evidence_path(self, job_id: str, evidence_id: str, suffix: str = ".jpg") -> Path:
        """Return the persisted path for one evidence artifact."""
        return self.evidence_dir(job_id) / f"{evidence_id}{suffix}"

    def replay_path(self, job_id: str, replay_id: str, suffix: str = ".gif") -> Path:
        """Return the persisted path for one finding replay artifact."""
        return self.replay_dir(job_id) / f"{replay_id}{suffix}"

    def product_events_path(self) -> Path:
        """Return the JSONL file holding product analytics events."""
        return self.meta_dir() / "product_events.jsonl"

    def waitlist_path(self) -> Path:
        """Return the JSONL file holding waitlist submissions."""
        return self.meta_dir() / "waitlist.jsonl"

    def eval_candidates_dir(self) -> Path:
        """Return the directory holding exported eval candidates."""
        path = self.meta_dir() / "eval_candidates"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def eval_candidate_path(self, candidate_id: str) -> Path:
        """Return the persisted JSON path for one eval candidate."""
        return self.eval_candidates_dir() / f"{candidate_id}.json"

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

    def append_product_event(self, event: dict[str, Any]) -> None:
        """Append one product analytics event to the JSONL log."""
        self._append_json_line(self.product_events_path(), event)

    def load_product_events(self) -> list[dict[str, Any]]:
        """Load all persisted product analytics events."""
        return self._load_json_lines(self.product_events_path())

    def append_waitlist_entry(self, entry: dict[str, Any]) -> None:
        """Append one waitlist signup entry to the JSONL log."""
        self._append_json_line(self.waitlist_path(), entry)

    def load_waitlist_entries(self) -> list[dict[str, Any]]:
        """Load all persisted waitlist submissions."""
        return self._load_json_lines(self.waitlist_path())

    def save_eval_candidate(self, candidate_id: str, payload: dict[str, Any]) -> Path:
        """Persist one exported evaluation candidate payload."""
        path = self.eval_candidate_path(candidate_id)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
        return path

    def load_eval_candidates(self) -> list[dict[str, Any]]:
        """Load all saved evaluation candidate payloads."""
        entries: list[dict[str, Any]] = []
        for path in sorted(self.eval_candidates_dir().glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "upload_store_eval_candidate_skipped",
                    path=str(path),
                    error=str(exc),
                )
                continue
            if isinstance(payload, dict):
                entries.append(payload)
        return entries

    def save_original_upload(self, job_id: str, filename: str, content: bytes) -> Path | None:
        """Persist the original upload bytes when enabled."""
        if not self._save_original_uploads:
            return None

        suffix = Path(filename).suffix or ".bin"
        upload_path = self.upload_path(job_id, suffix=suffix)
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        upload_path.write_bytes(content)
        return upload_path

    def save_job_input(self, job_id: str, filename: str, content: bytes) -> Path:
        """Persist the queued worker input for a job regardless of retention policy."""
        suffix = Path(filename).suffix or ".bin"
        input_path = self.queued_input_path(job_id, suffix=suffix)
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_bytes(content)
        return input_path

    def load_job_input(self, job_id: str) -> bytes | None:
        """Load queued worker input bytes for one job, if still present."""
        job_dir = self.job_dir(job_id)
        for path in sorted(job_dir.glob("queued-input.*")):
            if path.is_file():
                return path.read_bytes()
        return None

    def has_job_input(self, job_id: str) -> bool:
        """Return True when queued worker input exists for one job."""
        job_dir = self.job_dir(job_id)
        return any(path.is_file() for path in job_dir.glob("queued-input.*"))

    def remove_job_input(self, job_id: str) -> None:
        """Delete queued worker input after the job reaches a terminal state."""
        job_dir = self.job_dir(job_id)
        for path in job_dir.glob("queued-input.*"):
            if path.is_file():
                path.unlink(missing_ok=True)

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

    def save_replay_gif(self, job_id: str, replay_id: str, content: bytes) -> Path:
        """Persist one replay GIF for a job."""
        replay_path = self.replay_path(job_id, replay_id, suffix=".gif")
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_bytes(content)
        return replay_path

    def load_replay_gif(self, job_id: str, replay_id: str) -> tuple[bytes, str] | None:
        """Load one persisted finding replay and its content type."""
        replay_path = self.replay_path(job_id, replay_id, suffix=".gif")
        if replay_path.exists():
            return (replay_path.read_bytes(), "image/gif")
        return None

    def delete_job(self, job_id: str) -> bool:
        """Delete all persisted artifacts for a job."""
        job_dir = self.job_dir(job_id)
        if not job_dir.exists():
            return False
        self._delete_tree(job_dir)
        return True

    def artifact_pointer(
        self,
        job_id: str,
        relative_path: str | Path,
        *,
        kind: str,
        media_type: str | None = None,
        url: str | None = None,
    ) -> dict[str, Any]:
        """Build one artifact pointer using storage-key semantics."""
        rel_path = Path(relative_path)
        full_path = self.job_dir(job_id) / rel_path
        size_bytes = full_path.stat().st_size if full_path.exists() else 0
        return {
            "kind": kind,
            "storage_backend": _ARTIFACT_BACKEND,
            "storage_key": f"jobs/{job_id}/{rel_path.as_posix()}",
            "relative_path": rel_path.as_posix(),
            "media_type": media_type,
            "size_bytes": size_bytes,
            "url": url,
        }

    def storage_summary(self) -> dict[str, int]:
        """Return a coarse storage summary for operator diagnostics."""
        persisted_jobs = 0
        bytes_used = 0
        queued_inputs = 0
        reports = 0
        evidence_files = 0
        replay_files = 0
        original_uploads = 0

        for job_dir in self._iter_job_dirs():
            if not job_dir.is_dir():
                continue
            persisted_jobs += 1
            for path in job_dir.rglob("*"):
                if not path.is_file():
                    continue
                bytes_used += path.stat().st_size
                name = path.name
                if name.startswith("queued-input."):
                    queued_inputs += 1
                elif name.startswith("upload."):
                    original_uploads += 1
                elif name == "report.pdf":
                    reports += 1
                elif path.parent.name == "evidence":
                    evidence_files += 1
                elif path.parent.name == "replays":
                    replay_files += 1

        meta_bytes = sum(
            path.stat().st_size for path in self.meta_dir().rglob("*") if path.is_file()
        )
        return {
            "persisted_jobs": persisted_jobs,
            "bytes_used": bytes_used + meta_bytes,
            "byte_budget": self._max_storage_bytes,
            "usage_percent": math.floor(((bytes_used + meta_bytes) / self._max_storage_bytes) * 100)
            if self._max_storage_bytes > 0
            else 0,
            "queued_inputs": queued_inputs,
            "original_uploads": original_uploads,
            "reports": reports,
            "evidence_files": evidence_files,
            "replay_files": replay_files,
            "meta_bytes": meta_bytes,
            "waitlist_entries": len(self.load_waitlist_entries()),
            "eval_candidates": len(self.load_eval_candidates()),
        }

    def _prune_old_jobs(self) -> None:
        """Best-effort pruning to keep persisted job storage bounded."""
        job_dirs = sorted(
            self._iter_job_dirs(),
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
                self._iter_job_dirs(),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )

        for path in job_dirs[self._max_persisted_jobs :]:
            try:
                self._delete_tree(path)
            except OSError as exc:
                logger.warning("upload_store_prune_failed", path=str(path), error=str(exc))

        if self._max_storage_bytes > 0:
            job_dirs = sorted(
                self._iter_job_dirs(),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            total_bytes = sum(self._job_size_bytes(path) for path in job_dirs)
            for path in reversed(job_dirs):
                if total_bytes <= self._max_storage_bytes:
                    break
                try:
                    size_bytes = self._job_size_bytes(path)
                    self._delete_tree(path)
                    total_bytes -= size_bytes
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

    def _job_size_bytes(self, job_dir: Path) -> int:
        """Return the total byte size for one job directory."""
        return sum(path.stat().st_size for path in job_dir.rglob("*") if path.is_file())

    def _iter_job_dirs(self) -> list[Path]:
        """Return only real persisted job directories under the storage root."""
        return [
            path
            for path in self.root_dir.iterdir()
            if path.is_dir() and (path / "job.json").exists()
        ]

    def _append_json_line(self, path: Path, payload: dict[str, Any]) -> None:
        """Append one JSON line to a metadata log file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")

    def _load_json_lines(self, path: Path) -> list[dict[str, Any]]:
        """Load newline-delimited JSON metadata entries from disk."""
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "upload_store_jsonl_entry_skipped",
                            path=str(path),
                            error=str(exc),
                        )
                        continue
                    if isinstance(payload, dict):
                        rows.append(payload)
        except OSError as exc:
            logger.warning("upload_store_jsonl_load_failed", path=str(path), error=str(exc))
        return rows
