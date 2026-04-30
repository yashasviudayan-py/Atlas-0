"""Check whether the current Atlas-0 runtime profile is ready to host."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_ROOT = _REPO_ROOT / "python"
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from atlas.api import server  # noqa: E402
from atlas.api.upload_store import UploadStore  # noqa: E402


def exercise_object_store_backend() -> dict[str, Any]:
    """Run a write/read/delete exercise against the object_store_fs backend."""
    with tempfile.TemporaryDirectory(prefix="atlas0-object-store-") as tmp:
        root = Path(tmp) / "jobs"
        object_root = Path(tmp) / "objects"
        store = UploadStore(
            root,
            artifact_backend="object_store_fs",
            artifact_object_dir=object_root,
            artifact_base_url="https://example.invalid/atlas-artifacts",
        )
        job = {
            "job_id": "preflight-job",
            "filename": "preflight.mp4",
            "status": "complete",
            "progress": 1.0,
            "stage": "complete",
        }
        store.create_job(job)
        report_path = store.save_report_pdf("preflight-job", b"%PDF-1.4\npreflight\n")
        evidence_path = store.save_evidence_image("preflight-job", "f00-r01", b"jpeg-bytes")
        report_pointer = store.artifact_pointer(
            "preflight-job",
            store.job_relative_path("preflight-job", report_path),
            kind="report_pdf",
            media_type="application/pdf",
            url="/reports/preflight-job.pdf",
        )

        loaded_report = store.load_report_pdf("preflight-job")
        loaded_evidence = store.load_evidence_image("preflight-job", "f00-r01")
        deleted = store.delete_job("preflight-job")

        checks = [
            {
                "name": "object_report_roundtrip",
                "status": "pass" if loaded_report == b"%PDF-1.4\npreflight\n" else "fail",
                "detail": "Report PDF persisted and loaded through object_store_fs.",
            },
            {
                "name": "object_evidence_roundtrip",
                "status": "pass" if loaded_evidence == (b"jpeg-bytes", "image/jpeg") else "fail",
                "detail": "Evidence image persisted and loaded through object_store_fs.",
            },
            {
                "name": "object_pointer",
                "status": "pass"
                if report_pointer["storage_backend"] == "object_store_fs"
                and report_pointer["storage_key"] == "jobs/preflight-job/report.pdf"
                else "fail",
                "detail": f"Artifact pointer URL: {report_pointer['url']}",
            },
            {
                "name": "object_delete",
                "status": "pass"
                if deleted and not report_path.exists() and not evidence_path.exists()
                else "fail",
                "detail": "Job manifest and object artifacts were deleted together.",
            },
        ]
        return {
            "ready": all(check["status"] == "pass" for check in checks),
            "checks": checks,
            "object_root": str(object_root),
        }


def build_report(*, strict_warnings: bool = False) -> dict[str, Any]:
    """Return deployment checks plus the issues that should block hosting."""
    summary = server._run_startup_checks()
    checks = list(summary.get("checks", []))
    blockers = [
        check
        for check in checks
        if check.get("status") == "fail"
        or (strict_warnings and check.get("status") == "warn")
    ]
    return {
        "ready": bool(summary.get("ready")) and not blockers,
        "strict_warnings": strict_warnings,
        "summary": summary.get("summary", ""),
        "checked_at": summary.get("checked_at"),
        "checks": checks,
        "blockers": blockers,
    }


def _print_text(report: dict[str, Any]) -> None:
    status = "READY" if report["ready"] else "NOT READY"
    print(f"Atlas-0 deployment preflight: {status}")
    print(report.get("summary") or "No summary available.")
    for check in report["checks"]:
        name = str(check.get("name", "check"))
        check_status = str(check.get("status", "unknown")).upper()
        detail = str(check.get("detail", ""))
        print(f"- {check_status}: {name} - {detail}")
    if "object_store_exercise" in report:
        exercise = report["object_store_exercise"]
        exercise_status = "READY" if exercise.get("ready") else "NOT READY"
        print(f"Object-store exercise: {exercise_status}")
        for check in exercise.get("checks", []):
            print(
                f"- {str(check.get('status', 'unknown')).upper()}: "
                f"{check.get('name', 'check')} - {check.get('detail', '')}"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text.",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Treat warning checks as blocking issues.",
    )
    parser.add_argument(
        "--exercise-object-store",
        action="store_true",
        help="Run a temporary object_store_fs artifact roundtrip exercise.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(strict_warnings=bool(args.strict_warnings))
    if args.exercise_object_store:
        report["object_store_exercise"] = exercise_object_store_backend()
        report["ready"] = bool(report["ready"]) and bool(report["object_store_exercise"]["ready"])
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
