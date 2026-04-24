"""Check whether the current Atlas-0 runtime profile is ready to host."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_ROOT = _REPO_ROOT / "python"
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from atlas.api import server  # noqa: E402


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(strict_warnings=bool(args.strict_warnings))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
