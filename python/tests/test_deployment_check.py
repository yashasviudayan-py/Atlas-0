"""Tests for the deployment preflight helper script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_deployment.py"
_SPEC = importlib.util.spec_from_file_location("check_deployment", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
check_deployment = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_deployment)


def test_build_report_exposes_startup_checks() -> None:
    report = check_deployment.build_report()

    assert report["ready"] is True
    assert report["blockers"] == []
    assert any(check["name"] == "storage_root" for check in report["checks"])


def test_main_returns_success_for_default_profile(capsys) -> None:
    exit_code = check_deployment.main([])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Atlas-0 deployment preflight: READY" in output
