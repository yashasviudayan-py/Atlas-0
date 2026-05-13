"""Generate a small repo-native SBOM for release artifacts.

The output is CycloneDX-inspired JSON with the dependency coordinates Atlas-0
can derive locally from Cargo, Python, npm, and GitHub Actions manifests. It is
not a replacement for a full scanner, but gives release attestations a stable
software inventory without adding another runtime dependency.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _component(name: str, version: str, ecosystem: str) -> dict[str, str]:
    return {
        "type": "library",
        "name": name,
        "version": version,
        "purl": f"pkg:{ecosystem}/{name}@{version}",
    }


def _python_components() -> list[dict[str, str]]:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    specs = list(pyproject.get("project", {}).get("dependencies", []))
    for values in pyproject.get("project", {}).get("optional-dependencies", {}).values():
        specs.extend(values)
    components = []
    for spec in specs:
        match = re.match(r"^([A-Za-z0-9_.-]+)(?:\[.*\])?==(.+)$", str(spec))
        if match:
            components.append(_component(match.group(1), match.group(2), "pypi"))
    return components


def _npm_components() -> list[dict[str, str]]:
    package = json.loads((_REPO_ROOT / "package.json").read_text())
    components = []
    for deps_key in ("dependencies", "devDependencies"):
        for name, version in package.get(deps_key, {}).items():
            components.append(_component(name, str(version).lstrip("^~"), "npm"))
    return components


def _cargo_components() -> list[dict[str, str]]:
    lock_path = _REPO_ROOT / "Cargo.lock"
    if not lock_path.exists():
        return []
    lock = lock_path.read_text()
    components = []
    for block in lock.split("[[package]]"):
        name_match = re.search(r'^name = "([^"]+)"$', block, flags=re.MULTILINE)
        version_match = re.search(r'^version = "([^"]+)"$', block, flags=re.MULTILINE)
        source_match = re.search(r'^source = "([^"]+)"$', block, flags=re.MULTILINE)
        if name_match and version_match and source_match:
            components.append(_component(name_match.group(1), version_match.group(1), "cargo"))
    return components


def _github_action_components() -> list[dict[str, str]]:
    components = []
    for workflow in (_REPO_ROOT / ".github" / "workflows").glob("*.yml"):
        for action, version in re.findall(
            r"uses:\s*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@([^\s]+)", workflow.read_text()
        ):
            components.append(_component(action, version, "githubactions"))
    return components


def build_sbom() -> dict[str, Any]:
    """Return a deterministic dependency inventory for release attestation."""
    components = (
        _python_components() + _npm_components() + _cargo_components() + _github_action_components()
    )
    unique = {
        (component["purl"], component["name"], component["version"]): component
        for component in components
    }
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "component": {"type": "application", "name": "atlas-0", "version": "0.1.0"},
        },
        "components": sorted(unique.values(), key=lambda item: item["purl"]),
    }


def main(argv: list[str] | None = None) -> int:
    output = Path(argv[0]) if argv else _REPO_ROOT / "dist" / "atlas0-sbom.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_sbom(), indent=2, sort_keys=True) + "\n")
    print(f"Wrote SBOM to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
