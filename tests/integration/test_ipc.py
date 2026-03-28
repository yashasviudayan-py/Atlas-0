"""Integration tests for the Rust↔Python shared-memory IPC bridge.

Tests cover:
- Round-trip data integrity (Python writer → Python reader, matching the Rust
  binary layout exactly).
- Cross-language compatibility: a Rust example binary writes a snapshot and
  Python reads it (requires ``cargo`` to be available).
- IPC read latency benchmark (target < 1 ms per snapshot).
- Protobuf message encode/decode round-trips for structured IPC types.
"""

import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

# Root of the workspace (two levels up from tests/integration/).
WORKSPACE_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "python"))

from atlas.utils.shared_mem import (  # noqa: E402
    ATLAS_MMAP_MAGIC,
    ATLAS_MMAP_VERSION,
    BYTES_PER_GAUSSIAN,
    HEADER_SIZE,
    CameraPose,
    MapSnapshot,
    SharedMemReader,
    SharedMemWriter,
    measure_snapshot_latency,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_mmap(tmp_path: Path) -> Path:
    """Return a temporary path for a .mmap file."""
    return tmp_path / "atlas_test.mmap"


# ── Data-integrity tests ──────────────────────────────────────────────────────


def test_empty_snapshot_round_trip(tmp_mmap: Path) -> None:
    """Writer → reader round-trip with zero Gaussians."""
    max_g = 1000
    pose = CameraPose(tx=0.0, ty=0.0, tz=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

    with SharedMemWriter(tmp_mmap, max_g) as writer:
        empty = np.empty(0, dtype=SharedMemWriter.GAUSSIAN_DTYPE)
        writer.write_snapshot(empty, pose=pose, frame_id=42, timestamp_ns=100_000)

    with SharedMemReader(tmp_mmap, max_g) as reader:
        snap = reader.get_latest_snapshot()

    assert snap.frame_id == 42
    assert snap.timestamp_ns == 100_000
    assert len(snap.gaussians) == 0


def test_single_gaussian_round_trip(tmp_mmap: Path) -> None:
    """Writer → reader preserves position, opacity, and colour."""
    max_g = 100
    pose = CameraPose(tx=1.0, ty=2.0, tz=3.0, qw=0.707, qx=0.0, qy=0.707, qz=0.0)

    gaussians = np.array(
        [(0.5, -1.0, 2.0, 0.8, 1.0, 0.0, 0.0)],
        dtype=SharedMemWriter.GAUSSIAN_DTYPE,
    )

    with SharedMemWriter(tmp_mmap, max_g) as writer:
        writer.write_snapshot(gaussians, pose=pose, frame_id=7, timestamp_ns=999)

    with SharedMemReader(tmp_mmap, max_g) as reader:
        snap = reader.get_latest_snapshot()

    assert snap.frame_id == 7
    assert snap.timestamp_ns == 999
    assert len(snap.gaussians) == 1

    g = snap.gaussians[0]
    assert abs(g["x"] - 0.5) < 1e-5
    assert abs(g["y"] - (-1.0)) < 1e-5
    assert abs(g["z"] - 2.0) < 1e-5
    assert abs(g["opacity"] - 0.8) < 1e-5
    assert abs(g["r"] - 1.0) < 1e-5
    assert abs(g["g"] - 0.0) < 1e-5
    assert abs(g["b"] - 0.0) < 1e-5

    assert abs(snap.pose.tx - 1.0) < 1e-5
    assert abs(snap.pose.ty - 2.0) < 1e-5
    assert abs(snap.pose.tz - 3.0) < 1e-5
    assert abs(snap.pose.qw - 0.707) < 1e-4
    assert abs(snap.pose.qy - 0.707) < 1e-4


def test_large_gaussian_batch_round_trip(tmp_mmap: Path) -> None:
    """Round-trip for a large batch of Gaussians."""
    n = 500_000
    max_g = n

    rng = np.random.default_rng(42)
    gaussians = np.empty(n, dtype=SharedMemWriter.GAUSSIAN_DTYPE)
    for col in ("x", "y", "z"):
        gaussians[col] = rng.uniform(-10.0, 10.0, n).astype(np.float32)
    gaussians["opacity"] = rng.uniform(0.0, 1.0, n).astype(np.float32)
    for col in ("r", "g", "b"):
        gaussians[col] = rng.uniform(0.0, 1.0, n).astype(np.float32)

    pose = CameraPose(tx=0.1, ty=0.2, tz=0.3, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

    with SharedMemWriter(tmp_mmap, max_g) as writer:
        writer.write_snapshot(gaussians, pose=pose, frame_id=1, timestamp_ns=0)

    with SharedMemReader(tmp_mmap, max_g) as reader:
        snap = reader.get_latest_snapshot()

    assert len(snap.gaussians) == n
    np.testing.assert_allclose(snap.gaussians["x"], gaussians["x"], rtol=1e-5)
    np.testing.assert_allclose(snap.gaussians["opacity"], gaussians["opacity"], rtol=1e-5)


def test_double_buffer_flip(tmp_mmap: Path) -> None:
    """Successive writes to the same file produce distinct frame_ids."""
    max_g = 10
    empty = np.empty(0, dtype=SharedMemWriter.GAUSSIAN_DTYPE)
    pose = CameraPose(tx=0.0, ty=0.0, tz=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

    with SharedMemWriter(tmp_mmap, max_g) as writer:
        with SharedMemReader(tmp_mmap, max_g) as reader:
            for fid in (1, 2, 3):
                writer.write_snapshot(empty, pose=pose, frame_id=fid, timestamp_ns=0)
                snap = reader.get_latest_snapshot()
                assert snap.frame_id == fid


def test_truncation_at_max_gaussians(tmp_mmap: Path) -> None:
    """Gaussians beyond max_gaussians are silently truncated."""
    max_g = 5
    gaussians = np.zeros(20, dtype=SharedMemWriter.GAUSSIAN_DTYPE)

    with SharedMemWriter(tmp_mmap, max_g) as writer:
        writer.write_snapshot(gaussians, pose=None, frame_id=1, timestamp_ns=0)

    with SharedMemReader(tmp_mmap, max_g) as reader:
        snap = reader.get_latest_snapshot()

    assert len(snap.gaussians) == max_g


def test_invalid_magic_raises(tmp_mmap: Path) -> None:
    """Opening a file with wrong magic raises ValueError."""
    max_g = 10
    # Write a correctly-sized file filled with zeros (magic = 0, not ATLAS_MMAP_MAGIC).
    file_size = HEADER_SIZE + max_g * BYTES_PER_GAUSSIAN * 2
    tmp_mmap.write_bytes(bytes(file_size))
    with pytest.raises(ValueError, match="magic"):
        SharedMemReader(tmp_mmap, max_gaussians=max_g)


def test_identity_pose_when_none(tmp_mmap: Path) -> None:
    """Passing pose=None uses the identity pose."""
    with SharedMemWriter(tmp_mmap, 10) as writer:
        writer.write_snapshot(np.empty(0, dtype=SharedMemWriter.GAUSSIAN_DTYPE), pose=None, frame_id=0, timestamp_ns=0)

    with SharedMemReader(tmp_mmap, 10) as reader:
        snap = reader.get_latest_snapshot()

    assert abs(snap.pose.qw - 1.0) < 1e-5
    assert abs(snap.pose.tx) < 1e-5


# ── Latency benchmark ─────────────────────────────────────────────────────────


def test_snapshot_read_latency_under_1ms(tmp_mmap: Path) -> None:
    """IPC read latency must be < 1 ms per snapshot (target from dev plan)."""
    n = 500_000
    max_g = n

    rng = np.random.default_rng(0)
    gaussians = np.empty(n, dtype=SharedMemWriter.GAUSSIAN_DTYPE)
    for col in ("x", "y", "z", "opacity", "r", "g", "b"):
        gaussians[col] = rng.random(n).astype(np.float32)

    pose = CameraPose(tx=0.0, ty=0.0, tz=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

    with SharedMemWriter(tmp_mmap, max_g) as writer:
        writer.write_snapshot(gaussians, pose=pose, frame_id=1, timestamp_ns=0)
        with SharedMemReader(tmp_mmap, max_g) as reader:
            latency_s = measure_snapshot_latency(reader, iterations=100)

    latency_ms = latency_s * 1000
    print(f"\nSnapshot read latency: {latency_ms:.3f} ms (500K Gaussians)")
    assert latency_ms < 1.0, f"Expected < 1 ms, got {latency_ms:.3f} ms"


# ── Cross-language test (requires cargo) ─────────────────────────────────────


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo not available")
def test_rust_writer_python_reader(tmp_path: Path) -> None:
    """Rust writes a snapshot; Python reads and verifies data integrity.

    Builds and runs ``crates/atlas-core/examples/ipc_write_test`` which
    creates a known snapshot, then verifies the Python reader decodes it
    correctly.
    """
    mmap_path = tmp_path / "rust_test.mmap"

    # Build the Rust example.
    build_result = subprocess.run(
        [
            "cargo",
            "build",
            "--example",
            "ipc_write_test",
            "-p",
            "atlas-core",
            "--quiet",
        ],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
    )
    if build_result.returncode != 0:
        pytest.skip(f"Could not build ipc_write_test: {build_result.stderr}")

    # Run the example, passing the mmap path as an argument.
    run_result = subprocess.run(
        [
            str(WORKSPACE_ROOT / "target" / "debug" / "examples" / "ipc_write_test"),
            str(mmap_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run_result.returncode == 0, f"ipc_write_test failed: {run_result.stderr}"

    # Python reads the Rust-written snapshot.
    max_g = 1000
    with SharedMemReader(mmap_path, max_g) as reader:
        snap = reader.get_latest_snapshot()

    # ipc_write_test writes: frame_id=99, 3 Gaussians at known positions.
    assert snap.frame_id == 99
    assert len(snap.gaussians) == 3
    assert abs(snap.gaussians[0]["x"] - 1.0) < 1e-4
    assert abs(snap.gaussians[1]["x"] - 2.0) < 1e-4
    assert abs(snap.gaussians[2]["x"] - 3.0) < 1e-4


# ── Protobuf IPC round-trips ──────────────────────────────────────────────────


def test_protobuf_semantic_label_round_trip() -> None:
    """SemanticLabel encodes and decodes without data loss."""
    from atlas.utils.atlas_pb2 import Point3, SemanticLabel  # type: ignore[import]

    label = SemanticLabel(
        object_id=42,
        label="coffee_cup",
        material="ceramic",
        mass_kg=0.35,
        fragility=0.8,
        friction=0.5,
        confidence=0.95,
        position=Point3(x=1.0, y=0.5, z=0.2),
    )

    encoded = label.SerializeToString()
    decoded = SemanticLabel()
    decoded.ParseFromString(encoded)

    assert decoded.object_id == 42
    assert decoded.label == "coffee_cup"
    assert decoded.material == "ceramic"
    assert abs(decoded.mass_kg - 0.35) < 1e-5
    assert abs(decoded.fragility - 0.8) < 1e-5
    assert abs(decoded.position.x - 1.0) < 1e-5


def test_protobuf_risk_assessment_round_trip() -> None:
    """RiskAssessment encodes and decodes without data loss."""
    from atlas.utils.atlas_pb2 import Point3, RiskAssessment  # type: ignore[import]

    risk = RiskAssessment(
        object_id=7,
        risk_type="Fall",
        probability=0.72,
        impact_point=Point3(x=0.5, y=0.0, z=0.0),
        description="Glass near table edge",
    )

    encoded = risk.SerializeToString()
    decoded = RiskAssessment()
    decoded.ParseFromString(encoded)

    assert decoded.object_id == 7
    assert decoded.risk_type == "Fall"
    assert abs(decoded.probability - 0.72) < 1e-5
    assert decoded.description == "Glass near table edge"
    assert abs(decoded.impact_point.x - 0.5) < 1e-5


def test_protobuf_scene_state_round_trip() -> None:
    """SceneState with nested messages encodes and decodes correctly."""
    from atlas.utils.atlas_pb2 import (  # type: ignore[import]
        Point3,
        Pose,
        RiskAssessment,
        SceneState,
        SemanticLabel,
    )

    state = SceneState(
        frame_id=100,
        camera_pose=Pose(
            position=Point3(x=1.0, y=2.0, z=3.0),
            qw=1.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
        ),
        objects=[
            SemanticLabel(object_id=1, label="chair", confidence=0.9),
            SemanticLabel(object_id=2, label="table", confidence=0.85),
        ],
        risks=[
            RiskAssessment(object_id=1, risk_type="Instability", probability=0.3),
        ],
    )

    encoded = state.SerializeToString()
    decoded = SceneState()
    decoded.ParseFromString(encoded)

    assert decoded.frame_id == 100
    assert len(decoded.objects) == 2
    assert decoded.objects[0].label == "chair"
    assert decoded.objects[1].label == "table"
    assert len(decoded.risks) == 1
    assert decoded.risks[0].risk_type == "Instability"
    assert abs(decoded.camera_pose.position.x - 1.0) < 1e-5
