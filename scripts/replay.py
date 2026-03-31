#!/usr/bin/env python3
"""Atlas-0 scene replay tool.

Records Gaussian map snapshots from shared memory to a compact binary
archive, then replays them back into shared memory at the original
capture rate for deterministic VLM / query pipeline testing.

Two sub-commands:

``record``
    Poll the Atlas mmap file and save each unique snapshot (by frame_id)
    to an ``.atlas_replay`` archive.

``replay``
    Read a previously recorded archive and write each snapshot back into
    shared memory, pacing emission to match the original capture interval
    (or at a configurable speed multiplier).

Archive format
--------------
The ``.atlas_replay`` file is a simple stream of fixed-width records:

.. code-block:: text

    ┌── header (32 bytes) ──────────────────────────────────────────────┐
    │  magic(4)  version(4)  snapshot_count(4)  max_gaussians(4)        │
    │  capture_fps(4/f32)    _padding(12)                               │
    ├── snapshot 0 ─────────────────────────────────────────────────────┤
    │  frame_id(8)  timestamp_ns(8)  gaussian_count(4)  _pad(4)  (52 B)  │
    │  pose: tx,ty,tz,qw,qx,qy,qz  (7 x f32 = 28 bytes)                │
    │  gaussians: gaussian_count x 28 bytes                             │
    ├── snapshot 1 ──────────────────────────────────────────────────── ┤
    │  …                                                                 │
    └────────────────────────────────────────────────────────────────────┘

All integers are little-endian.

Usage::

    # Record 60 seconds from a running SLAM pipeline.
    python scripts/replay.py record --duration 60 --output session.atlas_replay

    # Replay at original speed.
    python scripts/replay.py replay session.atlas_replay

    # Replay at 2x speed.
    python scripts/replay.py replay session.atlas_replay --speed 2.0

    # Replay in a loop (useful for continuous integration).
    python scripts/replay.py replay session.atlas_replay --loop
"""

from __future__ import annotations

import argparse
import io
import os
import signal
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure repo python/ is importable.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python"))

import numpy as np  # noqa: E402
from atlas.utils.shared_mem import (  # noqa: E402
    CameraPose,
    SharedMemReader,
    SharedMemWriter,
)

# ── Archive constants ──────────────────────────────────────────────────────────

_ARCHIVE_MAGIC: int = 0xA71A_AA00  # Atlas-0 replay archive
_ARCHIVE_VERSION: int = 1
_ARCHIVE_HEADER_SIZE: int = 32
_SNAPSHOT_FIXED_SIZE: int = 52  # frame_id(8) + ts(8) + count(4) + pad(4) + pose(7*f32=28)

_GAUSSIAN_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("opacity", "<f4"),
        ("r", "<f4"),
        ("g", "<f4"),
        ("b", "<f4"),
    ]
)


# ── Archive dataclasses ───────────────────────────────────────────────────────


@dataclass
class ArchiveHeader:
    """Top-level header for a replay archive file."""

    snapshot_count: int
    max_gaussians: int
    capture_fps: float


@dataclass
class SnapshotRecord:
    """One serialised snapshot stored in the archive."""

    frame_id: int
    timestamp_ns: int
    pose: CameraPose
    gaussians: np.ndarray


# ── Serialization helpers ─────────────────────────────────────────────────────


def _write_archive_header(buf: io.RawIOBase, header: ArchiveHeader) -> None:
    buf.write(
        struct.pack(
            "<IIIIf12s",
            _ARCHIVE_MAGIC,
            _ARCHIVE_VERSION,
            header.snapshot_count,
            header.max_gaussians,
            header.capture_fps,
            b"\x00" * 12,
        )
    )


def _read_archive_header(buf: io.RawIOBase) -> ArchiveHeader:
    raw = buf.read(_ARCHIVE_HEADER_SIZE)
    if len(raw) < _ARCHIVE_HEADER_SIZE:
        raise ValueError("Truncated archive header")
    magic, version, count, max_g, fps = struct.unpack_from("<IIIIf", raw)
    if magic != _ARCHIVE_MAGIC:
        raise ValueError(f"Invalid archive magic: {magic:#010x}")
    if version != _ARCHIVE_VERSION:
        raise ValueError(f"Unsupported archive version: {version}")
    return ArchiveHeader(snapshot_count=count, max_gaussians=max_g, capture_fps=fps)


def _write_snapshot(buf: io.RawIOBase, rec: SnapshotRecord) -> None:
    n = len(rec.gaussians)
    p = rec.pose
    buf.write(
        struct.pack(
            "<QQIi7f",
            rec.frame_id,
            rec.timestamp_ns,
            n,
            0,  # padding
            p.tx,
            p.ty,
            p.tz,
            p.qw,
            p.qx,
            p.qy,
            p.qz,
        )
    )
    if n > 0:
        buf.write(rec.gaussians.tobytes())


def _read_snapshot(buf: io.RawIOBase, max_gaussians: int) -> SnapshotRecord | None:
    fixed = buf.read(_SNAPSHOT_FIXED_SIZE)
    if not fixed:
        return None
    if len(fixed) < _SNAPSHOT_FIXED_SIZE:
        raise ValueError(f"Truncated snapshot fixed header ({len(fixed)} bytes)")
    unpacked = struct.unpack_from("<QQIi7f", fixed)
    frame_id, timestamp_ns, n, _pad = unpacked[:4]
    tx, ty, tz, qw, qx, qy, qz = unpacked[4:]
    n_actual = min(n, max_gaussians)
    n_bytes = n_actual * _GAUSSIAN_DTYPE.itemsize
    gs_bytes = buf.read(n_bytes)
    if len(gs_bytes) < n_bytes:
        raise ValueError(f"Truncated Gaussian data ({len(gs_bytes)}/{n_bytes} bytes)")
    gaussians = np.frombuffer(gs_bytes, dtype=_GAUSSIAN_DTYPE).copy()
    return SnapshotRecord(
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
        pose=CameraPose(tx=tx, ty=ty, tz=tz, qw=qw, qx=qx, qy=qy, qz=qz),
        gaussians=gaussians,
    )


# ── Record command ────────────────────────────────────────────────────────────


def cmd_record(args: argparse.Namespace) -> None:
    """Poll the Atlas mmap file and record unique snapshots to disk.

    Args:
        args: Parsed CLI arguments (mmap_path, output, duration, fps).
    """
    mmap_path = Path(args.mmap_path)
    output_path = Path(args.output)
    duration_s: float = args.duration
    poll_fps: float = args.fps

    if not mmap_path.exists():
        print(f"[replay] ERROR: mmap file not found: {mmap_path}", file=sys.stderr)
        print("         Start the Atlas SLAM pipeline first.", file=sys.stderr)
        sys.exit(1)

    max_gaussians = args.max_gaussians
    reader = SharedMemReader(mmap_path, max_gaussians=max_gaussians)
    print(f"[replay] Recording from {mmap_path} for {duration_s:.0f}s → {output_path}")
    print(f"[replay] Poll rate: {poll_fps:.1f} fps  |  Ctrl+C to stop early")

    records: list[SnapshotRecord] = []
    last_frame_id = -1
    deadline = time.monotonic() + duration_s
    poll_interval = 1.0 / poll_fps
    start_ts: float | None = None

    stop = False

    def _handle_sigint(_sig: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sigint)

    while time.monotonic() < deadline and not stop:
        snap = reader.get_latest_snapshot()
        if snap.frame_id != last_frame_id and snap.frame_id > 0:
            if start_ts is None:
                start_ts = time.monotonic()
            records.append(
                SnapshotRecord(
                    frame_id=snap.frame_id,
                    timestamp_ns=snap.timestamp_ns,
                    pose=snap.pose,
                    gaussians=snap.gaussians.copy(),
                )
            )
            last_frame_id = snap.frame_id
            elapsed = time.monotonic() - (start_ts or time.monotonic())
            print(
                f"\r[replay] Recorded {len(records)} snapshots  "
                f"(frame_id={snap.frame_id}  {elapsed:.1f}s)   ",
                end="",
                flush=True,
            )
        time.sleep(poll_interval)

    reader.close()
    print(f"\n[replay] Captured {len(records)} snapshots.")

    if not records:
        print("[replay] Nothing captured — no snapshots written.", file=sys.stderr)
        sys.exit(1)

    # Compute approximate capture FPS from first and last timestamp.
    if len(records) > 1:
        dt_ns = records[-1].timestamp_ns - records[0].timestamp_ns
        capture_fps = (len(records) - 1) / (dt_ns / 1e9) if dt_ns > 0 else poll_fps
    else:
        capture_fps = poll_fps

    header = ArchiveHeader(
        snapshot_count=len(records),
        max_gaussians=max_gaussians,
        capture_fps=capture_fps,
    )

    with open(output_path, "wb") as fh:
        _write_archive_header(fh, header)
        for rec in records:
            _write_snapshot(fh, rec)

    size_mb = output_path.stat().st_size / 1_048_576
    print(f"[replay] Saved to {output_path} ({size_mb:.2f} MB, {len(records)} snapshots)")


# ── Replay command ────────────────────────────────────────────────────────────


def cmd_replay(args: argparse.Namespace) -> None:
    """Write archive snapshots back into shared memory at the original rate.

    Args:
        args: Parsed CLI arguments (input, mmap_path, speed, loop, max_gaussians).
    """
    archive_path = Path(args.input)
    mmap_path = Path(args.mmap_path)
    speed: float = args.speed
    do_loop: bool = args.loop

    if not archive_path.exists():
        print(f"[replay] ERROR: archive not found: {archive_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load archive ──────────────────────────────────────────────────────────
    records: list[SnapshotRecord] = []
    with open(archive_path, "rb") as fh:
        header = _read_archive_header(fh)
        for _ in range(header.snapshot_count):
            rec = _read_snapshot(fh, header.max_gaussians)
            if rec is None:
                break
            records.append(rec)

    if not records:
        print("[replay] Archive contains no snapshots.", file=sys.stderr)
        sys.exit(1)

    max_g = max(len(r.gaussians) for r in records)
    writer = SharedMemWriter(mmap_path, max_gaussians=max(max_g, 1))

    capture_fps = header.capture_fps
    frame_interval_s = (1.0 / capture_fps) / speed if capture_fps > 0 else 0.0

    print(
        f"[replay] Replaying {len(records)} snapshots from {archive_path}"
        f"  (capture_fps={capture_fps:.1f}, speed={speed:.1f}x,"
        f"  interval={frame_interval_s * 1000:.1f}ms)"
    )
    print(f"[replay] Writing to {mmap_path}  |  Ctrl+C to stop")

    stop = False

    def _handle_sigint(_sig: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sigint)

    iteration = 0
    while not stop:
        for i, rec in enumerate(records):
            if stop:
                break
            t_frame_start = time.monotonic()

            writer.write_snapshot(
                gaussians=rec.gaussians,
                pose=rec.pose,
                frame_id=rec.frame_id,
                timestamp_ns=int(time.time_ns()),
            )

            print(
                f"\r[replay] iter={iteration}  frame {i + 1}/{len(records)}"
                f"  gaussians={len(rec.gaussians)}"
                f"  frame_id={rec.frame_id}   ",
                end="",
                flush=True,
            )

            elapsed = time.monotonic() - t_frame_start
            sleep_s = frame_interval_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

        iteration += 1
        if not do_loop:
            break

    writer.close()
    print(f"\n[replay] Done ({iteration} pass(es)).")


# ── CLI ────────────────────────────────────────────────────────────────────────


def _default_mmap() -> str:
    return os.environ.get("ATLAS_MMAP_PATH", "/tmp/atlas.mmap")


def main() -> None:
    """Entry point: dispatch to record or replay sub-command."""
    parser = argparse.ArgumentParser(
        description="Atlas-0 scene replay tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── record sub-command ────────────────────────────────────────────────────
    rec_p = sub.add_parser("record", help="Record snapshots from shared memory to a file")
    rec_p.add_argument(
        "--output",
        "-o",
        default="session.atlas_replay",
        help="Output archive file (default: session.atlas_replay)",
    )
    rec_p.add_argument(
        "--duration",
        "-d",
        type=float,
        default=30.0,
        help="Recording duration in seconds (default: 30)",
    )
    rec_p.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Snapshot poll rate in Hz (default: 5)",
    )
    rec_p.add_argument(
        "--mmap-path",
        default=_default_mmap(),
        help=f"Path to Atlas mmap file (default: {_default_mmap()})",
    )
    rec_p.add_argument(
        "--max-gaussians",
        type=int,
        default=100_000,
        help="Max Gaussians per snapshot (default: 100 000)",
    )

    # ── replay sub-command ────────────────────────────────────────────────────
    rep_p = sub.add_parser("replay", help="Replay a recorded archive into shared memory")
    rep_p.add_argument("input", help="Archive file to replay (.atlas_replay)")
    rep_p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )
    rep_p.add_argument(
        "--loop",
        action="store_true",
        help="Loop the archive indefinitely",
    )
    rep_p.add_argument(
        "--mmap-path",
        default=_default_mmap(),
        help=f"Path to write the mmap file (default: {_default_mmap()})",
    )

    args = parser.parse_args()

    if args.command == "record":
        cmd_record(args)
    else:
        cmd_replay(args)


if __name__ == "__main__":
    main()
