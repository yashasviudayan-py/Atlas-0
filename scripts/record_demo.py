#!/usr/bin/env python3
"""Atlas-0 demo recorder.

Records a live Atlas-0 session to a binary ``.atlas_demo`` archive, then
replays it to a sequence of PNG frames that can be encoded into a video with
``ffmpeg``.

Recording captures two streams in parallel:
- **WebSocket risk deltas** from ``/ws/risks`` — risk assessments, overlay
  primitives, and trajectory data.
- **HTTP scene snapshots** from ``/scene`` — full object and risk list, polled
  at a configurable interval.

Both streams are time-stamped and stored together in a single archive so the
replay is deterministic.

Archive format
--------------
The ``.atlas_demo`` file is a length-prefixed binary archive.  Each record is:

    [4 bytes: little-endian uint32 payload length]
    [N bytes: JSON-encoded DemoRecord payload]

The records are written in wall-clock order.

Usage::

    # Record 30 seconds to atlas_demo.atlas_demo
    python scripts/record_demo.py record --duration 30

    # Record indefinitely until Ctrl-C
    python scripts/record_demo.py record --output my_session.atlas_demo

    # Replay to PNG frames in ./frames/
    python scripts/record_demo.py replay my_session.atlas_demo --out-dir ./frames

    # Show a summary of a recording
    python scripts/record_demo.py info my_session.atlas_demo

Environment::

    ATLAS_API_HOST   Host of the running Atlas-0 API  (default: localhost)
    ATLAS_API_PORT   Port of the running Atlas-0 API  (default: 8420)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional dependencies — fail gracefully with a helpful message.
# ---------------------------------------------------------------------------

try:
    import httpx
    import websockets  # type: ignore[import-untyped]
except ImportError as _e:
    print(
        f"[record_demo] Missing dependency: {_e}.\n"
        "Install with: pip install httpx websockets",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_HOST = os.environ.get("ATLAS_API_HOST", "localhost")
_DEFAULT_PORT = int(os.environ.get("ATLAS_API_PORT", "8420"))
_RECORD_MAGIC = b"ATLS"
_RECORD_VERSION = 1
_HEADER_FMT = "!4sHH"  # magic (4) + version (2) + reserved (2)
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_LEN_FMT = "<I"  # little-endian uint32 payload length
_LEN_SIZE = struct.calcsize(_LEN_FMT)

# ── Record schema ─────────────────────────────────────────────────────────────


def _make_record(kind: str, timestamp: float, payload: Any) -> bytes:
    """Encode one record as length-prefixed JSON bytes.

    Args:
        kind: Source identifier — ``"ws_delta"`` or ``"scene_snapshot"``.
        timestamp: Wall-clock time (``time.time()``).
        payload: JSON-serialisable data.

    Returns:
        Length-prefixed bytes ready to write to the archive.
    """
    doc = {"kind": kind, "ts": timestamp, "data": payload}
    body = json.dumps(doc, separators=(",", ":")).encode()
    return struct.pack(_LEN_FMT, len(body)) + body


def _iter_records(path: Path):
    """Yield decoded records from an ``.atlas_demo`` archive.

    Args:
        path: Path to the ``.atlas_demo`` file.

    Yields:
        Decoded record dicts (``{"kind": ..., "ts": ..., "data": ...}``).
    """
    with path.open("rb") as f:
        header = f.read(_HEADER_SIZE)
        if header[:4] != _RECORD_MAGIC:
            raise ValueError(f"Not an Atlas-0 demo file: {path}")
        while True:
            raw_len = f.read(_LEN_SIZE)
            if not raw_len:
                break
            (length,) = struct.unpack(_LEN_FMT, raw_len)
            body = f.read(length)
            if len(body) != length:
                break
            yield json.loads(body)


# ── Record command ────────────────────────────────────────────────────────────


async def _record(
    host: str,
    port: int,
    out_path: Path,
    duration: float | None,
    snapshot_interval: float,
) -> None:
    """Record a live Atlas-0 session to *out_path*.

    Args:
        host: API server host.
        port: API server port.
        out_path: Destination ``.atlas_demo`` file path.
        duration: Stop after this many seconds; ``None`` records until Ctrl-C.
        snapshot_interval: Seconds between scene snapshots via HTTP.
    """
    ws_url = f"ws://{host}:{port}/ws/risks"
    http_url = f"http://{host}:{port}/scene"
    start = time.time()

    print(f"[record_demo] Recording → {out_path}")
    print(f"[record_demo] WebSocket: {ws_url}")
    print(f"[record_demo] Scene snapshots every {snapshot_interval}s from {http_url}")
    if duration:
        print(f"[record_demo] Duration: {duration}s")
    else:
        print("[record_demo] Press Ctrl-C to stop.")

    record_count = 0

    with out_path.open("wb") as f:
        # Write file header.
        f.write(struct.pack(_HEADER_FMT, _RECORD_MAGIC, _RECORD_VERSION, 0))

        async def _ws_reader() -> None:
            nonlocal record_count
            async for msg in websockets.connect(ws_url):
                payload = json.loads(msg)
                f.write(_make_record("ws_delta", time.time(), payload))
                record_count += 1
                if duration and time.time() - start >= duration:
                    return

        async def _snapshot_poller() -> None:
            nonlocal record_count
            async with httpx.AsyncClient(timeout=5.0) as client:
                while True:
                    if duration and time.time() - start >= duration:
                        return
                    try:
                        resp = await client.get(http_url)
                        if resp.status_code == 200:
                            f.write(_make_record("scene_snapshot", time.time(), resp.json()))
                            record_count += 1
                    except Exception as exc:
                        print(f"[record_demo] snapshot error: {exc}", file=sys.stderr)
                    await asyncio.sleep(snapshot_interval)

        async def _progress() -> None:
            while True:
                elapsed = time.time() - start
                if duration:
                    print(
                        f"\r[record_demo] {elapsed:.1f}s / {duration}s — "
                        f"{record_count} records",
                        end="",
                        flush=True,
                    )
                    if elapsed >= duration:
                        return
                else:
                    print(
                        f"\r[record_demo] {elapsed:.1f}s — {record_count} records",
                        end="",
                        flush=True,
                    )
                await asyncio.sleep(0.5)

        try:
            await asyncio.gather(
                _ws_reader(),
                _snapshot_poller(),
                _progress(),
            )
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass

    elapsed = time.time() - start
    print(f"\n[record_demo] Done. {record_count} records in {elapsed:.1f}s → {out_path}")


# ── Replay command ────────────────────────────────────────────────────────────


def _replay(src: Path, out_dir: Path) -> None:
    """Replay a ``.atlas_demo`` archive, writing one JSON frame per record.

    Each record is written as a zero-padded numbered JSON file in *out_dir*,
    suitable for post-processing or encoding to video with ffmpeg.

    Args:
        src: Source ``.atlas_demo`` archive.
        out_dir: Directory for output ``frame_NNNNNN.json`` files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(_iter_records(src))
    if not records:
        print("[record_demo] No records found.", file=sys.stderr)
        return

    print(f"[record_demo] Replaying {len(records)} records → {out_dir}/")
    for i, rec in enumerate(records):
        frame_path = out_dir / f"frame_{i:06d}.json"
        frame_path.write_text(json.dumps(rec, indent=2))

    print(f"[record_demo] Done. {len(records)} frames in {out_dir}/")
    print(
        "[record_demo] Encode to video with:\n"
        "  ffmpeg -framerate 10 -i frame_%06d.json -vf "
        "scale=1920:1080 demo.mp4"
    )


# ── Info command ──────────────────────────────────────────────────────────────


def _info(src: Path) -> None:
    """Print a summary of a ``.atlas_demo`` archive.

    Args:
        src: Source ``.atlas_demo`` archive.
    """
    records = list(_iter_records(src))
    if not records:
        print("[record_demo] Empty or invalid archive.", file=sys.stderr)
        return

    by_kind: dict[str, int] = {}
    for rec in records:
        by_kind[rec["kind"]] = by_kind.get(rec["kind"], 0) + 1

    first_ts = records[0]["ts"]
    last_ts = records[-1]["ts"]
    duration = last_ts - first_ts

    print(f"File    : {src}")
    print(f"Records : {len(records)}")
    print(f"Duration: {duration:.1f}s")
    print(f"Start   : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(first_ts))}")
    for kind, count in sorted(by_kind.items()):
        print(f"  {kind}: {count}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="record_demo",
        description="Record and replay Atlas-0 demo sessions",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="Record a live session")
    rec.add_argument(
        "--host",
        default=_DEFAULT_HOST,
        help=f"API server host (default: {_DEFAULT_HOST})",
    )
    rec.add_argument(
        "--port",
        type=int,
        default=_DEFAULT_PORT,
        help=f"API server port (default: {_DEFAULT_PORT})",
    )
    rec.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("atlas_demo.atlas_demo"),
        help="Output file path (default: atlas_demo.atlas_demo)",
    )
    rec.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="Stop recording after N seconds (default: run until Ctrl-C)",
    )
    rec.add_argument(
        "--snapshot-interval",
        type=float,
        default=1.0,
        help="Seconds between /scene HTTP snapshots (default: 1.0)",
    )

    rep = sub.add_parser("replay", help="Replay a recorded session to JSON frames")
    rep.add_argument("src", type=Path, help="Source .atlas_demo file")
    rep.add_argument(
        "--out-dir",
        type=Path,
        default=Path("frames"),
        help="Output directory for frame JSON files (default: frames/)",
    )

    inf = sub.add_parser("info", help="Print summary of a recording")
    inf.add_argument("src", type=Path, help="Source .atlas_demo file")

    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "record":
        try:
            asyncio.run(
                _record(
                    host=args.host,
                    port=args.port,
                    out_path=args.output,
                    duration=args.duration,
                    snapshot_interval=args.snapshot_interval,
                )
            )
        except KeyboardInterrupt:
            print("\n[record_demo] Interrupted.")
    elif args.command == "replay":
        _replay(src=args.src, out_dir=args.out_dir)
    elif args.command == "info":
        _info(src=args.src)


if __name__ == "__main__":
    main()
