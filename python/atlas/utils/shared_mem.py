"""Shared-memory IPC helpers for Rust→Python Gaussian snapshot transfer.

Binary layout (must match ``crates/atlas-core/src/shared_mem.rs`` exactly):

.. code-block:: text

    ┌────────────────────────── 64-byte header ─────────────────────────────┐
    │  magic(4)  version(4)  frame_id(8)  timestamp_ns(8)                    │
    │  gaussian_count(4)  pose_tx(4)  pose_ty(4)  pose_tz(4)                 │
    │  pose_qw(4)  pose_qx(4)  pose_qy(4)  pose_qz(4)                       │
    │  write_index(4)  _padding(4)                                            │
    ├──────────────────────── Buffer 0 ─────────────────────────────────────┤
    │  gaussian_count x 28 bytes  (x,y,z,opacity,r,g,b -- all float32 LE)   │
    ├──────────────────────── Buffer 1 ─────────────────────────────────────┤
    │  gaussian_count x 28 bytes                                              │
    └────────────────────────────────────────────────────────────────────────┘

All multi-byte integers are **little-endian**.  Double-buffering avoids
read-write contention: the Rust writer fills the *inactive* buffer and
then atomically flips ``write_index``.
"""

import mmap
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── Constants (mirror crates/atlas-core/src/shared_mem.rs) ──────────────────

ATLAS_MMAP_MAGIC: int = 0xA71A5000
ATLAS_MMAP_VERSION: int = 1
HEADER_SIZE: int = 64
BYTES_PER_GAUSSIAN: int = 28  # x,y,z,opacity,r,g,b -- 7 x float32

# Header byte offsets.
_OFF_MAGIC: int = 0
_OFF_VERSION: int = 4
_OFF_FRAME_ID: int = 8
_OFF_TIMESTAMP_NS: int = 16
_OFF_GAUSSIAN_COUNT: int = 24
_OFF_POSE_TX: int = 28
_OFF_POSE_TY: int = 32
_OFF_POSE_TZ: int = 36
_OFF_POSE_QW: int = 40
_OFF_POSE_QX: int = 44
_OFF_POSE_QY: int = 48
_OFF_POSE_QZ: int = 52
_OFF_WRITE_INDEX: int = 56

# Numpy dtype matching one serialised Gaussian entry.
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


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class CameraPose:
    """Camera pose extracted from a snapshot header."""

    tx: float
    ty: float
    tz: float
    qw: float
    qx: float
    qy: float
    qz: float


@dataclass
class MapSnapshot:
    """A snapshot of the Gaussian map as written by the Rust SLAM pipeline.

    Args:
        frame_id: SLAM frame index when the snapshot was written.
        timestamp_ns: UNIX timestamp in nanoseconds at write time.
        pose: Camera pose at snapshot time.
        gaussians: Structured numpy array of shape ``(N,)`` with dtype
            ``{x,y,z,opacity,r,g,b}`` (all float32).  Zero-copy view into
            the mmap buffer when possible.
    """

    frame_id: int
    timestamp_ns: int
    pose: CameraPose
    gaussians: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=_GAUSSIAN_DTYPE))


# ── Reader ───────────────────────────────────────────────────────────────────


class SharedMemReader:
    """Zero-copy reader for Atlas-0 Gaussian snapshots.

    Opens the memory-mapped file produced by the Rust ``SharedMemWriter``
    and exposes the latest snapshot via :meth:`get_latest_snapshot`.

    Args:
        path: Path to the ``.mmap`` file created by the Rust writer.
        max_gaussians: Capacity declared when the file was created.  Must
            match the ``max_gaussians`` used by the Rust writer.

    Raises:
        ValueError: If the file header contains an invalid magic number or
            an unsupported format version.
        OSError: If the file cannot be opened or memory-mapped.

    Example::

        reader = SharedMemReader(Path("/tmp/atlas.mmap"), max_gaussians=100_000)
        with reader:
            snap = reader.get_latest_snapshot()
            print(f"frame {snap.frame_id}: {len(snap.gaussians)} Gaussians")
    """

    def __init__(self, path: Path, max_gaussians: int) -> None:
        self._path = path
        self._max_gaussians = max_gaussians
        file_size = HEADER_SIZE + max_gaussians * BYTES_PER_GAUSSIAN * 2

        self._file = open(path, "rb")  # noqa: SIM115 — kept open for mmap lifetime
        self._mmap = mmap.mmap(self._file.fileno(), file_size, access=mmap.ACCESS_READ)
        self._validate_header()

    def _validate_header(self) -> None:
        raw = self._mmap[:HEADER_SIZE]
        (magic,) = struct.unpack_from("<I", raw, _OFF_MAGIC)
        if magic != ATLAS_MMAP_MAGIC:
            raise ValueError(f"Invalid Atlas mmap magic: {magic:#010x}")
        (version,) = struct.unpack_from("<I", raw, _OFF_VERSION)
        if version != ATLAS_MMAP_VERSION:
            raise ValueError(f"Unsupported Atlas mmap version: {version}")

    def get_latest_snapshot(self) -> MapSnapshot:
        """Return the latest fully-written snapshot from the active buffer.

        This is a non-blocking read: it picks up whichever snapshot the Rust
        writer last committed.  Call repeatedly to poll for new frames.

        Returns:
            :class:`MapSnapshot` containing the decoded Gaussian array and
            camera pose.  The ``gaussians`` field is a numpy structured array
            with zero-copy semantics when the underlying mmap buffer is still
            valid.
        """
        # Re-read header bytes on each call (mmap is always up-to-date).
        header = self._mmap[:HEADER_SIZE]

        (write_index,) = struct.unpack_from("<I", header, _OFF_WRITE_INDEX)
        (raw_count,) = struct.unpack_from("<I", header, _OFF_GAUSSIAN_COUNT)
        gaussian_count = min(raw_count, self._max_gaussians)
        (frame_id,) = struct.unpack_from("<Q", header, _OFF_FRAME_ID)
        (timestamp_ns,) = struct.unpack_from("<Q", header, _OFF_TIMESTAMP_NS)

        pose = CameraPose(
            tx=struct.unpack_from("<f", header, _OFF_POSE_TX)[0],
            ty=struct.unpack_from("<f", header, _OFF_POSE_TY)[0],
            tz=struct.unpack_from("<f", header, _OFF_POSE_TZ)[0],
            qw=struct.unpack_from("<f", header, _OFF_POSE_QW)[0],
            qx=struct.unpack_from("<f", header, _OFF_POSE_QX)[0],
            qy=struct.unpack_from("<f", header, _OFF_POSE_QY)[0],
            qz=struct.unpack_from("<f", header, _OFF_POSE_QZ)[0],
        )

        if gaussian_count == 0:
            gaussians = np.empty(0, dtype=_GAUSSIAN_DTYPE)
        else:
            buf_offset = HEADER_SIZE + write_index * self._max_gaussians * BYTES_PER_GAUSSIAN
            n_bytes = gaussian_count * BYTES_PER_GAUSSIAN
            # Zero-copy view: frombuffer returns an array backed by the mmap.
            buf_slice = self._mmap[buf_offset : buf_offset + n_bytes]
            gaussians = np.frombuffer(buf_slice, dtype=_GAUSSIAN_DTYPE)

        return MapSnapshot(
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            pose=pose,
            gaussians=gaussians,
        )

    def close(self) -> None:
        """Release the mmap mapping and close the file."""
        self._mmap.close()
        self._file.close()

    def __enter__(self) -> "SharedMemReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


# ── Writer (used for testing and Python-side snapshot production) ────────────


class SharedMemWriter:
    """Writes Gaussian snapshots to an Atlas-0 mmap file.

    Creates the same binary layout as the Rust ``SharedMemWriter``, enabling
    Python-only tests and future Python-authored snapshots.

    Args:
        path: Destination file path.  Created or truncated on open.
        max_gaussians: Per-buffer capacity.  Gaussians beyond this limit are
            silently truncated.

    Example::

        writer = SharedMemWriter(Path("/tmp/atlas.mmap"), max_gaussians=1000)
        gaussians = np.zeros(10, dtype=writer.GAUSSIAN_DTYPE)
        writer.write_snapshot(gaussians, pose=None, frame_id=1, timestamp_ns=0)
        writer.close()
    """

    GAUSSIAN_DTYPE = _GAUSSIAN_DTYPE

    def __init__(self, path: Path, max_gaussians: int) -> None:
        self._path = path
        self._max_gaussians = max_gaussians
        file_size = HEADER_SIZE + max_gaussians * BYTES_PER_GAUSSIAN * 2

        self._file = open(path, "w+b")  # noqa: SIM115
        self._file.truncate(file_size)
        self._file.flush()
        self._mmap = mmap.mmap(self._file.fileno(), file_size)
        self._init_header()
        self._write_index = 0

    def _init_header(self) -> None:
        struct.pack_into("<I", self._mmap, _OFF_MAGIC, ATLAS_MMAP_MAGIC)
        struct.pack_into("<I", self._mmap, _OFF_VERSION, ATLAS_MMAP_VERSION)
        struct.pack_into("<I", self._mmap, _OFF_WRITE_INDEX, 0)
        self._mmap.flush()

    def write_snapshot(
        self,
        gaussians: np.ndarray,
        pose: CameraPose | None,
        frame_id: int,
        timestamp_ns: int,
    ) -> None:
        """Write a snapshot into the inactive double-buffer slot.

        Args:
            gaussians: Structured array with dtype matching
                :attr:`GAUSSIAN_DTYPE`.  Silently truncated to
                ``max_gaussians``.
            pose: Camera pose for this snapshot.  Identity pose is used when
                ``None``.
            frame_id: SLAM frame counter value.
            timestamp_ns: UNIX timestamp in nanoseconds.
        """
        if pose is None:
            pose = CameraPose(tx=0.0, ty=0.0, tz=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)

        current_idx = self._write_index
        next_idx = 1 - current_idx
        n = min(len(gaussians), self._max_gaussians)

        buf_offset = HEADER_SIZE + next_idx * self._max_gaussians * BYTES_PER_GAUSSIAN
        if n > 0:
            data = gaussians[:n].tobytes()
            self._mmap[buf_offset : buf_offset + len(data)] = data

        # Update header fields.
        struct.pack_into("<Q", self._mmap, _OFF_FRAME_ID, frame_id)
        struct.pack_into("<Q", self._mmap, _OFF_TIMESTAMP_NS, timestamp_ns)
        struct.pack_into("<I", self._mmap, _OFF_GAUSSIAN_COUNT, n)
        struct.pack_into("<f", self._mmap, _OFF_POSE_TX, pose.tx)
        struct.pack_into("<f", self._mmap, _OFF_POSE_TY, pose.ty)
        struct.pack_into("<f", self._mmap, _OFF_POSE_TZ, pose.tz)
        struct.pack_into("<f", self._mmap, _OFF_POSE_QW, pose.qw)
        struct.pack_into("<f", self._mmap, _OFF_POSE_QX, pose.qx)
        struct.pack_into("<f", self._mmap, _OFF_POSE_QY, pose.qy)
        struct.pack_into("<f", self._mmap, _OFF_POSE_QZ, pose.qz)

        # Flip the active buffer index (release fence equivalent).
        self._mmap.flush()
        struct.pack_into("<I", self._mmap, _OFF_WRITE_INDEX, next_idx)
        self._mmap.flush()
        self._write_index = next_idx

    def close(self) -> None:
        """Flush and close the mmap file."""
        self._mmap.flush()
        self._mmap.close()
        self._file.close()

    def __enter__(self) -> "SharedMemWriter":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


# ── Latency helper ───────────────────────────────────────────────────────────


def measure_snapshot_latency(reader: SharedMemReader, iterations: int = 1000) -> float:
    """Measure the average wall-clock time to read one snapshot in seconds.

    Args:
        reader: An open :class:`SharedMemReader`.
        iterations: Number of read calls to average over.

    Returns:
        Mean latency in **seconds** per snapshot read.
    """
    start = time.perf_counter()
    for _ in range(iterations):
        reader.get_latest_snapshot()
    elapsed = time.perf_counter() - start
    return elapsed / iterations
