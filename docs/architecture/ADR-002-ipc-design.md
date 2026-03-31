# ADR-002: Shared Memory + Protobuf IPC over gRPC for Rust↔Python Communication

**Status**: Accepted  
**Date**: 2026-03-31  
**Authors**: Atlas-0 core team  
**Phase**: Phase 2 (The Brain)

---

## Context

Atlas-0 has a strict polyglot split: Rust owns the hot path (camera → SLAM → Gaussian map at 60fps) and Python owns the reasoning path (VLM inference, semantic labeling, spatial queries, API server). These two runtimes must exchange data in real time.

Two classes of data need to cross the boundary:

| Data Class | Size | Rate | Characteristics |
|---|---|---|---|
| **Gaussian map snapshots** | 2–28 MB (100K–1M Gaussians × 28 bytes) | ~1 snapshot / 5 keyframes (~0.1–1 Hz) | Large, unstructured float32 arrays; Python needs zero-copy access |
| **Structured messages** (semantic labels, risk assessments, queries) | < 1 KB each | ~1–10 Hz | Small, typed, need versioned schema |

Three IPC approaches were evaluated:

| Approach | Latency | Throughput | Schema | Complexity |
|---|---|---|---|---|
| **gRPC (HTTP/2 + protobuf)** | ~1–5 ms network + serialisation | Moderate | Excellent (protobuf IDL) | High (codegen, TLS, service stubs) |
| **Shared memory mmap + protobuf** | < 0.1 ms (kernel mmap) | Excellent | Good (binary layout + protobuf) | Moderate |
| **Unix socket / named pipe** | < 0.5 ms | Good | Manual | Low |

---

## Decision

We use a **two-channel IPC design**:

1. **Shared memory (mmap)** for Gaussian map snapshots — Rust writes via `memmap2`; Python reads via `mmap` + numpy with zero-copy semantics.
2. **Protobuf messages** for structured semantic / risk data — serialised to the same mmap region (or a secondary channel for Phase 3) using `prost` (Rust) and `protobuf` (Python).

The Rust↔Python boundary is defined in `proto/atlas.proto`, compiled independently for each language.

---

## Rationale

### Why shared memory for map snapshots

**Zero-copy access is mandatory at this data size.**  
A 100 K Gaussian snapshot is 2.8 MB. At 1 Hz, that is 2.8 MB/s — well within mmap capability. gRPC serialisation of 2.8 MB per call adds ~5–20ms of copying and encoding overhead per snapshot, which exceeds the 5ms IPC budget from the Phase 2 spec.

With mmap, Python receives a `numpy.ndarray` backed directly by the kernel-mapped page. There is no memcpy. The round-trip latency measured in benchmarks is < 0.003 ms for 1 000 Gaussians and sub-millisecond for 100 K Gaussians.

**Double-buffer prevents read-write contention without locks.**  
The binary layout uses two equal-sized Gaussian buffers. The Rust writer fills the inactive buffer and flips an atomic `write_index` field in the 64-byte header. The Python reader always reads from the buffer pointed to by `write_index`. There is no mutex required between the processes — the flip is a single 32-bit store with mmap flush as a release fence.

**Simplicity of the binary layout.**  
The layout (`header[64] || buffer0[N×28] || buffer1[N×28]`) is defined once in `crates/atlas-core/src/shared_mem.rs` and mirrored in `python/atlas/utils/shared_mem.py` as documented constants. Any mismatch (e.g., `BYTES_PER_GAUSSIAN`) causes an immediate validation error at reader construction time.

### Why protobuf for structured messages

**Schema evolution without breaking the other language.**  
Semantic labels, risk assessments, and query responses have evolving fields (e.g., adding `density` to `SemanticLabel`). Protobuf field numbering guarantees backwards compatibility: old Rust binaries can send messages to new Python readers and vice versa. Raw binary structs or JSON lack this guarantee.

**Existing dependency, no new build complexity.**  
`prost` is already a workspace dependency. `protobuf` is already in `pyproject.toml`. Using protobuf for structured messages requires no additional infrastructure.

**Compact encoding.**  
A `SemanticLabel` message (label, material, mass_kg, fragility, friction, confidence) encodes to ~60 bytes in protobuf vs ~130 bytes in JSON. This matters for the WebSocket risk stream in Phase 3 where many labels are pushed per second.

### Why not gRPC

**gRPC is over-engineered for same-host IPC.**  
gRPC was designed for cross-network service meshes with TLS, load balancing, and service discovery. For Atlas-0 where Rust and Python always run on the same machine, the full gRPC stack adds ~5ms latency (HTTP/2 framing + TLS negotiation + protobuf marshal) per call — more than the entire 5ms IPC budget.

**gRPC requires a running server and connection handshake.**  
If the Python server starts before the Rust SLAM pipeline, gRPC connections fail until Rust is ready. mmap files are created once by the writer and persist on disk; the reader can open them at any time without coordination.

**gRPC introduces network-facing attack surface.**  
Even on localhost, a listening gRPC port can be probed by other processes. mmap files have UNIX file permissions — access control is OS-enforced at the file level.

---

## Alternatives Considered

### gRPC

**Pros**: Auto-generated clients/servers; strong typing; streaming support; works across machines.  
**Cons**: 5–20ms per call latency for large payloads; requires running gRPC server; complex setup (codegen, channel management, TLS); overkill for same-host IPC.

**Rejected** for primary data path due to latency budget violation. May be reconsidered if Atlas-0 becomes a distributed system (Phase 4+).

### Unix domain sockets (raw)

**Pros**: Very low latency; no server setup; bidirectional.  
**Cons**: Copy-on-receive (no zero-copy); message framing must be implemented manually; no schema versioning.

**Rejected** because zero-copy is critical for Gaussian snapshots, and manual framing adds maintenance burden.

### Redis / message queue (ZeroMQ, NATS)

**Pros**: Off-the-shelf, familiar to many engineers.  
**Cons**: Adds a daemon dependency; adds serialisation latency; significantly over-engineered for a two-process same-machine use case.

**Rejected** as unnecessary operational complexity for a two-process pipeline.

---

## Binary Layout (Canonical Reference)

```
Offset  Size  Field            Description
──────  ────  ───────────────  ──────────────────────────────────────────
0       4     magic            0xA71A5000 — identifies Atlas mmap files
4       4     version          Format version (currently 1)
8       8     frame_id         SLAM frame counter at last write
16      8     timestamp_ns     UNIX timestamp in nanoseconds
24      4     gaussian_count   Number of Gaussians in the active buffer
28      4     pose_tx          Camera translation X (metres, float32)
32      4     pose_ty          Camera translation Y
36      4     pose_tz          Camera translation Z
40      4     pose_qw          Camera rotation quaternion W
44      4     pose_qx          Quaternion X
48      4     pose_qy          Quaternion Y
52      4     pose_qz          Quaternion Z
56      4     write_index      Active buffer index (0 or 1)
60      4     _padding         Reserved, must be zero
64      N×28  buffer 0         Gaussian array (x,y,z,opacity,r,g,b — float32 LE)
64+N×28 N×28  buffer 1         Double-buffer slot
```

Where `N = max_gaussians` (configured in `[ipc] max_gaussians`, default 100 000).

---

## Consequences

- `crates/atlas-core/src/shared_mem.rs` — Rust writer; single source of truth for the binary layout.
- `python/atlas/utils/shared_mem.py` — Python reader/writer; constants must stay in sync with Rust.
- `proto/atlas.proto` — canonical schema for all structured messages; compiled via `prost-build` in `atlas-core/build.rs` and `protoc` for Python.
- Any change to the binary layout requires incrementing `ATLAS_MMAP_VERSION` and updating both files simultaneously.
- Phase 3 physics risk assessments will be serialised as `RiskAssessment` protobuf messages written to a secondary mmap channel (or a ring buffer appended after the Gaussian buffers).
