# Atlas-0

**Spatial Reasoning & Physical World-Model Engine**

Atlas-0 transforms a live 2D camera feed into a Semantic 3D Digital Twin that predicts physical consequences in real time. Walk around a room and the system simultaneously:

1. **Builds a 3D Gaussian Splatting reconstruction** of the scene (Phase 1 — The Eye)
2. **Labels every object** with mass, material, fragility, and spatial relationships via a local VLM (Phase 2 — The Brain)
3. **Simulates physics** to predict which objects will fall, spill, or collide, and streams the results as an AR overlay (Phase 3 — The Ghost)

---

## Architecture

```
Camera ──► Rust SLAM Pipeline ──► Shared Memory (mmap)
              │                          │
              ▼                          ▼
        3DGS Reconstruction       Python World-Model Agent
        (atlas-slam crate)          │
              │                    ├── VLM (Ollama)
              ▼                    ├── Spatial Query Engine
        Physics Simulator          └── Risk Aggregator
        (atlas-physics crate)             │
                                          ▼
                                    FastAPI Server
                                    ├── GET  /health
                                    ├── POST /query
                                    ├── GET  /objects
                                    ├── GET  /scene
                                    ├── GET  /metrics  (Prometheus)
                                    ├── GET  /app      (AR Overlay UI)
                                    └── WS   /ws/risks (delta stream)
```

**Polyglot design**: Rust owns the 60 fps hot path (frame ingestion, SLAM, physics). Python owns reasoning (VLM inference, semantic labeling, spatial queries). They share data via memory-mapped files and protobuf messages.

---

## Quick Start

### Prerequisites

- Rust toolchain (`rustup.rs`)
- Python 3.11+
- [Ollama](https://ollama.ai/) running locally (`ollama serve`)
- A webcam or an MP4/MKV test video

### Install

```bash
git clone https://github.com/yashasviudayan-py/Atlas-0
cd Atlas-0

# Python dependencies
pip install -e ".[ml]"

# Pull the default VLM
ollama pull llava:7b
```

### Run (single command)

```bash
python scripts/run_atlas.py
```

This starts the Rust SLAM pipeline (release build) and the Python API server in the correct order, then monitors both processes. Press `Ctrl-C` to stop.

Additional options:

```bash
python scripts/run_atlas.py --dev              # Rust debug build + uvicorn --reload
python scripts/run_atlas.py --no-slam          # Python API only (for testing)
python scripts/run_atlas.py --config configs/custom.toml
```

### Run with Docker

```bash
docker compose -f docker/docker-compose.yml up
```

The compose file starts the Atlas-0 API server and an Ollama instance. The AR overlay is served at `http://localhost:8420/app`.

---

## AR Overlay

Open `http://localhost:8420/app` in a browser. The overlay connects to the WebSocket risk stream and renders:

- **Red spheres** around at-risk objects
- **Dashed arc lines** showing predicted fall trajectories
- **Impact rings** on the surface where objects are predicted to land
- **Alert badges** with severity-coded descriptions

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System status, component liveness, staleness indicator |
| `POST` | `/query` | Natural language spatial query |
| `GET` | `/objects` | All labeled objects with physical properties |
| `GET` | `/scene` | Full scene state snapshot |
| `GET` | `/metrics` | Prometheus metrics (text exposition format) |
| `WS` | `/ws/risks` | Real-time risk delta stream |

### Example Query

```bash
curl -X POST http://localhost:8420/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Where is the most unstable object?", "max_results": 3}'
```

---

## Prometheus Metrics

The `/metrics` endpoint exposes:

| Metric | Type | Description |
|--------|------|-------------|
| `atlas_risk_count` | Gauge | Current number of active risks |
| `atlas_object_count` | Gauge | Current number of labeled objects |
| `atlas_query_total` | Counter | Total spatial queries processed |
| `atlas_ws_clients_active` | Gauge | Connected WebSocket clients |
| `atlas_slam_active` | Gauge | 1 if Rust SLAM pipeline is connected |
| `atlas_assessment_age_seconds` | Gauge | Seconds since last successful assessment |
| `atlas_vlm_request_seconds` | Histogram | VLM inference latency |

---

## Demo Recording

Record a full session and export frames:

```bash
# Record 60 seconds
python scripts/record_demo.py record --duration 60 --output my_session.atlas_demo

# Show recording info
python scripts/record_demo.py info my_session.atlas_demo

# Replay to JSON frames
python scripts/record_demo.py replay my_session.atlas_demo --out-dir ./frames

# Encode to video (requires ffmpeg)
ffmpeg -framerate 10 -i frames/frame_%06d.json demo.mp4
```

---

## Development

### Run Tests

```bash
# Rust
cargo test --all

# Python
pytest python/tests/ -v
```

### Lint & Format

```bash
# Rust
cargo fmt --all
cargo clippy --all-targets -- -D warnings

# Python
ruff check python/
ruff format python/
```

### Benchmarks

```bash
# Rust (criterion)
cargo bench

# Python pipeline benchmark
python scripts/benchmark.py
```

---

## Repository Structure

```
crates/
  atlas-core/      # Shared types, IPC, error handling
  atlas-slam/      # 3DGS-SLAM pipeline
  atlas-physics/   # Rigid-body physics simulation
  atlas-stream/    # Camera capture & frame pipeline
python/atlas/
  vlm/             # Ollama VLM client & inference engine
  world_model/     # Semantic labeling, spatial queries, risk aggregation
  api/             # FastAPI server, WebSocket stream, AR overlay, metrics
  utils/           # Config loader, shared memory reader
frontend/          # Three.js AR overlay web app
proto/             # Protobuf schema (Rust ↔ Python IPC)
configs/           # Runtime TOML configuration
scripts/           # run_atlas.py, benchmark.py, replay.py, record_demo.py
docker/            # Dockerfile + docker-compose.yml
docs/architecture/ # ADRs + performance report
tests/integration/ # Cross-language integration tests
```

---

## Performance

| Stage | Budget | Status |
|-------|--------|--------|
| Frame capture + preprocess | < 5 ms | ✅ |
| Feature extraction + matching | < 8 ms | ✅ |
| Pose estimation | < 3 ms | ✅ |
| Gaussian update (per keyframe) | < 20 ms | ✅ |
| IPC Rust → Python | < 5 ms | ✅ |
| VLM inference (per region) | < 2000 ms | ✅ |
| Physics simulation (full scene) | < 10 ms | ✅ |
| API query response | < 200 ms | ✅ |
| WebSocket risk push | < 50 ms | ✅ |

See [`docs/architecture/performance_report.md`](docs/architecture/performance_report.md) for benchmark details.

---

## Architecture Decisions

- [ADR-001: 3DGS-SLAM over traditional SLAM](docs/architecture/ADR-001-3dgs-slam.md)
- [ADR-002: Shared memory IPC over gRPC](docs/architecture/ADR-002-ipc-design.md)

---

## License

MIT — see [LICENSE](LICENSE).
