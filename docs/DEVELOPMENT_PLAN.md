# Atlas-0: Detailed Phase Development Plan

> Internal reference document. Maps every deliverable, file, module, and
> integration point needed to take Atlas-0 from scaffolding to production.

---

## Real-World Use Cases

Atlas-0 is not a research demo — it solves concrete problems that people have today.
Every phase must be understood in terms of user value, not just engineering completeness.

### Primary Target: Home & Space Safety Audit
**The user:** A parent childproofing a home. A tenant doing a move-in inspection.
A homeowner prepping for an earthquake or insurance claim.
**The job:** Scan a room with a phone or webcam, get a ranked list of physical
hazards — objects likely to fall, tip, spill, or block an exit — with photos and
positions, exported as a PDF or sent to a Slack channel.
**Why Atlas-0 wins:** It quantifies risk with physics. A human can see a wobbly
lamp, but Atlas-0 calculates the tipping torque, predicts where it lands, and
flags the impact zone on a child's play area.

### Secondary Target: Warehouse & Site Safety Compliance
**The user:** A warehouse manager, construction site supervisor, OSHA auditor.
**The job:** Walk the floor with a camera, auto-generate a hazard report with GPS
coordinates and severity scores. Re-run weekly to detect changes.
**Why Atlas-0 wins:** Continuous monitoring at scale. 10,000 sq ft can't be
manually audited daily, but a camera + Atlas-0 can.

### Tertiary Target: Insurance & Property Inventory
**The user:** Homeowner filing a claim, real estate agent listing a property,
property manager doing a condition assessment.
**The job:** Scan the space, auto-generate an inventory of objects with
estimated values, conditions, and photos embedded in a report.
**Why Atlas-0 wins:** Combines 3D spatial map + semantic labeling into a
structured document that adjusters and agents can actually use.

---

## Current State (Post Phase 1–3 Implementation)

What is built:
- Full Rust workspace: `atlas-core`, `atlas-slam`, `atlas-stream`, `atlas-physics`.
- Python package: VLM engine (Ollama), world model agent, FastAPI server,
  Prometheus metrics, AR overlay, upload pipeline (images only).
- Frontend: Three.js scene viewer, risk overlay, upload UI, WebSocket risk stream.
- Docker: Dockerfile + docker-compose with health checks.
- Tests: 266 Python + 92 Rust tests passing. CI green.

What is **NOT** production-ready:
- No direct MP4/video ingestion (requires manual frame extraction).
- VLM is hard-wired to Ollama — no cloud model option.
- No export formats (.glb, PDF report) that users can take away.
- No real demo footage — frontend shows hardcoded fake data.
- Setup requires 5+ manual prerequisites before seeing anything.

---

## PHASE 1: THE EYE (Parts 1–4) ✅ COMPLETE

### Real-World Value
Without the Eye, there is no 3D map. No 3D map means no spatial understanding,
no object positions, no physics simulation. This phase is the sensor layer — it
turns a cheap webcam into a spatial measurement device.

### Goal
Walk around a room with a camera. See a high-fidelity 3D Gaussian Splatting
reconstruction generate on screen in < 100ms latency.

---

### Part 1: Camera Ingestion Pipeline ✅

**Objective**: Capture live video frames at 60fps and distribute them through
the pipeline with minimal latency.

#### Deliverables

1. **Camera capture backend** (`crates/atlas-stream/src/capture.rs`)
   - Integrate `nokhwa` crate (cross-platform camera access) or `v4l2` on Linux.
   - Implement `CameraCapture` struct: opens device, captures frames at target FPS.
   - Convert raw camera buffers (YUYV/MJPEG) to RGB using `image` crate.
   - Push `Frame` structs into the `FramePipeline` bounded channel.
   - Handle camera disconnection, timeout, and format negotiation errors.

2. **Video file source** (`crates/atlas-stream/src/file_source.rs`)
   - Accept an image-sequence directory for offline testing and benchmarking.
   - Simulate real-time playback by pacing frame emission to match target FPS.
   - This is critical for reproducible development — most SLAM work uses recorded sequences.
   - ⚠️ Direct MP4/MKV decoding gated behind `ffmpeg-next` feature (see Phase 4 Part 13).

3. **Frame preprocessing** (`crates/atlas-stream/src/preprocess.rs`)
   - Resize frames to SLAM working resolution (configurable, default 640x480).
   - Convert to grayscale for feature extraction (keep color copy for Gaussian color).
   - Undistort frames using camera intrinsics (pinhole + radial distortion model).

4. **Camera calibration loader** (`crates/atlas-stream/src/calibration.rs`)
   - Load camera intrinsics (fx, fy, cx, cy) from TOML config.
   - Support standard calibration formats (OpenCV-compatible).

#### Success Criteria ✅
- `cargo run --example capture_demo` opens camera, shows FPS counter.

---

### Part 2: Feature Extraction & Visual Odometry ✅

**Objective**: Extract visual features from each frame and estimate camera motion.

#### Deliverables
1. ORB feature detection + matching (`features.rs`, `matching.rs`)
2. Essential matrix decomposition → R|t pose (`pose_estimation.rs`)
3. Visual odometry pipeline wired in `tracker.rs`

#### Success Criteria ✅
- Process a 30-second video clip, output a camera trajectory.

---

### Part 3: 3D Gaussian Splatting Reconstruction ✅

**Objective**: Build and incrementally update a 3D Gaussian map.

#### Deliverables
1. Monocular depth estimation via ONNX Runtime (`depth.rs`)
2. Gaussian initialization from point cloud (`gaussian_init.rs`)
3. Differentiable Gaussian optimization / Adam (`optimizer.rs`)
4. Keyframe management + pruning (`keyframe.rs`)

#### Success Criteria ✅
- Feed a recorded video, get a recognizable 3D Gaussian reconstruction.

---

### Part 4: Real-time Visualization & Integration ✅

**Objective**: Tie the full pipeline together with a live viewer.

#### Deliverables
1. Tile-based 3DGS renderer (`renderer.rs`)
2. Visualization window: `winit` + `wgpu` (`viewer.rs`)
3. End-to-end pipeline binary (`examples/live_slam.rs`)
4. Shared memory bridge scaffold for Phase 2 (`shared_mem.rs`)

#### Phase 1 Exit Criteria ✅
- Camera captures frames at 60fps.
- SLAM tracks camera pose continuously.
- 3D Gaussian map contains 50K+ Gaussians after 30s.
- End-to-end latency < 100ms.

---

## PHASE 2: THE BRAIN (Parts 5–8) ✅ COMPLETE

### Real-World Value
Geometry alone isn't enough. A glass vase and a stone statue look similar in a
point cloud — only semantics tells them apart. This phase makes the map
*queryable*: "Where is the most fragile object?" returns a coordinate with a
confidence score, not just a blob of points.

For the home safety use case: this is the layer that understands *what* every
object is, so the physics engine in Phase 3 can use the right material
properties when simulating falls.

### Goal
Query the map with natural language and get a correct spatial coordinate back.
Every object has semantic metadata: mass, friction, fragility, material,
relationships.

---

### Part 5: Rust→Python IPC Bridge ✅
- Memory-mapped buffer for `GaussianCloud` snapshots (`shared_mem.rs`)
- Protobuf for structured messages (`proto/atlas.proto`)
- Round-trip latency < 1ms

### Part 6: VLM Integration & Semantic Labeling ✅
- Ollama async client (`ollama_client.py`)
- VLM inference engine with JSON parsing (`inference.py`)
- Versioned prompt templates (`prompts.py`)
- DBSCAN region extractor (`region_extractor.py`)
- Confidence-weighted label store (`label_store.py`)

**Note**: VLM is currently Ollama-only. Multi-provider support (Claude, OpenAI)
added in Phase 4 Part 14.

### Part 7: Spatial Query Engine ✅
- Rule-based NL query parser (`query_parser.py`)
- Spatial relationship detector (`relationships.py`)
- WorldModelAgent scene assessment loop (`agent.py`)
- FastAPI endpoints: `/health`, `/query`, `/objects`, `/scene`

### Part 8: Integration, Benchmarking & Polish ✅
- Typed Pydantic config loader with `ATLAS_` env overrides (`config.py`)
- Process manager script (`scripts/run_atlas.py`)
- Benchmark suite — all budgets passing (`scripts/benchmark.py`)
- Scene replay tool (`scripts/replay.py`)
- ADR-001 (3DGS-SLAM choice), ADR-002 (IPC design)

#### Phase 2 Exit Criteria ✅
- VLM correctly labels ≥ 80% of common objects.
- Spatial queries return correct positions within 0.5m.
- End-to-end query latency < 200ms.

---

## PHASE 3: THE GHOST (Parts 9–12) ✅ COMPLETE

### Real-World Value
This is what Atlas-0 can do that a human inspection cannot: *simulate the
future*. A toddler bumps the table — where does the glass land? A minor
earthquake hits — which shelf collapses first? The physics engine runs these
simulations continuously, in the background, assigning risk scores to every
object.

For the warehouse use case: a pallet stacked at the wrong angle, a shelf
overloaded at the top — Atlas-0 flags these *before* they become incidents,
with quantified probability and predicted impact zones.

### Goal
Real-time AR overlay showing predicted failure paths and risk zones.
Alerts triggered before things go wrong.

---

### Part 9: Physics Simulation Engine ✅
- `RigidBody` struct + bounding shapes (`rigid_body.rs`)
- RANSAC plane extractor for surfaces (`surfaces.rs`)
- Sphere/AABB collision detection + SAT (`collision.rs`)
- Semi-implicit Euler integrator with Baumgarte correction (`integrator.rs`)
- `assess_risks()` with perturbation + classification (`simulator.rs`)
- 51 unit tests, criterion benchmarks.

### Part 10: Risk Prediction Pipeline ✅
- 4-strategy perturbation engine (`perturbations.rs`)
- Trajectory recorder + spill zone predictor (`trajectory.rs`)
- Background `RiskLoop` thread (`simulator.rs`)
- Python risk aggregator: weighted merge, top-N ranking (`risk_aggregator.py`)

### Part 11: AR Overlay & Frontend ✅
- WebSocket delta stream `/ws/risks` (added/updated/removed diffs)
- `OverlayBuilder`: `RiskZone`, `TrajectoryArc`, `ImpactZone`, `Alert` (`overlay.py`)
- Three.js AR overlay: Gaussian particle cloud, neural tether lines,
  knowledge graph, physics ghost (`scene_viewer.js`, `overlay.js`, `app.js`)

### Part 12: Integration, Optimization & Production Hardening ✅
- Prometheus metrics: 7 gauges/counters/histograms (`metrics.py`)
- `/metrics` endpoint + StaticFiles frontend mount (`server.py`)
- Docker: Dockerfile with frontend + HEALTHCHECK, docker-compose health deps
- Demo recording tool (`scripts/record_demo.py`)
- Full README, architecture performance report

#### Phase 3 Exit Criteria ✅
- Physics correctly predicts falls and tips.
- AR overlay streams risk zones in real-time.
- Full pipeline stable for 10+ minutes.
- Docker deployment works out of the box.

---

## PHASE 4: PRODUCTION (Parts 13–16) 🔲 NEXT

### Goal
Turn Atlas-0 from an impressive engineering project into something a real
person can use on a real problem in under 5 minutes. The test: hand it to
someone who has never seen it before. If they can scan a room and get a useful
safety report without reading any docs, Phase 4 is done.

---

### Part 13: Direct Video Ingestion

**Real-world impact**: Right now, a user who records a video has to manually
extract frames with FFmpeg before Atlas-0 can process it. That kills the product
before it starts. This part removes that friction entirely.

**Target**: Drop a `.mp4` / `.mov` / `.webm` on the upload UI → Atlas-0 handles
the rest, no terminal required.

#### Deliverables

1. **Python video frame extractor** (`python/atlas/utils/video.py`)
   - `extract_frames(video_bytes, max_frames, sample_fps) -> list[bytes]`
   - Uses `PyAV` (`av` package) — pure Python ffmpeg bindings, pip-installable.
   - Samples frames evenly across the video duration (not just the beginning).
   - Falls back to a clear `RuntimeError` if `av` is not installed, with install hint.
   - Returns JPEG-encoded frames ready for VLM and point cloud generation.

2. **Update upload pipeline** (`python/atlas/api/server.py`)
   - Replace the "requires Rust SLAM pipeline" stub with real frame extraction.
   - For each sampled frame: run VLM labeling + generate depth point cloud.
   - Merge all frame results: deduplicate objects by label similarity,
     union point clouds, take max risk scores across frames.
   - Report incremental progress per frame (0.1 → 0.9 as frames are processed).

3. **Rust `ffmpeg-next` feature flag** (`crates/atlas-stream/`)
   - Add optional `[features] video = ["ffmpeg-next"]` to `atlas-stream/Cargo.toml`.
   - Behind the flag: `FileSource` accepts a video file path directly, not just
     an image directory.
   - Default: feature disabled (no system FFmpeg dependency at compile time).
   - Document: `cargo build --features video` to enable.

4. **Add `av` optional dependency** (`pyproject.toml`)
   - `[project.optional-dependencies]` `video = ["av>=14.0.0"]`
   - Document in README: `pip install atlas-0[video]` for video support.

#### New Dependencies
- Python: `av>=14.0.0` (optional)
- Rust: `ffmpeg-next` (optional feature, requires system FFmpeg)

#### Tests
- Unit: Frame extraction from synthetic video bytes, even sampling verification.
- Unit: Graceful error when `av` not installed.
- Integration: Upload a short MP4, verify objects and point cloud returned.

#### Success Criteria
- User uploads a 30-second room video via the web UI.
- Gets back labeled objects, risk scores, and a 3D point cloud.
- No terminal interaction required.

---

### Part 14: Multi-Provider VLM Switch

**Real-world impact**: Ollama + `moondream` (the default) gives mediocre
labels. Claude or GPT-4V gives dramatically better semantic understanding,
especially for material classification and mass estimation — which directly
affects risk score accuracy.

A local Ollama setup is a 5-step process with GPU requirements. Many users
won't do it. An API key to Claude or OpenAI unlocks the full system instantly.

**Target**: Set `ATLAS_VLM_PROVIDER=claude` (or `openai`) in the environment,
provide an API key, and get production-quality labels. No code changes.

#### Deliverables

1. **VLM provider protocol** (`python/atlas/vlm/providers/base.py`)
   - `VLMProvider` Protocol: `initialize()`, `generate(image_bytes, prompt) -> str`,
     `close()`.
   - All providers must implement this interface.

2. **Ollama provider** (`python/atlas/vlm/providers/ollama_provider.py`)
   - Wraps existing `OllamaClient`. Backward compatible — default behavior unchanged.

3. **Anthropic (Claude) provider** (`python/atlas/vlm/providers/anthropic_provider.py`)
   - Uses `anthropic` SDK with `claude-sonnet-4-6` (default, configurable).
   - API key from `ANTHROPIC_API_KEY` env var.
   - Sends image as base64 in the `image` content block.
   - Maps `VLMConfig.max_tokens` and `temperature` to Anthropic API params.

4. **OpenAI provider** (`python/atlas/vlm/providers/openai_provider.py`)
   - Uses `openai` SDK with `gpt-4o` (default, configurable).
   - API key from `OPENAI_API_KEY` env var.
   - Sends image as base64 data URL in the `image_url` content block.

5. **Provider factory** (`python/atlas/vlm/providers/__init__.py`)
   - `get_provider(config: VLMConfig) -> VLMProvider`
   - Raises `ValueError` with a clear message if provider string is unknown.
   - Raises `ImportError` with install instructions if the SDK isn't installed.

6. **Update `VLMEngine`** (`python/atlas/vlm/inference.py`)
   - Replace hardcoded `OllamaClient` with `get_provider(config)`.
   - `initialize()`, `label_region()`, `close()` delegate to the provider.
   - All existing tests remain valid (Ollama is still the default).

7. **Config additions** (`python/atlas/utils/config.py`, `configs/default.toml`)
   ```toml
   [vlm]
   provider = "ollama"               # "ollama" | "claude" | "openai"
   claude_model = "claude-sonnet-4-6"
   openai_model = "gpt-4o"
   # API keys read from ANTHROPIC_API_KEY / OPENAI_API_KEY env vars only.
   ```

8. **Optional SDK dependencies** (`pyproject.toml`)
   - `[project.optional-dependencies]`
   - `claude = ["anthropic>=0.40.0"]`
   - `openai = ["openai>=1.50.0"]`

#### Quick-switch cheat-sheet (for README)
```bash
# Use Claude (best quality)
export ATLAS_VLM_PROVIDER=claude
export ANTHROPIC_API_KEY=sk-ant-...
python -m atlas.api.server

# Use OpenAI
export ATLAS_VLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
python -m atlas.api.server

# Use local Ollama (default, free, needs GPU)
python -m atlas.api.server
```

#### Tests
- Unit: Factory returns correct provider type for each string.
- Unit: Factory raises `ValueError` on unknown provider.
- Unit: Factory raises `ImportError` (with install hint) when SDK missing.
- Unit: Anthropic + OpenAI providers format images and parse responses correctly
  (mock SDK clients, no real API calls).

---

### Part 15: Export Formats & Safety Report

**Real-world impact**: The current output lives only in the browser. A property
manager, parent, or insurance adjuster needs something they can *hand to
someone*. This part adds three export paths that make Atlas-0 results portable.

**Target**: One-click export from the frontend → downloadable file.

#### Deliverables

1. **PDF safety report** (`python/atlas/api/export.py`)
   - `generate_pdf_report(scene, risks) -> bytes`
   - Uses `reportlab` or `weasyprint`.
   - Report sections: Summary table, per-object cards (label, material, risk score,
     photo if available), top-5 risks with descriptions, timestamp and location.
   - Endpoint: `GET /export/report.pdf`

2. **3D model export** (`python/atlas/api/export.py`)
   - `generate_glb(point_cloud) -> bytes` — exports the Gaussian point cloud as a
     GLB file (viewable in Blender, Apple Vision Pro, Google Model Viewer).
   - Uses `trimesh` for GLB serialization.
   - Endpoint: `GET /export/scene.glb`

3. **JSON inventory export**
   - `GET /export/inventory.json` — all labeled objects with properties + risks.
   - Machine-readable. Useful for integration with property management systems.

4. **Frontend export buttons**
   - "Download PDF Report", "Download 3D Model (.glb)", "Export Inventory (JSON)"
   - Each button calls the corresponding endpoint and triggers browser download.

#### New Dependencies (Python)
- `reportlab` or `weasyprint` (PDF generation)
- `trimesh` (GLB export)

#### Success Criteria
- Download a PDF from the browser after scanning a room.
- Open the `.glb` in Blender and see the point cloud.

---

### Part 16: One-Command Setup & Demo

**Real-world impact**: A project that takes 45 minutes to set up gets no users.
This part reduces time-to-first-result to under 3 minutes for a new user.

**Target**: `docker compose up` → open browser → upload a video → get results.
No Rust toolchain, no Python environment, no Ollama setup required.

#### Deliverables

1. **Consolidated Docker setup**
   - `docker/docker-compose.yml`: single `atlas-api` service that bundles Python
     API + frontend + optional Ollama sidecar.
   - Pre-built image on GitHub Container Registry (via CI).
   - `ATLAS_VLM_PROVIDER=claude` env var path: no Ollama needed, just an API key.

2. **Real demo footage**
   - Record a real 60-second room walkthrough video.
   - Process it through Atlas-0, capture the output (point cloud, labeled objects,
     risk report).
   - Ship this as `demo/sample_room.mp4` + `demo/expected_output.json` for CI
     regression testing.

3. **Interactive demo mode**
   - If no camera and no upload: serve the pre-processed demo scene automatically.
   - User sees real data, real risk scores, real point cloud — not fake hardcoded
     objects.

4. **Setup script** (`scripts/setup.sh`)
   - Detects OS, checks prerequisites (Docker only), pulls image, starts stack.
   - Prints: "Atlas-0 is running at http://localhost:8420"
   - Total time from zero to running: < 3 minutes.

5. **Update README**
   - Lead with the use case, not the architecture.
   - Quick-start: 3 commands.
   - Screenshot / GIF of the real demo scene.
   - Link to hosted live demo (if applicable).

#### Phase 4 Exit Criteria
- [ ] User uploads an MP4 video, gets labeled objects + risk report. No terminal.
- [ ] `ATLAS_VLM_PROVIDER=claude` works with just an API key.
- [ ] PDF report downloads from the browser.
- [ ] `docker compose up` is the only setup step.
- [ ] README leads with a GIF of the real demo, not architecture jargon.
- [ ] A new user can go from zero to useful output in < 5 minutes.

---

## Cross-Cutting Concerns (All Phases)

### Performance Budgets
| Stage | Budget | Measured By |
|---|---|---|
| Frame capture + preprocess | < 5ms | `tracing` span |
| Feature extraction + matching | < 8ms | `tracing` span |
| Pose estimation | < 3ms | `tracing` span |
| Gaussian update (per keyframe) | < 20ms | `criterion` bench |
| IPC Rust→Python | < 5ms | integration test |
| VLM inference — Ollama local | < 2000ms | Ollama timing |
| VLM inference — Claude/OpenAI | < 5000ms | httpx timing |
| Video frame extraction (per frame) | < 100ms | pytest-benchmark |
| Physics simulation (full scene) | < 10ms | `criterion` bench |
| API query response | < 200ms | pytest-benchmark |
| WebSocket risk push | < 50ms | `tracing` span |
| PDF report generation | < 3000ms | pytest-benchmark |

### Dependency Summary
| Phase | Rust Additions | Python Additions |
|---|---|---|
| Phase 1 | `nokhwa`, `ort`, `winit`, `wgpu`, `criterion` | — |
| Phase 2 | `prost-build` (build) | `scikit-learn`, `httpx` |
| Phase 3 | — | `prometheus-client` |
| Phase 4 | `ffmpeg-next` (optional feature) | `av` (opt), `anthropic` (opt), `openai` (opt), `reportlab` (opt), `trimesh` (opt) |

### Risk Register
| Risk | Impact | Mitigation |
|---|---|---|
| Real-time 3DGS too slow on CPU | Blocks Phase 1 | Start with sparse map; GPU path (WGPU/CUDA) as stretch goal |
| VLM hallucinations on material properties | Degrades risk accuracy | Confidence thresholds; multi-angle consensus; prefer Claude/GPT-4V |
| Monocular depth estimation inaccurate | Degrades 3D accuracy | Use known-scale objects for calibration; RGBD camera as option |
| Physics sim divergence | Incorrect risk predictions | Cap simulation steps; Baumgarte correction; validate on known scenarios |
| Shared memory race conditions | Data corruption at IPC boundary | Double-buffer with atomic flags; integration tests |
| PyAV / ffmpeg not available | Video upload fails silently | Clear error message with install instructions; graceful degradation to image-only |
| Cloud VLM API costs | Blocks adoption | Ollama always available as free fallback; clear cost docs |

---

## File Creation & Modification Map

### Phase 4 — New Files
```
python/atlas/utils/video.py              # PyAV video frame extractor
python/atlas/vlm/providers/__init__.py   # provider factory + exports
python/atlas/vlm/providers/base.py       # VLMProvider protocol
python/atlas/vlm/providers/ollama_provider.py
python/atlas/vlm/providers/anthropic_provider.py
python/atlas/vlm/providers/openai_provider.py
python/atlas/api/export.py               # PDF, GLB, JSON export endpoints
python/tests/test_providers.py           # provider factory + mock tests
python/tests/test_video.py               # video extraction tests
demo/sample_room.mp4                     # real demo footage
demo/expected_output.json                # regression baseline
scripts/setup.sh                         # one-command setup script
```

### Phase 4 — Modified Files
```
python/atlas/vlm/inference.py            # use provider abstraction
python/atlas/utils/config.py             # add provider, claude_model, openai_model fields
python/atlas/api/server.py               # real video processing, export endpoint mounts
configs/default.toml                     # add vlm.provider, vlm.claude_model, etc.
pyproject.toml                           # add video/claude/openai optional dep groups
crates/atlas-stream/Cargo.toml           # add optional ffmpeg-next feature
frontend/index.html                      # add export buttons
docker/docker-compose.yml                # add Ollama sidecar option
README.md                                # rewrite lead section, quick-start
```
