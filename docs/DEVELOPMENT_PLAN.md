# Atlas-0: Detailed Phase Development Plan

> Internal reference document. Maps every deliverable, file, module, and
> integration point needed to take Atlas-0 from scaffolding to production.

---

## Current State (Post-Initialization)

What exists today:
- Rust workspace with 4 crates (`atlas-core`, `atlas-slam`, `atlas-stream`, `atlas-physics`) — all compile, 8 unit tests pass, zero clippy warnings.
- Python package with VLM engine, world model agent, and FastAPI server — all scaffolded with TODOs marking every implementation point.
- Protobuf schema (`proto/atlas.proto`) defining the Rust<->Python IPC contract.
- CI pipeline, Docker setup, config system, and project infrastructure.

What does NOT exist yet (the actual work):
- No real camera capture or video decoding.
- No feature extraction, pose estimation, or Gaussian splatting.
- No VLM inference or Ollama integration.
- No physics simulation logic.
- No shared memory IPC between Rust and Python.
- No AR overlay or frontend.

---

## PHASE 1: THE EYE (Parts 1–4)

### Goal
Walk around a room with a camera. See a high-fidelity 3D Gaussian Splatting reconstruction generate on screen in < 100ms latency. The SLAM system tracks camera pose and builds the map in real-time.

### Why This Phase First
Everything downstream (semantic labeling, physics simulation, risk prediction) requires a working 3D spatial representation. Without the Eye, there is nothing to reason about.

---

### Part 1: Camera Ingestion Pipeline

**Objective**: Capture live video frames at 60fps and distribute them through the pipeline with minimal latency.

#### Deliverables

1. **Camera capture backend** (`crates/atlas-stream/src/capture.rs`)
   - Integrate `nokhwa` crate (cross-platform camera access) or `v4l2` on Linux.
   - Implement `CameraCapture` struct: opens device, captures frames at target FPS.
   - Convert raw camera buffers (YUYV/MJPEG) to RGB using `image` crate.
   - Push `Frame` structs into the `FramePipeline` bounded channel.
   - Handle camera disconnection, timeout, and format negotiation errors.

2. **Video file source** (`crates/atlas-stream/src/file_source.rs`)
   - Accept a video file path (MP4/MKV) for offline testing and benchmarking.
   - Decode frames using `ffmpeg-next` crate bindings.
   - Simulate real-time playback by pacing frame emission to match target FPS.
   - This is critical for reproducible development — most SLAM work will use recorded sequences.

3. **Frame preprocessing** (`crates/atlas-stream/src/preprocess.rs`)
   - Resize frames to SLAM working resolution (configurable, default 640x480 for speed).
   - Convert to grayscale for feature extraction (keep color copy for Gaussian color).
   - Undistort frames using camera intrinsics (pinhole + radial distortion model).
   - All operations must be zero-copy where possible using `image::ImageBuffer` views.

4. **Camera calibration loader** (`crates/atlas-stream/src/calibration.rs`)
   - Load camera intrinsics (fx, fy, cx, cy) and distortion coefficients from TOML config.
   - Add `[camera]` section to `configs/default.toml`.
   - Support standard calibration formats (OpenCV-compatible).

5. **Config loading** (`crates/atlas-stream/src/config.rs` — extend existing)
   - Wire up TOML config loading from `configs/default.toml`.
   - Add `ATLAS_` env var override support.

#### New Dependencies (Rust)
- `nokhwa` (camera capture) or `eye` crate
- `ffmpeg-next` (video file decoding)

#### Tests
- Unit: Frame capture mock, format conversion correctness.
- Benchmark: Frame capture + preprocessing latency (must be < 5ms per frame).

#### Success Criteria
- `cargo run --example capture_demo` opens camera, shows FPS counter, frames flow through pipeline.

---

### Part 2: Feature Extraction & Visual Odometry

**Objective**: Extract visual features from each frame and estimate camera motion between consecutive frames.

#### Deliverables

1. **Feature extractor** (`crates/atlas-slam/src/features.rs`)
   - Implement ORB (Oriented FAST and Rotated BRIEF) feature detection.
   - Alternatively, integrate a Rust binding to OpenCV's feature pipeline if pure-Rust is too slow.
   - Extract up to 1000 keypoints per frame with descriptors.
   - Consider `cv` or `imageproc` crates for pure-Rust path; fallback to `opencv` crate.
   - Must complete in < 5ms for 640x480 grayscale.

2. **Feature matching** (`crates/atlas-slam/src/matching.rs`)
   - Brute-force Hamming distance matching for ORB descriptors.
   - Apply ratio test (Lowe's ratio = 0.75) to filter ambiguous matches.
   - RANSAC-based outlier rejection using fundamental/essential matrix estimation.
   - Return a set of verified inlier correspondences.

3. **Pose estimation** (`crates/atlas-slam/src/pose_estimation.rs`)
   - From 2D-2D correspondences (initial case): Compute essential matrix → decompose into R|t.
   - From 3D-2D correspondences (after map exists): PnP solver (P3P + RANSAC).
   - Use `nalgebra` for all matrix/quaternion operations.
   - Output a `Pose` (position + rotation) relative to the world frame.

4. **Visual odometry pipeline** (`crates/atlas-slam/src/tracker.rs` — implement existing TODOs)
   - Wire feature extraction → matching → pose estimation into `process_frame()`.
   - Maintain a reference frame for tracking.
   - Detect tracking loss (too few inlier matches) and trigger relocalization.

#### New Dependencies (Rust)
- `opencv` crate (if pure-Rust feature extraction is insufficient)
- `rand` (RANSAC sampling)

#### Tests
- Unit: Feature extraction on synthetic images, known-pose recovery from synthetic correspondences.
- Integration: Process a sequence of frames, verify pose trajectory is roughly correct.
- Benchmark: Feature extraction + matching < 8ms total per frame.

#### Success Criteria
- Process a 30-second video clip, output a camera trajectory that roughly matches ground truth.

---

### Part 3: 3D Gaussian Splatting Reconstruction

**Objective**: Build and incrementally update a 3D Gaussian map from tracked frames.

#### Deliverables

1. **Depth estimation** (`crates/atlas-slam/src/depth.rs`)
   - Monocular depth estimation using a lightweight neural network (MiDaS small / Depth Anything V2 small).
   - Alternatively: stereo matching if stereo camera is available.
   - Run inference via ONNX Runtime (`ort` crate) for the depth model.
   - Convert depth map + camera intrinsics → 3D point cloud per frame.

2. **Gaussian initialization** (`crates/atlas-slam/src/gaussian_init.rs`)
   - From a point cloud, initialize a Gaussian per point.
   - Set initial covariance based on local point density.
   - Set color from the source pixel RGB value.
   - Set initial opacity to 0.5 (will be optimized).
   - Set scale from nearest-neighbor distance.

3. **Gaussian optimization** (`crates/atlas-slam/src/optimizer.rs`)
   - Implement differentiable Gaussian splatting rendering (forward pass).
   - Compute photometric loss between rendered and actual frame.
   - Backpropagate gradients to Gaussian parameters (position, covariance, color, opacity).
   - Use Adam optimizer for parameter updates.
   - This is the core of 3DGS — consider using `burn` crate or a custom CUDA kernel if CPU is too slow.

4. **Keyframe management** (`crates/atlas-slam/src/keyframe.rs`)
   - Decide when to insert a keyframe based on translation/rotation thresholds (from config).
   - Maintain a keyframe graph with co-visibility information.
   - Trigger Gaussian addition only on keyframes (not every frame).
   - Prune low-opacity Gaussians periodically to stay under `max_gaussians`.

5. **Map data structure** (`crates/atlas-core/src/gaussian.rs` — extend `GaussianCloud`)
   - Add spatial indexing (octree or k-d tree) for efficient region queries.
   - Support serialization for map saving/loading.
   - Add methods: `query_region(bbox) -> Vec<&Gaussian3D>`, `prune(min_opacity)`, `merge_near(threshold)`.

#### New Dependencies (Rust)
- `ort` (ONNX Runtime for depth model inference)
- `burn` or custom implementation for differentiable rendering

#### Tests
- Unit: Gaussian initialization from known point cloud, rendering correctness.
- Integration: Process 10 keyframes, verify Gaussian count grows and PSNR improves.
- Benchmark: Single optimization iteration < 10ms for 100K Gaussians.

#### Success Criteria
- Feed a recorded video, get a recognizable 3D Gaussian reconstruction.

---

### Part 4: Real-time Visualization & Integration

**Objective**: Tie the full pipeline together. Visualize the 3D reconstruction in real-time as the camera moves.

#### Deliverables

1. **3DGS renderer** (`crates/atlas-slam/src/renderer.rs`)
   - Implement tile-based Gaussian splatting renderer.
   - Sort Gaussians by depth, splat to screen using 2D Gaussian projection.
   - Alpha compositing front-to-back.
   - Target: render 500K Gaussians at 30fps on CPU (GPU path is a stretch goal).

2. **Visualization window** (`crates/atlas-slam/src/viewer.rs` or separate binary)
   - Open a window using `winit` + `wgpu` (or `minifb` for simplicity).
   - Display: original camera frame | depth estimate | Gaussian splat render (side by side).
   - Overlay: current pose, FPS counter, Gaussian count, tracking status.

3. **End-to-end pipeline binary** (`crates/atlas-slam/src/main.rs` or `examples/live_slam.rs`)
   - Wire: Camera → Preprocess → Feature Extract → Track → Keyframe Check → Gaussian Update → Render.
   - All on separate threads with crossbeam channels between stages.
   - Measure and log per-stage latency using `tracing` spans.

4. **Shared memory bridge scaffold** (`crates/atlas-core/src/shared_mem.rs`)
   - Design the memory-mapped buffer layout for Rust→Python data sharing.
   - Implement writer side in Rust: serialize `GaussianCloud` snapshots to mmap.
   - This prepares the foundation for Phase 2 Python integration.

5. **Performance profiling & optimization**
   - Profile with `cargo flamegraph`.
   - Identify and fix bottlenecks to hit < 100ms end-to-end latency.
   - Add `criterion` benchmarks for the hot path.

#### New Dependencies (Rust)
- `winit` + `wgpu` (or `minifb`) for visualization
- `criterion` for benchmarks

#### Tests
- Integration: Full pipeline test with a recorded sequence — verify no panics, output is reasonable.
- Benchmark: End-to-end latency < 100ms on target hardware.

#### Phase 1 Exit Criteria
- [ ] Camera captures frames at 60fps.
- [ ] SLAM tracks camera pose continuously (no tracking loss on smooth motion).
- [ ] 3D Gaussian map contains 50K+ Gaussians after 30s of mapping.
- [ ] Visualization shows recognizable 3D reconstruction.
- [ ] End-to-end latency from frame capture to display < 100ms.
- [ ] All existing + new tests pass. CI green.

---

## PHASE 2: THE BRAIN (Parts 5–8)

### Goal
Feed shards of the 3D Gaussian map into a local VLM. Query the map with natural language ("Where is the most unstable object?") and get a correct spatial coordinate back. Every object in the scene has semantic metadata: mass, friction, fragility, material, relationships.

### Why This Phase Second
The Eye gives us geometry (where things are). The Brain gives us semantics (what things are, what they're made of, how they relate). Physics simulation in Phase 3 needs both.

---

### Part 5: Rust→Python IPC Bridge

**Objective**: Establish the real-time data bridge between the Rust spatial engine and the Python reasoning layer.

#### Deliverables

1. **Shared memory writer (Rust)** (`crates/atlas-core/src/shared_mem.rs` — implement)
   - Memory-map a file (or anonymous mmap) using `memmap2`.
   - Define a binary layout: header (frame_id, timestamp, gaussian_count, camera_pose) + Gaussian array.
   - Write map snapshots at a configurable rate (e.g., every 5th keyframe).
   - Use a double-buffer or ring-buffer pattern to avoid read/write contention.

2. **Shared memory reader (Python)** (`python/atlas/utils/shared_mem.py`)
   - `mmap` the same file from Python.
   - Parse the binary header and Gaussian array using `numpy` for zero-copy access.
   - Expose `get_latest_snapshot() -> MapSnapshot` API.

3. **Protobuf IPC for structured messages** (`proto/atlas.proto` — already defined)
   - Compile protobuf definitions for both Rust (`prost-build`) and Python (`protobuf`).
   - Add `build.rs` to atlas-core for protobuf compilation.
   - Use protobuf for semantic labels, risk assessments, and queries (not frame data).

4. **IPC integration test** (`tests/integration/test_ipc.py`)
   - Rust writes a known map snapshot → Python reads and verifies data integrity.
   - Measure IPC latency (target < 1ms for snapshot transfer).

#### New Dependencies
- Rust: `prost-build` (build dependency)
- Python: `numpy`, `protobuf` (already in pyproject.toml)

#### Tests
- Integration: Round-trip data integrity test (Rust write → Python read → verify).
- Benchmark: Snapshot transfer latency.

---

### Part 6: VLM Integration & Semantic Labeling

**Objective**: Run a local VLM to understand what objects are in the scene and assign physical properties.

#### Deliverables

1. **Ollama client** (`python/atlas/vlm/ollama_client.py`)
   - Async HTTP client using `httpx` to communicate with Ollama API.
   - `check_model(name)` — verify model is pulled.
   - `pull_model(name)` — pull if missing.
   - `generate(prompt, image_b64, model)` — send inference request, stream response.
   - Robust error handling: connection refused, timeout, malformed response.

2. **VLM inference engine** (`python/atlas/vlm/inference.py` — implement existing TODOs)
   - Implement `VLMEngine.initialize()`: connect to Ollama, verify/pull model.
   - Implement `VLMEngine.label_region()`:
     - Receive a rendered image of a map region (from Gaussian splatting render).
     - Build a structured prompt asking for: object label, material, estimated mass, fragility, friction.
     - Parse VLM response into `SemanticLabel` dataclass.
   - Implement response parsing with fallback heuristics (VLMs don't always give clean JSON).

3. **Prompt engineering** (`python/atlas/vlm/prompts.py`)
   - Design prompts that extract structured physical metadata from the VLM.
   - Template: "You are a physics-aware scene analyzer. Describe the object in this image. Respond in JSON: {label, material, mass_kg, fragility_0_to_1, friction_0_to_1}."
   - Version prompts so we can A/B test different strategies.

4. **Region extractor** (`python/atlas/vlm/region_extractor.py`)
   - Given a `MapSnapshot`, identify distinct object regions.
   - Strategy: cluster Gaussians by spatial proximity (DBSCAN or connected components on the octree).
   - For each cluster, render a cropped view from the nearest keyframe camera pose.
   - Output: list of (region_image_bytes, region_bbox, gaussian_indices).

5. **Semantic label store** (`python/atlas/world_model/label_store.py`)
   - In-memory store mapping object_id → SemanticLabel.
   - Support updates (re-labeling as VLM sees objects from new angles).
   - Confidence-weighted update: new labels with higher confidence overwrite lower.
   - Serialize to protobuf `SemanticLabel` messages for IPC.

#### New Dependencies (Python)
- `ollama` Python client (or raw `httpx`)
- `scikit-learn` (DBSCAN clustering)

#### Tests
- Unit: Prompt construction, response parsing (mock VLM responses).
- Integration: Label a test image with Ollama running locally — verify output structure.
- Manual: Label objects in a test scene, visually verify correctness.

---

### Part 7: Spatial Query Engine

**Objective**: Answer natural language questions about the 3D scene by combining spatial data with semantic labels.

#### Deliverables

1. **Query parser** (`python/atlas/world_model/query_parser.py`)
   - Parse natural language queries into structured spatial operations.
   - Types of queries:
     - Location: "Where is the [object]?" → find object by label, return position.
     - Property: "What is the [object] made of?" → look up material property.
     - Spatial relation: "What is on top of the [object]?" → compute spatial relationships.
     - Risk: "What is the most unstable object?" → query risk assessments.
   - Use the VLM itself for query understanding (send query + scene description).

2. **Spatial relationship detector** (`python/atlas/world_model/relationships.py`)
   - Given labeled objects with bounding boxes, compute relationships:
     - "on top of": object A's bbox_min.y > object B's bbox_max.y (within threshold) AND horizontal overlap.
     - "inside": A's bbox fully within B's bbox.
     - "adjacent to": bboxes are within proximity threshold.
     - "supporting": A is below B and B would fall without A.
     - "leaning": A's center of mass is not above its support polygon.
   - Store relationships in `SemanticObject.relationships`.

3. **World model agent** (`python/atlas/world_model/agent.py` — implement existing TODOs)
   - Implement `_assess_scene()`:
     1. Get latest map snapshot from shared memory.
     2. Extract object regions via region extractor.
     3. Label new/changed regions via VLM.
     4. Compute spatial relationships.
     5. Run stability checks (basic physics heuristics before full simulation).
     6. Update risk list.
   - Rate-limit VLM calls to avoid overwhelming Ollama.

4. **API endpoints** (`python/atlas/api/server.py` — implement existing TODOs)
   - Wire `/health` to actual component status.
   - Implement `/query` endpoint:
     - Parse query → search label store → compute answer → return results with positions.
   - Add `/objects` endpoint: list all labeled objects with properties.
   - Add `/scene` endpoint: return full `SceneState` protobuf as JSON.

#### Tests
- Unit: Query parsing, spatial relationship detection on known object layouts.
- Integration: Full query pipeline with mocked VLM responses.
- API: FastAPI test client tests for all endpoints.

---

### Part 8: Integration, Benchmarking & Polish

**Objective**: Wire Phase 1 and Phase 2 together into a seamless system. Benchmark and optimize.

#### Deliverables

1. **End-to-end integration** (`scripts/run_atlas.py` or entry point)
   - Single command to start: Rust SLAM pipeline + Python world model + API server.
   - Process manager: start Rust binary, wait for shared memory to be ready, start Python.
   - Graceful shutdown: signal handling, resource cleanup.

2. **Config validation** (`python/atlas/utils/config.py`)
   - Load and validate `configs/default.toml` on startup.
   - Type-check all fields, report clear errors for missing/invalid values.
   - Support `ATLAS_*` env var overrides.

3. **Benchmarking suite** (`scripts/benchmark.py`)
   - Measure and report:
     - SLAM latency per frame (feature extraction, tracking, Gaussian update).
     - IPC transfer latency (Rust→Python snapshot).
     - VLM inference latency per region.
     - Spatial query response time.
   - Output results as JSON for CI tracking.

4. **Scene replay tool** (`scripts/replay.py`)
   - Record a mapping session (frames + poses + timestamps) to disk.
   - Replay for deterministic testing of the VLM/query pipeline.

5. **Documentation**
   - ADR-001: Choice of 3DGS-SLAM over traditional SLAM.
   - ADR-002: Shared memory IPC vs gRPC for Rust<->Python.
   - API documentation (auto-generated from FastAPI OpenAPI spec).

#### Phase 2 Exit Criteria
- [ ] VLM correctly labels at least 80% of common household objects.
- [ ] Spatial queries return correct positions (within 0.5m of ground truth).
- [ ] "Where is the most unstable object?" returns a reasonable answer.
- [ ] End-to-end query latency < 200ms (excluding VLM inference, which is bounded by model speed).
- [ ] IPC bridge transfers 500K Gaussian snapshots in < 5ms.
- [ ] All existing + new tests pass. CI green.

---

## PHASE 3: THE GHOST (Parts 9–12)

### Goal
If the AI detects a glass on a ledge, it "ghosts" a simulation of it falling — calculating the impact zone, probability, and risk level. Real-time AR overlay shows predicted failure paths. The system alerts before things go wrong.

### Why This Phase Last
Physics simulation requires both geometry (Phase 1) and semantics (Phase 2). You can't simulate a fall without knowing the object's mass, and you can't predict a spill without knowing the object contains liquid.

---

### Part 9: Physics Simulation Engine

**Objective**: Build a lightweight rigid-body physics simulator that can predict object trajectories.

#### Deliverables

1. **Rigid body representation** (`crates/atlas-physics/src/rigid_body.rs`)
   - `RigidBody` struct: position, velocity, angular_velocity, mass, inertia_tensor, bounding_shape.
   - Support shapes: sphere, box, convex hull (simplified from Gaussian clusters).
   - Convert `SemanticObject` (from Python via protobuf) into `RigidBody`.

2. **Collision detection** (`crates/atlas-physics/src/collision.rs`)
   - Broad phase: sweep-and-prune or spatial hash grid.
   - Narrow phase: GJK algorithm for convex shape intersection.
   - Contact point generation for collision response.
   - Detect floor, walls, and surfaces from the Gaussian map (plane extraction).

3. **Physics integrator** (`crates/atlas-physics/src/integrator.rs`)
   - Semi-implicit Euler integration (stable, fast, good enough for prediction).
   - Apply gravity, friction, normal forces, and restitution.
   - Constraint solver for resting contacts (prevent objects sinking through surfaces).
   - Configurable timestep and max steps from `PhysicsConfig`.

4. **Simulator** (`crates/atlas-physics/src/simulator.rs` — implement existing TODOs)
   - Implement `assess_risks()`:
     1. For each semantic object, create a RigidBody.
     2. Apply a small perturbation (nudge, vibration, slight push).
     3. Step simulation forward until object comes to rest or exits bounds.
     4. If object moved significantly → it's at risk. Record trajectory.
     5. Classify risk: Fall, Spill, Collision, TripHazard, Instability.
     6. Calculate impact point and probability.

5. **Surface/plane extraction** (`crates/atlas-physics/src/surfaces.rs`)
   - Extract supporting surfaces (tables, floors, shelves) from the Gaussian map.
   - RANSAC plane fitting on Gaussian center points.
   - Classify planes: horizontal (floor/table), vertical (wall), angled (ramp).
   - These form the collision environment for simulation.

#### New Dependencies (Rust)
- `parry3d` (collision detection library from the Rapier physics ecosystem) — or implement from scratch for learning value.

#### Tests
- Unit: Ball drop test (known height → known impact time), resting object stability.
- Integration: Place virtual objects in simulated scene, verify physics predictions.
- Benchmark: Simulation of 50 objects for 1000 steps < 10ms.

---

### Part 10: Risk Prediction Pipeline

**Objective**: Continuously run physics simulations in the background and maintain a ranked list of physical risks in the scene.

#### Deliverables

1. **Risk assessment loop** (extend `crates/atlas-physics/src/simulator.rs`)
   - Background thread that continuously re-evaluates scene risks.
   - Triggered by: new objects detected, objects moved, scene change.
   - Priority queue: simulate highest-risk objects first (objects on edges, leaning, top-heavy).

2. **Perturbation strategies** (`crates/atlas-physics/src/perturbations.rs`)
   - Define a set of realistic perturbations to test:
     - Gravity only (is the object stable as-is?).
     - Small horizontal push (simulating bump/vibration).
     - Surface tilt (simulating uneven/shaky surface).
     - Removal of supporting object (what if the shelf breaks?).
   - Each perturbation produces an independent risk score.
   - Final risk = weighted combination of perturbation results.

3. **Trajectory prediction** (`crates/atlas-physics/src/trajectory.rs`)
   - Record full trajectory of simulated objects (position at each timestep).
   - Identify impact point (where object lands after falling).
   - Calculate impact energy (mass × velocity² / 2) for damage estimation.
   - Predict spill zone for liquid-containing objects (cone projection from tip-over point).

4. **Risk→Python bridge**
   - Serialize `RiskAssessment` results via protobuf.
   - Write to shared memory or a dedicated risk channel.
   - Python world model agent reads and serves via API.

5. **Python risk aggregator** (`python/atlas/world_model/risk_aggregator.py`)
   - Receive risk assessments from Rust physics engine.
   - Merge with VLM-based heuristic risks (e.g., "knife near edge" detected by VLM).
   - Rank by combined probability × severity.
   - Maintain a top-N risk list for the API and WebSocket stream.

#### Tests
- Unit: Perturbation strategies produce expected force vectors.
- Integration: Known unstable scene → verify correct risks are identified and ranked.

---

### Part 11: AR Overlay & Frontend

**Objective**: Build a real-time AR overlay that visualizes the AI's predictions — showing risk zones, predicted trajectories, and alerts.

#### Deliverables

1. **WebSocket risk stream** (`python/atlas/api/server.py` — implement `/ws/risks`)
   - Stream risk updates as JSON via WebSocket.
   - Only send deltas (new/changed/removed risks) to minimize bandwidth.
   - Include: risk_type, object_label, position, probability, impact_point, trajectory_points.

2. **Web-based AR overlay** (`frontend/` — new directory)
   - Simple web app (vanilla JS + Three.js or Babylon.js).
   - Connect to WebSocket for risk updates.
   - Render:
     - Semi-transparent red zones around at-risk objects.
     - Dotted trajectory arcs showing predicted fall paths.
     - Impact zones highlighted on the floor/surface.
     - Alert badges with risk descriptions ("Trip hazard: cable near walkway").
   - Camera feed background with overlay composited on top.

3. **Overlay data format** (`python/atlas/api/overlay.py`)
   - Convert internal risk data into renderable overlay primitives:
     - `RiskZone`: center, radius, color, opacity (for highlighting).
     - `TrajectoryArc`: list of 3D points for the predicted path.
     - `ImpactZone`: polygon on the impact surface.
     - `Alert`: text, severity level, screen position.
   - Project 3D coordinates to 2D screen space using current camera pose.

4. **Mobile/tablet viewer** (stretch goal)
   - Responsive web app that works on mobile browsers.
   - Use device camera + AR.js or WebXR for true AR experience.

#### New Dependencies
- Frontend: Three.js (via CDN), WebSocket client.
- Python: No new deps (FastAPI already handles WebSocket).

#### Tests
- Unit: Overlay data projection (known 3D point → expected 2D position).
- Manual: Visual verification of overlay rendering on test scenes.

---

### Part 12: Integration, Optimization & Production Hardening

**Objective**: Full system integration test. Optimize for real-time performance. Harden for reliability.

#### Deliverables

1. **Full pipeline integration**
   - Camera → SLAM → Shared Memory → VLM Labeling → Physics Sim → Risk Stream → AR Overlay.
   - Single `docker-compose up` starts the entire system.
   - Health checks on every component with automatic restart on failure.

2. **Performance optimization**
   - Profile full pipeline end-to-end.
   - GPU acceleration for Gaussian splatting (WGPU compute shaders or CUDA).
   - Batch VLM inference (multiple regions in one request if model supports it).
   - Physics simulation on a dedicated thread pool (`rayon`).

3. **Graceful degradation**
   - If VLM is slow: show geometry without semantic labels.
   - If physics sim is behind: show last-known risks with staleness indicator.
   - If camera drops frames: interpolate pose, don't crash.
   - If tracking is lost: show warning, attempt relocalization.

4. **Logging, metrics & observability**
   - Structured logging throughout (already using `tracing` in Rust, `structlog` in Python).
   - Expose Prometheus metrics: frame rate, SLAM latency, VLM latency, Gaussian count, risk count.
   - Add `/metrics` endpoint to FastAPI.

5. **Demo recording tool** (`scripts/record_demo.py`)
   - Record a full session: camera feed + SLAM output + risk predictions + overlay.
   - Export as video for portfolio showcase.

6. **Final documentation**
   - README.md with project overview, setup instructions, demo GIF.
   - Architecture diagram (component → component data flow).
   - Performance report with benchmark numbers.

#### Phase 3 Exit Criteria
- [ ] Physics simulation correctly predicts object falls (ball drop test, tipping test).
- [ ] Risk prediction identifies at least 3 types of risks in a cluttered scene.
- [ ] AR overlay renders risk zones and trajectories in real-time.
- [ ] WebSocket streams updates with < 100ms latency from detection to display.
- [ ] Full pipeline runs stable for 10+ minutes without crash.
- [ ] Docker deployment works out of the box.
- [ ] All tests pass. CI green.

---

## Cross-Cutting Concerns (All Phases)

### Performance Budgets (Enforced from Phase 1)
| Stage | Budget | Measured By |
|---|---|---|
| Frame capture + preprocess | < 5ms | `tracing` span |
| Feature extraction + matching | < 8ms | `tracing` span |
| Pose estimation | < 3ms | `tracing` span |
| Gaussian update (per keyframe) | < 20ms | `criterion` bench |
| IPC Rust→Python | < 5ms | integration test |
| VLM inference (per region) | < 2000ms | Ollama timing |
| Physics simulation (full scene) | < 10ms | `criterion` bench |
| API query response | < 200ms | `pytest-benchmark` |
| WebSocket risk push | < 50ms | `tracing` span |

### Dependency Summary
| Phase | Rust Additions | Python Additions |
|---|---|---|
| Phase 1 | `nokhwa`/`eye`, `ffmpeg-next`, `ort`, `winit`, `wgpu`, `criterion` | — |
| Phase 2 | `prost-build` (build) | `scikit-learn`, `ollama` |
| Phase 3 | `parry3d` (or manual collision) | Three.js (frontend) |

### Risk Register
| Risk | Impact | Mitigation |
|---|---|---|
| Real-time 3DGS too slow on CPU | Blocks Phase 1 | Start with sparse map; add GPU path (WGPU/CUDA) early |
| VLM hallucinations on material properties | Degrades Phase 2 quality | Confidence thresholds; multi-angle consensus; prompt iteration |
| Monocular depth estimation inaccurate | Degrades 3D accuracy | Use known-scale objects for calibration; consider RGBD camera |
| Physics sim divergence | Incorrect Phase 3 predictions | Cap simulation steps; use stable integrator; validate against known scenarios |
| Shared memory race conditions | Data corruption at IPC boundary | Double-buffer with atomic flags; thorough integration tests |

---

## File Creation & Modification Map

### Phase 1 — New Files
```
crates/atlas-stream/src/capture.rs
crates/atlas-stream/src/file_source.rs
crates/atlas-stream/src/preprocess.rs
crates/atlas-stream/src/calibration.rs
crates/atlas-slam/src/features.rs
crates/atlas-slam/src/matching.rs
crates/atlas-slam/src/pose_estimation.rs
crates/atlas-slam/src/depth.rs
crates/atlas-slam/src/gaussian_init.rs
crates/atlas-slam/src/optimizer.rs
crates/atlas-slam/src/keyframe.rs
crates/atlas-slam/src/renderer.rs
crates/atlas-slam/src/viewer.rs
crates/atlas-core/src/shared_mem.rs (scaffold)
examples/capture_demo.rs
examples/live_slam.rs
```

### Phase 1 — Modified Files
```
crates/atlas-stream/src/lib.rs (add new modules)
crates/atlas-stream/src/config.rs (add camera config)
crates/atlas-stream/Cargo.toml (new deps)
crates/atlas-slam/src/lib.rs (add new modules)
crates/atlas-slam/src/tracker.rs (implement TODOs)
crates/atlas-slam/Cargo.toml (new deps)
crates/atlas-core/src/gaussian.rs (add spatial indexing)
crates/atlas-core/src/lib.rs (re-export new types)
configs/default.toml (add camera section)
Cargo.toml (workspace deps)
```

### Phase 2 — New Files
```
crates/atlas-core/src/shared_mem.rs (implement)
crates/atlas-core/build.rs (protobuf compilation)
python/atlas/utils/shared_mem.py
python/atlas/vlm/ollama_client.py
python/atlas/vlm/prompts.py
python/atlas/vlm/region_extractor.py
python/atlas/world_model/label_store.py
python/atlas/world_model/query_parser.py
python/atlas/world_model/relationships.py
python/atlas/utils/config.py
tests/integration/test_ipc.py
scripts/benchmark.py
scripts/replay.py
scripts/run_atlas.py
docs/architecture/ADR-001-3dgs-slam.md
docs/architecture/ADR-002-ipc-design.md
```

### Phase 2 — Modified Files
```
python/atlas/vlm/inference.py (implement TODOs)
python/atlas/world_model/agent.py (implement TODOs)
python/atlas/api/server.py (implement TODOs)
pyproject.toml (new deps)
```

### Phase 3 — New Files
```
crates/atlas-physics/src/rigid_body.rs
crates/atlas-physics/src/collision.rs
crates/atlas-physics/src/integrator.rs
crates/atlas-physics/src/surfaces.rs
crates/atlas-physics/src/perturbations.rs
crates/atlas-physics/src/trajectory.rs
python/atlas/world_model/risk_aggregator.py
python/atlas/api/overlay.py
frontend/index.html
frontend/js/app.js
frontend/js/overlay.js
scripts/record_demo.py
```

### Phase 3 — Modified Files
```
crates/atlas-physics/src/lib.rs (add new modules)
crates/atlas-physics/src/simulator.rs (implement TODOs)
crates/atlas-physics/Cargo.toml (new deps)
python/atlas/api/server.py (WebSocket implementation)
docker/Dockerfile (add frontend)
docker/docker-compose.yml (add frontend service)
```
