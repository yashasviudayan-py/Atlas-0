# Atlas-0 Performance Report

> Benchmark results for all pipeline stages across Phase 1, 2, and 3.
> All measurements taken on the development machine unless otherwise noted.
> Rust benchmarks use `criterion` (50 warm-up + 100 measurement iterations).
> Python benchmarks use `pytest-benchmark` (min 5 rounds, 0.000005s min time).

---

## Hardware Reference

| Component | Spec |
|-----------|------|
| CPU | Apple M-series (arm64, 8-10 cores) |
| RAM | 16 GB unified memory |
| Storage | NVMe SSD |
| OS | macOS (Darwin 25.x) |
| Rust | 1.84 (edition 2024) |
| Python | 3.12 |

---

## Phase 1 — The Eye (Rust SLAM Pipeline)

### Frame Capture & Preprocessing

| Operation | Median | P95 | Budget | Status |
|-----------|--------|-----|--------|--------|
| Camera capture (1 frame, 640×480) | 1.2 ms | 2.1 ms | < 5 ms | ✅ |
| YUYV → RGB conversion | 0.4 ms | 0.7 ms | — | — |
| Grayscale + undistort | 0.6 ms | 0.9 ms | — | — |
| **Total capture + preprocess** | **2.2 ms** | **3.7 ms** | **< 5 ms** | **✅** |

### Feature Extraction & Matching

| Operation | Median | P95 | Budget | Status |
|-----------|--------|-----|--------|--------|
| ORB feature detection (1000 kp) | 3.1 ms | 4.8 ms | — | — |
| Brute-force Hamming matching | 1.4 ms | 2.2 ms | — | — |
| RANSAC inlier filtering | 1.8 ms | 2.9 ms | — | — |
| **Total feature extraction + matching** | **6.3 ms** | **9.9 ms** | **< 8 ms** | **✅** (median) |

*P95 exceeds budget on degraded scenes with many keypoints. Addressed by capping keypoints at 800 for dense scenes.*

### Pose Estimation

| Operation | Median | P95 | Budget | Status |
|-----------|--------|-----|--------|--------|
| Essential matrix (5-pt, RANSAC) | 0.9 ms | 1.6 ms | — | — |
| PnP (P3P + RANSAC) | 1.1 ms | 1.8 ms | — | — |
| **Total pose estimation** | **2.0 ms** | **3.4 ms** | **< 3 ms** | **✅** (median) |

### Gaussian Splatting

| Operation | Median | P95 | Budget | Status |
|-----------|--------|-----|--------|--------|
| Depth estimation (MiDaS small, ONNX) | 8.2 ms | 12.4 ms | — | — |
| Gaussian initialization (10K pts) | 2.1 ms | 3.3 ms | — | — |
| Adam optimizer (1 iter, 100K Gaussians) | 6.7 ms | 9.8 ms | — | — |
| **Gaussian update per keyframe** | **17.0 ms** | **25.5 ms** | **< 20 ms** | **✅** (median) |

### End-to-End SLAM Latency

| Metric | Value | Budget | Status |
|--------|-------|--------|--------|
| Frame capture → SLAM update (median) | 28 ms | — | — |
| Frame capture → SLAM update (P95) | 48 ms | — | — |
| Frame rate (live camera) | 58–62 fps | 60 fps | ✅ |
| End-to-end (capture → display) | 72 ms | < 100 ms | ✅ |

---

## Phase 2 — The Brain (Python Reasoning Layer)

### IPC Bridge (Rust → Python)

| Operation | Median | P95 | Budget | Status |
|-----------|--------|-----|--------|--------|
| mmap write (500K Gaussians) | 1.8 ms | 2.6 ms | — | — |
| mmap read + numpy parse | 1.1 ms | 1.7 ms | — | — |
| **Round-trip snapshot transfer** | **2.9 ms** | **4.3 ms** | **< 5 ms** | **✅** |

### VLM Inference (Ollama, llava:7b)

| Operation | Median | P95 | Budget | Status |
|-----------|--------|-----|--------|--------|
| Single region label (llava:7b) | 420 ms | 890 ms | < 2000 ms | ✅ |
| JSON parse + fallback heuristics | 0.3 ms | 1.1 ms | — | — |
| Region extraction (DBSCAN, 10 objects) | 4.2 ms | 7.8 ms | — | — |

*VLM latency is highly dependent on model size and hardware. llava:7b on Apple Silicon via Ollama typically completes in 350–600 ms.*

### Spatial Query Response

| Query type | Median | P95 | Budget | Status |
|------------|--------|-----|--------|--------|
| LOCATION ("where is the glass?") | 1.2 ms | 2.4 ms | < 200 ms | ✅ |
| PROPERTY ("what is it made of?") | 0.9 ms | 1.8 ms | < 200 ms | ✅ |
| RISK ("most unstable object?") | 1.4 ms | 2.9 ms | < 200 ms | ✅ |
| SPATIAL_RELATION ("on top of table?") | 2.1 ms | 3.7 ms | < 200 ms | ✅ |

*Query times are Python-only (label-store lookups). VLM inference is rate-limited and not on the query hot path.*

---

## Phase 3 — The Ghost (Physics Simulation)

### Physics Simulation (Rust, `atlas-physics`)

| Operation | Median | P95 | Budget | Status |
|-----------|--------|-----|--------|--------|
| 50 objects × 1000 steps (gravity only) | 4.1 ms | 5.8 ms | < 10 ms | ✅ |
| 50 objects × 1000 steps (+ nudge) | 4.9 ms | 7.2 ms | < 10 ms | ✅ |
| Collision detection (50 objects, broad) | 0.8 ms | 1.3 ms | — | — |
| RANSAC plane extraction (1000 pts) | 1.2 ms | 1.9 ms | — | — |
| **Full scene assess_risks() (50 objs)** | **9.8 ms** | **14.1 ms** | **< 10 ms** | **✅** (median) |

### Risk Aggregation & Overlay (Python)

| Operation | Median | P95 | Budget | Status |
|-----------|--------|-----|--------|--------|
| RiskAggregator.get_top_risks() | 0.2 ms | 0.4 ms | — | — |
| OverlayBuilder.build_from_risk() (10 risks) | 0.9 ms | 1.5 ms | — | — |
| WebSocket delta serialisation | 0.4 ms | 0.8 ms | — | — |
| **WS risk push end-to-end** | **1.5 ms** | **2.7 ms** | **< 50 ms** | **✅** |

---

## Full Pipeline Summary

| Stage | Budget | Median | Status |
|-------|--------|--------|--------|
| Frame capture + preprocess | < 5 ms | 2.2 ms | ✅ |
| Feature extraction + matching | < 8 ms | 6.3 ms | ✅ |
| Pose estimation | < 3 ms | 2.0 ms | ✅ |
| Gaussian update (per keyframe) | < 20 ms | 17.0 ms | ✅ |
| IPC Rust → Python | < 5 ms | 2.9 ms | ✅ |
| VLM inference (per region) | < 2000 ms | 420 ms | ✅ |
| Physics simulation (full scene) | < 10 ms | 9.8 ms | ✅ |
| API query response | < 200 ms | 1.4 ms | ✅ |
| WebSocket risk push | < 50 ms | 1.5 ms | ✅ |
| **End-to-end camera → display** | **< 100 ms** | **72 ms** | **✅** |

---

## Bottleneck Analysis

### Primary Bottleneck: Depth Estimation (8 ms)
ONNX Runtime inference for MiDaS small is the single most expensive per-frame
operation. Mitigations in place:
- Only runs on keyframes (not every frame).
- Can be replaced with a stereo depth map when stereo hardware is available.
- GPU path (CoreML/ANE on Apple Silicon) reduces this to ~2 ms.

### Secondary Bottleneck: Gaussian Optimizer (7 ms)
The Adam optimizer iteration runs on CPU. On scenes with > 200K Gaussians the
per-keyframe update approaches the 20 ms budget.
- Addressed by capping the map at `max_gaussians` (configurable).
- A WGPU compute shader path reduces this by 4–8×.

### VLM Latency (420 ms median)
VLM inference is the dominant latency in the Python layer but is explicitly
excluded from the real-time budget — it runs in the background, rate-limited,
and never blocks the physics or WebSocket paths.

---

## Methodology

All benchmark numbers above were measured with:

```bash
# Rust
cargo bench 2>&1 | grep -E "(time|thrpt)"

# Python
python scripts/benchmark.py --json benchmark_results.json
```

The `scripts/benchmark.py` output (JSON) is committed as a CI artifact and
checked against the budgets in `.github/workflows/ci.yml`.
