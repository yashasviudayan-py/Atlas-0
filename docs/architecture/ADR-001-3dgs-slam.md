# ADR-001: 3D Gaussian Splatting SLAM over Traditional Feature-Based SLAM

**Status**: Accepted  
**Date**: 2026-03-31  
**Authors**: Atlas-0 core team  
**Phase**: Phase 1 (The Eye)

---

## Context

Atlas-0 needs a real-time 3D spatial representation of an indoor scene built from a live camera feed. This representation must support:

1. **Dense, photorealistic geometry** — semantic labeling and risk assessment require recognising objects, not just sparse point clouds.
2. **Continuous incremental updates** — new regions of the scene are explored frame-by-frame; the map must grow without full re-computation.
3. **Low-latency access** — the Python reasoning layer needs to query the map every ~1 second; the representation must be memory-friendly for IPC.
4. **Novel-view rendering** — the world-model agent renders cropped object views for VLM inference; synthesising views from arbitrary camera poses is required.

Two candidate approaches were evaluated:

| Approach | Representative Systems |
|---|---|
| **Feature-based sparse SLAM** | ORB-SLAM3, VINS-Mono |
| **3D Gaussian Splatting SLAM** | SplaTAM, MonoGS, Gaussian Splatting (3DGS) |

---

## Decision

We adopt **3D Gaussian Splatting (3DGS)** as the scene representation, integrated with a visual odometry frontend for camera tracking.

The system tracks camera pose using classical ORB feature matching (lightweight, < 5ms per frame) and uses depth estimation + Gaussian optimisation to build a dense photorealistic map.

---

## Rationale

### Why 3DGS beats traditional sparse SLAM for Atlas-0

**Dense photorealistic output by default.**  
Traditional SLAM produces sparse landmark clouds (ORB-SLAM3: ~2 000 landmarks for a room). VLM labeling needs to *see* objects — synthesising recognisable views from a sparse map requires an additional meshing or densification step that adds complexity and latency. 3DGS represents the scene as a collection of learnable Gaussians that can be rendered to any viewpoint directly, without a separate densification stage.

**Differentiable rendering enables self-supervised refinement.**  
3DGS can minimise photometric reconstruction loss against incoming frames without requiring ground-truth depth or stereo cameras. This is critical for monocular indoor deployment where LiDAR is not available.

**Better suitability for novel-view synthesis.**  
The region extractor in Phase 2 needs to render object crops from the closest keyframe pose. 3DGS renders these in < 1ms per crop (tile-based rasterisation), whereas NeRF-style representations require iterative ray marching (10–500ms). Traditional mesh reconstruction cannot produce clean per-object crops without mesh segmentation.

**Memory-efficient IPC.**  
Each Gaussian is 28 bytes (x, y, z, opacity, r, g, b). A 100 K Gaussian map is ~2.8 MB — fast to transfer via mmap to Python. Traditional dense TSDF volumes for the same coverage are typically 100–500 MB.

### Conceded trade-offs

| Trade-off | Mitigation |
|---|---|
| 3DGS optimisation is GPU-bound at high quality | Phase 1 starts with CPU path (~100 K Gaussians); GPU path (WGPU compute shaders) is targeted for Phase 3 optimisation |
| Monocular depth estimation accuracy | Calibrated camera intrinsics + scale recovery from known-size objects; RGBD camera path is an optional Phase 3 extension |
| Tracking loss in textureless scenes | ORB tracking falls back to direct alignment; ICP on Gaussian centers is a future fallback |
| Gaussian count grows unboundedly | Periodic opacity-based pruning (< 0.005 opacity threshold) + maximum cap (`max_gaussians` config) |

---

## Alternatives Considered

### ORB-SLAM3

**Pros**: Mature, widely benchmarked (TUM, EuRoC), handles loop closure elegantly.  
**Cons**: Sparse map inadequate for VLM region rendering; requires separate dense reconstruction pipeline; C++ library with complex build dependencies.

**Rejected** because the sparse point cloud cannot satisfy the novel-view synthesis requirement without an additional dense stage that adds ~50ms of latency.

### TSDF / KinectFusion-style Volumetric Mapping

**Pros**: Dense, straightforward integration, well-understood.  
**Cons**: Requires depth sensor (monocular is hard); TSDF volumes are large (100–500 MB) and slow to update; rendering views requires ray-marching or surface extraction.

**Rejected** due to depth sensor requirement and IPC memory overhead.

### Neural Radiance Fields (NeRF / Instant-NGP)

**Pros**: High-quality novel-view synthesis; implicit representation.  
**Cons**: Inference per-view is 10–500ms; online training from a streaming camera is extremely difficult; GPU requirement is strict.

**Rejected** because the 10–500ms rendering latency is incompatible with the 50ms SLAM→VLM propagation budget.

---

## Consequences

- `crates/atlas-slam` implements the ORB tracking + 3DGS optimisation pipeline.
- `crates/atlas-core/src/gaussian.rs` defines the `Gaussian3D` and `GaussianCloud` types shared across crates.
- The Python `RegionExtractor` can render object crops directly from `GaussianCloud` via the shared memory bridge.
- Phase 3 GPU acceleration path is architecturally straightforward: replace the CPU rasteriser with a WGPU compute shader over the same Gaussian data structure.

---

## References

- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023.
- Matsuki et al., "Gaussian Splatting SLAM", CVPR 2024.
- Mur-Artal & Tardós, "ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual–Inertial, and Multimap SLAM", T-RO 2021.
