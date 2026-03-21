# Atlas-0: Development Rules & Guidelines

## Project Overview
Atlas-0 is a Spatial Reasoning & Physical World-Model Engine that transforms 2D camera feeds into Semantic 3D Digital Twins capable of predicting physical consequences.

**Architecture**: Polyglot — Rust (spatial engine, performance-critical paths) + Python (ML/VLM reasoning, API layer).

## Repository Structure
```
crates/           # Rust workspace crates
  atlas-core/     # Shared types, traits, error handling
  atlas-slam/     # 3DGS-SLAM pipeline (camera pose, Gaussian splatting)
  atlas-physics/  # Causal physics simulation engine
  atlas-stream/   # Video ingestion & frame pipeline
python/atlas/     # Python package
  vlm/            # Vision-Language Model inference
  world_model/    # Semantic labeling & risk assessment loop
  api/            # FastAPI server for AR overlay & queries
  utils/          # Shared Python utilities
proto/            # Protobuf definitions for Rust<->Python IPC
configs/          # Runtime configuration files (TOML)
scripts/          # Dev scripts (setup, benchmarks, data prep)
docker/           # Container definitions
tests/integration # Cross-language integration tests
```

## Development Rules

### 1. Code Quality
- **Rust**: All code must pass `cargo clippy -- -D warnings` with zero warnings. Use `#[must_use]` on fallible functions. No `unwrap()` in library code — use proper error types via `thiserror`.
- **Python**: All code must pass `ruff check` and `ruff format`. Type hints are mandatory on all public functions. Use `pydantic` for data validation at system boundaries.
- **Both**: No dead code. No commented-out code in main branch. If it's not used, delete it.

### 2. Architecture Principles
- **Rust owns the hot path**: Frame ingestion, SLAM, physics simulation — anything that must run at 60fps or faster lives in Rust.
- **Python owns reasoning**: VLM inference, semantic labeling, world-model queries — anything that benefits from ML ecosystem lives in Python.
- **IPC via shared memory + protobuf**: Rust and Python communicate through memory-mapped buffers for frame data and protobuf messages for structured data. No REST between internal components.
- **Zero-copy where possible**: Use `Arc`, `Cow`, and memory-mapped I/O to avoid unnecessary allocations in the frame pipeline.

### 3. Error Handling
- Rust: Use `Result<T, E>` everywhere. Define domain-specific error enums per crate. Propagate with `?`. Log at the boundary.
- Python: Use structured exceptions inheriting from `AtlasError`. Never bare `except:`. Log with `structlog`.

### 4. Testing
- Unit tests live next to the code (`#[cfg(test)]` in Rust, `tests/` directories in Python).
- Integration tests in `tests/integration/` test Rust<->Python boundaries.
- Performance-sensitive code must have benchmarks (`criterion` for Rust, `pytest-benchmark` for Python).
- CI must pass before merge. No exceptions.

### 5. Performance Budgets
- Frame ingestion to SLAM update: < 16ms (60fps target)
- SLAM to semantic label propagation: < 50ms
- Physics simulation tick: < 10ms
- End-to-end query response ("where is the most unstable object?"): < 200ms

### 6. Configuration
- All runtime config in TOML files under `configs/`.
- No hardcoded paths, ports, model names, or thresholds in source code.
- Environment-specific overrides via `ATLAS_` prefixed env vars.

### 7. Git & Branching
- Branch naming: `phase-{N}/{feature-name}` (e.g., `phase-1/slam-pipeline`)
- Commit messages: imperative mood, < 72 chars first line. Reference phase in body.
- PRs require at least passing CI. Squash merge to main.

### 8. Dependencies
- Pin all dependency versions. No `*` or `>=` ranges.
- Rust: Use workspace dependencies in root `Cargo.toml`.
- Python: Use `uv` for dependency management. Lock file must be committed.
- Evaluate new dependencies carefully — prefer well-maintained crates/packages with permissive licenses (MIT/Apache-2.0).

### 9. Safety & Security
- No `unsafe` Rust without a `// SAFETY:` comment explaining the invariant.
- Sanitize all external input (camera streams, API queries, config files).
- No secrets in code or config files. Use env vars or a secrets manager.

### 10. Documentation
- Public Rust APIs must have doc comments with examples.
- Python public functions must have docstrings (Google style).
- Architecture decisions go in `docs/architecture/` as ADRs (Architecture Decision Records).

## Phase Development Plan

### Phase 1: The Eye (Weeks 1-4) — Real-time 3DGS-SLAM
Focus: Camera ingestion, pose estimation, Gaussian splatting reconstruction.
Target: Walk around a room, see 3D reconstruction < 100ms latency.

### Phase 2: The Brain (Weeks 5-8) — Semantic Understanding
Focus: VLM integration, spatial queries, metadata assignment.
Target: Query "where is the most unstable object?" and get a spatial coordinate.

### Phase 3: The Ghost (Weeks 9-12) — Physics Simulation
Focus: Physics engine integration, risk prediction, AR overlay.
Target: Real-time visual alerts for physical risks.
