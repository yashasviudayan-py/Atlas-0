<p align="center">
  <img src="docs/assets/atlas0-logo.svg" alt="ATLAS-0 room safety scan hero" width="940" />
</p>

<p align="center">
  <a href="https://github.com/yashasviudayan-py/Atlas-0/actions/workflows/ci.yml">
    <img src="https://github.com/yashasviudayan-py/Atlas-0/actions/workflows/ci.yml/badge.svg" alt="CI status" />
  </a>
  <a href="docs/DEVELOPMENT_PLAN.md">
    <img src="https://img.shields.io/badge/status-beta%20MVP-0c8074" alt="Status: beta MVP" />
  </a>
  <a href="docs/DEVELOPMENT_PLAN.md">
    <img src="https://img.shields.io/badge/product-room%20safety%20scan-d67f30" alt="Product: room safety scan" />
  </a>
  <img src="https://img.shields.io/badge/deploy-preflight-125d69" alt="Deployment preflight" />
  <img src="https://img.shields.io/badge/python-ruff%20%7C%20pytest-3776AB?logo=python&logoColor=white" alt="Python checks" />
  <img src="https://img.shields.io/badge/rust-fmt%20%7C%20clippy%20%7C%20test-000000?logo=rust" alt="Rust checks" />
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-0a5f57" alt="License: MIT" />
  </a>
</p>

<h3 align="center">Upload one room walkthrough. Leave with a warm, evidence-backed Room Safety Brief.</h3>

<p align="center">
  <strong>ATLAS-0</strong> is a warm-trust room safety assistant for renters,
  parents, pet owners, and anyone who wants a practical second look before
  something falls, tips, spills, blocks a path, or becomes tomorrow's small
  household chaos.
</p>

<table>
  <tr>
    <td width="33%">
      <strong>Input</strong><br/>
      A 20-60 second phone walkthrough of a real room.
    </td>
    <td width="33%">
      <strong>Output</strong><br/>
      A Room Safety Brief with top actions, evidence frames, approximate
      locations, confidence, and fix-first recommendations.
    </td>
    <td width="33%">
      <strong>Share</strong><br/>
      A downloadable PDF report you can send to a landlord, partner,
      contractor, or care team.
    </td>
  </tr>
</table>

> Current promise: decision support, not safety certification. ATLAS-0 is
> intentionally honest about confidence, scan quality, and where the pipeline is
> still approximate.

The design direction is **Warm Trust** with a premium Safety Brief feel: calm
language, high contrast, visible uncertainty, polished scan guidance, and
practical next steps instead of demo theater or fake certainty.

## Current Standing

ATLAS-0 is now a self-hosted beta product, not a broad demo shell. The current
spec is:

**Upload a room walkthrough, get a trustworthy Room Safety Brief, fix one
thing, and optionally rescan to compare progress.**

What is strong today:
- The first-run path includes a sample report, scan playbooks, audience modes,
  Room Mystery Mode, file preflight, and capture guidance before upload.
- The optional Live Capture Coach checks lighting, steadiness, duration, floor
  path, corners, and approximate room coverage in the browser before upload.
- Uploads persist through a durable job/artifact pipeline, with detached-worker
  support and an IndexedDB retry queue for files selected while offline.
- The Safety Brief surfaces top actions, Calm Score, confidence/uncertainty,
  evidence frames, fix difficulty, Field Notes, approximate evidence maps, and
  deterministic "Ask this Safety Brief" answers.
- Privacy Receipt now shows room label, evidence inclusion, local blur/redaction
  previews, retention posture, delete controls, and decision-support wording
  before sharing/exporting.
- Same-room rescans produce before/after comparison data, including score delta,
  hazard delta, and compact previous/current evidence snapshots.
- Home Journal, Room Health Passport, Room Rituals, One Thing Today, Room Care
  Calendar, Fix Library, challenges, streaks, and weekly recaps are all local
  first and accountless.
- Trust Proof and operator views expose aggregate quality, funnel, storage,
  worker, feedback, and eval-corpus readiness signals without leaking private
  user data.
- Settings has become a local control center for theme, accessibility, report
  defaults, scan defaults, privacy/data clearing, backup/import, beta feedback,
  changelog, known limits, and replaying onboarding.
- The frontend identity uses the Warm Trust palette, editorial typography,
  polished report artifacts, responsive layouts, PWA manifest, and offline shell.
- Rust, Python, frontend smoke, deployment preflight, benchmark smoke, and API
  tests are represented in the local/CI quality gate.

What is still rough:
- Upload-side grounding is approximate and evidence-backed, not survey-grade 3D
  reconstruction.
- Live Capture Coach is lightweight browser-side guidance, not full real-time
  room understanding.
- Offline support covers the app shell, local journal/settings, and queued
  upload retry; it does not make private reports or upload artifacts available
  offline.
- The eval corpus and beta feedback loops exist, but they still need more real
  labeled scans from beta users before quality claims can become stronger.
- This remains a self-hosted beta. The next production step is hosted operations,
  real user onboarding, object-storage-backed deployment hardening, and a larger
  reviewed evaluation corpus.

## What Atlas-0 Is Now

Atlas-0 is best thought of as:

- A **home or room safety scan** from a phone video
- A **report-first workflow** instead of a live demo-first workflow
- A **trust-improving system** that tries to show evidence and uncertainty,
  not fake certainty

Atlas-0 is not currently positioned as:

- a full digital twin platform
- a warehouse compliance suite
- a real-time AR product for everyday users
- a general-purpose reasoning agent

## What Works Today

The current product slice supports:

1. Uploading an image or room walkthrough video
2. Guided capture setup with room label, audience mode, scan checklist, and
   optional browser-side Live Capture Coach
3. Offline upload queuing and retry when a user selects a file without
   connection
4. Sampling frames and extracting salient regions
5. Labeling likely objects and estimating approximate multi-view positions
6. Generating hazard findings with:
   - severity
   - confidence
   - evidence frames
   - reasoning signals
   - actionable recommendations
7. Reviewing findings in a report-first frontend at `/app`
8. Asking bounded, deterministic questions of the active Safety Brief
9. Exporting a PDF report and copyable share/action packets
10. Previewing privacy receipts, evidence inclusion, and local blur choices
11. Saving local room history, care plans, challenges, fixes, and before/after
    comparisons
12. Inspecting operator quality, funnel, storage, worker, and eval readiness
    signals

## Why Someone Would Use This

The core user value is not "AI 3D magic." It is:

- **Quick room triage**: surface the objects most likely to fall, tip, spill,
  or break
- **Actionable output**: tell the user what is wrong, why it matters, and what
  to do next
- **Shareable evidence**: give them a report they can send to a landlord,
  partner, contractor, or insurer
- **Tiny home-care loop**: give them one fix today, then a reason to rescan and
  see whether the room got calmer

If ATLAS-0 becomes genuinely useful, it will be because it helps a person go
from "this room feels unsafe or cluttered" to "here are the 3 things I should
fix first."

## Known Limitations

This README is intentionally honest about the current state:

- Spatial positions are **estimated**, not measured with high precision.
- Upload reports should be treated as **decision support**, not professional
  safety certification.
- Weak scans still degrade results. Blur, darkness, short coverage, and low
  motion all reduce report quality.
- Local redaction/blur controls affect share previews and copied wording. They
  do not mutate stored server artifacts.
- The PWA/offline shell intentionally avoids caching private upload, report,
  operator, and artifact routes.
- The scene view is secondary. The report is the product.

## Quick Start

### Prerequisites

- Rust toolchain
- Python 3.11+
- `uv` for Python environment management
- One VLM path:
  - local Ollama, or
  - OpenAI, or
  - Anthropic

### Install

```bash
git clone https://github.com/yashasviudayan-py/Atlas-0
cd Atlas-0

uv sync --extra dev --extra video
```

Optional provider extras:

```bash
uv sync --extra dev --extra video --extra openai
uv sync --extra dev --extra video --extra claude
```

If you want the default local path, start Ollama separately and make sure your
configured model is available.

### Run The Upload-First Product

For the current product wedge, the easiest path is API + web app:

```bash
uv run python scripts/run_atlas.py --no-slam
```

Then open:

```text
http://localhost:8420/app
```

This gives you the report-first frontend where you can upload a scan and review
the resulting hazard report.

### Run The Full Stack

If you want to run the experimental Rust SLAM path as well:

```bash
uv run python scripts/run_atlas.py
```

Useful variants:

```bash
uv run python scripts/run_atlas.py --dev
uv run python scripts/run_atlas.py --config configs/default.toml
uv run python scripts/run_atlas.py --no-api
```

### Run The Production-Like Docker Stack

The Docker path runs the public API and upload worker as separate services with
shared durable artifact storage. This is the closest local shape to a hosted
beta deployment.

```bash
cp .env.example .env
# Edit .env and replace ATLAS_API_ACCESS_TOKEN before exposing the stack.
docker compose -f docker/docker-compose.yml up --build
```

Then open:

```text
http://localhost:8420/app
```

The Compose stack uses:

- `atlas-api` for FastAPI, static frontend, uploads, reports, and metrics
- `atlas-worker` for detached upload analysis
- `atlas_data` for persisted job manifests, PDFs, evidence, and replay assets
- `object_store_fs` artifact storage so manifests stay pointer-based
- `ollama` as the default local VLM provider

Use the `Authorization: Bearer <ATLAS_API_ACCESS_TOKEN>` header for private
upload/report endpoints when loopback auth is disabled.

Before exposing a hosted beta environment, run the deployment preflight:

```bash
python scripts/check_deployment.py
```

For stricter hosting gates, warnings can be treated as failures:

```bash
python scripts/check_deployment.py --strict-warnings
```

## Core API Surface

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/upload` | Upload an image or room walkthrough |
| `GET` | `/jobs` | List upload jobs |
| `GET` | `/jobs/{job_id}` | Fetch one job and its report payload |
| `POST` | `/jobs/{job_id}/feedback` | Mark a finding as useful, wrong, or duplicate |
| `POST` | `/jobs/{job_id}/follow-up` | Mark findings resolved, monitoring, or ignored |
| `POST` | `/jobs/{job_id}/evaluation` | Save human review / missed-hazard evaluation |
| `POST` | `/jobs/{job_id}/eval-candidate` | Export review-ready eval candidates |
| `GET` | `/reports/{job_id}.pdf` | Download the PDF report |
| `DELETE` | `/jobs/{job_id}` | Delete a job and persisted artifacts |
| `GET` | `/product/privacy` | Public retention/delete/privacy posture |
| `GET` | `/product/upload-guidance` | Public upload limits and accepted media guidance |
| `GET` | `/product/trust-proof` | Privacy-safe aggregate product quality signals |
| `POST` | `/product/events` | Public product telemetry allowlist |
| `POST` | `/product/waitlist` | Public beta waitlist capture |
| `GET` | `/sample-report` | Built-in sample Safety Brief |
| `GET` | `/operator/settings` | Token-protected operator diagnostics and beta inbox |
| `POST` | `/operator/storage/prune` | Token-protected storage lifecycle pruning |
| `GET` | `/health` | Runtime health and status |
| `POST` | `/query` | Experimental spatial query interface |
| `GET` | `/objects` | Experimental object listing |
| `GET` | `/scene` | Experimental scene snapshot |
| `GET` | `/metrics` | Prometheus metrics |
| `WS` | `/ws/risks` | Experimental risk delta stream |

## Development

### Required Checks

Before pushing, this repo expects all of the following to pass:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all
env RUSTFLAGS='-D warnings' cargo test --all
ruff check python/ scripts/check_deployment.py scripts/run_upload_worker.py
ruff format --check python/
pytest python/tests/ -v
node --check frontend/js/api.js
node --check frontend/js/app.js
node --check frontend/js/upload.js
python -m py_compile scripts/check_deployment.py
python -m py_compile scripts/run_upload_worker.py
python scripts/check_deployment.py
python scripts/benchmark.py --iterations 1 --skip-vlm
```

### Benchmarks

```bash
uv run python scripts/benchmark.py --skip-vlm
```

The benchmark suite includes the committed sample walkthrough report fixture so
the upload/report path can be checked for regressions.

## Repository Layout

```text
crates/            Rust crates for SLAM, physics, streaming, and shared core code
python/atlas/      Python API, VLM integration, world-model logic, utilities
frontend/          Report-first web UI
configs/           Runtime TOML configuration
scripts/           Process manager, benchmarks, and support scripts
docs/              Architecture docs and development plan
data/              Sample walkthrough fixtures and expected report output
tests/             Cross-language integration tests
```

## Roadmap

The active roadmap lives in [docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md).

The current order of attack is:

1. Keep gathering real beta scans and convert them into labeled eval cases.
2. Improve upload grounding beyond the current heuristic/multi-frame pipeline.
3. Harden hosted deployment, object storage, worker operations, and artifact
   retention for real production traffic.
4. Keep the report more useful than the visualization: clearer evidence,
   stronger recommendations, better before/after verification.
5. Preserve Warm Trust: honest uncertainty, privacy controls, accessible UI,
   and no safety-certification claims.

## License

MIT. See [LICENSE](LICENSE).
