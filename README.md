<div align="center">

<img src="docs/assets/atlas0-logo.svg" alt="ATLAS-0 — Room Safety Brief" width="900" />

<h1>ATLAS-0</h1>

<strong>Upload one room walkthrough. Leave with a warm, evidence-backed Room Safety Brief.</strong>

<em>A second pair of eyes on your room — before something tips, falls, spills, or blocks a path.</em>

<br/><br/>

<a href="https://github.com/yashasviudayan-py/Atlas-0/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/yashasviudayan-py/Atlas-0/ci.yml?style=flat-square&labelColor=13201F&color=0C8074&label=CI" alt="CI status" /></a>
<a href="docs/DEVELOPMENT_PLAN.md"><img src="https://img.shields.io/badge/status-beta_MVP-0C8074?style=flat-square&labelColor=13201F" alt="Status: beta MVP" /></a>
<a href="docs/DEVELOPMENT_PLAN.md"><img src="https://img.shields.io/badge/product-room_safety_scan-D67F30?style=flat-square&labelColor=13201F" alt="Product: room safety scan" /></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-0A5F57?style=flat-square&labelColor=13201F" alt="License: MIT" /></a>
<br/>
<img src="https://img.shields.io/badge/python-ruff_•_pytest-0C8074?style=flat-square&labelColor=13201F&logo=python&logoColor=white" alt="Python checks" />
<img src="https://img.shields.io/badge/rust-fmt_•_clippy_•_test-D67F30?style=flat-square&labelColor=13201F&logo=rust&logoColor=white" alt="Rust checks" />
<img src="https://img.shields.io/badge/deploy-preflight-125D69?style=flat-square&labelColor=13201F&logo=docker&logoColor=white" alt="Deployment preflight" />

<br/>

<sub>
  <a href="#-how-it-works">How it works</a> &nbsp;·&nbsp;
  <a href="#-what-atlas-0-is-and-isnt">What it is</a> &nbsp;·&nbsp;
  <a href="#-quick-start">Quick start</a> &nbsp;·&nbsp;
  <a href="#-core-api-surface">API</a> &nbsp;·&nbsp;
  <a href="#-development">Development</a> &nbsp;·&nbsp;
  <a href="#-roadmap">Roadmap</a>
</sub>

</div>

> [!IMPORTANT]
> **Decision support, not safety certification.** ATLAS-0 is intentionally honest
> about confidence, scan quality, and where the pipeline is still approximate.
> The design direction is **Warm Trust**: calm language, high contrast, visible
> uncertainty, and practical next steps — not demo theater or fake certainty.

---

## 🔁 How it works

<table>
<tr>
<td width="33%" valign="top">

### 📥 Input
A **20–60 second** phone walkthrough of a real room.

</td>
<td width="33%" valign="top">

### 🧾 Output
A **Room Safety Brief**: top actions, evidence frames, approximate locations,
confidence, and fix-first recommendations.

</td>
<td width="33%" valign="top">

### 📤 Share
A downloadable **PDF report** to send to a landlord, partner, contractor, or
care team.

</td>
</tr>
</table>

> If ATLAS-0 becomes genuinely useful, it will be because it helps a person go
> from *"this room feels unsafe or cluttered"* to *"here are the 3 things I
> should fix first."*

---

## 🧭 What ATLAS-0 is (and isn't)

<table>
<tr>
<th width="50%" align="left">✅ It is</th>
<th width="50%" align="left">🚫 It is not</th>
</tr>
<tr>
<td valign="top">

- A **home/room safety scan** from a phone video
- A **report-first** workflow, not a live demo
- A **trust-improving** system that shows evidence and uncertainty

</td>
<td valign="top">

- A full digital-twin platform
- A warehouse compliance suite
- A real-time AR product for everyday users
- A general-purpose reasoning agent

</td>
</tr>
</table>

**Why someone reaches for it**

- 🩺 **Quick room triage** — surface the objects most likely to fall, tip, spill, or break
- 🔧 **Actionable output** — what's wrong, why it matters, and what to do next
- 📎 **Shareable evidence** — a report for a landlord, partner, contractor, or insurer
- 🔁 **Tiny home-care loop** — one fix today, then a reason to rescan and watch the room get calmer

---

## ✨ What works today

<details open>
<summary><strong>The current product slice</strong></summary>

<br/>

1. Upload an image or room walkthrough video
2. Guided capture: room label, audience mode, scan checklist, and optional browser-side **Live Capture Coach**
3. Offline upload queuing and retry when a file is selected without connection
4. Frame sampling and salient-region extraction
5. Object labeling and approximate multi-view position estimates
6. Hazard findings with **severity · confidence · evidence frames · reasoning signals · recommendations**
7. Report-first review frontend at `/app`
8. Bounded, deterministic "Ask this Safety Brief" answers
9. PDF export and copyable share/action packets
10. Privacy receipts: evidence inclusion and local blur/redaction previews
11. Local room history, care plans, challenges, fixes, and before/after comparisons
12. Operator quality, funnel, storage, worker, and eval-readiness signals

</details>

<details>
<summary><strong>Current standing — strong vs. still rough</strong></summary>

<br/>

ATLAS-0 is a self-hosted beta product, not a broad demo shell. The spec:
**upload a walkthrough → get a trustworthy Safety Brief → fix one thing →
optionally rescan to compare progress.**

**💪 Strong today**
- First-run path: sample report, scan playbooks, audience modes, Room Mystery Mode, file preflight, capture guidance
- Live Capture Coach checks lighting, steadiness, duration, floor path, corners, and coverage in-browser before upload
- Durable job/artifact pipeline with detached-worker support and an IndexedDB retry queue for offline-selected files
- Brief surfaces top actions, Calm Score, confidence/uncertainty, evidence frames, fix difficulty, Field Notes, approximate evidence maps
- Privacy Receipt shows room label, evidence inclusion, local blur/redaction previews, retention posture, and delete controls
- Same-room rescans produce before/after deltas (score, hazards, compact evidence snapshots)
- Home Journal, Room Health Passport, Rituals, One Thing Today, Care Calendar, Fix Library, challenges, streaks, recaps — all local-first and accountless
- Trust Proof + operator views expose aggregate quality/funnel/storage/worker/eval signals without leaking private data
- Settings is a local control center: theme, accessibility, report/scan defaults, privacy clearing, backup/import, feedback, changelog
- Rust, Python, frontend smoke, deployment preflight, benchmark smoke, and API tests are in the local/CI quality gate

**🚧 Still rough**
- Upload-side grounding is approximate and evidence-backed, not survey-grade 3D reconstruction
- Live Capture Coach is lightweight browser guidance, not full real-time room understanding
- Offline covers the app shell, local journal/settings, and queued retry — not private reports or upload artifacts
- The eval corpus and feedback loops exist but need more real labeled scans before quality claims strengthen
- Still self-hosted beta — next is hosted operations, real onboarding, object-storage hardening, and a larger reviewed eval corpus

</details>

> [!WARNING]
> **Known limitations — stated plainly.**
> - Spatial positions are **estimated**, not precisely measured.
> - Reports are **decision support**, not professional safety certification.
> - Weak scans degrade results — blur, darkness, short coverage, and low motion all reduce quality.
> - Local redaction/blur affects share previews and copied wording; it does **not** mutate stored server artifacts.
> - The PWA/offline shell intentionally avoids caching private upload, report, operator, and artifact routes.
> - The scene view is secondary. **The report is the product.**

---

## 🚀 Quick start

**Prerequisites** — Rust toolchain · Python 3.11+ · [`uv`](https://github.com/astral-sh/uv) · one VLM path (local **Ollama**, **OpenAI**, or **Anthropic**).

```bash
git clone https://github.com/yashasviudayan-py/Atlas-0
cd Atlas-0
uv sync --extra dev --extra video
```

<details>
<summary>Optional provider extras</summary>

<br/>

```bash
uv sync --extra dev --extra video --extra openai
uv sync --extra dev --extra video --extra claude
```

For the default local path, start Ollama separately and make sure your
configured model is available.

</details>

**Run the upload-first product** (the current product wedge — API + web app):

```bash
uv run python scripts/run_atlas.py --no-slam
```

Then open **`http://localhost:8420/app`** — the report-first frontend where you
upload a scan and review the resulting hazard report.

<details>
<summary>Run the full stack (experimental Rust SLAM path)</summary>

<br/>

```bash
uv run python scripts/run_atlas.py

# Useful variants
uv run python scripts/run_atlas.py --dev
uv run python scripts/run_atlas.py --config configs/default.toml
uv run python scripts/run_atlas.py --no-api
```

</details>

<details>
<summary>Run the production-like Docker stack</summary>

<br/>

The closest local shape to a hosted beta: public API + upload worker as separate
services with shared durable artifact storage.

```bash
cp .env.example .env
# Edit .env and replace ATLAS_API_ACCESS_TOKEN before exposing the stack.
docker compose -f docker/docker-compose.yml up --build
```

Then open **`http://localhost:8420/app`**. The Compose stack uses:

| Service | Role |
|--------|------|
| `atlas-api` | FastAPI, static frontend, uploads, reports, metrics |
| `atlas-worker` | detached upload analysis |
| `atlas_data` | persisted job manifests, PDFs, evidence, replay assets |
| `object_store_fs` | pointer-based artifact storage |
| `ollama` | default local VLM provider |

Use `Authorization: Bearer <ATLAS_API_ACCESS_TOKEN>` for private upload/report
endpoints when loopback auth is disabled. Before exposing a hosted environment,
run the preflight (add `--strict-warnings` for stricter gates):

```bash
python scripts/check_deployment.py
```

</details>

---

## 🔌 Core API surface

<details>
<summary><strong>Endpoints</strong> — product, operator, and experimental routes</summary>

<br/>

| Method | Endpoint | Purpose |
|:------:|----------|---------|
| `POST` | `/upload` | Upload an image or room walkthrough |
| `GET` | `/jobs` | List upload jobs |
| `GET` | `/jobs/{job_id}` | Fetch one job and its report payload |
| `POST` | `/jobs/{job_id}/feedback` | Mark a finding useful, wrong, or duplicate |
| `POST` | `/jobs/{job_id}/follow-up` | Mark findings resolved, monitoring, or ignored |
| `POST` | `/jobs/{job_id}/evaluation` | Save human review / missed-hazard evaluation |
| `POST` | `/jobs/{job_id}/eval-candidate` | Export review-ready eval candidates |
| `GET` | `/reports/{job_id}.pdf` | Download the PDF report |
| `DELETE` | `/jobs/{job_id}` | Delete a job and persisted artifacts |
| `GET` | `/product/privacy` | Public retention/delete/privacy posture |
| `GET` | `/product/upload-guidance` | Public upload limits and accepted media |
| `GET` | `/product/trust-proof` | Privacy-safe aggregate quality signals |
| `POST` | `/product/events` | Public product telemetry allowlist |
| `POST` | `/product/waitlist` | Public beta waitlist capture |
| `GET` | `/sample-report` | Built-in sample Safety Brief |
| `GET` | `/operator/settings` | Token-protected operator diagnostics + beta inbox |
| `POST` | `/operator/storage/prune` | Token-protected storage lifecycle pruning |
| `GET` | `/health` | Runtime health and status |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/query` | _Experimental_ — spatial query interface |
| `GET` | `/objects` | _Experimental_ — object listing |
| `GET` | `/scene` | _Experimental_ — scene snapshot |
| `WS` | `/ws/risks` | _Experimental_ — risk delta stream |

</details>

---

## 💻 Development

<details>
<summary><strong>Required checks</strong> — everything that must pass before pushing</summary>

<br/>

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

</details>

**Benchmarks** — the suite includes the committed sample walkthrough fixture so
the upload/report path is regression-checked:

```bash
uv run python scripts/benchmark.py --skip-vlm
```

### 🗂️ Repository layout

```text
crates/        Rust crates — SLAM, physics, streaming, shared core
python/atlas/  Python API, VLM integration, world-model logic, utilities
frontend/      Report-first web UI
configs/       Runtime TOML configuration
scripts/       Process manager, benchmarks, support scripts
docs/          Architecture docs and development plan
data/          Sample walkthrough fixtures and expected output
tests/         Cross-language integration tests
```

---

## 📍 Roadmap

The active roadmap lives in **[docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md)**. The current order of attack:

1. **Gather real beta scans** and convert them into labeled eval cases
2. **Improve upload grounding** beyond the current heuristic/multi-frame pipeline
3. **Harden hosted deployment** — object storage, worker ops, artifact retention for production traffic
4. **Keep the report > the visualization** — clearer evidence, stronger recommendations, better before/after verification
5. **Preserve Warm Trust** — honest uncertainty, privacy controls, accessible UI, no safety-certification claims

---

<div align="center">

### 📄 License

**MIT** — see [LICENSE](LICENSE).

<sub>Built with calm language, visible uncertainty, and a fix-first mindset. 🏠</sub>

</div>
