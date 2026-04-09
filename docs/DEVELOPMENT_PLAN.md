# ATLAS-0: Product Reset Development Plan

> April 2026 reset. This document replaces the old "everything at once"
> roadmap with a narrower plan built around one outcome:
> a user uploads a phone walkthrough and gets a trustworthy hazard report.

---

## Executive Summary

Atlas-0 has real engineering underneath it, but it is not yet a product people
can trust or recommend.

Today the repo is strongest as an experimental engine:
- Rust test coverage is solid.
- Python services and the frontend are reasonably modular.
- The codebase can demo reconstruction, labeling, and risk overlays.

Today the product is weak:
- Upload-based "3D" is still pseudo-depth generated from image brightness.
- Uploaded objects are not grounded by real multi-view geometry.
- Reasoning is mostly VLM labeling plus rule-based queries, not reliable
  evidence-backed scene understanding.
- The frontend still contains demo fallback behavior, which makes the product
  feel more polished than it really is.
- Setup, trust, and output quality are not good enough for repeat usage.

The fix is not "more features." The fix is focus.

---

## Product Decision

### What We Are Actually Building

We should stop positioning Atlas-0 as a general "spatial reasoning and world
model engine" for now.

We should build one opinionated product:

**Upload a 20-60 second phone walkthrough of a room. Get back a ranked hazard
report with evidence frames, locations, severity, and a downloadable PDF.**

If that works well, users can understand it, trust it, share it, and come
back to it.

### What We Are Not Building Yet

These should be explicitly deprioritized until the core product works:
- Live AR overlays as the primary user experience
- Real-time 60 fps end-to-end consumer workflows
- Broad warehouse/compliance workflows
- Insurance inventory automation
- "General reasoning agent" messaging
- Full 3D scene editing or digital twin platform ambitions

Those may become expansion paths later, but they are not the wedge.

---

## Identity / Positioning

### Recommendation

Keep `ATLAS-0` as the project and product name for now.

Do not force a rebrand before the product wedge is proven. The identity problem
is real, but the first fix should be better positioning, not a premature rename.

### Suggested tagline

`Scan your room. Catch what could fall, tip, or break.`

### Messaging shift

Current messaging is too ambitious and too abstract.

Replace:
- "Spatial reasoning & physical world-model engine"

With:
- "Home safety scan from a phone video"
- "Find unstable objects before they fall"
- "Turn a room walkthrough into a hazard report"

---

## Reality Check

These are the current blockers preventing usefulness and user growth.

### 1. The core value is not trustworthy yet

The app can display objects, risks, and point clouds, but much of the current
upload path is heuristic or synthetic. That is acceptable for internal
development and unacceptable for external trust.

### 2. The product promise is too broad

Home safety, warehouse compliance, insurance inventory, real-time AR, and
multi-provider reasoning all at once create a diluted roadmap and scattered UX.

### 3. The output is not share-worthy

A browser visualization is not enough. Users need:
- a report
- evidence
- a link or export they can send to someone else

### 4. Demo polish is masking real gaps

Demo-mode fallbacks and aesthetic overlays currently create a mismatch between
what the interface implies and what the system actually knows.

---

## North Star

### Primary User

A renter, homeowner, or parent who wants a fast hazard scan of a room.

### Core Job To Be Done

"Show me the objects most likely to fall, tip, spill, or create a hazard, and
give me enough evidence to act on it."

### North Star Experience

1. User opens ATLAS-0.
2. User uploads a phone walkthrough video.
3. Within 3-5 minutes, the user gets:
   - top hazards
   - evidence frames
   - approximate room locations
   - a confidence score
   - a downloadable PDF report
4. The user can share the report with a landlord, partner, contractor, or
   insurance contact.

### Success Metrics

- Time to first useful result: under 5 minutes
- First-session completion rate: above 60%
- Report download/share rate: above 30%
- Returning users within 14 days: above 20%
- At least 10 beta users submit 3+ real scans each

---

## Build Principles

1. Trust beats spectacle.
   Remove fake confidence, fake geometry, and fake reasoning.

2. Async upload beats live demo complexity.
   A reliable asynchronous pipeline is more valuable than a fragile real-time one.

3. Evidence beats cleverness.
   Every hazard should link to the frames, objects, and signals that produced it.

4. A report beats a visualization.
   The scene viewer is supporting UX. The report is the product.

5. One wedge beats five markets.
   Win home safety first. Expand only after repeated user pull.

---

## Build Plan

## Phase 0: Truth Reset

### Status

In progress. The repo now has an honest upload/report shell, trust notes,
runtime-config fixes, PDF export, and no primary-flow demo fallback in the
frontend scene viewer. Known limitations are still not centralized enough, and
the upload pipeline remains heuristic on grounding.

### Goal

Remove product dishonesty and align the codebase, docs, and UI with reality.

### Deliverables

1. Eliminate demo-only fallback in primary product flows.
2. Mark synthetic or heuristic outputs clearly in the UI and API.
3. Add health/status reporting that reflects real SLAM/runtime state.
4. Wire runtime config correctly so provider switches and config overrides
   actually work.
5. Create a single "known limitations" section in README and product docs.

### Exit Criteria

- No primary user flow silently falls back to fake data.
- Status endpoints reflect real runtime state.
- Config-driven provider/runtime switches are verified by tests.

### Completed in repo

- Runtime config now drives provider/runtime wiring correctly.
- Upload jobs now expose `summary`, `recommendations`, `evidence_frames`,
  `trust_notes`, `scene_source`, and `report_url`.
- The primary frontend flow is now upload-first and report-first instead of a
  demo dashboard.
- The scene viewer no longer silently injects fake demo objects when the API is
  empty or unavailable.
- Completed jobs can export a downloadable PDF report.

---

## Phase 1: Useful Upload-to-Report MVP

### Status

Started. The first report-first MVP is now in the repo, but it is not yet a
beta-ready user flow because jobs are still in-memory, grounding is still
heuristic, and there is no polished sample scan or hosted onboarding path.

### Goal

Make one room scan produce one useful report without touching the terminal.

### Deliverables

1. Robust upload pipeline
   - MP4/MOV/WEBM upload via browser
   - clear progress states
   - graceful error messages
   - job persistence beyond in-memory-only state

2. Better semantic extraction
   - use cloud VLM providers by default for best-quality beta experience
   - keep Ollama as fallback, not as the main pitch
   - add object evidence thumbnails per finding

3. Report-first UX
   - summary card: total hazards, top risks, scan timestamp
   - evidence gallery per hazard
   - actions/recommendations per hazard
   - PDF export

4. Real demo asset
   - one polished sample walkthrough
   - expected output committed for regression testing

### File Areas

- `python/atlas/api/server.py`
- `python/atlas/api/export.py`
- `python/atlas/utils/video.py`
- `python/atlas/vlm/*`
- `frontend/index.html`
- `frontend/js/upload.js`
- `frontend/js/intelligence.js`
- `README.md`

### Exit Criteria

- A new user can upload one room video and download one useful report.
- No Rust toolchain knowledge is required for the beta path.

### Completed in repo

- Browser upload now lands in a report-first UI with processing states.
- The report view now shows top hazards, recommendations, evidence, trust
  notes, and PDF export.
- The scene view has been demoted to a secondary inspection tool.

### Still missing

- persistent jobs and storage beyond process memory
- polished empty/error states for all backend failures
- one committed sample walkthrough with expected output
- a clearer "how to record a good scan" onboarding path

---

## Phase 2: Spatial Grounding That Is Not Fake

### Status

Not done. Approximate locations and trust notes are now explicit, but the
pipeline is still based on heuristic positioning rather than real multi-view
grounding.

### Goal

Replace pseudo-3D shortcuts with enough real geometry to support believable
hazard localization.

### Deliverables

1. Multi-view object grounding
   - track object observations across sampled frames
   - merge repeated sightings into scene objects
   - retain per-object evidence frames

2. Replace luminance-as-depth for uploads
   - use a real monocular depth model or sparse reconstruction path
   - estimate scale from camera motion, known objects, or calibration hints

3. Confidence-aware spatial output
   - object location uncertainty
   - "approximate" vs "high confidence" spatial labels
   - do not fabricate precise positions when evidence is weak

4. Scene snapshot schema
   - make the JSON scene model explicit about source:
     `measured`, `estimated`, or `heuristic`

### Exit Criteria

- Uploaded objects have evidence-backed approximate locations.
- No fabricated room placement remains in the upload pipeline.

---

## Phase 3: Reasoning Users Can Trust

### Status

Partially started. We now generate simple recommendation cards and evidence
frames, but the reasoning layer is still shallow and the hazard ontology is not
formalized.

### Goal

Turn labels into actionable hazard judgments with explicit evidence.

### Deliverables

1. Hazard ontology
   - define the first 15-20 hazard types
   - examples: tipping, falling glass, blocked exit, overloaded shelf,
     unstable stack, liquid spill near edge

2. Evidence-backed reasoning layer
   - each finding stores:
     - object(s) involved
     - supporting evidence frame(s)
     - spatial relationship(s)
     - physics/heuristic contribution
     - confidence

3. Recommendation engine
   - simple, deterministic guidance first
   - examples:
     - "move vase away from edge"
     - "anchor tall shelf"
     - "lower heavy object"

4. User feedback loop
   - mark finding as useful / wrong / duplicate
   - capture corrections for later evaluation

### Exit Criteria

- Every top hazard in the report has visible supporting evidence.
- Users can understand why the system made the claim.

### Completed in repo

- Findings now surface evidence frames and location labels.
- Deterministic recommendations are generated alongside risks.

### Still missing

- explicit hazard ontology and evaluation set
- reasoning traces that show which relationships produced a claim
- user feedback capture for wrong/duplicate findings

---

## Phase 4: Productization and Growth

### Goal

Make ATLAS-0 easy to try, easy to share, and easy to talk about.

### Deliverables

1. Hosted beta path
   - landing page
   - sample scan
   - email capture / waitlist / beta invite flow

2. Shareable outputs
   - PDF report
   - share link
   - inventory JSON for integrations later

3. Onboarding
   - "how to record a good scan" guidance
   - expected scan length
   - progress and retry UX

4. Analytics
   - upload success rate
   - time to completed report
   - drop-off points
   - report download/share events

5. Demo and social proof
   - before/after sample scans
   - real screenshots
   - one short product video

### Exit Criteria

- A stranger can understand the product in under 30 seconds.
- A beta user can go from landing page to report without help.

---

## Proposed 8-Week Sequence

### Weeks 1-2

- Phase 0 complete
- remove misleading fallback behavior
- stabilize config/runtime wiring
- define brand direction and rewrite README/landing copy

### Weeks 3-4

- Phase 1 MVP complete
- upload to report working end-to-end
- PDF export shipped
- real sample demo shipped

### Weeks 5-6

- Phase 2 underway
- real upload grounding replaces pseudo-3D shortcuts
- confidence-aware spatial output added

### Weeks 7-8

- Phase 3 first pass
- hazard ontology implemented
- evidence-backed explanations added
- beta onboarding and share flow shipped

---

## Immediate Engineering Priorities

These are the next concrete tasks that matter most.

1. Remove or clearly label synthetic geometry and demo fallback behavior.
2. Finish the report/export path so output is portable.
3. Make cloud VLM usage the best-path beta experience.
4. Replace upload pseudo-depth with a real grounded spatial estimate.
5. Add evidence-linked findings and confidence scoring.
6. Ship one real walkthrough demo and benchmark against it on every change.

---

## Non-Negotiable Quality Bar

Before calling the product usable:

- no fake scene data in the main UX
- no silent fallback that implies confidence the system does not have
- no claim without evidence frames or rationale
- no setup path that requires reading the source to succeed
- no marketing copy that outruns actual product behavior

---

## Final Recommendation

Do not try to "win" on general 3D world modeling right now.

Win on this instead:

**ATLAS-0 helps a person scan a room and get a credible hazard report they can
immediately act on or share.**

If we can make that experience fast, trustworthy, and easy to demo, users will
show up. If we keep shipping broad technical ambition without a tight product
loop, we will keep producing impressive scaffolding and weak adoption.
