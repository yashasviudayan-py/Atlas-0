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

Substantially complete for the current MVP slice. Upload jobs and report
artifacts now persist to disk, the report path has a committed walkthrough
fixture plus expected output, and the remaining gaps are now more about hosted
productization than core upload/report usefulness.

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
- Upload jobs and generated PDFs now persist beyond in-memory state.
- A committed sample walkthrough fixture and expected report now exist for
  regression coverage.
- The benchmark suite now includes a sample walkthrough report run.

### Still missing

- polished empty/error states for all backend failures
- a clearer "how to record a good scan" onboarding path

---

## Phase 2: Spatial Grounding That Is Not Fake

### Status

First pass complete. The upload pipeline now tracks salient regions across
sampled frames, estimates a simple camera path from frame motion, localizes
objects from repeated observations, and emits confidence-aware multi-view
positions. It is still approximate rather than full reconstruction.

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

### Completed in repo

- Upload video analysis now tracks repeated region observations across frames.
- Estimated object positions now come from multi-view frame tracks instead of a
  fixed per-frame Z offset.
- Scene output explicitly distinguishes `estimated_multiview` from
  `single_view_estimate`.
- Point clouds are now anchored around tracked observations instead of a fake
  room layout.

---

## Phase 3: Reasoning Users Can Trust

### Status

First pass complete. The upload report now uses a named hazard ontology,
stores explicit evidence and signals per finding, and derives deterministic
recommendations from those findings. The reasoning is still rules-first, not
learned or user-corrected.

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
- A formal upload hazard ontology now exists with named hazard codes and
  families.
- Findings now store explicit evidence IDs, signals, confidence, and reasoning
  text.

### Still missing

- reasoning traces that show which relationships produced a claim
- persistent per-finding follow-through states such as resolved / monitor / ignored

---

## Phase 4: Productization and Growth

### Goal

Make ATLAS-0 easy to try, easy to share, and easy to talk about.

### Status

Partially complete. The product now has a more trustworthy report path,
feedback capture, replay artifacts, hosted token access, durable queued job
execution, storage lifecycle controls, and clearer beta-facing trust language.
The remaining work is to make that foundation production ready and worth coming
back to.

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

6. Retention and repeat usage
   - saved scan history by room
   - rescan flow for before/after comparison
   - room safety score with trend over time

7. Audience-specific modes
   - toddler mode
   - pet mode
   - move-in / move-out renter mode
   - reorder findings and recommendations based on context

### Exit Criteria

- A stranger can understand the product in under 30 seconds.
- A beta user can go from landing page to report without help.
- A returning user has a clear reason to rescan the same room after making
  changes.

### Completed in repo

- Browser upload, report generation, replay artifacts, PDF export, and hosted
  token access now exist in one coherent flow.
- Users can submit feedback on findings, and operators can inspect evaluation
  summaries plus runtime/storage diagnostics.
- The product now uses safer low-confidence language instead of implying that
  "no strong hazard found" means "safe."
- Share-link flow for reports is live.
- Room history, before/after rescan comparisons, and a room score foundation are
  live.
- Audience-specific modes such as toddler / pet / renter are live.
- A public built-in sample walkthrough report now exists for first-time users.
- Findings can now carry a persistent follow-through state so the report can
  track resolved / monitor / ignored work over time.
- A basic public waitlist capture path now exists in the frontend and API.
- Product funnel events now persist so operator metrics can track sample opens,
  share actions, PDF downloads, and CTA usage.
- Reviewed reports can now be exported as eval candidates, which turns the eval
  corpus from a vague goal into an operator workflow.

### Still missing

- broader landing page / invite funnel beyond the current hero + waitlist
- richer analytics and attribution beyond the current first-pass event stream
- a genuinely larger reviewed eval corpus built from real beta scans

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

## Pre-Mortem Optimization Backlog

This section converts the latest product/system audit into concrete build work.
The goal is to move ATLAS-0 from "functional" to "industry-leading" without
losing the current product wedge.

### Why the project would fail if we do nothing

- Users would try it once, see a report they cannot fully trust, and not come
  back.
- Processing would become inconsistent under real usage because ingestion,
  analysis, persistence, and API serving are still too tightly coupled.
- The product would still read as a "cool spatial demo" rather than a clear
  hazard-report tool.
- Privacy and retention concerns around room videos and evidence crops would
  become blockers before user growth.

### Workstream A: User Experience & Friction

#### Goal

Reduce drop-off, reduce cognitive load, and make the report immediately
actionable.

#### Planned fixes

1. Scan quality gate before full processing
   - score uploads for blur, darkness, overexposure, low motion coverage, and
     low saliency coverage
   - warn early when a scan is unlikely to produce a trustworthy report
   - return explicit retry guidance such as "move slower" or "keep the table in frame"

2. "Fix First" report summary
   - add a top-of-report section with the three most important actions
   - rank by severity × confidence × actionability, not only raw risk score
   - collapse duplicate hazards on the same object into a single primary action

3. Report simplification
   - hide low-confidence findings behind a toggle instead of treating them as
     equal to strong findings
   - standardize every finding card to:
     `what`, `why it matters`, `what to do next`
   - keep ontology metadata visible but secondary

4. Scene view demotion and clarification
   - label scene view as estimated spatial context
   - surface object observation count and grounding confidence in the scene UI
   - always land users on the report, never on the scene tab

5. Sticky follow-up loop
   - add saved-scan history
   - allow room labels and repeat scans
   - add finding status such as `resolved`, `still present`, or `ignored`

### Workstream B: Scalability & Robustness

#### Goal

Make the system survive 10x more usage and 10x more data diversity without
falling apart.

#### Planned fixes

1. Split API from job execution
   - FastAPI should accept uploads and enqueue jobs only
   - move upload analysis into dedicated worker processes
   - enforce concurrency limits and backpressure at the queue boundary

2. Shrink manifests and externalize artifacts
   - remove inline base64 evidence from job manifests
   - store evidence crops, replays, and PDFs as files/object-store blobs
   - keep manifests small and pointer-based

3. Artifact storage abstraction
   - keep local filesystem backend for dev
   - add an interface for object storage later
   - separate manifest storage from binary artifact storage

4. Provider resilience
   - add provider-level timeouts, retries, and degraded-mode reporting
   - cache repeated crop inferences by image hash + prompt version
   - split local interactive mode from production provider mode in config/docs

5. Data-volume controls
   - enforce upload duration and size limits
   - track artifact bytes per job
   - add pruning by age and total storage, not only by job count

### Workstream C: Trust, Reasoning, and Product Defensibility

#### Goal

Make the report explainable and obviously more trustworthy than a generic AI
viewer.

#### Planned fixes

1. Evidence replay
   - generate a short replay per top finding
   - highlight the object and show the associated risk caption
   - pair replay links with PDF/share output

2. Stronger confidence calibration
   - distinguish "no high-confidence hazards detected" from "room is safe"
   - surface report coverage warnings when scan quality or observation count is weak
   - suppress overconfident claims when grounding quality is poor

3. Reasoning trace expansion
   - attach object relationships and rule hits to each finding
   - show which signals materially contributed to the claim
   - expose low-level reasoning in a debug view, not in the primary UX

4. Evaluation corpus
   - build a 20-50 scan evaluation set
   - measure false positives, false negatives, and confidence calibration
   - benchmark every major report-path change against that set

5. Multi-provider quality strategy
   - use stronger cloud vision models as the best-path beta experience
   - keep Ollama/local mode as fallback and developer path
   - compare provider output quality on the same evaluation scans

### Workstream D: Edge Cases, Privacy, and Security

#### Goal

Prevent non-obvious failures that would block real-world use.

#### Planned fixes

1. Retention and deletion controls
   - default artifact retention window
   - manual delete per job
   - clear operator policy for persisted uploads/evidence

2. Sensitive-room-content handling
   - add optional redaction for text-heavy or sensitive evidence crops
   - detect documents, screens, labels, and photo walls before persistence/export
   - avoid retaining more raw room imagery than the product actually needs

3. Multimodal prompt-injection hardening
   - detect dense text regions in frames
   - isolate or redact OCR-heavy regions before VLM calls
   - add adversarial image-text evaluation cases

4. Safer "clear" results
   - avoid absolute "safe" language
   - require a minimum evidence/coverage threshold before presenting a low-risk result prominently
   - show confidence and scan-quality caveats in the summary itself

5. Access control readiness
   - prepare artifact endpoints for authenticated access before any hosted beta
   - avoid assuming local-only trust boundaries in report/evidence delivery

### Ordered Execution Plan

This is the preferred order of attack after the work already completed in the
current repo.

1. Evaluation corpus expansion and release gating
   - required before broader beta or more model complexity
   - converts correctness work from intuition into measurable quality bars

2. Scan acceptance and refusal flow
   - reject or downgrade low-quality scans before they create misleading reports
   - protects trust and reduces support burden

3. Privacy-first beta controls
   - visible delete-my-scan controls, retention messaging, and stronger
     redaction for sensitive imagery
   - required before meaningful external beta growth

4. Shareable growth loop
   - before/after rescans, room safety score, and improved share output
   - highest leverage feature set for word of mouth and repeat usage

5. Audience-specific modes
   - toddler mode, pet mode, and renter move-in mode
   - strongest path to "this feels made for me" retention and referrals

6. Proper deployment infrastructure
   - move from app-local durable queue + disk storage toward dedicated workers,
     object storage, and production observability
   - required for a serious hosted beta

7. Provider routing maturity
   - benchmark local vs hosted providers on the eval set
   - use routing only where it measurably improves accuracy or latency

### Starting Today

If we continue from this plan today, the first concrete build slice should
be:

1. build the first real labeled evaluation set and define release gates
2. implement hard scan refusal / downgrade logic for low-quality uploads
3. add user-visible privacy controls and delete-my-scan UX
4. add before/after rescans plus a room safety score foundation

### Production Readiness Track

This is the most important path if the goal is to make ATLAS-0 safe to expose
to outside beta users.

### April 23 Production Deployment Slice

Today changes the default production-readiness focus from more UI polish to a
repeatable deployment shape:

1. CI parity and pinned toolchains
   - Rust is pinned with `rust-toolchain.toml` so local checks and GitHub
     Actions run the same compiler and Clippy rules.
   - The CI workflow now checks the worker entrypoint, frontend JavaScript
     syntax, and the benchmark smoke path in addition to Rust and Python tests.

2. API / worker split in Docker
   - Docker Compose now runs `atlas-api` and `atlas-worker` as separate
     services.
   - Upload analysis runs through `ATLAS_UPLOADS_WORKER_MODE=external` instead
     of hiding work inside the API process.

3. Durable artifact storage for beta-like runs
   - Compose uses shared `atlas_data` storage and `object_store_fs` artifacts.
   - Reports, evidence, replay assets, and PDFs are persisted as file-backed
     objects instead of being treated as disposable container state.

4. Hosted-beta guardrails
   - The stack requires `ATLAS_API_ACCESS_TOKEN` before startup.
   - Loopback auth bypass is disabled in Compose.
   - Broad job listing remains disabled unless explicitly enabled.

5. Operator runbook
   - README now includes the production-like Docker path, service roles, and
     the full pre-push verification list.

Remaining production gap: this is still a single-host, filesystem-backed
deployment. The next infrastructure jump is a managed queue plus remote object
storage such as S3/R2/GCS, with structured tracing around provider latency,
queue age, and artifact lifecycle events.

### April 24 Production Tightening Slice

Today focuses on smaller loose ends that make a hosted beta less brittle:

1. Public upload guidance
   - The API now exposes upload limits, accepted media families, capture
     checklist items, and retry guidance through `/product/upload-guidance`.
   - The frontend uses that contract to show the real hosted upload limit and
     reject obviously oversized files before they consume queue capacity.

2. Deployment preflight
   - `scripts/check_deployment.py` gives operators a repeatable readiness
     command backed by the same startup checks as the API.
   - CI now compiles and runs the preflight so deployment regressions surface
     before a push becomes another red workflow.

3. Beta UX tightening
   - First-run capture copy now comes from backend guidance instead of drifting
     away from runtime limits.
   - Upload errors are faster and more specific for files that exceed the
     configured hosted limit.

Remaining production gap: the preflight still validates the current single-host
runtime. A fully hosted production path still needs remote object storage,
managed queue workers, structured tracing, and environment-specific secret
management.

### April 25 Public-Face And Convenience Slice

Today focuses on the places beta users see before they ever run a scan:

1. README production introduction
   - The public README hero has been rebuilt as a cleaner product banner with
     one clear promise instead of overlapping title treatments.
   - The intro now explains the input, output, and share value in a compact
     GitHub-friendly layout.

2. Product trust copy
   - The README now leads with decision support, evidence, confidence, and
     limitations instead of broad spatial-reasoning language.
   - The duplicated title/intro pattern has been removed so the page feels more
     intentional and less like stitched-together launch copy.

3. Upload guidance fallback
   - If `/product/upload-guidance` is unavailable, the frontend no longer leaves
     the upload size pill stuck on a loading state.
   - The UI falls back to honest guidance while server-side upload validation
     remains the source of truth.

Remaining production gap: the public face is stronger, but beta acquisition
still needs a hosted landing/waitlist path, a polished sample walkthrough, and
real conversion analytics tied to scan completion and report sharing.

1. Close the truth gap
   - build a 50-100 scan labeled evaluation set across clean rooms, cluttered
     rooms, difficult lighting, and intentionally adversarial cases
   - track false positives, false negatives, missed dangerous hazards, and
     confidence calibration quality
   - require report-path changes to beat or match the current benchmark before
     release

2. Make report quality predictable
   - score uploads for coverage, blur, darkness, shakiness, occlusion, and
     weak motion diversity
   - refuse analysis when the scan is clearly below minimum quality
   - downgrade claims more aggressively when evidence quality is weak

3. Harden privacy and trust
   - add automatic face, screen, document, and family-photo redaction
   - show clear retention policy and deletion controls in the actual product
   - keep "delete my scan and artifacts now" available to end users, not only
     operators

4. Prepare real deployment infrastructure
   - move beyond the current detached local workers to a managed queue
   - move beyond filesystem-backed object storage to remote object storage
   - add observability for queue depth, latency, failure rate, timeout rate,
     storage growth, and provider failures

5. Tighten the hosted beta UX
   - simplify first run to upload -> wait -> read top risks
   - provide a guided sample scan and a capture-quality checklist
   - make the report feel printable, shareable, and immediately actionable

### Beta User Acquisition Track

This is the path that makes the product easier to recommend, easier to share,
and more likely to get repeat use.

1. Keep the wedge narrow
   - continue positioning ATLAS-0 as a room safety scan, not a general 3D
     world-model platform
   - speak to real jobs such as toddler safety, pet hazards, clutter risk, and
     move-in safety checks

2. Make the report share-worthy
   - improve report visuals and export quality for landlord / partner /
     contractor sharing
   - generate a cleaner summary link with top hazards, confidence posture, and
     top actions
   - add a "weekend fix list" output so the report turns into a punch list

3. Build one repeat-use loop
   - let users rescan a room after changes and compare before/after risk
   - show progress over time instead of isolated single-scan output
   - reward improvement with clear deltas, not gamified noise

4. Add personalized modes
   - toddler mode prioritizes reachable, topple, choke, and climb hazards
   - pet mode prioritizes chew, spill, trip, ingestible, and unstable-object
     hazards
   - renter mode prioritizes move-in / move-out issues, blocked exits, broken
     furniture, and liability-relevant evidence

5. Add one light social / daily-use layer
   - room safety score
   - before / after comparison
   - "3 things to fix this weekend"
   - optional "room wins" section so the product can feel useful and uplifting,
     not only critical

### Feature Ideas That Can Create Word of Mouth

These are not random extras. They are the most promising features for making
people come back and tell others.

1. Room Safety Score
   - one score per room
   - explain the drivers of the score
   - make improvement visible after a cleanup or fix session

2. Before / After Improvement Mode
   - compare two scans of the same room
   - call out what got safer, what stayed risky, and what worsened

3. Weekend Fix List
   - convert findings into a short, action-first checklist
   - sort by effort and impact so users can actually finish it

4. Toddler / Pet / Renter Modes
   - re-rank hazards for the user context
   - make the product feel specifically built for the person using it

5. Room Wins
   - add a lightweight "you already did these things right" section
   - makes the report more shareable and less emotionally punishing

### Beta Audiences To Target

The likely early users who can both benefit from the product and talk about it
to others are:

- parents of toddlers
- pet owners
- renters moving in or moving out
- people decluttering or reorganizing rooms
- elder-care households
- organizers, cleaners, or handymen who can use the report with clients

### Release Gates For Public Beta

Before trying to scale acquisition, ATLAS-0 should meet these bars:

1. correctness is benchmarked on a labeled eval set and tracked in CI or
   release checks
2. weak scans are refused or clearly downgraded before report generation
3. privacy controls and delete-my-scan flows are exposed in the product
4. report sharing works without exposing unrelated artifacts or internal state
5. hosted beta observability is sufficient to debug failures quickly

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
