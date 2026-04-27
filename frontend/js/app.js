/**
 * app.js — ATLAS-0 report-first web application.
 */

import * as api from './api.js';
import { SceneViewer } from './scene_viewer.js';
import { UploadView } from './upload.js';

const THEME_STORAGE_KEY = 'atlas0.theme';
const MOTION_STORAGE_KEY = 'atlas0.reducedMotion';
const LOW_CONFIDENCE_STORAGE_KEY = 'atlas0.showLowConfidenceDefault';

function readStoredPreference(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStoredPreference(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {}
}

const VIEW_LABELS = {
  scan: 'Scan',
  report: 'Report',
  scene: 'Scene',
  settings: 'Settings',
};

const state = {
  activeView: 'scan',
  activeJobId: null,
  activeSampleKey: null,
  jobs: new Map(),
  showLowConfidence: readStoredPreference(LOW_CONFIDENCE_STORAGE_KEY) === 'true',
  accessPolicy: null,
  privacyPolicy: null,
  uploadGuidance: null,
  operatorSettings: null,
  reportViewEvents: new Set(),
};

const navButtons = /** @type {NodeListOf<HTMLButtonElement>} */ (
  document.querySelectorAll('.nav-btn[data-view]')
);
const jumpButtons = /** @type {NodeListOf<HTMLButtonElement>} */ (
  document.querySelectorAll('[data-jump-view]')
);
const sampleButtons = /** @type {NodeListOf<HTMLButtonElement>} */ (
  document.querySelectorAll('[data-load-sample]')
);
const viewElements = document.querySelectorAll('.view');
const viewLabel = document.getElementById('hdr-view-label');

const healthStatus = document.getElementById('health-status');
const uploadStatus = document.getElementById('upload-status');
const reportStatus = document.getElementById('report-status');

const processStage = document.getElementById('process-stage');
const processCopy = document.getElementById('process-copy');
const processBar = document.getElementById('process-bar');
const processMeta = document.getElementById('process-meta');
const processGuidance = document.getElementById('process-guidance');
const roomLabelInput = /** @type {HTMLInputElement} */ (document.getElementById('room-label-input'));
const audienceModeInput = /** @type {HTMLSelectElement} */ (document.getElementById('audience-mode-input'));

const recentList = document.getElementById('upload-list');
const recentEmpty = document.getElementById('upload-empty');
const accessBanner = document.getElementById('access-banner');
const accessHelp = document.getElementById('access-help');
const uploadGuidanceCopy = document.getElementById('upload-guidance-copy');
const uploadDurationPill = document.getElementById('upload-duration-pill');
const uploadSizePill = document.getElementById('upload-size-pill');
const privacyPolicy = document.getElementById('privacy-policy');
const operatorPolicy = document.getElementById('operator-policy');
const operatorQueue = document.getElementById('operator-queue');
const operatorSystem = document.getElementById('operator-system');
const operatorEval = document.getElementById('operator-eval');
const operatorProduct = document.getElementById('operator-product');
const operatorPruneBtn = /** @type {HTMLButtonElement} */ (document.getElementById('operator-prune-btn'));
const accessTokenInput = /** @type {HTMLInputElement} */ (document.getElementById('access-token-input'));
const accessTokenSave = /** @type {HTMLButtonElement} */ (document.getElementById('access-token-save'));
const accessTokenClear = /** @type {HTMLButtonElement} */ (document.getElementById('access-token-clear'));
const waitlistEmailInput = /** @type {HTMLInputElement} */ (document.getElementById('waitlist-email-input'));
const waitlistUseCaseInput = /** @type {HTMLInputElement} */ (document.getElementById('waitlist-use-case-input'));
const waitlistSubmitBtn = /** @type {HTMLButtonElement} */ (document.getElementById('waitlist-submit-btn'));
const waitlistNote = document.getElementById('waitlist-note');

const reportHero = document.getElementById('report-hero');
const reportHeroMeta = document.getElementById('report-hero-meta');
const summaryObjects = document.getElementById('summary-objects');
const summaryHazards = document.getElementById('summary-hazards');
const summarySeverity = document.getElementById('summary-severity');
const summaryConfidence = document.getElementById('summary-confidence');
const summaryCoverage = document.getElementById('summary-coverage');
const summarySource = document.getElementById('summary-source');
const fixFirstList = document.getElementById('fix-first-list');
const scanQualityCard = document.getElementById('scan-quality-card');
const reportPostureCard = document.getElementById('report-posture-card');
const reportEvalCard = document.getElementById('report-eval-card');
const weekendFixList = document.getElementById('weekend-fix-list');
const roomWinsList = document.getElementById('room-wins-list');
const lowConfidenceToggle = /** @type {HTMLInputElement} */ (document.getElementById('low-confidence-toggle'));
const settingsLowConfidenceToggle = /** @type {HTMLInputElement} */ (
  document.getElementById('settings-low-confidence-toggle')
);
const findingToggleNote = document.getElementById('finding-toggle-note');
const shareLinkNote = document.getElementById('share-link-note');
const reportHeadline = document.getElementById('report-headline');
const reportSubhead = document.getElementById('report-subhead');
const reportHazards = document.getElementById('risk-report-list');
const reportRecommendations = document.getElementById('rec-list');
const reportEvidence = document.getElementById('evidence-grid');
const trustNotes = document.getElementById('trust-notes');
const exportPdfBtn = /** @type {HTMLAnchorElement} */ (document.getElementById('export-pdf-btn'));
const copyShareBtn = /** @type {HTMLButtonElement} */ (document.getElementById('copy-share-btn'));
const deleteJobBtn = /** @type {HTMLButtonElement} */ (document.getElementById('delete-job-btn'));
const themeToggle = /** @type {HTMLInputElement} */ (document.getElementById('theme-toggle'));
const themeStatus = document.getElementById('theme-status');
const motionToggle = /** @type {HTMLInputElement} */ (document.getElementById('motion-toggle'));
const motionStatus = document.getElementById('motion-status');
const settingsTokenStatus = document.getElementById('settings-token-status');
const settingsTokenClear = /** @type {HTMLButtonElement} */ (document.getElementById('settings-token-clear'));
const settingsSampleBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settings-sample-btn'));

const sceneCanvas = /** @type {HTMLCanvasElement} */ (document.getElementById('scene-canvas'));
const sceneEmpty = document.getElementById('scene-empty');
const sceneObjList = document.getElementById('scene-obj-list');
const sceneViewer = new SceneViewer(sceneCanvas, sceneEmpty, sceneObjList);
let sceneReady = false;

const toast = document.getElementById('toast');
let toastTimer = null;

function showToast(message, timeout = 2600) {
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), timeout);
}

function requestedJobId() {
  return new URLSearchParams(window.location.search).get('job');
}

function requestedSampleKey() {
  return new URLSearchParams(window.location.search).get('sample');
}

function requestedView() {
  const value = new URLSearchParams(window.location.search).get('view');
  return value && VIEW_LABELS[value] ? value : null;
}

function reportDeepLink(job) {
  if (!job?.job_id) {
    return '';
  }
  const relative = job.share_url || (job.sample_key
    ? `/app?view=report&sample=${encodeURIComponent(job.sample_key)}`
    : `/app?view=report&job=${encodeURIComponent(job.job_id)}`);
  return new URL(relative, window.location.origin).toString();
}

function syncUrlState() {
  const url = new URL(window.location.href);
  if (state.activeSampleKey) {
    url.searchParams.delete('job');
    url.searchParams.set('sample', state.activeSampleKey);
  } else if (state.activeJobId) {
    url.searchParams.set('job', state.activeJobId);
    url.searchParams.delete('sample');
  } else {
    url.searchParams.delete('job');
    url.searchParams.delete('sample');
  }

  const view = state.activeView || 'scan';
  if (view === 'scan' && !state.activeJobId) {
    url.searchParams.delete('view');
  } else {
    url.searchParams.set('view', view);
  }

  const next = `${url.pathname}${url.search}${url.hash}`;
  const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (next !== current) {
    window.history.replaceState({}, '', next);
  }
}

async function copyText(value) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }

  const ghost = document.createElement('textarea');
  ghost.value = value;
  ghost.setAttribute('readonly', 'true');
  ghost.style.position = 'absolute';
  ghost.style.left = '-9999px';
  document.body.appendChild(ghost);
  ghost.select();
  document.execCommand('copy');
  document.body.removeChild(ghost);
}

async function trackProductEvent(eventName, extra = {}) {
  try {
    await api.logProductEvent({ event_name: eventName, ...extra });
  } catch {}
}

function applyThemePreference(theme) {
  const nextTheme = theme === 'dark' ? 'dark' : 'light';
  document.documentElement.dataset.theme = nextTheme;
  writeStoredPreference(THEME_STORAGE_KEY, nextTheme);

  if (themeToggle) {
    themeToggle.checked = nextTheme === 'dark';
  }
  if (themeStatus) {
    themeStatus.textContent = nextTheme === 'dark' ? 'Using dark mode' : 'Using light mode';
  }
}

function applyMotionPreference(reduced) {
  const enabled = Boolean(reduced);
  if (enabled) {
    document.documentElement.dataset.reducedMotion = 'true';
  } else {
    delete document.documentElement.dataset.reducedMotion;
  }
  writeStoredPreference(MOTION_STORAGE_KEY, enabled ? 'true' : 'false');

  if (motionToggle) {
    motionToggle.checked = enabled;
  }
  if (motionStatus) {
    motionStatus.textContent = enabled ? 'Animations are reduced' : 'Animations are on';
  }
}

function syncLowConfidenceControls() {
  if (lowConfidenceToggle) {
    lowConfidenceToggle.checked = state.showLowConfidence;
  }
  if (settingsLowConfidenceToggle) {
    settingsLowConfidenceToggle.checked = state.showLowConfidence;
  }
}

function setLowConfidenceVisibility(enabled, persist = true) {
  state.showLowConfidence = Boolean(enabled);
  if (persist) {
    writeStoredPreference(LOW_CONFIDENCE_STORAGE_KEY, state.showLowConfidence ? 'true' : 'false');
  }
  syncLowConfidenceControls();
  renderReport(activeJob());
}

function syncSettingsAccessStatus() {
  const tokenStored = Boolean(api.getAccessToken());
  if (settingsTokenStatus) {
    settingsTokenStatus.textContent = tokenStored
      ? 'Private-beta token stored locally'
      : 'No private-beta token stored';
  }
  if (settingsTokenClear) {
    settingsTokenClear.disabled = !tokenStored;
  }
}

async function loadSampleReport() {
  try {
    const sample = await api.fetchSampleReport();
    upsertJob(sample);
    setActiveJob(sample.job_id);
    switchView('report');
    showToast('Sample report loaded.');
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not load the sample report.', 3600);
  }
}

function switchView(id) {
  state.activeView = id;
  navButtons.forEach((button) => {
    button.classList.toggle('active', button.dataset.view === id);
  });
  viewElements.forEach((element) => {
    element.classList.toggle('active', element.id === `view-${id}`);
  });
  viewLabel.textContent = VIEW_LABELS[id] || id;

  if (id === 'scene') {
    ensureScene();
    sceneViewer.refresh();
  }
  syncUrlState();
}

function ensureScene() {
  if (!sceneReady) {
    sceneViewer.init();
    sceneReady = true;
  }
}

function activeJob() {
  return state.activeJobId ? state.jobs.get(state.activeJobId) : null;
}

function setActiveJob(jobId) {
  state.activeJobId = jobId;
  state.activeSampleKey = state.jobs.get(jobId)?.sample_key || null;
  renderUploads();
  renderProcessing(activeJob());
  renderReport(activeJob());
  syncUrlState();
}

function upsertJob(job) {
  state.jobs.set(job.job_id, job);
  if (!state.activeJobId || job.job_id === state.activeJobId || job.status === 'complete') {
    state.activeJobId = job.job_id;
  }

  renderUploads();
  renderProcessing(job);

  if (job.status === 'complete') {
    renderReport(job);
    switchView('report');
    showToast(job.is_sample ? 'Sample report ready.' : 'Scan complete. Report is ready.');
  } else if (job.status === 'error') {
    showToast(job.error || 'Scan failed.', 3600);
  }
}

function removeJob(jobId) {
  state.jobs.delete(jobId);
  if (state.activeJobId === jobId) {
    const nextJob = [...state.jobs.values()].sort((a, b) => b.job_id.localeCompare(a.job_id))[0] || null;
    state.activeJobId = nextJob?.job_id || null;
  }
  renderUploads();
  renderProcessing(activeJob());
  renderReport(activeJob());
}

function renderUploads() {
  const jobs = [...state.jobs.values()]
    .filter((job) => !job.is_sample)
    .sort((a, b) => b.job_id.localeCompare(a.job_id));
  recentEmpty.style.display = jobs.length ? 'none' : '';
  recentEmpty.textContent = jobs.length
    ? ''
    : 'No scans yet. Start with one room you know well, keep the camera steady, and the first report will appear here.';

  recentList.innerHTML = jobs.map((job) => {
    const summary = job.summary || {};
    const hazardLabel = summary.top_hazard_label || 'No hazards yet';
    const roomLabel = job.room_label || summary.room_label || '';
    const roomScore = typeof summary.room_score === 'number' ? `${summary.room_score}/100 room score` : '';
    const audienceLabel = summary.audience_label || '';
    const activeClass = job.job_id === state.activeJobId ? 'active' : '';
    return `
      <button class="scan-card ${activeClass}" data-job-id="${job.job_id}">
        <div class="scan-card-top">
          <span class="scan-file">${escapeHtml(job.filename)}</span>
          <span class="scan-pill ${job.status}">${escapeHtml(job.status)}</span>
        </div>
        <div class="scan-card-meta">
          <span>${Math.round((job.progress || 0) * 100)}%</span>
          <span>${escapeHtml(job.stage || 'queued')}</span>
          <span>${escapeHtml(roomLabel || hazardLabel)}</span>
          ${audienceLabel ? `<span>${escapeHtml(audienceLabel)}</span>` : ''}
          ${roomScore ? `<span>${escapeHtml(roomScore)}</span>` : ''}
        </div>
      </button>
    `;
  }).join('');

  recentList.querySelectorAll('.scan-card').forEach((button) => {
    button.addEventListener('click', () => {
      setActiveJob(button.dataset.jobId);
      if (activeJob()?.status === 'complete') {
        switchView('report');
      }
    });
  });
}

function renderProcessing(job) {
  if (!job) {
    processStage.textContent = 'Ready for upload';
    processCopy.textContent = 'Upload a short walkthrough and ATLAS-0 will build a calmer, evidence-backed room safety report.';
    processBar.style.width = '0%';
    processMeta.textContent = 'No active scan yet';
    processGuidance.innerHTML = renderProcessGuidance(null);
    uploadStatus.textContent = 'Ready for first scan';
    reportStatus.textContent = 'No report yet';
    return;
  }

  if (job.is_sample) {
    processStage.textContent = 'Sample report loaded';
    processCopy.textContent = 'The built-in walkthrough is open so you can explore the report before recording your own room.';
    processBar.style.width = '100%';
    processMeta.textContent = `${escapeHtml(job.room_label || 'Sample room')} · ${escapeHtml(job.summary?.audience_label || 'General home safety')} · sample`;
    processGuidance.innerHTML = `
      <article class="guidance-card">
        <strong>Use this as a reference</strong>
        <p>This sample shows the tone, evidence, and follow-through structure we want first-time users to understand quickly.</p>
      </article>
      <article class="guidance-card">
        <strong>What to do next</strong>
        <p>When you are ready, switch back to Scan and record one real room with the same calm, steady walkthrough style.</p>
      </article>
    `;
    uploadStatus.textContent = 'Sample report loaded';
    reportStatus.textContent = 'Sample report ready';
    return;
  }

  const statusLabel = `${capitalize(job.stage || 'upload')} · ${Math.round((job.progress || 0) * 100)}%`;
  processStage.textContent = statusLabel;
  processBar.style.width = `${Math.round((job.progress || 0) * 100)}%`;

  if (job.status === 'complete') {
    processCopy.textContent = 'Your report is ready with top hazards, evidence frames, and practical next steps.';
  } else if (job.status === 'error') {
    processCopy.textContent = job.error || 'The scan could not be processed.';
  } else {
    processCopy.textContent = 'ATLAS-0 is analyzing the upload, grounding the findings, and assembling the report.';
  }

  processMeta.textContent = `${escapeHtml(job.filename)} · ${escapeHtml(job.room_label || 'Unlabeled room')} · ${escapeHtml(job.summary?.audience_label || 'General home safety')} · ${escapeHtml(job.status)}`;
  processGuidance.innerHTML = renderProcessGuidance(job);
  uploadStatus.textContent = job.status === 'complete' ? 'Latest scan complete' : 'Scan in progress';
  reportStatus.textContent = job.status === 'complete' ? 'Report ready' : 'Waiting for report';
}

function renderReport(job) {
  syncLowConfidenceControls();
  if (!job || job.status !== 'complete') {
    reportHero.classList.add('empty');
    reportHeadline.textContent = 'No report yet';
    reportSubhead.textContent = 'Run one room scan to review hazards, evidence, and practical next steps in a single place.';
    reportHeroMeta.innerHTML = `
      <span class="soft-badge">Action-first</span>
      <span class="soft-badge">Evidence-backed</span>
      <span class="soft-badge">Shareable PDF</span>
    `;
    summaryObjects.textContent = '0';
    summaryHazards.textContent = '0';
    summarySeverity.textContent = '—';
    summaryConfidence.textContent = '—';
    summaryCoverage.textContent = '—';
    summarySource.textContent = '—';
    fixFirstList.innerHTML = emptyMarkup('Priority actions will appear here once a scan finishes.');
    scanQualityCard.innerHTML = emptyMarkup('Scan quality diagnostics will appear here after the upload is processed.');
    reportPostureCard.innerHTML = emptyMarkup('Report posture details will appear here after a completed scan.');
    reportEvalCard.innerHTML = emptyMarkup('Feedback and review coverage will appear here after a completed scan.');
    weekendFixList.innerHTML = emptyMarkup('Weekend-friendly fixes will appear here after a completed scan.');
    roomWinsList.innerHTML = emptyMarkup('Positive scan signals will appear here after a completed scan.');
    reportHazards.innerHTML = emptyMarkup('Hazards will appear here once ATLAS-0 has something evidence-backed to show.');
    reportRecommendations.innerHTML = emptyMarkup('Recommendations will appear here after a completed scan.');
    reportEvidence.innerHTML = emptyMarkup('Evidence frames will appear here after a completed scan.');
    trustNotes.innerHTML = emptyMarkup('Trust notes will appear here after a completed scan.');
    findingToggleNote.textContent = 'Low-confidence findings stay hidden until a report is ready.';
    shareLinkNote.textContent = 'Share links open the exact report view. Hosted environments still respect token protection.';
    exportPdfBtn.removeAttribute('href');
    exportPdfBtn.classList.add('disabled');
    copyShareBtn.classList.add('disabled');
    copyShareBtn.disabled = true;
    deleteJobBtn.classList.add('disabled');
    deleteJobBtn.disabled = true;
    return;
  }

  reportHero.classList.remove('empty');
  const summary = job.summary || {};
  const hazards = job.risks || [];
  const visibleHazards = state.showLowConfidence ? hazards : hazards.filter((risk) => !isLowConfidenceRisk(risk));
  const hiddenCount = Math.max(0, hazards.length - visibleHazards.length);
  const fixFirst = job.fix_first || [];
  const recommendations = job.recommendations || [];
  const evidence = job.evidence_frames || [];
  const scanQuality = job.scan_quality || {};
  const notes = job.trust_notes || [];
  const evaluation = job.evaluation_summary || {};
  const resolution = job.resolution_summary || {};
  const comparison = job.room_comparison || null;
  const roomLabel = job.room_label || summary.room_label || '';
  const audienceLabel = summary.audience_label || 'General home safety';
  const weekendFixes = job.weekend_fix_list || [];
  const roomWins = job.room_wins || [];
  const viewKey = `${job.sample_key || 'job'}:${job.job_id}`;
  if (!state.reportViewEvents.has(viewKey)) {
    state.reportViewEvents.add(viewKey);
    trackProductEvent('report_viewed', {
      surface: 'report_view',
      job_id: job.job_id,
      sample_key: job.sample_key || null,
      audience_mode: job.audience_mode || 'general',
      room_labeled: Boolean(roomLabel),
    });
  }

  reportHeadline.textContent = summary.headline || (summary.top_hazard_label
    ? `Top concern: ${summary.top_hazard_label}`
    : 'Hazard report');
  reportSubhead.textContent = summary.overview || `${roomLabel || job.filename} · ${audienceLabel} · ${summary.confidence_label || 'Approximate grounding'}`;
  reportHeroMeta.innerHTML = renderHeroBadges(summary, roomLabel, comparison, resolution, job.is_sample);

  summaryObjects.textContent = String(summary.object_count || 0);
  summaryHazards.textContent = String(summary.hazard_count || 0);
  summarySeverity.textContent = capitalize(summary.top_severity || 'none');
  summaryConfidence.textContent = summary.scan_quality_label
    ? `${summary.confidence_label || 'Approximate grounding'} · ${summary.scan_quality_label} scan`
    : summary.confidence_label || 'Approximate grounding';
  summaryCoverage.textContent = capitalize(summary.coverage_label || 'unknown');
  summarySource.textContent = summary.scene_source || 'unknown';
  findingToggleNote.textContent = hiddenCount > 0 && !state.showLowConfidence
    ? `${hiddenCount} lower-confidence finding${hiddenCount === 1 ? '' : 's'} hidden to keep the report focused.`
    : 'Showing every finding, including weak or approximate ones.';

  fixFirstList.innerHTML = fixFirst.length
    ? fixFirst.map((action, index) => `
        <article class="fix-first-card">
          <div class="report-card-top">
            <h3>${index + 1}. ${escapeHtml(action.title || 'Fix first')}</h3>
            <span class="severity-pill ${action.severity || 'low'}">${escapeHtml(action.severity || 'low')}</span>
          </div>
          <div class="report-copy-block">
            <strong>What to do next</strong>
            <span>${escapeHtml(action.action || '')}</span>
          </div>
          <div class="report-copy-block">
            <strong>Why this moved to the top</strong>
            <span>${escapeHtml(action.why || '')}</span>
          </div>
          <div class="report-card-meta">
            <span>${escapeHtml(action.location || 'scan area')}</span>
            <span>${escapeHtml(action.confidence_label || 'weak')} evidence</span>
          </div>
        </article>
      `).join('')
    : emptyMarkup('No high-priority actions were generated for this scan.');

  scanQualityCard.innerHTML = renderScanQuality(scanQuality);
  reportPostureCard.innerHTML = renderReportPosture(summary, scanQuality, job);
  reportEvalCard.innerHTML = renderEvaluationSummary(evaluation, job);
  weekendFixList.innerHTML = weekendFixes.length
    ? weekendFixes.map((item, index) => `
        <article class="report-card recommendation">
          <div class="report-card-top">
            <h3>${index + 1}. ${escapeHtml(item.title || 'Weekend fix')}</h3>
            <span class="severity-pill medium">${escapeHtml(item.effort || '20-30 minutes')}</span>
          </div>
          <div class="report-copy-block">
            <strong>Quick task</strong>
            <span>${escapeHtml(item.task || '')}</span>
          </div>
          <div class="report-copy-block">
            <strong>Why it helps</strong>
            <span>${escapeHtml(item.benefit || '')}</span>
          </div>
          <div class="report-card-meta">
            <span>${escapeHtml(item.location || 'scan area')}</span>
            <span>${escapeHtml(item.audience_label || audienceLabel)}</span>
          </div>
        </article>
      `).join('')
    : emptyMarkup('No weekend-friendly fixes were generated for this scan.');
  roomWinsList.innerHTML = roomWins.length
    ? roomWins.map((win) => `
        <article class="report-card recommendation">
          <div class="report-card-top">
            <h3>${escapeHtml(win.title || 'Positive sign')}</h3>
            <span class="severity-pill low">Calm signal</span>
          </div>
          <p>${escapeHtml(win.detail || '')}</p>
        </article>
      `).join('')
    : emptyMarkup('No positive scan signals were generated for this report.');

  reportHazards.innerHTML = visibleHazards.length
    ? visibleHazards.map((risk) => `
        <article class="report-card hazard ${risk.severity || 'low'}">
          <div class="report-card-top">
            <h3>${escapeHtml(risk.hazard_title || risk.object_label || 'Object')}</h3>
            <span class="severity-pill ${risk.severity || 'low'}">${escapeHtml(risk.severity || 'low')}</span>
          </div>
          <div class="report-copy-block">
            <strong>What is wrong</strong>
            <span>${escapeHtml(risk.what || risk.description || '')}</span>
          </div>
          <div class="report-copy-block">
            <strong>Why it matters</strong>
            <span>${escapeHtml(risk.why_it_matters || risk.why || '')}</span>
          </div>
          <div class="report-copy-block">
            <strong>What to do next</strong>
            <span>${escapeHtml(risk.what_to_do_next || risk.recommendation || '')}</span>
          </div>
          <ul class="signal-list">
            ${(risk.reasoning?.signals || []).slice(0, 3).map((signal) => `<li>${escapeHtml(signal)}</li>`).join('')}
          </ul>
          ${renderReplayPreview(risk)}
          ${renderReasoningPanel(risk)}
          <div class="report-card-meta">
            <span>${Math.round((risk.risk_score || 0) * 100)} risk score</span>
            <span>${escapeHtml(risk.location_label || 'Approximate location')}</span>
            <span>${escapeHtml(risk.hazard_code || 'finding')}</span>
            <span>${escapeHtml(risk.confidence_label || 'weak')} evidence</span>
            <span>${escapeHtml(risk.reasoning?.support_summary || 'Limited support')}</span>
            ${risk.follow_up_status ? `<span>${escapeHtml(formatFollowUpLabel(risk.follow_up_status))}</span>` : ''}
          </div>
          ${job.is_sample ? '' : `
            <div class="feedback-row" data-follow-up-controls data-job-id="${job.job_id}" data-hazard-code="${escapeHtml(risk.hazard_code || '')}" data-object-id="${escapeHtml(risk.object_id || '')}" data-active-status="${escapeHtml(risk.follow_up_status || '')}">
              ${renderFollowUpButton('resolved', risk.follow_up_status)}
              ${renderFollowUpButton('monitor', risk.follow_up_status)}
              ${renderFollowUpButton('ignored', risk.follow_up_status)}
            </div>
            <div class="feedback-row" data-feedback-controls data-job-id="${job.job_id}" data-hazard-code="${escapeHtml(risk.hazard_code || '')}" data-object-id="${escapeHtml(risk.object_id || '')}">
              ${renderFeedbackButton('useful', risk.latest_feedback)}
              ${renderFeedbackButton('wrong', risk.latest_feedback)}
              ${renderFeedbackButton('duplicate', risk.latest_feedback)}
            </div>
          `}
        </article>
      `).join('')
    : emptyMarkup(
        summary.analysis_outcome === 'rejected'
          ? 'ATLAS-0 rejected this scan as a normal room report. Follow the retry guidance and rescan before trusting any “all clear” takeaway.'
          : hazards.length
          ? 'Only lower-confidence findings remain. Toggle them on if you want the full raw report.'
          : summary.rescan_recommended
            ? 'No high-confidence hazards were detected, but this scan had limited coverage. Rescan before treating the room as low risk.'
            : 'No high-confidence hazards were detected. This is still a screening result, not a safety clearance.',
      );

  reportRecommendations.innerHTML = recommendations.length
    ? recommendations.map((rec) => `
        <article class="report-card recommendation">
          <div class="report-card-top">
            <h3>${escapeHtml(rec.title || 'Recommendation')}</h3>
            <span class="severity-pill ${rec.priority || 'low'}">${escapeHtml(rec.priority || 'low')}</span>
          </div>
          <p>${escapeHtml(rec.action || '')}</p>
          <div class="report-card-meta">
            <span>${escapeHtml(rec.location || 'scan area')}</span>
            <span>${escapeHtml(rec.why || '')}</span>
          </div>
        </article>
      `).join('')
    : emptyMarkup('No actions were generated for this scan.');

  reportEvidence.innerHTML = evidence.length
    ? evidence.map((frame) => `
        <article class="evidence-card">
          <img src="${api.withAccessToken(frame.image_url || '')}" alt="${escapeHtml(frame.caption || 'Evidence frame')}" />
          <div class="evidence-copy">
            <strong>${escapeHtml(frame.caption || 'Evidence frame')}</strong>
            <span>${Math.round((frame.confidence || 0) * 100)}% label confidence</span>
            <span>${escapeHtml(formatEvidenceMeta(frame))}</span>
          </div>
        </article>
      `).join('')
    : emptyMarkup('No evidence frames were stored for this scan.');

  trustNotes.innerHTML = notes.length
    ? notes.map((note) => `<li>${escapeHtml(note)}</li>`).join('')
    : '<li>No additional trust notes.</li>';

  exportPdfBtn.href = job.is_sample ? api.sampleReportPdfUrl() : api.reportPdfUrl(job.job_id);
  exportPdfBtn.classList.remove('disabled');
  copyShareBtn.classList.remove('disabled');
  copyShareBtn.disabled = false;
  deleteJobBtn.classList.toggle('disabled', Boolean(job.is_sample));
  deleteJobBtn.disabled = Boolean(job.is_sample);
  const expiryNote = job.expires_at
    ? ` Artifacts are scheduled to expire on ${new Date(job.expires_at).toLocaleDateString()}.`
    : '';
  shareLinkNote.textContent = job.is_sample
    ? 'Share link opens this built-in sample report view.'
    : `${summary.share_summary || 'Share link opens this exact report view.'}${expiryNote}`;
  if (!job.is_sample) {
    attachFollowUpHandlers(job.job_id);
    attachFeedbackHandlers(job.job_id);
    attachEvaluationHandlers(job.job_id);
  }
}

function renderHeroBadges(summary, roomLabel, comparison, resolution, isSample) {
  const badges = [
    summary.audience_label || 'General home safety',
    summary.report_posture || 'screening',
    summary.coverage_label ? `${summary.coverage_label} coverage` : 'Coverage pending',
    summary.rescan_recommended ? 'Rescan recommended' : 'Evidence-backed',
  ];
  if (isSample) {
    badges.unshift('Built-in sample');
  }
  if (roomLabel) {
    badges.unshift(roomLabel);
  }
  if (typeof summary.room_score === 'number') {
    badges.push(`${summary.room_score}/100 room score`);
  }
  if (Number(resolution?.resolved_count || 0) > 0) {
    badges.push(`${resolution.resolved_count} resolved`);
  }
  if (comparison?.score_delta) {
    const delta = Number(comparison.score_delta || 0);
    badges.push(`${delta > 0 ? '+' : ''}${delta} vs last scan`);
  }
  return badges.map((badge) => `<span class="soft-badge">${escapeHtml(badge)}</span>`).join('');
}

function renderReplayPreview(risk) {
  const replay = risk?.replay;
  if (!replay?.image_url) {
    return '';
  }

  return `
    <div class="finding-replay">
      <div class="finding-replay-copy">
        <strong>Evidence replay</strong>
        <span>${escapeHtml(replay.caption || 'Short replay from the strongest supporting crops.')}</span>
      </div>
      <img src="${api.withAccessToken(replay.image_url)}" alt="${escapeHtml(replay.caption || 'Finding replay')}" />
      <div class="finding-replay-meta">
        <span>${Number(replay.frame_count || 0)} supporting frame${Number(replay.frame_count || 0) === 1 ? '' : 's'}</span>
        <span>${escapeHtml(risk.location_label || 'scan area')}</span>
      </div>
    </div>
  `;
}

function renderReasoningPanel(risk) {
  const reasoning = risk?.reasoning || {};
  const objectSnapshot = reasoning.object_snapshot || {};
  const ruleHits = Array.isArray(reasoning.rule_hits) ? reasoning.rule_hits : [];
  const evidenceIds = Array.isArray(reasoning.evidence_ids) ? reasoning.evidence_ids : [];
  const confidenceReasons = Array.isArray(reasoning.confidence_reasons) ? reasoning.confidence_reasons : [];
  const facts = [];

  if (objectSnapshot.material) {
    facts.push(`Material: ${objectSnapshot.material}`);
  }
  if (Number(objectSnapshot.estimated_height_m || 0) > 0) {
    facts.push(`Estimated height ${Number(objectSnapshot.estimated_height_m).toFixed(2)} m`);
  }
  if (Number(objectSnapshot.estimated_width_m || 0) > 0) {
    facts.push(`Estimated width ${Number(objectSnapshot.estimated_width_m).toFixed(2)} m`);
  }
  if (Number(objectSnapshot.observation_count || 0) > 0) {
    facts.push(`${Number(objectSnapshot.observation_count)} supporting observation${Number(objectSnapshot.observation_count) === 1 ? '' : 's'}`);
  }

  if (!ruleHits.length && !facts.length && !evidenceIds.length && !confidenceReasons.length) {
    return '';
  }

  return `
    <details class="reasoning-panel">
      <summary>Why ATLAS-0 surfaced this finding</summary>
      <div class="reasoning-grid">
        ${ruleHits.length ? `
          <div class="reasoning-block">
            <strong>Triggered rules</strong>
            <ul>
              ${ruleHits.map((hit) => `<li>${escapeHtml(hit)}</li>`).join('')}
            </ul>
          </div>
        ` : ''}
        ${facts.length ? `
          <div class="reasoning-block">
            <strong>Object snapshot</strong>
            <ul>
              ${facts.map((fact) => `<li>${escapeHtml(fact)}</li>`).join('')}
            </ul>
          </div>
        ` : ''}
        ${evidenceIds.length ? `
          <div class="reasoning-block">
            <strong>Evidence references</strong>
            <div class="reasoning-chips">
              ${evidenceIds.map((id) => `<span>${escapeHtml(id)}</span>`).join('')}
            </div>
          </div>
        ` : ''}
        ${confidenceReasons.length ? `
          <div class="reasoning-block">
            <strong>Confidence calibration</strong>
            <ul>
              ${confidenceReasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join('')}
            </ul>
          </div>
        ` : ''}
      </div>
    </details>
  `;
}

function renderProcessGuidance(job) {
  if (!job) {
    return `
      <article class="guidance-card">
        <strong>Best first scan</strong>
        <p>Choose one room, move steadily, and keep shelves, tables, and corners visible for a moment so weaker hazards do not look stronger than the scan supports.</p>
      </article>
      <article class="guidance-card">
        <strong>What you’ll get</strong>
        <p>A ranked screening report with top hazards, evidence crops, approximate locations, confidence labels, and a downloadable PDF.</p>
      </article>
    `;
  }

  if (job.status === 'error') {
    return `
      <article class="guidance-card">
        <strong>Scan needs another try</strong>
        <p>${escapeHtml(job.error || 'The scan could not be processed this time.')}</p>
        <ul class="guidance-list">
          <li>Retry with one room only.</li>
          <li>Keep the motion slower and steadier than feels natural.</li>
          <li>If lighting was poor, add light before rescanning.</li>
        </ul>
      </article>
    `;
  }

  if (job.status === 'complete') {
    const scanQuality = job.scan_quality || {};
    const guidance = scanQuality.retry_guidance || [];
    const summary = job.summary || {};
    return `
      <article class="guidance-card">
        <strong>${escapeHtml(scanQuality.rescan_recommended ? 'Use this report carefully' : 'Report is ready')}</strong>
        <p>${escapeHtml(summary.screening_statement || 'This report flags likely hazards from the uploaded scan. It does not certify that the room is safe.')}</p>
      </article>
      <article class="guidance-card">
        <strong>Next best move</strong>
        <p>${escapeHtml(scanQuality.capture_summary || 'Review the top hazards and trust notes before acting on smaller details.')}</p>
        ${guidance.length ? `<ul class="guidance-list">${guidance.slice(0, 2).map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>` : ''}
      </article>
    `;
  }

  return `
    <article class="guidance-card">
      <strong>What ATLAS-0 is doing now</strong>
      <p>${escapeHtml(stageExplanation(job.stage))}</p>
    </article>
    <article class="guidance-card">
      <strong>While you wait</strong>
      <p>Keep this tab open. When the report is ready, we’ll switch you straight into the report view.</p>
    </article>
  `;
}

function emptyMarkup(message) {
  return `<div class="empty-card">${escapeHtml(message)}</div>`;
}

function renderPolicyItems(items) {
  return items.map((item) => `
    <div class="policy-item">
      <span>${escapeHtml(item.label)}</span>
      <strong>${escapeHtml(item.value)}</strong>
    </div>
  `).join('');
}

function renderReleaseGates(releaseGates) {
  if (!releaseGates || !Array.isArray(releaseGates.gates)) {
    return '';
  }

  return `
    <p class="subsection-label">Release gates</p>
    ${renderPolicyItems(releaseGates.gates.map((gate) => ({
      label: gate.label || gate.id || 'Gate',
      value: `${gate.passed ? 'Pass' : 'Open'} · ${formatGateValue(gate.actual)} / ${formatGateValue(gate.target)}`,
    })))}
    <p class="meta-copy">${escapeHtml(releaseGates.summary || '')}</p>
  `;
}

function applyUploadGuidance(guidance) {
  const recommended = guidance?.recommended_duration_seconds || {};
  const minSeconds = Number(recommended.min || 20);
  const maxSeconds = Number(recommended.max || guidance?.max_video_duration_seconds || 60);
  if (uploadDurationPill) {
    uploadDurationPill.textContent = `${minSeconds}-${maxSeconds} seconds`;
  }
  if (uploadSizePill) {
    uploadSizePill.textContent = guidance?.max_upload_bytes
      ? `${formatBytes(guidance.max_upload_bytes)} max`
      : 'Limit checked on upload';
  }
  if (uploadGuidanceCopy) {
    const checklist = Array.isArray(guidance?.checklist) ? guidance.checklist : [];
    uploadGuidanceCopy.textContent = checklist[1]
      || 'Record one bright, steady room walkthrough before uploading.';
  }
}

function renderAccessPanels(errorMessage = '') {
  const access = state.accessPolicy;
  const privacy = state.privacyPolicy;
  const settings = state.operatorSettings;
  const tokenStored = Boolean(api.getAccessToken());

  accessTokenClear.disabled = !tokenStored;
  syncSettingsAccessStatus();

  if (!privacy) {
    privacyPolicy.innerHTML = emptyMarkup('Privacy defaults are unavailable right now.');
  } else {
    privacyPolicy.innerHTML = `
      <p class="subsection-label">User-visible privacy</p>
      ${renderPolicyItems([
        { label: 'Retention window', value: `${privacy.retention_days} day(s)` },
        { label: 'Keep originals', value: privacy.save_original_uploads ? 'Yes' : 'No by default' },
        { label: 'Text redaction', value: privacy.text_redaction_enabled ? 'Enabled' : 'Disabled' },
        { label: 'Delete support', value: privacy.delete_supported ? 'Available in report view' : 'Unavailable' },
      ])}
      <p class="meta-copy">${escapeHtml(privacy.summary || '')}</p>
    `;
  }

  if (!access) {
    accessBanner.className = 'status-banner';
    accessBanner.innerHTML = '<strong>Could not load access policy.</strong>';
    accessHelp.textContent = 'The hosted access policy could not be loaded from the API.';
    operatorPolicy.innerHTML = emptyMarkup('Operator policy details are unavailable right now.');
    operatorQueue.innerHTML = emptyMarkup('Queue diagnostics are unavailable right now.');
    operatorSystem.innerHTML = emptyMarkup('Deployment diagnostics are unavailable right now.');
    operatorEval.innerHTML = emptyMarkup('Evaluation metrics are unavailable right now.');
    operatorProduct.innerHTML = emptyMarkup('Product metrics are unavailable right now.');
    operatorPruneBtn.disabled = true;
    return;
  }

  const locked = access.requires_token && !settings;
  accessBanner.className = `status-banner ${locked ? 'locked' : 'ready'}`;
  accessBanner.innerHTML = locked
    ? '<strong>Hosted upload/report access is locked.</strong> Add the private-beta token to use protected scans, reports, and diagnostics.'
    : access.requires_token
      ? '<strong>Hosted access is unlocked.</strong> Protected upload/report endpoints are available with the stored token.'
      : access.mode === 'loopback'
        ? '<strong>Local access is open.</strong> Loopback requests can use upload/report flows without a token.'
        : '<strong>Upload/report access is restricted.</strong> This environment is not accepting unauthenticated hosted requests.';

  accessHelp.textContent = locked
    ? (errorMessage || 'Your token stays in this browser only and is sent as a Bearer token for protected Atlas-0 endpoints.')
    : tokenStored
      ? 'A token is stored locally for this browser session and will be appended to protected API requests.'
      : 'No token is stored locally. You only need one when the hosted environment requires it.';

  if (!settings) {
    operatorPolicy.innerHTML = emptyMarkup(
      access.requires_token
        ? 'Enter a valid token to load retention, queue, and access settings.'
        : 'Operator settings are not available yet.',
    );
    operatorQueue.innerHTML = emptyMarkup('Queue diagnostics will appear here once operator access is available.');
    operatorSystem.innerHTML = emptyMarkup('Deployment diagnostics will appear here once operator access is available.');
    operatorEval.innerHTML = emptyMarkup('Evaluation metrics will appear here once operator access is available.');
    operatorProduct.innerHTML = emptyMarkup('Product metrics will appear here once operator access is available.');
    operatorPruneBtn.disabled = true;
    return;
  }

  operatorPruneBtn.disabled = false;

  operatorPolicy.innerHTML = renderPolicyItems([
    { label: 'Access mode', value: settings.access.mode === 'token' ? 'Token protected' : settings.access.mode === 'loopback' ? 'Loopback-friendly' : 'Restricted' },
    { label: 'Primary provider', value: settings.providers.primary_provider || 'unknown' },
    { label: 'Fallback provider', value: settings.providers.fallback_provider || 'None' },
    { label: 'Worker mode', value: settings.uploads.worker_mode || settings.system?.worker_mode || 'unknown' },
    { label: 'Job listing', value: settings.access.enable_job_listing ? 'Enabled' : 'Direct job IDs only' },
    { label: 'Retention window', value: `${settings.uploads.retention_days} day(s)` },
    { label: 'Keep originals', value: settings.uploads.save_original_uploads ? 'Yes' : 'No' },
    { label: 'Artifact backend', value: settings.uploads.artifact_backend || 'unknown' },
    { label: 'Queue depth limit', value: String(settings.uploads.max_queue_depth) },
    { label: 'Retry budget', value: `${settings.uploads.max_job_attempts} attempt(s)` },
  ]);

  operatorQueue.innerHTML = renderPolicyItems([
    { label: 'Workers', value: String(settings.queue.worker_count) },
    { label: 'Configured capacity', value: String(settings.queue.configured_capacity || settings.uploads.max_concurrent_jobs || 0) },
    { label: 'Queued jobs', value: String(settings.queue.queued_jobs) },
    { label: 'Processing jobs', value: String(settings.queue.processing_jobs) },
    { label: 'Failed jobs', value: String(settings.queue.failed_jobs) },
    { label: 'Active claims', value: String(settings.storage.active_claims || 0) },
    { label: 'Stored jobs', value: String(settings.storage.persisted_jobs) },
    { label: 'Storage budget', value: formatBytes(settings.storage.byte_budget || 0) },
    { label: 'Disk used', value: formatBytes(settings.storage.bytes_used || 0) },
    { label: 'Budget used', value: `${settings.storage.usage_percent || 0}%` },
  ]);

  const startupChecks = Array.isArray(settings.system?.startup_checks) ? settings.system.startup_checks : [];
  const startupSummary = settings.system?.startup_summary || '';
  const recentFailures = Array.isArray(settings.system?.recent_failures) ? settings.system.recent_failures : [];
  operatorSystem.innerHTML = `
    <p class="subsection-label">Deployment readiness</p>
    ${renderPolicyItems([
      { label: 'Status', value: settings.system?.deployment_ready ? 'Ready' : 'Needs operator fixes' },
      { label: 'Worker mode', value: settings.system?.worker_mode || 'unknown' },
      { label: 'Service uptime', value: `${Math.round(settings.system?.uptime_seconds || 0)}s` },
      { label: 'Active workers', value: String(settings.system?.active_workers || 0) },
      { label: 'Artifact backend', value: settings.system?.artifact_backend || settings.uploads.artifact_backend || 'unknown' },
      { label: 'Object store root', value: settings.system?.artifact_object_dir || settings.uploads.artifact_object_dir || 'local job storage' },
      { label: 'Storage root', value: settings.system?.storage_root || 'unknown' },
      { label: 'Recent failures', value: String(recentFailures.length) },
    ])}
    <p class="meta-copy">${escapeHtml(startupSummary)}</p>
    ${startupChecks.length ? renderPolicyItems(startupChecks.map((check) => ({
      label: check.name || 'Check',
      value: `${check.status || 'unknown'} · ${check.detail || ''}`,
    }))) : ''}
    ${recentFailures.length ? `<p class="subsection-label">Recent job failures</p>${renderPolicyItems(recentFailures.map((failure) => ({
      label: `${failure.job_id || 'job'} · ${failure.stage || 'stage'}`,
      value: `${failure.will_retry ? 'Retrying' : 'Terminal'} · ${failure.error || 'Unknown failure'}`,
    })))}` : '<p class="meta-copy">No recent terminal worker failures recorded.</p>'}
  `;

  operatorEval.innerHTML = `
    <p class="subsection-label">Evaluation loop</p>
    ${renderPolicyItems([
      { label: 'Reviewed jobs', value: String(settings.evaluation.reviewed_jobs || 0) },
      { label: 'Benchmarked jobs', value: String(settings.evaluation.benchmarked_jobs || 0) },
      { label: 'Benchmark match rate', value: `${Math.round((settings.evaluation.benchmark_match_rate || 0) * 100)}%` },
      { label: 'Missed-hazard jobs', value: String(settings.evaluation.jobs_with_missed_hazards || 0) },
      { label: 'False-positive job rate', value: `${Math.round((settings.evaluation.false_positive_job_rate || 0) * 100)}%` },
      { label: 'Average review coverage', value: `${Math.round((settings.evaluation.avg_review_coverage || 0) * 100)}%` },
      { label: 'Committed eval fixtures', value: String(settings.evaluation.seed_fixture_count || 0) },
      { label: 'Saved eval candidates', value: String(settings.evaluation.saved_eval_candidates || 0) },
      { label: 'Review-ready eval cases', value: `${settings.evaluation.available_eval_cases || 0} / ${settings.evaluation.target_corpus_size || 0}` },
    ])}
    ${renderReleaseGates(settings.evaluation.release_gates)}
  `;

  operatorProduct.innerHTML = `
    <p class="subsection-label">Beta product loop</p>
    ${renderPolicyItems([
      { label: 'Upload success rate', value: `${Math.round((settings.product.upload_success_rate || 0) * 100)}%` },
      { label: 'Rescan recommended rate', value: `${Math.round((settings.product.rescan_recommended_rate || 0) * 100)}%` },
      { label: 'Report usefulness rate', value: `${Math.round((settings.product.report_usefulness_rate || 0) * 100)}%` },
      { label: 'Average report time', value: `${settings.product.avg_report_seconds || 0}s` },
      { label: 'Completed jobs', value: String(settings.product.completed_jobs || 0) },
      { label: 'Terminal jobs', value: String(settings.product.terminal_jobs || 0) },
      { label: 'Labeled rooms', value: String(settings.product.labeled_rooms || 0) },
      { label: 'Repeat-scan rooms', value: String(settings.product.repeat_scan_rooms || 0) },
      { label: 'Waitlist signups', value: String(settings.product.waitlist_signups || 0) },
      { label: 'Sample report opens', value: String(settings.product.sample_report_opens || 0) },
      { label: 'Share events', value: String(settings.product.share_events || 0) },
      { label: 'PDF downloads', value: String(settings.product.pdf_download_events || 0) },
      { label: 'CTA start-scan taps', value: String(settings.product.cta_start_scan_events || 0) },
    ])}
  `;
}

async function pollHealth() {
  try {
    const health = await api.fetchHealth();
    if (Array.isArray(health.warnings) && health.warnings.length) {
      healthStatus.textContent = `Needs attention · ${health.warnings[0]}`;
    } else {
      healthStatus.textContent = health.slam_active ? 'Live scene connected' : 'Upload-first mode';
    }
  } catch {
    healthStatus.textContent = 'API unavailable';
  }
}

async function bootstrapJobs() {
  if (!state.operatorSettings?.access?.enable_job_listing) {
    renderUploads();
    renderProcessing(activeJob());
    renderReport(activeJob());
    return;
  }

  try {
    const jobs = await api.fetchJobs();
    state.jobs.clear();
    jobs.forEach((job) => state.jobs.set(job.job_id, job));
    const latest = jobs.at(-1) || jobs[jobs.length - 1];
    if (latest) {
      setActiveJob(latest.job_id);
      renderProcessing(latest);
      if (latest.status === 'complete') {
        renderReport(latest);
      }
    } else {
      renderUploads();
      renderProcessing(null);
      renderReport(null);
    }
  } catch {
    renderUploads();
    renderProcessing(null);
    renderReport(null);
  }
}

async function refreshOperatorState(errorMessage = '') {
  try {
    state.operatorSettings = await api.fetchOperatorSettings();
  } catch {
    state.operatorSettings = null;
  }
  renderAccessPanels(errorMessage);
}

async function bootstrapApp() {
  try {
    state.accessPolicy = await api.fetchAccessPolicy();
  } catch {
    state.accessPolicy = null;
  }
  try {
    state.privacyPolicy = await api.fetchPrivacyPolicy();
  } catch {
    state.privacyPolicy = null;
  }
  try {
    state.uploadGuidance = await api.fetchUploadGuidance();
    uploadView.setGuidance(state.uploadGuidance);
    applyUploadGuidance(state.uploadGuidance);
  } catch {
    state.uploadGuidance = null;
    applyUploadGuidance(null);
  }
  await refreshOperatorState();
  await bootstrapJobs();
  const sampleKey = requestedSampleKey();
  if (sampleKey) {
    try {
      const sample = await api.fetchSampleReport();
      upsertJob(sample);
      setActiveJob(sample.job_id);
    } catch (error) {
      showToast(error instanceof Error ? error.message : 'Could not open the sample report.', 3600);
    }
  }
  const jobId = requestedJobId();
  if (jobId) {
    try {
      upsertJob(await api.fetchJob(jobId));
      setActiveJob(jobId);
    } catch (error) {
      showToast(error instanceof Error ? error.message : 'Could not open the shared report link.', 3600);
    }
  }
  switchView(requestedView() || (activeJob()?.status === 'complete' ? 'report' : 'scan'));
}

navButtons.forEach((button) => {
  button.addEventListener('click', () => switchView(button.dataset.view || 'scan'));
});

jumpButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const destination = button.dataset.jumpView || 'scan';
    if (destination === 'scan') {
      trackProductEvent('cta_start_scan', { surface: 'hero' });
    }
    switchView(destination);
  });
});

sampleButtons.forEach((button) => {
  button.addEventListener('click', loadSampleReport);
});

document.getElementById('scene-refresh-btn')?.addEventListener('click', () => {
  ensureScene();
  sceneViewer.refresh();
});

const uploadView = new UploadView({
  dropZone: document.getElementById('drop-zone'),
  fileInput: /** @type {HTMLInputElement} */ (document.getElementById('file-input')),
  roomLabelInput,
  audienceModeInput,
  onJobCreated: async (job) => {
    upsertJob(job);
    await refreshOperatorState();
    switchView('scan');
  },
  onJobUpdate: (job) => upsertJob(job),
  onJobError: (error) => showToast(error.message, 3600),
});

uploadView.init();
applyThemePreference(readStoredPreference(THEME_STORAGE_KEY) || document.documentElement.dataset.theme || 'light');
applyMotionPreference(readStoredPreference(MOTION_STORAGE_KEY) === 'true');
syncLowConfidenceControls();
syncSettingsAccessStatus();
bootstrapApp();
pollHealth();
setInterval(pollHealth, 6000);

lowConfidenceToggle?.addEventListener('change', (event) => {
  setLowConfidenceVisibility(/** @type {HTMLInputElement} */ (event.currentTarget).checked);
});

settingsLowConfidenceToggle?.addEventListener('change', (event) => {
  setLowConfidenceVisibility(/** @type {HTMLInputElement} */ (event.currentTarget).checked);
});

themeToggle?.addEventListener('change', (event) => {
  const enabled = /** @type {HTMLInputElement} */ (event.currentTarget).checked;
  applyThemePreference(enabled ? 'dark' : 'light');
  trackProductEvent('settings_theme_changed', { theme: enabled ? 'dark' : 'light' });
});

motionToggle?.addEventListener('change', (event) => {
  const enabled = /** @type {HTMLInputElement} */ (event.currentTarget).checked;
  applyMotionPreference(enabled);
  trackProductEvent('settings_motion_changed', { reduced_motion: enabled });
});

settingsSampleBtn?.addEventListener('click', () => {
  trackProductEvent('settings_sample_opened');
  loadSampleReport();
});

accessTokenSave?.addEventListener('click', async () => {
  const token = accessTokenInput.value.trim();
  if (!token) {
    showToast('Enter a token before saving.', 3200);
    return;
  }

  api.setAccessToken(token);
  accessTokenInput.value = '';
  await refreshOperatorState('The stored token did not unlock operator access.');
  await bootstrapJobs();
  if (requestedJobId()) {
    try {
      upsertJob(await api.fetchJob(requestedJobId()));
    } catch {}
  }
  if (requestedSampleKey()) {
    try {
      upsertJob(await api.fetchSampleReport());
    } catch {}
  }
  syncSettingsAccessStatus();
  showToast(state.operatorSettings ? 'Access token saved.' : 'Token saved, but access is still blocked.', 3200);
});

async function clearStoredAccessToken() {
  api.clearAccessToken();
  accessTokenInput.value = '';
  state.operatorSettings = null;
  state.jobs.clear();
  state.activeJobId = null;
  state.activeSampleKey = null;
  renderAccessPanels();
  renderUploads();
  renderProcessing(null);
  renderReport(null);
  syncSettingsAccessStatus();
  switchView('scan');
  showToast('Stored access token cleared.');
}

accessTokenClear?.addEventListener('click', async () => {
  await clearStoredAccessToken();
});

settingsTokenClear?.addEventListener('click', async () => {
  await clearStoredAccessToken();
});

operatorPruneBtn?.addEventListener('click', async () => {
  try {
    const result = await api.pruneOperatorStorage();
    await refreshOperatorState();
    showToast(
      result.deleted_jobs
        ? `Pruned ${result.deleted_jobs} job(s) and reclaimed ${formatBytes(result.bytes_reclaimed || 0)}.`
        : 'Storage prune completed with no expired jobs removed.',
      3400,
    );
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not run storage prune.', 3600);
  }
});

waitlistSubmitBtn?.addEventListener('click', async () => {
  const email = waitlistEmailInput.value.trim();
  const useCase = waitlistUseCaseInput.value.trim();
  if (!email) {
    showToast('Enter an email to join the beta waitlist.', 3200);
    return;
  }

  try {
    const response = await api.submitWaitlist({
      email,
      use_case: useCase || null,
    });
    waitlistEmailInput.value = '';
    waitlistUseCaseInput.value = '';
    if (waitlistNote) {
      waitlistNote.textContent = response.message;
    }
    await refreshOperatorState();
    showToast(`Waitlist joined. Position ${response.waitlist_count}.`, 3200);
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not join the waitlist.', 3600);
  }
});

deleteJobBtn?.addEventListener('click', async () => {
  const job = activeJob();
  if (!job) {
    return;
  }

  try {
    await api.deleteJob(job.job_id);
    removeJob(job.job_id);
    await refreshOperatorState();
    showToast('Report deleted.');
    switchView('scan');
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not delete report.', 3600);
  }
});

copyShareBtn?.addEventListener('click', async () => {
  const job = activeJob();
  if (!job || job.status !== 'complete') {
    return;
  }

  try {
    const link = reportDeepLink(job);
    await copyText(link);
    await trackProductEvent('report_share_copied', {
      surface: 'report_toolbar',
      job_id: job.job_id,
      sample_key: job.sample_key || null,
      audience_mode: job.audience_mode || 'general',
      room_labeled: Boolean(job.room_label || job.summary?.room_label),
    });
    showToast('Report link copied.');
  } catch (error) {
    showToast(error instanceof Error ? error.message : 'Could not copy report link.', 3600);
  }
});

exportPdfBtn?.addEventListener('click', () => {
  const job = activeJob();
  if (!job || job.status !== 'complete') {
    return;
  }
  trackProductEvent('report_pdf_downloaded', {
    surface: 'report_toolbar',
    job_id: job.job_id,
    sample_key: job.sample_key || null,
    audience_mode: job.audience_mode || 'general',
    room_labeled: Boolean(job.room_label || job.summary?.room_label),
  });
});

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function capitalize(value) {
  const text = String(value || '');
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}

function formatEvidenceMeta(frame) {
  const parts = [];
  if (typeof frame.frame_index === 'number') {
    parts.push(`Frame ${frame.frame_index}`);
  }
  if (typeof frame.timestamp_s === 'number') {
    parts.push(`${frame.timestamp_s.toFixed(1)}s`);
  }
  if (frame.object_label) {
    parts.push(String(frame.object_label));
  }
  if (frame.redacted) {
    parts.push('Text-heavy crop blurred');
  }
  return parts.join(' · ') || 'Stored evidence crop';
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatGateValue(value) {
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return String(value ?? '—');
  }
  if (numeric >= 0 && numeric <= 1 && !Number.isInteger(numeric)) {
    return `${Math.round(numeric * 100)}%`;
  }
  return String(Math.round(numeric * 100) / 100);
}

function isLowConfidenceRisk(risk) {
  return Number(risk?.confidence || 0) < 0.6 || Number(risk?.reasoning?.grounding_confidence || 0) < 0.55;
}

function renderFeedbackButton(verdict, activeVerdict) {
  const activeClass = verdict === activeVerdict ? 'active' : '';
  return `<button class="feedback-btn ${activeClass}" type="button" data-feedback="${verdict}">${capitalize(verdict)}</button>`;
}

function renderEvalButton(status, activeStatus) {
  const activeClass = status === activeStatus ? 'active' : '';
  const label = status === 'needs_review'
    ? 'Needs Review'
    : status === 'missed_hazard'
      ? 'Missed Hazard'
      : 'Confirmed';
  return `<button class="feedback-btn ${activeClass}" type="button" data-eval-status="${status}">${label}</button>`;
}

function renderFollowUpButton(status, activeStatus) {
  const activeClass = status === activeStatus ? 'active' : '';
  return `<button class="feedback-btn ${activeClass}" type="button" data-follow-up="${status}">${formatFollowUpLabel(status)}</button>`;
}

function formatFollowUpLabel(status) {
  if (status === 'resolved') return 'Resolved';
  if (status === 'monitor') return 'Monitor';
  if (status === 'ignored') return 'Ignored';
  return 'Open';
}

function renderScanQuality(scanQuality) {
  if (!scanQuality || Object.keys(scanQuality).length === 0) {
    return emptyMarkup('No scan quality diagnostics were recorded for this job.');
  }

  const warnings = scanQuality.warnings || [];
  const guidance = scanQuality.retry_guidance || [];
  const metrics = scanQuality.metrics || {};

  return `
    <div class="quality-score">
      <strong>${Math.round((scanQuality.score || 0) * 100)}</strong>
      <span class="severity-pill ${qualityTone(scanQuality.status)}">${escapeHtml(scanQuality.status || 'unknown')}</span>
    </div>
    <div class="quality-copy">
      <strong>What this means</strong>
      <span>${escapeHtml(scanQuality.capture_summary || (scanQuality.usable ? 'This scan is usable, but better lighting, steadier motion, and fuller coverage can still strengthen the report.' : 'This scan is likely to produce weaker findings and should ideally be rescanned before trusting smaller details.'))}</span>
    </div>
    <div class="report-card-meta">
      <span>${escapeHtml(capitalize(scanQuality.reportability || 'accepted'))} reportability</span>
      <span>${scanQuality.hard_reject ? 'Normal report refused' : scanQuality.rescan_recommended ? 'Report downgraded' : 'Normal report allowed'}</span>
    </div>
    <div class="report-card-meta">
      <span>${escapeHtml(`${metrics.frame_count || 0} sampled frame(s)`)}</span>
      <span>${escapeHtml(`${Math.round((metrics.motion_coverage || 0) * 100)}% motion coverage`)}</span>
      <span>${escapeHtml(`${Math.round((metrics.saliency_coverage || 0) * 100)}% object coverage`)}</span>
    </div>
    ${warnings.length ? `<ul class="quality-warning-list">${warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join('')}</ul>` : '<div class="empty-card">No major scan-quality warnings were detected.</div>'}
    ${scanQuality.rejection_reasons?.length ? `<ul class="quality-warning-list">${scanQuality.rejection_reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join('')}</ul>` : ''}
    ${guidance.length ? `<div class="quality-copy"><strong>Retry guidance</strong><span>${escapeHtml(guidance.slice(0, 2).join(' '))}</span></div>` : ''}
  `;
}

function renderReportPosture(summary, scanQuality, job) {
  const posture = summary.report_posture || 'screening';
  const guidance = scanQuality.retry_guidance || [];
  const comparison = job.room_comparison || null;
  const history = Array.isArray(job.room_history) ? job.room_history : [];
  const resolution = job.resolution_summary || {};

  return `
    <div class="report-copy-block">
      <strong>What this report supports</strong>
      <span>${escapeHtml(summary.overview || 'This upload produced a first-pass room hazard screen.')}</span>
    </div>
    <div class="report-copy-block">
      <strong>Coverage summary</strong>
      <span>${escapeHtml(summary.coverage_summary || 'Coverage details were not recorded for this scan.')}</span>
    </div>
    <div class="report-copy-block">
      <strong>What it does not claim</strong>
      <span>${escapeHtml(summary.screening_statement || 'This report flags likely hazards from the uploaded scan. It does not certify that the room is safe.')}</span>
    </div>
    ${typeof summary.room_score === 'number' ? `
      <div class="report-copy-block">
        <strong>Room safety score foundation</strong>
        <span>${escapeHtml(`${summary.room_score}/100 · ${summary.room_score_band || 'screening score'}. ${summary.room_score_summary || ''}`)}</span>
      </div>
    ` : ''}
    ${comparison ? `
      <div class="report-copy-block">
        <strong>Before / after comparison</strong>
        <span>${escapeHtml(`${comparison.summary} Score change: ${comparison.score_delta > 0 ? '+' : ''}${comparison.score_delta}. Hazard delta: ${comparison.hazard_delta > 0 ? '+' : ''}${comparison.hazard_delta}.`)}</span>
      </div>
    ` : ''}
    ${resolution.total_findings ? `
      <div class="report-copy-block">
        <strong>Follow-through state</strong>
        <span>${escapeHtml(resolution.summary || 'No follow-up state yet.')}</span>
      </div>
    ` : ''}
    <div class="report-card-meta">
      <span>${escapeHtml(posture)}</span>
      <span>${escapeHtml(summary.coverage_label || 'Unknown')} coverage</span>
      <span>${summary.rescan_recommended ? 'Rescan recommended' : 'No rescan required for first-pass review'}</span>
    </div>
    ${resolution.total_findings ? `
      <div class="report-card-meta">
        <span>${escapeHtml(String(resolution.resolved_count || 0))} resolved</span>
        <span>${escapeHtml(String(resolution.monitor_count || 0))} monitor</span>
        <span>${escapeHtml(String(resolution.ignored_count || 0))} ignored</span>
        <span>${escapeHtml(String(resolution.open_count || 0))} open</span>
      </div>
    ` : ''}
    ${history.length ? `<div class="report-card-meta">${history.slice(0, 3).map((entry) => `<span>${escapeHtml(`${entry.filename} · ${entry.room_score ?? '—'}/100`)}</span>`).join('')}</div>` : ''}
    ${guidance.length ? `<ul class="quality-warning-list">${guidance.slice(0, 2).map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>` : ''}
  `;
}

function renderEvaluationSummary(evaluation, job) {
  if (!evaluation || Object.keys(evaluation).length === 0) {
    return emptyMarkup('No review coverage has been recorded for this report yet.');
  }

  const controls = job.is_sample
    ? '<p class="meta-copy">Sample reports stay read-only so the built-in walkthrough remains stable for every visitor.</p>'
    : `
      <div class="evaluation-controls" data-eval-controls data-job-id="${job.job_id}">
        <div class="feedback-row">
          ${renderEvalButton('confirmed', evaluation.human_status)}
          ${renderEvalButton('needs_review', evaluation.human_status)}
          ${renderEvalButton('missed_hazard', evaluation.human_status)}
        </div>
        <div class="feedback-row">
          <button class="feedback-btn" type="button" data-export-eval-candidate="true">Save Eval Case</button>
        </div>
        <p class="meta-copy">Use these controls to build the eval set, log missed hazards, and keep the beta report loop honest.</p>
      </div>
    `;

  return `
    <div class="report-copy-block">
      <strong>Review status</strong>
      <span>${escapeHtml(evaluation.summary || 'No review summary available.')}</span>
    </div>
    <div class="meta-grid">
      <div class="meta-tile">
        <span>Review coverage</span>
        <strong>${Math.round((evaluation.review_coverage || 0) * 100)}%</strong>
      </div>
      <div class="meta-tile">
        <span>Pending findings</span>
        <strong>${escapeHtml(String(evaluation.pending_findings ?? 0))}</strong>
      </div>
      <div class="meta-tile">
        <span>Marked useful</span>
        <strong>${escapeHtml(String(evaluation.useful_events ?? 0))}</strong>
      </div>
      <div class="meta-tile">
        <span>Wrong or duplicate</span>
        <strong>${escapeHtml(String((evaluation.wrong_events || 0) + (evaluation.duplicate_events || 0)))}</strong>
      </div>
      <div class="meta-tile">
        <span>Precision proxy</span>
        <strong>${Math.round((evaluation.precision_proxy || 0) * 100)}%</strong>
      </div>
      <div class="meta-tile">
        <span>Recall proxy</span>
        <strong>${Math.round((evaluation.recall_proxy || 0) * 100)}%</strong>
      </div>
    </div>
    <div class="report-card-meta">
      <span>${escapeHtml(String(evaluation.high_priority_pending ?? 0))} high-priority findings unreviewed</span>
      <span>${escapeHtml(String(evaluation.missed_hazard_count ?? 0))} missed hazards logged</span>
      <span>${escapeHtml(evaluation.benchmark_label ? `${evaluation.benchmark_label} benchmark` : 'No benchmark tag')}</span>
      <span>${evaluation.needs_review ? 'More review still needed' : 'Review loop covered current findings'}</span>
    </div>
    <div class="evaluation-status">
      <span>Human verdict: ${escapeHtml(evaluation.human_status || 'not set')}</span>
      <span>${escapeHtml(evaluation.benchmark_match === true ? 'Benchmark matched' : evaluation.benchmark_match === false ? 'Benchmark mismatch' : 'No benchmark comparison')}</span>
    </div>
    ${controls}
  `;
}

function qualityTone(status) {
  if (status === 'good') return 'low';
  if (status === 'fair') return 'medium';
  return 'high';
}

function stageExplanation(stage) {
  if (stage === 'upload') return 'The upload was accepted and queued for processing.';
  if (stage === 'ingest') return 'The media is being unpacked so frames can be sampled cleanly.';
  if (stage === 'vlm') return 'ATLAS-0 is labeling observations and grounding them across the scan.';
  if (stage === 'risk') return 'Findings, recommendations, and evidence artifacts are being assembled into the report.';
  if (stage === 'complete') return 'The report is ready to review.';
  return 'ATLAS-0 is processing the scan.';
}

function attachFeedbackHandlers(jobId) {
  reportHazards.querySelectorAll('[data-feedback-controls]').forEach((container) => {
    container.querySelectorAll('[data-feedback]').forEach((button) => {
      button.addEventListener('click', async () => {
        try {
          const updated = await api.submitFindingFeedback(jobId, {
            hazard_code: container.dataset.hazardCode,
            object_id: container.dataset.objectId || null,
            verdict: button.dataset.feedback,
          });
          upsertJob(updated);
          showToast('Feedback saved.');
        } catch (error) {
          showToast(error instanceof Error ? error.message : 'Could not save feedback.', 3600);
        }
      });
    });
  });
}

function attachFollowUpHandlers(jobId) {
  reportHazards.querySelectorAll('[data-follow-up-controls]').forEach((container) => {
    container.querySelectorAll('[data-follow-up]').forEach((button) => {
      button.addEventListener('click', async () => {
        const nextStatus = button.dataset.followUp || '';
        const activeStatus = container.dataset.activeStatus || '';
        const status = nextStatus === activeStatus ? 'open' : nextStatus;
        try {
          const updated = await api.submitFindingFollowUp(jobId, {
            hazard_code: container.dataset.hazardCode,
            object_id: container.dataset.objectId || null,
            status,
          });
          upsertJob(updated);
          showToast(status === 'open' ? 'Finding reset to open.' : 'Follow-up saved.');
        } catch (error) {
          showToast(error instanceof Error ? error.message : 'Could not save follow-up.', 3600);
        }
      });
    });
  });
}

function attachEvaluationHandlers(jobId) {
  reportEvalCard.querySelectorAll('[data-eval-controls]').forEach((container) => {
    container.querySelectorAll('[data-export-eval-candidate]').forEach((button) => {
      button.addEventListener('click', async () => {
        const suggested = `${jobId}-eval`;
        const label = window.prompt(
          'Optional eval-case label. Leave it as-is or clear it to use the job ID.',
          suggested,
        );
        if (label === null) {
          return;
        }

        try {
          const updated = await api.exportEvalCandidate(jobId, {
            label: label.trim() || null,
          });
          upsertJob(updated);
          await refreshOperatorState();
          showToast('Eval case saved.');
        } catch (error) {
          showToast(error instanceof Error ? error.message : 'Could not save eval case.', 3600);
        }
      });
    });

    container.querySelectorAll('[data-eval-status]').forEach((button) => {
      button.addEventListener('click', async () => {
        const status = button.dataset.evalStatus;
        if (!status) {
          return;
        }

        let missedHazards = [];
        let note = '';
        if (status === 'missed_hazard') {
          const answer = window.prompt(
            'List any missed hazards as a comma-separated note. Leave blank if you just want to flag a miss.',
            '',
          );
          if (answer === null) {
            return;
          }
          note = answer.trim();
          missedHazards = note
            ? note.split(',').map((item) => item.trim()).filter(Boolean)
            : [];
        }

        try {
          const updated = await api.submitJobEvaluation(jobId, {
            status,
            missed_hazards: missedHazards,
            note: note || null,
          });
          upsertJob(updated);
          await refreshOperatorState();
          showToast('Evaluation saved.');
        } catch (error) {
          showToast(error instanceof Error ? error.message : 'Could not save evaluation.', 3600);
        }
      });
    });
  });
}
