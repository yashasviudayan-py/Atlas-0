/**
 * app.js — ATLAS-0 report-first web application.
 */

import * as api from './api.js';
import { SceneViewer } from './scene_viewer.js';
import { UploadView } from './upload.js';

const VIEW_LABELS = {
  scan: 'Scan',
  report: 'Report',
  scene: 'Scene',
};

const state = {
  activeView: 'scan',
  activeJobId: null,
  jobs: new Map(),
  showLowConfidence: false,
  accessPolicy: null,
  operatorSettings: null,
};

const navButtons = /** @type {NodeListOf<HTMLButtonElement>} */ (
  document.querySelectorAll('.nav-btn[data-view]')
);
const jumpButtons = /** @type {NodeListOf<HTMLButtonElement>} */ (
  document.querySelectorAll('[data-jump-view]')
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

const recentList = document.getElementById('upload-list');
const recentEmpty = document.getElementById('upload-empty');
const accessBanner = document.getElementById('access-banner');
const accessHelp = document.getElementById('access-help');
const operatorPolicy = document.getElementById('operator-policy');
const operatorQueue = document.getElementById('operator-queue');
const accessTokenInput = /** @type {HTMLInputElement} */ (document.getElementById('access-token-input'));
const accessTokenSave = /** @type {HTMLButtonElement} */ (document.getElementById('access-token-save'));
const accessTokenClear = /** @type {HTMLButtonElement} */ (document.getElementById('access-token-clear'));

const reportHero = document.getElementById('report-hero');
const summaryObjects = document.getElementById('summary-objects');
const summaryHazards = document.getElementById('summary-hazards');
const summarySeverity = document.getElementById('summary-severity');
const summaryConfidence = document.getElementById('summary-confidence');
const summarySource = document.getElementById('summary-source');
const fixFirstList = document.getElementById('fix-first-list');
const scanQualityCard = document.getElementById('scan-quality-card');
const lowConfidenceToggle = /** @type {HTMLInputElement} */ (document.getElementById('low-confidence-toggle'));
const findingToggleNote = document.getElementById('finding-toggle-note');
const reportHeadline = document.getElementById('report-headline');
const reportSubhead = document.getElementById('report-subhead');
const reportHazards = document.getElementById('risk-report-list');
const reportRecommendations = document.getElementById('rec-list');
const reportEvidence = document.getElementById('evidence-grid');
const trustNotes = document.getElementById('trust-notes');
const exportPdfBtn = /** @type {HTMLAnchorElement} */ (document.getElementById('export-pdf-btn'));
const deleteJobBtn = /** @type {HTMLButtonElement} */ (document.getElementById('delete-job-btn'));

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
  renderUploads();
  renderProcessing(activeJob());
  renderReport(activeJob());
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
    showToast('Scan complete. Report is ready.');
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
  const jobs = [...state.jobs.values()].sort((a, b) => b.job_id.localeCompare(a.job_id));
  recentEmpty.style.display = jobs.length ? 'none' : '';
  recentEmpty.textContent = jobs.length
    ? ''
    : 'No scans yet. Start with one room you know well, keep the camera steady, and the first report will appear here.';

  recentList.innerHTML = jobs.map((job) => {
    const summary = job.summary || {};
    const hazardLabel = summary.top_hazard_label || 'No hazards yet';
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
          <span>${escapeHtml(hazardLabel)}</span>
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
    uploadStatus.textContent = 'Ready for first scan';
    reportStatus.textContent = 'No report yet';
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

  processMeta.textContent = `${escapeHtml(job.filename)} · ${escapeHtml(job.status)}`;
  uploadStatus.textContent = job.status === 'complete' ? 'Latest scan complete' : 'Scan in progress';
  reportStatus.textContent = job.status === 'complete' ? 'Report ready' : 'Waiting for report';
}

function renderReport(job) {
  if (!job || job.status !== 'complete') {
    reportHero.classList.add('empty');
    reportHeadline.textContent = 'No report yet';
    reportSubhead.textContent = 'Run one room scan to review hazards, evidence, and practical next steps in a single place.';
    summaryObjects.textContent = '0';
    summaryHazards.textContent = '0';
    summarySeverity.textContent = '—';
    summaryConfidence.textContent = '—';
    summarySource.textContent = '—';
    fixFirstList.innerHTML = emptyMarkup('Priority actions will appear here once a scan finishes.');
    scanQualityCard.innerHTML = emptyMarkup('Scan quality diagnostics will appear here after the upload is processed.');
    reportHazards.innerHTML = emptyMarkup('Hazards will appear here once ATLAS-0 has something evidence-backed to show.');
    reportRecommendations.innerHTML = emptyMarkup('Recommendations will appear here after a completed scan.');
    reportEvidence.innerHTML = emptyMarkup('Evidence frames will appear here after a completed scan.');
    trustNotes.innerHTML = emptyMarkup('Trust notes will appear here after a completed scan.');
    findingToggleNote.textContent = 'Low-confidence findings stay hidden until a report is ready.';
    exportPdfBtn.removeAttribute('href');
    exportPdfBtn.classList.add('disabled');
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

  reportHeadline.textContent = summary.top_hazard_label
    ? `Top concern: ${summary.top_hazard_label}`
    : 'Hazard report';
  reportSubhead.textContent = `${escapeHtml(job.filename)} · ${summary.confidence_label || 'Approximate grounding'}`;

  summaryObjects.textContent = String(summary.object_count || 0);
  summaryHazards.textContent = String(summary.hazard_count || 0);
  summarySeverity.textContent = capitalize(summary.top_severity || 'none');
  summaryConfidence.textContent = summary.scan_quality_label
    ? `${summary.confidence_label || 'Approximate grounding'} · ${summary.scan_quality_label} scan`
    : summary.confidence_label || 'Approximate grounding';
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
          <div class="report-card-meta">
            <span>${Math.round((risk.risk_score || 0) * 100)} risk score</span>
            <span>${escapeHtml(risk.location_label || 'Approximate location')}</span>
            <span>${escapeHtml(risk.hazard_code || 'finding')}</span>
            <span>${escapeHtml(risk.confidence_label || 'weak')} evidence</span>
            <span>${escapeHtml(risk.reasoning?.support_summary || 'Limited support')}</span>
          </div>
          <div class="feedback-row" data-feedback-controls data-job-id="${job.job_id}" data-hazard-code="${escapeHtml(risk.hazard_code || '')}" data-object-id="${escapeHtml(risk.object_id || '')}">
            ${renderFeedbackButton('useful', risk.latest_feedback)}
            ${renderFeedbackButton('wrong', risk.latest_feedback)}
            ${renderFeedbackButton('duplicate', risk.latest_feedback)}
          </div>
        </article>
      `).join('')
    : emptyMarkup(
        hazards.length
          ? 'Only lower-confidence findings remain. Toggle them on if you want the full raw report.'
          : 'No significant hazards were detected in this scan.',
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

  exportPdfBtn.href = api.reportPdfUrl(job.job_id);
  exportPdfBtn.classList.remove('disabled');
  deleteJobBtn.classList.remove('disabled');
  deleteJobBtn.disabled = false;
  attachFeedbackHandlers(job.job_id);
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

function renderAccessPanels(errorMessage = '') {
  const access = state.accessPolicy;
  const settings = state.operatorSettings;
  const tokenStored = Boolean(api.getAccessToken());

  accessTokenClear.disabled = !tokenStored;

  if (!access) {
    accessBanner.className = 'status-banner';
    accessBanner.innerHTML = '<strong>Could not load access policy.</strong>';
    accessHelp.textContent = 'The hosted access policy could not be loaded from the API.';
    operatorPolicy.innerHTML = emptyMarkup('Operator policy details are unavailable right now.');
    operatorQueue.innerHTML = emptyMarkup('Queue diagnostics are unavailable right now.');
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
    return;
  }

  operatorPolicy.innerHTML = renderPolicyItems([
    { label: 'Access mode', value: settings.access.mode === 'token' ? 'Token protected' : settings.access.mode === 'loopback' ? 'Loopback-friendly' : 'Restricted' },
    { label: 'Job listing', value: settings.access.enable_job_listing ? 'Enabled' : 'Direct job IDs only' },
    { label: 'Retention window', value: `${settings.uploads.retention_days} day(s)` },
    { label: 'Keep originals', value: settings.uploads.save_original_uploads ? 'Yes' : 'No' },
    { label: 'Queue depth limit', value: String(settings.uploads.max_queue_depth) },
    { label: 'Retry budget', value: `${settings.uploads.max_job_attempts} attempt(s)` },
  ]);

  operatorQueue.innerHTML = renderPolicyItems([
    { label: 'Workers', value: String(settings.queue.worker_count) },
    { label: 'Queued jobs', value: String(settings.queue.queued_jobs) },
    { label: 'Processing jobs', value: String(settings.queue.processing_jobs) },
    { label: 'Failed jobs', value: String(settings.queue.failed_jobs) },
    { label: 'Stored jobs', value: String(settings.storage.persisted_jobs) },
    { label: 'Disk used', value: formatBytes(settings.storage.bytes_used || 0) },
  ]);
}

async function pollHealth() {
  try {
    const health = await api.fetchHealth();
    healthStatus.textContent = health.slam_active ? 'Live scene connected' : 'Upload-first mode';
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
  await refreshOperatorState();
  await bootstrapJobs();
}

navButtons.forEach((button) => {
  button.addEventListener('click', () => switchView(button.dataset.view || 'scan'));
});

jumpButtons.forEach((button) => {
  button.addEventListener('click', () => switchView(button.dataset.jumpView || 'scan'));
});

document.getElementById('scene-refresh-btn')?.addEventListener('click', () => {
  ensureScene();
  sceneViewer.refresh();
});

const uploadView = new UploadView({
  dropZone: document.getElementById('drop-zone'),
  fileInput: /** @type {HTMLInputElement} */ (document.getElementById('file-input')),
  onJobCreated: async (job) => {
    upsertJob(job);
    await refreshOperatorState();
    switchView('scan');
  },
  onJobUpdate: (job) => upsertJob(job),
  onJobError: (error) => showToast(error.message, 3600),
});

uploadView.init();
bootstrapApp();
pollHealth();
setInterval(pollHealth, 6000);
switchView('scan');

lowConfidenceToggle?.addEventListener('change', (event) => {
  state.showLowConfidence = /** @type {HTMLInputElement} */ (event.currentTarget).checked;
  renderReport(activeJob());
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
  showToast(state.operatorSettings ? 'Access token saved.' : 'Token saved, but access is still blocked.', 3200);
});

accessTokenClear?.addEventListener('click', async () => {
  api.clearAccessToken();
  accessTokenInput.value = '';
  state.operatorSettings = null;
  state.jobs.clear();
  state.activeJobId = null;
  renderAccessPanels();
  renderUploads();
  renderProcessing(null);
  renderReport(null);
  showToast('Stored access token cleared.');
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

function isLowConfidenceRisk(risk) {
  return Number(risk?.confidence || 0) < 0.6 || Number(risk?.reasoning?.grounding_confidence || 0) < 0.55;
}

function renderFeedbackButton(verdict, activeVerdict) {
  const activeClass = verdict === activeVerdict ? 'active' : '';
  return `<button class="feedback-btn ${activeClass}" type="button" data-feedback="${verdict}">${capitalize(verdict)}</button>`;
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
      <span>${scanQuality.usable ? 'This scan is usable, but better lighting, steadier motion, and fuller coverage can still strengthen the report.' : 'This scan is likely to produce weaker findings and should ideally be rescanned before trusting smaller details.'}</span>
    </div>
    <div class="report-card-meta">
      <span>${escapeHtml(`${metrics.frame_count || 0} sampled frame(s)`)}</span>
      <span>${escapeHtml(`${Math.round((metrics.motion_coverage || 0) * 100)}% motion coverage`)}</span>
      <span>${escapeHtml(`${Math.round((metrics.saliency_coverage || 0) * 100)}% object coverage`)}</span>
    </div>
    ${warnings.length ? `<ul class="quality-warning-list">${warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join('')}</ul>` : '<div class="empty-card">No major scan-quality warnings were detected.</div>'}
    ${guidance.length ? `<div class="quality-copy"><strong>Retry guidance</strong><span>${escapeHtml(guidance[0])}</span></div>` : ''}
  `;
}

function qualityTone(status) {
  if (status === 'good') return 'low';
  if (status === 'fair') return 'medium';
  return 'high';
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
