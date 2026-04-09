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
};

const navButtons = /** @type {NodeListOf<HTMLButtonElement>} */ (
  document.querySelectorAll('.nav-btn[data-view]')
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

const reportHero = document.getElementById('report-hero');
const summaryObjects = document.getElementById('summary-objects');
const summaryHazards = document.getElementById('summary-hazards');
const summarySeverity = document.getElementById('summary-severity');
const summaryConfidence = document.getElementById('summary-confidence');
const summarySource = document.getElementById('summary-source');
const reportHeadline = document.getElementById('report-headline');
const reportSubhead = document.getElementById('report-subhead');
const reportHazards = document.getElementById('risk-report-list');
const reportRecommendations = document.getElementById('rec-list');
const reportEvidence = document.getElementById('evidence-grid');
const trustNotes = document.getElementById('trust-notes');
const exportPdfBtn = /** @type {HTMLAnchorElement} */ (document.getElementById('export-pdf-btn'));

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

function renderUploads() {
  const jobs = [...state.jobs.values()].sort((a, b) => b.job_id.localeCompare(a.job_id));
  recentEmpty.style.display = jobs.length ? 'none' : '';

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
    processCopy.textContent = 'Upload a 20-60 second room walkthrough to generate a safety report.';
    processBar.style.width = '0%';
    processMeta.textContent = 'No active scan';
    uploadStatus.textContent = 'Waiting for scan';
    reportStatus.textContent = 'No report yet';
    return;
  }

  const statusLabel = `${capitalize(job.stage || 'upload')} · ${Math.round((job.progress || 0) * 100)}%`;
  processStage.textContent = statusLabel;
  processBar.style.width = `${Math.round((job.progress || 0) * 100)}%`;

  if (job.status === 'complete') {
    processCopy.textContent = 'Evidence frames, hazard ranking, and recommendations are ready to review.';
  } else if (job.status === 'error') {
    processCopy.textContent = job.error || 'The scan could not be processed.';
  } else {
    processCopy.textContent = 'ATLAS-0 is analyzing the upload and building an evidence-backed report.';
  }

  processMeta.textContent = `${escapeHtml(job.filename)} · ${escapeHtml(job.status)}`;
  uploadStatus.textContent = job.status === 'complete' ? 'Latest scan complete' : 'Scan in progress';
  reportStatus.textContent = job.status === 'complete' ? 'Report ready' : 'Waiting for report';
}

function renderReport(job) {
  if (!job || job.status !== 'complete') {
    reportHero.classList.add('empty');
    reportHeadline.textContent = 'No report yet';
    reportSubhead.textContent = 'Run a scan to see hazards, evidence, and recommended actions.';
    summaryObjects.textContent = '0';
    summaryHazards.textContent = '0';
    summarySeverity.textContent = '—';
    summaryConfidence.textContent = '—';
    summarySource.textContent = '—';
    reportHazards.innerHTML = emptyMarkup('No hazards to review yet.');
    reportRecommendations.innerHTML = emptyMarkup('Recommendations will appear after a completed scan.');
    reportEvidence.innerHTML = emptyMarkup('Evidence frames will appear after a completed scan.');
    trustNotes.innerHTML = emptyMarkup('Trust notes will appear after a completed scan.');
    exportPdfBtn.removeAttribute('href');
    exportPdfBtn.classList.add('disabled');
    return;
  }

  reportHero.classList.remove('empty');
  const summary = job.summary || {};
  const hazards = job.risks || [];
  const recommendations = job.recommendations || [];
  const evidence = job.evidence_frames || [];
  const notes = job.trust_notes || [];

  reportHeadline.textContent = summary.top_hazard_label
    ? `Top concern: ${summary.top_hazard_label}`
    : 'Hazard report';
  reportSubhead.textContent = `${escapeHtml(job.filename)} · ${summary.confidence_label || 'Approximate grounding'}`;

  summaryObjects.textContent = String(summary.object_count || 0);
  summaryHazards.textContent = String(summary.hazard_count || 0);
  summarySeverity.textContent = capitalize(summary.top_severity || 'none');
  summaryConfidence.textContent = summary.confidence_label || 'Approximate grounding';
  summarySource.textContent = summary.scene_source || 'unknown';

  reportHazards.innerHTML = hazards.length
    ? hazards.map((risk) => `
        <article class="report-card hazard ${risk.severity || 'low'}">
          <div class="report-card-top">
            <h3>${escapeHtml(risk.object_label || 'Object')}</h3>
            <span class="severity-pill ${risk.severity || 'low'}">${escapeHtml(risk.severity || 'low')}</span>
          </div>
          <p>${escapeHtml(risk.description || '')}</p>
          <div class="report-card-meta">
            <span>${Math.round((risk.risk_score || 0) * 100)} risk score</span>
            <span>${escapeHtml(risk.location_label || 'Approximate location')}</span>
          </div>
        </article>
      `).join('')
    : emptyMarkup('No significant hazards were detected in this scan.');

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
          <img src="${frame.image_url}" alt="${escapeHtml(frame.caption || 'Evidence frame')}" />
          <div class="evidence-copy">
            <strong>${escapeHtml(frame.caption || 'Evidence frame')}</strong>
            <span>${Math.round((frame.confidence || 0) * 100)}% label confidence</span>
          </div>
        </article>
      `).join('')
    : emptyMarkup('No evidence frames were stored for this scan.');

  trustNotes.innerHTML = notes.length
    ? notes.map((note) => `<li>${escapeHtml(note)}</li>`).join('')
    : '<li>No additional trust notes.</li>';

  exportPdfBtn.href = api.reportPdfUrl(job.job_id);
  exportPdfBtn.classList.remove('disabled');
}

function emptyMarkup(message) {
  return `<div class="empty-card">${escapeHtml(message)}</div>`;
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
  try {
    const jobs = await api.fetchJobs();
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

navButtons.forEach((button) => {
  button.addEventListener('click', () => switchView(button.dataset.view || 'scan'));
});

document.getElementById('scene-refresh-btn')?.addEventListener('click', () => {
  ensureScene();
  sceneViewer.refresh();
});

const uploadView = new UploadView({
  dropZone: document.getElementById('drop-zone'),
  fileInput: /** @type {HTMLInputElement} */ (document.getElementById('file-input')),
  onJobCreated: (job) => {
    upsertJob(job);
    switchView('scan');
  },
  onJobUpdate: (job) => upsertJob(job),
  onJobError: (error) => showToast(error.message, 3600),
});

uploadView.init();
bootstrapJobs();
pollHealth();
setInterval(pollHealth, 6000);
switchView('scan');

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
