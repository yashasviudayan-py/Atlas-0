/**
 * app.js — Atlas-0 web app main controller.
 *
 * Handles:
 *   - Sidebar navigation between the four views
 *   - Health polling (header chips)
 *   - WebSocket risk stream (Monitor view)
 *   - Lazy init of Scene and Intelligence views
 *   - Toast notifications
 */

import { UploadView }       from './upload.js';
import { SceneViewer }      from './scene_viewer.js';
import { IntelligenceView } from './intelligence.js';
import { OverlayRenderer }  from './overlay.js';
import * as api             from './api.js';

// ── Navigation ────────────────────────────────────────────────────────────────

const VIEW_LABELS = { ingest: 'Ingest', scene: 'Scene', intelligence: 'Intelligence', monitor: 'Monitor' };

const navBtns     = /** @type {NodeListOf<HTMLButtonElement>} */ (document.querySelectorAll('.nav-btn[data-view]'));
const viewEls     = document.querySelectorAll('.view');
const hdrLabel    = document.getElementById('hdr-view-label');

let activeView = 'ingest';

navBtns.forEach(btn => btn.addEventListener('click', () => switchView(btn.dataset.view)));

function switchView(id) {
  if (id === activeView) return;
  activeView = id;

  navBtns.forEach(b => b.classList.toggle('active', b.dataset.view === id));
  viewEls.forEach(v => v.classList.toggle('active', v.id === `view-${id}`));
  hdrLabel.textContent = VIEW_LABELS[id] || id;

  if (id === 'scene')        { ensureScene(); sceneViewer.refresh(); }
  if (id === 'intelligence') { intelView.refresh(); }
  if (id === 'monitor')      { tryCamera(); }
  if (id !== 'monitor')      { stopCamera(); }
}

// ── Upload view ───────────────────────────────────────────────────────────────

const uploadView = new UploadView({
  dropZone:  document.getElementById('drop-zone'),
  fileInput: document.getElementById('file-input'),
  queue:     document.getElementById('upload-queue'),
  queueWrap: document.getElementById('queue-wrap'),
});
uploadView.init();

// ── Scene viewer (lazy init) ──────────────────────────────────────────────────

const sceneCanvas    = /** @type {HTMLCanvasElement} */ (document.getElementById('scene-canvas'));
const sceneEmpty     = document.getElementById('scene-empty');
const sceneObjList   = document.getElementById('scene-obj-list');
const sceneViewer    = new SceneViewer(sceneCanvas, sceneEmpty, sceneObjList);
let   sceneReady     = false;

function ensureScene() {
  if (!sceneReady) { sceneViewer.init(); sceneReady = true; }
}

document.getElementById('scene-refresh-btn').addEventListener('click', () => {
  ensureScene();
  sceneViewer.refresh();
});

const autoBtn = document.getElementById('scene-auto-btn');
let   autoOn  = false;

autoBtn.addEventListener('click', () => {
  autoOn = !autoOn;
  autoBtn.textContent = `Auto ${autoOn ? '●' : '○'}`;
  ensureScene();
  sceneViewer.setAutoRefresh(autoOn);
});

// ── Intelligence view ─────────────────────────────────────────────────────────

const intelView = new IntelligenceView({
  queryInput:   document.getElementById('q-input'),
  queryBtn:     document.getElementById('q-btn'),
  queryResults: document.getElementById('q-results'),
  objList:      document.getElementById('i-obj-list'),
  riskList:     document.getElementById('i-risk-list'),
  objCount:     document.getElementById('i-obj-cnt'),
  riskCount:    document.getElementById('i-risk-cnt'),
});
intelView.init();

// ── Monitor view — AR overlay + WebSocket ─────────────────────────────────────

const overlayCanvas  = /** @type {HTMLCanvasElement} */ (document.getElementById('overlay-canvas'));
const alertContainer = document.getElementById('alert-container');
const wsDot          = document.getElementById('ws-dot');
const wsLabel        = document.getElementById('ws-label');
const connBtn        = document.getElementById('conn-btn');
const riskBadge      = document.getElementById('risk-badge');
const statRisks      = document.getElementById('stat-risks');
const statObjs       = document.getElementById('stat-objs');
const cameraFeed     = /** @type {HTMLVideoElement} */ (document.getElementById('camera-feed'));

const overlay    = new OverlayRenderer(overlayCanvas);
overlay.init();

const alertBadges = new Map();
const riskStore   = new Map();

let ws              = null;
let userDisconnected = false;
let reconnectDelay  = 1000;

function wsUrl() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${location.host}/ws/risks`;
}

function connect() {
  if (ws && ws.readyState <= WebSocket.OPEN) return;
  userDisconnected = false;
  setWsStatus('connecting');
  ws = new WebSocket(wsUrl());

  ws.addEventListener('open', () => {
    reconnectDelay = 1000;
    setWsStatus('connected');
  });
  ws.addEventListener('message', evt => {
    try { applyDelta(JSON.parse(evt.data)); } catch { /* ignore malformed */ }
  });
  ws.addEventListener('close', () => {
    setWsStatus('disconnected');
    if (!userDisconnected) {
      setTimeout(() => { if (!userDisconnected) connect(); }, reconnectDelay);
      reconnectDelay = Math.min(reconnectDelay * 2, 16000);
    }
  });
  ws.addEventListener('error', () => setWsStatus('disconnected'));
}

function disconnect() {
  userDisconnected = true;
  ws?.close();
  setWsStatus('disconnected');
}

connBtn.addEventListener('click', () => {
  if (ws && ws.readyState <= WebSocket.OPEN) disconnect();
  else connect();
});

function applyDelta({ added = [], updated = [], removed = [] }) {
  for (const e of added)   { riskStore.set(e.object_id, e); overlay.addRisk(e);    upsertBadge(e); }
  for (const e of updated) { riskStore.set(e.object_id, e); overlay.updateRisk(e); upsertBadge(e); }
  for (const id of removed){ riskStore.delete(id);          overlay.removeRisk(id); removeBadge(id); }
  updateMonHUD();
}

function upsertBadge(entry) {
  let b = alertBadges.get(entry.object_id);
  if (!b) {
    b = document.createElement('div');
    b.className = 'alert-badge';
    b.innerHTML = '<div class="dot"></div><span class="lbl"></span>';
    alertContainer.appendChild(b);
    alertBadges.set(entry.object_id, b);
  }

  const score = entry.combined_score ?? 0;
  b.classList.remove('severity-low', 'severity-mid', 'severity-high');
  b.classList.add(score >= 0.65 ? 'severity-high' : score >= 0.35 ? 'severity-mid' : 'severity-low');

  const alertData = entry.overlay?.alert;
  const text = alertData?.text ?? entry.object_label ?? `Object ${entry.object_id}`;
  b.querySelector('.lbl').textContent = text.length > 50 ? text.slice(0, 48) + '…' : text;

  let px = null, py = null;
  if (alertData?.screen_position) {
    [px, py] = alertData.screen_position;
  } else if (entry.position) {
    const ndc = overlay.projectToNDC(entry.position);
    px = (ndc.x * 0.5 + 0.5) * window.innerWidth;
    py = (-ndc.y * 0.5 + 0.5) * window.innerHeight;
  }

  if (px !== null) {
    b.style.left    = `${px}px`;
    b.style.top     = `${py}px`;
    b.style.display = '';
  } else {
    b.style.display = 'none';
  }
}

function removeBadge(id) {
  const b = alertBadges.get(id);
  if (b) { b.remove(); alertBadges.delete(id); }
}

function updateMonHUD() {
  const c = riskStore.size;
  statRisks.textContent = String(c);
  statObjs.textContent  = String(c);
  riskBadge.textContent = `${c} risk${c === 1 ? '' : 's'}`;
  riskBadge.style.display = c > 0 ? '' : 'none';
}

function setWsStatus(state) {
  wsDot.className   = state;
  wsLabel.textContent = state.charAt(0).toUpperCase() + state.slice(1);
  connBtn.textContent = (state === 'connected' || state === 'connecting') ? 'Disconnect' : 'Connect';
  document.getElementById('sb-conn-dot').className = state;
}

// ── Camera (monitor view) ─────────────────────────────────────────────────────

async function tryCamera() {
  if (cameraFeed.srcObject) return;   // already running
  if (!navigator.mediaDevices?.getUserMedia) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 } },
    });
    cameraFeed.srcObject = stream;
  } catch { /* permission denied or no camera — dark bg is the fallback */ }
}

function stopCamera() {
  if (!cameraFeed.srcObject) return;
  cameraFeed.srcObject.getTracks().forEach(t => t.stop());
  cameraFeed.srcObject = null;
}

// ── Health polling ────────────────────────────────────────────────────────────

const chipVlm   = document.getElementById('chip-vlm');
const chipSlam  = document.getElementById('chip-slam');
const chipObjs  = document.getElementById('chip-objs');
const chipRisks = document.getElementById('chip-risks');
const cntObjs   = document.getElementById('cnt-objs');
const cntRisks  = document.getElementById('cnt-risks');

async function pollHealth() {
  try {
    const h = await api.fetchHealth();
    chipVlm.className  = `chip ${h.vlm_active  ? 'on'    : ''}`;
    chipSlam.className = `chip ${h.slam_active  ? 'on'    : ''}`;
    cntObjs.textContent   = String(h.object_count);
    cntRisks.textContent  = String(h.risk_count);
    chipObjs.className  = `chip ${h.object_count > 0 ? 'on'    : ''}`;
    chipRisks.className = `chip ${h.risk_count   > 0 ? 'alert' : ''}`;
  } catch { /* server not up yet */ }
}

// ── Toast ─────────────────────────────────────────────────────────────────────

const toastEl = document.getElementById('toast');
let   toastTm = null;

window.showToast = (msg, ms = 3000) => {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastTm);
  toastTm = setTimeout(() => toastEl.classList.remove('show'), ms);
};

// ── Bootstrap ─────────────────────────────────────────────────────────────────

connect();
pollHealth();
setInterval(pollHealth, 6000);
