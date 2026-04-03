/**
 * app.js — Atlas-0 AR overlay application entry point.
 *
 * Responsibilities:
 *   1. Initialise the Three.js OverlayRenderer on the overlay canvas.
 *   2. Manage the WebSocket connection to /ws/risks.
 *   3. Apply delta updates (added / updated / removed) to the renderer and
 *      the DOM alert badge container.
 *   4. Update the HUD counters and connection status indicator.
 *
 * The app auto-connects on page load and exponentially backs off on
 * disconnection (max 16 s).
 */

import { OverlayRenderer } from './overlay.js';

// ── Constants ─────────────────────────────────────────────────────────────────

const WS_PATH     = '/ws/risks';
const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS  = 16000;

// ── DOM references ────────────────────────────────────────────────────────────

const overlayCanvas   = /** @type {HTMLCanvasElement} */ (document.getElementById('overlay-canvas'));
const alertContainer  = document.getElementById('alert-container');
const wsDot           = document.getElementById('ws-dot');
const wsLabel         = document.getElementById('ws-label');
const connectBtn      = document.getElementById('connect-btn');
const riskCountBadge  = document.getElementById('risk-count-badge');
const statRisks       = document.getElementById('stat-risks');
const statObjs        = document.getElementById('stat-objs');
const cameraFeed      = /** @type {HTMLVideoElement} */ (document.getElementById('camera-feed'));

// ── State ─────────────────────────────────────────────────────────────────────

/** @type {OverlayRenderer} */
const renderer = new OverlayRenderer(overlayCanvas);

/** @type {WebSocket|null} */
let ws = null;

/** @type {boolean} */
let userDisconnected = false;

/** @type {number} */
let reconnectDelay = RECONNECT_BASE_MS;

/** Alert badge DOM nodes keyed by objectId. @type {Map<number, HTMLElement>} */
const alertBadges = new Map();

/** Last known risk entries keyed by objectId. @type {Map<number, object>} */
const riskStore = new Map();

// ── Renderer init ─────────────────────────────────────────────────────────────

renderer.init();
tryAttachCamera();

// ── Camera feed ───────────────────────────────────────────────────────────────

/**
 * Try to attach the device camera to the video element.
 * Falls silently if not supported or permission is denied.
 */
async function tryAttachCamera() {
  if (!navigator.mediaDevices?.getUserMedia) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 } },
    });
    cameraFeed.srcObject = stream;
  } catch (_) {
    // Camera unavailable — the dark background acts as the "feed".
  }
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

function wsUrl() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${location.host}${WS_PATH}`;
}

function connect() {
  if (ws && ws.readyState <= WebSocket.OPEN) return;
  userDisconnected = false;
  setStatus('connecting');

  ws = new WebSocket(wsUrl());

  ws.addEventListener('open', () => {
    reconnectDelay = RECONNECT_BASE_MS;
    setStatus('connected');
  });

  ws.addEventListener('message', (evt) => {
    let msg;
    try {
      msg = JSON.parse(evt.data);
    } catch (e) {
      console.warn('[atlas] Bad WS message', e);
      return;
    }
    applyDelta(msg);
  });

  ws.addEventListener('close', () => {
    setStatus('disconnected');
    if (!userDisconnected) scheduleReconnect();
  });

  ws.addEventListener('error', () => {
    setStatus('disconnected');
  });
}

function disconnect() {
  userDisconnected = true;
  ws?.close();
  setStatus('disconnected');
}

function scheduleReconnect() {
  setTimeout(() => {
    if (!userDisconnected) connect();
  }, reconnectDelay);
  reconnectDelay = Math.min(reconnectDelay * 2, RECONNECT_MAX_MS);
}

// ── Delta application ─────────────────────────────────────────────────────────

/**
 * Apply a RiskDeltaMessage to the renderer and the DOM.
 * @param {{ added: object[], updated: object[], removed: number[] }} delta
 */
function applyDelta(delta) {
  const { added = [], updated = [], removed = [] } = delta;

  for (const entry of added) {
    riskStore.set(entry.object_id, entry);
    renderer.addRisk(entry);
    upsertAlertBadge(entry);
  }

  for (const entry of updated) {
    riskStore.set(entry.object_id, entry);
    renderer.updateRisk(entry);
    upsertAlertBadge(entry);
  }

  for (const id of removed) {
    riskStore.delete(id);
    renderer.removeRisk(id);
    removeAlertBadge(id);
  }

  updateHUD();
}

// ── Alert badges ──────────────────────────────────────────────────────────────

/**
 * Create or update the CSS alert badge for a risk entry.
 * @param {object} entry
 */
function upsertAlertBadge(entry) {
  let badge = alertBadges.get(entry.object_id);
  if (!badge) {
    badge = document.createElement('div');
    badge.className = 'alert-badge';
    badge.innerHTML = '<div class="dot"></div><span class="label-text"></span>';
    alertContainer.appendChild(badge);
    alertBadges.set(entry.object_id, badge);
  }

  const score = entry.combined_score ?? 0;
  badge.classList.remove('severity-low', 'severity-mid', 'severity-high');
  if (score >= 0.65)      badge.classList.add('severity-high');
  else if (score >= 0.35) badge.classList.add('severity-mid');
  else                    badge.classList.add('severity-low');

  const alertData = entry.overlay?.alert;
  const text = alertData?.text ?? entry.object_label ?? `Object ${entry.object_id}`;
  badge.querySelector('.label-text').textContent =
    text.length > 50 ? text.slice(0, 48) + '…' : text;

  // Position: prefer server-side projected screen_position if available.
  let px = null, py = null;

  if (alertData?.screen_position) {
    [px, py] = alertData.screen_position;
  } else if (entry.position) {
    // Fall back to Three.js projection.
    const ndc = renderer.projectToNDC(entry.position);
    px = (ndc.x * 0.5 + 0.5) * window.innerWidth;
    py = (-ndc.y * 0.5 + 0.5) * window.innerHeight;
  }

  if (px !== null && py !== null) {
    badge.style.left = `${px}px`;
    badge.style.top  = `${py}px`;
    badge.style.display = '';
  } else {
    badge.style.display = 'none';
  }
}

/**
 * Remove the alert badge for a deactivated risk.
 * @param {number} id
 */
function removeAlertBadge(id) {
  const badge = alertBadges.get(id);
  if (badge) {
    badge.remove();
    alertBadges.delete(id);
  }
}

// ── HUD update ────────────────────────────────────────────────────────────────

function updateHUD() {
  const count = riskStore.size;
  statRisks.textContent = String(count);
  statObjs.textContent  = String(count);  // Approximation until /objects feed is wired.

  if (count > 0) {
    riskCountBadge.textContent = `${count} risk${count === 1 ? '' : 's'}`;
    riskCountBadge.style.display = '';
  } else {
    riskCountBadge.style.display = 'none';
  }
}

// ── Status helpers ────────────────────────────────────────────────────────────

/** @param {'connected'|'connecting'|'disconnected'} state */
function setStatus(state) {
  wsDot.className  = state;
  wsLabel.textContent = state.charAt(0).toUpperCase() + state.slice(1);
  connectBtn.textContent = (state === 'connected' || state === 'connecting')
    ? 'Disconnect'
    : 'Connect';
}

// ── Button handler ────────────────────────────────────────────────────────────

connectBtn.addEventListener('click', () => {
  if (ws && ws.readyState <= WebSocket.OPEN) {
    disconnect();
  } else {
    connect();
  }
});

// ── Auto-connect on page load ─────────────────────────────────────────────────

connect();
