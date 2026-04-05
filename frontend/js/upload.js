/**
 * upload.js — Drag-and-drop upload view with pipeline stage animation.
 */

import * as api from './api.js';

const STAGES      = ['upload', 'ingest', 'vlm', 'risk', 'complete'];
const STAGE_LABEL = { upload: 'UPLOAD', ingest: 'INGEST', vlm: 'VLM', risk: 'RISK', complete: 'DONE' };
const POLL_MS     = 1400;

export class UploadView {
  /** @param {{ dropZone, fileInput, queue, queueWrap }} els */
  constructor(els) {
    this._dz        = els.dropZone;
    this._input     = els.fileInput;
    this._queue     = els.queue;
    this._queueWrap = els.queueWrap;
    this._cards     = new Map();   // jobId → card element
    this._pollers   = new Map();   // jobId → intervalId
  }

  init() {
    const dz = this._dz;

    dz.addEventListener('click', () => this._input.click());

    this._input.addEventListener('change', (e) => {
      Array.from(e.target.files || []).forEach(f => this._handle(f));
      e.target.value = '';
    });

    dz.addEventListener('dragover', (e) => {
      e.preventDefault();
      dz.classList.add('drag-over');
    });

    dz.addEventListener('dragleave', (e) => {
      if (!dz.contains(/** @type {Node} */(e.relatedTarget))) {
        dz.classList.remove('drag-over');
      }
    });

    dz.addEventListener('drop', (e) => {
      e.preventDefault();
      dz.classList.remove('drag-over');
      Array.from(e.dataTransfer.files).forEach(f => this._handle(f));
    });
  }

  _handle(file) {
    this._queueWrap.style.display = '';
    this._doUpload(file);
  }

  async _doUpload(file) {
    const card = this._makeCard(file);
    this._queue.prepend(card);

    // Image thumbnail
    if (file.type.startsWith('image/')) {
      const img = card.querySelector('.job-thumb img');
      if (img) img.src = URL.createObjectURL(file);
    }

    try {
      const job = await api.uploadFile(file);
      card.dataset.jobId = job.job_id;
      this._cards.set(job.job_id, card);
      this._sync(card, job);
      this._poll(job.job_id, card);
    } catch (err) {
      this._setError(card, String(err.message));
    }
  }

  _poll(jobId, card) {
    const id = setInterval(async () => {
      try {
        const job = await api.fetchJob(jobId);
        this._sync(card, job);
        if (job.status === 'complete' || job.status === 'error') {
          clearInterval(id);
          this._pollers.delete(jobId);
        }
      } catch {
        clearInterval(id);
      }
    }, POLL_MS);
    this._pollers.set(jobId, id);
  }

  _makeCard(file) {
    const isVid = file.type.startsWith('video/');

    const videoIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <polygon points="23 7 16 12 23 17 23 7"/>
      <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
    </svg>`;

    const stagesHtml = STAGES.map((s, i) => `
      <div class="ps ${s === 'upload' ? 'active' : ''}" data-stage="${s}">
        <div class="ps-dot"></div>
        <span>${STAGE_LABEL[s]}</span>
      </div>
      ${i < STAGES.length - 1 ? '<div class="ps-line"></div>' : ''}
    `).join('');

    const card = document.createElement('div');
    card.className = 'job-card';
    card.innerHTML = `
      <div class="job-thumb">
        ${isVid ? videoIcon : '<img alt="" />'}
      </div>
      <div class="job-body">
        <div class="job-name">${esc(file.name)}</div>
        <div class="job-meta">${fmtSize(file.size)} · ${file.type || 'unknown'}</div>
        <div class="pipeline">${stagesHtml}</div>
        <div class="job-tags" style="display:none"></div>
      </div>
      <div class="job-badge processing">UPLOADING</div>
    `;
    return card;
  }

  _sync(card, job) {
    // Badge
    const badge = card.querySelector('.job-badge');
    badge.className = `job-badge ${job.status}`;
    badge.textContent = job.status.toUpperCase();

    card.classList.remove('complete', 'error');
    if (job.status === 'complete' || job.status === 'error') {
      card.classList.add(job.status);
    }

    // Stages
    const curIdx = STAGES.indexOf(job.stage);
    STAGES.forEach((s, i) => {
      const el = card.querySelector(`.ps[data-stage="${s}"]`);
      const ln = card.querySelectorAll('.ps-line')[i];
      if (!el) return;

      el.classList.remove('done', 'active');

      const isDone = job.status === 'complete' || job.status === 'error';
      if (i < curIdx || (isDone && i <= curIdx)) {
        el.classList.add('done');
        if (ln) ln.classList.add('done');
      } else if (i === curIdx && !isDone) {
        el.classList.add('active');
      }
    });

    // Results
    if (job.status === 'complete') {
      const tagsEl = card.querySelector('.job-tags');
      const objs   = job.objects || [];

      if (objs.length > 0) {
        tagsEl.style.display = '';
        tagsEl.innerHTML = objs.map(o => {
          const rs  = (job.risks || []).find(r => r.object_label === o.label);
          const cls = rs
            ? rs.risk_score >= 0.65 ? 'hi' : rs.risk_score >= 0.35 ? 'md' : ''
            : '';
          return `<span class="jtag ${cls}">${esc(o.label)} · ${esc(o.material)}</span>`;
        }).join('');
      }

      if (job.error) {
        card.querySelector('.job-meta').textContent = job.error;
      }
    }

    if (job.status === 'error' && job.error) {
      card.querySelector('.job-meta').textContent = job.error;
    }
  }

  _setError(card, msg) {
    card.classList.add('error');
    card.querySelector('.job-badge').className = 'job-badge error';
    card.querySelector('.job-badge').textContent = 'ERROR';
    card.querySelector('.job-meta').textContent = msg;
  }
}

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function fmtSize(b) {
  if (b < 1024)        return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
}
