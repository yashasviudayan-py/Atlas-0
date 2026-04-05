/**
 * intelligence.js — Objects / Risks panels and natural-language query UI.
 */

import * as api from './api.js';

export class IntelligenceView {
  constructor(els) {
    this._qInput    = els.queryInput;
    this._qBtn      = els.queryBtn;
    this._qResults  = els.queryResults;
    this._objList   = els.objList;
    this._riskList  = els.riskList;
    this._objCnt    = els.objCount;
    this._riskCnt   = els.riskCount;
  }

  init() {
    this._qBtn.addEventListener('click', () => this._query());
    this._qInput.addEventListener('keydown', e => {
      if (e.key === 'Enter') this._query();
    });
  }

  async refresh() {
    try {
      const [objects, scene] = await Promise.all([api.fetchObjects(), api.fetchScene()]);
      this._renderObjs(objects, scene.risks || []);
      this._renderRisks(scene.risks || []);
    } catch { /* server may not be up */ }
  }

  // ── Objects ────────────────────────────────────────────────────────────────

  _renderObjs(objects, risks) {
    this._objCnt.textContent = String(objects.length);

    if (!objects.length) {
      this._objList.innerHTML = `
        <div class="empty">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
          </svg>
          <p>No objects detected yet.<br>Upload images or start the SLAM pipeline.</p>
        </div>`;
      return;
    }

    const riskMap = {};
    risks.forEach(r => { riskMap[r.object_id] = r.risk_score; });

    this._objList.innerHTML = objects.map(obj => {
      const frag = Math.round(obj.fragility * 100);
      const fric = Math.round(obj.friction  * 100);
      const conf = Math.round(obj.confidence * 100);
      const mass = Math.min(100, obj.mass_kg * 8);
      return `
        <div class="obj-card">
          <div class="oc-top">
            <span class="oc-label">${esc(obj.label)}</span>
            <span class="oc-mat">${esc(obj.material)}</span>
          </div>
          <div class="oc-props">
            <div class="oc-prop">
              <span class="oc-plabel">Fragility</span>
              <div class="oc-bar"><div class="oc-fill frag" style="width:${frag}%"></div></div>
              <span class="oc-pval">${obj.fragility.toFixed(2)}</span>
            </div>
            <div class="oc-prop">
              <span class="oc-plabel">Friction</span>
              <div class="oc-bar"><div class="oc-fill fric" style="width:${fric}%"></div></div>
              <span class="oc-pval">${obj.friction.toFixed(2)}</span>
            </div>
            <div class="oc-prop">
              <span class="oc-plabel">Mass</span>
              <div class="oc-bar"><div class="oc-fill" style="width:${mass}%"></div></div>
              <span class="oc-pval">${obj.mass_kg.toFixed(2)} kg</span>
            </div>
            <div class="oc-prop">
              <span class="oc-plabel">Confidence</span>
              <div class="oc-bar"><div class="oc-fill" style="width:${conf}%"></div></div>
              <span class="oc-pval">${conf}%</span>
            </div>
          </div>
        </div>`;
    }).join('');
  }

  // ── Risks ──────────────────────────────────────────────────────────────────

  _renderRisks(risks) {
    this._riskCnt.textContent = String(risks.length);

    if (!risks.length) {
      this._riskList.innerHTML = `
        <div class="empty">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
          <p>No risks assessed yet.</p>
        </div>`;
      return;
    }

    const sorted = [...risks].sort((a, b) => b.risk_score - a.risk_score);
    this._riskList.innerHTML = sorted.map(r => {
      const cls = r.risk_score >= 0.65 ? 'hi' : r.risk_score >= 0.35 ? 'md' : 'lo';
      const pct = Math.round(r.risk_score * 100);
      return `
        <div class="risk-card ${cls}">
          <div class="rc-top">
            <span class="rc-label">${esc(r.object_label)}</span>
            <span class="rc-score">${pct}%</span>
          </div>
          <p class="rc-desc">${esc(r.description)}</p>
          <div class="rc-bar"><div class="rc-fill" style="width:${pct}%"></div></div>
        </div>`;
    }).join('');
  }

  // ── Query ──────────────────────────────────────────────────────────────────

  async _query() {
    const q = this._qInput.value.trim();
    if (!q) return;

    this._qBtn.textContent = '…';
    this._qBtn.disabled    = true;

    try {
      const results = await api.postQuery(q);
      this._showQueryResults(results);
    } catch (err) {
      this._qResults.className = 'vis';
      this._qResults.innerHTML = `
        <div class="qr-item" style="color:var(--accent-red);font-size:12px">${esc(err.message)}</div>`;
    } finally {
      this._qBtn.textContent = 'Ask';
      this._qBtn.disabled    = false;
    }
  }

  _showQueryResults(results) {
    this._qResults.className = 'vis';
    if (!results.length) {
      this._qResults.innerHTML = `
        <div class="qr-item" style="color:var(--text-muted);font-size:12px">No results found.</div>`;
      return;
    }
    this._qResults.innerHTML = results.map(r => `
      <div class="qr-item">
        <div>
          <div class="qr-label">${esc(r.object_label)}</div>
          <div class="qr-desc">${esc(r.description)}</div>
        </div>
        <div class="qr-pos">(${r.position.map(v => v.toFixed(2)).join(', ')})</div>
      </div>`).join('');
  }
}

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
