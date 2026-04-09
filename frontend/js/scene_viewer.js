/**
 * scene_viewer.js — Atlas-0 Neural Scene Viewer v2.
 *
 * Visual systems:
 *   1. Gaussian particle clouds  — 3DGS crystallisation simulation
 *   2. Neural Tether lines       — canvas overlay + DOM Reasoning Cards
 *   3. Knowledge Graph           — animated force-directed canvas (top-right panel)
 *   4. Physics Ghost             — semi-transparent ghost + dotted arc + collision warning
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as api from './api.js';

// ── Palette ───────────────────────────────────────────────────────────────────

const S = {
  CYAN:   '#00f0ff',
  GREEN:  '#39ff14',
  DANGER: '#ff2040',
  GHOST:  '#b48eff',
  ORANGE: '#ff8c00',
  YELLOW: '#ffd700',
  MUTED:  'rgba(80,140,180,0.5)',
};

// ── Gaussian Particle Cloud ───────────────────────────────────────────────────

const VERT = `
  attribute float aSize;
  attribute float aAlpha;
  attribute vec3  aOffset;
  uniform   float uCryst;
  uniform   float uTime;
  uniform   vec3  uColor;
  varying   float vAlpha;
  varying   vec3  vCol;

  void main() {
    float t  = uCryst * uCryst * (3.0 - 2.0 * uCryst);
    vec3 scattered    = position + aOffset;
    vec3 crystallised = position;

    float wave = sin(uTime * 4.0 + position.x * 6.0 + position.z * 4.0);
    vec3 pos   = mix(scattered, crystallised, t);
    pos.y     += wave * (1.0 - t) * 0.07;
    pos.y     += sin(uTime * 1.6 + aOffset.x * 2.5) * t * 0.025;

    vAlpha = aAlpha * (0.2 + t * 0.8);
    vCol   = uColor;

    vec4 mv = modelViewMatrix * vec4(pos, 1.0);
    gl_Position  = projectionMatrix * mv;
    // Clamp point size: base 1-3 px range, modest distance scale, hard cap at 28px
    float rawSize = aSize * (1.0 + (1.0 - t) * 0.8) * (55.0 / -mv.z);
    gl_PointSize  = clamp(rawSize, 0.5, 28.0);
  }
`;

const FRAG = `
  varying float vAlpha;
  varying vec3  vCol;

  void main() {
    vec2  uv = gl_PointCoord - 0.5;
    float d  = length(uv);
    if (d > 0.5) discard;
    float soft = exp(-d * d * 9.0) * vAlpha;
    float core = exp(-d * d * 48.0) * 0.85;
    gl_FragColor = vec4(vCol + core, soft);
  }
`;

class GaussianCloud {
  constructor(scene, position, hexColor) {
    this._scene  = scene;
    this._target = new THREE.Vector3(...position);
    this._phase  = Math.random() * 100;
    this._cryst  = 0;

    const N   = 240;
    const pos = new Float32Array(N * 3);
    const off = new Float32Array(N * 3);
    const sz  = new Float32Array(N);
    const alp = new Float32Array(N);

    for (let i = 0; i < N; i++) {
      const r = 0.5 + Math.random() * 2.2;
      const θ = Math.random() * Math.PI * 2;
      const φ = Math.acos(2 * Math.random() - 1);
      off[i*3]   = r * Math.sin(φ) * Math.cos(θ);
      off[i*3+1] = r * Math.sin(φ) * Math.sin(θ);
      off[i*3+2] = r * Math.cos(φ);
      pos[i*3]   = this._target.x;
      pos[i*3+1] = this._target.y;
      pos[i*3+2] = this._target.z;
      sz[i]  = 0.8 + Math.random() * 2.8;
      alp[i] = 0.35 + Math.random() * 0.65;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('aOffset',  new THREE.BufferAttribute(off, 3));
    geo.setAttribute('aSize',    new THREE.BufferAttribute(sz,  1));
    geo.setAttribute('aAlpha',   new THREE.BufferAttribute(alp, 1));

    this._mat = new THREE.ShaderMaterial({
      vertexShader:   VERT,
      fragmentShader: FRAG,
      uniforms: {
        uColor: { value: new THREE.Color(hexColor) },
        uCryst: { value: 0 },
        uTime:  { value: 0 },
      },
      transparent: true,
      depthWrite:  false,
      blending:    THREE.AdditiveBlending,
    });

    this._pts = new THREE.Points(geo, this._mat);
    scene.add(this._pts);
  }

  tick(dt) {
    this._phase += dt;
    this._cryst  = Math.min(1, this._cryst + dt * 0.20);
    this._mat.uniforms.uCryst.value = this._cryst;
    this._mat.uniforms.uTime.value  = this._phase;
  }

  get position() { return this._target; }

  dispose() {
    this._scene.remove(this._pts);
    this._pts.geometry.dispose();
    this._mat.dispose();
  }
}

// ── Knowledge Graph ───────────────────────────────────────────────────────────

class KnowledgeGraph {
  constructor(canvas) {
    this._canvas = canvas;
    this._ctx    = canvas.getContext('2d');
    this._nodes  = [];
    this._edges  = [];
    this._t      = 0;
    this._animId = null;
  }

  setObjects(objects) {
    const W  = this._canvas.width  || 270;
    const H  = this._canvas.height || 210;
    const cx = W / 2;
    const cy = H / 2;

    // Object nodes — arranged in a circle
    const n = Math.min(objects.length, 5);
    const objNodes = objects.slice(0, n).map((o, i) => {
      const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
      const r = Math.min(W, H) * 0.30;
      return {
        id:     o.object_id,
        label:  o.label.length > 9 ? o.label.slice(0, 8) + '…' : o.label,
        type:   'object',
        x:      cx + r * Math.cos(angle),
        y:      cy + r * Math.sin(angle),
        vx: 0, vy: 0,
        color:  S.CYAN,
        r:      16,
        _obj:   o,
      };
    });

    // Concept nodes + edges derived from object properties
    const concepts = new Map();
    const edges = [];

    const getConcept = (label, color) => {
      if (!concepts.has(label)) {
        concepts.set(label, {
          id: 'c_' + label,
          label: label.length > 9 ? label.slice(0, 8) + '…' : label,
          type: 'concept',
          x: cx + (Math.random() - 0.5) * W * 0.45,
          y: cy + (Math.random() - 0.5) * H * 0.45,
          vx: 0, vy: 0,
          color: color || S.MUTED,
          r: 11,
        });
      }
      return concepts.get(label);
    };

    for (const n of objNodes) {
      const o = n._obj;
      if (o.fragility > 0.60) {
        edges.push({ from: n, to: getConcept('Breakage', S.DANGER), label: 'Risk' });
      }
      if (/cup|glass|bottle|mug|vase|bowl/i.test(o.label)) {
        edges.push({ from: n, to: getConcept('Liquid', S.ORANGE), label: 'Contains' });
      }
      if (o.mass_kg > 8) {
        edges.push({ from: n, to: getConcept('Heavy', S.MUTED), label: 'Weight' });
      }
      if (/table|desk|shelf|counter|surface/i.test(o.label)) {
        edges.push({ from: n, to: getConcept('Surface', S.GREEN), label: 'Is' });
      }
      if (/lamp|light|bulb/i.test(o.label)) {
        edges.push({ from: n, to: getConcept('Illuminates', S.YELLOW), label: 'Emits' });
      }
    }

    this._nodes = [...objNodes, ...concepts.values()];
    this._edges = edges;
  }

  start() {
    if (this._animId) return;
    const loop = () => {
      this._animId = requestAnimationFrame(loop);
      this._tick(1 / 60);
    };
    loop();
  }

  stop() {
    if (this._animId) { cancelAnimationFrame(this._animId); this._animId = null; }
  }

  _tick(dt) {
    this._t += dt;
    const W  = this._canvas.width  || 270;
    const H  = this._canvas.height || 210;
    const cx = W / 2;
    const cy = H / 2;

    // Force-directed layout
    const REPEL  = 700;
    const SPRING = 12;
    const TARGET = 68;
    const CENTER = 0.008;
    const DAMP   = 0.85;

    for (const n of this._nodes) {
      let fx = 0, fy = 0;
      for (const m of this._nodes) {
        if (m === n) continue;
        const dx = n.x - m.x;
        const dy = n.y - m.y;
        const d2 = dx*dx + dy*dy + 1;
        const f  = REPEL / d2;
        const d  = Math.sqrt(d2);
        fx += (dx / d) * f;
        fy += (dy / d) * f;
      }
      for (const e of this._edges) {
        if (e.from !== n && e.to !== n) continue;
        const other = e.from === n ? e.to : e.from;
        const dx = other.x - n.x;
        const dy = other.y - n.y;
        const d  = Math.sqrt(dx*dx + dy*dy) + 0.01;
        const stretch = d - TARGET;
        fx += (dx / d) * stretch * SPRING * dt;
        fy += (dy / d) * stretch * SPRING * dt;
      }
      fx += (cx - n.x) * CENTER;
      fy += (cy - n.y) * CENTER;

      n.vx = (n.vx + fx * dt) * DAMP;
      n.vy = (n.vy + fy * dt) * DAMP;
      n.x  = Math.max(n.r + 4, Math.min(W - n.r - 4, n.x + n.vx));
      n.y  = Math.max(n.r + 4, Math.min(H - n.r - 4, n.y + n.vy));
    }

    this._draw(W, H);
  }

  _draw(W, H) {
    const ctx = this._ctx;
    ctx.clearRect(0, 0, W, H);

    // Edges
    for (const e of this._edges) {
      const mx = (e.from.x + e.to.x) / 2;
      const my = (e.from.y + e.to.y) / 2 - 5;

      ctx.save();
      ctx.strokeStyle = 'rgba(0,240,255,0.18)';
      ctx.lineWidth   = 1.5;
      ctx.shadowColor = '#00f0ff';
      ctx.shadowBlur  = 8;
      ctx.beginPath();
      ctx.moveTo(e.from.x, e.from.y);
      ctx.lineTo(e.to.x, e.to.y);
      ctx.stroke();
      ctx.restore();

      // Edge label
      ctx.save();
      ctx.fillStyle   = 'rgba(120,190,220,0.65)';
      ctx.font        = '7px "JetBrains Mono", monospace';
      ctx.textAlign   = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(e.label, mx, my);
      ctx.restore();

      // Travelling dot
      const prog = ((this._t * 0.45) % 1.0);
      const tx = e.from.x + (e.to.x - e.from.x) * prog;
      const ty = e.from.y + (e.to.y - e.from.y) * prog;
      ctx.save();
      ctx.fillStyle   = '#00f0ff';
      ctx.shadowColor = '#00f0ff';
      ctx.shadowBlur  = 7;
      ctx.beginPath();
      ctx.arc(tx, ty, 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Nodes
    for (const n of this._nodes) {
      const pulse = 1 + Math.sin(this._t * 2.2 + n.x * 0.05) * 0.05;
      const r = n.r * pulse;

      // Outer glow ring
      ctx.save();
      ctx.strokeStyle  = n.color;
      ctx.lineWidth    = 0.8;
      ctx.globalAlpha  = 0.28;
      ctx.shadowColor  = n.color;
      ctx.shadowBlur   = 14;
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      // Node fill
      ctx.save();
      ctx.fillStyle   = 'rgba(2,7,16,0.92)';
      ctx.strokeStyle = n.color;
      ctx.lineWidth   = n.type === 'object' ? 1.2 : 0.8;
      ctx.shadowColor = n.color;
      ctx.shadowBlur  = 10;
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();

      // Label
      ctx.save();
      ctx.fillStyle    = n.type === 'object' ? '#d8eeff' : n.color;
      ctx.font         = `${n.type === 'object' ? 8 : 7}px "JetBrains Mono", monospace`;
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(n.label, n.x, n.y);
      ctx.restore();
    }

    // Scan-line shimmer
    const scanY = ((this._t * 0.25) % 1) * H;
    const grad = ctx.createLinearGradient(0, scanY - 5, 0, scanY + 5);
    grad.addColorStop(0, 'transparent');
    grad.addColorStop(0.5, 'rgba(0,240,255,0.035)');
    grad.addColorStop(1, 'transparent');
    ctx.fillStyle = grad;
    ctx.fillRect(0, scanY - 5, W, 10);
  }

  dispose() { this.stop(); }
}

// ── Physics Ghost ─────────────────────────────────────────────────────────────

class PhysicsGhost {
  constructor(scene) {
    this._scene     = scene;
    this._active    = false;
    this._t         = 0;
    this._period    = 4.8;
    this._onCollide = null;
    this._meshes    = [];
    this._ghost     = null;
    this._impactRing = null;
    this._start     = new THREE.Vector3(0.5, 0.9, 0.3);
    this._end       = new THREE.Vector3(1.9, 0.0, 1.4);
  }

  setTarget(position) {
    this._start.set(position[0], position[1] + 0.1, position[2]);
    this._end.set(
      position[0] + 1.1 + (Math.random() - 0.5) * 0.5,
      0.0,
      position[2] + 0.9 + (Math.random() - 0.5) * 0.5,
    );
    if (this._active) {
      this._clear();
      this._build();
    }
  }

  setCollideCallback(fn) { this._onCollide = fn; }

  activate() {
    if (this._active) return;
    this._active = true;
    this._build();
  }

  _build() {
    this._clear();

    // Ghost sphere
    const geo = new THREE.SphereGeometry(0.20, 20, 20);
    const mat = new THREE.MeshPhongMaterial({
      color:            0xb48eff,
      emissive:         0xb48eff,
      emissiveIntensity: 0.5,
      transparent:      true,
      opacity:          0.22,
      depthWrite:       false,
    });
    this._ghost = new THREE.Mesh(geo, mat);
    this._scene.add(this._ghost);
    this._meshes.push(this._ghost);

    // Trajectory arc — dashed via LineSegments (skip every other pair)
    const pts = [];
    for (let i = 0; i <= 48; i++) {
      pts.push(this._getPos(i / 48));
    }
    const dashPts = [];
    for (let i = 0; i < pts.length - 1; i += 2) dashPts.push(pts[i], pts[i+1]);

    const arcGeo = new THREE.BufferGeometry().setFromPoints(dashPts);
    const arcMat = new THREE.LineBasicMaterial({
      color: 0xb48eff, transparent: true, opacity: 0.45, depthWrite: false,
    });
    const arc = new THREE.LineSegments(arcGeo, arcMat);
    this._scene.add(arc);
    this._meshes.push(arc);

    // Impact ring at landing zone
    const rg = new THREE.RingGeometry(0.16, 0.26, 28);
    const rm = new THREE.MeshBasicMaterial({
      color: 0xff2040, transparent: true, opacity: 0.55,
      side: THREE.DoubleSide, depthWrite: false,
    });
    this._impactRing = new THREE.Mesh(rg, rm);
    this._impactRing.rotation.x = -Math.PI / 2;
    this._impactRing.position.copy(this._end).setY(0.01);
    this._scene.add(this._impactRing);
    this._meshes.push(this._impactRing);
  }

  _getPos(tt) {
    const x = this._start.x + (this._end.x - this._start.x) * tt;
    const z = this._start.z + (this._end.z - this._start.z) * tt;
    const y = this._start.y * (1 - tt)
            + (-4.9 * tt * tt + 1.1 * tt * (1 - tt)) * 1.4;
    return new THREE.Vector3(x, Math.max(0, y), z);
  }

  tick(dt) {
    if (!this._active) return;
    this._t = (this._t + dt / this._period) % 1;
    const pos = this._getPos(this._t);

    if (this._ghost) {
      this._ghost.position.copy(pos);
      this._ghost.rotation.z += dt * 1.8;
      this._ghost.rotation.x += dt * 0.9;
    }

    if (this._impactRing) {
      const s = 1 + Math.sin(this._t * Math.PI * 8) * 0.18 * (1 - this._t);
      this._impactRing.scale.set(s, s, s);
    }

    // Collision window: last 18% of arc
    const colliding = this._t > 0.82;
    this._onCollide?.(colliding);
  }

  _clear() {
    for (const m of this._meshes) {
      this._scene.remove(m);
      m.geometry?.dispose();
      m.material?.dispose();
    }
    this._meshes = [];
    this._ghost      = null;
    this._impactRing = null;
  }

  dispose() { this._clear(); }
}

// ── Neural Tether System ──────────────────────────────────────────────────────

class NeuralTetherSystem {
  constructor(tetherCanvas, cardsContainer) {
    this._canvas    = tetherCanvas;
    this._ctx       = tetherCanvas.getContext('2d');
    this._container = cardsContainer;
    this._objects   = [];
    this._cards     = new Map();
    this._camera    = null;
    this._t         = 0;
  }

  setCamera(camera) { this._camera = camera; }

  setObjects(objects, risks) {
    const riskMap = {};
    for (const r of (risks || [])) riskMap[r.object_id] = r;
    this._objects = objects.map(o => ({ ...o, _risk: riskMap[o.object_id] }));

    const seen = new Set();
    for (const obj of this._objects) {
      seen.add(obj.object_id);
      if (!this._cards.has(obj.object_id)) {
        const el = this._buildCard(obj);
        this._container.appendChild(el);
        this._cards.set(obj.object_id, el);
      } else {
        this._updateCard(this._cards.get(obj.object_id), obj);
      }
    }
    for (const [id, el] of this._cards) {
      if (!seen.has(id)) { el.remove(); this._cards.delete(id); }
    }
  }

  _buildCard(obj) {
    const el = document.createElement('div');
    el.className = 'reasoning-card';
    this._updateCard(el, obj);
    return el;
  }

  _updateCard(el, obj) {
    const rs  = obj._risk?.risk_score ?? 0;
    const cls = rs >= 0.65 ? 'rc-hi' : rs >= 0.35 ? 'rc-md' : 'rc-lo';
    el.className = `reasoning-card ${cls}`;
    el.innerHTML = `
      <div class="rc-header">
        <span class="rc-name">${esc(obj.label)}</span>
        <span class="rc-pct">${Math.round(rs * 100)}%</span>
      </div>
      <div class="rc-grid">
        <span class="rck">MASS</span>     <span class="rcv">${obj.mass_kg.toFixed(2)} kg</span>
        <span class="rck">FRICTION</span> <span class="rcv">${obj.friction.toFixed(2)}</span>
        <span class="rck">MATERIAL</span> <span class="rcv">${esc(obj.material)}</span>
        <span class="rck">FRAGILITY</span><span class="rcv">${obj.fragility.toFixed(2)}</span>
      </div>`;
  }

  draw(dt) {
    if (!this._camera) return;
    this._t += dt;

    const ctx = this._ctx;
    const W   = this._canvas.width;
    const H   = this._canvas.height;
    ctx.clearRect(0, 0, W, H);

    for (const obj of this._objects) {
      const card = this._cards.get(obj.object_id);
      if (!card) continue;

      const v3  = new THREE.Vector3(...obj.position);
      const ndc = v3.clone().project(this._camera);
      if (ndc.z > 1.0) { card.style.opacity = '0'; continue; }

      const sx = (ndc.x *  0.5 + 0.5) * W;
      const sy = (-ndc.y * 0.5 + 0.5) * H;

      const cardW = 172;
      const cardH = 100;
      let   cx    = sx + 56;
      let   cy    = sy - cardH / 2;

      if (cx + cardW > W - 10) cx = sx - cardW - 56;
      if (cy < 8)               cy = 8;
      if (cy + cardH > H - 8)  cy = H - cardH - 8;

      card.style.left    = `${cx}px`;
      card.style.top     = `${cy}px`;
      card.style.opacity = '1';

      const anchorX = cx < sx ? cx + cardW : cx;
      const anchorY = cy + cardH / 2;
      const pulse   = 0.45 + Math.sin(this._t * 2.8 + obj.position[0]) * 0.28;

      // Glow pass
      ctx.save();
      ctx.strokeStyle = 'rgba(0,240,255,0.12)';
      ctx.lineWidth   = 3;
      ctx.shadowColor = '#00f0ff';
      ctx.shadowBlur  = 14;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(anchorX, anchorY);
      ctx.stroke();
      // Sharp core
      ctx.shadowBlur  = 0;
      ctx.strokeStyle = `rgba(0,240,255,${(0.38 + pulse * 0.38).toFixed(2)})`;
      ctx.lineWidth   = 0.75;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(anchorX, anchorY);
      ctx.stroke();
      // Origin node dot
      ctx.fillStyle   = '#00f0ff';
      ctx.shadowColor = '#00f0ff';
      ctx.shadowBlur  = 12;
      ctx.beginPath();
      ctx.arc(sx, sy, 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  dispose() {
    for (const el of this._cards.values()) el.remove();
    this._cards.clear();
  }
}

// ── Upload Point Cloud renderer ───────────────────────────────────────────────
// Renders image-derived pseudo-depth points as coloured Gaussian splats.

const PC_VERT = `
  attribute float aSize;
  attribute vec3  aColor;
  uniform   float uTime;
  uniform   float uAlpha;
  varying   vec3  vColor;
  varying   float vAlpha;

  void main() {
    // Gentle depth-pulsing shimmer
    float jitter = sin(uTime * 2.0 + position.x * 4.0 + position.z * 3.0) * 0.02;
    vec3 pos = position + vec3(0.0, jitter, 0.0);

    vColor = aColor;
    vAlpha = uAlpha;

    vec4 mv     = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mv;
    float sz    = aSize * (50.0 / -mv.z);
    gl_PointSize = clamp(sz, 0.5, 20.0);
  }
`;

const PC_FRAG = `
  varying vec3  vColor;
  varying float vAlpha;

  void main() {
    vec2  uv = gl_PointCoord - 0.5;
    float d  = length(uv);
    if (d > 0.5) discard;
    float soft = exp(-d * d * 10.0) * vAlpha;
    float core = exp(-d * d * 50.0) * 0.6;
    gl_FragColor = vec4(vColor + core, soft);
  }
`;

class UploadPointCloud {
  constructor(scene) {
    this._scene  = scene;
    this._pts    = null;
    this._mat    = null;
    this._t      = 0;
    this._fadeIn = 0; // 0 → 1
  }

  /** @param {number[][]} points  — [[x,y,z,r,g,b], …] */
  setPoints(points) {
    this._clear();
    if (!points || points.length === 0) return;

    const N   = points.length;
    const pos = new Float32Array(N * 3);
    const col = new Float32Array(N * 3);
    const sz  = new Float32Array(N);

    for (let i = 0; i < N; i++) {
      const [x, y, z, r, g, b] = points[i];
      pos[i*3]   = x;
      pos[i*3+1] = y;
      pos[i*3+2] = z;
      col[i*3]   = r;
      col[i*3+1] = g;
      col[i*3+2] = b;
      sz[i]      = 0.6 + Math.random() * 1.4;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('aColor',   new THREE.BufferAttribute(col, 3));
    geo.setAttribute('aSize',    new THREE.BufferAttribute(sz,  1));

    this._mat = new THREE.ShaderMaterial({
      vertexShader:   PC_VERT,
      fragmentShader: PC_FRAG,
      uniforms: {
        uTime:  { value: 0 },
        uAlpha: { value: 0 },
      },
      transparent: true,
      depthWrite:  false,
      blending:    THREE.AdditiveBlending,
    });

    this._pts   = new THREE.Points(geo, this._mat);
    this._fadeIn = 0;
    this._scene.add(this._pts);
  }

  tick(dt) {
    if (!this._mat) return;
    this._t      += dt;
    this._fadeIn  = Math.min(1, this._fadeIn + dt * 0.4);
    this._mat.uniforms.uTime.value  = this._t;
    this._mat.uniforms.uAlpha.value = this._fadeIn * 0.75;
  }

  _clear() {
    if (this._pts) {
      this._scene.remove(this._pts);
      this._pts.geometry.dispose();
      this._mat?.dispose();
      this._pts = null;
      this._mat = null;
    }
  }

  dispose() { this._clear(); }
}

// ── SceneViewer ───────────────────────────────────────────────────────────────

export class SceneViewer {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {HTMLElement}       emptyEl
   * @param {HTMLElement}       objListEl
   */
  constructor(canvas, emptyEl, objListEl) {
    this._canvas    = canvas;
    this._emptyEl   = emptyEl;
    this._objListEl = objListEl;

    this._renderer  = null;
    this._scene     = null;
    this._camera    = null;
    this._controls  = null;
    this._clouds    = new Map();
    this._animId    = null;
    this._prevTime  = 0;
    this._colliding = false;
    this._autoRefresh = false;
    this._autoTimer   = null;

    this._tether   = null;
    this._kg       = null;
    this._ghost    = null;
    this._pcCloud  = null;
  }

  init() {
    const canvas = this._canvas;
    const w = canvas.clientWidth  || canvas.offsetWidth  || 900;
    const h = canvas.clientHeight || canvas.offsetHeight || 600;

    // Renderer
    this._renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this._renderer.setSize(w, h, false);
    this._renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this._renderer.setClearColor(0x020408, 1);

    // Scene
    this._scene = new THREE.Scene();
    this._scene.fog = new THREE.FogExp2(0x020810, 0.042);

    // Camera
    this._camera = new THREE.PerspectiveCamera(46, w / h, 0.1, 200);
    this._camera.position.set(5, 7, 11);
    this._camera.lookAt(0, 1, 0);

    // Orbit controls
    this._controls = new OrbitControls(this._camera, canvas);
    this._controls.enableDamping  = true;
    this._controls.dampingFactor  = 0.065;
    this._controls.minDistance    = 2;
    this._controls.maxDistance    = 55;
    this._controls.maxPolarAngle  = Math.PI * 0.84;
    this._controls.target.set(0, 1, 0);

    // Fine grid floor
    const g1 = new THREE.Color(0x040c1a);
    const g2 = new THREE.Color(0x071525);
    const grid = new THREE.GridHelper(32, 52, g1, g2);
    /** @type {THREE.Material} */ (grid.material).opacity = 0.55;
    /** @type {THREE.Material} */ (grid.material).transparent = true;
    this._scene.add(grid);

    // Lights
    this._scene.add(new THREE.AmbientLight(0x0a1428, 1.1));
    const keyLight = new THREE.DirectionalLight(0x0055cc, 0.45);
    keyLight.position.set(7, 14, 8);
    this._scene.add(keyLight);
    const fillLight = new THREE.PointLight(0x00f0ff, 0.3, 25);
    fillLight.position.set(-3, 4, -3);
    this._scene.add(fillLight);

    // Neural Tether system
    const tetherCanvas = /** @type {HTMLCanvasElement|null} */ (document.getElementById('tether-canvas'));
    const cardsEl      = document.getElementById('reasoning-cards');
    if (tetherCanvas && cardsEl) {
      this._tether = new NeuralTetherSystem(tetherCanvas, cardsEl);
      this._tether.setCamera(this._camera);
    }

    // Knowledge Graph
    const kgCanvas = /** @type {HTMLCanvasElement|null} */ (document.getElementById('kg-canvas'));
    if (kgCanvas) {
      this._kg = new KnowledgeGraph(kgCanvas);
      this._kg.start();
    }

    // Upload point cloud renderer
    this._pcCloud = new UploadPointCloud(this._scene);

    // Physics Ghost
    this._ghost = new PhysicsGhost(this._scene);
    this._ghost.setCollideCallback(colliding => {
      const warn = document.getElementById('collision-warning');
      if (!warn) return;
      if (colliding && !this._colliding) {
        this._colliding = true;
        warn.classList.add('active');
      } else if (!colliding && this._colliding) {
        this._colliding = false;
        warn.classList.remove('active');
      }
    });

    new ResizeObserver(() => this._resize()).observe(
      canvas.parentElement || document.body,
    );

    this._animate();
    this.refresh();
  }

  setAutoRefresh(on) {
    this._autoRefresh = on;
    clearInterval(this._autoTimer);
    if (on) this._autoTimer = setInterval(() => this.refresh(), 3000);
  }

  async refresh() {
    try {
      const scene = await api.fetchScene();
      const objs  = scene.objects || [];
      const risks = scene.risks   || [];
      this._draw(objs, risks, scene);
    } catch {
      this._draw([], [], null);
    }
  }

  _draw(objs, risks, scene) {
    if (!objs.length) {
      this._pcCloud?.setPoints([]);
      this._emptyEl.style.display = '';
      this._objListEl.innerHTML = '<div style="font-size:11px;color:var(--text-muted)">—</div>';
      return;
    }

    this._emptyEl.style.display = 'none';

    const riskMap = {};
    for (const r of risks) riskMap[r.object_id] = r.risk_score ?? 0;

    // Rebuild Gaussian clouds
    this._clouds.forEach(c => c.dispose());
    this._clouds.clear();

    for (const obj of objs) {
      const rs    = riskMap[obj.object_id] ?? 0;
      const color = this._riskColor(rs);
      this._clouds.set(obj.object_id, new GaussianCloud(this._scene, obj.position, color));
    }

    // Upload-derived point cloud (pseudo-depth reconstruction)
    this._pcCloud?.setPoints(scene?.point_cloud || []);

    // Neural tethers + cards
    this._tether?.setObjects(objs, risks);

    // Knowledge graph
    this._kg?.setObjects(objs);

    // Physics ghost — target highest fragility × risk object
    const ghostTarget = [...objs].sort((a, b) => {
      const ra = riskMap[a.object_id] ?? 0;
      const rb = riskMap[b.object_id] ?? 0;
      return (rb + b.fragility * 0.5) - (ra + a.fragility * 0.5);
    })[0];

    if (ghostTarget) {
      this._ghost?.setTarget(ghostTarget.position);
      this._ghost?.activate();

      const warn = document.getElementById('collision-warning');
      if (warn) {
        const ql = warn.querySelector('.cw-label');
        if (ql) ql.textContent = `COLLISION PREDICTED · ${ghostTarget.label.toUpperCase()}`;
      }
    }

    // HUD object list
    this._objListEl.innerHTML = objs.map(o => {
      const rs  = riskMap[o.object_id] ?? 0;
      const hex = this._riskColor(rs);
      return `<div class="s-obj-item">
        <div class="s-obj-dot" style="background:${hex};box-shadow:0 0 4px ${hex}66"></div>
        <span>${esc(o.label)}</span>
      </div>`;
    }).join('');
  }

  _riskColor(rs) {
    if (rs >= 0.65) return '#ff2040';
    if (rs >= 0.35) return '#ff8c00';
    if (rs >= 0.10) return '#ffd700';
    return '#00f0ff';
  }

  _animate() {
    this._animId = requestAnimationFrame(() => this._animate());

    const now = performance.now() * 0.001;
    const dt  = Math.min(now - (this._prevTime || now), 0.05);
    this._prevTime = now;

    this._controls.update();
    this._clouds.forEach(c => c.tick(dt));
    this._pcCloud?.tick(dt);
    this._ghost?.tick(dt);

    // Sync tether canvas size with scene canvas
    const tc = /** @type {HTMLCanvasElement|null} */ (document.getElementById('tether-canvas'));
    if (tc) {
      const sw = this._canvas.clientWidth;
      const sh = this._canvas.clientHeight;
      if (tc.width !== sw || tc.height !== sh) { tc.width = sw; tc.height = sh; }
    }

    this._tether?.draw(dt);
    this._renderer.render(this._scene, this._camera);
  }

  _resize() {
    const canvas = this._canvas;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    this._camera.aspect = w / h;
    this._camera.updateProjectionMatrix();
    this._renderer.setSize(w, h, false);
  }

  dispose() {
    if (this._animId) cancelAnimationFrame(this._animId);
    clearInterval(this._autoTimer);
    this._clouds.forEach(c => c.dispose());
    this._pcCloud?.dispose();
    this._tether?.dispose();
    this._kg?.dispose();
    this._ghost?.dispose();
    this._renderer?.dispose();
  }
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
