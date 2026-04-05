/**
 * scene_viewer.js — Three.js 3-D scene viewer.
 *
 * Fetches objects + risks from the API and renders them as labelled,
 * colour-coded spheres with a grid floor.  OrbitControls let the user
 * orbit, zoom, and pan freely.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as api from './api.js';

const CLR = {
  high:   0xff3050,
  mid:    0xff8c00,
  low:    0xffd700,
  safe:   0x00c8ff,
  grid:   0x0a1525,
  gridDiv: 0x0d1c30,
};

const HEX = {
  high:  '#ff3050',
  mid:   '#ff8c00',
  low:   '#ffd700',
  safe:  '#00c8ff',
};

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
    this._meshes    = new Map();   // objectId → mesh
    this._rings     = [];
    this._animId    = null;
    this._autoRefresh = false;
    this._autoTimer   = null;
  }

  init() {
    const canvas = this._canvas;
    const w = canvas.clientWidth  || canvas.offsetWidth  || 800;
    const h = canvas.clientHeight || canvas.offsetHeight || 600;

    // Renderer
    this._renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this._renderer.setSize(w, h, false);
    this._renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this._renderer.setClearColor(0x000000, 0);

    // Scene
    this._scene = new THREE.Scene();
    this._scene.fog = new THREE.FogExp2(0x05060e, 0.06);

    // Camera
    this._camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 200);
    this._camera.position.set(6, 9, 12);
    this._camera.lookAt(0, 0, 0);

    // Controls
    this._controls = new OrbitControls(this._camera, canvas);
    this._controls.enableDamping    = true;
    this._controls.dampingFactor    = 0.07;
    this._controls.minDistance      = 2;
    this._controls.maxDistance      = 60;
    this._controls.maxPolarAngle    = Math.PI * 0.87;
    this._controls.mouseButtons     = {
      LEFT:   THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT:  THREE.MOUSE.PAN,
    };

    // Grid floor
    const grid = new THREE.GridHelper(30, 30, CLR.grid, CLR.gridDiv);
    this._scene.add(grid);

    // Axis helper (subtle)
    const axes = new THREE.AxesHelper(2);
    axes.material.opacity = 0.25;
    axes.material.transparent = true;
    this._scene.add(axes);

    // Lights
    this._scene.add(new THREE.AmbientLight(0x223355, 1.2));
    const sun = new THREE.DirectionalLight(0x00aaff, 0.5);
    sun.position.set(8, 14, 8);
    this._scene.add(sun);

    // Resize observer
    const ro = new ResizeObserver(() => this._resize());
    ro.observe(canvas.parentElement || document.body);

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
      this._draw(scene);
    } catch {
      // server may not be reachable yet
    }
  }

  _draw(scene) {
    // Remove old meshes + rings
    this._meshes.forEach(m => this._scene.remove(m));
    this._meshes.clear();
    this._rings.forEach(r => this._scene.remove(r));
    this._rings = [];

    const objs  = scene.objects || [];
    const risks = scene.risks   || [];

    if (objs.length === 0) {
      this._emptyEl.style.display = '';
      this._objListEl.innerHTML = '<div style="font-size:11px;color:var(--text-muted)">—</div>';
      return;
    }

    this._emptyEl.style.display = 'none';

    const riskMap = {};
    risks.forEach(r => { riskMap[r.object_id] = r.risk_score; });

    objs.forEach(obj => {
      const rs    = riskMap[obj.object_id] || 0;
      const color = this._color(rs);
      const hex   = this._hex(rs);

      // Sphere
      const geo = new THREE.SphereGeometry(0.28, 20, 20);
      const mat = new THREE.MeshPhongMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.35,
        transparent: true,
        opacity: 0.88,
      });
      const mesh = new THREE.Mesh(geo, mat);
      const [x, y, z] = obj.position;
      mesh.position.set(x, y + 0.28, z);
      mesh.userData = { rs };
      this._scene.add(mesh);
      this._meshes.set(obj.object_id, mesh);

      // Floor ring
      const rg = new THREE.RingGeometry(0.32, 0.38, 28);
      const rm = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.35,
        side: THREE.DoubleSide,
      });
      const ring = new THREE.Mesh(rg, rm);
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(x, 0.01, z);
      this._scene.add(ring);
      this._rings.push(ring);

      // HUD list entry
      const li = document.createElement('div');
      li.className = 's-obj-item';
      li.innerHTML = `<div class="s-obj-dot" style="background:${hex}"></div><span>${esc(obj.label)}</span>`;
      if (this._objListEl.querySelector('div[style]')) this._objListEl.innerHTML = '';
      this._objListEl.appendChild(li);
    });
  }

  _color(rs) {
    if (rs >= 0.65) return CLR.high;
    if (rs >= 0.35) return CLR.mid;
    if (rs >= 0.1)  return CLR.low;
    return CLR.safe;
  }

  _hex(rs) {
    if (rs >= 0.65) return HEX.high;
    if (rs >= 0.35) return HEX.mid;
    if (rs >= 0.1)  return HEX.low;
    return HEX.safe;
  }

  _animate() {
    this._animId = requestAnimationFrame(() => this._animate());
    this._controls.update();
    // Gentle idle bob
    const t = performance.now() * 0.0008;
    this._meshes.forEach((mesh, id) => {
      mesh.position.y = mesh.userData.baseY ?? (mesh.position.y + (mesh.userData.baseY = mesh.position.y, 0));
      mesh.position.y = (mesh.userData.baseY || 0.28) + Math.sin(t + id * 0.9) * 0.05;
    });
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
    if (this._animId)  cancelAnimationFrame(this._animId);
    clearInterval(this._autoTimer);
    this._renderer?.dispose();
  }
}

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
