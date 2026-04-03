/**
 * overlay.js — Three.js AR overlay renderer for Atlas-0 risk visualisation.
 *
 * Manages a Three.js scene rendered on top of the camera feed canvas.
 * Each risk is represented as a set of 3D objects:
 *
 *   RiskZone     — semi-transparent wireframe sphere at the object centre.
 *   TrajectoryArc — dashed line following the predicted fall path.
 *   ImpactZone   — flat ring on the impact surface.
 *
 * Alert badges are managed as DOM elements by app.js using the projected
 * screen positions supplied in the WebSocket payload.
 *
 * Usage:
 *   import { OverlayRenderer } from './overlay.js';
 *   const renderer = new OverlayRenderer(document.getElementById('overlay-canvas'));
 *   await renderer.init();
 *   renderer.addRisk(riskEntry);     // from WebSocket "added"
 *   renderer.updateRisk(riskEntry);  // from WebSocket "updated"
 *   renderer.removeRisk(objectId);  // from WebSocket "removed"
 */

import * as THREE from 'three';

// ── Colour helpers ────────────────────────────────────────────────────────────

/**
 * Convert a [0,255,255,255] RGB array to a THREE.Color.
 * @param {number[]} rgb
 * @returns {THREE.Color}
 */
function rgbToColor(rgb) {
  return new THREE.Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
}

// ── Geometry builders ─────────────────────────────────────────────────────────

/**
 * Build a wireframe sphere mesh for a risk zone.
 * @param {number[]} center  [x, y, z]
 * @param {number}   radius
 * @param {number[]} color   [r, g, b] 0-255
 * @param {number}   opacity
 * @returns {THREE.Mesh}
 */
function buildRiskZoneMesh(center, radius, color, opacity) {
  const geo = new THREE.SphereGeometry(radius, 16, 12);
  const mat = new THREE.MeshBasicMaterial({
    color: rgbToColor(color),
    transparent: true,
    opacity,
    wireframe: true,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(...center);
  return mesh;
}

/**
 * Build a solid (filled) inner sphere for the risk zone glow effect.
 * @param {number[]} center
 * @param {number}   radius
 * @param {number[]} color
 * @param {number}   opacity  (applied at 30% of outer opacity)
 * @returns {THREE.Mesh}
 */
function buildRiskZoneGlow(center, radius, color, opacity) {
  const geo = new THREE.SphereGeometry(radius * 0.92, 16, 12);
  const mat = new THREE.MeshBasicMaterial({
    color: rgbToColor(color),
    transparent: true,
    opacity: opacity * 0.28,
    depthWrite: false,
    side: THREE.BackSide,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(...center);
  return mesh;
}

/**
 * Build a Three.js Line following the trajectory arc points.
 * @param {number[][]} points  Array of [x, y, z] positions.
 * @returns {THREE.Line}
 */
function buildTrajectoryLine(points) {
  const positions = points.flatMap(p => p);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const mat = new THREE.LineBasicMaterial({
    color: 0xffdd00,
    transparent: true,
    opacity: 0.85,
    linewidth: 2,   // Note: linewidth > 1 is only respected on some platforms
  });
  return new THREE.Line(geo, mat);
}

/**
 * Build a ring mesh at the impact zone centre (flat on Y plane).
 * @param {number[]} center  [x, y, z]
 * @param {number[]} polygon List of [x, y, z] polygon vertices.
 * @returns {THREE.Mesh}
 */
function buildImpactRing(center, polygon) {
  // Estimate radius from the polygon vertex distances.
  let maxR = 0;
  for (const v of polygon) {
    const dx = v[0] - center[0];
    const dz = v[2] - center[2];
    const r = Math.sqrt(dx * dx + dz * dz);
    if (r > maxR) maxR = r;
  }
  const innerR = Math.max(0.01, maxR * 0.6);
  const geo = new THREE.RingGeometry(innerR, maxR, Math.max(8, polygon.length));
  // Rotate flat on the XZ plane (ring geometry lies in XY by default).
  geo.rotateX(-Math.PI / 2);
  const mat = new THREE.MeshBasicMaterial({
    color: 0xff2020,
    transparent: true,
    opacity: 0.45,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(center[0], center[1] + 0.01, center[2]);
  return mesh;
}

// ── OverlayRenderer ───────────────────────────────────────────────────────────

export class OverlayRenderer {
  /**
   * @param {HTMLCanvasElement} canvas  The overlay canvas element.
   */
  constructor(canvas) {
    this._canvas = canvas;
    /** @type {THREE.WebGLRenderer|null} */
    this._renderer = null;
    /** @type {THREE.Scene} */
    this._scene = new THREE.Scene();
    /** @type {THREE.PerspectiveCamera} */
    this._camera = null;
    /**
     * Map of objectId → { zone, glow, arc, impact } Three.js objects.
     * @type {Map<number, object>}
     */
    this._riskObjects = new Map();
    this._animFrameId = null;
  }

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /**
   * Initialise the Three.js renderer and start the animation loop.
   */
  init() {
    const w = this._canvas.clientWidth  || window.innerWidth;
    const h = this._canvas.clientHeight || window.innerHeight;

    this._renderer = new THREE.WebGLRenderer({
      canvas: this._canvas,
      alpha: true,
      antialias: true,
    });
    this._renderer.setSize(w, h, false);
    this._renderer.setPixelRatio(window.devicePixelRatio);
    this._renderer.setClearColor(0x000000, 0);   // transparent background

    this._camera = new THREE.PerspectiveCamera(60, w / h, 0.01, 200);
    this._camera.position.set(0, 3, 6);
    this._camera.lookAt(0, 1, 0);

    // Ambient fill so wireframes look slightly better.
    const ambient = new THREE.AmbientLight(0xffffff, 0.5);
    this._scene.add(ambient);

    // Floor grid (subtle guide).
    const grid = new THREE.GridHelper(20, 20, 0x222233, 0x151520);
    grid.position.y = 0;
    this._scene.add(grid);

    this._startLoop();
    window.addEventListener('resize', () => this._onResize());
  }

  /** Dispose all resources and stop the loop. */
  dispose() {
    if (this._animFrameId !== null) {
      cancelAnimationFrame(this._animFrameId);
    }
    this._riskObjects.forEach((_, id) => this.removeRisk(id));
    this._renderer?.dispose();
  }

  // ── Risk management ───────────────────────────────────────────────────────

  /**
   * Add visual objects for a new risk entry.
   * @param {object} entry  OverlayRiskEntry from the WebSocket "added" list.
   */
  addRisk(entry) {
    if (this._riskObjects.has(entry.object_id)) {
      this.updateRisk(entry);
      return;
    }
    this._riskObjects.set(entry.object_id, this._buildGroup(entry));
  }

  /**
   * Replace visual objects for an updated risk entry.
   * @param {object} entry  OverlayRiskEntry from the WebSocket "updated" list.
   */
  updateRisk(entry) {
    this.removeRisk(entry.object_id);
    this._riskObjects.set(entry.object_id, this._buildGroup(entry));
  }

  /**
   * Remove all visual objects for a risk that is no longer active.
   * @param {number} objectId
   */
  removeRisk(objectId) {
    const group = this._riskObjects.get(objectId);
    if (!group) return;
    for (const obj of Object.values(group)) {
      if (obj instanceof THREE.Object3D) {
        this._scene.remove(obj);
        obj.geometry?.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) {
            obj.material.forEach(m => m.dispose());
          } else {
            obj.material.dispose();
          }
        }
      }
    }
    this._riskObjects.delete(objectId);
  }

  /** Remove all risk visuals from the scene. */
  clearAll() {
    for (const id of [...this._riskObjects.keys()]) {
      this.removeRisk(id);
    }
  }

  /** @returns {number} Count of currently rendered risk objects. */
  get riskCount() {
    return this._riskObjects.size;
  }

  /**
   * Project a world-space position to normalised device coordinates using
   * the current Three.js camera.  Used by app.js for alert badge placement.
   *
   * @param {number[]} worldPos  [x, y, z]
   * @returns {{ x: number, y: number }}  NDC in [-1, 1]; y is NOT flipped.
   */
  projectToNDC(worldPos) {
    const v = new THREE.Vector3(...worldPos);
    v.project(this._camera);
    return { x: v.x, y: v.y };
  }

  // ── Private ───────────────────────────────────────────────────────────────

  /**
   * Build and add all Three.js objects for a risk entry.
   * @param {object} entry
   * @returns {object}  { zone, glow, arc?, impact? }
   */
  _buildGroup(entry) {
    const overlay = entry.overlay ?? {};
    const zoneData = overlay.risk_zone;
    const arcData  = overlay.trajectory_arc;
    const impData  = overlay.impact_zone;

    const group = {};

    if (zoneData) {
      const zone = buildRiskZoneMesh(
        zoneData.center, zoneData.radius, zoneData.color, zoneData.opacity,
      );
      const glow = buildRiskZoneGlow(
        zoneData.center, zoneData.radius, zoneData.color, zoneData.opacity,
      );
      this._scene.add(zone, glow);
      group.zone = zone;
      group.glow = glow;
    }

    if (arcData && Array.isArray(arcData.points) && arcData.points.length >= 2) {
      const arc = buildTrajectoryLine(arcData.points);
      this._scene.add(arc);
      group.arc = arc;
    }

    if (impData && Array.isArray(impData.polygon) && impData.polygon.length >= 3) {
      const ring = buildImpactRing(impData.center, impData.polygon);
      this._scene.add(ring);
      group.impact = ring;
    }

    return group;
  }

  _startLoop() {
    const tick = () => {
      this._animFrameId = requestAnimationFrame(tick);
      this._render();
    };
    tick();
  }

  _render() {
    if (!this._renderer) return;
    // Slow camera orbit for demo aesthetics when no live camera feed is present.
    const t = performance.now() * 0.0002;
    this._camera.position.x = Math.sin(t) * 6;
    this._camera.position.z = Math.cos(t) * 6;
    this._camera.lookAt(0, 1, 0);

    this._renderer.render(this._scene, this._camera);
  }

  _onResize() {
    if (!this._renderer || !this._camera) return;
    const w = window.innerWidth;
    const h = window.innerHeight;
    this._camera.aspect = w / h;
    this._camera.updateProjectionMatrix();
    this._renderer.setSize(w, h, false);
  }
}
