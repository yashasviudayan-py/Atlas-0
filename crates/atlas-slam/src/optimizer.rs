//! Gaussian Splatting optimizer — differentiable renderer + Adam updates.
//!
//! This module implements the core 3DGS optimization loop:
//!
//! 1. **Render** the current set of [`Gaussian3D`] splats onto an image by
//!    projecting each Gaussian to screen space and alpha-compositing front to
//!    back.
//! 2. **Compute** the photometric MSE loss between the rendered image and the
//!    actual camera frame.
//! 3. **Backpropagate** analytical gradients for *opacity* and *SH DC colour*
//!    coefficients and apply [Adam] updates.
//! 4. **Prune** Gaussians whose opacity has fallen below a configurable floor.
//!
//! [Adam]: https://arxiv.org/abs/1412.6980

use atlas_core::{Gaussian3D, Pose};
use nalgebra::{Matrix3, Matrix4};

use crate::config::CameraIntrinsics;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Tuning parameters for the [`GaussianOptimizer`].
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Adam learning rate for the opacity parameter.
    pub lr_opacity: f32,
    /// Adam learning rate for the SH DC colour coefficients.
    pub lr_color: f32,
    /// Gaussians with opacity below this threshold are pruned.
    pub min_opacity: f32,
    /// Number of optimization steps to run per keyframe.
    pub iterations: u32,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            lr_opacity: 0.05,
            lr_color: 0.01,
            min_opacity: 0.005,
            iterations: 10,
        }
    }
}

// ─── Adam state ───────────────────────────────────────────────────────────────

/// Per-parameter Adam first- and second-moment accumulators.
struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    step: u32,
    beta1: f32,
    beta2: f32,
    eps: f32,
}

impl AdamState {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            step: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// Grow internal buffers to accommodate `n` parameters.
    fn ensure_capacity(&mut self, n: usize) {
        if n > self.m.len() {
            self.m.resize(n, 0.0);
            self.v.resize(n, 0.0);
        }
    }

    /// Advance the global step counter.  Call once per optimisation step
    /// *before* calling [`Self::delta`] for any parameter.
    fn advance(&mut self) {
        self.step += 1;
    }

    /// Compute the Adam parameter delta for index `i` with gradient `grad`.
    ///
    /// Returns the value to *subtract* from the parameter (i.e. δ = lr × m̂ / √v̂).
    fn delta(&mut self, i: usize, grad: f32, lr: f32) -> f32 {
        self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad;
        self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad * grad;
        let m_hat = self.m[i] / (1.0 - self.beta1.powi(self.step as i32));
        let v_hat = self.v[i] / (1.0 - self.beta2.powi(self.step as i32));
        lr * m_hat / (v_hat.sqrt() + self.eps)
    }
}

// ─── GaussianOptimizer ────────────────────────────────────────────────────────

/// Runs the differentiable 3DGS rendering loop and updates Gaussian parameters.
pub struct GaussianOptimizer {
    config: OptimizerConfig,
    opacity_adam: AdamState,
    color_adam: AdamState,
}

impl GaussianOptimizer {
    /// Create a new optimizer with the given configuration.
    #[must_use]
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            opacity_adam: AdamState::new(0),
            color_adam: AdamState::new(0),
        }
    }

    // ── Rendering ─────────────────────────────────────────────────────────────

    /// Render `gaussians` into an RGB image of size `width × height` as seen
    /// from `pose`, using the supplied camera `intrinsics`.
    ///
    /// Gaussians are projected to 2-D, sorted front-to-back, and
    /// alpha-composited using the standard 3DGS formula.
    ///
    /// Returns a `width × height × 3` byte slice (row-major, RGB).
    #[must_use]
    pub fn render(
        gaussians: &[Gaussian3D],
        width: u32,
        height: u32,
        pose: &Pose,
        intrinsics: &CameraIntrinsics,
    ) -> Vec<u8> {
        let (rendered_f, _transmittance) = render_float(gaussians, width, height, pose, intrinsics);

        let mut out = vec![0u8; (width * height * 3) as usize];
        for (i, rgb) in rendered_f.iter().enumerate() {
            out[i * 3] = (rgb[0].clamp(0.0, 1.0) * 255.0) as u8;
            out[i * 3 + 1] = (rgb[1].clamp(0.0, 1.0) * 255.0) as u8;
            out[i * 3 + 2] = (rgb[2].clamp(0.0, 1.0) * 255.0) as u8;
        }
        out
    }

    // ── Optimization ─────────────────────────────────────────────────────────

    /// Run one optimization step: render, compute MSE loss, apply Adam updates
    /// to opacity and SH DC colour coefficients.
    ///
    /// Returns the scalar MSE loss for this step.
    pub fn optimize_step(
        &mut self,
        gaussians: &mut [Gaussian3D],
        target_rgb: &[u8],
        width: u32,
        height: u32,
        pose: &Pose,
        intrinsics: &CameraIntrinsics,
    ) -> f32 {
        let n_gauss = gaussians.len();
        if n_gauss == 0 {
            return 0.0;
        }

        self.opacity_adam.ensure_capacity(n_gauss);
        // Each Gaussian has 3 colour channels.
        self.color_adam.ensure_capacity(n_gauss * 3);

        // ── Forward pass ──────────────────────────────────────────────────────
        let (rendered_f, transmittance) = render_float(gaussians, width, height, pose, intrinsics);

        // Collect per-splat screen projections to compute per-pixel gradients.
        let splats = project_splats(gaussians, width, height, pose, intrinsics);

        // ── Loss ──────────────────────────────────────────────────────────────
        let n_pixels = (width * height) as f32;
        let mut loss = 0.0f32;
        for (i, rgb) in rendered_f.iter().enumerate() {
            for c in 0..3 {
                let diff = rgb[c] - target_rgb[i * 3 + c] as f32 / 255.0;
                loss += diff * diff;
            }
        }
        loss /= n_pixels * 3.0;

        // ── Gradient accumulation ─────────────────────────────────────────────
        // dL/d(rendered_color[px][c]) = 2/(N*3) * (rendered - target)
        let mut opacity_grads = vec![0.0f32; n_gauss];
        let mut color_grads = vec![[0.0f32; 3]; n_gauss];

        for splat in &splats {
            let g = &gaussians[splat.gauss_idx];
            let cov = &splat.cov2d;
            let det = cov[0] * cov[2] - cov[1] * cov[1];
            if det < 1e-6 {
                continue;
            }
            let inv_det = 1.0 / det;
            let inv00 = cov[2] * inv_det;
            let inv01 = -cov[1] * inv_det;
            let inv11 = cov[0] * inv_det;

            let sigma = (cov[0].max(cov[2])).sqrt() * 3.0;
            let x_min = ((splat.px - sigma) as i32).max(0) as u32;
            let x_max = ((splat.px + sigma) as i32 + 1).min(width as i32) as u32;
            let y_min = ((splat.py - sigma) as i32).max(0) as u32;
            let y_max = ((splat.py + sigma) as i32 + 1).min(height as i32) as u32;

            for y in y_min..y_max {
                for x in x_min..x_max {
                    let dx = x as f32 - splat.px;
                    let dy = y as f32 - splat.py;
                    let power = -0.5 * (inv00 * dx * dx + 2.0 * inv01 * dx * dy + inv11 * dy * dy);
                    if power < -4.0 {
                        continue;
                    }
                    let gauss_val = power.exp();
                    let alpha = (g.opacity * gauss_val).min(0.999);

                    let px_idx = (y * width + x) as usize;
                    let t = transmittance[px_idx];

                    // dL/d(rendered_c) for each channel.
                    for c in 0..3 {
                        let rendered_c = rendered_f[px_idx][c];
                        let target_c = target_rgb[px_idx * 3 + c] as f32 / 255.0;
                        let dl_dr = 2.0 / (n_pixels * 3.0) * (rendered_c - target_c);

                        // Gradient w.r.t. SH DC colour: contribution is T·α·dl_dr.
                        color_grads[splat.gauss_idx][c] += t * alpha * dl_dr;

                        // Gradient w.r.t. opacity: gauss_val * T * color * dl_dr.
                        opacity_grads[splat.gauss_idx] += gauss_val * t * splat.color[c] * dl_dr;
                    }
                }
            }
        }

        // ── Adam updates ──────────────────────────────────────────────────────
        self.opacity_adam.advance();
        self.color_adam.advance();

        const C0: f32 = 0.282_094_8;
        for (i, g) in gaussians.iter_mut().enumerate() {
            // Opacity update (clamp to (0, 1)).
            let d_op = self
                .opacity_adam
                .delta(i, opacity_grads[i], self.config.lr_opacity);
            g.opacity = (g.opacity - d_op).clamp(0.0, 1.0);

            // SH DC colour update.
            if g.sh_coefficients.len() >= 3 {
                for (c, &cg) in color_grads[i].iter().enumerate() {
                    let d_sh = self.color_adam.delta(i * 3 + c, cg, self.config.lr_color);
                    g.sh_coefficients[c] -= d_sh;
                }
                // Keep the resulting colour plausible.
                let max_sh = 0.5 / C0; // maps to colour = 1.0 after SH → RGB
                for sh in g.sh_coefficients.iter_mut().take(3) {
                    *sh = sh.clamp(-max_sh, max_sh);
                }
            }
        }

        loss
    }

    /// Remove Gaussians whose opacity has dropped below `config.min_opacity`.
    pub fn prune_low_opacity(&self, gaussians: &mut Vec<Gaussian3D>) {
        gaussians.retain(|g| g.opacity >= self.config.min_opacity);
    }

    /// Run `config.iterations` optimization steps, then prune.
    ///
    /// Convenience wrapper used by the tracker.
    pub fn optimize(
        &mut self,
        gaussians: &mut Vec<Gaussian3D>,
        target_rgb: &[u8],
        width: u32,
        height: u32,
        pose: &Pose,
        intrinsics: &CameraIntrinsics,
    ) {
        for _ in 0..self.config.iterations {
            self.optimize_step(gaussians, target_rgb, width, height, pose, intrinsics);
        }
        self.prune_low_opacity(gaussians);
    }
}

// ─── Projected splat ─────────────────────────────────────────────────────────

/// Cached screen-space data for a single Gaussian.
struct ProjectedSplat {
    /// Index into the Gaussian array.
    gauss_idx: usize,
    /// 2-D projected position (pixels).
    px: f32,
    py: f32,
    /// Camera-space depth (used for depth sorting).
    depth: f32,
    /// 2-D covariance upper triangle [σ11, σ12, σ22].
    cov2d: [f32; 3],
    /// View-independent colour (SH DC → RGB, clamped to [0, 1]).
    color: [f32; 3],
}

// ─── Rendering helpers ────────────────────────────────────────────────────────

/// Project all Gaussians to screen space, returning visible splats sorted by
/// depth (front to back).
fn project_splats(
    gaussians: &[Gaussian3D],
    width: u32,
    height: u32,
    pose: &Pose,
    intrinsics: &CameraIntrinsics,
) -> Vec<ProjectedSplat> {
    let world_to_cam = pose
        .to_matrix()
        .try_inverse()
        .unwrap_or_else(Matrix4::identity);

    let cam_rot = Matrix3::new(
        world_to_cam[(0, 0)],
        world_to_cam[(0, 1)],
        world_to_cam[(0, 2)],
        world_to_cam[(1, 0)],
        world_to_cam[(1, 1)],
        world_to_cam[(1, 2)],
        world_to_cam[(2, 0)],
        world_to_cam[(2, 1)],
        world_to_cam[(2, 2)],
    );

    let fx = intrinsics.fx as f32;
    let fy = intrinsics.fy as f32;
    let cx = intrinsics.cx as f32;
    let cy = intrinsics.cy as f32;

    const C0: f32 = 0.282_094_8;

    let mut splats = Vec::with_capacity(gaussians.len());

    for (gauss_idx, g) in gaussians.iter().enumerate() {
        // Transform Gaussian centre to camera space.
        let pw = nalgebra::Vector3::new(g.center.x, g.center.y, g.center.z);
        let t_cam = nalgebra::Vector3::new(
            world_to_cam[(0, 3)],
            world_to_cam[(1, 3)],
            world_to_cam[(2, 3)],
        );
        let pc = cam_rot * pw + t_cam;

        if pc[2] <= 0.1 {
            continue; // Behind the camera.
        }

        let inv_z = 1.0 / pc[2];
        let px = fx * pc[0] * inv_z + cx;
        let py = fy * pc[1] * inv_z + cy;

        // Cull splats completely off-screen.
        let spread = (g.scale[0].max(g.scale[1]) * fx * inv_z * 3.0).max(4.0);
        if px < -spread
            || px >= width as f32 + spread
            || py < -spread
            || py >= height as f32 + spread
        {
            continue;
        }

        // 2-D covariance  Σ₂D = J · R_cw · Σ₃D · R_cw^T · J^T
        // J = [[fx/z, 0, −fx·x/z²], [0, fy/z, −fy·y/z²]]
        let j00 = fx * inv_z;
        let j02 = -fx * pc[0] * inv_z * inv_z;
        let j11 = fy * inv_z;
        let j12 = -fy * pc[1] * inv_z * inv_z;

        // Unpack upper-triangle covariance → full 3×3.
        let s = &g.covariance;
        let cov3d = [[s[0], s[1], s[2]], [s[1], s[3], s[4]], [s[2], s[4], s[5]]];

        // Rotate 3-D covariance to camera space: R_cw · Σ₃D · R_cw^T.
        let mut cov_cam = [[0.0f32; 3]; 3];
        for r in 0..3usize {
            for c in 0..3usize {
                let mut v = 0.0f32;
                for k in 0..3usize {
                    for l in 0..3usize {
                        v += cam_rot[(r, k)] * cov3d[k][l] * cam_rot[(c, l)];
                    }
                }
                cov_cam[r][c] = v;
            }
        }

        // Project to 2-D: J · cov_cam · J^T.
        let jc = [
            [
                j00 * cov_cam[0][0] + j02 * cov_cam[2][0],
                j00 * cov_cam[0][1] + j02 * cov_cam[2][1],
                j00 * cov_cam[0][2] + j02 * cov_cam[2][2],
            ],
            [
                j11 * cov_cam[1][0] + j12 * cov_cam[2][0],
                j11 * cov_cam[1][1] + j12 * cov_cam[2][1],
                j11 * cov_cam[1][2] + j12 * cov_cam[2][2],
            ],
        ];
        // Low-pass filter offset (0.3) prevents excessively sharp splats.
        let s11 = jc[0][0] * j00 + jc[0][2] * j02 + 0.3;
        let s12 = jc[0][1] * j11 + jc[0][2] * j12;
        let s22 = jc[1][1] * j11 + jc[1][2] * j12 + 0.3;

        // View-independent colour from SH DC coefficient.
        let color = if g.sh_coefficients.len() >= 3 {
            [
                (C0 * g.sh_coefficients[0] + 0.5).clamp(0.0, 1.0),
                (C0 * g.sh_coefficients[1] + 0.5).clamp(0.0, 1.0),
                (C0 * g.sh_coefficients[2] + 0.5).clamp(0.0, 1.0),
            ]
        } else {
            [0.5, 0.5, 0.5]
        };

        splats.push(ProjectedSplat {
            gauss_idx,
            px,
            py,
            depth: pc[2],
            cov2d: [s11, s12, s22],
            color,
        });
    }

    // Sort front-to-back so that alpha compositing is correct.
    splats.sort_by(|a, b| {
        a.depth
            .partial_cmp(&b.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    splats
}

/// Alpha-composite the projected splats into a float RGB buffer.
///
/// Returns `(rendered: Vec<[f32;3]>, transmittance: Vec<f32>)`.
fn render_float(
    gaussians: &[Gaussian3D],
    width: u32,
    height: u32,
    pose: &Pose,
    intrinsics: &CameraIntrinsics,
) -> (Vec<[f32; 3]>, Vec<f32>) {
    let splats = project_splats(gaussians, width, height, pose, intrinsics);

    let n_pixels = (width * height) as usize;
    let mut rendered = vec![[0.0f32; 3]; n_pixels];
    let mut transmittance = vec![1.0f32; n_pixels];

    for splat in &splats {
        let cov = &splat.cov2d;
        let det = cov[0] * cov[2] - cov[1] * cov[1];
        if det < 1e-6 {
            continue;
        }
        let inv_det = 1.0 / det;
        let inv00 = cov[2] * inv_det;
        let inv01 = -cov[1] * inv_det;
        let inv11 = cov[0] * inv_det;

        let sigma = (cov[0].max(cov[2])).sqrt() * 3.0;
        let x_min = ((splat.px - sigma) as i32).max(0) as u32;
        let x_max = ((splat.px + sigma) as i32 + 1).min(width as i32) as u32;
        let y_min = ((splat.py - sigma) as i32).max(0) as u32;
        let y_max = ((splat.py + sigma) as i32 + 1).min(height as i32) as u32;

        let g = &gaussians[splat.gauss_idx];

        for y in y_min..y_max {
            for x in x_min..x_max {
                let dx = x as f32 - splat.px;
                let dy = y as f32 - splat.py;
                let power = -0.5 * (inv00 * dx * dx + 2.0 * inv01 * dx * dy + inv11 * dy * dy);
                if power < -4.0 {
                    continue;
                }
                let gauss_val = power.exp();
                let alpha = (g.opacity * gauss_val).min(0.999);

                let px_idx = (y * width + x) as usize;
                let t = transmittance[px_idx];
                if t < 1e-4 {
                    continue;
                }

                let contrib = t * alpha;
                rendered[px_idx][0] += contrib * splat.color[0];
                rendered[px_idx][1] += contrib * splat.color[1];
                rendered[px_idx][2] += contrib * splat.color[2];
                transmittance[px_idx] *= 1.0 - alpha;
            }
        }
    }

    (rendered, transmittance)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::{Gaussian3D, Point3};

    fn default_intrinsics() -> CameraIntrinsics {
        CameraIntrinsics {
            fx: 525.0,
            fy: 525.0,
            cx: 320.0,
            cy: 240.0,
        }
    }

    fn make_gaussian_at(x: f32, y: f32, z: f32, opacity: f32, color: [f32; 3]) -> Gaussian3D {
        let mut g = Gaussian3D::new(Point3::new(x, y, z), color, opacity);
        // Convert colour to SH DC representation.
        const C0: f32 = 0.282_094_8;
        g.sh_coefficients = vec![
            (color[0] - 0.5) / C0,
            (color[1] - 0.5) / C0,
            (color[2] - 0.5) / C0,
        ];
        // Small scale: ~2-pixel radius when 1 m in front of camera.
        g.scale = [0.01, 0.01, 0.01];
        let s2 = 0.01_f32 * 0.01;
        g.covariance = [s2, 0.0, 0.0, s2, 0.0, s2];
        g
    }

    // ── Render ────────────────────────────────────────────────────────────────

    #[test]
    fn test_render_empty_gives_black_image() {
        let gaussians: Vec<Gaussian3D> = Vec::new();
        let rendered =
            GaussianOptimizer::render(&gaussians, 64, 64, &Pose::identity(), &default_intrinsics());
        assert!(
            rendered.iter().all(|&b| b == 0),
            "empty scene must render black"
        );
    }

    #[test]
    fn test_render_output_size_correct() {
        let gaussians: Vec<Gaussian3D> = Vec::new();
        let rendered =
            GaussianOptimizer::render(&gaussians, 32, 48, &Pose::identity(), &default_intrinsics());
        assert_eq!(rendered.len(), 32 * 48 * 3);
    }

    #[test]
    fn test_render_single_gaussian_visible() {
        // Place a bright opaque Gaussian directly in front of the camera.
        let g = make_gaussian_at(0.0, 0.0, 2.0, 0.95, [1.0, 0.5, 0.0]);
        let rendered =
            GaussianOptimizer::render(&[g], 640, 480, &Pose::identity(), &default_intrinsics());
        // At least some pixels should be non-zero.
        assert!(
            rendered.iter().any(|&b| b > 0),
            "a Gaussian in front of the camera must produce non-zero pixels"
        );
    }

    #[test]
    fn test_render_gaussian_behind_camera_invisible() {
        // Gaussian at z = -1 (behind the camera) should not appear.
        let g = make_gaussian_at(0.0, 0.0, -1.0, 1.0, [1.0, 1.0, 1.0]);
        let rendered =
            GaussianOptimizer::render(&[g], 64, 64, &Pose::identity(), &default_intrinsics());
        assert!(
            rendered.iter().all(|&b| b == 0),
            "Gaussian behind camera must not appear"
        );
    }

    // ── AdamState ─────────────────────────────────────────────────────────────

    #[test]
    fn test_adam_update_decreases_parameter_for_positive_gradient() {
        let mut adam = AdamState::new(1);
        adam.advance();
        let delta = adam.delta(0, 1.0, 0.01);
        assert!(
            delta > 0.0,
            "positive gradient → positive delta (subtract from param)"
        );
    }

    #[test]
    fn test_adam_update_increases_parameter_for_negative_gradient() {
        let mut adam = AdamState::new(1);
        adam.advance();
        let delta = adam.delta(0, -1.0, 0.01);
        assert!(
            delta < 0.0,
            "negative gradient → negative delta (subtract means increase)"
        );
    }

    #[test]
    fn test_adam_ensure_capacity_grows() {
        let mut adam = AdamState::new(2);
        adam.ensure_capacity(10);
        assert_eq!(adam.m.len(), 10);
    }

    // ── Optimization ─────────────────────────────────────────────────────────

    #[test]
    fn test_optimize_step_returns_zero_for_empty_gaussians() {
        let mut optimizer = GaussianOptimizer::new(OptimizerConfig::default());
        let mut gaussians: Vec<Gaussian3D> = Vec::new();
        let target = vec![0u8; 64 * 64 * 3];
        let loss = optimizer.optimize_step(
            &mut gaussians,
            &target,
            64,
            64,
            &Pose::identity(),
            &default_intrinsics(),
        );
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_optimize_step_loss_is_non_negative() {
        let mut optimizer = GaussianOptimizer::new(OptimizerConfig::default());
        let g = make_gaussian_at(0.0, 0.0, 2.0, 0.8, [0.5, 0.5, 0.5]);
        let mut gaussians = vec![g];
        let target = vec![128u8; 64 * 64 * 3];
        let loss = optimizer.optimize_step(
            &mut gaussians,
            &target,
            64,
            64,
            &Pose::identity(),
            &default_intrinsics(),
        );
        assert!(loss >= 0.0, "MSE loss must be non-negative");
    }

    #[test]
    fn test_optimize_multiple_steps_does_not_panic() {
        let mut optimizer = GaussianOptimizer::new(OptimizerConfig::default());
        let g = make_gaussian_at(0.0, 0.0, 2.0, 0.5, [0.8, 0.4, 0.2]);
        let mut gaussians = vec![g];
        let target = vec![200u8; 32 * 32 * 3];
        for _ in 0..5 {
            optimizer.optimize_step(
                &mut gaussians,
                &target,
                32,
                32,
                &Pose::identity(),
                &default_intrinsics(),
            );
        }
        // Must not panic; opacity clamped to valid range.
        for g in &gaussians {
            assert!(g.opacity >= 0.0 && g.opacity <= 1.0);
        }
    }

    // ── Pruning ───────────────────────────────────────────────────────────────

    #[test]
    fn test_prune_removes_low_opacity() {
        let config = OptimizerConfig {
            min_opacity: 0.1,
            ..Default::default()
        };
        let optimizer = GaussianOptimizer::new(config);
        let mut gaussians = vec![
            make_gaussian_at(0.0, 0.0, 1.0, 0.05, [0.5, 0.5, 0.5]), // below threshold
            make_gaussian_at(1.0, 0.0, 1.0, 0.5, [0.5, 0.5, 0.5]),  // above threshold
        ];
        optimizer.prune_low_opacity(&mut gaussians);
        assert_eq!(gaussians.len(), 1);
    }

    #[test]
    fn test_prune_keeps_all_above_threshold() {
        let optimizer = GaussianOptimizer::new(OptimizerConfig::default()); // min = 0.005
        let mut gaussians = vec![
            make_gaussian_at(0.0, 0.0, 1.0, 0.5, [0.5, 0.5, 0.5]),
            make_gaussian_at(1.0, 0.0, 1.0, 0.9, [0.5, 0.5, 0.5]),
        ];
        optimizer.prune_low_opacity(&mut gaussians);
        assert_eq!(gaussians.len(), 2);
    }
}
