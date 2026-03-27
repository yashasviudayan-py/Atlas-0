//! CPU 3D Gaussian Splatting renderer for display output.
//!
//! [`SplatRenderer`] wraps the core EWA splatting pipeline and produces
//! display-ready image buffers. It is intentionally separate from the
//! photometric [`GaussianOptimizer`] so the viewer can call it without
//! carrying optimizer state.
//!
//! # Display layout
//!
//! [`SplatRenderer::compose_display`] produces a 3-panel side-by-side image:
//!
//! ```text
//! ┌──────────────────┬──────────────────┬──────────────────┐
//! │  Camera frame    │   Depth map      │  Gaussian splat  │
//! └──────────────────┴──────────────────┴──────────────────┘
//! ```

use atlas_core::{Gaussian3D, Pose};

use crate::config::CameraIntrinsics;
use crate::optimizer::GaussianOptimizer;

// ─── SplatRenderer ────────────────────────────────────────────────────────────

/// Display-oriented 3DGS renderer.
///
/// Wraps [`GaussianOptimizer::render`] and provides helpers for converting the
/// output to formats suitable for on-screen display.
///
/// # Example
///
/// ```
/// # use atlas_slam::renderer::SplatRenderer;
/// # use atlas_slam::config::CameraIntrinsics;
/// # use atlas_core::Pose;
/// let renderer = SplatRenderer::new(CameraIntrinsics::default(), 640, 480);
/// let rgb = renderer.render_rgb(&[], &Pose::identity());
/// assert_eq!(rgb.len(), 640 * 480 * 3);
/// ```
pub struct SplatRenderer {
    intrinsics: CameraIntrinsics,
    /// Rendering width in pixels.
    pub width: u32,
    /// Rendering height in pixels.
    pub height: u32,
}

impl SplatRenderer {
    /// Create a new renderer for the given camera and resolution.
    #[must_use]
    pub fn new(intrinsics: CameraIntrinsics, width: u32, height: u32) -> Self {
        Self {
            intrinsics,
            width,
            height,
        }
    }

    /// Render `gaussians` as seen from `pose`, returning a row-major RGB byte
    /// buffer of size `width × height × 3`.
    ///
    /// Gaussians are projected, depth-sorted, and alpha-composited using the
    /// EWA splatting formula.
    #[must_use]
    pub fn render_rgb(&self, gaussians: &[Gaussian3D], pose: &Pose) -> Vec<u8> {
        GaussianOptimizer::render(gaussians, self.width, self.height, pose, &self.intrinsics)
    }

    /// Render to a BGRA `u32` buffer suitable for `minifb`.
    ///
    /// Each `u32` is packed as `0x00_RR_GG_BB`.
    #[must_use]
    pub fn render_bgra(&self, gaussians: &[Gaussian3D], pose: &Pose) -> Vec<u32> {
        rgb_to_bgra(&self.render_rgb(gaussians, pose))
    }

    /// Compose a 3-panel BGRA buffer: camera | depth | splat.
    ///
    /// The output is `3 × width` wide and `height` tall (BGRA `u32`).
    ///
    /// - `camera_rgb`: raw camera frame (`width × height × 3` bytes, row-major RGB)
    /// - `depth_rgb`:  visualised depth map (same dimensions, RGB)
    /// - `gaussians`:  current Gaussian map
    /// - `pose`:       current camera pose
    #[must_use]
    pub fn compose_display(
        &self,
        camera_rgb: &[u8],
        depth_rgb: &[u8],
        gaussians: &[Gaussian3D],
        pose: &Pose,
    ) -> Vec<u32> {
        let splat_rgb = self.render_rgb(gaussians, pose);
        compose_panels_bgra(
            &[camera_rgb, depth_rgb, &splat_rgb],
            self.width,
            self.height,
        )
    }
}

// ─── Public helpers ───────────────────────────────────────────────────────────

/// Convert a row-major RGB byte slice to a BGRA `u32` buffer for `minifb`.
///
/// Each output `u32` is packed as `0x00_RR_GG_BB`.
///
/// # Example
///
/// ```
/// # use atlas_slam::renderer::rgb_to_bgra;
/// let bgra = rgb_to_bgra(&[255, 128, 64]);
/// assert_eq!(bgra[0], 0x00_FF_80_40);
/// ```
#[must_use]
pub fn rgb_to_bgra(rgb: &[u8]) -> Vec<u32> {
    rgb.chunks_exact(3)
        .map(|c| ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | (c[2] as u32))
        .collect()
}

/// Convert a grayscale `u8` slice to a row-major RGB byte slice.
///
/// Each grey value is replicated to R, G, and B channels.
///
/// # Example
///
/// ```
/// # use atlas_slam::renderer::gray_to_rgb;
/// assert_eq!(gray_to_rgb(&[100, 200]), vec![100,100,100, 200,200,200]);
/// ```
#[must_use]
pub fn gray_to_rgb(gray: &[u8]) -> Vec<u8> {
    gray.iter().flat_map(|&v| [v, v, v]).collect()
}

/// Compose equally-sized RGB panels into a horizontal strip (BGRA `u32`).
///
/// Panels are placed left-to-right.  Each panel must contain
/// `width × height × 3` bytes (row-major RGB).  Undersized panels are filled
/// with black for missing pixels.
///
/// # Example
///
/// ```
/// # use atlas_slam::renderer::compose_panels_bgra;
/// // Two 1×1 panels: red and green.
/// let red   = [255u8, 0, 0];
/// let green = [0u8, 255, 0];
/// let out = compose_panels_bgra(&[&red, &green], 1, 1);
/// assert_eq!(out.len(), 2);
/// assert_eq!(out[0], 0x00_FF_00_00);
/// assert_eq!(out[1], 0x00_00_FF_00);
/// ```
#[must_use]
pub fn compose_panels_bgra(panels: &[&[u8]], width: u32, height: u32) -> Vec<u32> {
    let n = panels.len() as u32;
    let total_w = width * n;
    let mut out = vec![0u32; (total_w * height) as usize];

    for (pi, panel) in panels.iter().enumerate() {
        let x_off = pi as u32 * width;
        for y in 0..height {
            for x in 0..width {
                let src = (y * width + x) as usize * 3;
                let dst = (y * total_w + x_off + x) as usize;
                if let Some(px) = panel.get(src..src + 3) {
                    out[dst] = ((px[0] as u32) << 16) | ((px[1] as u32) << 8) | (px[2] as u32);
                }
            }
        }
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::{Gaussian3D, Point3};

    fn make_renderer(w: u32, h: u32) -> SplatRenderer {
        SplatRenderer::new(CameraIntrinsics::default(), w, h)
    }

    #[test]
    fn test_rgb_to_bgra_encoding() {
        let rgb = [255u8, 128, 64, 0, 255, 0];
        let bgra = rgb_to_bgra(&rgb);
        assert_eq!(bgra.len(), 2);
        assert_eq!(bgra[0], 0x00_FF_80_40);
        assert_eq!(bgra[1], 0x00_00_FF_00);
    }

    #[test]
    fn test_gray_to_rgb_expands_channels() {
        let rgb = gray_to_rgb(&[100u8, 200]);
        assert_eq!(rgb, vec![100, 100, 100, 200, 200, 200]);
    }

    #[test]
    fn test_compose_panels_output_width() {
        let w = 4u32;
        let h = 2u32;
        let panel = vec![0u8; (w * h * 3) as usize];
        let out = compose_panels_bgra(&[&panel, &panel, &panel], w, h);
        assert_eq!(out.len(), (3 * w * h) as usize);
    }

    #[test]
    fn test_compose_two_panels_colors() {
        let red = vec![255u8, 0, 0];
        let green = vec![0u8, 255, 0];
        let out = compose_panels_bgra(&[&red, &green], 1, 1);
        assert_eq!(out[0], 0x00_FF_00_00, "first panel should be red");
        assert_eq!(out[1], 0x00_00_FF_00, "second panel should be green");
    }

    #[test]
    fn test_render_rgb_empty_is_black() {
        let r = make_renderer(16, 16);
        let rgb = r.render_rgb(&[], &atlas_core::Pose::identity());
        assert_eq!(rgb.len(), 16 * 16 * 3);
        assert!(rgb.iter().all(|&b| b == 0), "empty Gaussian map → black");
    }

    #[test]
    fn test_render_bgra_correct_size() {
        let r = make_renderer(32, 24);
        let bgra = r.render_bgra(&[], &atlas_core::Pose::identity());
        assert_eq!(bgra.len(), 32 * 24);
    }

    #[test]
    fn test_render_rgb_single_gaussian_colors_center() {
        // Place a red Gaussian directly in front of the camera.
        let intrinsics = CameraIntrinsics {
            fx: 525.0,
            fy: 525.0,
            cx: 32.0,
            cy: 24.0,
        };
        let r = SplatRenderer::new(intrinsics, 64, 48);

        const C0: f32 = 0.282_094_8;
        let color = [1.0f32, 0.0, 0.0];
        let mut g = Gaussian3D::new(Point3::new(0.0, 0.0, 2.0), color, 0.9);
        g.sh_coefficients = vec![
            (color[0] - 0.5) / C0,
            (color[1] - 0.5) / C0,
            (color[2] - 0.5) / C0,
        ];
        let s2 = 0.05_f32 * 0.05;
        g.covariance = [s2, 0.0, 0.0, s2, 0.0, s2];
        g.scale = [0.05, 0.05, 0.05];

        let rgb = r.render_rgb(&[g], &atlas_core::Pose::identity());
        let idx = (24 * 64 + 32) * 3; // pixel at (cx=32, cy=24)
        assert!(rgb[idx] > 0, "center pixel red channel must be nonzero");
    }

    #[test]
    fn test_compose_display_output_width() {
        let r = make_renderer(16, 16);
        let panel = vec![0u8; 16 * 16 * 3];
        let out = r.compose_display(&panel, &panel, &[], &atlas_core::Pose::identity());
        assert_eq!(out.len(), 3 * 16 * 16);
    }
}
