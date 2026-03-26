//! Gaussian initialization from sparse 3-D point clouds.
//!
//! After triangulation, each 3-D point becomes the seed for a new
//! [`Gaussian3D`].  [`GaussianInitializer`] sets the initial size (scale),
//! colour (SH DC coefficient), and opacity for every point so that the map
//! can be refined by the downstream optimizer.

use atlas_core::{Gaussian3D, Point3};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Tuning parameters for Gaussian initialisation.
#[derive(Debug, Clone)]
pub struct GaussianInitConfig {
    /// Starting opacity for every new Gaussian (range 0–1).
    pub initial_opacity: f32,
    /// Floor on the Gaussian scale (metres).  Prevents degenerate splats.
    pub min_scale: f32,
    /// Ceiling on the Gaussian scale (metres).  Prevents sky-size splats.
    pub max_scale: f32,
}

impl Default for GaussianInitConfig {
    fn default() -> Self {
        Self {
            initial_opacity: 0.5,
            min_scale: 0.001,
            max_scale: 1.0,
        }
    }
}

// ─── GaussianInitializer ──────────────────────────────────────────────────────

/// Converts a sparse 3-D point cloud into a set of [`Gaussian3D`] splats.
pub struct GaussianInitializer {
    config: GaussianInitConfig,
}

impl GaussianInitializer {
    /// Create a new initialiser with the given configuration.
    #[must_use]
    pub fn new(config: GaussianInitConfig) -> Self {
        Self { config }
    }

    /// Create one [`Gaussian3D`] per point in `points`.
    ///
    /// `points` is a slice of `(position, RGB colour)` pairs where colour
    /// channels are in the range \[0, 1\].
    ///
    /// For each point:
    /// - **Position**: set to `position`.
    /// - **Scale**: estimated as half the distance to the nearest neighbour
    ///   (clamped to `[min_scale, max_scale]`).
    /// - **Colour (SH DC)**: converted from linear RGB using the degree-0
    ///   spherical-harmonics basis function
    ///   `sh = (rgb − 0.5) / C₀`  where `C₀ = 1/(2√π) ≈ 0.282`.
    /// - **Covariance**: diagonal `diag(s², s², s²)` from the computed scale.
    /// - **Opacity**: `config.initial_opacity`.
    #[must_use]
    pub fn from_point_cloud(&self, points: &[(Point3, [f32; 3])]) -> Vec<Gaussian3D> {
        if points.is_empty() {
            return Vec::new();
        }

        // Precompute squared positions for nearest-neighbour search.
        let mut gaussians = Vec::with_capacity(points.len());

        for (i, (pos, color)) in points.iter().enumerate() {
            let scale = self.nearest_neighbour_scale(pos, points, i);
            let s2 = scale * scale;

            // Degree-0 SH coefficient: sh = (color − 0.5) / C₀
            // C₀ = 1 / (2√π)
            const C0: f32 = 0.282_094_8;
            let sh = [
                (color[0] - 0.5) / C0,
                (color[1] - 0.5) / C0,
                (color[2] - 0.5) / C0,
            ];

            let mut g = Gaussian3D::new(*pos, *color, self.config.initial_opacity);
            g.sh_coefficients = sh.to_vec();
            g.scale = [scale, scale, scale];
            // Diagonal covariance Σ = diag(s², s², s²).
            g.covariance = [s2, 0.0, 0.0, s2, 0.0, s2];

            gaussians.push(g);
        }

        gaussians
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Compute the initial Gaussian scale for the point at index `skip_idx`.
    ///
    /// The scale is set to half the distance to the nearest *other* point,
    /// clamped to `[min_scale, max_scale]`.  For an isolated point the
    /// default scale is `min_scale`.
    fn nearest_neighbour_scale(
        &self,
        pos: &Point3,
        all_points: &[(Point3, [f32; 3])],
        skip_idx: usize,
    ) -> f32 {
        let mut min_dist_sq = f32::MAX;

        for (i, (other, _)) in all_points.iter().enumerate() {
            if i == skip_idx {
                continue;
            }
            let dx = pos.x - other.x;
            let dy = pos.y - other.y;
            let dz = pos.z - other.z;
            let dsq = dx * dx + dy * dy + dz * dz;
            if dsq < min_dist_sq {
                min_dist_sq = dsq;
            }
        }

        if min_dist_sq == f32::MAX {
            return self.config.min_scale;
        }

        (min_dist_sq.sqrt() * 0.5)
            .max(self.config.min_scale)
            .min(self.config.max_scale)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_point(x: f32, y: f32, z: f32) -> (Point3, [f32; 3]) {
        (Point3::new(x, y, z), [0.8, 0.4, 0.2])
    }

    // ── from_point_cloud ─────────────────────────────────────────────────────

    #[test]
    fn test_empty_point_cloud_returns_empty() {
        let init = GaussianInitializer::new(GaussianInitConfig::default());
        let result = init.from_point_cloud(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_point_uses_min_scale() {
        let config = GaussianInitConfig {
            min_scale: 0.05,
            ..Default::default()
        };
        let init = GaussianInitializer::new(config.clone());
        let points = [make_point(0.0, 0.0, 0.0)];
        let result = init.from_point_cloud(&points);
        assert_eq!(result.len(), 1);
        // Single point has no neighbour → falls back to min_scale.
        assert!((result[0].scale[0] - config.min_scale).abs() < 1e-6);
    }

    #[test]
    fn test_two_points_scale_is_half_distance() {
        let init = GaussianInitializer::new(GaussianInitConfig::default());
        let points = [make_point(0.0, 0.0, 0.0), make_point(1.0, 0.0, 0.0)];
        let result = init.from_point_cloud(&points);
        // Distance = 1.0 m, so scale = 0.5 m.
        assert!((result[0].scale[0] - 0.5).abs() < 1e-5);
        assert!((result[1].scale[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_scale_clamped_to_max() {
        let config = GaussianInitConfig {
            max_scale: 0.1,
            ..Default::default()
        };
        let init = GaussianInitializer::new(config.clone());
        // 100 m apart → unclamped scale = 50 m, but max is 0.1 m.
        let points = [make_point(0.0, 0.0, 0.0), make_point(100.0, 0.0, 0.0)];
        let result = init.from_point_cloud(&points);
        assert!((result[0].scale[0] - config.max_scale).abs() < 1e-6);
    }

    #[test]
    fn test_scale_clamped_to_min() {
        let config = GaussianInitConfig {
            min_scale: 0.1,
            ..Default::default()
        };
        let init = GaussianInitializer::new(config.clone());
        // Very close points → unclamped scale ≈ 0, clamped to min_scale.
        let points = [make_point(0.0, 0.0, 0.0), make_point(0.0001, 0.0, 0.0)];
        let result = init.from_point_cloud(&points);
        assert!(result[0].scale[0] >= config.min_scale - 1e-6);
    }

    #[test]
    fn test_opacity_set_correctly() {
        let config = GaussianInitConfig {
            initial_opacity: 0.3,
            ..Default::default()
        };
        let init = GaussianInitializer::new(config);
        let points = [make_point(0.0, 0.0, 1.0), make_point(1.0, 0.0, 1.0)];
        let result = init.from_point_cloud(&points);
        for g in &result {
            assert!((g.opacity - 0.3).abs() < 1e-6);
        }
    }

    #[test]
    fn test_covariance_is_diagonal_scale_squared() {
        let init = GaussianInitializer::new(GaussianInitConfig::default());
        let points = [make_point(0.0, 0.0, 0.0), make_point(0.4, 0.0, 0.0)];
        let result = init.from_point_cloud(&points);
        // scale = 0.2, so s² = 0.04.
        let s = result[0].scale[0];
        let s2 = s * s;
        assert!((result[0].covariance[0] - s2).abs() < 1e-6, "σ_xx = s²");
        assert!((result[0].covariance[3] - s2).abs() < 1e-6, "σ_yy = s²");
        assert!((result[0].covariance[5] - s2).abs() < 1e-6, "σ_zz = s²");
        // Off-diagonal entries must be zero.
        assert_eq!(result[0].covariance[1], 0.0, "σ_xy = 0");
        assert_eq!(result[0].covariance[2], 0.0, "σ_xz = 0");
        assert_eq!(result[0].covariance[4], 0.0, "σ_yz = 0");
    }

    #[test]
    fn test_sh_coefficients_have_three_components() {
        let init = GaussianInitializer::new(GaussianInitConfig::default());
        let points = [make_point(0.0, 0.0, 0.0), make_point(1.0, 0.0, 0.0)];
        let result = init.from_point_cloud(&points);
        for g in &result {
            assert_eq!(g.sh_coefficients.len(), 3);
        }
    }

    #[test]
    fn test_sh_to_color_round_trip() {
        // Verify that SH → colour conversion recovers the original [r, g, b].
        const C0: f32 = 0.282_094_8;
        let color = [0.8_f32, 0.4, 0.2];
        let init = GaussianInitializer::new(GaussianInitConfig::default());
        let points = [
            (Point3::new(0.0, 0.0, 0.0), color),
            (Point3::new(1.0, 0.0, 0.0), color),
        ];
        let result = init.from_point_cloud(&points);
        let sh = &result[0].sh_coefficients;
        let recovered = [
            (C0 * sh[0] + 0.5).clamp(0.0, 1.0),
            (C0 * sh[1] + 0.5).clamp(0.0, 1.0),
            (C0 * sh[2] + 0.5).clamp(0.0, 1.0),
        ];
        for i in 0..3 {
            assert!(
                (recovered[i] - color[i]).abs() < 1e-5,
                "channel {i}: expected {}, got {}",
                color[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_position_is_preserved() {
        let init = GaussianInitializer::new(GaussianInitConfig::default());
        let pos = Point3::new(1.5, 2.5, 3.5);
        let points = [(pos, [0.5, 0.5, 0.5])];
        let result = init.from_point_cloud(&points);
        assert!((result[0].center.x - 1.5).abs() < 1e-6);
        assert!((result[0].center.y - 2.5).abs() < 1e-6);
        assert!((result[0].center.z - 3.5).abs() < 1e-6);
    }
}
