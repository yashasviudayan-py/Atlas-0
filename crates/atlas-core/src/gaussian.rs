//! 3D Gaussian primitives for Gaussian Splatting.

use crate::spatial::Point3;
use serde::{Deserialize, Serialize};

/// A single 3D Gaussian used in the splatting representation.
///
/// Each Gaussian encodes position, shape (covariance), color, and opacity.
/// The collection of all Gaussians forms the 3D scene representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gaussian3D {
    /// Center position in world coordinates.
    pub center: Point3,
    /// Covariance matrix (stored as upper triangle: [σ_xx, σ_xy, σ_xz, σ_yy, σ_yz, σ_zz]).
    pub covariance: [f32; 6],
    /// Spherical harmonics coefficients for view-dependent color (degree 0 = RGB).
    pub sh_coefficients: Vec<f32>,
    /// Opacity (0.0 = transparent, 1.0 = opaque).
    pub opacity: f32,
    /// Scale factors for the three axes.
    pub scale: [f32; 3],
    /// Rotation as quaternion (w, x, y, z).
    pub rotation: [f32; 4],
}

impl Gaussian3D {
    /// Create a new Gaussian with default spherical shape.
    #[must_use]
    pub fn new(center: Point3, color_rgb: [f32; 3], opacity: f32) -> Self {
        Self {
            center,
            covariance: [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            sh_coefficients: color_rgb.to_vec(),
            opacity,
            scale: [1.0, 1.0, 1.0],
            rotation: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

/// A collection of 3D Gaussians representing a scene or scene fragment.
#[derive(Debug, Clone, Default)]
pub struct GaussianCloud {
    pub gaussians: Vec<Gaussian3D>,
}

impl GaussianCloud {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, gaussian: Gaussian3D) {
        self.gaussians.push(gaussian);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.gaussians.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.gaussians.is_empty()
    }
}
