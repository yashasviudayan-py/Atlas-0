//! 3D Gaussian primitives for Gaussian Splatting.

use crate::spatial::Point3;
use serde::{Deserialize, Serialize};

// ─── Bounding box ─────────────────────────────────────────────────────────────

/// An axis-aligned bounding box in 3D space.
///
/// Used for spatial queries against a [`GaussianCloud`].
///
/// # Example
/// ```
/// # use atlas_core::gaussian::BoundingBox3D;
/// # use atlas_core::Point3;
/// let bbox = BoundingBox3D::new(
///     Point3::new(-1.0, -1.0, -1.0),
///     Point3::new( 1.0,  1.0,  1.0),
/// );
/// assert!(bbox.contains(&Point3::new(0.0, 0.0, 0.0)));
/// assert!(!bbox.contains(&Point3::new(2.0, 0.0, 0.0)));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox3D {
    /// Minimum corner (smallest x, y, z).
    pub min: Point3,
    /// Maximum corner (largest x, y, z).
    pub max: Point3,
}

impl BoundingBox3D {
    /// Create a new bounding box from its minimum and maximum corners.
    #[must_use]
    pub fn new(min: Point3, max: Point3) -> Self {
        Self { min, max }
    }

    /// Return `true` if point `p` lies inside or on the boundary of this box.
    #[must_use]
    pub fn contains(&self, p: &Point3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }
}

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

    // ── Spatial queries ───────────────────────────────────────────────────────

    /// Return the indices of all Gaussians whose centres lie inside `bbox`.
    ///
    /// Uses a linear scan; suitable for sparse maps. For dense maps (> 1M
    /// Gaussians) consider adding an octree index.
    #[must_use]
    pub fn query_region(&self, bbox: &BoundingBox3D) -> Vec<usize> {
        self.gaussians
            .iter()
            .enumerate()
            .filter_map(|(i, g)| bbox.contains(&g.center).then_some(i))
            .collect()
    }

    /// Remove all Gaussians whose opacity is below `min_opacity`.
    ///
    /// This is the primary densification/pruning mechanism: Gaussians that the
    /// optimizer has driven toward transparent get culled to keep the map lean.
    pub fn prune(&mut self, min_opacity: f32) {
        self.gaussians.retain(|g| g.opacity >= min_opacity);
    }

    /// Merge Gaussians whose centres are within `threshold` metres of each
    /// other.
    ///
    /// When two Gaussians are close enough they are collapsed into one:
    /// the survivor keeps the position of the *more opaque* one and its
    /// opacity is set to the maximum of the pair.  The redundant Gaussian is
    /// removed.  This prevents the map from accumulating duplicate splats in
    /// static regions.
    pub fn merge_near(&mut self, threshold: f32) {
        let threshold_sq = threshold * threshold;
        let n = self.gaussians.len();
        // Mark the lower-opacity duplicate in each close pair for removal.
        let mut remove = vec![false; n];

        for i in 0..n {
            if remove[i] {
                continue;
            }
            for j in (i + 1)..n {
                if remove[j] {
                    continue;
                }
                let gi = &self.gaussians[i];
                let gj = &self.gaussians[j];
                let dx = gi.center.x - gj.center.x;
                let dy = gi.center.y - gj.center.y;
                let dz = gi.center.z - gj.center.z;
                if dx * dx + dy * dy + dz * dz < threshold_sq {
                    // Keep the more opaque one; absorb the other.
                    if gi.opacity >= gj.opacity {
                        let max_opacity = gi.opacity.max(gj.opacity);
                        self.gaussians[i].opacity = max_opacity;
                        remove[j] = true;
                    } else {
                        let max_opacity = gi.opacity.max(gj.opacity);
                        self.gaussians[j].opacity = max_opacity;
                        remove[i] = true;
                        break; // i is gone — stop looking for j partners
                    }
                }
            }
        }

        let mut idx = 0usize;
        self.gaussians.retain(|_| {
            let keep = !remove[idx];
            idx += 1;
            keep
        });
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point3;

    fn make_gaussian(x: f32, y: f32, z: f32, opacity: f32) -> Gaussian3D {
        Gaussian3D::new(Point3::new(x, y, z), [0.5, 0.5, 0.5], opacity)
    }

    // ── BoundingBox3D ─────────────────────────────────────────────────────────

    #[test]
    fn test_bbox_contains_interior() {
        let bbox = BoundingBox3D::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        assert!(bbox.contains(&Point3::new(0.0, 0.0, 0.0)));
    }

    #[test]
    fn test_bbox_contains_on_boundary() {
        let bbox = BoundingBox3D::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        assert!(bbox.contains(&Point3::new(0.0, 0.5, 1.0)));
    }

    #[test]
    fn test_bbox_excludes_outside() {
        let bbox = BoundingBox3D::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        assert!(!bbox.contains(&Point3::new(2.0, 0.0, 0.0)));
        assert!(!bbox.contains(&Point3::new(0.0, -2.0, 0.0)));
    }

    // ── query_region ─────────────────────────────────────────────────────────

    #[test]
    fn test_query_region_all_inside() {
        let mut cloud = GaussianCloud::new();
        cloud.add(make_gaussian(0.0, 0.0, 0.0, 0.8));
        cloud.add(make_gaussian(0.5, 0.5, 0.5, 0.6));
        let bbox = BoundingBox3D::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let hits = cloud.query_region(&bbox);
        assert_eq!(hits, vec![0, 1]);
    }

    #[test]
    fn test_query_region_partial_overlap() {
        let mut cloud = GaussianCloud::new();
        cloud.add(make_gaussian(0.0, 0.0, 0.0, 0.8));
        cloud.add(make_gaussian(5.0, 5.0, 5.0, 0.6));
        let bbox = BoundingBox3D::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let hits = cloud.query_region(&bbox);
        assert_eq!(hits, vec![0]);
    }

    #[test]
    fn test_query_region_empty_cloud() {
        let cloud = GaussianCloud::new();
        let bbox = BoundingBox3D::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        assert!(cloud.query_region(&bbox).is_empty());
    }

    // ── prune ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_prune_removes_low_opacity() {
        let mut cloud = GaussianCloud::new();
        cloud.add(make_gaussian(0.0, 0.0, 0.0, 0.001)); // below threshold
        cloud.add(make_gaussian(1.0, 0.0, 0.0, 0.5)); // above
        cloud.add(make_gaussian(2.0, 0.0, 0.0, 0.01)); // exactly on threshold
        cloud.prune(0.01);
        // 0.001 is removed; 0.5 and 0.01 stay.
        assert_eq!(cloud.len(), 2);
    }

    #[test]
    fn test_prune_all_removed() {
        let mut cloud = GaussianCloud::new();
        cloud.add(make_gaussian(0.0, 0.0, 0.0, 0.001));
        cloud.prune(0.01);
        assert!(cloud.is_empty());
    }

    #[test]
    fn test_prune_none_removed() {
        let mut cloud = GaussianCloud::new();
        cloud.add(make_gaussian(0.0, 0.0, 0.0, 0.5));
        cloud.add(make_gaussian(1.0, 0.0, 0.0, 1.0));
        cloud.prune(0.1);
        assert_eq!(cloud.len(), 2);
    }

    // ── merge_near ────────────────────────────────────────────────────────────

    #[test]
    fn test_merge_near_removes_duplicate() {
        let mut cloud = GaussianCloud::new();
        cloud.add(make_gaussian(0.0, 0.0, 0.0, 0.8));
        // Very close to the first — should be merged.
        cloud.add(make_gaussian(0.001, 0.0, 0.0, 0.6));
        cloud.add(make_gaussian(5.0, 0.0, 0.0, 0.5)); // far away
        cloud.merge_near(0.1);
        assert_eq!(cloud.len(), 2, "close pair collapses to 1; far one remains");
    }

    #[test]
    fn test_merge_near_keeps_higher_opacity() {
        let mut cloud = GaussianCloud::new();
        cloud.add(make_gaussian(0.0, 0.0, 0.0, 0.4));
        cloud.add(make_gaussian(0.0, 0.0, 0.001, 0.9)); // closer, higher opacity
        cloud.merge_near(0.1);
        assert_eq!(cloud.len(), 1);
        assert!((cloud.gaussians[0].opacity - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_merge_near_nothing_to_merge() {
        let mut cloud = GaussianCloud::new();
        cloud.add(make_gaussian(0.0, 0.0, 0.0, 0.8));
        cloud.add(make_gaussian(1.0, 0.0, 0.0, 0.6));
        cloud.add(make_gaussian(0.0, 1.0, 0.0, 0.5));
        cloud.merge_near(0.1);
        assert_eq!(cloud.len(), 3);
    }
}
