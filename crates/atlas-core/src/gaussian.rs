//! 3D Gaussian primitives for Gaussian Splatting.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::spatial::Point3;

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

// ─── GaussianCloud ────────────────────────────────────────────────────────────

/// A collection of 3D Gaussians representing a scene or scene fragment.
///
/// Spatial queries use a uniform-grid index that is rebuilt lazily whenever
/// the Gaussian list is mutated.  The grid (and dirty flag) are excluded from
/// serialisation — they are reconstructed on the first query after loading.
///
/// # Persistence
///
/// ```no_run
/// # use atlas_core::gaussian::GaussianCloud;
/// # use std::path::Path;
/// let mut cloud = GaussianCloud::new();
/// cloud.save(Path::new("/tmp/map.json")).unwrap();
/// let _loaded = GaussianCloud::load(Path::new("/tmp/map.json")).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianCloud {
    pub gaussians: Vec<Gaussian3D>,

    // ── Spatial index (not serialised) ────────────────────────────────────────
    /// Side length of each grid cell in metres.  Smaller = finer, but more
    /// memory.  Default: 0.5 m.
    #[serde(default = "default_cell_size")]
    grid_cell_size: f32,
    /// Uniform-grid spatial index: cell key → list of Gaussian indices.
    #[serde(skip)]
    grid: HashMap<(i32, i32, i32), Vec<usize>>,
    /// Set to `true` whenever `gaussians` is mutated so the next
    /// `query_region` call triggers a full index rebuild.
    #[serde(skip)]
    grid_dirty: bool,
}

fn default_cell_size() -> f32 {
    0.5
}

impl Default for GaussianCloud {
    fn default() -> Self {
        Self {
            gaussians: Vec::new(),
            grid_cell_size: default_cell_size(),
            grid: HashMap::new(),
            grid_dirty: false,
        }
    }
}

impl GaussianCloud {
    /// Create an empty cloud with the default grid cell size (0.5 m).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a single Gaussian and mark the spatial index as dirty.
    pub fn add(&mut self, gaussian: Gaussian3D) {
        self.gaussians.push(gaussian);
        self.grid_dirty = true;
    }

    /// Number of Gaussians in this cloud.
    #[must_use]
    pub fn len(&self) -> usize {
        self.gaussians.len()
    }

    /// Return `true` when the cloud contains no Gaussians.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.gaussians.is_empty()
    }

    // ── Spatial queries ───────────────────────────────────────────────────────

    /// Return the indices of all Gaussians whose centres lie inside `bbox`.
    ///
    /// Uses a uniform hash-grid index for sub-linear average-case performance.
    /// The index is rebuilt lazily on the first call after any mutation.
    pub fn query_region(&mut self, bbox: &BoundingBox3D) -> Vec<usize> {
        if self.grid_dirty || (self.grid.is_empty() && !self.gaussians.is_empty()) {
            self.rebuild_grid();
        }

        let min_key = Self::cell_key(&bbox.min, self.grid_cell_size);
        let max_key = Self::cell_key(&bbox.max, self.grid_cell_size);

        let mut result = Vec::new();
        for ix in min_key.0..=max_key.0 {
            for iy in min_key.1..=max_key.1 {
                for iz in min_key.2..=max_key.2 {
                    if let Some(indices) = self.grid.get(&(ix, iy, iz)) {
                        for &idx in indices {
                            if bbox.contains(&self.gaussians[idx].center) {
                                result.push(idx);
                            }
                        }
                    }
                }
            }
        }
        result
    }

    /// Remove all Gaussians whose opacity is below `min_opacity`.
    ///
    /// This is the primary densification/pruning mechanism: Gaussians that the
    /// optimizer has driven toward transparent get culled to keep the map lean.
    pub fn prune(&mut self, min_opacity: f32) {
        self.gaussians.retain(|g| g.opacity >= min_opacity);
        self.grid_dirty = true;
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
                    if gi.opacity >= gj.opacity {
                        let max_opacity = gi.opacity.max(gj.opacity);
                        self.gaussians[i].opacity = max_opacity;
                        remove[j] = true;
                    } else {
                        let max_opacity = gi.opacity.max(gj.opacity);
                        self.gaussians[j].opacity = max_opacity;
                        remove[i] = true;
                        break;
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
        self.grid_dirty = true;
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    /// Serialise the map to `path` as a JSON file.
    ///
    /// The spatial index is not stored — it is rebuilt lazily after loading.
    ///
    /// # Errors
    ///
    /// Returns [`AtlasError::Io`] on file-system errors or
    /// [`AtlasError::Serialization`] on encoding failures.
    ///
    /// [`AtlasError::Io`]: crate::error::AtlasError::Io
    /// [`AtlasError::Serialization`]: crate::error::AtlasError::Serialization
    pub fn save(&self, path: &Path) -> crate::Result<()> {
        use std::io::BufWriter;
        let file = std::fs::File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)
            .map_err(|e| crate::error::AtlasError::Serialization(e.to_string()))
    }

    /// Load a map previously saved with [`save`].
    ///
    /// The spatial index starts dirty and is rebuilt on the first
    /// [`query_region`] call.
    ///
    /// # Errors
    ///
    /// Returns [`AtlasError::Io`] on file-system errors or
    /// [`AtlasError::Serialization`] on decoding failures.
    ///
    /// [`query_region`]: GaussianCloud::query_region
    /// [`AtlasError::Io`]: crate::error::AtlasError::Io
    /// [`AtlasError::Serialization`]: crate::error::AtlasError::Serialization
    pub fn load(path: &Path) -> crate::Result<Self> {
        use std::io::BufReader;
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        let mut cloud: Self = serde_json::from_reader(reader)
            .map_err(|e| crate::error::AtlasError::Serialization(e.to_string()))?;
        // Mark dirty so the first query_region rebuilds the index.
        cloud.grid_dirty = true;
        Ok(cloud)
    }

    // ── Grid index helpers ────────────────────────────────────────────────────

    /// Rebuild the uniform-grid spatial index from scratch.
    fn rebuild_grid(&mut self) {
        self.grid.clear();
        for (idx, g) in self.gaussians.iter().enumerate() {
            let key = Self::cell_key(&g.center, self.grid_cell_size);
            self.grid.entry(key).or_default().push(idx);
        }
        self.grid_dirty = false;
    }

    /// Map a world-space point to its grid cell key.
    fn cell_key(p: &Point3, cell_size: f32) -> (i32, i32, i32) {
        (
            (p.x / cell_size).floor() as i32,
            (p.y / cell_size).floor() as i32,
            (p.z / cell_size).floor() as i32,
        )
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
        let mut hits = cloud.query_region(&bbox);
        hits.sort_unstable();
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
        let mut cloud = GaussianCloud::new();
        let bbox = BoundingBox3D::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        assert!(cloud.query_region(&bbox).is_empty());
    }

    // ── prune ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_prune_removes_low_opacity() {
        let mut cloud = GaussianCloud::new();
        cloud.add(make_gaussian(0.0, 0.0, 0.0, 0.001));
        cloud.add(make_gaussian(1.0, 0.0, 0.0, 0.5));
        cloud.add(make_gaussian(2.0, 0.0, 0.0, 0.01));
        cloud.prune(0.01);
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
        cloud.add(make_gaussian(0.001, 0.0, 0.0, 0.6));
        cloud.add(make_gaussian(5.0, 0.0, 0.0, 0.5));
        cloud.merge_near(0.1);
        assert_eq!(cloud.len(), 2, "close pair collapses to 1; far one remains");
    }

    #[test]
    fn test_merge_near_keeps_higher_opacity() {
        let mut cloud = GaussianCloud::new();
        cloud.add(make_gaussian(0.0, 0.0, 0.0, 0.4));
        cloud.add(make_gaussian(0.0, 0.0, 0.001, 0.9));
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

    // ── save / load ───────────────────────────────────────────────────────────

    #[test]
    fn test_save_and_load_round_trip() {
        let mut cloud = GaussianCloud::new();
        cloud.add(make_gaussian(1.0, 2.0, 3.0, 0.7));
        cloud.add(make_gaussian(-1.0, 0.0, 0.5, 0.3));

        let dir = std::env::temp_dir();
        let path = dir.join("atlas_test_cloud.json");
        cloud.save(&path).expect("save failed");

        let loaded = GaussianCloud::load(&path).expect("load failed");
        assert_eq!(loaded.len(), cloud.len());
        assert!((loaded.gaussians[0].center.x - 1.0).abs() < 1e-6);
        assert!((loaded.gaussians[1].opacity - 0.3).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_missing_file_returns_error() {
        let result = GaussianCloud::load(Path::new("/nonexistent/atlas_test.json"));
        assert!(result.is_err());
    }

    // ── grid index ────────────────────────────────────────────────────────────

    #[test]
    fn test_query_region_uses_grid_after_rebuild() {
        let mut cloud = GaussianCloud::new();
        for i in 0..20 {
            cloud.add(make_gaussian(i as f32 * 2.0, 0.0, 0.0, 0.5));
        }
        // Query a narrow region — only x in [0, 1] should match (index 0).
        let bbox = BoundingBox3D::new(Point3::new(-0.1, -0.1, -0.1), Point3::new(1.0, 0.1, 0.1));
        let hits = cloud.query_region(&bbox);
        assert_eq!(
            hits,
            vec![0],
            "grid query should return only the first point"
        );
    }

    #[test]
    fn test_grid_stays_consistent_after_prune() {
        let mut cloud = GaussianCloud::new();
        cloud.add(make_gaussian(0.0, 0.0, 0.0, 0.001)); // will be pruned
        cloud.add(make_gaussian(0.0, 0.0, 0.0, 0.8)); // stays
        cloud.prune(0.01);
        // After prune the grid must be dirty; a query should rebuild cleanly.
        let bbox = BoundingBox3D::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let hits = cloud.query_region(&bbox);
        assert_eq!(hits.len(), 1);
    }
}
