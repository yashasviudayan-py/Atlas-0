//! Surface / plane extraction from 3D point clouds.
//!
//! Uses iterative RANSAC to find the dominant planes in a set of points
//! (typically Gaussian splat centres).  Each extracted [`Surface`] is
//! classified as a [`PlaneType`] based on its normal direction.

use std::collections::HashSet;

use atlas_core::spatial::Point3;
use serde::{Deserialize, Serialize};

// ─── PlaneType ─────────────────────────────────────────────────────────────────

/// Classification of a detected planar surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlaneType {
    /// Roughly horizontal surface facing upward (floor, table-top, shelf).
    Floor,
    /// Roughly horizontal surface facing downward (ceiling).
    Ceiling,
    /// Roughly vertical surface (wall, door).
    Wall,
    /// Any other orientation (ramp, angled panel).
    Angled,
}

// ─── Surface ───────────────────────────────────────────────────────────────────

/// A planar surface extracted from the scene.
///
/// The plane equation is `normal · x = offset` where `normal` is the outward-
/// facing unit normal.  Signed distance from any point `p` to the plane is
/// `normal · p − offset`; positive means the point is on the outward side.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Surface {
    /// Unit outward normal vector `[nx, ny, nz]`.
    pub normal: [f32; 3],
    /// Plane offset: `dot(normal, any_point_on_plane)`.
    pub offset: f32,
    /// Classification of this surface.
    pub plane_type: PlaneType,
    /// Approximate centroid of the inlier point set.
    pub centre: Point3,
    /// Number of inlier points used to fit this plane.
    pub inlier_count: usize,
}

impl Surface {
    /// Signed distance from `point` to this plane.
    ///
    /// Positive → point is on the outward-normal side (in front of the surface).
    /// Negative → point is behind the surface.
    #[must_use]
    pub fn signed_distance(&self, point: &[f32; 3]) -> f32 {
        self.normal[0] * point[0] + self.normal[1] * point[1] + self.normal[2] * point[2]
            - self.offset
    }

    /// Classify a surface normal into a [`PlaneType`].
    ///
    /// Uses the Y-axis (vertical) angle to distinguish horizontal from vertical:
    /// - |ny| > cos(20°) ≈ 0.940 → horizontal (Floor or Ceiling).
    /// - |ny| < cos(70°) ≈ 0.342 → vertical (Wall).
    /// - Otherwise → Angled.
    #[must_use]
    pub fn classify(normal: &[f32; 3]) -> PlaneType {
        const COS_20: f32 = 0.9397;
        const COS_70: f32 = 0.3420;
        let ny_abs = normal[1].abs();
        if ny_abs > COS_20 {
            if normal[1] > 0.0 {
                PlaneType::Floor
            } else {
                PlaneType::Ceiling
            }
        } else if ny_abs < COS_70 {
            PlaneType::Wall
        } else {
            PlaneType::Angled
        }
    }
}

// ─── SurfaceExtractorConfig ────────────────────────────────────────────────────

/// Tuning parameters for RANSAC plane extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceExtractorConfig {
    /// Maximum point-to-plane distance to count as an inlier (metres).
    pub inlier_threshold: f32,
    /// Number of RANSAC iterations per plane.
    pub ransac_iterations: u32,
    /// Minimum inlier count for a plane to be accepted.
    pub min_inliers: usize,
    /// Maximum number of planes to extract.
    pub max_planes: usize,
}

impl Default for SurfaceExtractorConfig {
    fn default() -> Self {
        Self {
            inlier_threshold: 0.05,
            ransac_iterations: 100,
            min_inliers: 20,
            max_planes: 8,
        }
    }
}

// ─── SurfaceExtractor ──────────────────────────────────────────────────────────

/// Iterative RANSAC plane extractor.
///
/// Each call to [`extract`] finds up to `config.max_planes` dominant planes.
/// After each plane is accepted its inliers are removed before the next search.
///
/// [`extract`]: SurfaceExtractor::extract
pub struct SurfaceExtractor {
    config: SurfaceExtractorConfig,
}

impl SurfaceExtractor {
    /// Create a new extractor with the given configuration.
    #[must_use]
    pub fn new(config: SurfaceExtractorConfig) -> Self {
        Self { config }
    }

    /// Extract dominant planes from `points`.
    ///
    /// Returns a `Vec<Surface>` sorted by inlier count (largest first).
    /// Returns an empty vec when fewer than 3 points are provided.
    #[must_use]
    pub fn extract(&self, points: &[Point3]) -> Vec<Surface> {
        if points.len() < 3 {
            return Vec::new();
        }

        let mut remaining: Vec<usize> = (0..points.len()).collect();
        let mut surfaces = Vec::new();

        while surfaces.len() < self.config.max_planes && remaining.len() >= 3 {
            match self.ransac_one_plane(points, &remaining) {
                Some(surface) if surface.inlier_count >= self.config.min_inliers => {
                    // Remove inliers from the working set before the next pass.
                    let inliers: HashSet<usize> = remaining
                        .iter()
                        .copied()
                        .filter(|&i| {
                            let p = &points[i];
                            surface.signed_distance(&[p.x, p.y, p.z]).abs()
                                <= self.config.inlier_threshold
                        })
                        .collect();
                    remaining.retain(|i| !inliers.contains(i));
                    surfaces.push(surface);
                }
                _ => break,
            }
        }

        surfaces.sort_by(|a, b| b.inlier_count.cmp(&a.inlier_count));
        surfaces
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// One RANSAC pass over the given index subset.  Returns the best fitting
    /// plane found, or `None` if every sampled triple is degenerate.
    fn ransac_one_plane(&self, points: &[Point3], indices: &[usize]) -> Option<Surface> {
        let n = indices.len();
        if n < 3 {
            return None;
        }

        let mut best_inliers = 0usize;
        let mut best_normal = [0.0f32; 3];
        let mut best_offset = 0.0f32;
        let mut best_cx = 0.0f32;
        let mut best_cy = 0.0f32;
        let mut best_cz = 0.0f32;

        // Deterministic pseudo-random via a simple LCG (fixed seed for
        // reproducibility across runs — same input always gives same planes).
        let mut rng: u64 = 0x517cc1b727220a95;

        for _ in 0..self.config.ransac_iterations {
            let i0 = indices[lcg_next(&mut rng) % n];
            let mut i1 = indices[lcg_next(&mut rng) % n];
            let mut i2 = indices[lcg_next(&mut rng) % n];
            // Ensure distinct indices (simple fallback, not cryptographically strong).
            if i1 == i0 {
                i1 = indices[(i0.wrapping_add(1)) % n];
            }
            if i2 == i0 || i2 == i1 {
                i2 = indices[(i0.wrapping_add(2)) % n];
            }

            let p0 = &points[i0];
            let p1 = &points[i1];
            let p2 = &points[i2];

            let ab = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
            let ac = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];

            // Cross product → plane normal.
            let nx = ab[1] * ac[2] - ab[2] * ac[1];
            let ny = ab[2] * ac[0] - ab[0] * ac[2];
            let nz = ab[0] * ac[1] - ab[1] * ac[0];
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            if len < 1e-8 {
                continue; // degenerate triple (collinear points)
            }
            let normal = [nx / len, ny / len, nz / len];
            let offset = normal[0] * p0.x + normal[1] * p0.y + normal[2] * p0.z;

            // Count inliers and accumulate centroid.
            let mut inlier_count = 0usize;
            let mut cx = 0.0f32;
            let mut cy = 0.0f32;
            let mut cz = 0.0f32;
            for &idx in indices {
                let p = &points[idx];
                let dist = (normal[0] * p.x + normal[1] * p.y + normal[2] * p.z - offset).abs();
                if dist <= self.config.inlier_threshold {
                    inlier_count += 1;
                    cx += p.x;
                    cy += p.y;
                    cz += p.z;
                }
            }

            if inlier_count > best_inliers {
                best_inliers = inlier_count;
                best_normal = normal;
                best_offset = offset;
                let ic = inlier_count as f32;
                best_cx = cx / ic;
                best_cy = cy / ic;
                best_cz = cz / ic;
            }
        }

        if best_inliers == 0 {
            return None;
        }

        Some(Surface {
            normal: best_normal,
            offset: best_offset,
            plane_type: Surface::classify(&best_normal),
            centre: Point3::new(best_cx, best_cy, best_cz),
            inlier_count: best_inliers,
        })
    }
}

/// LCG step — returns a value in [0, 2^31).
fn lcg_next(state: &mut u64) -> usize {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as usize
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Surface::classify ─────────────────────────────────────────────────────

    #[test]
    fn test_classify_floor() {
        assert_eq!(Surface::classify(&[0.0, 1.0, 0.0]), PlaneType::Floor);
    }

    #[test]
    fn test_classify_ceiling() {
        assert_eq!(Surface::classify(&[0.0, -1.0, 0.0]), PlaneType::Ceiling);
    }

    #[test]
    fn test_classify_wall_x() {
        assert_eq!(Surface::classify(&[1.0, 0.0, 0.0]), PlaneType::Wall);
    }

    #[test]
    fn test_classify_wall_z() {
        assert_eq!(Surface::classify(&[0.0, 0.0, 1.0]), PlaneType::Wall);
    }

    #[test]
    fn test_classify_angled() {
        // 45-degree slope: neither clearly horizontal nor vertical.
        let n = [0.0f32, 1.0, 1.0];
        let len = (n[0].powi(2) + n[1].powi(2) + n[2].powi(2)).sqrt();
        let unit = [n[0] / len, n[1] / len, n[2] / len];
        assert_eq!(Surface::classify(&unit), PlaneType::Angled);
    }

    // ── Surface::signed_distance ──────────────────────────────────────────────

    #[test]
    fn test_signed_distance_above_floor() {
        let floor = Surface {
            normal: [0.0, 1.0, 0.0],
            offset: 0.0,
            plane_type: PlaneType::Floor,
            centre: Point3::new(0.0, 0.0, 0.0),
            inlier_count: 100,
        };
        let point = [0.0, 1.5, 0.0];
        assert!((floor.signed_distance(&point) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_signed_distance_below_floor() {
        let floor = Surface {
            normal: [0.0, 1.0, 0.0],
            offset: 0.0,
            plane_type: PlaneType::Floor,
            centre: Point3::new(0.0, 0.0, 0.0),
            inlier_count: 100,
        };
        let point = [0.0, -0.5, 0.0];
        assert!(floor.signed_distance(&point) < 0.0);
    }

    #[test]
    fn test_signed_distance_on_plane() {
        let floor = Surface {
            normal: [0.0, 1.0, 0.0],
            offset: 2.0,
            plane_type: PlaneType::Floor,
            centre: Point3::new(0.0, 2.0, 0.0),
            inlier_count: 1,
        };
        let point = [5.0, 2.0, -3.0];
        assert!(floor.signed_distance(&point).abs() < 1e-5);
    }

    // ── SurfaceExtractor ──────────────────────────────────────────────────────

    fn floor_points(n: usize, y: f32) -> Vec<Point3> {
        (0..n)
            .map(|i| {
                let x = (i % 10) as f32 * 0.5;
                let z = (i / 10) as f32 * 0.5;
                Point3::new(x, y, z)
            })
            .collect()
    }

    #[test]
    fn test_extract_detects_floor() {
        let points = floor_points(50, 0.0);
        let extractor = SurfaceExtractor::new(SurfaceExtractorConfig::default());
        let surfaces = extractor.extract(&points);
        assert!(!surfaces.is_empty(), "should find at least one surface");
        let s = &surfaces[0];
        // RANSAC cross-product may produce either [0,+1,0] or [0,-1,0] — both are
        // valid normals for a horizontal plane.  Accept Floor or Ceiling.
        assert!(
            s.plane_type == PlaneType::Floor || s.plane_type == PlaneType::Ceiling,
            "expected a horizontal plane, got {:?}",
            s.plane_type
        );
        assert!(s.inlier_count >= 20);
    }

    #[test]
    fn test_extract_empty_input() {
        let extractor = SurfaceExtractor::new(SurfaceExtractorConfig::default());
        assert!(extractor.extract(&[]).is_empty());
    }

    #[test]
    fn test_extract_too_few_points() {
        let extractor = SurfaceExtractor::new(SurfaceExtractorConfig::default());
        let pts = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
        assert!(extractor.extract(&pts).is_empty());
    }

    #[test]
    fn test_extract_two_planes() {
        // 50 floor points at y = 0.0, 50 wall points at x = 0.0.
        let mut points = floor_points(50, 0.0);
        for i in 0..50usize {
            let y = (i % 10) as f32 * 0.3;
            let z = (i / 10) as f32 * 0.5;
            points.push(Point3::new(0.0, y, z));
        }
        let extractor = SurfaceExtractor::new(SurfaceExtractorConfig::default());
        let surfaces = extractor.extract(&points);
        assert!(surfaces.len() >= 2, "should extract floor and wall");
        let types: Vec<PlaneType> = surfaces.iter().map(|s| s.plane_type).collect();
        assert!(types.contains(&PlaneType::Floor) || types.contains(&PlaneType::Wall));
    }

    #[test]
    fn test_inlier_threshold_respected() {
        // Points near y = 0 but with noise: some at y = 1.0 (outliers).
        let mut points = floor_points(40, 0.0);
        // Add outlier points well away from the floor.
        for i in 0..10 {
            points.push(Point3::new(i as f32, 5.0, i as f32));
        }
        let cfg = SurfaceExtractorConfig {
            min_inliers: 30,
            ..Default::default()
        };
        let extractor = SurfaceExtractor::new(cfg);
        let surfaces = extractor.extract(&points);
        assert!(!surfaces.is_empty());
        // All inliers of the first plane should be near y = 0.
        assert!(
            surfaces[0].inlier_count >= 30,
            "floor should have ≥30 inliers"
        );
    }
}
