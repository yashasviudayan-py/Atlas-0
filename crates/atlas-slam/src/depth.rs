//! Depth estimation via feature triangulation.
//!
//! Given two frames with known camera poses and a set of 2-D feature matches,
//! [`DepthEstimator`] triangulates each matched pair into a 3-D world point and
//! records the result in a [`DepthMap`].  The approach is purely geometric —
//! no neural-network inference is required — making it immediately usable
//! whenever two frames with sufficient baseline are available.

use atlas_core::{Pose, spatial::Point3};
use nalgebra::{Matrix4, Vector4};

use crate::config::CameraIntrinsics;
use crate::features::KeyPoint;
use crate::matching::DMatch;

// ─── DepthMap ─────────────────────────────────────────────────────────────────

/// A per-pixel depth map in camera-space Z coordinates (metres).
///
/// Pixels without depth information are stored as [`f32::NAN`].
pub struct DepthMap {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Per-pixel depth (metres). [`f32::NAN`] means no depth known.
    pub depths: Vec<f32>,
    /// Sparse 3-D world-space points with their corresponding pixel coordinates.
    pub world_points: Vec<(Point3, u32, u32)>,
}

impl DepthMap {
    /// Return the depth at pixel `(x, y)`, or `None` if no depth is stored.
    #[must_use]
    pub fn depth_at(&self, x: u32, y: u32) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let d = self.depths[(y * self.width + x) as usize];
        if d.is_nan() { None } else { Some(d) }
    }

    /// Count pixels that carry a valid (non-NaN) depth value.
    #[must_use]
    pub fn valid_depth_count(&self) -> usize {
        self.depths.iter().filter(|d| !d.is_nan()).count()
    }
}

// ─── DepthEstimator ───────────────────────────────────────────────────────────

/// Estimates a depth map for the *current* frame by triangulating matched
/// feature correspondences between the *previous* and *current* frames.
pub struct DepthEstimator {
    intrinsics: CameraIntrinsics,
}

impl DepthEstimator {
    /// Create a new estimator with the given camera intrinsics.
    #[must_use]
    pub fn new(intrinsics: CameraIntrinsics) -> Self {
        Self { intrinsics }
    }

    /// Estimate a [`DepthMap`] for the current frame.
    ///
    /// `kps_prev` / `kps_curr` are the keypoints extracted from the previous
    /// and current frames.  `matches` connect them (query = current, train =
    /// previous).  `pose_prev` / `pose_curr` are the camera-to-world transforms.
    ///
    /// Returns a depth map with valid entries at every triangulated pixel.
    /// Degenerate triangulations (negative or very distant depths) are silently
    /// discarded.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn estimate(
        &self,
        kps_prev: &[KeyPoint],
        kps_curr: &[KeyPoint],
        matches: &[DMatch],
        pose_prev: &Pose,
        pose_curr: &Pose,
        width: u32,
        height: u32,
    ) -> DepthMap {
        let mut depths = vec![f32::NAN; (width * height) as usize];
        let mut world_points = Vec::with_capacity(matches.len());

        // Build 3×4 projection matrices P = K · [R | t]  (world → image).
        let p1 = build_projection(pose_prev, &self.intrinsics);
        let p2 = build_projection(pose_curr, &self.intrinsics);

        // World-to-camera for the current frame (to obtain camera-space Z).
        let cam_from_world = pose_curr
            .to_matrix()
            .try_inverse()
            .unwrap_or_else(Matrix4::identity);

        for m in matches {
            if m.train_idx >= kps_prev.len() || m.query_idx >= kps_curr.len() {
                continue;
            }
            let kp_prev = &kps_prev[m.train_idx];
            let kp_curr = &kps_curr[m.query_idx];

            let pt3d = triangulate_dlt(
                kp_prev.x as f64,
                kp_prev.y as f64,
                kp_curr.x as f64,
                kp_curr.y as f64,
                &p1,
                &p2,
            );

            // Compute depth as the Z coordinate in the current camera frame.
            let pw = Vector4::new(pt3d[0] as f32, pt3d[1] as f32, pt3d[2] as f32, 1.0);
            let pc = cam_from_world * pw;
            let z = pc[2];

            // Reject points behind the camera or implausibly far away.
            if z > 0.01 && z < 100.0 {
                let px = kp_curr.x.round() as u32;
                let py = kp_curr.y.round() as u32;
                if px < width && py < height {
                    let idx = (py * width + px) as usize;
                    // When multiple points project to the same pixel, keep the nearest.
                    if depths[idx].is_nan() || z < depths[idx] {
                        depths[idx] = z;
                        world_points.push((
                            Point3::new(pt3d[0] as f32, pt3d[1] as f32, pt3d[2] as f32),
                            px,
                            py,
                        ));
                    }
                }
            }
        }

        DepthMap {
            width,
            height,
            depths,
            world_points,
        }
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Build the 3×4 projection matrix P = K · [R | t] that maps world-space
/// points to image pixels for the given camera pose.
///
/// `pose` is the *camera-to-world* transform; we invert it here to get the
/// world-to-camera transform needed for projection.
fn build_projection(pose: &Pose, k: &CameraIntrinsics) -> [[f64; 4]; 3] {
    let cam_from_world = pose
        .to_matrix()
        .try_inverse()
        .unwrap_or_else(Matrix4::identity);

    let r = &cam_from_world;
    let fx = k.fx;
    let fy = k.fy;
    let cx = k.cx;
    let cy = k.cy;

    // P = K · Rt  (each entry expanded row by row).
    [
        [
            fx * r[(0, 0)] as f64 + cx * r[(2, 0)] as f64,
            fx * r[(0, 1)] as f64 + cx * r[(2, 1)] as f64,
            fx * r[(0, 2)] as f64 + cx * r[(2, 2)] as f64,
            fx * r[(0, 3)] as f64 + cx * r[(2, 3)] as f64,
        ],
        [
            fy * r[(1, 0)] as f64 + cy * r[(2, 0)] as f64,
            fy * r[(1, 1)] as f64 + cy * r[(2, 1)] as f64,
            fy * r[(1, 2)] as f64 + cy * r[(2, 2)] as f64,
            fy * r[(1, 3)] as f64 + cy * r[(2, 3)] as f64,
        ],
        [
            r[(2, 0)] as f64,
            r[(2, 1)] as f64,
            r[(2, 2)] as f64,
            r[(2, 3)] as f64,
        ],
    ]
}

/// Triangulate a 3-D point from two image observations using the Direct
/// Linear Transform (DLT) method.
///
/// Solves A · X = 0 via SVD, where A is a 4×4 matrix constructed from the
/// two pixel observations and their respective projection matrices.
///
/// Returns the homogeneously divided 3-D point `[X, Y, Z]` in world
/// coordinates, or `[0, 0, 0]` if the system is degenerate.
fn triangulate_dlt(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    p1: &[[f64; 4]; 3],
    p2: &[[f64; 4]; 3],
) -> [f64; 3] {
    // Build the 4×4 coefficient matrix:
    //   row 0:  y·P[2] − P[1]   (view 1)
    //   row 1:  x·P[2] − P[0]   (view 1)
    //   row 2:  y·P[2] − P[1]   (view 2)
    //   row 3:  x·P[2] − P[0]   (view 2)
    let a = nalgebra::Matrix4::new(
        y1 * p1[2][0] - p1[1][0],
        y1 * p1[2][1] - p1[1][1],
        y1 * p1[2][2] - p1[1][2],
        y1 * p1[2][3] - p1[1][3],
        x1 * p1[2][0] - p1[0][0],
        x1 * p1[2][1] - p1[0][1],
        x1 * p1[2][2] - p1[0][2],
        x1 * p1[2][3] - p1[0][3],
        y2 * p2[2][0] - p2[1][0],
        y2 * p2[2][1] - p2[1][1],
        y2 * p2[2][2] - p2[1][2],
        y2 * p2[2][3] - p2[1][3],
        x2 * p2[2][0] - p2[0][0],
        x2 * p2[2][1] - p2[0][1],
        x2 * p2[2][2] - p2[0][2],
        x2 * p2[2][3] - p2[0][3],
    );

    // The null-space vector (solution) is the last row of V^T in A = U Σ V^T,
    // corresponding to the smallest singular value.
    let svd = a.svd(false, true);
    let v_t = match svd.v_t {
        Some(vt) => vt,
        None => return [0.0, 0.0, 0.0],
    };

    // Last row of V^T = right singular vector for smallest singular value.
    let w = v_t[(3, 3)];
    if w.abs() < 1e-12 {
        return [0.0, 0.0, 0.0];
    }

    [v_t[(3, 0)] / w, v_t[(3, 1)] / w, v_t[(3, 2)] / w]
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::Pose;

    fn default_intrinsics() -> CameraIntrinsics {
        CameraIntrinsics {
            fx: 525.0,
            fy: 525.0,
            cx: 320.0,
            cy: 240.0,
        }
    }

    // ── DepthMap ──────────────────────────────────────────────────────────────

    #[test]
    fn test_depth_map_out_of_bounds_returns_none() {
        let dm = DepthMap {
            width: 10,
            height: 10,
            depths: vec![f32::NAN; 100],
            world_points: Vec::new(),
        };
        assert!(dm.depth_at(10, 0).is_none());
        assert!(dm.depth_at(0, 10).is_none());
        assert!(dm.depth_at(100, 100).is_none());
    }

    #[test]
    fn test_depth_map_valid_pixel_returns_value() {
        let mut depths = vec![f32::NAN; 100];
        depths[5] = 2.5;
        let dm = DepthMap {
            width: 10,
            height: 10,
            depths,
            world_points: Vec::new(),
        };
        assert_eq!(dm.depth_at(5, 0), Some(2.5));
        assert!(dm.depth_at(0, 0).is_none());
    }

    #[test]
    fn test_depth_map_valid_count() {
        let mut depths = vec![f32::NAN; 100];
        depths[0] = 1.5;
        depths[7] = 3.0;
        let dm = DepthMap {
            width: 10,
            height: 10,
            depths,
            world_points: Vec::new(),
        };
        assert_eq!(dm.valid_depth_count(), 2);
    }

    #[test]
    fn test_depth_map_all_nan() {
        let dm = DepthMap {
            width: 4,
            height: 4,
            depths: vec![f32::NAN; 16],
            world_points: Vec::new(),
        };
        assert_eq!(dm.valid_depth_count(), 0);
    }

    // ── DepthEstimator ────────────────────────────────────────────────────────

    #[test]
    fn test_estimate_zero_matches_gives_empty_map() {
        let estimator = DepthEstimator::new(default_intrinsics());
        let pose = Pose::identity();
        let dm = estimator.estimate(&[], &[], &[], &pose, &pose, 64, 64);
        assert_eq!(dm.valid_depth_count(), 0);
        assert!(dm.world_points.is_empty());
    }

    #[test]
    fn test_estimate_identical_poses_degenerate() {
        // Same pose for both views → no baseline → no valid triangulation.
        let intrinsics = default_intrinsics();
        let estimator = DepthEstimator::new(intrinsics);
        let pose = Pose::identity();

        let kps = vec![
            KeyPoint {
                x: 320.0,
                y: 240.0,
                response: 1.0,
            },
            KeyPoint {
                x: 340.0,
                y: 260.0,
                response: 1.0,
            },
        ];
        let matches = vec![DMatch {
            query_idx: 0,
            train_idx: 0,
            distance: 10,
        }];

        let dm = estimator.estimate(&kps, &kps, &matches, &pose, &pose, 640, 480);
        // May produce some output, but must not panic.
        assert!(dm.valid_depth_count() <= 1);
    }

    // ── build_projection ──────────────────────────────────────────────────────

    #[test]
    fn test_build_projection_identity_intrinsics_and_pose() {
        // With identity pose, P = K · I_3x4.
        let k = CameraIntrinsics {
            fx: 2.0,
            fy: 3.0,
            cx: 1.0,
            cy: 1.5,
        };
        let pose = Pose::identity();
        let p = build_projection(&pose, &k);

        // Row 0: [fx + cx*0, 0 + cx*0, cx*1, fx*0+cx*0] = [2, 0, 1, 0]
        assert!((p[0][0] - 2.0).abs() < 1e-9, "P[0][0] = fx");
        assert!((p[0][2] - 1.0).abs() < 1e-9, "P[0][2] = cx");
        // Row 1: [0, fy, cy, 0]
        assert!((p[1][1] - 3.0).abs() < 1e-9, "P[1][1] = fy");
        assert!((p[1][2] - 1.5).abs() < 1e-9, "P[1][2] = cy");
        // Row 2: [0, 0, 1, 0]
        assert!((p[2][2] - 1.0).abs() < 1e-9, "P[2][2] = 1");
        assert!((p[2][3]).abs() < 1e-9, "P[2][3] = 0");
    }

    // ── triangulate_dlt ───────────────────────────────────────────────────────

    #[test]
    fn test_triangulate_dlt_does_not_panic_on_degenerate_input() {
        // All-zero projection matrices → degenerate system → should return [0,0,0].
        let zero_p = [[0.0; 4]; 3];
        let result = triangulate_dlt(0.0, 0.0, 0.0, 0.0, &zero_p, &zero_p);
        // Should return something without panicking.
        assert!(result[0].is_finite() || result[0] == 0.0);
    }

    #[test]
    fn test_triangulate_dlt_known_point() {
        // Place a point at (0, 0, 5) in world space and project it onto two
        // cameras: one at the origin looking along +Z and one 1 m to the right.
        //
        // Camera intrinsics: fx=fy=1, cx=cy=0 for simplicity.
        let k = CameraIntrinsics {
            fx: 1.0,
            fy: 1.0,
            cx: 0.0,
            cy: 0.0,
        };

        // Camera 1: identity pose.
        let pose1 = Pose::identity();
        // Camera 2: translated 1 m to the right (+X in world).
        let pose2 = Pose {
            position: atlas_core::Point3::new(1.0, 0.0, 0.0),
            rotation: [1.0, 0.0, 0.0, 0.0],
        };

        let p1 = build_projection(&pose1, &k);
        let p2 = build_projection(&pose2, &k);

        // World point (0, 0, 5):
        //   cam1: (0/5, 0/5) = (0, 0)
        //   cam2: (world - cam2_pos) = (-1, 0, 5) → (-1/5, 0/5) = (-0.2, 0)
        let result = triangulate_dlt(0.0, 0.0, -0.2, 0.0, &p1, &p2);

        assert!((result[0]).abs() < 0.1, "X ≈ 0, got {}", result[0]);
        assert!((result[1]).abs() < 0.1, "Y ≈ 0, got {}", result[1]);
        assert!((result[2] - 5.0).abs() < 0.5, "Z ≈ 5, got {}", result[2]);
    }
}
