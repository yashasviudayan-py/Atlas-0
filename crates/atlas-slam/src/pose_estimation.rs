//! Relative pose estimation from 2D–2D feature correspondences.
//!
//! Implements the normalised 8-point algorithm inside a RANSAC loop to
//! compute the essential matrix **E**, then decomposes **E** into the four
//! `(R, t)` candidates and selects the physically correct one via a
//! cheirality check (triangulated points must be in front of both cameras).
//!
//! # Note on scale ambiguity
//! Monocular visual odometry recovers rotation exactly but translation only
//! up to an unknown scale factor.  The translation vector returned here is
//! a unit vector; absolute metric scale is not recoverable without additional
//! information (e.g. known object size or stereo baseline).

use atlas_core::{Point3, Pose};
use nalgebra::{DMatrix, Matrix3, Rotation3, UnitQuaternion, Vector3};

use crate::config::CameraIntrinsics;

// ─── Public surface ───────────────────────────────────────────────────────────

/// A 2D point in image (pixel or normalised) coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    #[must_use]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Estimates the relative camera pose from 2D–2D pixel correspondences.
///
/// # Examples
/// ```
/// use atlas_slam::{config::CameraIntrinsics, pose_estimation::PoseEstimator};
///
/// let intrinsics = CameraIntrinsics::default();
/// let estimator = PoseEstimator::new(intrinsics, 200, 1e-3);
/// ```
pub struct PoseEstimator {
    intrinsics: CameraIntrinsics,
    ransac_iterations: usize,
    ransac_threshold: f64,
}

impl PoseEstimator {
    /// Create a new pose estimator.
    ///
    /// # Arguments
    /// * `intrinsics` — Pinhole camera intrinsics used to convert pixel
    ///   coordinates to normalised image coordinates.
    /// * `ransac_iterations` — Number of RANSAC trials (typical: 100–500).
    /// * `ransac_threshold` — Sampson-distance threshold (in normalised image
    ///   coords) below which a point pair is counted as an inlier.
    ///   Typical value: `1e-3`.
    #[must_use]
    pub fn new(
        intrinsics: CameraIntrinsics,
        ransac_iterations: usize,
        ransac_threshold: f64,
    ) -> Self {
        Self {
            intrinsics,
            ransac_iterations,
            ransac_threshold,
        }
    }

    /// Estimate the relative camera pose from pixel-coordinate correspondences.
    ///
    /// Each element of `correspondences` is `([x1, y1], [x2, y2])` where
    /// `(x1, y1)` is the point in the **previous** frame and `(x2, y2)` is
    /// the point in the **current** frame.
    ///
    /// Returns `Ok(Some(pose))` on success.  Returns `Ok(None)` when there are
    /// not enough correspondences or too few inliers to produce a reliable
    /// estimate.
    ///
    /// # Errors
    /// Returns [`atlas_core::error::SlamError::PoseEstimationFailed`] when a
    /// required linear-algebra operation fails numerically.
    pub fn estimate_pose(
        &self,
        correspondences: &[([f64; 2], [f64; 2])],
        frame_id: u64,
    ) -> crate::Result<Option<Pose>> {
        use atlas_core::error::SlamError;

        if correspondences.len() < 8 {
            return Ok(None);
        }

        // Pixel → normalised image coordinates.
        let pts_prev: Vec<Point2> = correspondences
            .iter()
            .map(|([x, y], _)| self.to_normalised(*x, *y))
            .collect();
        let pts_curr: Vec<Point2> = correspondences
            .iter()
            .map(|(_, [x, y])| self.to_normalised(*x, *y))
            .collect();

        // RANSAC essential matrix.
        let Some((_, inlier_mask)) = self.ransac_essential(&pts_prev, &pts_curr) else {
            return Ok(None);
        };

        let inlier_count = inlier_mask.iter().filter(|&&b| b).count();
        if inlier_count < 8 {
            tracing::warn!(
                frame_id,
                inliers = inlier_count,
                "insufficient inliers for pose estimation"
            );
            return Ok(None);
        }

        // Collect inlier subsets and refit E on all of them.
        let in_prev: Vec<Point2> = pts_prev
            .iter()
            .zip(&inlier_mask)
            .filter_map(|(p, &keep)| keep.then_some(*p))
            .collect();
        let in_curr: Vec<Point2> = pts_curr
            .iter()
            .zip(&inlier_mask)
            .filter_map(|(p, &keep)| keep.then_some(*p))
            .collect();

        let e = compute_essential_matrix(&in_prev, &in_curr).ok_or_else(|| {
            SlamError::PoseEstimationFailed(format!(
                "essential matrix refit failed at frame {frame_id}"
            ))
        })?;

        self.decompose_essential(&e, &in_prev, &in_curr)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Convert a pixel coordinate to a normalised image coordinate using K⁻¹.
    fn to_normalised(&self, x: f64, y: f64) -> Point2 {
        Point2 {
            x: (x - self.intrinsics.cx) / self.intrinsics.fx,
            y: (y - self.intrinsics.cy) / self.intrinsics.fy,
        }
    }

    /// Run RANSAC over the 8-point algorithm and return the best `(E, inlier_mask)`.
    fn ransac_essential(
        &self,
        pts1: &[Point2],
        pts2: &[Point2],
    ) -> Option<(Matrix3<f64>, Vec<bool>)> {
        let n = pts1.len();
        if n < 8 {
            return None;
        }

        let mut rng = Lcg::new(0xDEAD_BEEF_1234_5678_u64);
        let mut best_inliers: Vec<bool> = vec![false; n];
        let mut best_count = 0usize;
        let mut best_e = Matrix3::identity();

        for _ in 0..self.ransac_iterations {
            let idx = sample_8(&mut rng, n);
            let s1: Vec<Point2> = idx.iter().map(|&i| pts1[i]).collect();
            let s2: Vec<Point2> = idx.iter().map(|&i| pts2[i]).collect();

            let Some(e) = compute_essential_matrix(&s1, &s2) else {
                continue;
            };

            let inliers: Vec<bool> = pts1
                .iter()
                .zip(pts2)
                .map(|(p1, p2)| sampson_distance(&e, p1, p2) < self.ransac_threshold)
                .collect();
            let count = inliers.iter().filter(|&&b| b).count();

            if count > best_count {
                best_count = count;
                best_inliers = inliers;
                best_e = e;
            }
        }

        (best_count >= 8).then_some((best_e, best_inliers))
    }

    /// Decompose **E** into the four `(R, t)` candidates and pick the one
    /// where the most triangulated points are in front of both cameras.
    fn decompose_essential(
        &self,
        e: &Matrix3<f64>,
        pts1: &[Point2],
        pts2: &[Point2],
    ) -> crate::Result<Option<Pose>> {
        use atlas_core::error::SlamError;

        let svd = e.svd(true, true);
        let u = svd
            .u
            .ok_or_else(|| SlamError::PoseEstimationFailed("SVD(E) failed to produce U".into()))?;
        let vt = svd.v_t.ok_or_else(|| {
            SlamError::PoseEstimationFailed("SVD(E) failed to produce V^T".into())
        })?;

        // Enforce det(U) = det(V^T) = +1 so that R = U W V^T is a proper rotation.
        let u = if u.determinant() < 0.0 { -u } else { u };
        let vt = if vt.determinant() < 0.0 { -vt } else { vt };

        #[rustfmt::skip]
        let w = Matrix3::new(
            0.0, -1.0, 0.0,
            1.0,  0.0, 0.0,
            0.0,  0.0, 1.0,
        );

        let t = u.column(2).into_owned();
        let r1 = u * w * vt;
        let r2 = u * w.transpose() * vt;

        let candidates: [(Matrix3<f64>, Vector3<f64>); 4] = [(r1, t), (r1, -t), (r2, t), (r2, -t)];

        // Select the candidate that puts the most points in front of both cameras.
        let best = candidates
            .iter()
            .max_by_key(|(r, t)| count_positive_depth(pts1, pts2, r, t));

        let Some((r_best, t_best)) = best else {
            return Ok(None);
        };

        // Require at least half the inliers to be in front.
        let positive = count_positive_depth(pts1, pts2, r_best, t_best);
        if positive < pts1.len() / 2 {
            return Ok(None);
        }

        // Rotation matrix → unit quaternion.
        let rot = Rotation3::from_matrix_unchecked(*r_best);
        let uq: UnitQuaternion<f64> = UnitQuaternion::from_rotation_matrix(&rot);
        let q = uq.into_inner();

        let pose = Pose {
            position: Point3::new(t_best[0] as f32, t_best[1] as f32, t_best[2] as f32),
            rotation: [q.w as f32, q.i as f32, q.j as f32, q.k as f32],
        };

        Ok(Some(pose))
    }
}

// ─── Core algorithm functions ────────────────────────────────────────────────

/// Compute the essential matrix from ≥8 normalised point correspondences
/// using the 8-point algorithm with Hartley normalisation.
///
/// Returns `None` if the SVD computation fails (e.g. degenerate configuration).
fn compute_essential_matrix(pts1: &[Point2], pts2: &[Point2]) -> Option<Matrix3<f64>> {
    let n = pts1.len();
    if n < 8 {
        return None;
    }

    let (t1, norm1) = hartley_normalise(pts1);
    let (t2, norm2) = hartley_normalise(pts2);

    // Build the epipolar constraint matrix A (n × 9).
    // Row i: [x2·x1, x2·y1, x2, y2·x1, y2·y1, y2, x1, y1, 1]
    let mut a_data = Vec::with_capacity(n * 9);
    for (p1, p2) in norm1.iter().zip(&norm2) {
        a_data.extend_from_slice(&[
            p2.x * p1.x,
            p2.x * p1.y,
            p2.x,
            p2.y * p1.x,
            p2.y * p1.y,
            p2.y,
            p1.x,
            p1.y,
            1.0_f64,
        ]);
    }

    let a = DMatrix::from_row_slice(n, 9, &a_data);
    let svd = a.svd(false, true);
    let vt = svd.v_t?;

    // The null-space vector corresponds to the smallest singular value
    // (last row of V^T since nalgebra orders them descending).
    // Clone to an owned row vector so we can call `as_slice` (view is non-contiguous).
    let last_row: Vec<f64> = vt.row(vt.nrows() - 1).iter().copied().collect();
    let e_norm = Matrix3::from_row_slice(&last_row);

    // Enforce rank-2 constraint: set the smallest singular value to 0
    // and balance the remaining two.
    let svd_e = e_norm.svd(true, true);
    let eu = svd_e.u?;
    let evt = svd_e.v_t?;
    let mut sigma = svd_e.singular_values;
    let avg = (sigma[0] + sigma[1]) / 2.0;
    sigma[0] = avg;
    sigma[1] = avg;
    sigma[2] = 0.0;
    let e_rank2 = eu * Matrix3::from_diagonal(&sigma) * evt;

    // Denormalise: E = T2^T · Ê · T1
    Some(t2.transpose() * e_rank2 * t1)
}

/// Hartley normalisation of a point set.
///
/// Translates points so their centroid is at the origin, then scales so their
/// mean distance from the origin is √2.  Returns the 3×3 normalisation matrix
/// `T` (such that `x_norm = T * [x, y, 1]^T`) and the normalised points.
fn hartley_normalise(pts: &[Point2]) -> (Matrix3<f64>, Vec<Point2>) {
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p.x).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p.y).sum::<f64>() / n;

    let mean_dist = pts
        .iter()
        .map(|p| ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt())
        .sum::<f64>()
        / n;

    let scale = if mean_dist > 1e-10 {
        std::f64::consts::SQRT_2 / mean_dist
    } else {
        1.0
    };

    #[rustfmt::skip]
    let t = Matrix3::new(
        scale, 0.0,   -scale * cx,
        0.0,   scale, -scale * cy,
        0.0,   0.0,    1.0,
    );

    let normalised = pts
        .iter()
        .map(|p| Point2 {
            x: scale * (p.x - cx),
            y: scale * (p.y - cy),
        })
        .collect();

    (t, normalised)
}

/// Compute the Sampson (first-order epipolar) distance for a correspondence.
///
/// A value below a small threshold (e.g. `1e-3`) indicates an inlier.
fn sampson_distance(e: &Matrix3<f64>, p1: &Point2, p2: &Point2) -> f64 {
    let x1 = Vector3::new(p1.x, p1.y, 1.0);
    let x2 = Vector3::new(p2.x, p2.y, 1.0);

    let ex1 = e * x1;
    let etx2 = e.transpose() * x2;

    let numer = x2.dot(&ex1).powi(2);
    let denom = ex1[0].powi(2) + ex1[1].powi(2) + etx2[0].powi(2) + etx2[1].powi(2);

    if denom < 1e-20 {
        f64::MAX
    } else {
        numer / denom
    }
}

/// Count correspondence pairs whose triangulated 3D point has positive depth
/// in both camera frames.
///
/// For camera 1 (at origin) a positive depth simply means `Z > 0`.
/// For camera 2 (`[R | t]`) depth is `(R·X + t).z > 0`.
/// This function approximates this check by solving the 2×2 linear system
/// arising from the `x`- and `z`-rows of the constraint `R·X ≈ λ₂·x₂ - t`.
fn count_positive_depth(
    pts1: &[Point2],
    pts2: &[Point2],
    r: &Matrix3<f64>,
    t: &Vector3<f64>,
) -> usize {
    pts1.iter()
        .zip(pts2)
        .filter(|(p1, p2)| positive_depth_check(p1, p2, r, t))
        .count()
}

/// Return `true` when both depths `(λ₁, λ₂)` are positive.
///
/// Solves the 2×2 system from the `x`- and `z`-components of:
/// `λ₁ · R · x̂₁ − λ₂ · x̂₂ = −t`
fn positive_depth_check(p1: &Point2, p2: &Point2, r: &Matrix3<f64>, t: &Vector3<f64>) -> bool {
    let x1 = Vector3::new(p1.x, p1.y, 1.0);
    let x2 = Vector3::new(p2.x, p2.y, 1.0);
    let rx1 = r * x1;

    // Use x (row 0) and z (row 2) equations.
    let a00 = rx1[0];
    let a01 = -x2[0];
    let a10 = rx1[2];
    let a11 = -x2[2];
    let b0 = -t[0];
    let b1 = -t[2];

    let det = a00 * a11 - a01 * a10;
    if det.abs() < 1e-10 {
        return false;
    }

    let d1 = (b0 * a11 - b1 * a01) / det;
    let d2 = (a00 * b1 - a10 * b0) / det;

    d1 > 0.0 && d2 > 0.0
}

// ─── PRNG helpers ─────────────────────────────────────────────────────────────

/// Minimal linear congruential generator used for RANSAC sampling.
/// Avoids a dependency on the `rand` crate.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn next_bounded(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Sample 8 unique indices from `[0, n)` using a partial Fisher–Yates shuffle.
fn sample_8(rng: &mut Lcg, n: usize) -> [usize; 8] {
    let mut buf: Vec<usize> = (0..n).collect();
    for i in 0..8 {
        let j = i + rng.next_bounded(n - i);
        buf.swap(i, j);
    }
    let mut out = [0usize; 8];
    out.copy_from_slice(&buf[..8]);
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hartley_centroid_zero() {
        let pts = vec![
            Point2::new(1.0, 0.0),
            Point2::new(-1.0, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(0.0, -1.0),
        ];
        let (_t, norm) = hartley_normalise(&pts);
        let cx = norm.iter().map(|p| p.x).sum::<f64>() / norm.len() as f64;
        let cy = norm.iter().map(|p| p.y).sum::<f64>() / norm.len() as f64;
        assert!(
            cx.abs() < 1e-10,
            "centroid x after normalisation must be ~0"
        );
        assert!(
            cy.abs() < 1e-10,
            "centroid y after normalisation must be ~0"
        );
    }

    #[test]
    fn test_sampson_perfect_epipolar() {
        // For a camera translated along Z, E = [[0,0,0],[0,0,-1],[0,1,0]].
        // Points on the same horizontal epipolar line satisfy x2^T E x1 = 0.
        #[rustfmt::skip]
        let e = Matrix3::new(
            0.0, 0.0,  0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0,  0.0,
        );
        let p1 = Point2::new(0.0, 0.0);
        let p2 = Point2::new(0.0, 0.0);
        assert!(
            sampson_distance(&e, &p1, &p2) < 1e-10,
            "perfect epipolar pair must have ~0 Sampson distance"
        );
    }

    #[test]
    fn test_lcg_bounded() {
        let mut rng = Lcg::new(42);
        for _ in 0..1_000 {
            assert!(rng.next_bounded(10) < 10);
        }
    }

    #[test]
    fn test_sample_8_unique_in_bounds() {
        let mut rng = Lcg::new(99);
        let sample = sample_8(&mut rng, 30);
        let mut seen = std::collections::HashSet::new();
        for &idx in &sample {
            assert!(idx < 30, "index {idx} out of bounds");
            assert!(seen.insert(idx), "duplicate index {idx}");
        }
    }

    #[test]
    fn test_positive_depth_pure_translation() {
        // Camera 2 is translated +1 along z from camera 1 (no rotation).
        // A world point at (1, 0, 5):
        //   cam1 projection: (1/5, 0/5) = (0.2, 0.0)
        //   cam2 position  : R·X + t = (1, 0, 5) + (0, 0, 1) = (1, 0, 6)
        //   cam2 projection: (1/6, 0/6) ≈ (0.1667, 0.0)
        // Both depths λ₁=5 and λ₂=6 are positive → should return true.
        let r = Matrix3::identity();
        let t = Vector3::new(0.0, 0.0, 1.0);
        let p1 = Point2::new(0.2, 0.0);
        let p2 = Point2::new(1.0 / 6.0, 0.0);
        assert!(
            positive_depth_check(&p1, &p2, &r, &t),
            "point in front of both cameras must have positive depth"
        );
    }
}
