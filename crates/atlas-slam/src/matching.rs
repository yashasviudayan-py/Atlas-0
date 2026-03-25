//! Feature matching for visual SLAM.
//!
//! Implements brute-force nearest-neighbour search in Hamming space combined
//! with Lowe's ratio test to find reliable correspondences between two sets of
//! BRIEF binary descriptors.

use crate::features::{Descriptor, KeyPoint};
use nalgebra::{DMatrix, Matrix3, Vector3};

// ─── DMatch ──────────────────────────────────────────────────────────────────

/// A verified correspondence between a query and a train descriptor.
#[derive(Debug, Clone)]
pub struct DMatch {
    /// Index into the query descriptor array.
    pub query_idx: usize,
    /// Index into the train descriptor array.
    pub train_idx: usize,
    /// Hamming distance between the matched descriptors.
    pub distance: u32,
}

// ─── FeatureMatcher ──────────────────────────────────────────────────────────

/// Matches BRIEF binary descriptors between two frames.
///
/// Uses brute-force nearest-neighbour search (O(n·m) in the number of
/// descriptors) followed by Lowe's ratio test to discard ambiguous matches,
/// and an absolute distance threshold to discard poor matches.
///
/// # Examples
/// ```
/// use atlas_slam::matching::FeatureMatcher;
/// let matcher = FeatureMatcher::new(0.75, 80);
/// ```
pub struct FeatureMatcher {
    /// Lowe's ratio test threshold (default: 0.75).
    ratio_threshold: f32,
    /// Absolute maximum Hamming distance accepted (out of 256).
    max_distance: u32,
}

impl FeatureMatcher {
    /// Create a new feature matcher.
    ///
    /// # Arguments
    /// * `ratio_threshold` — Ratio of best-to-second-best distance below which
    ///   a match is accepted.  Lower values → fewer but more reliable matches.
    ///   Typical value: 0.75 (Lowe 2004).
    /// * `max_distance` — Absolute maximum Hamming distance.  Matches further
    ///   apart than this are discarded even if they pass the ratio test.
    ///
    /// # Examples
    /// ```
    /// use atlas_slam::matching::FeatureMatcher;
    /// let matcher = FeatureMatcher::new(0.75, 80);
    /// ```
    #[must_use]
    pub fn new(ratio_threshold: f32, max_distance: u32) -> Self {
        Self {
            ratio_threshold,
            max_distance,
        }
    }

    /// Match `query` descriptors against `train` descriptors.
    ///
    /// For each query descriptor the nearest and second-nearest train
    /// descriptors are found.  A match is returned only when:
    /// 1. The nearest distance ≤ `max_distance`.
    /// 2. `nearest / second_nearest < ratio_threshold` (Lowe's ratio test).
    ///
    /// When there is only one train descriptor the ratio test is skipped.
    ///
    /// Returns an unsorted `Vec<DMatch>` with one entry per accepted query
    /// descriptor.
    #[must_use]
    pub fn match_descriptors(&self, query: &[Descriptor], train: &[Descriptor]) -> Vec<DMatch> {
        if query.is_empty() || train.is_empty() {
            return Vec::new();
        }

        query
            .iter()
            .enumerate()
            .filter_map(|(q_idx, q_desc)| self.find_best(q_idx, q_desc, train))
            .collect()
    }

    /// Filter an existing match set using RANSAC fundamental-matrix estimation.
    ///
    /// Estimates the fundamental matrix **F** using the normalised 8-point
    /// algorithm inside a RANSAC loop.  Returns only the matches whose
    /// Sampson distance to **F** is below `threshold`.
    ///
    /// When fewer than 8 matches are supplied the input slice is returned
    /// unchanged (not enough points to fit a fundamental matrix).
    ///
    /// # Arguments
    /// * `matches` — Initial match set (e.g. from [`Self::match_descriptors`]).
    /// * `query_kps` — Keypoints of the query (current) frame.
    /// * `train_kps` — Keypoints of the train (reference) frame.
    /// * `ransac_iterations` — Number of RANSAC trials.
    /// * `threshold` — Sampson-distance inlier threshold in pixel² units
    ///   (typical value: 1.0–4.0).
    #[must_use]
    pub fn filter_matches_ransac(
        &self,
        matches: &[DMatch],
        query_kps: &[KeyPoint],
        train_kps: &[KeyPoint],
        ransac_iterations: usize,
        threshold: f64,
    ) -> Vec<DMatch> {
        if matches.len() < 8 {
            return matches.to_vec();
        }

        let pts_q: Vec<[f64; 2]> = matches
            .iter()
            .map(|m| {
                [
                    query_kps[m.query_idx].x as f64,
                    query_kps[m.query_idx].y as f64,
                ]
            })
            .collect();
        let pts_t: Vec<[f64; 2]> = matches
            .iter()
            .map(|m| {
                [
                    train_kps[m.train_idx].x as f64,
                    train_kps[m.train_idx].y as f64,
                ]
            })
            .collect();

        let n = matches.len();
        let mut rng = Lcg::new(0xCAFE_BABE_DEAD_BEEF_u64);
        let mut best_inliers: Vec<bool> = vec![false; n];
        let mut best_count = 0usize;

        for _ in 0..ransac_iterations {
            let idx = sample_8(&mut rng, n);
            let s1: Vec<[f64; 2]> = idx.iter().map(|&i| pts_q[i]).collect();
            let s2: Vec<[f64; 2]> = idx.iter().map(|&i| pts_t[i]).collect();

            let Some(f) = compute_fundamental_matrix(&s1, &s2) else {
                continue;
            };

            let inliers: Vec<bool> = pts_q
                .iter()
                .zip(&pts_t)
                .map(|(p1, p2)| sampson_distance_px(&f, p1, p2) < threshold)
                .collect();
            let count = inliers.iter().filter(|&&b| b).count();

            if count > best_count {
                best_count = count;
                best_inliers = inliers;
            }
        }

        matches
            .iter()
            .zip(&best_inliers)
            .filter_map(|(m, &keep)| keep.then_some(m.clone()))
            .collect()
    }

    // ── Private helper ────────────────────────────────────────────────────────

    /// Find the best match for `q_desc` in `train`, applying both filters.
    fn find_best(&self, q_idx: usize, q_desc: &Descriptor, train: &[Descriptor]) -> Option<DMatch> {
        let mut best_dist = u32::MAX;
        let mut best_idx = 0usize;
        let mut second_dist = u32::MAX;

        for (t_idx, t_desc) in train.iter().enumerate() {
            let dist = q_desc.hamming_distance(t_desc);
            if dist < best_dist {
                second_dist = best_dist;
                best_dist = dist;
                best_idx = t_idx;
            } else if dist < second_dist {
                second_dist = dist;
            }
        }

        // Absolute distance filter.
        if best_dist > self.max_distance {
            return None;
        }

        // Ratio test (skip when there is only one candidate).
        if train.len() > 1 {
            let ratio = best_dist as f32 / second_dist.max(1) as f32;
            if ratio >= self.ratio_threshold {
                return None;
            }
        }

        Some(DMatch {
            query_idx: q_idx,
            train_idx: best_idx,
            distance: best_dist,
        })
    }
}

// ─── Fundamental matrix helpers ──────────────────────────────────────────────

/// Compute the fundamental matrix from ≥8 pixel-coordinate correspondences
/// using the normalised 8-point algorithm.
///
/// Returns `None` if the computation is numerically degenerate.
fn compute_fundamental_matrix(pts1: &[[f64; 2]], pts2: &[[f64; 2]]) -> Option<Matrix3<f64>> {
    let n = pts1.len();
    if n < 8 {
        return None;
    }

    let (t1, norm1) = hartley_normalise_px(pts1);
    let (t2, norm2) = hartley_normalise_px(pts2);

    // Build epipolar constraint matrix A (n × 9).
    let mut a_data = Vec::with_capacity(n * 9);
    for (p1, p2) in norm1.iter().zip(&norm2) {
        a_data.extend_from_slice(&[
            p2[0] * p1[0],
            p2[0] * p1[1],
            p2[0],
            p2[1] * p1[0],
            p2[1] * p1[1],
            p2[1],
            p1[0],
            p1[1],
            1.0_f64,
        ]);
    }

    let a = DMatrix::from_row_slice(n, 9, &a_data);
    let svd = a.svd(false, true);
    let vt = svd.v_t?;

    // Null-space vector = last row of V^T (smallest singular value).
    let last_row: Vec<f64> = vt.row(vt.nrows() - 1).iter().copied().collect();
    let f_norm = Matrix3::from_row_slice(&last_row);

    // Enforce rank-2 constraint: set smallest singular value to 0.
    let svd_f = f_norm.svd(true, true);
    let fu = svd_f.u?;
    let fvt = svd_f.v_t?;
    let mut sigma = svd_f.singular_values;
    sigma[2] = 0.0;
    let f_rank2 = fu * Matrix3::from_diagonal(&sigma) * fvt;

    // Denormalise: F = T2^T · F̂ · T1
    Some(t2.transpose() * f_rank2 * t1)
}

/// Hartley normalisation for pixel-coordinate point sets.
///
/// Translates points to zero centroid and scales so the mean distance from
/// the origin is √2.  Returns the 3×3 normalisation matrix `T` and the
/// normalised points.
fn hartley_normalise_px(pts: &[[f64; 2]]) -> (Matrix3<f64>, Vec<[f64; 2]>) {
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p[0]).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p[1]).sum::<f64>() / n;

    let mean_dist = pts
        .iter()
        .map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt())
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
        .map(|p| [scale * (p[0] - cx), scale * (p[1] - cy)])
        .collect();

    (t, normalised)
}

/// Sampson (first-order epipolar) distance for a pixel-coordinate pair.
///
/// Values below a small threshold (e.g. 1.0) indicate inliers.
fn sampson_distance_px(f: &Matrix3<f64>, p1: &[f64; 2], p2: &[f64; 2]) -> f64 {
    let x1 = Vector3::new(p1[0], p1[1], 1.0);
    let x2 = Vector3::new(p2[0], p2[1], 1.0);

    let fx1 = f * x1;
    let ftx2 = f.transpose() * x2;

    let numer = x2.dot(&fx1).powi(2);
    let denom = fx1[0].powi(2) + fx1[1].powi(2) + ftx2[0].powi(2) + ftx2[1].powi(2);

    if denom < 1e-20 {
        f64::MAX
    } else {
        numer / denom
    }
}

// ─── PRNG helpers ─────────────────────────────────────────────────────────────

/// Minimal LCG used for RANSAC sampling (avoids `rand` dependency).
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

    fn desc(byte: u8) -> Descriptor {
        Descriptor([byte; 32])
    }

    #[test]
    fn test_match_identical_descriptor() {
        let matcher = FeatureMatcher::new(0.75, 80);
        let query = vec![desc(0xAA)];
        let train = vec![desc(0xAA), desc(0xFF)];

        let matches = matcher.match_descriptors(&query, &train);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].query_idx, 0);
        assert_eq!(matches[0].train_idx, 0);
        assert_eq!(matches[0].distance, 0);
    }

    #[test]
    fn test_match_empty_input() {
        let matcher = FeatureMatcher::new(0.75, 80);
        assert!(matcher.match_descriptors(&[], &[desc(0)]).is_empty());
        assert!(matcher.match_descriptors(&[desc(0)], &[]).is_empty());
    }

    #[test]
    fn test_ratio_test_rejects_ambiguous() {
        let matcher = FeatureMatcher::new(0.75, 80);
        // query = 0x00; train has two descriptors each 1 bit away → ratio = 1.0 → reject.
        let query = vec![desc(0x00)];
        let train = vec![desc(0x01), desc(0x02)];
        // Hamming(0x00, 0x01) = 32 bits × 1 bit/byte = 32 (one bit set per byte)
        // Hamming(0x00, 0x02) = 32 (one different bit per byte)
        // ratio = 32/32 = 1.0 ≥ 0.75 → rejected
        let matches = matcher.match_descriptors(&query, &train);
        assert!(matches.is_empty(), "ambiguous match should be rejected");
    }

    #[test]
    fn test_max_distance_filter() {
        let matcher = FeatureMatcher::new(0.75, 10);
        // Hamming(0x00, 0xFF) = 256 > 10.
        let matches = matcher.match_descriptors(&[desc(0x00)], &[desc(0xFF)]);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_single_train_skips_ratio_test() {
        let matcher = FeatureMatcher::new(0.75, 80);
        // Only one train descriptor → ratio test is skipped; accepted if distance ≤ max.
        let query = vec![desc(0x00)];
        let train = vec![desc(0x01)]; // distance = 32 (one bit per byte × 32 bytes)
        let matches = matcher.match_descriptors(&query, &train);
        assert_eq!(matches.len(), 1);
    }

    // ── RANSAC filter tests ────────────────────────────────────────────────

    fn make_kp(x: f32, y: f32) -> KeyPoint {
        KeyPoint {
            x,
            y,
            response: 1.0,
        }
    }

    #[test]
    fn test_ransac_filter_too_few_matches_returns_all() {
        let matcher = FeatureMatcher::new(0.75, 80);
        // 7 matches — below the 8-point minimum; all should be returned unchanged.
        let n = 7usize;
        let query_kps: Vec<KeyPoint> = (0..n).map(|i| make_kp(i as f32 * 10.0, 100.0)).collect();
        let train_kps: Vec<KeyPoint> = (0..n)
            .map(|i| make_kp(i as f32 * 10.0 + 5.0, 100.0))
            .collect();
        let matches: Vec<DMatch> = (0..n)
            .map(|i| DMatch {
                query_idx: i,
                train_idx: i,
                distance: 0,
            })
            .collect();

        let filtered = matcher.filter_matches_ransac(&matches, &query_kps, &train_kps, 200, 1.0);
        assert_eq!(
            filtered.len(),
            n,
            "all matches must be returned when fewer than 8 are supplied"
        );
    }

    #[test]
    fn test_ransac_filter_consistent_matches_are_kept() {
        // Synthetic scene: camera 2 is translated by tx=30 px along X relative
        // to camera 1 (same focal length / principal point).  Correspondences
        // satisfy the epipolar geometry exactly so they must all be inliers.
        let fx = 525.0f64;
        let fy = 525.0f64;
        let cx = 320.0f64;
        let cy = 240.0f64;
        let tx = 0.5f64; // world-space translation (metres)

        let world_pts: &[(f64, f64, f64)] = &[
            (0.5, 0.3, 5.0),
            (-0.2, 0.4, 4.5),
            (0.1, -0.3, 6.0),
            (-0.4, -0.2, 5.5),
            (0.3, 0.1, 4.0),
            (-0.1, 0.5, 7.0),
            (0.6, -0.4, 5.0),
            (-0.5, 0.2, 6.0),
            (0.2, 0.6, 4.5),
            (-0.3, -0.5, 5.5),
            (0.4, 0.3, 8.0),
            (-0.6, 0.1, 6.5),
        ];

        let mut query_kps = Vec::new();
        let mut train_kps = Vec::new();
        let mut matches = Vec::new();

        for &(x, y, z) in world_pts {
            let u1 = fx * x / z + cx;
            let v1 = fy * y / z + cy;
            let u2 = fx * (x - tx) / z + cx;
            let v2 = fy * y / z + cy;
            let i = query_kps.len();
            query_kps.push(make_kp(u1 as f32, v1 as f32));
            train_kps.push(make_kp(u2 as f32, v2 as f32));
            matches.push(DMatch {
                query_idx: i,
                train_idx: i,
                distance: 0,
            });
        }

        let matcher = FeatureMatcher::new(0.75, 80);
        let filtered = matcher.filter_matches_ransac(&matches, &query_kps, &train_kps, 500, 2.0);

        // All 12 correspondences satisfy the epipolar geometry exactly, so at
        // least 10 of them must survive RANSAC.
        assert!(
            filtered.len() >= 10,
            "expected ≥10 inliers from 12 geometrically consistent matches, got {}",
            filtered.len()
        );
    }

    #[test]
    fn test_ransac_filter_rejects_gross_outliers() {
        // 12 consistent matches + 4 gross outliers.
        // The filter must retain most consistent ones and discard the outliers.
        let fx = 525.0f64;
        let fy = 525.0f64;
        let cx = 320.0f64;
        let cy = 240.0f64;
        let tx = 0.5f64;

        let world_pts: &[(f64, f64, f64)] = &[
            (0.5, 0.3, 5.0),
            (-0.2, 0.4, 4.5),
            (0.1, -0.3, 6.0),
            (-0.4, -0.2, 5.5),
            (0.3, 0.1, 4.0),
            (-0.1, 0.5, 7.0),
            (0.6, -0.4, 5.0),
            (-0.5, 0.2, 6.0),
            (0.2, 0.6, 4.5),
            (-0.3, -0.5, 5.5),
            (0.4, 0.3, 8.0),
            (-0.6, 0.1, 6.5),
        ];

        let mut query_kps = Vec::new();
        let mut train_kps = Vec::new();
        let mut matches = Vec::new();

        for &(x, y, z) in world_pts {
            let u1 = fx * x / z + cx;
            let v1 = fy * y / z + cy;
            let u2 = fx * (x - tx) / z + cx;
            let v2 = fy * y / z + cy;
            let i = query_kps.len();
            query_kps.push(make_kp(u1 as f32, v1 as f32));
            train_kps.push(make_kp(u2 as f32, v2 as f32));
            matches.push(DMatch {
                query_idx: i,
                train_idx: i,
                distance: 0,
            });
        }

        // Add 4 gross outliers: random large displacements.
        for k in 0..4usize {
            let i = query_kps.len();
            query_kps.push(make_kp(50.0 + k as f32 * 100.0, 50.0));
            train_kps.push(make_kp(500.0 - k as f32 * 80.0, 400.0)); // wildly inconsistent
            matches.push(DMatch {
                query_idx: i,
                train_idx: i,
                distance: 5,
            });
        }

        let matcher = FeatureMatcher::new(0.75, 80);
        let filtered = matcher.filter_matches_ransac(&matches, &query_kps, &train_kps, 500, 2.0);

        // At least the 12 inliers should be kept; the 4 outliers should mostly be gone.
        assert!(
            filtered.len() >= 10,
            "expected ≥10 inliers to survive, got {}",
            filtered.len()
        );
        assert!(
            filtered.len() <= 14,
            "expected at most 14 matches (all inliers + few outliers), got {}",
            filtered.len()
        );
    }

    #[test]
    fn test_multiple_queries() {
        let matcher = FeatureMatcher::new(0.75, 80);
        let query = vec![desc(0x00), desc(0xFF)];
        let train = vec![desc(0x00), desc(0xFF), desc(0x0F)];
        let matches = matcher.match_descriptors(&query, &train);
        // Each query should match its identical counterpart.
        let q0 = matches.iter().find(|m| m.query_idx == 0);
        let q1 = matches.iter().find(|m| m.query_idx == 1);
        assert!(q0.is_some_and(|m| m.train_idx == 0 && m.distance == 0));
        assert!(q1.is_some_and(|m| m.train_idx == 1 && m.distance == 0));
    }
}
