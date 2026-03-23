//! Feature matching for visual SLAM.
//!
//! Implements brute-force nearest-neighbour search in Hamming space combined
//! with Lowe's ratio test to find reliable correspondences between two sets of
//! BRIEF binary descriptors.

use crate::features::Descriptor;

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
