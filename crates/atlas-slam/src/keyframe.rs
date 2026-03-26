//! Keyframe graph with co-visibility information.
//!
//! A *keyframe* is a selected camera frame that anchors the Gaussian map.
//! The [`KeyframeGraph`] decides when to insert new keyframes (based on
//! camera motion thresholds) and tracks which pairs of keyframes share
//! visible features — the *co-visibility* relationship used to schedule
//! local bundle-adjustment windows and Gaussian densification.

use atlas_core::Pose;

use crate::features::{Descriptor, KeyPoint};

// ─── Types ────────────────────────────────────────────────────────────────────

/// A unique identifier for a keyframe within a [`KeyframeGraph`].
pub type KeyframeId = u64;

/// All data stored for a single keyframe.
pub struct KeyframeData {
    /// Unique identifier assigned by the graph.
    pub id: KeyframeId,
    /// Camera-to-world pose at the time this keyframe was selected.
    pub pose: Pose,
    /// Keypoints extracted from the frame.
    pub keypoints: Vec<KeyPoint>,
    /// Corresponding BRIEF descriptors (parallel to `keypoints`).
    pub descriptors: Vec<Descriptor>,
    /// Number of Gaussians in the map when this keyframe was inserted.
    pub gaussian_count: usize,
}

/// A directed co-visibility edge: this keyframe shares features with
/// `other_id`.
pub struct CoVisibilityEdge {
    /// The other keyframe in this co-visibility relationship.
    pub other_id: KeyframeId,
    /// Number of descriptor matches shared between the two keyframes.
    pub shared_feature_count: usize,
}

// ─── KeyframeGraph ────────────────────────────────────────────────────────────

/// Manages a collection of keyframes and their co-visibility relationships.
///
/// Keyframe insertion is gated by translation and rotation thresholds from
/// the [`crate::config::SlamConfig`].  When a keyframe is inserted the graph
/// automatically computes co-visibility edges to recent keyframes so that
/// Gaussian densification can prioritise well-observed regions.
pub struct KeyframeGraph {
    keyframes: Vec<KeyframeData>,
    /// `edges[i]` stores co-visibility edges for `keyframes[i]`.
    edges: Vec<Vec<CoVisibilityEdge>>,
    /// Pose at the time of the last keyframe insertion.
    last_keyframe_pose: Option<Pose>,
    translation_threshold: f32,
    rotation_threshold: f32,
    next_id: KeyframeId,
}

impl KeyframeGraph {
    /// Create an empty graph using the thresholds from `config`.
    #[must_use]
    pub fn new(translation_threshold: f32, rotation_threshold: f32) -> Self {
        Self {
            keyframes: Vec::new(),
            edges: Vec::new(),
            last_keyframe_pose: None,
            translation_threshold,
            rotation_threshold,
            next_id: 0,
        }
    }

    // ── Keyframe insertion ────────────────────────────────────────────────────

    /// Return `true` when the camera at `pose` is far enough from the last
    /// keyframe to warrant inserting a new one.
    #[must_use]
    pub fn should_insert(&self, pose: &Pose) -> bool {
        match &self.last_keyframe_pose {
            None => true,
            Some(kf_pose) => {
                translation_distance(kf_pose, pose) >= self.translation_threshold
                    || rotation_distance(kf_pose, pose) >= self.rotation_threshold
            }
        }
    }

    /// Insert a new keyframe and return its assigned [`KeyframeId`].
    ///
    /// Co-visibility edges to the most recent keyframes are computed
    /// automatically using descriptor matching.
    pub fn insert(
        &mut self,
        pose: Pose,
        keypoints: Vec<KeyPoint>,
        descriptors: Vec<Descriptor>,
        gaussian_count: usize,
    ) -> KeyframeId {
        let id = self.next_id;
        self.next_id += 1;

        // Compute backward co-visibility edges to up to the last 10 keyframes.
        let window_start = self.keyframes.len().saturating_sub(10);
        let mut backward_edges_for_new: Vec<CoVisibilityEdge> = Vec::new();

        for i in window_start..self.keyframes.len() {
            let shared = count_shared_features(&descriptors, &self.keyframes[i].descriptors);
            if shared >= 3 {
                // Forward edge: existing keyframe → new keyframe.
                self.edges[i].push(CoVisibilityEdge {
                    other_id: id,
                    shared_feature_count: shared,
                });
                // Backward edge: new keyframe → existing keyframe.
                backward_edges_for_new.push(CoVisibilityEdge {
                    other_id: self.keyframes[i].id,
                    shared_feature_count: shared,
                });
            }
        }

        self.keyframes.push(KeyframeData {
            id,
            pose,
            keypoints,
            descriptors,
            gaussian_count,
        });
        self.edges.push(backward_edges_for_new);
        self.last_keyframe_pose = Some(pose);

        id
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Return references to all keyframes that share features with `id`.
    #[must_use]
    pub fn get_covisible(&self, id: KeyframeId) -> Vec<&KeyframeData> {
        let Some(idx) = self.keyframes.iter().position(|kf| kf.id == id) else {
            return Vec::new();
        };
        self.edges[idx]
            .iter()
            .filter_map(|e| self.keyframes.iter().find(|kf| kf.id == e.other_id))
            .collect()
    }

    /// Number of keyframes currently in the graph.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    /// Return `true` when the graph contains no keyframes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    /// Immutable access to the ordered list of keyframes.
    #[must_use]
    pub fn keyframes(&self) -> &[KeyframeData] {
        &self.keyframes
    }

    /// The most recently inserted keyframe, or `None` if the graph is empty.
    #[must_use]
    pub fn latest(&self) -> Option<&KeyframeData> {
        self.keyframes.last()
    }

    // ── Test helpers ──────────────────────────────────────────────────────────

    #[cfg(test)]
    pub(crate) fn clear(&mut self) {
        self.keyframes.clear();
        self.edges.clear();
        self.last_keyframe_pose = None;
        self.next_id = 0;
    }

    #[cfg(test)]
    pub(crate) fn set_last_keyframe_pose(&mut self, pose: Option<Pose>) {
        self.last_keyframe_pose = pose;
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Count how many descriptors in `a` have a close match (Hamming < 50) in `b`.
fn count_shared_features(a: &[Descriptor], b: &[Descriptor]) -> usize {
    const THRESHOLD: u32 = 50;
    let mut count = 0usize;
    for da in a {
        for db in b {
            if da.hamming_distance(db) < THRESHOLD {
                count += 1;
                break;
            }
        }
    }
    count
}

/// Euclidean distance between the translation parts of two poses (metres).
fn translation_distance(a: &Pose, b: &Pose) -> f32 {
    let dx = b.position.x - a.position.x;
    let dy = b.position.y - a.position.y;
    let dz = b.position.z - a.position.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Geodesic angle (radians) between the rotation components of two poses.
fn rotation_distance(a: &Pose, b: &Pose) -> f32 {
    let dot = a.rotation[0] * b.rotation[0]
        + a.rotation[1] * b.rotation[1]
        + a.rotation[2] * b.rotation[2]
        + a.rotation[3] * b.rotation[3];
    2.0 * dot.abs().min(1.0).acos()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::Pose;

    fn empty_graph() -> KeyframeGraph {
        KeyframeGraph::new(0.1, 0.1)
    }

    fn make_keyframe_data(id_hint: u64, x: f32) -> (Pose, Vec<KeyPoint>, Vec<Descriptor>, usize) {
        let pose = Pose {
            position: atlas_core::Point3::new(x, 0.0, 0.0),
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        let kps: Vec<KeyPoint> = Vec::new();
        let descs: Vec<Descriptor> = Vec::new();
        let _ = id_hint;
        (pose, kps, descs, 0)
    }

    // ── should_insert ─────────────────────────────────────────────────────────

    #[test]
    fn test_should_insert_when_empty() {
        let graph = empty_graph();
        assert!(graph.should_insert(&Pose::identity()));
    }

    #[test]
    fn test_should_insert_after_large_translation() {
        let mut graph = empty_graph(); // threshold = 0.1 m
        let (pose, kps, descs, n) = make_keyframe_data(0, 0.0);
        graph.insert(pose, kps, descs, n);

        let far_pose = Pose {
            position: atlas_core::Point3::new(1.0, 0.0, 0.0),
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        assert!(graph.should_insert(&far_pose));
    }

    #[test]
    fn test_should_not_insert_for_small_motion() {
        let mut graph = empty_graph(); // threshold = 0.1 m
        let (pose, kps, descs, n) = make_keyframe_data(0, 0.0);
        graph.insert(pose, kps, descs, n);

        let close_pose = Pose {
            position: atlas_core::Point3::new(0.01, 0.0, 0.0),
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        assert!(!graph.should_insert(&close_pose));
    }

    // ── insert ────────────────────────────────────────────────────────────────

    #[test]
    fn test_insert_increments_count() {
        let mut graph = empty_graph();
        assert_eq!(graph.len(), 0);
        let (pose, kps, descs, n) = make_keyframe_data(0, 0.0);
        graph.insert(pose, kps, descs, n);
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn test_insert_assigns_sequential_ids() {
        let mut graph = empty_graph();
        let (p1, k1, d1, n1) = make_keyframe_data(0, 0.0);
        let (p2, k2, d2, n2) = make_keyframe_data(1, 1.0);
        let id1 = graph.insert(p1, k1, d1, n1);
        let id2 = graph.insert(p2, k2, d2, n2);
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
    }

    #[test]
    fn test_latest_returns_last_inserted() {
        let mut graph = empty_graph();
        assert!(graph.latest().is_none());
        let (pose, kps, descs, n) = make_keyframe_data(0, 42.0);
        graph.insert(pose, kps, descs, n);
        let latest = graph.latest().unwrap();
        assert!((latest.pose.position.x - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_insert_updates_last_keyframe_pose() {
        let mut graph = empty_graph();
        let (p1, k1, d1, n1) = make_keyframe_data(0, 0.0);
        graph.insert(p1, k1, d1, n1);
        // After inserting at x=0, the close pose (x=0.01) should NOT trigger another insert.
        let close_pose = Pose {
            position: atlas_core::Point3::new(0.01, 0.0, 0.0),
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        assert!(!graph.should_insert(&close_pose));
    }

    // ── co-visibility ──────────────────────────────────────────────────────────

    #[test]
    fn test_covisible_unknown_id_returns_empty() {
        let graph = empty_graph();
        let result = graph.get_covisible(999);
        assert!(result.is_empty());
    }

    #[test]
    fn test_covisible_no_shared_features() {
        // Insert two keyframes with no matching descriptors.
        let mut graph = empty_graph();
        let (p1, k1, d1, n1) = make_keyframe_data(0, 0.0);
        let (p2, k2, d2, n2) = make_keyframe_data(1, 1.0);
        let id1 = graph.insert(p1, k1, d1, n1);
        graph.insert(p2, k2, d2, n2);
        // No descriptors → no co-visibility.
        assert!(graph.get_covisible(id1).is_empty());
    }

    // ── clear / set_last_keyframe_pose (test helpers) ─────────────────────────

    #[test]
    fn test_clear_empties_graph() {
        let mut graph = empty_graph();
        let (pose, kps, descs, n) = make_keyframe_data(0, 0.0);
        graph.insert(pose, kps, descs, n);
        assert_eq!(graph.len(), 1);
        graph.clear();
        assert!(graph.is_empty());
    }

    #[test]
    fn test_set_last_keyframe_pose_none_triggers_insert() {
        let mut graph = empty_graph();
        // Pretend there was a prior keyframe so the default would block insertion.
        let (pose, kps, descs, n) = make_keyframe_data(0, 0.0);
        graph.insert(pose, kps, descs, n);
        // Reset — as if the graph had been freshly cleared.
        graph.set_last_keyframe_pose(None);
        assert!(graph.should_insert(&Pose::identity()));
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    #[test]
    fn test_translation_distance_zero() {
        let p = Pose::identity();
        assert!(translation_distance(&p, &p) < 1e-6);
    }

    #[test]
    fn test_translation_distance_known() {
        let a = Pose::identity();
        let b = Pose {
            position: atlas_core::Point3::new(3.0, 4.0, 0.0),
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        assert!((translation_distance(&a, &b) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_rotation_distance_identity() {
        let p = Pose::identity();
        assert!(rotation_distance(&p, &p) < 1e-5);
    }
}
