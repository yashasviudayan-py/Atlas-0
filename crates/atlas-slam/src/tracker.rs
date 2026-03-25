//! Camera pose tracker using visual odometry.
//!
//! The [`Tracker`] maintains the SLAM state across frames:
//! - Extracts FAST+BRIEF features from each frame.
//! - Matches them against the previous frame using Hamming-distance BF matching.
//! - Estimates the relative camera pose via the 8-point essential-matrix algorithm.
//! - Accumulates poses so that `current_pose()` always reflects the camera's
//!   estimated position and orientation in world space.

use atlas_core::{Frame, GaussianCloud, Point3, Pose};
use nalgebra::{Matrix3, Matrix4, Rotation3, UnitQuaternion};
use tracing::{info, warn};

use crate::config::SlamConfig;
use crate::features::{Descriptor, FeatureExtractor, KeyPoint};
use crate::matching::FeatureMatcher;
use crate::pose_estimation::PoseEstimator;

// ─── Keyframe ────────────────────────────────────────────────────────────────

/// A stored reference frame used for relocalization.
struct Keyframe {
    /// World-space pose of this keyframe.
    pose: Pose,
    /// Keypoints extracted from the frame.
    keypoints: Vec<KeyPoint>,
    /// Corresponding BRIEF descriptors.
    descriptors: Vec<Descriptor>,
}

// ─── Tracker ─────────────────────────────────────────────────────────────────

/// Tracks camera pose and maintains the 3D Gaussian map.
pub struct Tracker {
    config: SlamConfig,
    current_pose: Pose,
    map: GaussianCloud,
    frame_count: u64,
    is_initialized: bool,
    // Feature pipeline components.
    extractor: FeatureExtractor,
    matcher: FeatureMatcher,
    estimator: PoseEstimator,
    // State from the previous frame.
    prev_keypoints: Vec<KeyPoint>,
    prev_descriptors: Vec<Descriptor>,
    // Keyframe store for relocalization.
    keyframes: Vec<Keyframe>,
    /// Pose at the time of the last keyframe insertion.
    last_keyframe_pose: Option<Pose>,
}

impl Tracker {
    /// Create a new tracker with the given configuration.
    ///
    /// The tracker starts uninitialised; the first call to [`process_frame`]
    /// will capture the reference frame.
    #[must_use]
    pub fn new(config: SlamConfig) -> Self {
        let extractor = FeatureExtractor::new(config.fast_threshold, config.max_features);
        let matcher = FeatureMatcher::new(config.ratio_test_threshold, config.max_match_distance);
        let estimator = PoseEstimator::new(
            config.camera.clone(),
            config.ransac_iterations,
            config.ransac_threshold,
        );

        Self {
            config,
            current_pose: Pose::identity(),
            map: GaussianCloud::new(),
            frame_count: 0,
            is_initialized: false,
            extractor,
            matcher,
            estimator,
            prev_keypoints: Vec::new(),
            prev_descriptors: Vec::new(),
            keyframes: Vec::new(),
            last_keyframe_pose: None,
        }
    }

    /// Process a new frame and update the camera pose estimate.
    ///
    /// Returns the estimated camera pose for this frame (in world space).
    ///
    /// # Errors
    /// - [`SlamError::InsufficientFeatures`] when too few matches are found.
    /// - [`SlamError::TrackingLost`] when pose estimation cannot find a valid
    ///   solution (e.g. too few inliers or degenerate configuration).
    /// - [`SlamError::PoseEstimationFailed`] on a numerical failure in the
    ///   essential-matrix computation.
    pub fn process_frame(&mut self, frame: &Frame) -> crate::Result<Pose> {
        self.frame_count += 1;

        if !self.is_initialized {
            self.initialize(frame)?;
            return Ok(self.current_pose);
        }

        use atlas_core::error::SlamError;

        // 1. Extract features from the current frame.
        let gray = FeatureExtractor::rgb_to_gray(&frame.data, frame.width, frame.height);
        let (kps, descs) = self.extractor.extract(&gray);

        // 2. Match against features from the previous frame.
        let matches = self
            .matcher
            .match_descriptors(&descs, &self.prev_descriptors);

        if matches.len() < self.config.min_features {
            // Too few matches — attempt relocalization before giving up.
            if let Some(recovered_pose) = self.try_relocalize(&kps, &descs, frame.id) {
                self.current_pose = recovered_pose;
                self.prev_keypoints = kps;
                self.prev_descriptors = descs;
                warn!(frame_id = frame.id, "tracking recovered via relocalization");
                return Ok(self.current_pose);
            }
            return Err(SlamError::InsufficientFeatures {
                found: matches.len(),
                required: self.config.min_features,
            });
        }

        // 3. Build (prev_pixel, curr_pixel) correspondence list for PnP/E-matrix.
        //    Note: query = current frame, train = previous frame.
        let correspondences: Vec<([f64; 2], [f64; 2])> = matches
            .iter()
            .map(|m| {
                let curr = &kps[m.query_idx];
                let prev = &self.prev_keypoints[m.train_idx];
                // (previous, current) — the estimator models motion from prev→curr.
                (
                    [prev.x as f64, prev.y as f64],
                    [curr.x as f64, curr.y as f64],
                )
            })
            .collect();

        // 4. Estimate relative pose (prev → current).
        let relative_pose = match self.estimator.estimate_pose(&correspondences, frame.id)? {
            Some(p) => p,
            None => {
                // Insufficient inliers — try relocalization.
                if let Some(recovered_pose) = self.try_relocalize(&kps, &descs, frame.id) {
                    self.current_pose = recovered_pose;
                    self.prev_keypoints = kps;
                    self.prev_descriptors = descs;
                    warn!(frame_id = frame.id, "tracking recovered via relocalization");
                    return Ok(self.current_pose);
                }
                return Err(SlamError::TrackingLost { frame_id: frame.id });
            }
        };

        // 5. Compose with the accumulated world pose.
        self.current_pose = compose_poses(&self.current_pose, &relative_pose);

        // 6. Insert a keyframe if the camera has moved far enough from the last one.
        if self.should_insert_keyframe(&self.current_pose) {
            self.keyframes.push(Keyframe {
                pose: self.current_pose,
                keypoints: kps.clone(),
                descriptors: descs.clone(),
            });
            self.last_keyframe_pose = Some(self.current_pose);
            info!(
                frame_id = frame.id,
                total_keyframes = self.keyframes.len(),
                "keyframe inserted"
            );
        }

        // 7. Store current frame features for the next iteration.
        self.prev_keypoints = kps;
        self.prev_descriptors = descs;

        info!(
            frame_id = frame.id,
            matches = matches.len(),
            gaussians = self.map.len(),
            "processed frame"
        );

        Ok(self.current_pose)
    }

    /// Get the current estimated camera pose (in world space).
    #[must_use]
    pub fn current_pose(&self) -> Pose {
        self.current_pose
    }

    /// Get an immutable reference to the current Gaussian map.
    #[must_use]
    pub fn map(&self) -> &GaussianCloud {
        &self.map
    }

    /// Total number of frames processed so far.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Initialise the SLAM system from the first frame.
    ///
    /// Stores the feature set as the reference for subsequent frames and sets
    /// the initial pose to the world origin.
    fn initialize(&mut self, frame: &Frame) -> crate::Result<()> {
        use atlas_core::error::SlamError;

        info!(
            width = frame.width,
            height = frame.height,
            "initialising SLAM from first frame"
        );

        let gray = FeatureExtractor::rgb_to_gray(&frame.data, frame.width, frame.height);
        let (kps, descs) = self.extractor.extract(&gray);

        if kps.is_empty() {
            return Err(SlamError::InitFailed(
                "no features detected in the first frame — check camera and threshold settings"
                    .into(),
            ));
        }

        info!(features = kps.len(), "first frame features extracted");

        // Store the first frame as an initial keyframe for relocalization.
        self.keyframes.push(Keyframe {
            pose: Pose::identity(),
            keypoints: kps.clone(),
            descriptors: descs.clone(),
        });
        self.last_keyframe_pose = Some(Pose::identity());

        self.prev_keypoints = kps;
        self.prev_descriptors = descs;
        // Initial pose is the world origin.
        self.current_pose = Pose::identity();
        self.is_initialized = true;
        Ok(())
    }

    /// Decide whether to insert a new keyframe at `pose`.
    ///
    /// A keyframe is inserted when:
    /// - No keyframe has been inserted yet, or
    /// - The camera has translated by more than `keyframe_translation_threshold`,
    ///   or rotated by more than `keyframe_rotation_threshold` since the last one.
    fn should_insert_keyframe(&self, pose: &Pose) -> bool {
        match &self.last_keyframe_pose {
            None => true,
            Some(kf_pose) => {
                let dt = translation_distance(kf_pose, pose);
                let dr = rotation_distance(kf_pose, pose);
                dt >= self.config.keyframe_translation_threshold
                    || dr >= self.config.keyframe_rotation_threshold
            }
        }
    }

    /// Attempt to recover tracking by matching `kps`/`descs` against stored
    /// keyframes.
    ///
    /// For each keyframe the current features are matched, and the candidate
    /// with the most matches has its relative pose estimated.  If that
    /// succeeds the recovered world pose is returned; otherwise `None`.
    fn try_relocalize(
        &self,
        kps: &[KeyPoint],
        descs: &[Descriptor],
        frame_id: u64,
    ) -> Option<Pose> {
        // Find the keyframe with the most descriptor matches.
        let (best_kf, best_matches) = self
            .keyframes
            .iter()
            .map(|kf| {
                let m = self.matcher.match_descriptors(descs, &kf.descriptors);
                (kf, m)
            })
            .max_by_key(|(_, m)| m.len())?;

        if best_matches.len() < self.config.min_features {
            return None;
        }

        // 2D–2D pose estimation: keyframe points → current frame points.
        let correspondences: Vec<([f64; 2], [f64; 2])> = best_matches
            .iter()
            .map(|m| {
                let curr = &kps[m.query_idx];
                let kf_kp = &best_kf.keypoints[m.train_idx];
                (
                    [kf_kp.x as f64, kf_kp.y as f64],
                    [curr.x as f64, curr.y as f64],
                )
            })
            .collect();

        let relative_pose = self
            .estimator
            .estimate_pose(&correspondences, frame_id)
            .ok()
            .flatten()?;

        Some(compose_poses(&best_kf.pose, &relative_pose))
    }
}

// ─── Pose composition helpers ─────────────────────────────────────────────────

/// Compose two poses: `result = base ∘ relative`.
///
/// Multiplies the two 4×4 SE(3) transformation matrices and converts the
/// result back to the `Pose` representation.
fn compose_poses(base: &Pose, relative: &Pose) -> Pose {
    let m = base.to_matrix() * relative.to_matrix();
    matrix4_to_pose(&m)
}

/// Extract a [`Pose`] from a 4×4 homogeneous transformation matrix.
fn matrix4_to_pose(m: &Matrix4<f32>) -> Pose {
    let pos = Point3::new(m[(0, 3)], m[(1, 3)], m[(2, 3)]);

    let rot_mat = Matrix3::new(
        m[(0, 0)],
        m[(0, 1)],
        m[(0, 2)],
        m[(1, 0)],
        m[(1, 1)],
        m[(1, 2)],
        m[(2, 0)],
        m[(2, 1)],
        m[(2, 2)],
    );
    let rot = Rotation3::from_matrix_unchecked(rot_mat);
    let uq: UnitQuaternion<f32> = UnitQuaternion::from_rotation_matrix(&rot);
    let q = uq.into_inner();

    Pose {
        position: pos,
        rotation: [q.w, q.i, q.j, q.k],
    }
}

/// Euclidean distance between the translation components of two poses.
fn translation_distance(a: &Pose, b: &Pose) -> f32 {
    let dx = b.position.x - a.position.x;
    let dy = b.position.y - a.position.y;
    let dz = b.position.z - a.position.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Angular distance (in radians) between the rotation components of two poses.
///
/// Computes `2 · acos(|q_a · q_b|)`, i.e. the geodesic angle on SO(3).
fn rotation_distance(a: &Pose, b: &Pose) -> f32 {
    let dot = a.rotation[0] * b.rotation[0]
        + a.rotation[1] * b.rotation[1]
        + a.rotation[2] * b.rotation[2]
        + a.rotation[3] * b.rotation[3];
    let dot = dot.abs().min(1.0);
    2.0 * dot.acos()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(id: u64, width: u32, height: u32, fill: u8) -> Frame {
        let data = vec![fill; (width * height * 3) as usize];
        Frame::new(id, width, height, data).expect("valid frame")
    }

    #[test]
    fn test_tracker_initialization() {
        let config = SlamConfig::default();
        let tracker = Tracker::new(config);
        assert_eq!(tracker.frame_count(), 0);
        assert!(tracker.map().is_empty());
    }

    #[test]
    fn test_process_first_frame_uniform_fails_gracefully() {
        // A uniform (featureless) first frame should fail with InitFailed.
        let config = SlamConfig::default();
        let mut tracker = Tracker::new(config);
        let frame = make_frame(0, 320, 240, 128);
        let result = tracker.process_frame(&frame);
        // Uniform image → no FAST corners → InitFailed.
        assert!(
            result.is_err(),
            "uniform first frame should fail with InitFailed"
        );
        assert_eq!(tracker.frame_count(), 1);
    }

    #[test]
    fn test_process_first_frame_with_features_succeeds() {
        // An isolated bright pixel on a dark background is guaranteed to produce
        // at least one FAST corner (all 16 circle pixels are below centre−threshold).
        let config = SlamConfig::default();
        let mut tracker = Tracker::new(config);

        let mut data = vec![30u8; 64 * 64 * 3];
        let idx = (32 * 64 + 32) as usize * 3;
        data[idx] = 200;
        data[idx + 1] = 200;
        data[idx + 2] = 200;

        let frame = Frame::new(0, 64, 64, data).expect("valid frame");
        let result = tracker.process_frame(&frame);
        assert!(result.is_ok(), "first frame with features must succeed");
        assert_eq!(tracker.frame_count(), 1);
    }

    #[test]
    fn test_compose_poses_identity() {
        let identity = Pose::identity();
        let composed = compose_poses(&identity, &identity);
        // Identity ∘ Identity = Identity
        assert!((composed.position.x).abs() < 1e-5);
        assert!((composed.position.y).abs() < 1e-5);
        assert!((composed.position.z).abs() < 1e-5);
        assert!((composed.rotation[0] - 1.0).abs() < 1e-5, "w should be ~1");
    }

    // ── Keyframe / relocalization helpers ─────────────────────────────────

    #[test]
    fn test_translation_distance_zero() {
        let p = Pose::identity();
        assert!((translation_distance(&p, &p)).abs() < 1e-6);
    }

    #[test]
    fn test_translation_distance_known() {
        let a = Pose::identity();
        let b = Pose {
            position: atlas_core::Point3::new(3.0, 4.0, 0.0),
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        let d = translation_distance(&a, &b);
        assert!((d - 5.0).abs() < 1e-5, "expected 5.0, got {d}");
    }

    #[test]
    fn test_rotation_distance_identity() {
        let p = Pose::identity();
        assert!(rotation_distance(&p, &p) < 1e-5);
    }

    #[test]
    fn test_should_insert_keyframe_no_previous() {
        let config = SlamConfig::default();
        let mut tracker = Tracker::new(config);
        // Before initialization there is no last keyframe.
        tracker.last_keyframe_pose = None;
        let pose = Pose::identity();
        assert!(
            tracker.should_insert_keyframe(&pose),
            "must insert when no previous keyframe exists"
        );
    }

    #[test]
    fn test_should_insert_keyframe_large_translation() {
        let config = SlamConfig::default(); // threshold = 0.1 m
        let mut tracker = Tracker::new(config);
        let base = Pose::identity();
        tracker.last_keyframe_pose = Some(base);

        let far_pose = Pose {
            position: atlas_core::Point3::new(1.0, 0.0, 0.0), // 1 m away
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        assert!(
            tracker.should_insert_keyframe(&far_pose),
            "must insert when translation exceeds threshold"
        );
    }

    #[test]
    fn test_should_not_insert_keyframe_small_motion() {
        let config = SlamConfig::default(); // threshold = 0.1 m
        let mut tracker = Tracker::new(config);
        let base = Pose::identity();
        tracker.last_keyframe_pose = Some(base);

        let close_pose = Pose {
            position: atlas_core::Point3::new(0.01, 0.0, 0.0), // only 1 cm
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        assert!(
            !tracker.should_insert_keyframe(&close_pose),
            "must not insert when motion is below threshold"
        );
    }

    #[test]
    fn test_first_frame_inserts_keyframe() {
        // After the first frame with detectable features, one keyframe must
        // have been stored.
        let config = SlamConfig::default();
        let mut tracker = Tracker::new(config);

        let mut data = vec![30u8; 64 * 64 * 3];
        let idx = (32 * 64 + 32) as usize * 3;
        data[idx] = 200;
        data[idx + 1] = 200;
        data[idx + 2] = 200;

        let frame = Frame::new(0, 64, 64, data).expect("valid frame");
        tracker
            .process_frame(&frame)
            .expect("first frame with features must succeed");

        assert_eq!(
            tracker.keyframes.len(),
            1,
            "exactly one keyframe must be stored after initialization"
        );
    }

    #[test]
    fn test_try_relocalize_returns_none_without_keyframes() {
        let config = SlamConfig::default();
        let mut tracker = Tracker::new(config);
        // Remove all keyframes to simulate a fresh (un-initialized) state.
        tracker.keyframes.clear();

        let kps: Vec<KeyPoint> = vec![];
        let descs: Vec<crate::features::Descriptor> = vec![];
        assert!(
            tracker.try_relocalize(&kps, &descs, 0).is_none(),
            "relocalization without keyframes must return None"
        );
    }

    #[test]
    fn test_matrix4_to_pose_identity() {
        let m = Matrix4::identity();
        let pose = matrix4_to_pose(&m);
        assert!((pose.position.x).abs() < 1e-5);
        assert!((pose.rotation[0] - 1.0).abs() < 1e-5);
    }
}
