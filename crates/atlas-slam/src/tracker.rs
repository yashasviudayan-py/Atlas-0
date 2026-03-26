//! Camera pose tracker with integrated 3DGS reconstruction.
//!
//! The [`Tracker`] maintains the full SLAM state across frames:
//!
//! 1. Extracts FAST+BRIEF features from each frame.
//! 2. Matches them against the previous frame.
//! 3. Estimates the relative camera pose via the 8-point essential-matrix algorithm.
//! 4. On keyframe insertion: triangulates depth, initialises new Gaussians, and runs
//!    the photometric optimizer.
//! 5. Provides relocalization fallback when tracking is lost.

use atlas_core::{Frame, GaussianCloud, Point3, Pose};
use nalgebra::{Matrix3, Matrix4, Rotation3, UnitQuaternion};
use tracing::{info, warn};

use crate::config::SlamConfig;
use crate::depth::DepthEstimator;
use crate::features::{Descriptor, FeatureExtractor, KeyPoint};
use crate::gaussian_init::{GaussianInitConfig, GaussianInitializer};
use crate::keyframe::KeyframeGraph;
use crate::matching::{DMatch, FeatureMatcher};
use crate::optimizer::{GaussianOptimizer, OptimizerConfig};
use crate::pose_estimation::PoseEstimator;

// ─── Legacy relocalization keyframe (private, same-file only) ────────────────

/// A stored reference frame used *only* for relocalization.
///
/// Separate from [`KeyframeGraph`] so the existing relocalization logic and its
/// unit tests are not disrupted.
struct RelocalKeyframe {
    pose: Pose,
    keypoints: Vec<KeyPoint>,
    descriptors: Vec<Descriptor>,
}

// ─── Tracker ─────────────────────────────────────────────────────────────────

/// Tracks camera pose and maintains the 3D Gaussian map.
pub struct Tracker {
    config: SlamConfig,
    current_pose: Pose,
    /// Previous frame's estimated pose (used for triangulation).
    prev_pose: Pose,
    map: GaussianCloud,
    frame_count: u64,
    is_initialized: bool,

    // ── Feature pipeline ─────────────────────────────────────────────────────
    extractor: FeatureExtractor,
    matcher: FeatureMatcher,
    estimator: PoseEstimator,

    // ── Part 3: depth / Gaussian init / optimization ─────────────────────────
    depth_estimator: DepthEstimator,
    gaussian_init: GaussianInitializer,
    optimizer: GaussianOptimizer,
    /// Keyframe graph tracking co-visibility for Gaussian densification.
    keyframe_graph: KeyframeGraph,

    // ── State from the previous frame ────────────────────────────────────────
    prev_keypoints: Vec<KeyPoint>,
    prev_descriptors: Vec<Descriptor>,

    // ── Relocalization store (legacy, keeps existing tests green) ─────────────
    keyframes: Vec<RelocalKeyframe>,
    last_keyframe_pose: Option<Pose>,
}

impl Tracker {
    /// Create a new tracker with the given configuration.
    ///
    /// The tracker starts uninitialised; the first call to [`process_frame`]
    /// captures the reference frame.
    #[must_use]
    pub fn new(config: SlamConfig) -> Self {
        let extractor = FeatureExtractor::new(config.fast_threshold, config.max_features);
        let matcher = FeatureMatcher::new(config.ratio_test_threshold, config.max_match_distance);
        let estimator = PoseEstimator::new(
            config.camera.clone(),
            config.ransac_iterations,
            config.ransac_threshold,
        );
        let depth_estimator = DepthEstimator::new(config.camera.clone());
        let gaussian_init = GaussianInitializer::new(GaussianInitConfig::default());
        let optimizer = GaussianOptimizer::new(OptimizerConfig {
            iterations: config.optimization_iterations,
            ..OptimizerConfig::default()
        });
        let keyframe_graph = KeyframeGraph::new(
            config.keyframe_translation_threshold,
            config.keyframe_rotation_threshold,
        );

        Self {
            depth_estimator,
            gaussian_init,
            optimizer,
            keyframe_graph,
            config,
            current_pose: Pose::identity(),
            prev_pose: Pose::identity(),
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

        // 3. Build (prev_pixel, curr_pixel) correspondence list.
        let correspondences: Vec<([f64; 2], [f64; 2])> = matches
            .iter()
            .map(|m| {
                let curr = &kps[m.query_idx];
                let prev = &self.prev_keypoints[m.train_idx];
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
        let new_pose = compose_poses(&self.current_pose, &relative_pose);

        // 6. Keyframe insertion + Gaussian update.
        if self.should_insert_keyframe(&new_pose) {
            self.insert_keyframe_and_update_map(frame, &kps, &descs, &matches, &new_pose);
        }

        self.prev_pose = self.current_pose;
        self.current_pose = new_pose;

        // 7. Store current features for the next iteration.
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

    /// Number of keyframes in the graph.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.keyframe_graph.len()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Initialise the SLAM system from the first frame.
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

        // Store as the initial keyframe for relocalization and the graph.
        self.keyframes.push(RelocalKeyframe {
            pose: Pose::identity(),
            keypoints: kps.clone(),
            descriptors: descs.clone(),
        });
        self.keyframe_graph
            .insert(Pose::identity(), kps.clone(), descs.clone(), 0);
        self.last_keyframe_pose = Some(Pose::identity());

        self.prev_keypoints = kps;
        self.prev_descriptors = descs;
        self.current_pose = Pose::identity();
        self.prev_pose = Pose::identity();
        self.is_initialized = true;
        Ok(())
    }

    /// Insert a keyframe, triangulate depth, spawn new Gaussians, and optimize.
    fn insert_keyframe_and_update_map(
        &mut self,
        frame: &Frame,
        kps_curr: &[KeyPoint],
        descs_curr: &[Descriptor],
        matches: &[DMatch],
        pose_curr: &Pose,
    ) {
        // 1. Insert into both the legacy relocalization store and the graph.
        self.keyframes.push(RelocalKeyframe {
            pose: *pose_curr,
            keypoints: kps_curr.to_vec(),
            descriptors: descs_curr.to_vec(),
        });
        self.keyframe_graph.insert(
            *pose_curr,
            kps_curr.to_vec(),
            descs_curr.to_vec(),
            self.map.len(),
        );
        self.last_keyframe_pose = Some(*pose_curr);

        info!(
            total_keyframes = self.keyframes.len(),
            gaussians = self.map.len(),
            "keyframe inserted"
        );

        // 2. Triangulate depth from the current and previous frame.
        let depth_map = self.depth_estimator.estimate(
            &self.prev_keypoints,
            kps_curr,
            matches,
            &self.prev_pose,
            pose_curr,
            frame.width,
            frame.height,
        );

        if depth_map.world_points.is_empty() {
            return; // No triangulated points — nothing to add.
        }

        // 3. Look up RGB colour at each triangulated pixel.
        let new_points: Vec<(Point3, [f32; 3])> = depth_map
            .world_points
            .iter()
            .map(|(pos, px, py)| {
                let color = sample_pixel_rgb(&frame.data, frame.width, frame.height, *px, *py);
                (*pos, color)
            })
            .collect();

        // 4. Initialise new Gaussians and add them to the map.
        let new_gaussians = self.gaussian_init.from_point_cloud(&new_points);
        let new_count = new_gaussians.len();
        for g in new_gaussians {
            if self.map.len() < self.config.max_gaussians {
                self.map.add(g);
            }
        }

        info!(
            new_gaussians = new_count,
            total_gaussians = self.map.len(),
            "Gaussians added from triangulation"
        );

        // 5. Optimize the map against the current frame.
        if !self.map.is_empty() {
            self.optimizer.optimize(
                &mut self.map.gaussians,
                &frame.data,
                frame.width,
                frame.height,
                pose_curr,
                &self.config.camera,
            );
        }

        // 6. Prune any Gaussians that have grown over the map limit.
        if self.map.len() > self.config.max_gaussians {
            // Keep the highest-opacity Gaussians.
            self.map.gaussians.sort_by(|a, b| {
                b.opacity
                    .partial_cmp(&a.opacity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.map.gaussians.truncate(self.config.max_gaussians);
        }
    }

    /// Decide whether to insert a new keyframe at `pose`.
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

    /// Attempt to recover tracking by matching against stored keyframes.
    fn try_relocalize(
        &self,
        kps: &[KeyPoint],
        descs: &[Descriptor],
        frame_id: u64,
    ) -> Option<Pose> {
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

// ─── Pixel sampling ───────────────────────────────────────────────────────────

/// Sample an RGB pixel from `data` (row-major, 3 bytes per pixel) and return
/// it as a `[f32; 3]` in the range \[0, 1\].
fn sample_pixel_rgb(data: &[u8], width: u32, height: u32, x: u32, y: u32) -> [f32; 3] {
    if x >= width || y >= height {
        return [0.5, 0.5, 0.5];
    }
    let idx = (y * width + x) as usize * 3;
    if idx + 2 >= data.len() {
        return [0.5, 0.5, 0.5];
    }
    [
        data[idx] as f32 / 255.0,
        data[idx + 1] as f32 / 255.0,
        data[idx + 2] as f32 / 255.0,
    ]
}

// ─── Pose composition helpers ─────────────────────────────────────────────────

fn compose_poses(base: &Pose, relative: &Pose) -> Pose {
    let m = base.to_matrix() * relative.to_matrix();
    matrix4_to_pose(&m)
}

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

fn translation_distance(a: &Pose, b: &Pose) -> f32 {
    let dx = b.position.x - a.position.x;
    let dy = b.position.y - a.position.y;
    let dz = b.position.z - a.position.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn rotation_distance(a: &Pose, b: &Pose) -> f32 {
    let dot = a.rotation[0] * b.rotation[0]
        + a.rotation[1] * b.rotation[1]
        + a.rotation[2] * b.rotation[2]
        + a.rotation[3] * b.rotation[3];
    2.0 * dot.abs().min(1.0).acos()
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
        let config = SlamConfig::default();
        let mut tracker = Tracker::new(config);
        let frame = make_frame(0, 320, 240, 128);
        let result = tracker.process_frame(&frame);
        assert!(
            result.is_err(),
            "uniform first frame should fail with InitFailed"
        );
        assert_eq!(tracker.frame_count(), 1);
    }

    #[test]
    fn test_process_first_frame_with_features_succeeds() {
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
        assert!((composed.position.x).abs() < 1e-5);
        assert!((composed.position.y).abs() < 1e-5);
        assert!((composed.position.z).abs() < 1e-5);
        assert!((composed.rotation[0] - 1.0).abs() < 1e-5, "w should be ~1");
    }

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
        tracker.last_keyframe_pose = None;
        let pose = Pose::identity();
        assert!(tracker.should_insert_keyframe(&pose));
    }

    #[test]
    fn test_should_insert_keyframe_large_translation() {
        let config = SlamConfig::default();
        let mut tracker = Tracker::new(config);
        tracker.last_keyframe_pose = Some(Pose::identity());
        let far_pose = Pose {
            position: atlas_core::Point3::new(1.0, 0.0, 0.0),
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        assert!(tracker.should_insert_keyframe(&far_pose));
    }

    #[test]
    fn test_should_not_insert_keyframe_small_motion() {
        let config = SlamConfig::default();
        let mut tracker = Tracker::new(config);
        tracker.last_keyframe_pose = Some(Pose::identity());
        let close_pose = Pose {
            position: atlas_core::Point3::new(0.01, 0.0, 0.0),
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        assert!(!tracker.should_insert_keyframe(&close_pose));
    }

    #[test]
    fn test_first_frame_inserts_keyframe() {
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
        tracker.keyframes.clear();

        let kps: Vec<KeyPoint> = vec![];
        let descs: Vec<crate::features::Descriptor> = vec![];
        assert!(tracker.try_relocalize(&kps, &descs, 0).is_none());
    }

    #[test]
    fn test_matrix4_to_pose_identity() {
        let m = Matrix4::identity();
        let pose = matrix4_to_pose(&m);
        assert!((pose.position.x).abs() < 1e-5);
        assert!((pose.rotation[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_keyframe_count_increments_on_init() {
        let config = SlamConfig::default();
        let mut tracker = Tracker::new(config);

        let mut data = vec![30u8; 64 * 64 * 3];
        let idx = (32 * 64 + 32) as usize * 3;
        data[idx] = 200;
        data[idx + 1] = 200;
        data[idx + 2] = 200;

        let frame = Frame::new(0, 64, 64, data).expect("valid frame");
        tracker.process_frame(&frame).expect("init must succeed");
        assert_eq!(tracker.keyframe_count(), 1);
    }

    #[test]
    fn test_sample_pixel_rgb_in_bounds() {
        let width = 4u32;
        let height = 4u32;
        let mut data = vec![0u8; (width * height * 3) as usize];
        // Set pixel (2, 1) to (255, 128, 64).
        let idx = (width + 2) as usize * 3;
        data[idx] = 255;
        data[idx + 1] = 128;
        data[idx + 2] = 64;
        let rgb = sample_pixel_rgb(&data, width, height, 2, 1);
        assert!((rgb[0] - 1.0).abs() < 1e-3);
        assert!((rgb[1] - 128.0 / 255.0).abs() < 1e-3);
        assert!((rgb[2] - 64.0 / 255.0).abs() < 1e-3);
    }

    #[test]
    fn test_sample_pixel_rgb_out_of_bounds_returns_grey() {
        let data = vec![255u8; 12];
        let rgb = sample_pixel_rgb(&data, 2, 2, 5, 5);
        assert!((rgb[0] - 0.5).abs() < 1e-6);
    }
}
