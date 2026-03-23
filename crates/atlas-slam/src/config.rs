//! SLAM pipeline configuration.

use serde::{Deserialize, Serialize};

/// Pinhole camera intrinsics used for coordinate normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    /// Focal length along the X axis (pixels).
    pub fx: f64,
    /// Focal length along the Y axis (pixels).
    pub fy: f64,
    /// Principal point X coordinate (pixels).
    pub cx: f64,
    /// Principal point Y coordinate (pixels).
    pub cy: f64,
}

impl Default for CameraIntrinsics {
    fn default() -> Self {
        // Reasonable defaults for a 640×480 sensor with ~70° horizontal FOV.
        Self {
            fx: 525.0,
            fy: 525.0,
            cx: 320.0,
            cy: 240.0,
        }
    }
}

/// Configuration for the SLAM pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlamConfig {
    /// Minimum number of feature matches required for tracking.
    pub min_features: usize,
    /// Maximum number of Gaussians in the map before pruning.
    pub max_gaussians: usize,
    /// Keyframe insertion threshold (translation in meters).
    pub keyframe_translation_threshold: f32,
    /// Keyframe insertion threshold (rotation in radians).
    pub keyframe_rotation_threshold: f32,
    /// Number of optimization iterations per keyframe.
    pub optimization_iterations: u32,
    /// Enable loop closure detection.
    pub enable_loop_closure: bool,

    // --- Camera ---
    /// Pinhole camera intrinsics for pose estimation.
    pub camera: CameraIntrinsics,

    // --- Feature extraction ---
    /// FAST corner detection intensity-difference threshold (typical: 10–30).
    pub fast_threshold: u8,
    /// Maximum number of keypoints extracted per frame.
    pub max_features: usize,

    // --- Feature matching ---
    /// Lowe's ratio-test threshold (typical: 0.70–0.80).
    pub ratio_test_threshold: f32,
    /// Absolute maximum Hamming distance accepted as a match (out of 256).
    pub max_match_distance: u32,

    // --- Pose estimation ---
    /// Number of RANSAC iterations for essential-matrix estimation.
    pub ransac_iterations: usize,
    /// Sampson-distance inlier threshold in normalised image coordinates.
    pub ransac_threshold: f64,
}

impl Default for SlamConfig {
    fn default() -> Self {
        Self {
            min_features: 30,
            max_gaussians: 500_000,
            keyframe_translation_threshold: 0.1,
            keyframe_rotation_threshold: 0.1,
            optimization_iterations: 10,
            enable_loop_closure: true,
            camera: CameraIntrinsics::default(),
            fast_threshold: 20,
            max_features: 1000,
            ratio_test_threshold: 0.75,
            max_match_distance: 80,
            ransac_iterations: 200,
            ransac_threshold: 1e-3,
        }
    }
}
