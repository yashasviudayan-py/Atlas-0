//! SLAM pipeline configuration.

use serde::{Deserialize, Serialize};

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
}

impl Default for SlamConfig {
    fn default() -> Self {
        Self {
            min_features: 100,
            max_gaussians: 500_000,
            keyframe_translation_threshold: 0.1,
            keyframe_rotation_threshold: 0.1,
            optimization_iterations: 10,
            enable_loop_closure: true,
        }
    }
}
