//! Camera pose tracker using visual odometry.

use atlas_core::{Frame, GaussianCloud, Pose};
use tracing::info;

use crate::config::SlamConfig;

/// Tracks camera pose and maintains the 3D Gaussian map.
pub struct Tracker {
    config: SlamConfig,
    current_pose: Pose,
    map: GaussianCloud,
    frame_count: u64,
    is_initialized: bool,
}

impl Tracker {
    #[must_use]
    pub fn new(config: SlamConfig) -> Self {
        Self {
            config,
            current_pose: Pose::identity(),
            map: GaussianCloud::new(),
            frame_count: 0,
            is_initialized: false,
        }
    }

    /// Process a new frame and update the map.
    ///
    /// Returns the estimated camera pose for this frame.
    ///
    /// # Errors
    /// Returns `SlamError` if tracking is lost or features are insufficient.
    pub fn process_frame(&mut self, frame: &Frame) -> crate::Result<Pose> {
        self.frame_count += 1;

        if !self.is_initialized {
            self.initialize(frame)?;
            return Ok(self.current_pose);
        }

        // TODO(phase-1): Implement visual odometry using config thresholds
        let _ = &self.config;
        // 1. Extract features from frame
        // 2. Match against previous frame / keyframe
        // 3. Estimate relative pose via PnP or essential matrix
        // 4. Update current_pose
        // 5. Decide if this is a keyframe -> if so, add Gaussians

        info!(
            frame_id = frame.id,
            gaussians = self.map.len(),
            "processed frame"
        );

        Ok(self.current_pose)
    }

    /// Initialize the SLAM system from the first frame.
    fn initialize(&mut self, frame: &Frame) -> crate::Result<()> {
        info!(
            width = frame.width,
            height = frame.height,
            "initializing SLAM from first frame"
        );

        // TODO(phase-1): Initialize map from first frame
        // 1. Extract features
        // 2. Create initial Gaussians from depth estimation
        // 3. Set initial pose as origin

        self.is_initialized = true;
        Ok(())
    }

    /// Get the current estimated camera pose.
    #[must_use]
    pub fn current_pose(&self) -> Pose {
        self.current_pose
    }

    /// Get an immutable reference to the current Gaussian map.
    #[must_use]
    pub fn map(&self) -> &GaussianCloud {
        &self.map
    }

    /// Total number of frames processed.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_initialization() {
        let config = SlamConfig::default();
        let tracker = Tracker::new(config);
        assert_eq!(tracker.frame_count(), 0);
        assert!(tracker.map().is_empty());
    }

    #[test]
    fn test_process_first_frame() {
        let config = SlamConfig::default();
        let mut tracker = Tracker::new(config);

        let data = vec![128u8; 320 * 240 * 3];
        let frame = Frame::new(0, 320, 240, data).unwrap();

        let result = tracker.process_frame(&frame);
        assert!(result.is_ok());
        assert_eq!(tracker.frame_count(), 1);
    }
}
