//! Stream pipeline configuration.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Target frames per second.
    pub target_fps: u32,
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
    /// Number of frames to buffer in the pipeline.
    pub buffer_size: usize,
    /// Camera device index or path.
    pub device: String,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            target_fps: 60,
            frame_width: 1280,
            frame_height: 720,
            buffer_size: 4,
            device: "0".to_string(),
        }
    }
}
