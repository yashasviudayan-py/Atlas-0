//! Frame data structures for the video pipeline.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

/// Unique identifier for a frame in the pipeline.
pub type FrameId = u64;

/// A single video frame with metadata.
#[derive(Clone)]
pub struct Frame {
    /// Monotonically increasing frame identifier.
    pub id: FrameId,
    /// Timestamp when the frame was captured.
    pub timestamp: Instant,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Raw pixel data (RGB, 3 bytes per pixel). Shared via Arc for zero-copy passing.
    pub data: Arc<[u8]>,
}

impl Frame {
    /// Create a new frame from raw RGB pixel data.
    ///
    /// # Errors
    /// Returns `None` if data length doesn't match width * height * 3.
    #[must_use]
    pub fn new(id: FrameId, width: u32, height: u32, data: Vec<u8>) -> Option<Self> {
        let expected = (width as usize) * (height as usize) * 3;
        if data.len() != expected {
            return None;
        }
        Some(Self {
            id,
            timestamp: Instant::now(),
            width,
            height,
            data: Arc::from(data),
        })
    }

    /// Total number of pixels.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

/// Metadata about frame processing timing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameTiming {
    pub frame_id: FrameId,
    pub capture_ms: f64,
    pub slam_ms: f64,
    pub semantic_ms: f64,
    pub total_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_creation() {
        let data = vec![0u8; 640 * 480 * 3];
        let frame = Frame::new(0, 640, 480, data);
        assert!(frame.is_some());
        assert_eq!(frame.unwrap().pixel_count(), 640 * 480);
    }

    #[test]
    fn test_frame_invalid_size() {
        let data = vec![0u8; 100]; // wrong size
        let frame = Frame::new(0, 640, 480, data);
        assert!(frame.is_none());
    }
}
