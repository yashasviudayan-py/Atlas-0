//! Frame processing pipeline.

use atlas_core::Frame;
use crossbeam::channel::{self, Receiver, Sender};
use crate::config::StreamConfig;

/// A multi-producer, multi-consumer frame pipeline.
///
/// Frames flow from the capture source through processing stages.
pub struct FramePipeline {
    config: StreamConfig,
    sender: Sender<Frame>,
    receiver: Receiver<Frame>,
}

impl FramePipeline {
    #[must_use]
    pub fn new(config: StreamConfig) -> Self {
        let (sender, receiver) = channel::bounded(config.buffer_size);
        Self {
            config,
            sender,
            receiver,
        }
    }

    /// Get a sender handle for pushing frames into the pipeline.
    #[must_use]
    pub fn sender(&self) -> Sender<Frame> {
        self.sender.clone()
    }

    /// Get a receiver handle for consuming frames from the pipeline.
    #[must_use]
    pub fn receiver(&self) -> Receiver<Frame> {
        self.receiver.clone()
    }

    /// Target FPS for the pipeline.
    #[must_use]
    pub fn target_fps(&self) -> u32 {
        self.config.target_fps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_send_receive() {
        let config = StreamConfig {
            buffer_size: 2,
            ..StreamConfig::default()
        };
        let pipeline = FramePipeline::new(config);

        let data = vec![0u8; 1280 * 720 * 3];
        let frame = Frame::new(0, 1280, 720, data).unwrap();

        pipeline.sender().send(frame).unwrap();
        let received = pipeline.receiver().recv().unwrap();
        assert_eq!(received.id, 0);
    }
}
