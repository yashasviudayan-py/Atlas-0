//! Frame processing pipeline.

use crate::config::StreamConfig;
use atlas_core::Frame;
use crossbeam::channel::{self, Receiver, Sender};

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

    #[test]
    fn target_fps_returns_configured_value() {
        // Arrange
        let config = StreamConfig {
            target_fps: 30,
            ..StreamConfig::default()
        };
        // Act
        let pipeline = FramePipeline::new(config);
        // Assert
        assert_eq!(pipeline.target_fps(), 30);
    }

    #[test]
    fn multiple_frames_preserve_fifo_order() {
        // Arrange
        let config = StreamConfig {
            buffer_size: 8,
            ..StreamConfig::default()
        };
        let pipeline = FramePipeline::new(config);
        let sender = pipeline.sender();
        let receiver = pipeline.receiver();

        // Act — send three frames with different IDs.
        for id in 0u64..3 {
            let data = vec![0u8; 4 * 4 * 3];
            let frame = Frame::new(id, 4, 4, data).unwrap();
            sender.send(frame).unwrap();
        }

        // Assert — receive in the same order.
        for expected_id in 0u64..3 {
            let frame = receiver.recv().unwrap();
            assert_eq!(frame.id, expected_id);
        }
    }

    #[test]
    fn buffer_full_try_send_errors_without_panic() {
        // Arrange — buffer capacity of 1.
        let config = StreamConfig {
            buffer_size: 1,
            ..StreamConfig::default()
        };
        let pipeline = FramePipeline::new(config);
        let sender = pipeline.sender();

        let mk_frame = |id| Frame::new(id, 1, 1, vec![0u8; 3]).unwrap();

        // Act — fill the buffer.
        sender.send(mk_frame(0)).unwrap();
        // Try to push one more — should fail with Full, not panic.
        let result = sender.try_send(mk_frame(1));

        // Assert
        assert!(
            matches!(result, Err(crossbeam::channel::TrySendError::Full(_))),
            "expected TrySendError::Full, got {result:?}"
        );
    }

    #[test]
    fn dropped_sender_closes_receiver() {
        // Arrange
        let config = StreamConfig::default();
        let pipeline = FramePipeline::new(config);
        let receiver = pipeline.receiver();

        // Act — drop the pipeline (and its internal sender clone).
        drop(pipeline);

        // Assert — receiver sees disconnect immediately.
        assert!(receiver.recv().is_err());
    }

    #[test]
    fn multiple_sender_clones_all_route_to_same_receiver() {
        // Arrange
        let config = StreamConfig {
            buffer_size: 4,
            ..StreamConfig::default()
        };
        let pipeline = FramePipeline::new(config);
        let s1 = pipeline.sender();
        let s2 = pipeline.sender();
        let receiver = pipeline.receiver();

        // Act — send from two different sender clones.
        s1.send(Frame::new(10, 1, 1, vec![0u8; 3]).unwrap())
            .unwrap();
        s2.send(Frame::new(20, 1, 1, vec![0u8; 3]).unwrap())
            .unwrap();

        // Assert — both frames arrive.
        let ids: Vec<u64> = (0..2).map(|_| receiver.recv().unwrap().id).collect();
        assert!(ids.contains(&10));
        assert!(ids.contains(&20));
    }
}
