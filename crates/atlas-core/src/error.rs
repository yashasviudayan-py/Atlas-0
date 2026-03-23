//! Error types for Atlas-0.

use thiserror::Error;

/// Top-level error type for Atlas-0 operations.
#[derive(Debug, Error)]
pub enum AtlasError {
    #[error("stream error: {0}")]
    Stream(#[from] StreamError),

    #[error("slam error: {0}")]
    Slam(#[from] SlamError),

    #[error("physics error: {0}")]
    Physics(#[from] PhysicsError),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Error)]
pub enum StreamError {
    #[error("failed to open camera device: {0}")]
    CameraOpen(String),

    #[error("frame capture timeout after {0}ms")]
    CaptureTimeout(u64),

    #[error("invalid frame dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    #[error("camera disconnected during capture")]
    Disconnected,

    #[error("camera format negotiation failed: {0}")]
    FormatNegotiation(String),
}

#[derive(Debug, Error)]
pub enum SlamError {
    #[error("insufficient features for tracking: found {found}, need {required}")]
    InsufficientFeatures { found: usize, required: usize },

    #[error("tracking lost at frame {frame_id}")]
    TrackingLost { frame_id: u64 },

    #[error("map initialization failed: {0}")]
    InitFailed(String),

    #[error("pose estimation failed: {0}")]
    PoseEstimationFailed(String),
}

#[derive(Debug, Error)]
pub enum PhysicsError {
    #[error("simulation diverged at step {step}")]
    Diverged { step: u64 },

    #[error("invalid physical property: {0}")]
    InvalidProperty(String),
}

/// Convenience Result type for Atlas-0 operations.
pub type Result<T> = std::result::Result<T, AtlasError>;
