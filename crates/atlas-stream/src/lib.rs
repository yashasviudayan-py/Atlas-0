//! Atlas-0 Stream: Video ingestion and frame pipeline.
//!
//! Handles camera capture, frame buffering, and distributing
//! frames to downstream consumers (SLAM, VLM).

pub mod config;
pub mod pipeline;

use atlas_core::error::StreamError;

pub type Result<T> = std::result::Result<T, StreamError>;
