//! Atlas-0 Core: Shared types, traits, and error handling.
//!
//! This crate provides the foundational types used across all Atlas-0 components:
//! - Spatial primitives (poses, points, Gaussians)
//! - Frame data structures
//! - Semantic metadata types
//! - Common error types

pub mod error;
pub mod frame;
pub mod gaussian;
pub mod semantic;
pub mod shared_mem;
pub mod spatial;

pub use error::{AtlasError, Result};
pub use frame::{Frame, FrameId};
pub use gaussian::{BoundingBox3D, Gaussian3D, GaussianCloud};
pub use spatial::{Point3, Pose};
