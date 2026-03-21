//! Atlas-0 SLAM: 3D Gaussian Splatting SLAM pipeline.
//!
//! This crate implements the spatial mapping engine:
//! - Camera pose estimation via visual odometry
//! - 3D Gaussian Splatting for scene reconstruction
//! - Keyframe management and map optimization

pub mod config;
pub mod tracker;

use atlas_core::error::SlamError;

pub type Result<T> = std::result::Result<T, SlamError>;
