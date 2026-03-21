//! Atlas-0 Physics: Causal physics simulation engine.
//!
//! This crate provides lightweight physics simulation for predicting
//! physical consequences in the scene (falls, spills, collisions).

pub mod config;
pub mod simulator;

use atlas_core::error::PhysicsError;

pub type Result<T> = std::result::Result<T, PhysicsError>;
