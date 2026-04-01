//! Atlas-0 Physics: Causal physics simulation engine.
//!
//! This crate provides lightweight physics simulation for predicting
//! physical consequences in the scene (falls, spills, collisions).
//!
//! # Module Overview
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`rigid_body`] | `RigidBody` struct & `BoundingShape` |
//! | [`surfaces`] | RANSAC plane extraction & `Surface` type |
//! | [`collision`] | Narrow-phase shape-pair collision detection |
//! | [`integrator`] | Semi-implicit Euler integration step |
//! | [`simulator`] | High-level `assess_risks()` pipeline & `RiskLoop` |
//! | [`perturbations`] | Four-strategy perturbation set & combined scoring |
//! | [`trajectory`] | Trajectory recording, impact energy, spill zone |
//! | [`config`] | `PhysicsConfig` (loaded from TOML) |

pub mod collision;
pub mod config;
pub mod integrator;
pub mod perturbations;
pub mod rigid_body;
pub mod simulator;
pub mod surfaces;
pub mod trajectory;

use atlas_core::error::PhysicsError;

/// Convenience `Result` type for this crate.
pub type Result<T> = std::result::Result<T, PhysicsError>;
