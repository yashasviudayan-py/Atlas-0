//! Physics simulation configuration.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConfig {
    /// Simulation timestep in seconds.
    pub timestep: f32,
    /// Gravitational acceleration (m/s^2).
    pub gravity: f32,
    /// Maximum simulation steps per prediction.
    pub max_steps: u32,
    /// Velocity threshold below which objects are considered at rest.
    pub rest_threshold: f32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            timestep: 0.001,
            gravity: 9.81,
            max_steps: 1000,
            rest_threshold: 0.01,
        }
    }
}
