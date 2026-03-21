//! Physics simulator for "what-if" scenario predictions.

use atlas_core::semantic::{RiskAssessment, SemanticObject};
use tracing::debug;

use crate::config::PhysicsConfig;

/// Runs lightweight physics simulations to predict physical consequences.
pub struct Simulator {
    config: PhysicsConfig,
}

impl Simulator {
    #[must_use]
    pub fn new(config: PhysicsConfig) -> Self {
        Self { config }
    }

    /// Assess risks for all objects in the scene.
    ///
    /// Runs physics simulations for each potentially unstable object
    /// and returns a list of risk assessments.
    pub fn assess_risks(&self, objects: &[SemanticObject]) -> Vec<RiskAssessment> {
        let risks = Vec::new();

        for object in objects {
            // TODO(phase-3): Implement actual physics simulation
            // 1. Build simplified collision geometry from object bbox
            // 2. Apply forces (gravity, estimated perturbations)
            // 3. Step simulation forward
            // 4. Check if object leaves stable position

            debug!(
                object_id = object.id,
                label = %object.label,
                "assessing risk"
            );

            let _ = &self.config; // suppress unused warning during scaffolding
            let _ = &risks;
        }

        risks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulator_creation() {
        let config = PhysicsConfig::default();
        let sim = Simulator::new(config);
        let risks = sim.assess_risks(&[]);
        assert!(risks.is_empty());
    }
}
