//! Physics simulator for "what-if" scenario predictions.
//!
//! [`Simulator`] runs lightweight rigid-body simulations for each semantic
//! object to assess whether it is physically stable.  Two perturbation
//! strategies are applied per object:
//!
//! 1. **Gravity-only** — is the object already unstable in its current pose?
//! 2. **Horizontal nudge** — does a small push cause it to fall?
//!
//! The worst-case displacement is mapped to a probability and the risk is
//! classified by type (Fall, Spill, TripHazard, Instability).
//!
//! The returned list is sorted by probability (highest first).

use atlas_core::semantic::{MaterialType, RiskAssessment, RiskType, SemanticObject};
use atlas_core::spatial::Point3;
use tracing::{debug, instrument};

use crate::collision::CollisionDetector;
use crate::config::PhysicsConfig;
use crate::integrator::Integrator;
use crate::rigid_body::RigidBody;
use crate::surfaces::{PlaneType, Surface};

// ─── Simulator ─────────────────────────────────────────────────────────────────

/// Runs lightweight physics simulations to predict physical consequences.
///
/// # Example
/// ```
/// use atlas_physics::simulator::Simulator;
/// use atlas_physics::config::PhysicsConfig;
///
/// let sim = Simulator::new(PhysicsConfig::default());
/// let risks = sim.assess_risks(&[]);
/// assert!(risks.is_empty());
/// ```
pub struct Simulator {
    config: PhysicsConfig,
    /// Known planar surfaces in the scene (floor, walls, tables, …).
    surfaces: Vec<Surface>,
}

impl Simulator {
    /// Create a new simulator with default surfaces (a flat floor at y = 0).
    #[must_use]
    pub fn new(config: PhysicsConfig) -> Self {
        Self {
            surfaces: default_surfaces(),
            config,
        }
    }

    /// Replace the scene surfaces, e.g. surfaces extracted from a
    /// [`GaussianCloud`] via [`SurfaceExtractor`].
    ///
    /// [`GaussianCloud`]: atlas_core::gaussian::GaussianCloud
    /// [`SurfaceExtractor`]: crate::surfaces::SurfaceExtractor
    pub fn set_surfaces(&mut self, surfaces: Vec<Surface>) {
        self.surfaces = surfaces;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Assess physical risks for every object in `objects`.
    ///
    /// For each object the simulator:
    /// 1. Creates a [`RigidBody`] proxy from the object's bounding box and
    ///    material properties.
    /// 2. Runs a gravity-only simulation and a nudged simulation.
    /// 3. Computes displacement from the starting position.
    /// 4. If displacement exceeds a threshold, classifies the risk type,
    ///    estimates probability, and records the predicted impact point.
    ///
    /// Returns assessments sorted by probability (highest first).
    #[instrument(skip(self, objects), fields(object_count = objects.len()))]
    pub fn assess_risks(&self, objects: &[SemanticObject]) -> Vec<RiskAssessment> {
        let mut risks: Vec<RiskAssessment> = Vec::new();

        for object in objects {
            debug!(
                object_id = object.id,
                label = %object.label,
                "assessing risk"
            );

            let body = RigidBody::from_semantic_object(object);

            // Gravity-only: is the object already tipping?
            let (disp_gravity, impact_gravity) = self.run_simulation(&body, [0.0, 0.0, 0.0]);

            // Horizontal nudge in X+Z: does a small push cause instability?
            let (disp_nudge, impact_nudge) = self.run_simulation(&body, [0.5, 0.0, 0.5]);

            let max_displacement = disp_gravity.max(disp_nudge);
            // Prefer the impact point from the larger displacement scenario.
            let impact_point = if disp_gravity >= disp_nudge {
                impact_gravity
            } else {
                impact_nudge
            };

            // Only report risks above the noise floor (5 cm movement).
            if max_displacement > 0.05 {
                let risk_type = classify_risk(object, max_displacement);
                let probability = displacement_to_probability(max_displacement);
                let description = format!(
                    "{}'{}'  may {} — predicted displacement {:.2} m",
                    if object.confidence > 0.7 {
                        ""
                    } else {
                        "(uncertain) "
                    },
                    object.label,
                    risk_type_verb(&risk_type),
                    max_displacement,
                );
                risks.push(RiskAssessment {
                    object_id: object.id,
                    risk_type,
                    probability,
                    impact_point,
                    description,
                });
            }
        }

        // Sort descending by probability.
        risks.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        risks
    }

    // ── Private simulation kernel ─────────────────────────────────────────────

    /// Run one perturbation scenario for `body` and return
    /// `(displacement, Option<impact_point>)`.
    ///
    /// The simulation terminates when:
    /// - The body comes to rest (velocity below `rest_threshold`), or
    /// - The body falls below y = −10 m (out of scene bounds), or
    /// - `config.max_steps` have elapsed.
    fn run_simulation(
        &self,
        body: &RigidBody,
        initial_velocity: [f32; 3],
    ) -> (f32, Option<Point3>) {
        let mut sim = body.clone();
        sim.velocity = initial_velocity;

        let start = sim.position;
        let mut impact: Option<Point3> = None;

        for _ in 0..self.config.max_steps {
            // Gather contacts with every surface.
            let contacts: Vec<_> = self
                .surfaces
                .iter()
                .filter_map(|s| CollisionDetector::body_vs_surface(&sim, s))
                .collect();

            Integrator::step(&mut sim, &contacts, &self.config);

            // Came to rest.
            if sim.is_at_rest(self.config.rest_threshold) {
                impact = Some(Point3::new(
                    sim.position[0],
                    sim.position[1],
                    sim.position[2],
                ));
                break;
            }

            // Fell out of scene.
            if sim.position[1] < -10.0 {
                impact = Some(Point3::new(sim.position[0], -10.0, sim.position[2]));
                break;
            }
        }

        // If neither rest nor out-of-bounds was reached, record the final position.
        if impact.is_none() {
            impact = Some(Point3::new(
                sim.position[0],
                sim.position[1],
                sim.position[2],
            ));
        }

        let dx = sim.position[0] - start[0];
        let dy = sim.position[1] - start[1];
        let dz = sim.position[2] - start[2];
        let displacement = (dx * dx + dy * dy + dz * dz).sqrt();

        (displacement, impact)
    }
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

/// Default scene: a single flat floor at y = 0.
fn default_surfaces() -> Vec<Surface> {
    vec![Surface {
        normal: [0.0, 1.0, 0.0],
        offset: 0.0,
        plane_type: PlaneType::Floor,
        centre: Point3::new(0.0, 0.0, 0.0),
        inlier_count: usize::MAX,
    }]
}

/// Classify the risk type based on object properties and simulated displacement.
fn classify_risk(object: &SemanticObject, displacement: f32) -> RiskType {
    // Liquid containers spill rather than fall.
    if matches!(object.properties.material, MaterialType::Liquid) {
        return RiskType::Spill;
    }
    // Floor-level objects that move horizontally become trip hazards.
    if object.position.y < 0.3 && displacement < 1.0 {
        return RiskType::TripHazard;
    }
    // Fragile or large-displacement objects → fall risk.
    if displacement > 0.5 || object.properties.fragility > 0.7 {
        return RiskType::Fall;
    }
    RiskType::Instability
}

/// Map simulated displacement (metres) to a probability in [0, 1].
///
/// Uses a simple hyperbolic mapping: `p = d / (d + 0.5)` so that:
/// - d = 0.05 → p ≈ 0.09
/// - d = 0.50 → p ≈ 0.50
/// - d = 5.00 → p ≈ 0.91
fn displacement_to_probability(displacement: f32) -> f32 {
    (displacement / (displacement + 0.5)).clamp(0.0, 1.0)
}

fn risk_type_verb(rt: &RiskType) -> &'static str {
    match rt {
        RiskType::Fall => "fall",
        RiskType::Spill => "spill",
        RiskType::Collision => "collide",
        RiskType::TripHazard => "cause a trip",
        RiskType::Instability => "become unstable",
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::semantic::{MaterialProperties, SemanticObject};
    use atlas_core::spatial::Point3;

    fn make_object(
        id: u64,
        label: &str,
        pos: [f32; 3],
        bbox_half: f32,
        mass: f32,
        fragility: f32,
        material: MaterialType,
    ) -> SemanticObject {
        SemanticObject {
            id,
            label: label.into(),
            position: Point3::new(pos[0], pos[1], pos[2]),
            bbox_min: Point3::new(pos[0] - bbox_half, pos[1] - bbox_half, pos[2] - bbox_half),
            bbox_max: Point3::new(pos[0] + bbox_half, pos[1] + bbox_half, pos[2] + bbox_half),
            properties: MaterialProperties {
                mass_kg: mass,
                friction: 0.4,
                fragility,
                material,
            },
            confidence: 0.9,
            relationships: vec![],
        }
    }

    // ── Basic sanity ──────────────────────────────────────────────────────────

    #[test]
    fn test_empty_scene_returns_no_risks() {
        let sim = Simulator::new(PhysicsConfig::default());
        assert!(sim.assess_risks(&[]).is_empty());
    }

    #[test]
    fn test_stable_heavy_object_on_floor_has_low_risk() {
        // A heavy 5kg block sitting flat on the floor should be stable.
        let sim = Simulator::new(PhysicsConfig::default());
        let obj = make_object(
            1,
            "anvil",
            [0.0, 0.5, 0.0],
            0.25,
            5.0,
            0.1,
            MaterialType::Metal,
        );
        let risks = sim.assess_risks(&[obj]);
        // The anvil may or may not appear; if it does, probability should be low.
        if let Some(risk) = risks.first() {
            assert!(
                risk.probability < 0.5,
                "heavy stable object should have low probability, got {}",
                risk.probability
            );
        }
    }

    #[test]
    fn test_elevated_object_without_support_is_at_risk() {
        // A light mug placed high in the air with no surface beneath it must fall.
        let mut sim = Simulator::new(PhysicsConfig::default());
        // Remove the default floor so there is truly nothing to land on.
        sim.set_surfaces(vec![]);
        let obj = make_object(
            2,
            "mug",
            [0.0, 5.0, 0.0],
            0.05,
            0.3,
            0.8,
            MaterialType::Ceramic,
        );
        let risks = sim.assess_risks(&[obj]);
        assert!(
            !risks.is_empty(),
            "elevated object with no support must be flagged"
        );
        let risk = &risks[0];
        assert!(
            risk.probability > 0.3,
            "risk probability should be meaningful, got {}",
            risk.probability
        );
    }

    #[test]
    fn test_liquid_object_classified_as_spill() {
        let mut sim = Simulator::new(PhysicsConfig::default());
        sim.set_surfaces(vec![]); // no floor → guaranteed fall
        let obj = make_object(
            3,
            "glass of water",
            [0.0, 2.0, 0.0],
            0.05,
            0.2,
            0.9,
            MaterialType::Liquid,
        );
        let risks = sim.assess_risks(&[obj]);
        assert!(!risks.is_empty());
        assert!(
            matches!(risks[0].risk_type, RiskType::Spill),
            "liquid should be classified as Spill"
        );
    }

    #[test]
    fn test_risks_sorted_by_probability_descending() {
        let mut sim = Simulator::new(PhysicsConfig::default());
        sim.set_surfaces(vec![]); // all objects fall freely
        let objects = vec![
            // High up, light, fragile → very high risk.
            make_object(
                10,
                "vase",
                [0.0, 10.0, 0.0],
                0.1,
                0.2,
                0.9,
                MaterialType::Glass,
            ),
            // Moderate height.
            make_object(
                11,
                "book",
                [0.0, 3.0, 0.0],
                0.15,
                0.5,
                0.2,
                MaterialType::Wood,
            ),
        ];
        let risks = sim.assess_risks(&objects);
        if risks.len() >= 2 {
            assert!(
                risks[0].probability >= risks[1].probability,
                "risks not sorted: {} < {}",
                risks[0].probability,
                risks[1].probability
            );
        }
    }

    // ── Displacement → probability mapping ───────────────────────────────────

    #[test]
    fn test_displacement_to_probability_zero() {
        assert!((displacement_to_probability(0.0)).abs() < 1e-6);
    }

    #[test]
    fn test_displacement_to_probability_large() {
        let p = displacement_to_probability(100.0);
        assert!(
            p > 0.99,
            "very large displacement should give probability ≈ 1.0"
        );
    }

    #[test]
    fn test_displacement_to_probability_monotone() {
        let p1 = displacement_to_probability(0.1);
        let p2 = displacement_to_probability(1.0);
        let p3 = displacement_to_probability(10.0);
        assert!(p1 < p2, "probability must increase with displacement");
        assert!(p2 < p3);
    }

    // ── set_surfaces ─────────────────────────────────────────────────────────

    #[test]
    fn test_set_surfaces_replaces_defaults() {
        let mut sim = Simulator::new(PhysicsConfig::default());
        // Replace with a high floor at y = 2.0 → objects placed at y < 2.0 will
        // be "inside" the floor and immediately flagged.
        let high_floor = Surface {
            normal: [0.0, 1.0, 0.0],
            offset: 2.0,
            plane_type: PlaneType::Floor,
            centre: Point3::new(0.0, 2.0, 0.0),
            inlier_count: 100,
        };
        sim.set_surfaces(vec![high_floor]);
        // Object below the new floor is immediately in contact / resolves quickly.
        let obj = make_object(
            20,
            "widget",
            [0.0, 1.0, 0.0],
            0.05,
            1.0,
            0.5,
            MaterialType::Plastic,
        );
        // Just verify it doesn't panic and returns something reasonable.
        let _risks = sim.assess_risks(&[obj]);
    }

    // ── Impact point ─────────────────────────────────────────────────────────

    #[test]
    fn test_falling_object_has_impact_point() {
        let mut sim = Simulator::new(PhysicsConfig::default());
        sim.set_surfaces(vec![]); // no floor → falls out of bounds
        let obj = make_object(
            30,
            "bottle",
            [0.0, 5.0, 0.0],
            0.1,
            0.5,
            0.6,
            MaterialType::Glass,
        );
        let risks = sim.assess_risks(&[obj]);
        if !risks.is_empty() {
            // Should have a recorded impact point.
            assert!(
                risks[0].impact_point.is_some(),
                "falling object should record an impact point"
            );
        }
    }
}
