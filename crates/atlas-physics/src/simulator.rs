//! Physics simulator for "what-if" scenario predictions.
//!
//! This module provides two main abstractions:
//!
//! - [`Simulator`] — stateless engine that runs lightweight rigid-body
//!   simulations to assess whether scene objects are physically stable.
//!   Callers can choose between a quick two-perturbation assessment
//!   ([`Simulator::assess_risks`]) or the full four-perturbation variant with
//!   trajectory data ([`Simulator::assess_risks_detailed`]).
//!
//! - [`RiskLoop`] — background thread that continuously re-evaluates scene
//!   risks, triggered by scene-change events.  Uses a priority queue to
//!   simulate the most likely at-risk objects first.

use std::sync::{
    Arc, Mutex, RwLock,
    mpsc::{self, RecvTimeoutError},
};
use std::time::Duration;

use atlas_core::semantic::{MaterialType, RelationType, RiskAssessment, RiskType, SemanticObject};
use atlas_core::spatial::Point3;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, warn};

use crate::collision::CollisionDetector;
use crate::config::PhysicsConfig;
use crate::integrator::Integrator;
use crate::perturbations::Perturbation;
use crate::rigid_body::RigidBody;
use crate::surfaces::{PlaneType, Surface};
use crate::trajectory::{TrajectoryRecorder, TrajectoryResult};

// ─── DetailedRiskAssessment ───────────────────────────────────────────────────

/// A risk assessment enriched with full trajectory data.
///
/// Produced by [`Simulator::assess_risks_detailed`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedRiskAssessment {
    /// Base risk fields (mirrors [`RiskAssessment`]).
    pub base: RiskAssessment,
    /// Predicted trajectory for the worst-case perturbation scenario.
    pub trajectory: TrajectoryResult,
}

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

    /// Assess risks using the full four-perturbation set with trajectory
    /// recording.
    ///
    /// Unlike [`assess_risks`], this method:
    ///
    /// - Applies all four strategies from [`Perturbation::standard_set`].
    /// - Combines their displacement scores using weighted averaging.
    /// - Records the worst-case trajectory including impact energy and spill zone.
    ///
    /// Returns assessments sorted by probability (highest first).
    ///
    /// [`assess_risks`]: Simulator::assess_risks
    #[instrument(skip(self, objects), fields(object_count = objects.len()))]
    pub fn assess_risks_detailed(&self, objects: &[SemanticObject]) -> Vec<DetailedRiskAssessment> {
        let perturbations = Perturbation::standard_set();
        let mut results: Vec<DetailedRiskAssessment> = Vec::new();

        for object in objects {
            debug!(
                object_id = object.id,
                label = %object.label,
                "assessing risk (detailed)"
            );

            let body = RigidBody::from_semantic_object(object);
            let is_liquid = matches!(object.properties.material, MaterialType::Liquid);

            let mut perturbation_scores: Vec<(f32, f32)> = Vec::new();
            let mut worst_disp = 0.0f32;
            let mut worst_traj: Option<TrajectoryResult> = None;

            for p in &perturbations {
                let (disp, traj) = self.run_simulation_with_trajectory(&body, p, is_liquid);
                perturbation_scores.push((disp, p.weight));
                if disp > worst_disp {
                    worst_disp = disp;
                    worst_traj = Some(traj);
                }
            }

            if worst_disp <= 0.05 {
                continue;
            }

            let combined_prob = Perturbation::combined_score(&perturbation_scores);
            let risk_type = classify_risk(object, worst_disp);
            let trajectory = worst_traj.unwrap_or_else(|| TrajectoryRecorder::new(10).finish(None));
            let impact_point = trajectory.impact_point;
            let description = format!(
                "{}'{}'  may {} — predicted displacement {:.2} m",
                if object.confidence > 0.7 {
                    ""
                } else {
                    "(uncertain) "
                },
                object.label,
                risk_type_verb(&risk_type),
                worst_disp,
            );

            results.push(DetailedRiskAssessment {
                base: RiskAssessment {
                    object_id: object.id,
                    risk_type,
                    probability: combined_prob,
                    impact_point,
                    description,
                },
                trajectory,
            });
        }

        results.sort_by(|a, b| {
            b.base
                .probability
                .partial_cmp(&a.base.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
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

    /// Run one perturbation scenario with trajectory recording.
    ///
    /// Returns `(displacement_m, TrajectoryResult)`.
    ///
    /// The perturbation's `extra_gravity` is applied as additional X/Z
    /// acceleration each step (on top of the simulator's downward gravity).
    fn run_simulation_with_trajectory(
        &self,
        body: &RigidBody,
        perturbation: &Perturbation,
        is_liquid: bool,
    ) -> (f32, TrajectoryResult) {
        let mut sim = body.clone();
        sim.velocity = perturbation.initial_velocity;

        let start = sim.position;
        let mut recorder = TrajectoryRecorder::new(10).with_liquid(is_liquid);

        for _ in 0..self.config.max_steps {
            // Apply horizontal extra-gravity from this perturbation scenario.
            let dt = self.config.timestep;
            sim.velocity[0] += perturbation.extra_gravity[0] * dt;
            sim.velocity[2] += perturbation.extra_gravity[2] * dt;

            let contacts: Vec<_> = self
                .surfaces
                .iter()
                .filter_map(|s| CollisionDetector::body_vs_surface(&sim, s))
                .collect();

            if !contacts.is_empty() {
                recorder.record_impact(&sim);
            }

            Integrator::step(&mut sim, &contacts, &self.config);
            recorder.record_step(&sim);

            if sim.is_at_rest(self.config.rest_threshold) {
                break;
            }
            if sim.position[1] < -10.0 {
                break;
            }
        }

        let impact_point = Some(Point3::new(
            sim.position[0],
            sim.position[1],
            sim.position[2],
        ));
        let traj = recorder.finish(impact_point);

        let dx = sim.position[0] - start[0];
        let dy = sim.position[1] - start[1];
        let dz = sim.position[2] - start[2];
        let displacement = (dx * dx + dy * dy + dz * dz).sqrt();

        (displacement, traj)
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

// ─── Priority scoring ─────────────────────────────────────────────────────────

/// Heuristic priority score for scheduling objects in the risk loop.
///
/// Higher score → object is simulated first (more likely to be at risk).
///
/// Factors:
/// - Centre-of-mass height (elevated objects first).
/// - Material fragility.
/// - Structural relationship bonuses (leaning, hanging, on-top-of).
///
/// Returns a value in `[0, 1]`.
pub(crate) fn object_priority(obj: &SemanticObject) -> f32 {
    let height_factor = (obj.position.y / 2.0).clamp(0.0, 1.0);
    let fragility_factor = obj.properties.fragility.clamp(0.0, 1.0);

    let relation_bonus: f32 = obj
        .relationships
        .iter()
        .map(|r| match r.relation_type {
            RelationType::Leaning => 0.30,
            RelationType::OnTopOf => 0.20,
            RelationType::Hanging => 0.25,
            _ => 0.0,
        })
        .sum::<f32>()
        .min(0.5);

    (height_factor * 0.4 + fragility_factor * 0.3 + relation_bonus).clamp(0.0, 1.0)
}

// ─── RiskLoop ─────────────────────────────────────────────────────────────────

/// Background risk-assessment loop.
///
/// Spawns a worker thread that continuously re-evaluates scene risks whenever
/// [`RiskLoop::trigger`] is called (or every 5 seconds at minimum).  The
/// thread uses a priority queue (sorted by [`object_priority`]) to simulate
/// the most at-risk objects first.
///
/// # Example
/// ```no_run
/// use atlas_physics::{simulator::RiskLoop, config::PhysicsConfig};
///
/// let loop_ = RiskLoop::spawn(PhysicsConfig::default());
/// // Update the scene:
/// loop_.update_scene(vec![], vec![]);
/// // Read the latest results:
/// let risks = loop_.latest_risks();
/// loop_.stop();
/// ```
pub struct RiskLoop {
    objects: Arc<RwLock<Vec<SemanticObject>>>,
    surfaces: Arc<RwLock<Vec<Surface>>>,
    risks: Arc<RwLock<Vec<RiskAssessment>>>,
    trigger_tx: mpsc::Sender<bool>,
    thread_handle: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl RiskLoop {
    /// Spawn the background risk-assessment loop.
    ///
    /// The loop immediately waits for a trigger (or a 5-second timeout) before
    /// running the first assessment.
    #[must_use]
    pub fn spawn(config: PhysicsConfig) -> Self {
        let objects: Arc<RwLock<Vec<SemanticObject>>> = Arc::new(RwLock::new(Vec::new()));
        let surfaces: Arc<RwLock<Vec<Surface>>> = Arc::new(RwLock::new(default_surfaces()));
        let risks: Arc<RwLock<Vec<RiskAssessment>>> = Arc::new(RwLock::new(Vec::new()));

        let (trigger_tx, trigger_rx) = mpsc::channel::<bool>();

        let objs_clone = Arc::clone(&objects);
        let surfs_clone = Arc::clone(&surfaces);
        let risks_clone = Arc::clone(&risks);

        let handle = std::thread::spawn(move || {
            loop {
                let should_run = match trigger_rx.recv_timeout(Duration::from_secs(5)) {
                    Ok(false) | Err(RecvTimeoutError::Disconnected) => break, // stop signal
                    Ok(true) | Err(RecvTimeoutError::Timeout) => true,
                };
                if !should_run {
                    break;
                }

                let objs = match objs_clone.read() {
                    Ok(guard) => guard.clone(),
                    Err(e) => {
                        warn!("risk_loop_objects_lock_poisoned: {e}");
                        continue;
                    }
                };
                if objs.is_empty() {
                    continue;
                }

                let surfs = match surfs_clone.read() {
                    Ok(guard) => guard.clone(),
                    Err(e) => {
                        warn!("risk_loop_surfaces_lock_poisoned: {e}");
                        continue;
                    }
                };

                // Sort objects by heuristic priority (highest first).
                let mut sorted = objs;
                sorted.sort_by(|a, b| {
                    object_priority(b)
                        .partial_cmp(&object_priority(a))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let mut sim = Simulator::new(config.clone());
                sim.set_surfaces(surfs);
                let new_risks = sim.assess_risks(&sorted);

                match risks_clone.write() {
                    Ok(mut guard) => *guard = new_risks,
                    Err(e) => warn!("risk_loop_risks_lock_poisoned: {e}"),
                }
            }
        });

        Self {
            objects,
            surfaces,
            risks,
            trigger_tx,
            thread_handle: Mutex::new(Some(handle)),
        }
    }

    /// Replace the current scene and trigger an immediate re-assessment.
    pub fn update_scene(&self, objects: Vec<SemanticObject>, surfaces: Vec<Surface>) {
        if let Ok(mut guard) = self.objects.write() {
            *guard = objects;
        }
        if let Ok(mut guard) = self.surfaces.write() {
            *guard = surfaces;
        }
        self.trigger();
    }

    /// Trigger an immediate re-assessment without changing scene data.
    pub fn trigger(&self) {
        let _ = self.trigger_tx.send(true);
    }

    /// Return a snapshot of the latest risk assessments.
    #[must_use]
    pub fn latest_risks(&self) -> Vec<RiskAssessment> {
        self.risks.read().map(|g| g.clone()).unwrap_or_default()
    }

    /// Signal the background thread to stop and wait for it to exit.
    pub fn stop(&self) {
        let _ = self.trigger_tx.send(false);
        if let Ok(mut guard) = self.thread_handle.lock()
            && let Some(handle) = guard.take()
        {
            let _ = handle.join();
        }
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

    // ── assess_risks_detailed ─────────────────────────────────────────────────

    #[test]
    fn test_detailed_empty_scene_returns_empty() {
        let sim = Simulator::new(PhysicsConfig::default());
        assert!(sim.assess_risks_detailed(&[]).is_empty());
    }

    #[test]
    fn test_detailed_falling_object_has_trajectory() {
        let mut sim = Simulator::new(PhysicsConfig::default());
        sim.set_surfaces(vec![]); // no floor → guaranteed fall
        let obj = make_object(
            40,
            "vase",
            [0.0, 5.0, 0.0],
            0.1,
            0.3,
            0.9,
            MaterialType::Glass,
        );
        let detailed = sim.assess_risks_detailed(&[obj]);
        assert!(
            !detailed.is_empty(),
            "elevated vase with no support must produce a risk"
        );
        // Trajectory must have at least one waypoint.
        assert!(
            !detailed[0].trajectory.waypoints.is_empty(),
            "trajectory must have waypoints"
        );
    }

    #[test]
    fn test_detailed_liquid_has_spill_zone() {
        let mut sim = Simulator::new(PhysicsConfig::default());
        sim.set_surfaces(vec![]); // no floor
        let obj = make_object(
            41,
            "water_glass",
            [0.0, 3.0, 0.0],
            0.05,
            0.2,
            0.9,
            MaterialType::Liquid,
        );
        let detailed = sim.assess_risks_detailed(&[obj]);
        assert!(!detailed.is_empty());
        assert!(
            detailed[0].trajectory.spill_zone.is_some(),
            "liquid object must have a spill zone"
        );
    }

    #[test]
    fn test_detailed_sorted_by_probability_descending() {
        let mut sim = Simulator::new(PhysicsConfig::default());
        sim.set_surfaces(vec![]); // all fall freely
        let objects = vec![
            make_object(
                50,
                "vase",
                [0.0, 10.0, 0.0],
                0.1,
                0.2,
                0.9,
                MaterialType::Glass,
            ),
            make_object(
                51,
                "book",
                [0.0, 2.0, 0.0],
                0.15,
                0.5,
                0.2,
                MaterialType::Wood,
            ),
        ];
        let detailed = sim.assess_risks_detailed(&objects);
        if detailed.len() >= 2 {
            assert!(
                detailed[0].base.probability >= detailed[1].base.probability,
                "results must be sorted by probability descending"
            );
        }
    }

    // ── object_priority ───────────────────────────────────────────────────────

    #[test]
    fn test_priority_elevated_fragile_is_high() {
        use super::object_priority;
        use atlas_core::semantic::{ObjectRelation, RelationType};
        let mut obj = make_object(
            60,
            "vase",
            [0.0, 4.0, 0.0],
            0.1,
            0.3,
            0.95,
            MaterialType::Glass,
        );
        obj.relationships.push(ObjectRelation {
            target_id: 0,
            relation_type: RelationType::OnTopOf,
        });
        let p = object_priority(&obj);
        assert!(
            p > 0.5,
            "elevated fragile object should have high priority, got {p}"
        );
    }

    #[test]
    fn test_priority_floor_level_low_fragility_is_low() {
        use super::object_priority;
        let obj = make_object(
            61,
            "block",
            [0.0, 0.1, 0.0],
            0.5,
            5.0,
            0.05,
            MaterialType::Metal,
        );
        let p = object_priority(&obj);
        assert!(
            p < 0.5,
            "low, robust object should have low priority, got {p}"
        );
    }

    #[test]
    fn test_priority_leaning_adds_bonus() {
        use super::object_priority;
        use atlas_core::semantic::{ObjectRelation, RelationType};
        let base = make_object(
            62,
            "lamp",
            [0.0, 1.0, 0.0],
            0.1,
            0.5,
            0.5,
            MaterialType::Plastic,
        );
        let mut leaning = base.clone();
        leaning.relationships.push(ObjectRelation {
            target_id: 0,
            relation_type: RelationType::Leaning,
        });
        assert!(
            object_priority(&leaning) > object_priority(&base),
            "leaning object should have higher priority"
        );
    }

    #[test]
    fn test_priority_bounded_zero_one() {
        use super::object_priority;
        use atlas_core::semantic::{ObjectRelation, RelationType};
        let mut obj = make_object(
            63,
            "any",
            [0.0, 100.0, 0.0],
            1.0,
            1.0,
            1.0,
            MaterialType::Glass,
        );
        obj.relationships = vec![
            ObjectRelation {
                target_id: 0,
                relation_type: RelationType::Leaning,
            },
            ObjectRelation {
                target_id: 1,
                relation_type: RelationType::OnTopOf,
            },
            ObjectRelation {
                target_id: 2,
                relation_type: RelationType::Hanging,
            },
        ];
        let p = object_priority(&obj);
        assert!((0.0..=1.0).contains(&p), "priority {p} out of [0, 1]");
    }

    // ── RiskLoop ──────────────────────────────────────────────────────────────

    #[test]
    fn test_risk_loop_starts_with_empty_risks() {
        let loop_ = super::RiskLoop::spawn(PhysicsConfig::default());
        assert!(
            loop_.latest_risks().is_empty(),
            "risk loop must start with no risks"
        );
        loop_.stop();
    }

    #[test]
    fn test_risk_loop_produces_risks_for_unstable_scene() {
        let loop_ = super::RiskLoop::spawn(PhysicsConfig::default());

        // Elevated vase with no support → should be at risk.
        let obj = make_object(
            70,
            "vase",
            [0.0, 5.0, 0.0],
            0.05,
            0.2,
            0.9,
            MaterialType::Glass,
        );
        loop_.update_scene(vec![obj], vec![]); // no surfaces → free fall

        // Poll for up to 2 seconds.
        let mut risks = vec![];
        for _ in 0..200 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            risks = loop_.latest_risks();
            if !risks.is_empty() {
                break;
            }
        }
        loop_.stop();

        assert!(!risks.is_empty(), "risk loop must eventually produce risks");
    }

    #[test]
    fn test_risk_loop_stop_does_not_panic() {
        let loop_ = super::RiskLoop::spawn(PhysicsConfig::default());
        loop_.stop(); // Should complete cleanly with no objects.
    }

    #[test]
    fn test_risk_loop_trigger_reruns_assessment() {
        let loop_ = super::RiskLoop::spawn(PhysicsConfig::default());
        // Trigger without objects → stays empty.
        loop_.trigger();
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(loop_.latest_risks().is_empty());
        loop_.stop();
    }
}
