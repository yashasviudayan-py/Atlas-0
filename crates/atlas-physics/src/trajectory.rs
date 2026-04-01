//! Trajectory recording for simulated rigid body motion.
//!
//! [`TrajectoryRecorder`] captures sampled body states during a simulation run
//! and computes derived quantities on finalisation:
//!
//! - **Impact energy** — kinetic energy at the moment of first surface contact.
//! - **Spill zone** — cone-projected disk radius for liquid-containing objects.
//!
//! # Usage
//!
//! ```
//! use atlas_physics::trajectory::TrajectoryRecorder;
//! use atlas_core::spatial::Point3;
//!
//! let mut recorder = TrajectoryRecorder::new(5).with_liquid(false);
//! // … call recorder.record_step(&body) inside the simulation loop …
//! let result = recorder.finish(Some(Point3::new(1.0, 0.0, 2.0)));
//! ```

use atlas_core::spatial::Point3;
use serde::{Deserialize, Serialize};

use crate::rigid_body::RigidBody;

// ─── TrajectoryPoint ──────────────────────────────────────────────────────────

/// A sampled state of a rigid body at one simulation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    /// Body position `[x, y, z]` (metres) at this sample.
    pub position: [f32; 3],
    /// Body velocity `[vx, vy, vz]` (m/s) at this sample.
    pub velocity: [f32; 3],
}

// ─── SpillZone ────────────────────────────────────────────────────────────────

/// Approximate disk-shaped spill zone for liquid-containing objects.
///
/// The zone is centred at the projected impact position and has a radius
/// derived from the impact kinetic energy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpillZone {
    /// Projected centre of the spill on the landing surface (metres).
    pub centre: Point3,
    /// Estimated spill radius (metres), clamped to `[0.05, 2.0]`.
    pub radius_m: f32,
}

// ─── TrajectoryResult ─────────────────────────────────────────────────────────

/// Outcome of one simulated trajectory.
///
/// Produced by [`TrajectoryRecorder::finish`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryResult {
    /// Sparse waypoints sampled every `sample_every` steps.
    pub waypoints: Vec<TrajectoryPoint>,
    /// Final resting position or exit-bounds position.
    pub impact_point: Option<Point3>,
    /// Kinetic energy (joules) at the moment of first surface contact.
    /// `0.0` if the body never contacted a surface.
    pub impact_energy_joules: f32,
    /// Estimated spill zone for liquid objects; `None` for non-liquids.
    pub spill_zone: Option<SpillZone>,
}

// ─── TrajectoryRecorder ───────────────────────────────────────────────────────

/// Records a rigid body trajectory during a simulation run.
///
/// Sampling is sparse (every `sample_every` steps) to keep memory bounded.
/// The recorder also computes impact kinetic energy and (optionally) a spill
/// zone for liquid containers.
///
/// # Example
/// ```
/// use atlas_physics::trajectory::TrajectoryRecorder;
///
/// let rec = TrajectoryRecorder::new(10);
/// assert_eq!(rec.sample_every(), 10);
/// ```
pub struct TrajectoryRecorder {
    sample_every: u32,
    waypoints: Vec<TrajectoryPoint>,
    impact_ke: f32,
    step: u32,
    is_liquid: bool,
    impact_recorded: bool,
}

impl TrajectoryRecorder {
    /// Create a new recorder that samples every `sample_every` simulation steps.
    ///
    /// `sample_every` is clamped to a minimum of 1.
    #[must_use]
    pub fn new(sample_every: u32) -> Self {
        Self {
            sample_every: sample_every.max(1),
            waypoints: Vec::new(),
            impact_ke: 0.0,
            step: 0,
            is_liquid: false,
            impact_recorded: false,
        }
    }

    /// Configure whether the body contains liquid (enables spill zone output).
    #[must_use]
    pub fn with_liquid(mut self, is_liquid: bool) -> Self {
        self.is_liquid = is_liquid;
        self
    }

    /// The configured sampling interval.
    #[must_use]
    pub fn sample_every(&self) -> u32 {
        self.sample_every
    }

    /// Record the body's state for this simulation step.
    ///
    /// Call once per integration step.  Only every `sample_every`-th call
    /// actually appends a waypoint.
    pub fn record_step(&mut self, body: &RigidBody) {
        if self.step.is_multiple_of(self.sample_every) {
            self.waypoints.push(TrajectoryPoint {
                position: body.position,
                velocity: body.velocity,
            });
        }
        self.step += 1;
    }

    /// Mark the current state as the first surface contact and record its KE.
    ///
    /// Only the first call takes effect (subsequent calls are no-ops).
    /// Also appends a waypoint if the current step is not already sampled.
    pub fn record_impact(&mut self, body: &RigidBody) {
        if self.impact_recorded {
            return;
        }
        self.impact_ke = kinetic_energy(body);
        self.impact_recorded = true;

        // Append waypoint at the impact step if not already scheduled.
        if !self.step.is_multiple_of(self.sample_every) {
            self.waypoints.push(TrajectoryPoint {
                position: body.position,
                velocity: body.velocity,
            });
        }
    }

    /// Finalise the trajectory and return a [`TrajectoryResult`].
    ///
    /// `impact_point` is the final resting or out-of-bounds position.
    #[must_use]
    pub fn finish(self, impact_point: Option<Point3>) -> TrajectoryResult {
        let spill_zone = if self.is_liquid {
            impact_point.as_ref().map(|ip| SpillZone {
                centre: *ip,
                radius_m: spill_radius(self.impact_ke),
            })
        } else {
            None
        };

        TrajectoryResult {
            waypoints: self.waypoints,
            impact_point,
            impact_energy_joules: self.impact_ke,
            spill_zone,
        }
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Translational kinetic energy: ½mv².
#[inline]
fn kinetic_energy(body: &RigidBody) -> f32 {
    let v2 = body.velocity[0].powi(2) + body.velocity[1].powi(2) + body.velocity[2].powi(2);
    0.5 * body.mass * v2
}

/// Estimate spill radius from impact kinetic energy.
///
/// Heuristic: `r = √(E / 5)` clamped to `[0.05, 2.0]` metres.
fn spill_radius(impact_ke: f32) -> f32 {
    (impact_ke / 5.0).sqrt().clamp(0.05, 2.0)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rigid_body::{BoundingShape, RigidBody};

    fn body_with_vy(vy: f32) -> RigidBody {
        RigidBody {
            id: 1,
            position: [0.0, 5.0, 0.0],
            velocity: [0.0, vy, 0.0],
            angular_velocity: [0.0; 3],
            mass: 1.0,
            inertia: [1.0; 3],
            shape: BoundingShape::Sphere { radius: 0.1 },
            friction: 0.4,
            restitution: 0.3,
            is_static: false,
        }
    }

    // ── Constructor & accessors ───────────────────────────────────────────────

    #[test]
    fn test_new_recorder_empty_result() {
        let rec = TrajectoryRecorder::new(5);
        let result = rec.finish(None);
        assert!(result.waypoints.is_empty());
        assert!(result.impact_point.is_none());
        assert!(result.impact_energy_joules.abs() < 1e-6);
        assert!(result.spill_zone.is_none());
    }

    #[test]
    fn test_sample_every_min_is_one() {
        let rec = TrajectoryRecorder::new(0);
        assert_eq!(rec.sample_every(), 1);
    }

    #[test]
    fn test_sample_every_preserved() {
        let rec = TrajectoryRecorder::new(7);
        assert_eq!(rec.sample_every(), 7);
    }

    // ── record_step sampling ──────────────────────────────────────────────────

    #[test]
    fn test_sampling_every_five_steps() {
        let mut rec = TrajectoryRecorder::new(5);
        let body = body_with_vy(-1.0);
        for _ in 0..10 {
            rec.record_step(&body);
        }
        // Steps 0 and 5 are sampled → 2 waypoints.
        let result = rec.finish(None);
        assert_eq!(result.waypoints.len(), 2, "expected 2 sampled waypoints");
    }

    #[test]
    fn test_sample_every_one_records_all_steps() {
        let mut rec = TrajectoryRecorder::new(1);
        let body = body_with_vy(-1.0);
        for _ in 0..5 {
            rec.record_step(&body);
        }
        assert_eq!(rec.finish(None).waypoints.len(), 5);
    }

    #[test]
    fn test_waypoints_store_position_and_velocity() {
        let mut rec = TrajectoryRecorder::new(1);
        let body = body_with_vy(-3.0);
        rec.record_step(&body);
        let result = rec.finish(None);
        assert_eq!(result.waypoints[0].position, [0.0, 5.0, 0.0]);
        assert_eq!(result.waypoints[0].velocity, [0.0, -3.0, 0.0]);
    }

    // ── record_impact ─────────────────────────────────────────────────────────

    #[test]
    fn test_record_impact_sets_ke() {
        let mut rec = TrajectoryRecorder::new(100); // sparse sampling
        let body = body_with_vy(-5.0); // KE = ½ × 1 × 25 = 12.5 J
        rec.record_impact(&body);
        let result = rec.finish(None);
        assert!(
            (result.impact_energy_joules - 12.5).abs() < 1e-3,
            "expected 12.5 J, got {}",
            result.impact_energy_joules
        );
    }

    #[test]
    fn test_record_impact_only_records_first_call() {
        let mut rec = TrajectoryRecorder::new(100);
        let high_ke = body_with_vy(-10.0); // KE = 50 J
        let low_ke = body_with_vy(-1.0); // KE = 0.5 J
        rec.record_impact(&high_ke);
        rec.record_impact(&low_ke); // second call → no-op
        let result = rec.finish(None);
        assert!(
            result.impact_energy_joules > 10.0,
            "second record_impact call must be a no-op"
        );
    }

    // ── finish / impact_point ─────────────────────────────────────────────────

    #[test]
    fn test_impact_point_stored_in_result() {
        let rec = TrajectoryRecorder::new(10);
        let pt = Point3::new(1.5, 0.0, 2.5);
        let result = rec.finish(Some(pt));
        let ip = result.impact_point.unwrap();
        assert!((ip.x - 1.5).abs() < 1e-6);
        assert!((ip.z - 2.5).abs() < 1e-6);
    }

    // ── spill zone ────────────────────────────────────────────────────────────

    #[test]
    fn test_no_spill_zone_for_non_liquid() {
        let rec = TrajectoryRecorder::new(1).with_liquid(false);
        let result = rec.finish(Some(Point3::new(0.0, 0.0, 0.0)));
        assert!(result.spill_zone.is_none());
    }

    #[test]
    fn test_spill_zone_present_for_liquid() {
        let mut rec = TrajectoryRecorder::new(1).with_liquid(true);
        let body = body_with_vy(-5.0);
        rec.record_impact(&body);
        let result = rec.finish(Some(Point3::new(0.0, 0.0, 0.0)));
        assert!(
            result.spill_zone.is_some(),
            "liquid object must have a spill zone"
        );
    }

    #[test]
    fn test_spill_radius_in_valid_range() {
        let mut rec = TrajectoryRecorder::new(1).with_liquid(true);
        let body = body_with_vy(-5.0);
        rec.record_impact(&body);
        let result = rec.finish(Some(Point3::new(0.0, 0.0, 0.0)));
        let r = result.spill_zone.unwrap().radius_m;
        assert!(
            (0.05..=2.0).contains(&r),
            "spill radius {r} outside [0.05, 2.0]"
        );
    }

    #[test]
    fn test_spill_radius_increases_with_energy() {
        // Low velocity → small spill; high velocity → large spill.
        let make_result = |vy: f32| {
            let mut rec = TrajectoryRecorder::new(1).with_liquid(true);
            rec.record_impact(&body_with_vy(vy));
            rec.finish(Some(Point3::new(0.0, 0.0, 0.0)))
                .spill_zone
                .unwrap()
                .radius_m
        };
        let r_low = make_result(-1.0); // KE = 0.5 J
        let r_high = make_result(-10.0); // KE = 50 J
        assert!(
            r_high >= r_low,
            "high energy ({r_high:.3}) must produce radius >= low energy ({r_low:.3})"
        );
    }

    #[test]
    fn test_liquid_no_impact_no_spill() {
        // No record_impact called → KE = 0 → minimum radius spill.
        let rec = TrajectoryRecorder::new(1).with_liquid(true);
        let result = rec.finish(Some(Point3::new(0.0, 0.0, 0.0)));
        // KE = 0 → spill_radius = √(0/5) = 0, clamped to 0.05.
        let sz = result.spill_zone.unwrap();
        assert!((sz.radius_m - 0.05).abs() < 1e-5);
    }
}
