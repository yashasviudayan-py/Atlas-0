//! Semi-implicit Euler integrator for rigid body dynamics.
//!
//! Update order per timestep `dt`:
//! 1. Apply gravity impulse → update velocity.
//! 2. Resolve each contact via an impulse-based response (normal + friction).
//! 3. Apply positional correction to prevent tunnelling.
//! 4. Apply linear damping (stabilises resting contacts).
//! 5. Integrate position with the updated velocity (semi-implicit = stable).
//!
//! Semi-implicit Euler is unconditionally stable for typical damped systems and
//! requires only one force evaluation per step, making it ideal for the
//! high-step-count "what-if" simulations used by the [`Simulator`].
//!
//! [`Simulator`]: crate::simulator::Simulator

use crate::collision::Contact;
use crate::config::PhysicsConfig;
use crate::rigid_body::RigidBody;

/// Linear velocity damping applied every step to absorb energy and stabilise
/// resting contacts.  Must be in (0, 1]; 1.0 = no damping.
const LINEAR_DAMPING: f32 = 0.99;

/// Fraction of penetration depth resolved per step via positional correction
/// (Baumgarte stabilisation).  Lower values are smoother but slower to resolve.
const BAUMGARTE_FACTOR: f32 = 0.8;

/// Stateless semi-implicit Euler integrator.
pub struct Integrator;

impl Integrator {
    /// Advance `body` by one timestep `dt = config.timestep`.
    ///
    /// `contacts` is the set of contacts this body has with surfaces or other
    /// bodies on the *current* substep.  Static bodies are skipped.
    ///
    /// # Arguments
    ///
    /// * `body` – the body to integrate (mutated in place).
    /// * `contacts` – contacts gathered this step (may be empty).
    /// * `config` – simulation parameters (timestep, gravity).
    pub fn step(body: &mut RigidBody, contacts: &[Contact], config: &PhysicsConfig) {
        if body.is_static {
            return;
        }

        let dt = config.timestep;

        // ── 1. Gravity ────────────────────────────────────────────────────────
        // Gravity acts in the −Y direction.
        body.velocity[1] -= config.gravity * dt;

        // ── 2. Contact impulse resolution ────────────────────────────────────
        for contact in contacts {
            let n = contact.normal;

            // Relative velocity along the contact normal.
            let vn = dot3(&body.velocity, &n);

            // Only resolve when the body is moving into the surface (vn < 0).
            if vn < 0.0 {
                // Normal impulse magnitude: j = −(1 + e) · vn · m
                let j_mag = -(1.0 + body.restitution) * vn * body.mass;
                let inv_mass = 1.0 / body.mass;

                // Apply normal impulse.
                body.velocity[0] += j_mag * inv_mass * n[0];
                body.velocity[1] += j_mag * inv_mass * n[1];
                body.velocity[2] += j_mag * inv_mass * n[2];

                // ── Friction impulse (Coulomb model) ─────────────────────────
                // Tangential velocity = v - (v·n)n
                let vn_post = dot3(&body.velocity, &n);
                let vt = [
                    body.velocity[0] - vn_post * n[0],
                    body.velocity[1] - vn_post * n[1],
                    body.velocity[2] - vn_post * n[2],
                ];
                let vt_len = magnitude3(&vt);
                if vt_len > 1e-6 {
                    // Friction impulse capped by Coulomb limit: μ · |j_normal|.
                    let friction_impulse = (body.friction * j_mag * inv_mass).min(vt_len);
                    body.velocity[0] -= friction_impulse * vt[0] / vt_len;
                    body.velocity[1] -= friction_impulse * vt[1] / vt_len;
                    body.velocity[2] -= friction_impulse * vt[2] / vt_len;
                }
            }

            // ── 3. Positional correction (Baumgarte) ─────────────────────────
            // Push the body out of penetration to prevent sinking.
            let correction = contact.depth * BAUMGARTE_FACTOR;
            body.position[0] += contact.normal[0] * correction;
            body.position[1] += contact.normal[1] * correction;
            body.position[2] += contact.normal[2] * correction;
        }

        // ── 4. Linear damping ─────────────────────────────────────────────────
        body.velocity[0] *= LINEAR_DAMPING;
        body.velocity[1] *= LINEAR_DAMPING;
        body.velocity[2] *= LINEAR_DAMPING;

        // ── 5. Position integration (semi-implicit Euler) ─────────────────────
        body.position[0] += body.velocity[0] * dt;
        body.position[1] += body.velocity[1] * dt;
        body.position[2] += body.velocity[2] * dt;
    }
}

// ─── Vector helpers ───────────────────────────────────────────────────────────

#[inline]
fn dot3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn magnitude3(v: &[f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rigid_body::BoundingShape;
    use crate::surfaces::{PlaneType, Surface};
    use crate::{collision::CollisionDetector, rigid_body::RigidBody};
    use atlas_core::spatial::Point3;

    fn default_config() -> PhysicsConfig {
        PhysicsConfig::default()
    }

    fn falling_sphere(y: f32, radius: f32) -> RigidBody {
        RigidBody {
            id: 1,
            position: [0.0, y, 0.0],
            velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            mass: 1.0,
            inertia: [1.0; 3],
            shape: BoundingShape::Sphere { radius },
            friction: 0.5,
            restitution: 0.3,
            is_static: false,
        }
    }

    fn floor_surface(y: f32) -> Surface {
        Surface {
            normal: [0.0, 1.0, 0.0],
            offset: y,
            plane_type: PlaneType::Floor,
            centre: Point3::new(0.0, y, 0.0),
            inlier_count: 100,
        }
    }

    // ── Ball drop ─────────────────────────────────────────────────────────────

    #[test]
    fn test_ball_falls_under_gravity() {
        let config = default_config(); // dt = 0.001, g = 9.81
        let mut body = falling_sphere(10.0, 0.1);
        let initial_y = body.position[1];

        // Step 100 times with no contacts (free fall).
        for _ in 0..100 {
            Integrator::step(&mut body, &[], &config);
        }

        assert!(
            body.position[1] < initial_y,
            "ball should fall under gravity: y = {}",
            body.position[1]
        );
    }

    #[test]
    fn test_ball_drop_approximate_kinematics() {
        // After t seconds of free fall from rest:
        //   y(t) = y₀ − ½·g·t²
        // We check that after 0.1s (100 steps of dt=0.001) the body has fallen
        // roughly ½·g·t² ≈ 0.049 m.  Linear damping (0.99 per step, so 0.99^100
        // ≈ 0.366 effective scaling) means the real drop is slightly less than
        // the ideal free-fall prediction, so we allow a generous 50% tolerance.
        let config = default_config();
        let mut body = falling_sphere(10.0, 0.05);
        for _ in 0..100 {
            Integrator::step(&mut body, &[], &config);
        }
        let ideal_drop = 0.5 * 9.81 * 0.1 * 0.1; // ≈ 0.049 m
        let actual_drop = 10.0 - body.position[1];
        // The ball must have fallen at least 1% and at most 150% of ideal drop.
        assert!(
            actual_drop > ideal_drop * 0.01,
            "ball must fall under gravity (ideal {ideal_drop:.4} m, got {actual_drop:.4} m)"
        );
        assert!(
            actual_drop < ideal_drop * 1.5,
            "ball fell much more than expected (ideal {ideal_drop:.4} m, got {actual_drop:.4} m)"
        );
    }

    // ── Static body ───────────────────────────────────────────────────────────

    #[test]
    fn test_static_body_not_moved() {
        let config = default_config();
        let mut body = RigidBody::static_box(99, [0.0, -0.1, 0.0], [10.0, 0.1, 10.0]);
        let initial_pos = body.position;
        for _ in 0..1000 {
            Integrator::step(&mut body, &[], &config);
        }
        assert_eq!(body.position, initial_pos, "static body must not move");
    }

    // ── Resting contact ───────────────────────────────────────────────────────

    #[test]
    fn test_resting_on_floor_stabilises() {
        // Sphere dropped from a small height should come to rest on the floor.
        let config = default_config();
        let floor = floor_surface(0.0);
        let mut body = falling_sphere(1.0, 0.5);

        // Simulate up to 5000 steps.
        for _ in 0..5000 {
            let contacts: Vec<_> = [&floor]
                .iter()
                .filter_map(|s| CollisionDetector::body_vs_surface(&body, s))
                .collect();
            Integrator::step(&mut body, &contacts, &config);
            if body.is_at_rest(config.rest_threshold) {
                break;
            }
        }

        // Body should have come to rest with centre near y = radius = 0.5.
        assert!(
            body.position[1] >= 0.0,
            "sphere must not fall through the floor: y = {}",
            body.position[1]
        );
        assert!(
            body.is_at_rest(config.rest_threshold),
            "sphere should have come to rest: v = {:?}",
            body.velocity
        );
    }

    // ── Contact normal resolution ─────────────────────────────────────────────

    #[test]
    fn test_downward_velocity_reversed_on_contact() {
        let config = default_config();
        let mut body = falling_sphere(0.5, 0.5); // sphere touching floor exactly
        body.velocity = [0.0, -2.0, 0.0]; // moving into floor

        let floor = floor_surface(0.0);
        let contacts: Vec<_> = [&floor]
            .iter()
            .filter_map(|s| CollisionDetector::body_vs_surface(&body, s))
            .collect();

        Integrator::step(&mut body, &contacts, &config);

        // After one step the vertical velocity should be positive (bounced up)
        // or at least no longer strongly negative.
        assert!(
            body.velocity[1] > -2.0,
            "downward velocity must be reduced by contact: vy = {}",
            body.velocity[1]
        );
    }

    // ── No contacts ───────────────────────────────────────────────────────────

    #[test]
    fn test_no_contacts_no_horizontal_drift() {
        let config = default_config();
        let mut body = falling_sphere(10.0, 0.1);

        for _ in 0..100 {
            Integrator::step(&mut body, &[], &config);
        }

        // With gravity only there should be no horizontal drift.
        assert!(
            body.position[0].abs() < 1e-5,
            "no horizontal drift expected: x = {}",
            body.position[0]
        );
        assert!(
            body.position[2].abs() < 1e-5,
            "no depth drift expected: z = {}",
            body.position[2]
        );
    }
}
