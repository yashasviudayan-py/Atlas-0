//! Perturbation strategies for "what-if" risk assessment.
//!
//! A [`Perturbation`] encodes an initial condition (velocity impulse, extra
//! horizontal gravity) applied to a rigid body before the simulator runs.
//! Using multiple perturbations per object lets the risk loop distinguish
//! between *inherent* instability (gravity-only) and *conditional* instability
//! (knock, tilt, support removal).
//!
//! # Workflow
//!
//! 1. Obtain a `Vec<Perturbation>` from [`Perturbation::standard_set`].
//! 2. For each perturbation, run the simulation and record the resulting
//!    displacement.
//! 3. Call [`Perturbation::combined_score`] with `(displacement, weight)` pairs
//!    to get a single probability in `[0, 1]`.

use serde::{Deserialize, Serialize};

// ─── PerturbationKind ─────────────────────────────────────────────────────────

/// Category of the perturbation scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerturbationKind {
    /// No external force — only gravity acts.  Tests natural stability.
    GravityOnly,
    /// Small horizontal push, simulating a bump or vibration.
    HorizontalPush,
    /// Tilted effective gravity, simulating an uneven or shaking surface.
    SurfaceTilt,
    /// Combined tilt + push, proxying collapse of the supporting structure.
    SupportRemoval,
}

// ─── Perturbation ─────────────────────────────────────────────────────────────

/// A single perturbation scenario applied before simulation.
///
/// # Example
/// ```
/// use atlas_physics::perturbations::Perturbation;
///
/// let set = Perturbation::standard_set();
/// assert_eq!(set.len(), 4);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Perturbation {
    /// Category of this perturbation.
    pub kind: PerturbationKind,
    /// Initial linear velocity impulse `[vx, vy, vz]` (m/s) applied to the
    /// body at the start of the simulation run.
    pub initial_velocity: [f32; 3],
    /// Extra horizontal gravity `[ax, 0, az]` (m/s²) applied each step in
    /// addition to the simulator's downward gravity.  Use `[0, 0, 0]` for no
    /// override.
    pub extra_gravity: [f32; 3],
    /// Weighting factor `∈ (0, 1]` used when combining perturbation results.
    pub weight: f32,
}

impl Perturbation {
    /// Standard four-perturbation set used by the risk assessment pipeline.
    ///
    /// The set covers:
    ///
    /// | # | Kind | Initial vel | Extra gravity | Weight |
    /// |---|------|-------------|---------------|--------|
    /// | 1 | `GravityOnly` | `[0,0,0]` | `[0,0,0]` | 0.30 |
    /// | 2 | `HorizontalPush` | `[0.5,0,0.5]` | `[0,0,0]` | 0.30 |
    /// | 3 | `SurfaceTilt` | `[0,0,0]` | `[2,0,1]` | 0.20 |
    /// | 4 | `SupportRemoval` | `[1,-0.5,0.5]` | `[3,0,2]` | 0.20 |
    ///
    /// Weights sum to 1.0 so that [`combined_score`] naturally normalises.
    ///
    /// [`combined_score`]: Perturbation::combined_score
    #[must_use]
    pub fn standard_set() -> Vec<Self> {
        vec![
            // 1. Gravity only: is the object unstable in its rest pose?
            Self {
                kind: PerturbationKind::GravityOnly,
                initial_velocity: [0.0, 0.0, 0.0],
                extra_gravity: [0.0, 0.0, 0.0],
                weight: 0.30,
            },
            // 2. Small horizontal push (bump / vibration simulation).
            Self {
                kind: PerturbationKind::HorizontalPush,
                initial_velocity: [0.5, 0.0, 0.5],
                extra_gravity: [0.0, 0.0, 0.0],
                weight: 0.30,
            },
            // 3. Surface tilt — effective gravity rotated ~11° off vertical.
            Self {
                kind: PerturbationKind::SurfaceTilt,
                initial_velocity: [0.0, 0.0, 0.0],
                extra_gravity: [2.0, 0.0, 1.0],
                weight: 0.20,
            },
            // 4. Support removal — aggressive tilt + push to proxy shelf collapse.
            Self {
                kind: PerturbationKind::SupportRemoval,
                initial_velocity: [1.0, -0.5, 0.5],
                extra_gravity: [3.0, 0.0, 2.0],
                weight: 0.20,
            },
        ]
    }

    /// Compute a combined probability from per-perturbation `(displacement, weight)` pairs.
    ///
    /// Each displacement is first mapped to a probability via a hyperbolic
    /// function (`p = d / (d + 0.5)`), then combined as a weighted average.
    ///
    /// Returns `0.0` when `results` is empty or total weight is zero.
    ///
    /// # Example
    /// ```
    /// use atlas_physics::perturbations::Perturbation;
    ///
    /// // Two perturbations: one stable, one displaced 2 m.
    /// let score = Perturbation::combined_score(&[(0.0, 0.5), (2.0, 0.5)]);
    /// assert!(score > 0.0 && score < 1.0);
    /// ```
    #[must_use]
    pub fn combined_score(results: &[(f32, f32)]) -> f32 {
        let total_weight: f32 = results.iter().map(|(_, w)| w).sum();
        if total_weight < 1e-6 {
            return 0.0;
        }
        let weighted_sum: f32 = results
            .iter()
            .map(|(disp, w)| displacement_to_probability(*disp) * w)
            .sum();
        (weighted_sum / total_weight).clamp(0.0, 1.0)
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Map displacement (metres) to a probability via hyperbolic function.
///
/// `p = d / (d + 0.5)` gives p ≈ 0.09 at d = 0.05 m and p ≈ 0.91 at d = 5 m.
fn displacement_to_probability(d: f32) -> f32 {
    (d / (d + 0.5)).clamp(0.0, 1.0)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── standard_set ──────────────────────────────────────────────────────────

    #[test]
    fn test_standard_set_has_four_perturbations() {
        assert_eq!(Perturbation::standard_set().len(), 4);
    }

    #[test]
    fn test_standard_set_weights_positive() {
        for p in Perturbation::standard_set() {
            assert!(
                p.weight > 0.0,
                "all perturbations must have positive weight (kind={:?})",
                p.kind
            );
        }
    }

    #[test]
    fn test_standard_set_weights_sum_to_one() {
        let total: f32 = Perturbation::standard_set().iter().map(|p| p.weight).sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "weights should sum to 1.0, got {total}"
        );
    }

    #[test]
    fn test_all_four_kinds_present() {
        let kinds: Vec<PerturbationKind> = Perturbation::standard_set()
            .iter()
            .map(|p| p.kind)
            .collect();
        assert!(kinds.contains(&PerturbationKind::GravityOnly));
        assert!(kinds.contains(&PerturbationKind::HorizontalPush));
        assert!(kinds.contains(&PerturbationKind::SurfaceTilt));
        assert!(kinds.contains(&PerturbationKind::SupportRemoval));
    }

    #[test]
    fn test_gravity_only_has_zero_velocity() {
        let p = Perturbation::standard_set()
            .into_iter()
            .find(|p| p.kind == PerturbationKind::GravityOnly)
            .unwrap();
        assert_eq!(p.initial_velocity, [0.0, 0.0, 0.0]);
        assert_eq!(p.extra_gravity, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_horizontal_push_has_nonzero_velocity() {
        let p = Perturbation::standard_set()
            .into_iter()
            .find(|p| p.kind == PerturbationKind::HorizontalPush)
            .unwrap();
        let speed_sq: f32 = p.initial_velocity.iter().map(|v| v * v).sum();
        assert!(
            speed_sq > 0.0,
            "horizontal push must have a non-zero velocity"
        );
    }

    #[test]
    fn test_surface_tilt_has_extra_gravity() {
        let p = Perturbation::standard_set()
            .into_iter()
            .find(|p| p.kind == PerturbationKind::SurfaceTilt)
            .unwrap();
        let g_sq: f32 = p.extra_gravity.iter().map(|g| g * g).sum();
        assert!(g_sq > 0.0, "surface tilt must have non-zero extra gravity");
    }

    #[test]
    fn test_support_removal_has_both_velocity_and_gravity() {
        let p = Perturbation::standard_set()
            .into_iter()
            .find(|p| p.kind == PerturbationKind::SupportRemoval)
            .unwrap();
        let speed_sq: f32 = p.initial_velocity.iter().map(|v| v * v).sum();
        let g_sq: f32 = p.extra_gravity.iter().map(|g| g * g).sum();
        assert!(speed_sq > 0.0);
        assert!(g_sq > 0.0);
    }

    // ── combined_score ────────────────────────────────────────────────────────

    #[test]
    fn test_combined_score_zero_displacement_is_zero() {
        let results = vec![(0.0, 0.3), (0.0, 0.3), (0.0, 0.2), (0.0, 0.2)];
        assert!(Perturbation::combined_score(&results).abs() < 1e-6);
    }

    #[test]
    fn test_combined_score_empty_is_zero() {
        assert!(Perturbation::combined_score(&[]).abs() < 1e-6);
    }

    #[test]
    fn test_combined_score_large_displacement_near_one() {
        let score = Perturbation::combined_score(&[(100.0, 1.0)]);
        assert!(
            score > 0.99,
            "very large displacement should give score ≈ 1.0, got {score}"
        );
    }

    #[test]
    fn test_combined_score_is_monotone() {
        let s1 = Perturbation::combined_score(&[(0.1, 1.0)]);
        let s2 = Perturbation::combined_score(&[(1.0, 1.0)]);
        let s3 = Perturbation::combined_score(&[(10.0, 1.0)]);
        assert!(s1 < s2, "score must increase with displacement");
        assert!(s2 < s3);
    }

    #[test]
    fn test_combined_score_bounded_zero_one() {
        for d in [0.0f32, 0.001, 0.1, 1.0, 10.0, 1000.0] {
            let s = Perturbation::combined_score(&[(d, 1.0)]);
            assert!((0.0..=1.0).contains(&s), "score {s} out of [0,1] for d={d}");
        }
    }

    #[test]
    fn test_combined_score_high_weight_dominates() {
        // One zero-displacement result with low weight, one large with high weight.
        let s = Perturbation::combined_score(&[(0.0, 0.1), (10.0, 0.9)]);
        assert!(
            s > 0.5,
            "high-weight large displacement should dominate: got {s}"
        );
    }

    #[test]
    fn test_combined_score_zero_total_weight_is_zero() {
        let s = Perturbation::combined_score(&[(5.0, 0.0), (5.0, 0.0)]);
        assert!(s.abs() < 1e-6);
    }
}
