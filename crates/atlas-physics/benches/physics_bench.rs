//! Criterion benchmarks for the atlas-physics simulation pipeline.
//!
//! Budget (from DEVELOPMENT_PLAN.md):
//!   Simulation of 50 objects × 1000 steps < 10 ms.

use atlas_core::semantic::{MaterialProperties, MaterialType, SemanticObject};
use atlas_core::spatial::Point3;
use atlas_physics::{
    collision::CollisionDetector,
    config::PhysicsConfig,
    integrator::Integrator,
    rigid_body::RigidBody,
    simulator::Simulator,
    surfaces::{PlaneType, Surface},
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn make_objects(n: usize) -> Vec<SemanticObject> {
    (0..n)
        .map(|i| {
            let x = (i % 10) as f32 * 0.5;
            let z = (i / 10) as f32 * 0.5;
            SemanticObject {
                id: i as u64,
                label: format!("obj_{i}"),
                position: Point3::new(x, 1.0, z),
                bbox_min: Point3::new(x - 0.1, 0.9, z - 0.1),
                bbox_max: Point3::new(x + 0.1, 1.1, z + 0.1),
                properties: MaterialProperties {
                    mass_kg: 0.5,
                    friction: 0.4,
                    fragility: 0.5,
                    material: MaterialType::Wood,
                },
                confidence: 0.9,
                relationships: vec![],
            }
        })
        .collect()
}

fn floor_surface() -> Surface {
    Surface {
        normal: [0.0, 1.0, 0.0],
        offset: 0.0,
        plane_type: PlaneType::Floor,
        centre: Point3::new(0.0, 0.0, 0.0),
        inlier_count: usize::MAX,
    }
}

// ─── Benchmarks ───────────────────────────────────────────────────────────────

/// Benchmark: 50 objects × 1000 integration steps each.
/// Target: < 10 ms total.
fn bench_50_objects_1000_steps(c: &mut Criterion) {
    let config = PhysicsConfig::default(); // dt = 0.001
    let floor = floor_surface();
    let objects: Vec<SemanticObject> = make_objects(50);
    let bodies: Vec<RigidBody> = objects
        .iter()
        .map(RigidBody::from_semantic_object)
        .collect();

    c.bench_function("50_objects_1000_steps", |b| {
        b.iter(|| {
            for body_template in &bodies {
                let mut body = body_template.clone();
                body.velocity = [0.0, -0.1, 0.0]; // initial downward velocity
                for _ in 0..1000 {
                    let contacts: Vec<_> = std::iter::once(&floor)
                        .filter_map(|s| CollisionDetector::body_vs_surface(&body, s))
                        .collect();
                    Integrator::step(black_box(&mut body), black_box(&contacts), &config);
                }
            }
        });
    });
}

/// Benchmark: full assess_risks() pipeline on 50 objects (includes both perturbations).
fn bench_assess_risks_50_objects(c: &mut Criterion) {
    let sim = Simulator::new(PhysicsConfig::default());
    let objects = make_objects(50);

    c.bench_function("assess_risks_50_objects", |b| {
        b.iter(|| {
            black_box(sim.assess_risks(black_box(&objects)));
        });
    });
}

/// Benchmark: collision detection — 50 bodies vs 1 floor surface, 1000 rounds.
fn bench_collision_detection(c: &mut Criterion) {
    let floor = floor_surface();
    let objects = make_objects(50);
    let bodies: Vec<RigidBody> = objects
        .iter()
        .map(RigidBody::from_semantic_object)
        .collect();

    c.bench_function("collision_50_vs_floor_1000x", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                for body in &bodies {
                    let _ = black_box(CollisionDetector::body_vs_surface(body, &floor));
                }
            }
        });
    });
}

criterion_group!(
    benches,
    bench_50_objects_1000_steps,
    bench_assess_risks_50_objects,
    bench_collision_detection,
);
criterion_main!(benches);
