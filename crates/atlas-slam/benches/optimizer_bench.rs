//! Criterion benchmarks for the 3DGS optimization step.
//!
//! Run with:  `cargo bench -p atlas-slam`
//!
//! Performance budget (from DEVELOPMENT_PLAN.md):
//!   single optimize_step < 10 ms for 100 K Gaussians

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use atlas_core::{Gaussian3D, Point3, Pose};
use atlas_slam::config::CameraIntrinsics;
use atlas_slam::optimizer::{GaussianOptimizer, OptimizerConfig};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

fn make_gaussians(n: usize) -> Vec<Gaussian3D> {
    const C0: f32 = 0.282_094_8;
    let color = [0.6_f32, 0.4, 0.2];
    let s2 = 0.01_f32 * 0.01;
    (0..n)
        .map(|i| {
            let x = (i as f32 % 100.0) * 0.1 - 5.0;
            let y = (i as f32 / 100.0) * 0.1 - 5.0;
            let mut g = Gaussian3D::new(Point3::new(x, y, 2.0), color, 0.5);
            g.sh_coefficients = vec![
                (color[0] - 0.5) / C0,
                (color[1] - 0.5) / C0,
                (color[2] - 0.5) / C0,
            ];
            g.scale = [0.01, 0.01, 0.01];
            g.covariance = [s2, 0.0, 0.0, s2, 0.0, s2];
            g
        })
        .collect()
}

fn bench_optimize_step(c: &mut Criterion) {
    let intrinsics = CameraIntrinsics::default();
    let pose = Pose::identity();
    let target = vec![128u8; (WIDTH * HEIGHT * 3) as usize];

    let mut group = c.benchmark_group("optimize_step");
    // Budget: < 10 ms for 100 K Gaussians.
    group.measurement_time(std::time::Duration::from_secs(10));

    for n in [1_000usize, 10_000, 100_000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut gaussians = make_gaussians(n);
            let mut opt = GaussianOptimizer::new(OptimizerConfig::default());
            b.iter(|| {
                opt.optimize_step(&mut gaussians, &target, WIDTH, HEIGHT, &pose, &intrinsics)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_optimize_step);
criterion_main!(benches);
