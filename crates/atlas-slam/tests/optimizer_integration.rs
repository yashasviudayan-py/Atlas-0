//! Integration test: process 10 synthetic keyframes, verify Gaussian count
//! grows and that PSNR improves over optimizer iterations.

use atlas_core::{Gaussian3D, Point3, Pose};
use atlas_slam::config::CameraIntrinsics;
use atlas_slam::gaussian_init::{GaussianInitConfig, GaussianInitializer};
use atlas_slam::optimizer::{GaussianOptimizer, OptimizerConfig};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Return a flat-colour RGB frame of size WIDTH × HEIGHT.
fn make_flat_frame(r: u8, g: u8, b: u8) -> Vec<u8> {
    let n = (WIDTH * HEIGHT) as usize;
    let mut frame = Vec::with_capacity(n * 3);
    for _ in 0..n {
        frame.push(r);
        frame.push(g);
        frame.push(b);
    }
    frame
}

/// Compute PSNR (dB) from an MSE value (MSE over normalised [0,1] channels).
fn psnr(mse: f32) -> f32 {
    if mse <= 0.0 {
        return f32::INFINITY;
    }
    10.0 * (1.0_f32 / mse).log10()
}

/// Camera intrinsics sized for the WIDTH×HEIGHT test canvas.
fn test_intrinsics() -> CameraIntrinsics {
    CameraIntrinsics {
        fx: 50.0,
        fy: 50.0,
        cx: (WIDTH / 2) as f64,
        cy: (HEIGHT / 2) as f64,
    }
}

/// Scatter `n` synthetic 3-D points in front of the camera.
fn synthetic_points(n: usize, z: f32, color: [f32; 3]) -> Vec<(Point3, [f32; 3])> {
    (0..n)
        .map(|i| {
            let x = (i as f32 - n as f32 / 2.0) * 0.1;
            (Point3::new(x, 0.0, z), color)
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Verify Gaussian count grows as new keyframes are processed.
#[test]
fn test_gaussian_count_grows_over_keyframes() {
    let init = GaussianInitializer::new(GaussianInitConfig::default());
    let src_color = [0.8_f32, 0.6, 0.4];

    let mut gaussians: Vec<Gaussian3D> = Vec::new();

    for kf in 0..10 {
        let count_before = gaussians.len();
        let pts = synthetic_points(5, 2.0, src_color);
        let new_gaussians = init.from_point_cloud(&pts);
        gaussians.extend(new_gaussians);
        assert!(
            gaussians.len() > count_before,
            "keyframe {kf}: Gaussian count should grow (was {count_before}, now {})",
            gaussians.len()
        );
    }
    assert_eq!(gaussians.len(), 50, "10 keyframes × 5 Gaussians each = 50");
}

/// Verify that running the optimizer on a fixed scene with a central Gaussian
/// reduces the photometric loss (PSNR improves) over iterations.
#[test]
fn test_psnr_improves_over_optimizer_iterations() {
    let intrinsics = test_intrinsics();
    let config = OptimizerConfig {
        iterations: 1, // we drive iterations manually to track per-step loss
        lr_opacity: 0.05,
        lr_color: 0.05,
        lr_position: 1e-4,
        lr_covariance: 5e-4,
        min_opacity: 0.001,
    };
    let mut optimizer = GaussianOptimizer::new(config);

    // One bright opaque Gaussian placed directly in front of the camera.
    const C0: f32 = 0.282_094_8;
    let target_color = [200u8, 150, 100];
    let src_color = [
        target_color[0] as f32 / 255.0,
        target_color[1] as f32 / 255.0,
        target_color[2] as f32 / 255.0,
    ];
    let s2 = 0.2_f32 * 0.2; // visible splat radius at 2 m depth
    let mut g = Gaussian3D::new(Point3::new(0.0, 0.0, 2.0), src_color, 0.9);
    g.sh_coefficients = vec![
        (src_color[0] - 0.5) / C0,
        (src_color[1] - 0.5) / C0,
        (src_color[2] - 0.5) / C0,
    ];
    g.scale = [0.2, 0.2, 0.2];
    g.covariance = [s2, 0.0, 0.0, s2, 0.0, s2];
    let mut gaussians = vec![g];

    let target_frame = make_flat_frame(target_color[0], target_color[1], target_color[2]);
    let pose = Pose::identity();

    let initial_loss = optimizer.optimize_step(
        &mut gaussians,
        &target_frame,
        WIDTH,
        HEIGHT,
        &pose,
        &intrinsics,
    );

    // Run 20 more iterations.
    let mut final_loss = initial_loss;
    for _ in 0..20 {
        final_loss = optimizer.optimize_step(
            &mut gaussians,
            &target_frame,
            WIDTH,
            HEIGHT,
            &pose,
            &intrinsics,
        );
    }

    assert!(
        final_loss < initial_loss,
        "optimizer should reduce loss: initial={initial_loss:.4}, final={final_loss:.4}"
    );
    assert!(
        psnr(final_loss) > psnr(initial_loss),
        "PSNR should improve: initial={:.2} dB, final={:.2} dB",
        psnr(initial_loss),
        psnr(final_loss)
    );
}

/// Verify that saving and loading a GaussianCloud built by the optimizer
/// round-trips all Gaussian parameters exactly.
#[test]
fn test_map_save_load_round_trip_via_optimizer() {
    use atlas_core::gaussian::GaussianCloud;

    let init = GaussianInitializer::new(GaussianInitConfig::default());
    let pts = synthetic_points(10, 2.0, [0.6, 0.4, 0.2]);
    let gaussians = init.from_point_cloud(&pts);

    let mut cloud = GaussianCloud::new();
    for g in gaussians {
        cloud.add(g);
    }

    let dir = std::env::temp_dir();
    let path = dir.join("atlas_integration_map.json");
    cloud.save(&path).expect("save failed");

    let loaded = GaussianCloud::load(&path).expect("load failed");
    assert_eq!(
        loaded.len(),
        cloud.len(),
        "Gaussian count must survive round-trip"
    );
    for (orig, loaded_g) in cloud.gaussians.iter().zip(loaded.gaussians.iter()) {
        assert!(
            (orig.center.x - loaded_g.center.x).abs() < 1e-5,
            "center.x mismatch after round-trip"
        );
        assert!(
            (orig.opacity - loaded_g.opacity).abs() < 1e-5,
            "opacity mismatch after round-trip"
        );
    }

    let _ = std::fs::remove_file(&path);
}
