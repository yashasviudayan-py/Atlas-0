//! End-to-end SLAM pipeline example.
//!
//! Demonstrates the full Atlas-0 Phase 1 pipeline:
//!
//! ```text
//! synthetic frame → SLAM tracker → Gaussian map → EWA renderer → display
//! ```
//!
//! Per-stage latency is logged via `tracing` spans.
//!
//! # Run (terminal only)
//!
//! ```bash
//! cargo run --example live_slam -p atlas-slam
//! ```
//!
//! # Run with display window
//!
//! ```bash
//! cargo run --example live_slam -p atlas-slam --features viewer
//! ```

use std::time::{Duration, Instant};

use atlas_core::{Frame, Pose};
use atlas_slam::{
    config::{CameraIntrinsics, SlamConfig},
    renderer::SplatRenderer,
    tracker::Tracker,
};
use tracing::{info, warn};

#[cfg(feature = "viewer")]
use atlas_slam::{
    renderer::{compose_panels_bgra, gray_to_rgb},
    viewer::Viewer,
};
#[cfg(feature = "viewer")]
use tracing::error;

// ─── Pipeline constants ───────────────────────────────────────────────────────

const SLAM_WIDTH: u32 = 320;
const SLAM_HEIGHT: u32 = 240;
const TARGET_FPS: f64 = 30.0;
/// Number of synthetic frames to process before exiting.
const MAX_FRAMES: u64 = 200;

// ─── Entry point ─────────────────────────────────────────────────────────────

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Atlas-0 live SLAM example starting");

    let config = SlamConfig {
        min_features: 5,
        max_gaussians: 10_000,
        keyframe_translation_threshold: 0.05,
        keyframe_rotation_threshold: 0.05,
        optimization_iterations: 3,
        enable_loop_closure: false,
        camera: CameraIntrinsics {
            fx: 200.0,
            fy: 200.0,
            cx: f64::from(SLAM_WIDTH) / 2.0,
            cy: f64::from(SLAM_HEIGHT) / 2.0,
        },
        ..SlamConfig::default()
    };

    let intrinsics = config.camera.clone();
    let mut tracker = Tracker::new(config);
    let renderer = SplatRenderer::new(intrinsics, SLAM_WIDTH, SLAM_HEIGHT);

    // ── Optional display window (requires `--features viewer`) ───────────────
    #[cfg(feature = "viewer")]
    let mut viewer = {
        let dw = SLAM_WIDTH as usize * 3;
        let dh = SLAM_HEIGHT as usize;
        match Viewer::new("Atlas-0 SLAM  [camera | depth | splat]", dw, dh) {
            Ok(v) => {
                info!(dw, dh, "viewer window opened");
                Some(v)
            }
            Err(e) => {
                warn!(%e, "could not open viewer window — running headless");
                None
            }
        }
    };

    // ── Pipeline loop ─────────────────────────────────────────────────────────
    let mut fps_timer = Instant::now();
    let mut fps_count = 0u32;
    let frame_duration = Duration::from_secs_f64(1.0 / TARGET_FPS);

    for frame_id in 0..MAX_FRAMES {
        let t_frame = Instant::now();

        // ── 1. Capture (synthetic frame generation) ───────────────────────────
        let frame = {
            let _span = tracing::info_span!("capture", frame_id).entered();
            generate_synthetic_frame(frame_id, SLAM_WIDTH, SLAM_HEIGHT)
        };

        // ── 2. SLAM: feature extraction → matching → pose → Gaussian update ──
        let t_slam = Instant::now();
        let pose: Pose = {
            let _span = tracing::info_span!("slam_track", frame_id).entered();
            match tracker.process_frame(&frame) {
                Ok(p) => p,
                Err(e) => {
                    warn!(frame_id, %e, "SLAM tracking skipped — using last known pose");
                    tracker.current_pose()
                }
            }
        };
        let slam_ms = t_slam.elapsed().as_secs_f64() * 1_000.0;

        let gaussian_count = tracker.map().len();
        let keyframe_count = tracker.keyframe_count();

        // ── 3. Render: EWA splatting → RGB image ─────────────────────────────
        let t_render = Instant::now();
        #[allow(unused_variables)]
        let splat_rgb = {
            let _span = tracing::info_span!("render", frame_id).entered();
            renderer.render_rgb(tracker.map().gaussians.as_slice(), &pose)
        };
        let render_ms = t_render.elapsed().as_secs_f64() * 1_000.0;

        // ── 4. Display: compose 3-panel frame and push to window ──────────────
        #[cfg(feature = "viewer")]
        if let Some(ref mut v) = viewer {
            if !v.is_open() {
                info!("viewer window closed — exiting");
                break;
            }

            let cam_rgb = frame.data.as_slice();
            // Grey placeholder for the depth panel until depth estimation is
            // wired in (Phase 1 Part 4 prerequisite: depth comes from Part 3).
            let depth_gray = vec![80u8; (SLAM_WIDTH * SLAM_HEIGHT) as usize];
            let depth_rgb = gray_to_rgb(&depth_gray);
            let display_frame =
                compose_panels_bgra(&[cam_rgb, &depth_rgb, &splat_rgb], SLAM_WIDTH, SLAM_HEIGHT);

            if let Err(e) = v.update(&display_frame) {
                error!(%e, "viewer update error — stopping display");
                break;
            }
        }

        // ── 5. Per-second stats ───────────────────────────────────────────────
        fps_count += 1;
        if fps_timer.elapsed() >= Duration::from_secs(1) {
            let fps = fps_count as f64 / fps_timer.elapsed().as_secs_f64();
            info!(
                fps = format_args!("{fps:.1}"),
                gaussians = gaussian_count,
                keyframes = keyframe_count,
                slam_ms = format_args!("{slam_ms:.2}"),
                render_ms = format_args!("{render_ms:.2}"),
                "pipeline stats"
            );
            fps_count = 0;
            fps_timer = Instant::now();
        }

        // ── 6. Frame pacing ───────────────────────────────────────────────────
        let elapsed = t_frame.elapsed();
        if elapsed < frame_duration {
            std::thread::sleep(frame_duration - elapsed);
        }
    }

    info!(
        total_frames = MAX_FRAMES,
        final_gaussians = tracker.map().len(),
        final_keyframes = tracker.keyframe_count(),
        "pipeline finished"
    );
}

// ─── Synthetic frame generator ────────────────────────────────────────────────

/// Generate a deterministic synthetic RGB frame for testing.
///
/// Combines per-pixel LCG noise with a regular grid overlay so that the FAST
/// corner detector reliably finds keypoints even without a real camera.
fn generate_synthetic_frame(frame_id: u64, width: u32, height: u32) -> Frame {
    let n = (width * height * 3) as usize;
    let mut data = vec![30u8; n];

    // Per-pixel noise derived from a simple LCG seeded by `frame_id`.
    let mut rng = frame_id
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    for px in data.iter_mut() {
        rng = rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *px = ((rng >> 33) & 0xFF) as u8;
    }

    // Bright grid lines give FAST high-contrast corners to detect.
    for y in (0..height).step_by(20) {
        for x in 0..width {
            let idx = (y * width + x) as usize * 3;
            if idx + 2 < n {
                data[idx] = 220;
                data[idx + 1] = 220;
                data[idx + 2] = 220;
            }
        }
    }
    for x in (0..width).step_by(20) {
        for y in 0..height {
            let idx = (y * width + x) as usize * 3;
            if idx + 2 < n {
                data[idx] = 220;
                data[idx + 1] = 220;
                data[idx + 2] = 220;
            }
        }
    }

    Frame::new(frame_id, width, height, data).expect("synthetic frame dimensions must be valid")
}
