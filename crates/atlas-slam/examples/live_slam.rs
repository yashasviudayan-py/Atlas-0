//! End-to-end SLAM pipeline — dataset or synthetic frames.
//!
//! Reads an image-sequence directory and runs it through the full Atlas-0
//! Phase 1 pipeline, writing Gaussian-map snapshots to shared memory so the
//! Python world-model agent can pick them up.
//!
//! # Usage
//!
//! ```bash
//! # Real dataset (TUM RGB-D, EuRoC, or any numbered PNGs):
//! ATLAS_SEQUENCE_DIR=/path/to/dataset/rgb \
//!   cargo run --example live_slam -p atlas-slam --release
//!
//! # Synthetic frames (no dataset needed):
//! cargo run --example live_slam -p atlas-slam
//! ```
//!
//! # Environment variables
//!
//! | Variable              | Default                  | Description                        |
//! |-----------------------|--------------------------|------------------------------------|
//! | `ATLAS_SEQUENCE_DIR`  | *(unset)*                | Directory of numbered image files. |
//! | `ATLAS_MMAP_PATH`     | `/tmp/atlas.mmap`        | Shared-memory output path.         |
//! | `ATLAS_MAX_GAUSSIANS` | `100000`                 | mmap buffer capacity.              |

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use atlas_core::shared_mem::SharedMemWriter;
use atlas_core::{Frame, Pose};
use atlas_slam::{
    config::{CameraIntrinsics, SlamConfig},
    renderer::SplatRenderer,
    tracker::Tracker,
};
use atlas_stream::{config::StreamConfig, file_source::FileSource, pipeline::FramePipeline};
use tracing::{info, warn};

// ─── Constants ────────────────────────────────────────────────────────────────

const SLAM_WIDTH: u32 = 640;
const SLAM_HEIGHT: u32 = 480;
const TARGET_FPS: f64 = 30.0;
/// Write a new mmap snapshot every N frames.
const SNAPSHOT_INTERVAL: u64 = 5;

// ─── Entry point ─────────────────────────────────────────────────────────────

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let sequence_dir = std::env::var("ATLAS_SEQUENCE_DIR").ok().map(PathBuf::from);
    let mmap_path =
        std::env::var("ATLAS_MMAP_PATH").unwrap_or_else(|_| "/tmp/atlas.mmap".to_string());
    let max_gaussians: usize = std::env::var("ATLAS_MAX_GAUSSIANS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000);

    info!(
        sequence = ?sequence_dir,
        mmap = %mmap_path,
        "Atlas-0 SLAM starting"
    );

    // ── SLAM config ───────────────────────────────────────────────────────────
    // TUM fr1 camera intrinsics (used when a dataset is provided; synthetic
    // frames use a simplified model).
    let (config, intrinsics) = if sequence_dir.is_some() {
        let cam = CameraIntrinsics {
            fx: 517.3,
            fy: 516.5,
            cx: 318.6,
            cy: 255.3,
        };
        let cfg = SlamConfig {
            min_features: 50,
            max_gaussians: max_gaussians.min(500_000),
            keyframe_translation_threshold: 0.05,
            keyframe_rotation_threshold: 0.05,
            optimization_iterations: 5,
            enable_loop_closure: true,
            camera: cam.clone(),
            ..SlamConfig::default()
        };
        (cfg, cam)
    } else {
        let cam = CameraIntrinsics {
            fx: 200.0,
            fy: 200.0,
            cx: f64::from(SLAM_WIDTH) / 2.0,
            cy: f64::from(SLAM_HEIGHT) / 2.0,
        };
        let cfg = SlamConfig {
            min_features: 5,
            max_gaussians: 10_000,
            keyframe_translation_threshold: 0.05,
            keyframe_rotation_threshold: 0.05,
            optimization_iterations: 3,
            enable_loop_closure: false,
            camera: cam.clone(),
            ..SlamConfig::default()
        };
        (cfg, cam)
    };

    let mut tracker = Tracker::new(config);
    let renderer = SplatRenderer::new(intrinsics, SLAM_WIDTH, SLAM_HEIGHT);

    // ── Shared-memory writer ─────────────────────────────────────────────────
    let mut mmap_writer = match SharedMemWriter::create(
        std::path::Path::new(&mmap_path),
        max_gaussians,
    ) {
        Ok(w) => {
            info!(%mmap_path, "Shared memory file created");
            Some(w)
        }
        Err(e) => {
            warn!(%e, "Could not create shared memory file — Python agent will not receive data");
            None
        }
    };

    // ── Frame source ─────────────────────────────────────────────────────────
    // `_source_handle` must be declared here so it lives for the full pipeline
    // loop. If it were declared inside the `if` block it would be dropped
    // before the loop runs, stopping the FileSource immediately.
    let (_source_handle, frame_rx): (Option<_>, Box<dyn Iterator<Item = Frame>>) =
        if let Some(dir) = sequence_dir {
            info!(dir = %dir.display(), "Using FileSource (dataset mode)");
            let stream_cfg = StreamConfig {
                target_fps: TARGET_FPS as u32,
                frame_width: SLAM_WIDTH,
                frame_height: SLAM_HEIGHT,
                buffer_size: 8,
                device: "0".to_string(),
            };
            let pipeline = FramePipeline::new(stream_cfg.clone());
            let rx = pipeline.receiver();
            let handle = FileSource::new(dir, stream_cfg, pipeline.sender())
                .start()
                .expect("FileSource failed to start");

            // Bridge crossbeam receiver → mpsc so we can box it as Iterator.
            let (tx, channel_rx) = mpsc::channel::<Frame>();
            std::thread::spawn(move || {
                while let Ok(frame) = rx.recv() {
                    if tx.send(frame).is_err() {
                        break;
                    }
                }
            });
            (Some(handle), Box::new(channel_rx.into_iter()))
        } else {
            info!("No ATLAS_SEQUENCE_DIR set — using synthetic frames (200 frames)");
            let frames: Vec<Frame> = (0u64..200)
                .map(|id| generate_synthetic_frame(id, SLAM_WIDTH, SLAM_HEIGHT))
                .collect();
            (None, Box::new(frames.into_iter()))
        };

    // ── Pipeline loop ─────────────────────────────────────────────────────────
    let mut fps_timer = Instant::now();
    let mut fps_count = 0u32;
    let frame_duration = Duration::from_secs_f64(1.0 / TARGET_FPS);

    for frame in frame_rx {
        let t_frame = Instant::now();
        let frame_id = frame.id;

        // ── 1. SLAM ───────────────────────────────────────────────────────────
        let t_slam = Instant::now();
        let pose: Pose = match tracker.process_frame(&frame) {
            Ok(p) => p,
            Err(e) => {
                warn!(frame_id, %e, "SLAM tracking skipped — using last known pose");
                tracker.current_pose()
            }
        };
        let slam_ms = t_slam.elapsed().as_secs_f64() * 1_000.0;

        // ── 2. Render ─────────────────────────────────────────────────────────
        let t_render = Instant::now();
        let _splat_rgb = renderer.render_rgb(tracker.map().gaussians.as_slice(), &pose);
        let render_ms = t_render.elapsed().as_secs_f64() * 1_000.0;

        // ── 3. Write snapshot to shared memory ────────────────────────────────
        if frame_id % SNAPSHOT_INTERVAL == 0
            && let Some(ref mut writer) = mmap_writer
        {
            let ts_ns = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            writer.write_snapshot(tracker.map(), &pose, frame_id, ts_ns);
        }

        // ── 4. Per-second stats ───────────────────────────────────────────────
        fps_count += 1;
        if fps_timer.elapsed() >= Duration::from_secs(1) {
            let fps = fps_count as f64 / fps_timer.elapsed().as_secs_f64();
            info!(
                fps = format_args!("{fps:.1}"),
                gaussians = tracker.map().len(),
                keyframes = tracker.keyframe_count(),
                slam_ms = format_args!("{slam_ms:.2}"),
                render_ms = format_args!("{render_ms:.2}"),
                "pipeline stats"
            );
            fps_count = 0;
            fps_timer = Instant::now();
        }

        // ── 5. Frame pacing ───────────────────────────────────────────────────
        let elapsed = t_frame.elapsed();
        if elapsed < frame_duration {
            std::thread::sleep(frame_duration - elapsed);
        }
    }

    // Final snapshot with whatever was accumulated.
    if let Some(ref mut writer) = mmap_writer {
        let ts_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        writer.write_snapshot(tracker.map(), &tracker.current_pose(), u64::MAX, ts_ns);
    }

    info!(
        final_gaussians = tracker.map().len(),
        final_keyframes = tracker.keyframe_count(),
        "pipeline finished"
    );
}

// ─── Synthetic frame generator ────────────────────────────────────────────────

/// Generate a deterministic synthetic RGB frame for testing.
fn generate_synthetic_frame(frame_id: u64, width: u32, height: u32) -> Frame {
    let n = (width * height * 3) as usize;
    let mut data = vec![30u8; n];

    let mut rng = frame_id
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    for px in data.iter_mut() {
        rng = rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *px = ((rng >> 33) & 0xFF) as u8;
    }

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
