//! Live camera capture demo.
//!
//! Opens the camera device configured in `configs/default.toml` (or the
//! compiled-in defaults), starts the frame pipeline, and prints a rolling
//! FPS counter every second until the user presses Ctrl-C.
//!
//! # Usage
//! ```text
//! cargo run --example capture_demo -p atlas-stream
//! ```
//!
//! Override the camera device or FPS without editing the config file:
//! ```text
//! ATLAS_STREAM_DEVICE=/dev/video1 ATLAS_STREAM_TARGET_FPS=30 \
//!     cargo run --example capture_demo -p atlas-stream
//! ```

use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use atlas_stream::{capture::CameraCapture, config::StreamConfig, pipeline::FramePipeline};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

fn main() {
    // Initialise structured logging.  Set RUST_LOG=debug for verbose output.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

    // Load config — fall back to defaults when the file is absent.
    let config = StreamConfig::load(Path::new("configs/default.toml")).unwrap_or_else(|e| {
        error!("Config load failed: {e} — using defaults");
        StreamConfig::default()
    });

    info!(
        device       = %config.device,
        target_fps   = config.target_fps,
        resolution   = %format!("{}x{}", config.frame_width, config.frame_height),
        buffer_size  = config.buffer_size,
        "Starting capture_demo"
    );

    // Set up the frame pipeline.
    let pipeline = FramePipeline::new(config.clone());
    let receiver = pipeline.receiver();

    // Start camera capture on a background thread.
    let capture = CameraCapture::new(config.clone(), pipeline.sender());
    let handle = match capture.start() {
        Ok(h) => h,
        Err(e) => {
            error!("Failed to open camera: {e}");
            std::process::exit(1);
        }
    };

    // Install a Ctrl-C handler so we exit cleanly.
    let running = Arc::new(AtomicBool::new(true));
    let running_ctrlc = Arc::clone(&running);
    ctrlc::set_handler(move || {
        running_ctrlc.store(false, Ordering::SeqCst);
    })
    .expect("failed to install Ctrl-C handler");

    info!("Camera open — press Ctrl-C to stop");

    // FPS counter: count frames received in each 1-second window.
    let mut frames_in_window: u64 = 0;
    let mut window_start = Instant::now();

    while running.load(Ordering::SeqCst) {
        match receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(frame) => {
                frames_in_window += 1;

                let elapsed = window_start.elapsed();
                if elapsed >= Duration::from_secs(1) {
                    let fps = frames_in_window as f64 / elapsed.as_secs_f64();
                    info!(
                        fps = format!("{fps:.1}"),
                        frame_id = frame.id,
                        width = frame.width,
                        height = frame.height,
                        "FPS counter"
                    );
                    frames_in_window = 0;
                    window_start = Instant::now();
                }
            }
            Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                // No frame within 100 ms — check running flag and loop.
            }
            Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                info!("Pipeline disconnected — stopping");
                break;
            }
        }
    }

    info!("Shutting down capture_demo");
    handle.stop();
}
