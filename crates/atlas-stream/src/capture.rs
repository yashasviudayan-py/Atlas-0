//! Camera capture backend.
//!
//! Uses `nokhwa` for cross-platform camera access:
//! - macOS: AVFoundation
//! - Linux: V4L2
//! - Windows: MSMF
//!
//! Raw camera buffers (YUYV / MJPEG) are decoded to packed RGB before being
//! pushed into the [`FramePipeline`](crate::pipeline::FramePipeline).
//!
//! # Example
//! ```no_run
//! use atlas_stream::{capture::CameraCapture, config::StreamConfig, pipeline::FramePipeline};
//!
//! let config = StreamConfig::default();
//! let pipeline = FramePipeline::new(config.clone());
//! let capture = CameraCapture::new(config, pipeline.sender());
//! let handle = capture.start().expect("failed to open camera");
//! // ... consume frames from pipeline.receiver() ...
//! handle.stop();
//! ```

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use atlas_core::{
    error::StreamError,
    frame::{Frame, FrameId},
};
use crossbeam::channel::Sender;
use nokhwa::{
    Camera,
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
};
use tracing::{error, info, warn};

use crate::{Result, config::StreamConfig};

/// Handle to a running camera capture thread.
///
/// Dropping the handle without calling [`stop`](CaptureHandle::stop) will still
/// signal the background thread to exit, but will not wait for it to finish.
pub struct CaptureHandle {
    running: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl CaptureHandle {
    /// Signal the capture loop to stop and wait for the background thread to exit.
    pub fn stop(mut self) {
        self.halt();
    }

    fn halt(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(t) = self.thread.take()
            && let Err(e) = t.join()
        {
            error!("capture thread panicked: {:?}", e);
        }
    }
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        // Ensure the thread is stopped even if the caller drops the handle
        // without calling stop().  thread.take() is a no-op when already
        // consumed by stop(), so this is always safe.
        self.halt();
    }
}

/// Captures live frames from a physical camera and pushes them into a
/// [`FramePipeline`](crate::pipeline::FramePipeline) sender.
///
/// The actual camera I/O runs on a dedicated background thread so that the
/// caller's thread is never blocked waiting for frames.
pub struct CameraCapture {
    config: StreamConfig,
    sender: Sender<Frame>,
}

impl CameraCapture {
    /// Create a new `CameraCapture`.
    ///
    /// # Arguments
    /// * `config` – Device index/path, target FPS, and frame resolution.
    /// * `sender` – Channel sender for the downstream frame pipeline.
    #[must_use]
    pub fn new(config: StreamConfig, sender: Sender<Frame>) -> Self {
        Self { config, sender }
    }

    /// Open the camera and start the capture loop on a dedicated background thread.
    ///
    /// The camera is opened inside the background thread (required because
    /// `nokhwa::Camera` is not `Send`), but startup errors are relayed back to
    /// the caller via a one-shot channel so they surface synchronously.
    ///
    /// # Errors
    /// - [`StreamError::FormatNegotiation`] – no format satisfies the request,
    ///   or the device cannot be enumerated.
    /// - [`StreamError::CameraOpen`] – stream could not be started or the
    ///   capture thread could not be spawned.
    pub fn start(self) -> Result<CaptureHandle> {
        // One-shot channel: thread sends `Ok(())` once the camera is open and
        // streaming, or `Err(…)` if initialization fails.
        let (init_tx, init_rx) = crossbeam::channel::bounded::<Result<()>>(1);

        let running = Arc::new(AtomicBool::new(true));
        let running_clone = Arc::clone(&running);
        let sender = self.sender.clone();
        let config = self.config.clone();

        let handle = thread::Builder::new()
            .name("atlas-capture".into())
            .spawn(move || {
                // Camera must be created on this thread — it is not Send.
                let mut camera = match open_camera(&config) {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = init_tx.send(Err(e));
                        return;
                    }
                };
                if let Err(e) = camera.open_stream() {
                    let _ = init_tx.send(Err(StreamError::CameraOpen(format!("open stream: {e}"))));
                    return;
                }

                let fmt = camera.camera_format();
                info!(
                    device = %config.device,
                    width  = fmt.width(),
                    height = fmt.height(),
                    fps    = fmt.frame_rate(),
                    "Camera stream opened"
                );

                // Signal successful initialisation before entering the loop.
                let _ = init_tx.send(Ok(()));

                run_capture_loop(&mut camera, &sender, &running_clone, &config);
                info!("Camera capture thread exiting");
            })
            .map_err(|e| StreamError::CameraOpen(format!("spawn thread: {e}")))?;

        // Block until the camera is ready (or fails to open).
        match init_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let _ = handle.join();
                return Err(e);
            }
            Err(_) => {
                // Channel disconnected before a message was sent: thread panicked.
                let _ = handle.join();
                return Err(StreamError::CameraOpen(
                    "capture thread panicked during initialisation".to_string(),
                ));
            }
        }

        Ok(CaptureHandle {
            running,
            thread: Some(handle),
        })
    }
}

/// Open the camera device described by `config`.
///
/// Requests the absolute highest frame rate that the camera can deliver in an
/// RGB-decodable format. The actual negotiated resolution may differ from the
/// configured values; the capture loop logs a warning when that happens.
fn open_camera(config: &StreamConfig) -> Result<Camera> {
    let index = parse_device_index(&config.device);
    let requested =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    Camera::new(index, requested).map_err(|e| StreamError::FormatNegotiation(format!("{e}")))
}

/// Convert the device string from config into a `nokhwa` [`CameraIndex`].
///
/// Numeric strings (e.g. `"0"`) become [`CameraIndex::Index`]; device path
/// strings (e.g. `"/dev/video0"`) become [`CameraIndex::String`].
fn parse_device_index(device: &str) -> CameraIndex {
    if let Ok(n) = device.parse::<u32>() {
        CameraIndex::Index(n)
    } else {
        CameraIndex::String(device.to_string())
    }
}

/// Main capture loop — runs on the dedicated background thread.
///
/// The loop retries transient hardware errors within a per-frame timeout
/// window, handles back-pressure by dropping frames when the pipeline buffer
/// is full, and exits cleanly when the receiver is closed or `running` is
/// cleared.
fn run_capture_loop(
    camera: &mut Camera,
    sender: &Sender<Frame>,
    running: &AtomicBool,
    config: &StreamConfig,
) {
    let mut frame_id: FrameId = 0;
    // Allow up to 3 missed frames before declaring a timeout.
    let capture_timeout = Duration::from_secs_f64(3.0 / f64::from(config.target_fps));

    while running.load(Ordering::SeqCst) {
        let deadline = Instant::now() + capture_timeout;

        // Retry transient I/O errors until the per-frame deadline.
        let raw_buffer = loop {
            match camera.frame() {
                Ok(buf) => break buf,
                Err(e) => {
                    if Instant::now() >= deadline {
                        error!(
                            frame_id,
                            timeout_ms = capture_timeout.as_millis(),
                            "Camera capture timeout: {e}"
                        );
                        running.store(false, Ordering::SeqCst);
                        return;
                    }
                    warn!(frame_id, "Transient capture error, retrying: {e}");
                    thread::sleep(Duration::from_millis(1));
                }
            }
        };

        // Decode raw camera buffer (YUYV / MJPEG) to packed RGB888.
        let rgb = match raw_buffer.decode_image::<RgbFormat>() {
            Ok(img) => img,
            Err(e) => {
                warn!(frame_id, "Frame decode failed: {e}");
                continue;
            }
        };

        let (width, height) = (rgb.width(), rgb.height());
        if width != config.frame_width || height != config.frame_height {
            warn!(
                frame_id,
                actual_w = width,
                actual_h = height,
                expected_w = config.frame_width,
                expected_h = config.frame_height,
                "Camera returned unexpected resolution — using actual dimensions"
            );
        }

        let Some(frame) = Frame::new(frame_id, width, height, rgb.into_raw()) else {
            error!(
                frame_id,
                width, height, "Frame data length mismatch — skipping"
            );
            continue;
        };

        match sender.try_send(frame) {
            Ok(()) => {}
            Err(crossbeam::channel::TrySendError::Full(_)) => {
                warn!(frame_id, "Pipeline buffer full — dropping frame");
            }
            Err(crossbeam::channel::TrySendError::Disconnected(_)) => {
                info!("Pipeline receiver closed — stopping capture");
                running.store(false, Ordering::SeqCst);
                return;
            }
        }

        frame_id = frame_id.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_device_index_numeric() {
        assert!(matches!(parse_device_index("0"), CameraIndex::Index(0)));
        assert!(matches!(parse_device_index("3"), CameraIndex::Index(3)));
    }

    #[test]
    fn parse_device_index_path() {
        match parse_device_index("/dev/video0") {
            CameraIndex::String(s) => assert_eq!(s, "/dev/video0"),
            other => panic!("expected CameraIndex::String, got {other:?}"),
        }
    }

    /// An RGB buffer with the correct byte length must produce a valid [`Frame`].
    #[test]
    fn frame_from_rgb_buffer_valid() {
        let (w, h) = (8u32, 6u32);
        let data = vec![128u8; (w * h * 3) as usize];
        let frame = Frame::new(42, w, h, data).expect("valid frame");
        assert_eq!(frame.id, 42);
        assert_eq!(frame.width, w);
        assert_eq!(frame.height, h);
        assert_eq!(frame.pixel_count(), (w * h) as usize);
    }

    /// An RGB buffer with the wrong byte length must be rejected.
    #[test]
    fn frame_rejects_bad_buffer() {
        assert!(Frame::new(0, 640, 480, vec![0u8; 10]).is_none());
    }

    /// Requires a physical camera device — skipped in CI.
    #[test]
    #[ignore = "requires a physical camera device"]
    fn camera_open_and_capture_one_frame() {
        use crate::{config::StreamConfig, pipeline::FramePipeline};

        let config = StreamConfig {
            target_fps: 30,
            frame_width: 640,
            frame_height: 480,
            buffer_size: 4,
            device: "0".to_string(),
        };
        let pipeline = FramePipeline::new(config.clone());
        let capture = CameraCapture::new(config, pipeline.sender());
        let handle = capture.start().expect("camera open failed");

        let frame = pipeline
            .receiver()
            .recv_timeout(Duration::from_secs(3))
            .expect("no frame within 3 seconds");

        assert!(frame.width > 0 && frame.height > 0);
        handle.stop();
    }
}
