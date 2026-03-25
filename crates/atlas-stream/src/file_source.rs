//! Image-sequence file source for offline testing and benchmarking.
//!
//! Reads numbered image files (PNG, JPEG, BMP, TIFF) from a directory,
//! sorted lexicographically by filename, and emits [`Frame`]s into a
//! [`FramePipeline`](crate::pipeline::FramePipeline) at the configured FPS.
//!
//! This is the primary offline input mechanism for development and
//! reproducible testing. Standard SLAM benchmark datasets (EuRoC MAV, TUM
//! RGB-D, KITTI) are distributed as image sequences and work directly.
//!
//! For live MP4/MKV playback, install FFmpeg and add `ffmpeg-next` as a
//! feature — that path is not enabled by default.
//!
//! # Example
//! ```no_run
//! use std::path::PathBuf;
//! use atlas_stream::{config::StreamConfig, file_source::FileSource, pipeline::FramePipeline};
//!
//! let config  = StreamConfig::default();
//! let pipeline = FramePipeline::new(config.clone());
//! let source  = FileSource::new(PathBuf::from("data/sequence"), config, pipeline.sender());
//! let handle  = source.start().expect("failed to open sequence");
//! // Consume frames from pipeline.receiver() …
//! handle.stop();
//! ```

use std::{
    path::{Path, PathBuf},
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
use image::RgbImage;
use tracing::{error, info, warn};

use crate::{Result, config::StreamConfig};

/// Supported image file extensions (lowercase comparison).
const SUPPORTED_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "bmp", "tiff", "tif"];

/// Handle to a running file-source playback thread.
///
/// Dropping the handle without calling [`stop`](FileSourceHandle::stop) will
/// still signal the background thread to exit; it will not wait for it to
/// finish.
pub struct FileSourceHandle {
    running: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl FileSourceHandle {
    /// Signal the playback loop to stop and wait for the thread to exit.
    pub fn stop(mut self) {
        self.halt();
    }

    fn halt(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(t) = self.thread.take()
            && let Err(e) = t.join()
        {
            error!("file-source thread panicked: {:?}", e);
        }
    }
}

impl Drop for FileSourceHandle {
    fn drop(&mut self) {
        self.halt();
    }
}

/// Reads an image sequence from a directory and emits frames at a fixed rate.
pub struct FileSource {
    dir: PathBuf,
    config: StreamConfig,
    sender: Sender<Frame>,
}

impl FileSource {
    /// Create a new `FileSource`.
    ///
    /// # Arguments
    /// * `dir`    – Directory containing numbered image files.
    /// * `config` – Target FPS and pipeline buffer size.
    /// * `sender` – Channel sender for the downstream frame pipeline.
    #[must_use]
    pub fn new(dir: PathBuf, config: StreamConfig, sender: Sender<Frame>) -> Self {
        Self {
            dir,
            config,
            sender,
        }
    }

    /// Discover image files and start the playback thread.
    ///
    /// Files are discovered synchronously before the thread is spawned so that
    /// any directory-access error surfaces immediately in the caller.
    ///
    /// # Errors
    /// - [`StreamError::CameraOpen`] – directory not found, no images found,
    ///   or thread could not be spawned.
    pub fn start(self) -> Result<FileSourceHandle> {
        if self.config.target_fps == 0 {
            return Err(StreamError::CameraOpen(
                "target_fps must be greater than zero".to_string(),
            ));
        }

        let files = collect_image_files(&self.dir)?;
        info!(
            dir = %self.dir.display(),
            count = files.len(),
            "FileSource: discovered image sequence"
        );

        let running = Arc::new(AtomicBool::new(true));
        let running_clone = Arc::clone(&running);
        let sender = self.sender.clone();
        let config = self.config.clone();

        let handle = thread::Builder::new()
            .name("atlas-file-source".into())
            .spawn(move || {
                run_playback_loop(files, &sender, &running_clone, &config);
                info!("FileSource playback thread exiting");
            })
            .map_err(|e| StreamError::CameraOpen(format!("spawn file-source thread: {e}")))?;

        Ok(FileSourceHandle {
            running,
            thread: Some(handle),
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Collect and sort all supported image files in `dir`.
fn collect_image_files(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        return Err(StreamError::CameraOpen(format!(
            "image sequence directory not found: {}",
            dir.display()
        )));
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| StreamError::CameraOpen(format!("read dir {}: {e}", dir.display())))?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let ext = path.extension()?.to_str()?.to_lowercase();
            if SUPPORTED_EXTENSIONS.contains(&ext.as_str()) {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if files.is_empty() {
        return Err(StreamError::CameraOpen(format!(
            "no supported image files found in {}",
            dir.display()
        )));
    }

    files.sort();
    Ok(files)
}

/// Load one image file as an RGB [`Frame`].
fn load_frame(path: &Path, frame_id: FrameId) -> Option<Frame> {
    let img = match image::open(path) {
        Ok(i) => i,
        Err(e) => {
            warn!(path = %path.display(), "Failed to load image: {e}");
            return None;
        }
    };
    let rgb: RgbImage = img.into_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Frame::new(frame_id, w, h, rgb.into_raw())
}

/// Main playback loop — runs on the dedicated background thread.
///
/// Each iteration:
/// 1. Loads the next image file as a [`Frame`].
/// 2. Sends it into the pipeline (drops frame on back-pressure).
/// 3. Sleeps until the next scheduled emission time to simulate real-time FPS.
fn run_playback_loop(
    files: Vec<PathBuf>,
    sender: &Sender<Frame>,
    running: &AtomicBool,
    config: &StreamConfig,
) {
    let frame_interval = Duration::from_secs_f64(1.0 / f64::from(config.target_fps));
    let mut frame_id: FrameId = 0;

    for path in &files {
        if !running.load(Ordering::SeqCst) {
            break;
        }

        let deadline = Instant::now() + frame_interval;

        let Some(frame) = load_frame(path, frame_id) else {
            // Do not advance frame_id — keep IDs sequential for downstream.
            continue;
        };

        match sender.try_send(frame) {
            Ok(()) => {}
            Err(crossbeam::channel::TrySendError::Full(_)) => {
                warn!(frame_id, "Pipeline buffer full — dropping frame");
            }
            Err(crossbeam::channel::TrySendError::Disconnected(_)) => {
                info!("Pipeline receiver closed — stopping file source");
                running.store(false, Ordering::SeqCst);
                return;
            }
        }

        frame_id = frame_id.wrapping_add(1);

        // Pace emission to simulate real-time playback.
        let now = Instant::now();
        if now < deadline {
            thread::sleep(deadline - now);
        }
    }

    if running.load(Ordering::SeqCst) {
        info!("FileSource: sequence complete ({frame_id} frames emitted)");
    }
    running.store(false, Ordering::SeqCst);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::StreamConfig, pipeline::FramePipeline};
    use std::time::Duration;

    fn write_test_image(dir: &Path, name: &str, width: u32, height: u32, value: u8) {
        let img: RgbImage =
            image::ImageBuffer::from_pixel(width, height, image::Rgb([value, value, value]));
        img.save(dir.join(name)).expect("save test image");
    }

    #[test]
    fn collect_sorts_files_lexicographically() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_test_image(dir.path(), "frame_003.png", 4, 4, 30);
        write_test_image(dir.path(), "frame_001.png", 4, 4, 10);
        write_test_image(dir.path(), "frame_002.png", 4, 4, 20);

        let files = collect_image_files(dir.path()).expect("collect");
        assert_eq!(files.len(), 3);
        assert!(
            files[0]
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .contains("001")
        );
        assert!(
            files[1]
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .contains("002")
        );
        assert!(
            files[2]
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .contains("003")
        );
    }

    #[test]
    fn missing_directory_returns_error() {
        let result = collect_image_files(Path::new("/nonexistent/sequence/path"));
        assert!(result.is_err());
    }

    #[test]
    fn empty_directory_returns_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let result = collect_image_files(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn load_frame_decodes_png() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_test_image(dir.path(), "test.png", 8, 6, 200);
        let path = dir.path().join("test.png");
        let frame = load_frame(&path, 7).expect("load frame");
        assert_eq!(frame.id, 7);
        assert_eq!(frame.width, 8);
        assert_eq!(frame.height, 6);
        assert_eq!(frame.data.len(), 8 * 6 * 3);
    }

    #[test]
    fn file_source_emits_frames_and_stops() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_test_image(dir.path(), "f001.png", 4, 4, 100);
        write_test_image(dir.path(), "f002.png", 4, 4, 150);

        let config = StreamConfig {
            target_fps: 120, // fast for test
            buffer_size: 8,
            ..StreamConfig::default()
        };
        let pipeline = FramePipeline::new(config.clone());
        let source = FileSource::new(dir.path().to_path_buf(), config, pipeline.sender());
        let _handle = source.start().expect("start");

        let frame0 = pipeline
            .receiver()
            .recv_timeout(Duration::from_secs(5))
            .expect("first frame");
        let frame1 = pipeline
            .receiver()
            .recv_timeout(Duration::from_secs(5))
            .expect("second frame");

        assert_eq!(frame0.id, 0);
        assert_eq!(frame1.id, 1);
        assert_eq!(frame0.width, 4);
        assert_eq!(frame0.height, 4);
    }

    #[test]
    fn load_frame_returns_none_for_corrupted_file() {
        // Arrange — file has a .png extension but contains garbage bytes.
        let dir = tempfile::tempdir().expect("tempdir");
        let bad = dir.path().join("corrupt.png");
        std::fs::write(&bad, b"this is not a valid PNG file!!").unwrap();
        // Act
        let result = load_frame(&bad, 0);
        // Assert — returns None without panicking.
        assert!(result.is_none());
    }

    #[test]
    fn load_frame_decodes_jpeg() {
        // Arrange — save a JPEG image.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("frame.jpg");
        let img: RgbImage = image::ImageBuffer::from_pixel(8, 8, image::Rgb([100u8, 100, 100]));
        img.save(&path).expect("save jpeg");
        // Act
        let frame = load_frame(&path, 3).expect("load jpeg frame");
        // Assert
        assert_eq!(frame.id, 3);
        assert_eq!(frame.width, 8);
        assert_eq!(frame.height, 8);
    }

    #[test]
    fn all_supported_extensions_are_collected() {
        // Arrange — one file per supported extension.
        let dir = tempfile::tempdir().expect("tempdir");
        let extensions = ["png", "jpg", "jpeg", "bmp", "tiff", "tif"];
        for ext in &extensions {
            write_test_image(dir.path(), &format!("frame.{ext}"), 2, 2, 0);
        }
        // Act
        let files = collect_image_files(dir.path()).expect("collect");
        // Assert — every extension was picked up.
        assert_eq!(
            files.len(),
            extensions.len(),
            "expected {} files, got {}",
            extensions.len(),
            files.len()
        );
    }

    #[test]
    fn corrupted_image_in_sequence_is_skipped_not_aborted() {
        // Arrange — mix one corrupt file between two valid images.
        let dir = tempfile::tempdir().expect("tempdir");
        write_test_image(dir.path(), "f001.png", 4, 4, 10);
        std::fs::write(dir.path().join("f002.png"), b"garbage").unwrap();
        write_test_image(dir.path(), "f003.png", 4, 4, 30);

        let config = StreamConfig {
            target_fps: 240,
            buffer_size: 8,
            ..StreamConfig::default()
        };
        let pipeline = FramePipeline::new(config.clone());
        let source = FileSource::new(dir.path().to_path_buf(), config, pipeline.sender());
        let _handle = source.start().expect("start");

        // Act — collect whatever frames arrive within 5 s.
        let mut received_ids = Vec::new();
        while let Ok(frame) = pipeline
            .receiver()
            .recv_timeout(std::time::Duration::from_secs(5))
        {
            received_ids.push(frame.id);
            if received_ids.len() == 2 {
                break;
            }
        }

        // Assert — exactly 2 valid frames emitted (the corrupt one was skipped).
        assert_eq!(received_ids.len(), 2);
        // IDs must be consecutive: corrupt frames do not consume an ID.
        assert!(received_ids.contains(&0));
        assert!(received_ids.contains(&1));
    }

    #[test]
    fn dropped_receiver_stops_source_cleanly() {
        // Arrange — large sequence so the loop doesn't finish on its own.
        let dir = tempfile::tempdir().expect("tempdir");
        for i in 0..50u32 {
            write_test_image(dir.path(), &format!("f{i:03}.png"), 4, 4, 0);
        }
        let config = StreamConfig {
            target_fps: 240,
            buffer_size: 2,
            ..StreamConfig::default()
        };
        let pipeline = FramePipeline::new(config.clone());
        let receiver = pipeline.receiver();
        let source = FileSource::new(dir.path().to_path_buf(), config, pipeline.sender());
        let handle = source.start().expect("start");

        // Act — receive one frame then drop the receiver, causing disconnect.
        let _frame = receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("first frame");
        drop(receiver);

        // Assert — stop() returns in finite time (source detected disconnect).
        handle.stop();
    }

    #[test]
    fn zero_target_fps_returns_error_not_panic() {
        // Arrange
        let dir = tempfile::tempdir().expect("tempdir");
        write_test_image(dir.path(), "f001.png", 4, 4, 0);
        let config = StreamConfig {
            target_fps: 0,
            ..StreamConfig::default()
        };
        let pipeline = FramePipeline::new(StreamConfig::default());
        let source = FileSource::new(dir.path().to_path_buf(), config, pipeline.sender());
        // Act
        let result = source.start();
        // Assert — clean error, no panic.
        let Err(e) = result else {
            panic!("expected error, got Ok")
        };
        assert!(
            format!("{e}").contains("target_fps"),
            "error should mention target_fps, got: {e}"
        );
    }

    #[test]
    fn corrupted_image_in_sequence_has_sequential_ids() {
        // Arrange — corrupt middle file.
        let dir = tempfile::tempdir().expect("tempdir");
        write_test_image(dir.path(), "f001.png", 4, 4, 10);
        std::fs::write(dir.path().join("f002.png"), b"garbage").unwrap();
        write_test_image(dir.path(), "f003.png", 4, 4, 30);

        let config = StreamConfig {
            target_fps: 240,
            buffer_size: 8,
            ..StreamConfig::default()
        };
        let pipeline = FramePipeline::new(config.clone());
        let source = FileSource::new(dir.path().to_path_buf(), config, pipeline.sender());
        let _handle = source.start().expect("start");

        let mut ids = Vec::new();
        while let Ok(frame) = pipeline
            .receiver()
            .recv_timeout(std::time::Duration::from_secs(5))
        {
            ids.push(frame.id);
            if ids.len() == 2 {
                break;
            }
        }

        // Assert — IDs are sequential (0, 1), not (0, 2).
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn non_image_files_are_ignored() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_test_image(dir.path(), "frame.png", 2, 2, 0);
        std::fs::write(dir.path().join("readme.txt"), b"not an image").unwrap();
        std::fs::write(dir.path().join("data.csv"), b"1,2,3").unwrap();

        let files = collect_image_files(dir.path()).expect("collect");
        assert_eq!(files.len(), 1);
    }
}
