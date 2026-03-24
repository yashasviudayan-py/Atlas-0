//! Stream pipeline configuration.
//!
//! [`StreamConfig`] can be constructed three ways:
//!
//! 1. **`StreamConfig::default()`** — hard-coded sensible defaults.
//! 2. **`StreamConfig::from_file(path)`** — load the `[stream]` table from a
//!    TOML file (typically `configs/default.toml`).
//! 3. **`StreamConfig::load(path)`** — same as `from_file`, then apply
//!    `ATLAS_STREAM_*` environment variable overrides on top.
//!
//! # Environment variable overrides
//!
//! Every field has a corresponding `ATLAS_STREAM_` env var:
//!
//! | Env var | Field | Type |
//! |---|---|---|
//! | `ATLAS_STREAM_TARGET_FPS` | `target_fps` | `u32` |
//! | `ATLAS_STREAM_FRAME_WIDTH` | `frame_width` | `u32` |
//! | `ATLAS_STREAM_FRAME_HEIGHT` | `frame_height` | `u32` |
//! | `ATLAS_STREAM_BUFFER_SIZE` | `buffer_size` | `usize` |
//! | `ATLAS_STREAM_DEVICE` | `device` | `String` |
//!
//! # Example
//! ```no_run
//! use std::path::Path;
//! use atlas_stream::config::StreamConfig;
//!
//! // Load from file with env var overrides applied.
//! let cfg = StreamConfig::load(Path::new("configs/default.toml"))
//!     .unwrap_or_default();
//! assert!(cfg.target_fps > 0);
//! ```

use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::Result;
use atlas_core::error::StreamError;

/// Top-level TOML layout — only the `[stream]` section is used here.
#[derive(Debug, Deserialize)]
struct AtlasTomlConfig {
    stream: StreamConfig,
}

/// Runtime configuration for the video ingestion pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Target frames per second for capture and playback pacing.
    pub target_fps: u32,
    /// Native capture frame width in pixels.
    pub frame_width: u32,
    /// Native capture frame height in pixels.
    pub frame_height: u32,
    /// Bounded channel capacity between capture and downstream stages.
    pub buffer_size: usize,
    /// Camera device index (e.g. `"0"`) or device path (e.g. `"/dev/video0"`).
    pub device: String,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            target_fps: 60,
            frame_width: 1280,
            frame_height: 720,
            buffer_size: 4,
            device: "0".to_string(),
        }
    }
}

impl StreamConfig {
    /// Load `StreamConfig` from the `[stream]` section of a TOML file.
    ///
    /// # Errors
    /// Returns [`StreamError::CameraOpen`] (re-used as a config error) when
    /// the file cannot be read or the TOML is malformed.
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| StreamError::CameraOpen(format!("read config {}: {e}", path.display())))?;
        let top: AtlasTomlConfig = toml::from_str(&raw).map_err(|e| {
            StreamError::CameraOpen(format!("parse config {}: {e}", path.display()))
        })?;
        Ok(top.stream)
    }

    /// Load from a TOML file, then apply `ATLAS_STREAM_*` env var overrides.
    ///
    /// Missing or unreadable config files are silently replaced with
    /// [`StreamConfig::default()`]; a warning is logged.
    ///
    /// # Errors
    /// Only returns an error when a present env var contains a value that
    /// cannot be parsed into the expected type — in that case the override is
    /// skipped and a warning is logged (never a hard error).
    pub fn load(path: &Path) -> Result<Self> {
        let mut cfg = Self::from_file(path).unwrap_or_else(|e| {
            warn!(
                "Could not load config from {}: {e} — using defaults",
                path.display()
            );
            Self::default()
        });
        cfg.apply_env_overrides();
        Ok(cfg)
    }

    /// Apply `ATLAS_STREAM_*` environment variable overrides in-place.
    ///
    /// Fields are overridden only when the corresponding env var is set *and*
    /// parses successfully. Bad values are warned and ignored.
    fn apply_env_overrides(&mut self) {
        if let Some(v) = env_var("ATLAS_STREAM_TARGET_FPS") {
            match v.parse::<u32>() {
                Ok(n) => self.target_fps = n,
                Err(_) => warn!("ATLAS_STREAM_TARGET_FPS={v:?} is not a valid u32 — ignored"),
            }
        }
        if let Some(v) = env_var("ATLAS_STREAM_FRAME_WIDTH") {
            match v.parse::<u32>() {
                Ok(n) => self.frame_width = n,
                Err(_) => warn!("ATLAS_STREAM_FRAME_WIDTH={v:?} is not a valid u32 — ignored"),
            }
        }
        if let Some(v) = env_var("ATLAS_STREAM_FRAME_HEIGHT") {
            match v.parse::<u32>() {
                Ok(n) => self.frame_height = n,
                Err(_) => warn!("ATLAS_STREAM_FRAME_HEIGHT={v:?} is not a valid u32 — ignored"),
            }
        }
        if let Some(v) = env_var("ATLAS_STREAM_BUFFER_SIZE") {
            match v.parse::<usize>() {
                Ok(n) => self.buffer_size = n,
                Err(_) => warn!("ATLAS_STREAM_BUFFER_SIZE={v:?} is not a valid usize — ignored"),
            }
        }
        if let Some(v) = env_var("ATLAS_STREAM_DEVICE") {
            self.device = v;
        }
    }
}

/// Read an env var, returning `None` when unset or not valid UTF-8.
fn env_var(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Mutex;

    /// Serialize tests that mutate environment variables so they don't race.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn write_toml(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(content.as_bytes()).expect("write");
        f
    }

    #[test]
    fn default_has_sensible_values() {
        let cfg = StreamConfig::default();
        assert!(cfg.target_fps > 0);
        assert!(cfg.frame_width > 0);
        assert!(cfg.frame_height > 0);
        assert!(cfg.buffer_size > 0);
    }

    #[test]
    fn from_file_parses_stream_section() {
        let f = write_toml(
            r#"
            [stream]
            target_fps   = 30
            frame_width  = 640
            frame_height = 480
            buffer_size  = 8
            device       = "/dev/video1"
            "#,
        );
        let cfg = StreamConfig::from_file(f.path()).expect("parse");
        assert_eq!(cfg.target_fps, 30);
        assert_eq!(cfg.frame_width, 640);
        assert_eq!(cfg.frame_height, 480);
        assert_eq!(cfg.buffer_size, 8);
        assert_eq!(cfg.device, "/dev/video1");
    }

    #[test]
    fn from_file_missing_path_returns_error() {
        let result = StreamConfig::from_file(Path::new("/no/such/file.toml"));
        assert!(result.is_err());
    }

    #[test]
    fn from_file_malformed_toml_returns_error() {
        let f = write_toml("NOT VALID TOML ::::");
        assert!(StreamConfig::from_file(f.path()).is_err());
    }

    #[test]
    fn load_falls_back_to_default_on_missing_file() {
        let _guard = ENV_LOCK.lock().unwrap();
        let cfg = StreamConfig::load(Path::new("/no/such/file.toml")).expect("load");
        // Should not error — uses default instead.
        assert_eq!(cfg.target_fps, StreamConfig::default().target_fps);
    }

    #[test]
    fn env_var_overrides_target_fps() {
        let _guard = ENV_LOCK.lock().unwrap();
        let f = write_toml(
            r#"
            [stream]
            target_fps   = 60
            frame_width  = 1280
            frame_height = 720
            buffer_size  = 4
            device       = "0"
            "#,
        );
        // SAFETY: ENV_LOCK serialises all env-mutating tests; no concurrent access.
        unsafe { std::env::set_var("ATLAS_STREAM_TARGET_FPS", "24") };
        let cfg = StreamConfig::load(f.path()).expect("load");
        unsafe { std::env::remove_var("ATLAS_STREAM_TARGET_FPS") };
        assert_eq!(cfg.target_fps, 24);
    }

    #[test]
    fn env_var_overrides_device() {
        let _guard = ENV_LOCK.lock().unwrap();
        let f = write_toml(
            r#"
            [stream]
            target_fps   = 60
            frame_width  = 1280
            frame_height = 720
            buffer_size  = 4
            device       = "0"
            "#,
        );
        // SAFETY: ENV_LOCK serialises all env-mutating tests; no concurrent access.
        unsafe { std::env::set_var("ATLAS_STREAM_DEVICE", "/dev/video2") };
        let cfg = StreamConfig::load(f.path()).expect("load");
        unsafe { std::env::remove_var("ATLAS_STREAM_DEVICE") };
        assert_eq!(cfg.device, "/dev/video2");
    }

    #[test]
    fn from_file_valid_toml_missing_stream_section_returns_error() {
        // Arrange — valid TOML but no [stream] table.
        let f = write_toml(
            r#"
            [other_section]
            foo = 1
            "#,
        );
        // Act + Assert
        assert!(
            StreamConfig::from_file(f.path()).is_err(),
            "missing [stream] section should return an error"
        );
    }

    #[test]
    fn env_var_overrides_frame_width() {
        let _guard = ENV_LOCK.lock().unwrap();
        let f = write_toml(
            r#"
            [stream]
            target_fps   = 60
            frame_width  = 1280
            frame_height = 720
            buffer_size  = 4
            device       = "0"
            "#,
        );
        // SAFETY: ENV_LOCK serialises all env-mutating tests.
        unsafe { std::env::set_var("ATLAS_STREAM_FRAME_WIDTH", "320") };
        let cfg = StreamConfig::load(f.path()).expect("load");
        unsafe { std::env::remove_var("ATLAS_STREAM_FRAME_WIDTH") };
        assert_eq!(cfg.frame_width, 320);
    }

    #[test]
    fn env_var_overrides_frame_height() {
        let _guard = ENV_LOCK.lock().unwrap();
        let f = write_toml(
            r#"
            [stream]
            target_fps   = 60
            frame_width  = 1280
            frame_height = 720
            buffer_size  = 4
            device       = "0"
            "#,
        );
        // SAFETY: ENV_LOCK serialises all env-mutating tests.
        unsafe { std::env::set_var("ATLAS_STREAM_FRAME_HEIGHT", "240") };
        let cfg = StreamConfig::load(f.path()).expect("load");
        unsafe { std::env::remove_var("ATLAS_STREAM_FRAME_HEIGHT") };
        assert_eq!(cfg.frame_height, 240);
    }

    #[test]
    fn env_var_overrides_buffer_size() {
        let _guard = ENV_LOCK.lock().unwrap();
        let f = write_toml(
            r#"
            [stream]
            target_fps   = 60
            frame_width  = 1280
            frame_height = 720
            buffer_size  = 4
            device       = "0"
            "#,
        );
        // SAFETY: ENV_LOCK serialises all env-mutating tests.
        unsafe { std::env::set_var("ATLAS_STREAM_BUFFER_SIZE", "16") };
        let cfg = StreamConfig::load(f.path()).expect("load");
        unsafe { std::env::remove_var("ATLAS_STREAM_BUFFER_SIZE") };
        assert_eq!(cfg.buffer_size, 16);
    }

    #[test]
    fn invalid_frame_width_env_var_is_ignored() {
        let _guard = ENV_LOCK.lock().unwrap();
        let f = write_toml(
            r#"
            [stream]
            target_fps   = 60
            frame_width  = 1280
            frame_height = 720
            buffer_size  = 4
            device       = "0"
            "#,
        );
        // SAFETY: ENV_LOCK serialises all env-mutating tests.
        unsafe { std::env::set_var("ATLAS_STREAM_FRAME_WIDTH", "not_a_number") };
        let cfg = StreamConfig::load(f.path()).expect("load");
        unsafe { std::env::remove_var("ATLAS_STREAM_FRAME_WIDTH") };
        // Bad value is silently ignored — original TOML value is kept.
        assert_eq!(cfg.frame_width, 1280);
    }

    #[test]
    fn invalid_env_var_is_ignored() {
        let _guard = ENV_LOCK.lock().unwrap();
        let f = write_toml(
            r#"
            [stream]
            target_fps   = 60
            frame_width  = 1280
            frame_height = 720
            buffer_size  = 4
            device       = "0"
            "#,
        );
        // SAFETY: ENV_LOCK serialises all env-mutating tests; no concurrent access.
        unsafe { std::env::set_var("ATLAS_STREAM_TARGET_FPS", "not_a_number") };
        let cfg = StreamConfig::load(f.path()).expect("load");
        unsafe { std::env::remove_var("ATLAS_STREAM_TARGET_FPS") };
        // Bad value is ignored — original TOML value is kept.
        assert_eq!(cfg.target_fps, 60);
    }
}
