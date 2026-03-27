//! Real-time display window for SLAM visualisation.
//!
//! This module requires the `viewer` crate feature:
//!
//! ```toml
//! atlas-slam = { features = ["viewer"] }
//! ```
//!
//! [`Viewer`] opens a [`minifb`] window and renders composite frames produced
//! by [`crate::renderer::SplatRenderer::compose_display`].
//!
//! # Usage
//!
//! ```no_run
//! # use atlas_slam::viewer::Viewer;
//! let mut viewer = Viewer::new("Atlas SLAM", 1920, 480).expect("window open failed");
//! let frame = vec![0u32; 1920 * 480];
//! while viewer.is_open() {
//!     viewer.update(&frame).expect("draw failed");
//! }
//! ```

use minifb::{Key, Scale, Window, WindowOptions};
use tracing::warn;

use atlas_core::error::SlamError;

// ─── Viewer ───────────────────────────────────────────────────────────────────

/// A `minifb` display window for real-time SLAM output.
pub struct Viewer {
    window: Window,
    buffer: Vec<u32>,
    display_width: usize,
    display_height: usize,
}

impl Viewer {
    /// Open a new display window.
    ///
    /// `display_width` and `display_height` are the pixel dimensions of the
    /// composite frame (e.g. `3 × slam_width` for a three-panel view).
    ///
    /// # Errors
    ///
    /// Returns [`SlamError::InitFailed`] if the window cannot be created
    /// (e.g. no display server is available).
    pub fn new(title: &str, display_width: usize, display_height: usize) -> crate::Result<Self> {
        let window = Window::new(
            title,
            display_width,
            display_height,
            WindowOptions {
                scale: Scale::X1,
                ..WindowOptions::default()
            },
        )
        .map_err(|e| SlamError::InitFailed(format!("cannot open viewer window: {e}")))?;

        Ok(Self {
            window,
            buffer: vec![0u32; display_width * display_height],
            display_width,
            display_height,
        })
    }

    /// Return `true` if the window is still open and `Escape` has not been
    /// pressed.
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.window.is_open() && !self.window.is_key_down(Key::Escape)
    }

    /// Push a BGRA `u32` frame to the window.
    ///
    /// `frame` must have length `display_width × display_height`.  If the
    /// length does not match, the window is cleared to black and a warning is
    /// logged.
    ///
    /// # Errors
    ///
    /// Returns [`SlamError::InitFailed`] if the underlying window update
    /// fails.
    pub fn update(&mut self, frame: &[u32]) -> crate::Result<()> {
        if frame.len() == self.buffer.len() {
            self.buffer.copy_from_slice(frame);
        } else {
            warn!(
                expected = self.buffer.len(),
                got = frame.len(),
                "display frame size mismatch — clearing to black"
            );
            self.buffer.fill(0);
        }
        self.window
            .update_with_buffer(&self.buffer, self.display_width, self.display_height)
            .map_err(|e| SlamError::InitFailed(format!("window update failed: {e}")))?;
        Ok(())
    }

    /// Display width in pixels.
    #[must_use]
    pub fn display_width(&self) -> usize {
        self.display_width
    }

    /// Display height in pixels.
    #[must_use]
    pub fn display_height(&self) -> usize {
        self.display_height
    }
}
