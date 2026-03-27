//! Memory-mapped snapshot buffer for Rust→Python data sharing.
//!
//! This is the Phase 2 IPC scaffold: the Rust side writes [`GaussianCloud`]
//! snapshots to a memory-mapped file at a configurable rate; the Python side
//! reads them with zero-copy `numpy` access.
//!
//! # Binary layout
//!
//! ```text
//! ┌──────────────────────── 64-byte header ────────────────────────────────┐
//! │  magic(4)  version(4)  frame_id(8)  timestamp_ns(8)                    │
//! │  gaussian_count(4)  pose_tx(4)  pose_ty(4)  pose_tz(4)                 │
//! │  pose_qw(4)  pose_qx(4)  pose_qy(4)  pose_qz(4)                       │
//! │  write_index(4)  _padding(4)                                            │
//! ├─────────────────────── Buffer 0 ───────────────────────────────────────┤
//! │  gaussian_count × 28 bytes  (x,y,z + opacity + r,g,b)                  │
//! ├─────────────────────── Buffer 1 ───────────────────────────────────────┤
//! │  gaussian_count × 28 bytes                                              │
//! └────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Double-buffering avoids read-write contention: the writer fills the
//! *inactive* buffer, then atomically flips `write_index`.  Readers always
//! consume the buffer pointed to by `write_index`.
//!
//! All multi-byte integers are stored in **little-endian** byte order.

use std::fs::OpenOptions;
use std::path::Path;
use std::sync::atomic::Ordering;

use memmap2::{Mmap, MmapMut, MmapOptions};

use crate::{GaussianCloud, Point3, Pose};

// ─── Header byte offsets ─────────────────────────────────────────────────────

const OFF_MAGIC: usize = 0;
const OFF_VERSION: usize = 4;
const OFF_FRAME_ID: usize = 8;
const OFF_TIMESTAMP_NS: usize = 16;
const OFF_GAUSSIAN_COUNT: usize = 24;
const OFF_POSE_TX: usize = 28;
const OFF_POSE_TY: usize = 32;
const OFF_POSE_TZ: usize = 36;
const OFF_POSE_QW: usize = 40;
const OFF_POSE_QX: usize = 44;
const OFF_POSE_QY: usize = 48;
const OFF_POSE_QZ: usize = 52;
const OFF_WRITE_INDEX: usize = 56;

/// Total header size in bytes.
pub const HEADER_SIZE: usize = 64;

/// Magic number that identifies Atlas-0 shared-memory files.
pub const ATLAS_MMAP_MAGIC: u32 = 0xA7_1A_50_00;
/// Current binary format version.
pub const ATLAS_MMAP_VERSION: u32 = 1;
/// Bytes occupied by one serialised Gaussian entry.
pub const BYTES_PER_GAUSSIAN: usize = 28;

// ─── Per-Gaussian field offsets (relative to entry start) ────────────────────

const GF_CENTER_X: usize = 0;
const GF_CENTER_Y: usize = 4;
const GF_CENTER_Z: usize = 8;
const GF_OPACITY: usize = 12;
const GF_COLOR_R: usize = 16;
const GF_COLOR_G: usize = 20;
const GF_COLOR_B: usize = 24;

// ─── SharedMemWriter ──────────────────────────────────────────────────────────

/// Writes [`GaussianCloud`] snapshots to a memory-mapped file for consumption
/// by the Python world-model agent.
///
/// # Example
///
/// ```no_run
/// # use atlas_core::{GaussianCloud, Pose};
/// # use atlas_core::shared_mem::SharedMemWriter;
/// # use std::path::Path;
/// let mut writer = SharedMemWriter::create(Path::new("/tmp/atlas.mmap"), 100_000)?;
/// let cloud = GaussianCloud::new();
/// let pose  = Pose::identity();
/// writer.write_snapshot(&cloud, &pose, 1, 0);
/// # Ok::<(), atlas_core::error::AtlasError>(())
/// ```
pub struct SharedMemWriter {
    mmap: MmapMut,
    max_gaussians: usize,
}

impl SharedMemWriter {
    /// Create (or truncate) `path` and memory-map it for writing.
    ///
    /// `max_gaussians` sets the per-buffer capacity.  Gaussians beyond this
    /// limit are silently truncated during [`write_snapshot`].
    ///
    /// # Errors
    ///
    /// Returns [`AtlasError::Io`] on file-system or mmap failures.
    ///
    /// [`write_snapshot`]: SharedMemWriter::write_snapshot
    /// [`AtlasError::Io`]: crate::error::AtlasError::Io
    pub fn create(path: &Path, max_gaussians: usize) -> crate::Result<Self> {
        let file_size = HEADER_SIZE + max_gaussians * BYTES_PER_GAUSSIAN * 2;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(file_size as u64)?;

        // SAFETY: The file was just created by this process and is exclusively
        // owned by this `SharedMemWriter` for its lifetime.  The mapping covers
        // the full file, so all byte-range accesses within `file_size` are
        // in-bounds.
        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        let mut writer = Self {
            mmap,
            max_gaussians,
        };
        writer.init_header();
        Ok(writer)
    }

    /// Write a snapshot of `cloud` taken at `frame_id` / `timestamp_ns` from
    /// camera `pose` into the mmap buffer.
    ///
    /// Uses double-buffering: writes into the *inactive* buffer first, then
    /// flips `write_index` with a release fence so readers always see a fully
    /// written snapshot.
    pub fn write_snapshot(
        &mut self,
        cloud: &GaussianCloud,
        pose: &Pose,
        frame_id: u64,
        timestamp_ns: u64,
    ) {
        // Select the inactive buffer (the one readers are *not* currently using).
        let current_idx = read_u32(&self.mmap, OFF_WRITE_INDEX) as usize;
        let next_idx = 1 - current_idx;

        let n = cloud.len().min(self.max_gaussians);
        let buf_start = HEADER_SIZE + next_idx * self.max_gaussians * BYTES_PER_GAUSSIAN;

        // Write Gaussian data to the inactive buffer.
        const C0: f32 = 0.282_094_8; // SH degree-0 → RGB conversion constant.
        for (i, g) in cloud.gaussians.iter().take(n).enumerate() {
            let entry = buf_start + i * BYTES_PER_GAUSSIAN;

            let (r, green, b) = if g.sh_coefficients.len() >= 3 {
                (
                    (C0 * g.sh_coefficients[0] + 0.5).clamp(0.0, 1.0),
                    (C0 * g.sh_coefficients[1] + 0.5).clamp(0.0, 1.0),
                    (C0 * g.sh_coefficients[2] + 0.5).clamp(0.0, 1.0),
                )
            } else {
                (0.5, 0.5, 0.5)
            };

            write_f32(&mut self.mmap, entry + GF_CENTER_X, g.center.x);
            write_f32(&mut self.mmap, entry + GF_CENTER_Y, g.center.y);
            write_f32(&mut self.mmap, entry + GF_CENTER_Z, g.center.z);
            write_f32(&mut self.mmap, entry + GF_OPACITY, g.opacity);
            write_f32(&mut self.mmap, entry + GF_COLOR_R, r);
            write_f32(&mut self.mmap, entry + GF_COLOR_G, green);
            write_f32(&mut self.mmap, entry + GF_COLOR_B, b);
        }

        // Update header fields.
        write_u64(&mut self.mmap, OFF_FRAME_ID, frame_id);
        write_u64(&mut self.mmap, OFF_TIMESTAMP_NS, timestamp_ns);
        write_u32(&mut self.mmap, OFF_GAUSSIAN_COUNT, n as u32);
        write_f32(&mut self.mmap, OFF_POSE_TX, pose.position.x);
        write_f32(&mut self.mmap, OFF_POSE_TY, pose.position.y);
        write_f32(&mut self.mmap, OFF_POSE_TZ, pose.position.z);
        write_f32(&mut self.mmap, OFF_POSE_QW, pose.rotation[0]);
        write_f32(&mut self.mmap, OFF_POSE_QX, pose.rotation[1]);
        write_f32(&mut self.mmap, OFF_POSE_QY, pose.rotation[2]);
        write_f32(&mut self.mmap, OFF_POSE_QZ, pose.rotation[3]);

        // Release fence ensures all prior writes are visible before the index
        // flip.
        std::sync::atomic::fence(Ordering::Release);
        write_u32(&mut self.mmap, OFF_WRITE_INDEX, next_idx as u32);
    }

    fn init_header(&mut self) {
        write_u32(&mut self.mmap, OFF_MAGIC, ATLAS_MMAP_MAGIC);
        write_u32(&mut self.mmap, OFF_VERSION, ATLAS_MMAP_VERSION);
        write_u32(&mut self.mmap, OFF_WRITE_INDEX, 0);
    }
}

// ─── SharedMemReader ──────────────────────────────────────────────────────────

/// A single Gaussian entry decoded from a snapshot buffer.
#[derive(Debug, Clone, Copy)]
pub struct GaussianEntry {
    /// World-space centre X.
    pub x: f32,
    /// World-space centre Y.
    pub y: f32,
    /// World-space centre Z.
    pub z: f32,
    /// Opacity in \[0, 1\].
    pub opacity: f32,
    /// Red channel in \[0, 1\].
    pub r: f32,
    /// Green channel in \[0, 1\].
    pub g: f32,
    /// Blue channel in \[0, 1\].
    pub b: f32,
}

/// A snapshot decoded from the active double-buffer.
#[derive(Debug, Clone)]
pub struct GaussianSnapshot {
    /// Frame index from the SLAM pipeline when the snapshot was written.
    pub frame_id: u64,
    /// UNIX timestamp in nanoseconds when the snapshot was written.
    pub timestamp_ns: u64,
    /// Camera pose at snapshot time.
    pub pose: Pose,
    /// Decoded Gaussian entries (position + opacity + colour).
    pub gaussians: Vec<GaussianEntry>,
}

/// Reads [`GaussianSnapshot`]s from a memory-mapped file written by
/// [`SharedMemWriter`].
///
/// # Example
///
/// ```no_run
/// # use atlas_core::shared_mem::SharedMemReader;
/// # use std::path::Path;
/// let reader = SharedMemReader::open(Path::new("/tmp/atlas.mmap"), 100_000)?;
/// let snap = reader.read_snapshot();
/// println!("frame {} has {} Gaussians", snap.frame_id, snap.gaussians.len());
/// # Ok::<(), atlas_core::error::AtlasError>(())
/// ```
pub struct SharedMemReader {
    mmap: Mmap,
    max_gaussians: usize,
}

impl SharedMemReader {
    /// Open an existing mmap file for reading.
    ///
    /// # Errors
    ///
    /// Returns [`AtlasError::Io`] on file-system or mmap failures, or
    /// [`AtlasError::Serialization`] if the magic number or version do not
    /// match.
    ///
    /// [`AtlasError::Io`]: crate::error::AtlasError::Io
    /// [`AtlasError::Serialization`]: crate::error::AtlasError::Serialization
    pub fn open(path: &Path, max_gaussians: usize) -> crate::Result<Self> {
        let file = std::fs::File::open(path)?;
        // SAFETY: We never write through this mapping; multiple concurrent
        // readers are safe.
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        let magic = read_u32(&mmap, OFF_MAGIC);
        if magic != ATLAS_MMAP_MAGIC {
            return Err(crate::error::AtlasError::Serialization(format!(
                "invalid Atlas mmap magic: {magic:#010x}"
            )));
        }
        let version = read_u32(&mmap, OFF_VERSION);
        if version != ATLAS_MMAP_VERSION {
            return Err(crate::error::AtlasError::Serialization(format!(
                "unsupported Atlas mmap version: {version}"
            )));
        }

        Ok(Self {
            mmap,
            max_gaussians,
        })
    }

    /// Read the latest fully-written snapshot from the active double-buffer.
    ///
    /// Uses an acquire fence to ensure all writes by the producer are visible
    /// before the data is read.
    #[must_use]
    pub fn read_snapshot(&self) -> GaussianSnapshot {
        std::sync::atomic::fence(Ordering::Acquire);

        let read_idx = read_u32(&self.mmap, OFF_WRITE_INDEX) as usize;
        let n = (read_u32(&self.mmap, OFF_GAUSSIAN_COUNT) as usize).min(self.max_gaussians);
        let buf_start = HEADER_SIZE + read_idx * self.max_gaussians * BYTES_PER_GAUSSIAN;

        let mut gaussians = Vec::with_capacity(n);
        for i in 0..n {
            let entry = buf_start + i * BYTES_PER_GAUSSIAN;
            gaussians.push(GaussianEntry {
                x: read_f32(&self.mmap, entry + GF_CENTER_X),
                y: read_f32(&self.mmap, entry + GF_CENTER_Y),
                z: read_f32(&self.mmap, entry + GF_CENTER_Z),
                opacity: read_f32(&self.mmap, entry + GF_OPACITY),
                r: read_f32(&self.mmap, entry + GF_COLOR_R),
                g: read_f32(&self.mmap, entry + GF_COLOR_G),
                b: read_f32(&self.mmap, entry + GF_COLOR_B),
            });
        }

        GaussianSnapshot {
            frame_id: read_u64(&self.mmap, OFF_FRAME_ID),
            timestamp_ns: read_u64(&self.mmap, OFF_TIMESTAMP_NS),
            pose: Pose {
                position: Point3::new(
                    read_f32(&self.mmap, OFF_POSE_TX),
                    read_f32(&self.mmap, OFF_POSE_TY),
                    read_f32(&self.mmap, OFF_POSE_TZ),
                ),
                rotation: [
                    read_f32(&self.mmap, OFF_POSE_QW),
                    read_f32(&self.mmap, OFF_POSE_QX),
                    read_f32(&self.mmap, OFF_POSE_QY),
                    read_f32(&self.mmap, OFF_POSE_QZ),
                ],
            },
            gaussians,
        }
    }
}

// ─── Byte-level I/O ───────────────────────────────────────────────────────────

fn write_u32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn write_u64(buf: &mut [u8], off: usize, v: u64) {
    buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

fn write_f32(buf: &mut [u8], off: usize, v: f32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn read_u32(buf: &[u8], off: usize) -> u32 {
    let b: [u8; 4] = buf[off..off + 4].try_into().unwrap_or([0; 4]);
    u32::from_le_bytes(b)
}

fn read_u64(buf: &[u8], off: usize) -> u64 {
    let b: [u8; 8] = buf[off..off + 8].try_into().unwrap_or([0; 8]);
    u64::from_le_bytes(b)
}

fn read_f32(buf: &[u8], off: usize) -> f32 {
    let b: [u8; 4] = buf[off..off + 4].try_into().unwrap_or([0; 4]);
    f32::from_le_bytes(b)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Gaussian3D, GaussianCloud, Point3, Pose};

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("atlas_shm_{}_{}.mmap", std::process::id(), tag))
    }

    #[test]
    fn test_write_and_read_empty_snapshot() {
        let path = temp_path("empty");
        let mut writer = SharedMemWriter::create(&path, 1_000).unwrap();
        let cloud = GaussianCloud::new();
        let pose = Pose::identity();
        writer.write_snapshot(&cloud, &pose, 42, 100_000);

        let reader = SharedMemReader::open(&path, 1_000).unwrap();
        let snap = reader.read_snapshot();
        assert_eq!(snap.frame_id, 42);
        assert_eq!(snap.timestamp_ns, 100_000);
        assert_eq!(snap.gaussians.len(), 0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_write_and_read_with_gaussians() {
        let path = temp_path("gaussians");
        let mut writer = SharedMemWriter::create(&path, 1_000).unwrap();

        let mut cloud = GaussianCloud::new();
        const C0: f32 = 0.282_094_8;
        let mut g = Gaussian3D::new(Point3::new(1.0, 2.0, 3.0), [1.0, 0.0, 0.0], 0.8);
        g.sh_coefficients = vec![(1.0 - 0.5) / C0, (0.0 - 0.5) / C0, (0.0 - 0.5) / C0];
        cloud.add(g);

        let pose = Pose {
            position: Point3::new(0.1, 0.2, 0.3),
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        writer.write_snapshot(&cloud, &pose, 7, 999);

        let reader = SharedMemReader::open(&path, 1_000).unwrap();
        let snap = reader.read_snapshot();

        assert_eq!(snap.frame_id, 7);
        assert_eq!(snap.gaussians.len(), 1);
        assert!((snap.gaussians[0].x - 1.0).abs() < 1e-5);
        assert!((snap.gaussians[0].y - 2.0).abs() < 1e-5);
        assert!((snap.gaussians[0].z - 3.0).abs() < 1e-5);
        assert!((snap.gaussians[0].opacity - 0.8).abs() < 1e-5);
        assert!(
            (snap.gaussians[0].r - 1.0).abs() < 1e-3,
            "expected r≈1.0, got {}",
            snap.gaussians[0].r
        );
        assert!((snap.pose.position.x - 0.1).abs() < 1e-5);
        assert!((snap.pose.rotation[0] - 1.0).abs() < 1e-5);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_double_buffer_flips_correctly() {
        let path = temp_path("flip");
        let mut writer = SharedMemWriter::create(&path, 10).unwrap();
        let cloud = GaussianCloud::new();
        let pose = Pose::identity();
        let reader = SharedMemReader::open(&path, 10).unwrap();

        writer.write_snapshot(&cloud, &pose, 1, 0);
        assert_eq!(reader.read_snapshot().frame_id, 1);

        writer.write_snapshot(&cloud, &pose, 2, 0);
        assert_eq!(reader.read_snapshot().frame_id, 2);

        writer.write_snapshot(&cloud, &pose, 3, 0);
        assert_eq!(reader.read_snapshot().frame_id, 3);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_invalid_magic_returns_error() {
        let path = temp_path("magic");
        std::fs::write(&path, vec![0u8; 256]).unwrap();
        let result = SharedMemReader::open(&path, 10);
        assert!(result.is_err(), "wrong magic must return an error");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_pose_round_trip() {
        let path = temp_path("pose");
        let mut writer = SharedMemWriter::create(&path, 10).unwrap();
        let cloud = GaussianCloud::new();
        let pose = Pose {
            position: Point3::new(-1.5, 2.25, 0.0),
            rotation: [0.707, 0.0, 0.707, 0.0],
        };
        writer.write_snapshot(&cloud, &pose, 0, 0);

        let reader = SharedMemReader::open(&path, 10).unwrap();
        let snap = reader.read_snapshot();
        assert!((snap.pose.position.x - (-1.5)).abs() < 1e-5);
        assert!((snap.pose.position.y - 2.25).abs() < 1e-5);
        assert!((snap.pose.rotation[0] - 0.707).abs() < 1e-5);
        assert!((snap.pose.rotation[2] - 0.707).abs() < 1e-5);

        let _ = std::fs::remove_file(&path);
    }
}
