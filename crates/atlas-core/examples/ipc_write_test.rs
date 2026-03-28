//! IPC write-test helper for the Python integration test suite.
//!
//! Usage:
//! ```text
//! cargo run --example ipc_write_test -- /tmp/atlas_test.mmap
//! ```
//!
//! Writes a deterministic snapshot (frame_id=99, 3 known Gaussians) to the
//! given path.  The Python integration test (`tests/integration/test_ipc.py`)
//! reads the file and verifies the data decoded correctly.

use atlas_core::shared_mem::SharedMemWriter;
use atlas_core::{Gaussian3D, GaussianCloud, Point3, Pose};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: ipc_write_test <mmap_path>");
        std::process::exit(1);
    }
    let path = Path::new(&args[1]);

    let mut writer = SharedMemWriter::create(path, 1_000).expect("create mmap");

    // Build a small cloud with 3 Gaussians at known x positions (1.0, 2.0, 3.0).
    let mut cloud = GaussianCloud::new();
    for x in [1.0_f32, 2.0, 3.0] {
        cloud.add(Gaussian3D::new(
            Point3::new(x, 0.0, 0.0),
            [1.0, 0.0, 0.0],
            0.8,
        ));
    }

    let pose = Pose {
        position: Point3::new(0.1, 0.2, 0.3),
        rotation: [1.0, 0.0, 0.0, 0.0],
    };

    // frame_id=99 is the sentinel value the Python test checks for.
    writer.write_snapshot(&cloud, &pose, 99, 12_345_678);

    println!("Wrote snapshot to {}", path.display());
}
