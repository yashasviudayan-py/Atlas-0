//! Spatial primitives for 3D operations.

use nalgebra::{Isometry3, Matrix4, Point3 as NaPoint3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

/// A 3D point in world coordinates.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Point3 {
    #[must_use]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Euclidean distance to another point.
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    #[must_use]
    pub fn to_nalgebra(&self) -> NaPoint3<f32> {
        NaPoint3::new(self.x, self.y, self.z)
    }
}

/// Camera or object pose in 3D space (position + orientation).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Pose {
    /// Translation component (x, y, z).
    pub position: Point3,
    /// Rotation as quaternion (w, x, y, z).
    pub rotation: [f32; 4],
}

impl Pose {
    #[must_use]
    pub fn identity() -> Self {
        Self {
            position: Point3::new(0.0, 0.0, 0.0),
            rotation: [1.0, 0.0, 0.0, 0.0],
        }
    }

    /// Convert to a nalgebra 4x4 transformation matrix.
    #[must_use]
    pub fn to_matrix(&self) -> Matrix4<f32> {
        let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            self.rotation[0],
            self.rotation[1],
            self.rotation[2],
            self.rotation[3],
        ));
        let t = Vector3::new(self.position.x, self.position.y, self.position.z);
        Isometry3::from_parts(t.into(), q).to_homogeneous()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_distance() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(3.0, 4.0, 0.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_identity_pose() {
        let pose = Pose::identity();
        let mat = pose.to_matrix();
        assert!((mat[(0, 0)] - 1.0).abs() < 1e-6);
        assert!((mat[(3, 3)] - 1.0).abs() < 1e-6);
    }
}
