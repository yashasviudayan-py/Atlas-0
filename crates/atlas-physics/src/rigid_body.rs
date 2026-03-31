//! Rigid body representation for physics simulation.
//!
//! Each [`RigidBody`] is a simplified physics proxy for a [`SemanticObject`].
//! The bounding box of the semantic object is used to derive the collision
//! shape; mass and friction come from the object's material properties.

use atlas_core::semantic::{MaterialType, SemanticObject};
use serde::{Deserialize, Serialize};

// ─── Bounding shape ────────────────────────────────────────────────────────────

/// Simplified collision shape used for broad- and narrow-phase detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundingShape {
    /// Sphere with the given radius in metres.
    Sphere {
        /// Sphere radius (metres).
        radius: f32,
    },
    /// Axis-aligned box described by its half-extents along each axis.
    Box {
        /// Half-lengths along X, Y, Z (metres).
        half_extents: [f32; 3],
    },
    /// Convex hull – stored as sample points; collision uses the bounding sphere
    /// for broad-phase and falls back to a sphere narrow-phase until a full GJK
    /// implementation is added.
    ConvexHull {
        /// Sample points defining the hull.
        points: Vec<[f32; 3]>,
        /// Bounding sphere radius (conservative, used for collision).
        bounding_radius: f32,
    },
}

impl BoundingShape {
    /// Conservative bounding sphere radius used for broad-phase culling.
    #[must_use]
    pub fn bounding_radius(&self) -> f32 {
        match self {
            BoundingShape::Sphere { radius } => *radius,
            BoundingShape::Box { half_extents } => {
                (half_extents[0].powi(2) + half_extents[1].powi(2) + half_extents[2].powi(2)).sqrt()
            }
            BoundingShape::ConvexHull {
                bounding_radius, ..
            } => *bounding_radius,
        }
    }
}

// ─── RigidBody ─────────────────────────────────────────────────────────────────

/// A rigid body for physics simulation.
///
/// Coordinates and velocities are in SI units (metres, seconds).  All vectors
/// are stored as plain `[f32; 3]` arrays so the struct is trivially serialisable
/// without additional nalgebra serde configuration.
///
/// # Example
/// ```
/// use atlas_core::semantic::{SemanticObject, MaterialProperties, MaterialType};
/// use atlas_core::spatial::Point3;
/// use atlas_physics::rigid_body::RigidBody;
///
/// let obj = SemanticObject {
///     id: 1,
///     label: "mug".into(),
///     position: Point3::new(0.0, 1.0, 0.0),
///     bbox_min: Point3::new(-0.05, 0.9, -0.05),
///     bbox_max: Point3::new(0.05, 1.1, 0.05),
///     properties: MaterialProperties {
///         mass_kg: 0.3,
///         friction: 0.4,
///         fragility: 0.8,
///         material: MaterialType::Ceramic,
///     },
///     confidence: 0.95,
///     relationships: vec![],
/// };
/// let body = RigidBody::from_semantic_object(&obj);
/// assert!((body.mass - 0.3).abs() < 1e-6);
/// assert!(body.is_at_rest(0.01));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigidBody {
    /// Object identifier (matches `SemanticObject::id`).
    pub id: u64,
    /// Position of centre of mass in world coordinates `[x, y, z]` (metres).
    pub position: [f32; 3],
    /// Linear velocity `[vx, vy, vz]` (m/s).
    pub velocity: [f32; 3],
    /// Angular velocity `[ωx, ωy, ωz]` (rad/s).
    pub angular_velocity: [f32; 3],
    /// Mass (kg).  Must be positive.
    pub mass: f32,
    /// Diagonal inertia tensor `[Ixx, Iyy, Izz]` (kg·m²).
    pub inertia: [f32; 3],
    /// Collision shape.
    pub shape: BoundingShape,
    /// Coefficient of kinetic friction (0 = frictionless, 1 = very rough).
    pub friction: f32,
    /// Coefficient of restitution (0 = perfectly inelastic, 1 = perfectly elastic).
    pub restitution: f32,
    /// When `true` the body is immovable (e.g., a floor or wall).
    pub is_static: bool,
}

impl RigidBody {
    /// Build a dynamic `RigidBody` from a [`SemanticObject`].
    ///
    /// The semantic object's bounding box determines the box shape; mass and
    /// friction come from its [`MaterialProperties`].
    ///
    /// [`MaterialProperties`]: atlas_core::semantic::MaterialProperties
    #[must_use]
    pub fn from_semantic_object(obj: &SemanticObject) -> Self {
        let hx = (obj.bbox_max.x - obj.bbox_min.x).abs() / 2.0;
        let hy = (obj.bbox_max.y - obj.bbox_min.y).abs() / 2.0;
        let hz = (obj.bbox_max.z - obj.bbox_min.z).abs() / 2.0;

        // Guard: mass must be positive to avoid division by zero.
        let mass = obj.properties.mass_kg.max(0.001);
        let shape = BoundingShape::Box {
            half_extents: [hx, hy, hz],
        };
        let inertia = Self::box_inertia(mass, hx, hy, hz);
        let restitution = material_restitution(&obj.properties.material);

        Self {
            id: obj.id,
            position: [obj.position.x, obj.position.y, obj.position.z],
            velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            mass,
            inertia,
            shape,
            friction: obj.properties.friction,
            restitution,
            is_static: false,
        }
    }

    /// Create a static (immovable) box body, e.g. to represent a surface.
    #[must_use]
    pub fn static_box(id: u64, position: [f32; 3], half_extents: [f32; 3]) -> Self {
        Self {
            id,
            position,
            velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            mass: f32::INFINITY,
            inertia: [f32::INFINITY; 3],
            shape: BoundingShape::Box { half_extents },
            friction: 0.5,
            restitution: 0.3,
            is_static: true,
        }
    }

    /// Return `true` when both linear and angular speed are below `threshold`.
    ///
    /// Used by the simulator to detect when a body has come to rest.
    #[must_use]
    pub fn is_at_rest(&self, threshold: f32) -> bool {
        let v2 = self.velocity[0].powi(2) + self.velocity[1].powi(2) + self.velocity[2].powi(2);
        let w2 = self.angular_velocity[0].powi(2)
            + self.angular_velocity[1].powi(2)
            + self.angular_velocity[2].powi(2);
        v2 < threshold * threshold && w2 < threshold * threshold
    }

    // ── Inertia helpers ───────────────────────────────────────────────────────

    /// Diagonal inertia tensor for a uniform solid box with full side lengths
    /// `2hx × 2hy × 2hz`.
    fn box_inertia(mass: f32, hx: f32, hy: f32, hz: f32) -> [f32; 3] {
        let lx = 2.0 * hx;
        let ly = 2.0 * hy;
        let lz = 2.0 * hz;
        [
            mass / 12.0 * (ly * ly + lz * lz),
            mass / 12.0 * (lx * lx + lz * lz),
            mass / 12.0 * (lx * lx + ly * ly),
        ]
    }
}

/// Heuristic coefficient of restitution per material type.
fn material_restitution(material: &MaterialType) -> f32 {
    match material {
        MaterialType::Metal => 0.40,
        MaterialType::Glass => 0.30,
        MaterialType::Ceramic => 0.20,
        MaterialType::Plastic => 0.50,
        MaterialType::Wood => 0.30,
        MaterialType::Fabric => 0.10,
        MaterialType::Liquid => 0.00,
        MaterialType::Organic => 0.20,
        MaterialType::Unknown => 0.30,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::semantic::{MaterialProperties, SemanticObject};
    use atlas_core::spatial::Point3;

    fn test_object(mass: f32, material: MaterialType) -> SemanticObject {
        SemanticObject {
            id: 42,
            label: "test".into(),
            position: Point3::new(1.0, 2.0, 3.0),
            bbox_min: Point3::new(0.9, 1.9, 2.9),
            bbox_max: Point3::new(1.1, 2.1, 3.1),
            properties: MaterialProperties {
                mass_kg: mass,
                friction: 0.4,
                fragility: 0.5,
                material,
            },
            confidence: 0.9,
            relationships: vec![],
        }
    }

    #[test]
    fn test_from_semantic_object_preserves_position() {
        let obj = test_object(1.0, MaterialType::Wood);
        let body = RigidBody::from_semantic_object(&obj);
        assert!((body.position[0] - 1.0).abs() < 1e-6);
        assert!((body.position[1] - 2.0).abs() < 1e-6);
        assert!((body.position[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_semantic_object_preserves_mass() {
        let obj = test_object(2.5, MaterialType::Metal);
        let body = RigidBody::from_semantic_object(&obj);
        assert!((body.mass - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_mass_clamp_prevents_zero() {
        let obj = test_object(0.0, MaterialType::Fabric);
        let body = RigidBody::from_semantic_object(&obj);
        assert!(body.mass > 0.0, "mass must be positive even for 0 kg input");
    }

    #[test]
    fn test_starts_at_rest() {
        let obj = test_object(1.0, MaterialType::Wood);
        let body = RigidBody::from_semantic_object(&obj);
        assert!(body.is_at_rest(0.01));
    }

    #[test]
    fn test_moving_body_not_at_rest() {
        let obj = test_object(1.0, MaterialType::Plastic);
        let mut body = RigidBody::from_semantic_object(&obj);
        body.velocity = [0.0, -1.0, 0.0];
        assert!(!body.is_at_rest(0.01));
    }

    #[test]
    fn test_box_inertia_sphere_moment() {
        // For a cube side L: Ixx = Iyy = Izz = m L² / 6
        let mass = 1.0f32;
        let h = 0.5f32; // half-extent → L = 1.0
        let inertia = RigidBody::box_inertia(mass, h, h, h);
        let expected = mass * 1.0f32 * 1.0 / 6.0;
        assert!(
            (inertia[0] - expected).abs() < 1e-5,
            "Ixx = {}, expected {}",
            inertia[0],
            expected
        );
    }

    #[test]
    fn test_bounding_radius_sphere() {
        let shape = BoundingShape::Sphere { radius: 2.0 };
        assert!((shape.bounding_radius() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_bounding_radius_box() {
        // Half-extents [3, 4, 0] → radius = 5
        let shape = BoundingShape::Box {
            half_extents: [3.0, 4.0, 0.0],
        };
        assert!((shape.bounding_radius() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_static_box_is_immovable() {
        let body = RigidBody::static_box(99, [0.0, -0.1, 0.0], [10.0, 0.1, 10.0]);
        assert!(body.is_static);
        assert!(body.mass.is_infinite());
    }
}
