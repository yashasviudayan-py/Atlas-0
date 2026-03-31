//! Collision detection between rigid bodies and surfaces.
//!
//! Provides broad-phase bounding-sphere overlap tests and narrow-phase
//! shape-pair tests (sphere–sphere, AABB–AABB, sphere–AABB).  Plane tests
//! against static [`Surface`] objects are handled separately via
//! [`CollisionDetector::body_vs_surface`].

use crate::rigid_body::{BoundingShape, RigidBody};
use crate::surfaces::Surface;

// ─── Contact ───────────────────────────────────────────────────────────────────

/// A contact between a dynamic body and another body or a surface.
///
/// `normal` points **from the other object toward body A** — this is the
/// direction body A should be pushed to resolve the penetration.
#[derive(Debug, Clone)]
pub struct Contact {
    /// World-space contact point.
    pub point: [f32; 3],
    /// Unit contact normal (from B toward A).
    pub normal: [f32; 3],
    /// Penetration depth (metres, ≥ 0).
    pub depth: f32,
}

// ─── CollisionDetector ────────────────────────────────────────────────────────

/// Stateless collision detection utilities.
///
/// All methods are pure functions — no mutable state is required.
pub struct CollisionDetector;

impl CollisionDetector {
    // ── Body vs static surface ────────────────────────────────────────────────

    /// Test `body` against a static planar `surface`.
    ///
    /// Returns `Some(Contact)` when the body penetrates or touches the surface,
    /// `None` when it is fully separated.
    #[must_use]
    pub fn body_vs_surface(body: &RigidBody, surface: &Surface) -> Option<Contact> {
        match &body.shape {
            BoundingShape::Sphere { radius } => sphere_vs_plane(&body.position, *radius, surface),
            BoundingShape::Box { half_extents } => {
                aabb_vs_plane(&body.position, half_extents, surface)
            }
            BoundingShape::ConvexHull {
                bounding_radius, ..
            } => {
                // Conservative: treat as a bounding sphere until full GJK is added.
                sphere_vs_plane(&body.position, *bounding_radius, surface)
            }
        }
    }

    // ── Body vs body ──────────────────────────────────────────────────────────

    /// Test two dynamic bodies against each other.
    ///
    /// Returns `Some(Contact)` when the bodies intersect, `None` when separated.
    /// The `normal` in the returned contact points from `b` toward `a`.
    #[must_use]
    pub fn body_vs_body(a: &RigidBody, b: &RigidBody) -> Option<Contact> {
        // Broad-phase: bounding-sphere overlap.
        let ra = a.shape.bounding_radius();
        let rb = b.shape.bounding_radius();
        let dx = a.position[0] - b.position[0];
        let dy = a.position[1] - b.position[1];
        let dz = a.position[2] - b.position[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let sum_r = ra + rb;
        if dist_sq > sum_r * sum_r {
            return None; // separated — skip narrow phase
        }

        // Narrow phase by shape pair.
        match (&a.shape, &b.shape) {
            (BoundingShape::Sphere { radius: ra }, BoundingShape::Sphere { radius: rb }) => {
                sphere_vs_sphere(&a.position, *ra, &b.position, *rb)
            }
            (BoundingShape::Box { half_extents: ha }, BoundingShape::Box { half_extents: hb }) => {
                aabb_vs_aabb(&a.position, ha, &b.position, hb)
            }
            (BoundingShape::Sphere { radius }, BoundingShape::Box { half_extents }) => {
                sphere_vs_aabb(&a.position, *radius, &b.position, half_extents)
            }
            (BoundingShape::Box { half_extents }, BoundingShape::Sphere { radius }) => {
                // Reverse roles then flip normal so it still points from B → A.
                sphere_vs_aabb(&b.position, *radius, &a.position, half_extents).map(|mut c| {
                    c.normal = neg3(c.normal);
                    c
                })
            }
            // Fallback for ConvexHull pairings — use bounding spheres.
            _ => {
                let ra = a.shape.bounding_radius();
                let rb = b.shape.bounding_radius();
                sphere_vs_sphere(&a.position, ra, &b.position, rb)
            }
        }
    }
}

// ─── Primitive tests ──────────────────────────────────────────────────────────

/// Sphere (centre, radius) vs infinite plane.  Normal points from plane outward.
fn sphere_vs_plane(center: &[f32; 3], radius: f32, surface: &Surface) -> Option<Contact> {
    let dist = surface.signed_distance(center);
    let depth = radius - dist;
    if depth >= 0.0 {
        let n = surface.normal;
        // Closest point on the sphere surface to the plane.
        let point = [
            center[0] - n[0] * (dist - radius).max(0.0),
            center[1] - n[1] * (dist - radius).max(0.0),
            center[2] - n[2] * (dist - radius).max(0.0),
        ];
        Some(Contact {
            point,
            normal: n,
            depth,
        })
    } else {
        None
    }
}

/// AABB (centre, half-extents) vs infinite plane.
///
/// Uses the support-function projection: the "radius" of the box along the
/// plane normal is `Σ |hᵢ · nᵢ|`.
fn aabb_vs_plane(center: &[f32; 3], half_extents: &[f32; 3], surface: &Surface) -> Option<Contact> {
    let n = &surface.normal;
    let support_radius =
        half_extents[0] * n[0].abs() + half_extents[1] * n[1].abs() + half_extents[2] * n[2].abs();
    let dist = surface.signed_distance(center);
    let depth = support_radius - dist;
    if depth >= 0.0 {
        // Contact point: deepest corner of the AABB projected onto the plane.
        let point = [
            center[0] - n[0] * dist,
            center[1] - n[1] * dist,
            center[2] - n[2] * dist,
        ];
        Some(Contact {
            point,
            normal: *n,
            depth,
        })
    } else {
        None
    }
}

/// Sphere A vs sphere B.  Normal points from B → A.
fn sphere_vs_sphere(pa: &[f32; 3], ra: f32, pb: &[f32; 3], rb: f32) -> Option<Contact> {
    let dx = pa[0] - pb[0];
    let dy = pa[1] - pb[1];
    let dz = pa[2] - pb[2];
    let dist_sq = dx * dx + dy * dy + dz * dz;
    let sum_r = ra + rb;
    if dist_sq >= sum_r * sum_r {
        return None;
    }
    let dist = dist_sq.sqrt().max(1e-8);
    let normal = [dx / dist, dy / dist, dz / dist];
    let depth = sum_r - dist;
    // Contact point: surface of sphere B closest to A.
    let point = [
        pb[0] + normal[0] * rb,
        pb[1] + normal[1] * rb,
        pb[2] + normal[2] * rb,
    ];
    Some(Contact {
        point,
        normal,
        depth,
    })
}

/// AABB A vs AABB B (SAT on each axis).  Normal is the min-overlap axis.
/// Normal direction points from B toward A.
fn aabb_vs_aabb(pa: &[f32; 3], ha: &[f32; 3], pb: &[f32; 3], hb: &[f32; 3]) -> Option<Contact> {
    let dx = pa[0] - pb[0];
    let dy = pa[1] - pb[1];
    let dz = pa[2] - pb[2];
    let ox = ha[0] + hb[0] - dx.abs();
    let oy = ha[1] + hb[1] - dy.abs();
    let oz = ha[2] + hb[2] - dz.abs();

    if ox <= 0.0 || oy <= 0.0 || oz <= 0.0 {
        return None;
    }

    // Minimum-penetration axis → contact normal.
    let (depth, normal) = if ox <= oy && ox <= oz {
        (ox, [dx.signum(), 0.0, 0.0])
    } else if oy <= oz {
        (oy, [0.0, dy.signum(), 0.0])
    } else {
        (oz, [0.0, 0.0, dz.signum()])
    };

    let point = [
        (pa[0] + pb[0]) / 2.0,
        (pa[1] + pb[1]) / 2.0,
        (pa[2] + pb[2]) / 2.0,
    ];
    Some(Contact {
        point,
        normal,
        depth,
    })
}

/// Sphere (pos, radius) vs AABB (center, half_extents).
/// Normal points from AABB toward sphere.
fn sphere_vs_aabb(
    sphere_pos: &[f32; 3],
    radius: f32,
    box_pos: &[f32; 3],
    half_extents: &[f32; 3],
) -> Option<Contact> {
    // Closest point on the AABB to the sphere centre.
    let clamped = [
        sphere_pos[0].clamp(box_pos[0] - half_extents[0], box_pos[0] + half_extents[0]),
        sphere_pos[1].clamp(box_pos[1] - half_extents[1], box_pos[1] + half_extents[1]),
        sphere_pos[2].clamp(box_pos[2] - half_extents[2], box_pos[2] + half_extents[2]),
    ];

    let dx = sphere_pos[0] - clamped[0];
    let dy = sphere_pos[1] - clamped[1];
    let dz = sphere_pos[2] - clamped[2];
    let dist_sq = dx * dx + dy * dy + dz * dz;

    if dist_sq >= radius * radius {
        return None;
    }
    let dist = dist_sq.sqrt().max(1e-8);
    let normal = [dx / dist, dy / dist, dz / dist];
    Some(Contact {
        point: clamped,
        normal,
        depth: radius - dist,
    })
}

/// Negate a 3-vector.
fn neg3(v: [f32; 3]) -> [f32; 3] {
    [-v[0], -v[1], -v[2]]
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surfaces::{PlaneType, Surface};
    use atlas_core::spatial::Point3;

    fn floor_surface(y: f32) -> Surface {
        Surface {
            normal: [0.0, 1.0, 0.0],
            offset: y,
            plane_type: PlaneType::Floor,
            centre: Point3::new(0.0, y, 0.0),
            inlier_count: 100,
        }
    }

    fn sphere_body(y: f32, radius: f32) -> RigidBody {
        RigidBody {
            id: 1,
            position: [0.0, y, 0.0],
            velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            mass: 1.0,
            inertia: [1.0; 3],
            shape: BoundingShape::Sphere { radius },
            friction: 0.5,
            restitution: 0.3,
            is_static: false,
        }
    }

    fn box_body(pos: [f32; 3], half_extents: [f32; 3]) -> RigidBody {
        RigidBody {
            id: 2,
            position: pos,
            velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            mass: 1.0,
            inertia: [1.0; 3],
            shape: BoundingShape::Box { half_extents },
            friction: 0.5,
            restitution: 0.3,
            is_static: false,
        }
    }

    // ── sphere vs plane ───────────────────────────────────────────────────────

    #[test]
    fn test_sphere_vs_floor_contact() {
        let body = sphere_body(0.3, 0.5);
        let floor = floor_surface(0.0);
        let contact = CollisionDetector::body_vs_surface(&body, &floor);
        assert!(
            contact.is_some(),
            "sphere at y=0.3 with radius=0.5 must touch floor"
        );
        let c = contact.unwrap();
        assert!((c.depth - 0.2).abs() < 1e-5, "depth = {}", c.depth);
    }

    #[test]
    fn test_sphere_vs_floor_no_contact() {
        let body = sphere_body(2.0, 0.5);
        let floor = floor_surface(0.0);
        assert!(CollisionDetector::body_vs_surface(&body, &floor).is_none());
    }

    #[test]
    fn test_sphere_on_floor_exactly() {
        // Centre at y = radius → touching but not penetrating.
        let body = sphere_body(0.5, 0.5);
        let floor = floor_surface(0.0);
        let contact = CollisionDetector::body_vs_surface(&body, &floor);
        assert!(contact.is_some());
        let c = contact.unwrap();
        assert!(c.depth >= 0.0);
    }

    // ── AABB vs plane ─────────────────────────────────────────────────────────

    #[test]
    fn test_aabb_vs_floor_contact() {
        // Box centred at y = 0.4 with half-extent 0.5 → bottom at y = −0.1.
        let body = box_body([0.0, 0.4, 0.0], [0.3, 0.5, 0.3]);
        let floor = floor_surface(0.0);
        let contact = CollisionDetector::body_vs_surface(&body, &floor);
        assert!(contact.is_some(), "box bottom at -0.1 must penetrate floor");
        let c = contact.unwrap();
        assert!(c.depth > 0.0);
    }

    #[test]
    fn test_aabb_vs_floor_no_contact() {
        let body = box_body([0.0, 2.0, 0.0], [0.3, 0.5, 0.3]);
        let floor = floor_surface(0.0);
        assert!(CollisionDetector::body_vs_surface(&body, &floor).is_none());
    }

    // ── Sphere vs sphere ──────────────────────────────────────────────────────

    #[test]
    fn test_sphere_vs_sphere_overlap() {
        let a = sphere_body(0.0, 1.0);
        let mut b = sphere_body(1.5, 1.0);
        b.id = 3;
        let contact = CollisionDetector::body_vs_body(&a, &b);
        assert!(
            contact.is_some(),
            "spheres 1.5m apart with r=1 each must overlap"
        );
        let c = contact.unwrap();
        assert!((c.depth - 0.5).abs() < 1e-4, "depth = {}", c.depth);
    }

    #[test]
    fn test_sphere_vs_sphere_separated() {
        let a = sphere_body(0.0, 0.5);
        let mut b = sphere_body(5.0, 0.5);
        b.id = 4;
        assert!(CollisionDetector::body_vs_body(&a, &b).is_none());
    }

    // ── AABB vs AABB ──────────────────────────────────────────────────────────

    #[test]
    fn test_aabb_vs_aabb_overlap() {
        let a = box_body([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let mut b = box_body([1.5, 0.0, 0.0], [1.0, 1.0, 1.0]);
        b.id = 5;
        let contact = CollisionDetector::body_vs_body(&a, &b);
        assert!(contact.is_some());
        let c = contact.unwrap();
        assert!((c.depth - 0.5).abs() < 1e-4, "depth = {}", c.depth);
    }

    #[test]
    fn test_aabb_vs_aabb_separated() {
        let a = box_body([0.0, 0.0, 0.0], [0.5, 0.5, 0.5]);
        let mut b = box_body([5.0, 0.0, 0.0], [0.5, 0.5, 0.5]);
        b.id = 6;
        assert!(CollisionDetector::body_vs_body(&a, &b).is_none());
    }

    // ── Sphere vs AABB ────────────────────────────────────────────────────────

    #[test]
    fn test_sphere_vs_aabb_overlap() {
        let a = sphere_body(0.0, 1.5);
        let b = box_body([2.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let contact = CollisionDetector::body_vs_body(&a, &b);
        // Closest point on box to sphere: x=1, y=0, z=0 → dist = 1 < radius 1.5
        assert!(contact.is_some());
    }

    #[test]
    fn test_sphere_vs_aabb_separated() {
        let a = sphere_body(0.0, 0.4);
        let b = box_body([5.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(CollisionDetector::body_vs_body(&a, &b).is_none());
    }

    // ── Normal direction ──────────────────────────────────────────────────────

    #[test]
    fn test_sphere_vs_sphere_normal_direction() {
        // A is above B → normal should point upward (from B toward A).
        let a = sphere_body(1.0, 0.6);
        let mut b = sphere_body(0.0, 0.6);
        b.id = 7;
        let contact = CollisionDetector::body_vs_body(&a, &b).unwrap();
        assert!(
            contact.normal[1] > 0.0,
            "normal should point from B toward A (upward)"
        );
    }

    #[test]
    fn test_floor_normal_direction() {
        let body = sphere_body(0.3, 0.5);
        let floor = floor_surface(0.0);
        let c = CollisionDetector::body_vs_surface(&body, &floor).unwrap();
        assert!(
            (c.normal[1] - 1.0).abs() < 1e-6,
            "floor normal should be +Y"
        );
    }
}
