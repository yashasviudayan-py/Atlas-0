//! Semantic metadata types for scene understanding.

use crate::spatial::Point3;
use serde::{Deserialize, Serialize};

/// Physical material properties assigned by the VLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialProperties {
    /// Estimated mass in kilograms.
    pub mass_kg: f32,
    /// Coefficient of friction (0.0 = frictionless, 1.0 = very rough).
    pub friction: f32,
    /// Fragility score (0.0 = indestructible, 1.0 = extremely fragile).
    pub fragility: f32,
    /// Material category.
    pub material: MaterialType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaterialType {
    Metal,
    Wood,
    Glass,
    Ceramic,
    Plastic,
    Fabric,
    Liquid,
    Organic,
    Unknown,
}

/// A semantically labeled object in the scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticObject {
    /// Unique object identifier within the scene.
    pub id: u64,
    /// Human-readable label (e.g., "coffee mug", "laptop").
    pub label: String,
    /// Center of mass in world coordinates.
    pub position: Point3,
    /// Axis-aligned bounding box: (min, max) corners.
    pub bbox_min: Point3,
    pub bbox_max: Point3,
    /// Physical properties.
    pub properties: MaterialProperties,
    /// Confidence score from the VLM (0.0 to 1.0).
    pub confidence: f32,
    /// Relationships to other objects (e.g., "on top of", "adjacent to").
    pub relationships: Vec<ObjectRelation>,
}

/// A spatial relationship between two objects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectRelation {
    pub target_id: u64,
    pub relation_type: RelationType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationType {
    OnTopOf,
    Inside,
    AdjacentTo,
    Supporting,
    Hanging,
    Leaning,
}

/// A predicted physical risk in the scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// The object at risk.
    pub object_id: u64,
    /// Type of risk.
    pub risk_type: RiskType,
    /// Probability of occurrence (0.0 to 1.0).
    pub probability: f32,
    /// Predicted impact zone center.
    pub impact_point: Option<Point3>,
    /// Human-readable description.
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskType {
    Fall,
    Spill,
    Collision,
    TripHazard,
    Instability,
}
