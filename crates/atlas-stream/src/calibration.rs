//! Camera calibration loader.
//!
//! Loads camera intrinsics (fx, fy, cx, cy) and distortion coefficients from
//! the `[camera]` section of `configs/default.toml`. The model is the standard
//! OpenCV-compatible pinhole + radial/tangential distortion model.
//!
//! # Config layout
//! ```toml
//! [camera]
//! slam_width  = 640
//! slam_height = 480
//!
//! [camera.intrinsics]
//! fx = 458.654
//! fy = 457.296
//! cx = 367.215
//! cy = 248.375
//!
//! [camera.distortion]
//! k1 = -0.283408
//! k2 =  0.073959
//! p1 =  0.000194
//! p2 =  0.000018
//! ```
//!
//! # Example
//! ```no_run
//! use atlas_stream::calibration::CameraCalibration;
//!
//! let toml_str = std::fs::read_to_string("configs/default.toml").unwrap();
//! let cal: CameraCalibration = toml::from_str::<toml::Value>(&toml_str)
//!     .unwrap()["camera"]
//!     .clone()
//!     .try_into()
//!     .unwrap();
//! assert!(cal.intrinsics.fx > 0.0);
//! ```

use serde::{Deserialize, Serialize};

/// Pinhole camera intrinsic parameters in pixel units.
///
/// Matches the OpenCV `camera_matrix` convention:
/// ```text
/// K = [[fx,  0, cx],
///      [ 0, fy, cy],
///      [ 0,  0,  1]]
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    /// Focal length in x (pixels).
    pub fx: f64,
    /// Focal length in y (pixels).
    pub fy: f64,
    /// Principal point x (pixels).
    pub cx: f64,
    /// Principal point y (pixels).
    pub cy: f64,
}

/// Radial and tangential distortion coefficients (OpenCV 4-coefficient model).
///
/// The forward distortion model is:
/// ```text
/// r²       = x² + y²
/// radial   = 1 + k1·r² + k2·r⁴
/// x'       = x·radial + 2·p1·x·y + p2·(r² + 2·x²)
/// y'       = y·radial + p1·(r² + 2·y²) + 2·p2·x·y
/// ```
/// where `(x, y)` are normalised camera-frame coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistortionCoefficients {
    /// Radial distortion coefficient k1.
    pub k1: f64,
    /// Radial distortion coefficient k2.
    pub k2: f64,
    /// Tangential distortion coefficient p1.
    pub p1: f64,
    /// Tangential distortion coefficient p2.
    pub p2: f64,
}

impl Default for DistortionCoefficients {
    /// Zero distortion — behaves as a perfect pinhole lens.
    fn default() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }
}

/// Full camera calibration: intrinsics + distortion + SLAM working resolution.
///
/// Deserialize directly from the `[camera]` table in `configs/default.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraCalibration {
    /// Pinhole intrinsic parameters.
    pub intrinsics: CameraIntrinsics,
    /// Lens distortion coefficients.
    pub distortion: DistortionCoefficients,
    /// Target frame width for SLAM processing after resizing.
    pub slam_width: u32,
    /// Target frame height for SLAM processing after resizing.
    pub slam_height: u32,
}

impl Default for CameraCalibration {
    /// EuRoC MAV dataset cam0 parameters — reasonable defaults for development.
    fn default() -> Self {
        Self {
            intrinsics: CameraIntrinsics {
                fx: 458.654,
                fy: 457.296,
                cx: 367.215,
                cy: 248.375,
            },
            distortion: DistortionCoefficients {
                k1: -0.283_408,
                k2: 0.073_959,
                p1: 0.000_194,
                p2: 0.000_018,
            },
            slam_width: 640,
            slam_height: 480,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_has_positive_focal_lengths() {
        let cal = CameraCalibration::default();
        assert!(cal.intrinsics.fx > 0.0);
        assert!(cal.intrinsics.fy > 0.0);
    }

    #[test]
    fn default_slam_resolution_is_640x480() {
        let cal = CameraCalibration::default();
        assert_eq!(cal.slam_width, 640);
        assert_eq!(cal.slam_height, 480);
    }

    #[test]
    fn zero_distortion_default() {
        let d = DistortionCoefficients::default();
        assert_eq!(d.k1, 0.0);
        assert_eq!(d.k2, 0.0);
        assert_eq!(d.p1, 0.0);
        assert_eq!(d.p2, 0.0);
    }

    #[test]
    fn round_trip_toml_serialization() {
        let original = CameraCalibration::default();
        let serialized = toml::to_string(&original).expect("serialization failed");
        let deserialized: CameraCalibration =
            toml::from_str(&serialized).expect("deserialization failed");
        assert_eq!(original, deserialized);
    }

    #[test]
    fn calibration_clone_equals_original() {
        // Arrange
        let original = CameraCalibration::default();
        // Act
        let cloned = original.clone();
        // Assert
        assert_eq!(original, cloned);
    }

    #[test]
    fn intrinsic_fields_are_accessible() {
        // Arrange
        let cal = CameraCalibration::default();
        // Assert — spot-check against the known EuRoC defaults.
        assert!((cal.intrinsics.fx - 458.654).abs() < 1e-3);
        assert!((cal.intrinsics.fy - 457.296).abs() < 1e-3);
        assert!((cal.intrinsics.cx - 367.215).abs() < 1e-3);
        assert!((cal.intrinsics.cy - 248.375).abs() < 1e-3);
    }

    #[test]
    fn distortion_fields_are_accessible() {
        // Arrange
        let cal = CameraCalibration::default();
        // Assert — sign and magnitude correct for EuRoC defaults.
        assert!(cal.distortion.k1 < 0.0, "k1 should be negative (barrel)");
        assert!(cal.distortion.k2 > 0.0, "k2 should be positive");
    }

    #[test]
    fn toml_missing_intrinsics_section_returns_error() {
        // Arrange — [intrinsics] block is absent.
        let toml_str = r#"
            slam_width  = 640
            slam_height = 480

            [distortion]
            k1 = -0.28
            k2 =  0.07
            p1 =  0.0
            p2 =  0.0
        "#;
        // Act + Assert
        let result: Result<CameraCalibration, _> = toml::from_str(toml_str);
        assert!(result.is_err(), "missing [intrinsics] should fail to parse");
    }

    #[test]
    fn toml_missing_distortion_section_returns_error() {
        // Arrange — [distortion] block is absent.
        let toml_str = r#"
            slam_width  = 640
            slam_height = 480

            [intrinsics]
            fx = 458.654
            fy = 457.296
            cx = 367.215
            cy = 248.375
        "#;
        // Act + Assert
        let result: Result<CameraCalibration, _> = toml::from_str(toml_str);
        assert!(result.is_err(), "missing [distortion] should fail to parse");
    }

    #[test]
    fn distortion_coefficients_round_trip_toml() {
        // Arrange
        let original = DistortionCoefficients {
            k1: -0.5,
            k2: 0.1,
            p1: 0.002,
            p2: -0.003,
        };
        // Act
        let s = toml::to_string(&original).expect("serialize");
        let back: DistortionCoefficients = toml::from_str(&s).expect("deserialize");
        // Assert
        assert!((back.k1 - original.k1).abs() < 1e-9);
        assert!((back.k2 - original.k2).abs() < 1e-9);
        assert!((back.p1 - original.p1).abs() < 1e-9);
        assert!((back.p2 - original.p2).abs() < 1e-9);
    }

    #[test]
    fn deserialize_from_config_section() {
        let toml_str = r#"
            slam_width  = 640
            slam_height = 480

            [intrinsics]
            fx = 458.654
            fy = 457.296
            cx = 367.215
            cy = 248.375

            [distortion]
            k1 = -0.283408
            k2 =  0.073959
            p1 =  0.000194
            p2 =  0.000018
        "#;
        let cal: CameraCalibration = toml::from_str(toml_str).expect("parse failed");
        assert_eq!(cal.slam_width, 640);
        assert!((cal.intrinsics.fx - 458.654).abs() < 1e-6);
        assert!((cal.distortion.k1 - (-0.283408)).abs() < 1e-6);
    }
}
