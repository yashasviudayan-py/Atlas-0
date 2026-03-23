//! Feature detection and description for visual SLAM.
//!
//! Implements FAST-9 corner detection and BRIEF binary descriptors for
//! extracting and describing visual features from grayscale images.
//!
//! # Pipeline
//! 1. Convert an RGB [`atlas_core::Frame`] to grayscale via [`FeatureExtractor::rgb_to_gray`].
//! 2. Call [`FeatureExtractor::extract`] to obtain `(keypoints, descriptors)`.
//! 3. Pass the descriptors to [`crate::matching::FeatureMatcher`].

use image::{GrayImage, Luma};

/// The FAST-9 Bresenham circle of radius 3 — 16 pixel offsets in CCW order.
const FAST_CIRCLE: [(i32, i32); 16] = [
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
];

/// Indices of the four cardinal points used for the FAST early-rejection test.
const FAST_CARDINAL: [usize; 4] = [0, 4, 8, 12];

// ─── KeyPoint ───────────────────────────────────────────────────────────────

/// A detected corner keypoint in an image.
#[derive(Debug, Clone)]
pub struct KeyPoint {
    /// Horizontal pixel coordinate (column).
    pub x: f32,
    /// Vertical pixel coordinate (row).
    pub y: f32,
    /// Corner response score — higher values indicate stronger corners.
    pub response: f32,
}

// ─── Descriptor ─────────────────────────────────────────────────────────────

/// A 256-bit BRIEF binary descriptor stored as 32 bytes.
///
/// # Examples
/// ```
/// use atlas_slam::features::Descriptor;
///
/// let a = Descriptor([0x00u8; 32]);
/// let b = Descriptor([0xFFu8; 32]);
/// assert_eq!(a.hamming_distance(&b), 256);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Descriptor(pub [u8; 32]);

impl Descriptor {
    /// Compute the Hamming distance to another descriptor.
    ///
    /// The result is in `[0, 256]`: 0 means identical, 256 means all bits differ.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
}

// ─── FeatureExtractor ───────────────────────────────────────────────────────

/// Extracts keypoints and BRIEF descriptors from grayscale images.
///
/// # Examples
/// ```
/// use atlas_slam::features::FeatureExtractor;
/// use image::GrayImage;
///
/// let extractor = FeatureExtractor::new(20, 1000);
/// let img = GrayImage::from_pixel(64, 64, image::Luma([128u8]));
/// let (kps, descs) = extractor.extract(&img);
/// assert_eq!(kps.len(), descs.len());
/// ```
pub struct FeatureExtractor {
    /// FAST corner detection intensity-difference threshold.
    fast_threshold: u8,
    /// Maximum number of keypoints returned after non-maximum suppression.
    max_keypoints: usize,
    /// Pre-computed BRIEF test pairs `(dx1, dy1, dx2, dy2)` in a 31×31 patch.
    brief_pairs: [(i32, i32, i32, i32); 256],
}

impl FeatureExtractor {
    /// Create a new feature extractor.
    ///
    /// # Arguments
    /// * `fast_threshold` — Intensity-difference threshold for FAST (typical: 10–30).
    /// * `max_keypoints` — Maximum keypoints per frame (typical: 500–2000).
    ///
    /// # Examples
    /// ```
    /// use atlas_slam::features::FeatureExtractor;
    /// let extractor = FeatureExtractor::new(20, 1000);
    /// ```
    #[must_use]
    pub fn new(fast_threshold: u8, max_keypoints: usize) -> Self {
        Self {
            fast_threshold,
            max_keypoints,
            brief_pairs: Self::generate_brief_pairs(),
        }
    }

    /// Extract keypoints and BRIEF descriptors from a grayscale image.
    ///
    /// Returns `(keypoints, descriptors)` where `keypoints[i]` corresponds to
    /// `descriptors[i]`.  Both slices always have the same length.
    #[must_use]
    pub fn extract(&self, image: &GrayImage) -> (Vec<KeyPoint>, Vec<Descriptor>) {
        let (width, height) = image.dimensions();
        // Need at least a 7×7 image to accommodate the FAST radius-3 circle border.
        if width < 7 || height < 7 {
            return (Vec::new(), Vec::new());
        }

        let mut keypoints = self.detect_fast(image);
        keypoints = grid_nms(keypoints, width, height, 7, self.max_keypoints);

        let descriptors = keypoints
            .iter()
            .map(|kp| self.compute_brief(image, kp))
            .collect();

        (keypoints, descriptors)
    }

    /// Convert packed RGB data to a [`GrayImage`] using the standard luminance formula.
    ///
    /// # Panics
    /// Panics (in debug) if `rgb_data.len() != width * height * 3`.
    #[must_use]
    pub fn rgb_to_gray(rgb_data: &[u8], width: u32, height: u32) -> GrayImage {
        GrayImage::from_fn(width, height, |x, y| {
            let idx = (y * width + x) as usize * 3;
            let r = f32::from(rgb_data[idx]);
            let g = f32::from(rgb_data[idx + 1]);
            let b = f32::from(rgb_data[idx + 2]);
            Luma([(0.299 * r + 0.587 * g + 0.114 * b) as u8])
        })
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Detect FAST-9 corners across the image interior (3-pixel border excluded).
    fn detect_fast(&self, image: &GrayImage) -> Vec<KeyPoint> {
        let (width, height) = image.dimensions();
        let threshold = self.fast_threshold;
        let mut keypoints = Vec::new();

        for y in 3..height.saturating_sub(3) {
            for x in 3..width.saturating_sub(3) {
                let center = i32::from(image.get_pixel(x, y)[0]);
                let hi = center + i32::from(threshold);
                let lo = center - i32::from(threshold);

                // Early rejection: at least 3 of 4 cardinal pixels must be
                // clearly brighter or clearly darker than the centre.
                let mut bright_card = 0u8;
                let mut dark_card = 0u8;
                for &ci in &FAST_CARDINAL {
                    let (dx, dy) = FAST_CIRCLE[ci];
                    let v = i32::from(
                        image.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32)[0],
                    );
                    if v > hi {
                        bright_card += 1;
                    } else if v < lo {
                        dark_card += 1;
                    }
                }
                if bright_card < 3 && dark_card < 3 {
                    continue;
                }

                // Full FAST-9 test: any 9 consecutive circle pixels all bright
                // or all dark?
                let vals: Vec<i32> = FAST_CIRCLE
                    .iter()
                    .map(|&(dx, dy)| {
                        i32::from(
                            image.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32)[0],
                        )
                    })
                    .collect();

                if is_fast_corner(&vals, hi, lo) {
                    let response = compute_fast_score(&vals, center);
                    keypoints.push(KeyPoint {
                        x: x as f32,
                        y: y as f32,
                        response,
                    });
                }
            }
        }

        keypoints
    }

    /// Compute a BRIEF descriptor for `kp` by comparing 256 pixel intensity pairs.
    fn compute_brief(&self, image: &GrayImage, kp: &KeyPoint) -> Descriptor {
        let (w, h) = image.dimensions();
        let cx = kp.x as i32;
        let cy = kp.y as i32;
        let mut bits = [0u8; 32];

        for (i, &(dx1, dy1, dx2, dy2)) in self.brief_pairs.iter().enumerate() {
            let nx1 = (cx + dx1).clamp(0, w as i32 - 1) as u32;
            let ny1 = (cy + dy1).clamp(0, h as i32 - 1) as u32;
            let nx2 = (cx + dx2).clamp(0, w as i32 - 1) as u32;
            let ny2 = (cy + dy2).clamp(0, h as i32 - 1) as u32;

            let p1 = image.get_pixel(nx1, ny1)[0];
            let p2 = image.get_pixel(nx2, ny2)[0];

            if p1 > p2 {
                bits[i / 8] |= 1 << (i % 8);
            }
        }

        Descriptor(bits)
    }

    /// Generate 256 BRIEF test-pair offsets deterministically using an LCG.
    ///
    /// Each offset `(dx1, dy1, dx2, dy2)` is in `[-15, 15]` (a 31×31 patch).
    /// The same seed is always used so results are reproducible across runs.
    fn generate_brief_pairs() -> [(i32, i32, i32, i32); 256] {
        let mut pairs = [(0i32, 0i32, 0i32, 0i32); 256];
        // Knuth multiplicative LCG.
        let mut seed: u64 = 0x5EED_1234_ABCD_EF01;

        for pair in &mut pairs {
            let mut next = || -> i32 {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((seed >> 33) as i32).rem_euclid(31) - 15
            };
            *pair = (next(), next(), next(), next());
        }

        pairs
    }
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Return `true` if the 16 circle values contain 9 consecutive pixels that
/// are all above `hi` or all below `lo` (FAST-9 criterion).
fn is_fast_corner(vals: &[i32], hi: i32, lo: i32) -> bool {
    let n = vals.len(); // always 16
    let bright: Vec<bool> = vals.iter().map(|&v| v > hi).collect();
    let dark: Vec<bool> = vals.iter().map(|&v| v < lo).collect();

    (0..n).any(|i| (0..9).all(|j| bright[(i + j) % n]))
        || (0..n).any(|i| (0..9).all(|j| dark[(i + j) % n]))
}

/// Sum of absolute intensity differences from centre — used as the corner score.
fn compute_fast_score(vals: &[i32], center: i32) -> f32 {
    vals.iter()
        .map(|&v| (v - center).unsigned_abs() as f32)
        .sum()
}

/// Grid-based non-maximum suppression.
///
/// Divides the image into `cell_size × cell_size` cells and keeps only the
/// highest-response keypoint per cell.  Results are sorted by response
/// (descending) and capped at `max_kps`.
fn grid_nms(
    mut keypoints: Vec<KeyPoint>,
    width: u32,
    height: u32,
    cell_size: u32,
    max_kps: usize,
) -> Vec<KeyPoint> {
    keypoints.sort_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let cols = width.div_ceil(cell_size);
    let rows = height.div_ceil(cell_size);
    let mut occupied = vec![false; (cols * rows) as usize];
    let mut result = Vec::with_capacity(max_kps);

    for kp in keypoints {
        if result.len() >= max_kps {
            break;
        }
        let col = (kp.x as u32 / cell_size).min(cols - 1);
        let row = (kp.y as u32 / cell_size).min(rows - 1);
        let idx = (row * cols + col) as usize;
        if !occupied[idx] {
            occupied[idx] = true;
            result.push(kp);
        }
    }

    result
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance_same() {
        let d = Descriptor([0xABu8; 32]);
        assert_eq!(d.hamming_distance(&d), 0);
    }

    #[test]
    fn test_hamming_distance_all_bits() {
        let a = Descriptor([0x00u8; 32]);
        let b = Descriptor([0xFFu8; 32]);
        assert_eq!(a.hamming_distance(&b), 256);
    }

    #[test]
    fn test_extract_tiny_image_returns_empty() {
        let extractor = FeatureExtractor::new(20, 1000);
        let img = GrayImage::new(4, 4);
        let (kps, descs) = extractor.extract(&img);
        assert_eq!(kps.len(), descs.len());
        assert!(kps.is_empty());
    }

    #[test]
    fn test_extract_uniform_no_corners() {
        let extractor = FeatureExtractor::new(20, 1000);
        let img = GrayImage::from_pixel(100, 100, Luma([128u8]));
        let (kps, descs) = extractor.extract(&img);
        assert_eq!(kps.len(), descs.len());
        assert!(kps.is_empty(), "uniform image must have no corners");
    }

    #[test]
    fn test_extract_isolated_pixel_is_corner() {
        // FAST-9 is guaranteed to detect an isolated bright pixel on a dark
        // background: all 16 circle pixels are below centre−threshold, giving
        // 16 consecutive "dark" pixels which satisfies the FAST-9 criterion.
        let extractor = FeatureExtractor::new(20, 1000);
        let mut img = GrayImage::from_pixel(64, 64, Luma([30u8]));
        img.put_pixel(32, 32, Luma([200u8]));
        let (kps, descs) = extractor.extract(&img);
        assert_eq!(kps.len(), descs.len());
        assert!(
            !kps.is_empty(),
            "isolated bright pixel must be detected as a corner"
        );
    }

    #[test]
    fn test_brief_pairs_deterministic() {
        // Pairs depend only on the fixed LCG seed, not on extractor parameters.
        let e1 = FeatureExtractor::new(20, 100);
        let e2 = FeatureExtractor::new(10, 500);
        assert_eq!(e1.brief_pairs, e2.brief_pairs);
    }

    #[test]
    fn test_rgb_to_gray_dimensions() {
        let rgb = vec![100u8; 32 * 32 * 3];
        let gray = FeatureExtractor::rgb_to_gray(&rgb, 32, 32);
        assert_eq!(gray.dimensions(), (32, 32));
    }
}
