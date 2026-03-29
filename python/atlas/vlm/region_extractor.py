"""Region extractor: identify spatial clusters in a Gaussian map snapshot.

Segments the Gaussian point cloud into distinct object regions using DBSCAN
spatial clustering, then renders a simple top-down projection image for each
cluster to pass to the VLM.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import structlog
from PIL import Image

from atlas.utils.shared_mem import MapSnapshot

logger = structlog.get_logger(__name__)

try:
    from sklearn.cluster import DBSCAN as _DBSCAN  # type: ignore[import-untyped]

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning(
        "sklearn_not_available",
        note="DBSCAN clustering disabled; all Gaussians treated as one region.",
    )


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in 3D world space.

    Args:
        x_min: Minimum X coordinate.
        y_min: Minimum Y coordinate.
        z_min: Minimum Z coordinate.
        x_max: Maximum X coordinate.
        y_max: Maximum Y coordinate.
        z_max: Maximum Z coordinate.
    """

    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

    @property
    def center(self) -> tuple[float, float, float]:
        """Geometric centre of the bounding box."""
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
            (self.z_min + self.z_max) / 2.0,
        )

    @property
    def size(self) -> tuple[float, float, float]:
        """Width, height, depth of the bounding box."""
        return (
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min,
        )


@dataclass
class SceneRegion:
    """A spatially coherent cluster of Gaussians representing one object region.

    Args:
        region_id: Integer cluster identifier from DBSCAN (≥ 0).
        image_bytes: JPEG-encoded top-down projection image of the cluster.
        bbox: Axis-aligned bounding box in world space.
        gaussian_indices: Indices into the source ``MapSnapshot.gaussians`` array.
        gaussian_count: Number of Gaussians in this region.
    """

    region_id: int
    image_bytes: bytes
    bbox: BoundingBox
    gaussian_indices: np.ndarray
    gaussian_count: int


class RegionExtractor:
    """Extracts distinct object regions from a Gaussian map snapshot.

    Uses DBSCAN spatial clustering on Gaussian centres to identify cohesive
    object groups, then renders a top-down projection image for each cluster.

    Args:
        eps_metres: DBSCAN neighbourhood radius in metres.
        min_samples: Minimum Gaussians required to form a cluster.
        max_regions: Upper bound on regions returned per snapshot.
        image_size: Pixel side length of rendered region images.
        min_opacity: Gaussians below this opacity threshold are ignored.

    Example::

        extractor = RegionExtractor(eps_metres=0.15, min_samples=10)
        regions = extractor.extract_regions(snapshot)
        for region in regions:
            label = await vlm_engine.label_region(region.image_bytes)
    """

    def __init__(
        self,
        eps_metres: float = 0.15,
        min_samples: int = 10,
        max_regions: int = 20,
        image_size: int = 128,
        min_opacity: float = 0.1,
    ) -> None:
        self._eps = eps_metres
        self._min_samples = min_samples
        self._max_regions = max_regions
        self._image_size = image_size
        self._min_opacity = min_opacity

    def extract_regions(self, snapshot: MapSnapshot) -> list[SceneRegion]:
        """Segment *snapshot* into distinct object regions.

        Args:
            snapshot: A :class:`~atlas.utils.shared_mem.MapSnapshot` produced
                by the Rust SLAM pipeline.

        Returns:
            List of :class:`SceneRegion` objects, sorted by Gaussian count
            (largest first).  Returns an empty list when the snapshot contains
            no usable Gaussians.
        """
        gaussians = snapshot.gaussians
        if len(gaussians) == 0:
            logger.debug("region_extraction_empty_snapshot")
            return []

        opaque_mask = gaussians["opacity"] >= self._min_opacity
        opaque_indices = np.where(opaque_mask)[0]
        if len(opaque_indices) == 0:
            logger.debug("region_extraction_all_transparent")
            return []

        positions = np.column_stack(
            [
                gaussians["x"][opaque_indices].astype(np.float64),
                gaussians["y"][opaque_indices].astype(np.float64),
                gaussians["z"][opaque_indices].astype(np.float64),
            ]
        )

        cluster_labels = self._cluster(positions)

        unique_labels = np.unique(cluster_labels)
        regions: list[SceneRegion] = []

        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue  # DBSCAN noise

            mask = cluster_labels == cluster_id
            cluster_positions = positions[mask]
            original_indices = opaque_indices[mask]

            bbox = BoundingBox(
                x_min=float(cluster_positions[:, 0].min()),
                y_min=float(cluster_positions[:, 1].min()),
                z_min=float(cluster_positions[:, 2].min()),
                x_max=float(cluster_positions[:, 0].max()),
                y_max=float(cluster_positions[:, 1].max()),
                z_max=float(cluster_positions[:, 2].max()),
            )

            colors = np.column_stack(
                [
                    gaussians["r"][original_indices].astype(np.float64),
                    gaussians["g"][original_indices].astype(np.float64),
                    gaussians["b"][original_indices].astype(np.float64),
                ]
            )
            image_bytes = self._render_region(cluster_positions, colors, bbox)

            regions.append(
                SceneRegion(
                    region_id=int(cluster_id),
                    image_bytes=image_bytes,
                    bbox=bbox,
                    gaussian_indices=original_indices,
                    gaussian_count=int(mask.sum()),
                )
            )

        regions.sort(key=lambda r: r.gaussian_count, reverse=True)
        result = regions[: self._max_regions]

        logger.debug(
            "region_extraction_done",
            total_gaussians=len(gaussians),
            opaque_gaussians=len(opaque_indices),
            regions_found=len(result),
        )
        return result

    def _cluster(self, positions: np.ndarray) -> np.ndarray:
        """Assign DBSCAN cluster labels to *positions*.

        Args:
            positions: Float64 array of shape ``(N, 3)``.

        Returns:
            Integer array of shape ``(N,)`` with DBSCAN cluster labels.  Label
            ``-1`` indicates noise.  Returns all-zeros (single cluster) when
            scikit-learn is unavailable.
        """
        if not _SKLEARN_AVAILABLE:
            return np.zeros(len(positions), dtype=np.intp)

        db = _DBSCAN(eps=self._eps, min_samples=self._min_samples, algorithm="ball_tree")
        result: np.ndarray = db.fit_predict(positions)
        return result

    def _render_region(
        self,
        positions: np.ndarray,
        colors: np.ndarray,
        bbox: BoundingBox,
    ) -> bytes:
        """Render a top-down XZ projection of a Gaussian cluster as JPEG bytes.

        Projects Gaussian centres onto the XZ plane (Y is up) and paints each
        point using its mean colour on a light-grey canvas.

        Args:
            positions: Float64 array of shape ``(N, 3)`` — world-space centres.
            colors: Float64 array of shape ``(N, 3)`` — R/G/B values in [0, 1].
            bbox: Bounding box of the cluster used for normalisation.

        Returns:
            JPEG-encoded image bytes.
        """
        size = self._image_size
        canvas = np.full((size, size, 3), 240, dtype=np.uint8)

        x_range = bbox.x_max - bbox.x_min
        z_range = bbox.z_max - bbox.z_min
        pad = 0.05  # 5 % margin on each side

        x_span = x_range if x_range > 1e-6 else 1.0
        z_span = z_range if z_range > 1e-6 else 1.0

        # Map X → pixel column, Z → pixel row.
        u = ((positions[:, 0] - bbox.x_min) / x_span * (1.0 - 2 * pad) + pad) * (size - 1)
        v = ((positions[:, 2] - bbox.z_min) / z_span * (1.0 - 2 * pad) + pad) * (size - 1)

        u_px = np.clip(u.astype(np.int32), 0, size - 1)
        v_px = np.clip(v.astype(np.int32), 0, size - 1)

        rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        canvas[v_px, u_px] = rgb

        img = Image.fromarray(canvas, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
