"""World Model Agent: continuously assesses scene risks.

The agent runs as a background loop, periodically pulling the latest
Gaussian map snapshot from shared memory, labeling new regions via the
VLM, computing spatial relationships, and maintaining a ranked risk list.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from dataclasses import dataclass

import structlog

from atlas.vlm.inference import VLMConfig, VLMEngine
from atlas.vlm.region_extractor import BoundingBox, RegionExtractor
from atlas.world_model.label_store import LabelStore
from atlas.world_model.relationships import RelationshipDetector, SemanticObject

logger = structlog.get_logger(__name__)


@dataclass
class WorldModelConfig:
    """Configuration for the world model agent.

    Args:
        assessment_interval_seconds: Delay between assessment cycles.
        max_concurrent_queries: (Reserved) max parallel VLM calls.
        risk_threshold: Minimum risk_score for an object to appear in the list.
        vlm_rate_limit_seconds: Minimum wall-clock gap between VLM calls.
    """

    assessment_interval_seconds: float = 1.0
    max_concurrent_queries: int = 3
    risk_threshold: float = 0.3
    vlm_rate_limit_seconds: float = 0.5


@dataclass
class RiskEntry:
    """A single entry in the current risk list.

    Args:
        object_id: DBSCAN cluster ID of the at-risk object.
        object_label: Human-readable label.
        position: World-space centre (x, y, z).
        risk_score: Heuristic score in [0, 1].
        fragility: Fragility property of the object.
        mass_kg: Estimated mass.
        description: Human-readable risk summary.
    """

    object_id: int
    object_label: str
    position: tuple[float, float, float]
    risk_score: float
    fragility: float
    mass_kg: float
    description: str


class WorldModelAgent:
    """Background agent that continuously assesses physical risks.

    Orchestrates the full VLM labeling pipeline:

    1. Read the latest :class:`~atlas.utils.shared_mem.MapSnapshot` from Rust
       via shared memory.
    2. Segment into :class:`~atlas.vlm.region_extractor.SceneRegion` clusters.
    3. Label each region via :class:`~atlas.vlm.inference.VLMEngine` (rate-limited).
    4. Detect spatial relationships between labeled objects.
    5. Score risks and update the ranked risk list.

    All state (label store, risk list, cached objects) is managed internally.
    External callers use :meth:`get_objects` and :meth:`get_risks`.

    Example::

        agent = WorldModelAgent()
        await agent.start()
        risks = await agent.get_risks()
        await agent.stop()
    """

    def __init__(
        self,
        config: WorldModelConfig | None = None,
        vlm_engine: VLMEngine | None = None,
        label_store: LabelStore | None = None,
        region_extractor: RegionExtractor | None = None,
        relationship_detector: RelationshipDetector | None = None,
    ) -> None:
        self.config = config or WorldModelConfig()
        self._vlm = vlm_engine or VLMEngine(VLMConfig())
        self._label_store = label_store or LabelStore()
        self._region_extractor = region_extractor or RegionExtractor()
        self._rel_detector = relationship_detector or RelationshipDetector()

        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._vlm_initialized = False

        # Thread-safe(ish) state updated on every assessment cycle.
        self._cached_objects: list[SemanticObject] = []
        self._risks: list[RiskEntry] = []
        self._risks_lock = asyncio.Lock()
        self._objects_lock = asyncio.Lock()

        # Rate-limiting: monotonic timestamp of the last VLM call.
        self._last_vlm_time: float = 0.0

        # Staleness tracking: monotonic timestamp of the last *successful*
        # assessment cycle.  0.0 means no assessment has completed yet.
        self._last_assessment_time: float = 0.0

        # Shared-memory reader (lazy-initialised on first assessment).
        self._shared_mem_reader: object | None = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def label_store(self) -> LabelStore:
        """The label store used by this agent."""
        return self._label_store

    @property
    def vlm_active(self) -> bool:
        """``True`` if the VLM engine has been successfully initialised."""
        return self._vlm_initialized

    @property
    def risks_stale_seconds(self) -> float:
        """Seconds since the last successful assessment cycle.

        Returns ``float('inf')`` when no assessment has completed yet, so
        callers can treat any large value as "data not available".

        Returns:
            Elapsed seconds since the last successful :meth:`_assess_scene`
            call, or ``float('inf')`` if no assessment has run.
        """
        if self._last_assessment_time == 0.0:
            return float("inf")
        return time.monotonic() - self._last_assessment_time

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background assessment loop.

        Initialises the VLM engine and begins periodic scene assessments.
        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._running:
            return
        self._running = True
        await self._vlm.initialize()
        self._vlm_initialized = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("world_model_agent_started")

    async def stop(self) -> None:
        """Stop the background assessment loop and release resources."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self._vlm.close()
        logger.info("world_model_agent_stopped")

    # ── Public accessors ──────────────────────────────────────────────────────

    async def get_risks(self) -> list[RiskEntry]:
        """Return a snapshot of the current risk list, sorted by score.

        Returns:
            Copy of the current :class:`RiskEntry` list.
        """
        async with self._risks_lock:
            return list(self._risks)

    async def get_objects(self) -> list[SemanticObject]:
        """Return a snapshot of all currently labeled semantic objects.

        Returns:
            Copy of the cached :class:`SemanticObject` list.
        """
        async with self._objects_lock:
            return list(self._cached_objects)

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._assess_scene()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("world_model_assessment_error")
            await asyncio.sleep(self.config.assessment_interval_seconds)

    # ── Core pipeline ─────────────────────────────────────────────────────────

    async def _assess_scene(self) -> None:
        """Run one full scene assessment cycle.

        Steps:
        1. Get current Gaussian map snapshot from Rust via shared memory.
        2. Extract spatially coherent object regions (DBSCAN clustering).
        3. Label each new / changed region via the VLM (rate-limited).
        4. Compute pairwise spatial relationships.
        5. Score and rank object risks; update the cached lists.
        """
        snapshot = self._get_snapshot()
        if snapshot is None:
            return

        regions = self._region_extractor.extract_regions(snapshot)
        if not regions:
            return

        labeled_objects: list[SemanticObject] = []

        for region in regions:
            await self._rate_limit_vlm()

            _vlm_start = time.monotonic()
            label = await self._vlm.label_region(
                region.image_bytes,
                region_hint=f"region {region.region_id}",
            )
            _vlm_elapsed = time.monotonic() - _vlm_start
            self._last_vlm_time = time.monotonic()
            self._record_vlm_latency(_vlm_elapsed)

            updated = self._label_store.update(region.region_id, label)
            logger.debug(
                "region_labeled",
                region_id=region.region_id,
                label=label.label,
                store_updated=updated,
            )

            labeled_objects.append(
                SemanticObject(
                    object_id=region.region_id,
                    label=label.label,
                    material=label.material,
                    mass_kg=label.mass_kg,
                    fragility=label.fragility,
                    friction=label.friction,
                    confidence=label.confidence,
                    bbox=region.bbox,
                )
            )

        if not labeled_objects:
            return

        self._rel_detector.compute_relationships(labeled_objects)

        risks = self._compute_risks(labeled_objects)

        async with self._objects_lock:
            self._cached_objects = labeled_objects

        async with self._risks_lock:
            self._risks = risks

        self._last_assessment_time = time.monotonic()

        logger.debug(
            "scene_assessed",
            regions=len(labeled_objects),
            risks=len(risks),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _rate_limit_vlm(self) -> None:
        """Sleep until the VLM rate-limit interval has elapsed."""
        elapsed = time.monotonic() - self._last_vlm_time
        if elapsed < self.config.vlm_rate_limit_seconds:
            await asyncio.sleep(self.config.vlm_rate_limit_seconds - elapsed)

    def _compute_risks(self, objects: list[SemanticObject]) -> list[RiskEntry]:
        """Build and sort the risk list from *objects*.

        Args:
            objects: Labeled objects with relationships already computed.

        Returns:
            List of :class:`RiskEntry` instances above the risk threshold,
            sorted by descending risk score.
        """
        entries: list[RiskEntry] = []
        for obj in objects:
            score = obj.risk_score
            if score < self.config.risk_threshold:
                continue
            entries.append(
                RiskEntry(
                    object_id=obj.object_id,
                    object_label=obj.label,
                    position=obj.position,
                    risk_score=score,
                    fragility=obj.fragility,
                    mass_kg=obj.mass_kg,
                    description=self._risk_description(obj),
                )
            )
        entries.sort(key=lambda r: r.risk_score, reverse=True)
        return entries

    @staticmethod
    def _risk_description(obj: SemanticObject) -> str:
        """Generate a concise human-readable description of why obj is at risk.

        Args:
            obj: The at-risk semantic object.

        Returns:
            Short descriptive string, e.g. ``"glass (glass), fragile, elevated"``.
        """
        parts = [f"{obj.label} ({obj.material})"]
        if obj.fragility > 0.7:
            parts.append("fragile")
        if obj.mass_kg > 3.0:
            parts.append("heavy")
        rel_types = {r.relation.value for r in obj.relationships}
        if "on_top_of" in rel_types:
            parts.append("elevated")
        if "leaning" in rel_types:
            parts.append("unstable/leaning")
        return ", ".join(parts)

    @staticmethod
    def _record_vlm_latency(elapsed: float) -> None:
        """Record *elapsed* seconds to the Prometheus VLM histogram if available.

        Args:
            elapsed: Duration of the VLM call in seconds.
        """
        try:
            from atlas.api.metrics import vlm_request_seconds  # type: ignore[attr-defined]

            vlm_request_seconds.observe(elapsed)
        except Exception:  # pragma: no cover — metrics import failure is non-fatal
            pass

    def _get_snapshot(self) -> object | None:
        """Read the latest Gaussian map snapshot from shared memory.

        Lazily initialises the :class:`~atlas.utils.shared_mem.SharedMemReader`
        on the first call using the ``ATLAS_MMAP_PATH`` environment variable
        (default: ``/tmp/atlas_gaussians.bin``).

        Returns:
            A :class:`~atlas.utils.shared_mem.MapSnapshot` or ``None`` when
            shared memory is not available (e.g. Rust pipeline not running).
        """
        try:
            from atlas.utils.shared_mem import SharedMemReader  # type: ignore[attr-defined]

            if self._shared_mem_reader is None:
                mmap_path = os.environ.get("ATLAS_MMAP_PATH", "/tmp/atlas_gaussians.bin")
                self._shared_mem_reader = SharedMemReader(mmap_path)

            return self._shared_mem_reader.get_latest_snapshot()  # type: ignore[union-attr]
        except Exception:
            return None

    # ── Synchronous convenience (for server dependency injection) ─────────────

    def get_objects_sync(self) -> list[SemanticObject]:
        """Return the last cached object list without acquiring the async lock.

        Safe to call from synchronous code (e.g. FastAPI sync path) because
        the list reference is swapped atomically by the assessment loop.

        Returns:
            The cached :class:`SemanticObject` list (may be stale by up to
            *assessment_interval_seconds*).
        """
        return list(self._cached_objects)

    def build_objects_from_store(self) -> list[SemanticObject]:
        """Build a :class:`SemanticObject` list directly from the label store.

        Used as a fallback when the assessment loop has not yet run.
        Objects returned here have zero-sized bounding boxes (no spatial data).

        Returns:
            List of :class:`SemanticObject` with only label-store properties.
        """
        _zero_bbox = BoundingBox(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        result: list[SemanticObject] = []
        for entry in self._label_store.all_entries():
            sl = entry.label
            result.append(
                SemanticObject(
                    object_id=entry.object_id,
                    label=sl.label,
                    material=sl.material,
                    mass_kg=sl.mass_kg,
                    fragility=sl.fragility,
                    friction=sl.friction,
                    confidence=sl.confidence,
                    bbox=_zero_bbox,
                )
            )
        return result
