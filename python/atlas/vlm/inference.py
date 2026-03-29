"""VLM inference engine for labeling 3D scene objects.

Runs a local Vision-Language Model (e.g., Moondream via Ollama)
to assign semantic metadata to Gaussian map shards.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import structlog

from atlas.vlm.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaModelError,
)
from atlas.vlm.prompts import (
    DEFAULT_LABEL_PROPERTIES,
    LABEL_REGION_V1,
    MATERIAL_DEFAULTS,
)

logger = structlog.get_logger(__name__)

# Matches the first JSON object in a string (handles chain-of-thought responses).
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass
class VLMConfig:
    """Configuration for the VLM inference engine."""

    model_name: str = "moondream"
    ollama_host: str = "http://localhost:11434"
    max_tokens: int = 256
    temperature: float = 0.1
    timeout_seconds: float = 30.0


@dataclass
class SemanticLabel:
    """Semantic label assigned to a scene region."""

    label: str
    material: str
    mass_kg: float
    fragility: float
    friction: float
    confidence: float


def _parse_label_response(text: str) -> SemanticLabel | None:
    """Extract a :class:`SemanticLabel` from raw VLM output.

    Tries strict JSON first, then a regex scan to find an embedded JSON object.
    The last match is preferred so chain-of-thought prompts (v2) work correctly.

    Args:
        text: Raw response string from Ollama.

    Returns:
        A parsed :class:`SemanticLabel` on success, ``None`` when no valid JSON
        with the required fields can be found.
    """
    candidates: list[str] = []

    stripped = text.strip()
    if stripped.startswith("{"):
        candidates.append(stripped)

    # Prefer the last JSON block (chain-of-thought puts the answer last).
    matches = _JSON_RE.findall(text)
    if matches:
        candidates.extend(reversed(matches))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        label_str = str(data.get("label", "unknown")).strip()
        if not label_str:
            continue

        material = str(data.get("material", "unknown")).strip().lower()
        mass_default, frag_default, fric_default = MATERIAL_DEFAULTS.get(
            material, DEFAULT_LABEL_PROPERTIES
        )

        try:
            mass_kg = float(data.get("mass_kg", mass_default))
            fragility = float(data.get("fragility", frag_default))
            friction = float(data.get("friction", fric_default))
        except (TypeError, ValueError):
            continue

        # Clamp to valid ranges.
        mass_kg = max(0.001, mass_kg)
        fragility = max(0.0, min(1.0, fragility))
        friction = max(0.0, min(1.0, friction))

        return SemanticLabel(
            label=label_str,
            material=material,
            mass_kg=mass_kg,
            fragility=fragility,
            friction=friction,
            confidence=0.8,
        )

    return None


def _fallback_label() -> SemanticLabel:
    """Return a low-confidence default used when parsing fails."""
    mass_kg, fragility, friction = DEFAULT_LABEL_PROPERTIES
    return SemanticLabel(
        label="unknown",
        material="unknown",
        mass_kg=mass_kg,
        fragility=fragility,
        friction=friction,
        confidence=0.1,
    )


class VLMEngine:
    """Runs VLM inference to generate semantic labels for scene objects.

    Uses a local Ollama instance to query a Vision-Language Model
    about regions of the 3D Gaussian map.

    Example::

        engine = VLMEngine(VLMConfig(model_name="moondream"))
        await engine.initialize()
        label = await engine.label_region(image_bytes, region_hint="shelf item")
        await engine.close()
    """

    def __init__(self, config: VLMConfig | None = None) -> None:
        self.config = config or VLMConfig()
        self._initialized = False
        self._client: OllamaClient | None = None
        logger.info("vlm_engine_created", model=self.config.model_name)

    async def initialize(self) -> None:
        """Connect to Ollama and verify/pull the configured model.

        Logs a warning and continues when Ollama is unreachable so that the
        rest of the system can operate in degraded mode with fallback labels.
        """
        self._client = OllamaClient(
            host=self.config.ollama_host,
            timeout_seconds=self.config.timeout_seconds,
        )
        await self._client.__aenter__()

        try:
            available = await self._client.check_model(self.config.model_name)
            if not available:
                logger.info("vlm_model_not_found_pulling", model=self.config.model_name)
                await self._client.pull_model(self.config.model_name)
        except OllamaConnectionError:
            logger.warning(
                "vlm_ollama_unreachable",
                host=self.config.ollama_host,
                note="Proceeding without VLM; labels will use fallbacks.",
            )
        except OllamaModelError:
            logger.warning(
                "vlm_model_pull_failed",
                model=self.config.model_name,
                note="Proceeding without VLM; labels will use fallbacks.",
            )

        self._initialized = True
        logger.info("vlm_engine_initialized", model=self.config.model_name)

    async def close(self) -> None:
        """Release the Ollama HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def label_region(
        self,
        image_bytes: bytes,
        region_hint: str = "",
    ) -> SemanticLabel:
        """Send an image region to the VLM and parse the semantic response.

        Args:
            image_bytes: JPEG or PNG encoded image of the scene region.
            region_hint: Optional free-form hint about the region context,
                e.g. ``"object on a shelf"`` or ``"near window"``.

        Returns:
            Parsed :class:`SemanticLabel` with physical properties.  Returns a
            low-confidence fallback when the VLM is unavailable or produces
            unparseable output.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called first.
        """
        if not self._initialized:
            raise RuntimeError("VLMEngine not initialized. Call initialize() first.")

        if self._client is None:
            logger.warning("vlm_client_unavailable_using_fallback")
            return _fallback_label()

        prompt = LABEL_REGION_V1.build(region_hint=region_hint or "none")

        try:
            raw = await self._client.generate(
                model=self.config.model_name,
                prompt=prompt,
                image_bytes=image_bytes,
            )
        except Exception as exc:
            logger.warning("vlm_inference_failed", error=str(exc))
            return _fallback_label()

        label = _parse_label_response(raw)
        if label is None:
            logger.warning(
                "vlm_response_parse_failed",
                raw_preview=raw[:200],
                model=self.config.model_name,
            )
            return _fallback_label()

        logger.debug(
            "vlm_region_labeled",
            label=label.label,
            material=label.material,
            confidence=label.confidence,
        )
        return label
