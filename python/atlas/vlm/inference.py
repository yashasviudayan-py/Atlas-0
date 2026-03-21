"""VLM inference engine for labeling 3D scene objects.

Runs a local Vision-Language Model (e.g., Moondream-3 via Ollama)
to assign semantic metadata to Gaussian map shards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class VLMConfig:
    """Configuration for the VLM inference engine."""

    model_name: str = "moondream"
    ollama_host: str = "http://localhost:11434"
    max_tokens: int = 256
    temperature: float = 0.1
    timeout_seconds: float = 10.0


@dataclass
class SemanticLabel:
    """Semantic label assigned to a scene region."""

    label: str
    material: str
    mass_kg: float
    fragility: float
    friction: float
    confidence: float


class VLMEngine:
    """Runs VLM inference to generate semantic labels for scene objects.

    Uses a local Ollama instance to query a Vision-Language Model
    about regions of the 3D Gaussian map.
    """

    def __init__(self, config: VLMConfig | None = None) -> None:
        self.config = config or VLMConfig()
        self._initialized = False
        logger.info("vlm_engine_created", model=self.config.model_name)

    async def initialize(self) -> None:
        """Connect to Ollama and verify model availability."""
        # TODO(phase-2): Implement Ollama connection check
        # 1. GET /api/tags to verify model exists
        # 2. Pull model if not present
        self._initialized = True
        logger.info("vlm_engine_initialized")

    async def label_region(self, image_bytes: bytes, prompt: str) -> SemanticLabel:
        """Send an image region to the VLM and parse the semantic response.

        Args:
            image_bytes: JPEG/PNG encoded image of the scene region.
            prompt: Structured prompt asking for material properties.

        Returns:
            Parsed semantic label with physical properties.
        """
        if not self._initialized:
            raise RuntimeError("VLMEngine not initialized. Call initialize() first.")

        # TODO(phase-2): Implement actual VLM inference
        # 1. Encode image to base64
        # 2. POST to Ollama /api/generate with image
        # 3. Parse structured response into SemanticLabel
        raise NotImplementedError("VLM inference not yet implemented")
