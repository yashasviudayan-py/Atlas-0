"""VLM inference engine for labeling 3D scene objects.

Supports multiple VLM backends via a provider abstraction:
  - ``"ollama"``  — local Ollama server (default, free, no API key)
  - ``"claude"``  — Anthropic Claude vision API (best quality)
  - ``"openai"``  — OpenAI GPT-4o vision API

Switch provider via config or environment variable::

    # In configs/default.toml:
    [vlm]
    provider = "claude"
    claude_model = "claude-sonnet-4-6"

    # Or at runtime:
    ATLAS_VLM_PROVIDER=claude ANTHROPIC_API_KEY=sk-ant-... python -m atlas.api.server
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import structlog

from atlas.vlm.prompts import (
    DEFAULT_LABEL_PROPERTIES,
    LABEL_REGION_V1,
    MATERIAL_DEFAULTS,
)
from atlas.vlm.providers import VLMProvider, get_provider

logger = structlog.get_logger(__name__)

# Matches the first JSON object in a string (handles chain-of-thought responses).
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass
class VLMConfig:
    """Configuration for the VLM inference engine.

    The ``provider`` field selects the backend:
      - ``"ollama"``  — local Ollama (default). Uses ``model_name`` + ``ollama_host``.
      - ``"claude"``  — Anthropic Claude. Uses ``claude_model``. Needs ``ANTHROPIC_API_KEY``.
      - ``"openai"``  — OpenAI GPT-4o. Uses ``openai_model``. Needs ``OPENAI_API_KEY``.
    """

    provider: str = "ollama"
    fallback_provider: str | None = None
    model_name: str = "moondream"
    ollama_host: str = "http://localhost:11434"
    claude_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-4o"
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

    Delegates to a :class:`~atlas.vlm.providers.base.VLMProvider` backend
    selected by :attr:`VLMConfig.provider`. Switching providers requires only
    a config change — no code changes.

    Example (Ollama, default)::

        engine = VLMEngine(VLMConfig(provider="ollama", model_name="moondream"))
        await engine.initialize()
        label = await engine.label_region(image_bytes, region_hint="shelf item")
        await engine.close()

    Example (Claude)::

        engine = VLMEngine(VLMConfig(provider="claude"))
        # Reads ANTHROPIC_API_KEY from environment automatically.
        await engine.initialize()
        label = await engine.label_region(image_bytes)
        await engine.close()
    """

    def __init__(self, config: VLMConfig | None = None) -> None:
        self.config = config or VLMConfig()
        self._initialized = False
        self._providers: list[tuple[str, VLMProvider]] = []
        logger.info(
            "vlm_engine_created",
            provider=self.config.provider,
            model=self.config.model_name,
        )

    async def initialize(self) -> None:
        """Initialize the configured VLM provider.

        For Ollama: connects to the local server, pulls model if missing.
        For Claude/OpenAI: validates the API key.

        Logs a warning and continues in degraded mode (fallback labels) when
        initialization fails, so the rest of the system keeps running.
        """
        self._providers = []
        provider_names = [self.config.provider]
        if self.config.fallback_provider and self.config.fallback_provider != self.config.provider:
            provider_names.append(self.config.fallback_provider)

        for provider_name in provider_names:
            provider = get_provider(self.config, provider_name=provider_name)
            try:
                await provider.initialize()
            except (ImportError, RuntimeError) as exc:
                logger.warning(
                    "vlm_provider_init_failed",
                    provider=provider_name,
                    error=str(exc),
                    note="Proceeding with degraded routing for this provider slot.",
                )
            self._providers.append((provider_name, provider))

        self._initialized = True
        logger.info(
            "vlm_engine_initialized",
            provider=self.config.provider,
            model=self.config.model_name,
            fallback_provider=self.config.fallback_provider,
        )

    async def close(self) -> None:
        """Release resources held by the active provider."""
        for _provider_name, provider in self._providers:
            await provider.close()
        self._providers = []
        self._initialized = False

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
            low-confidence fallback when the provider is unavailable or
            produces unparseable output.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called first.
        """
        if not self._initialized:
            raise RuntimeError("VLMEngine not initialized. Call initialize() first.")

        if not self._providers:
            logger.warning("vlm_provider_unavailable_using_fallback")
            return _fallback_label()

        prompt = LABEL_REGION_V1.build(region_hint=region_hint or "none")

        for provider_name, provider in self._providers:
            try:
                raw = await provider.generate(image_bytes, prompt)
            except Exception as exc:
                logger.warning(
                    "vlm_inference_failed",
                    provider=provider_name,
                    error=str(exc),
                )
                continue

            label = _parse_label_response(raw)
            if label is None:
                logger.warning(
                    "vlm_response_parse_failed",
                    raw_preview=raw[:200],
                    provider=provider_name,
                )
                continue

            logger.debug(
                "vlm_region_labeled",
                label=label.label,
                material=label.material,
                confidence=label.confidence,
                provider=provider_name,
            )
            return label

        logger.warning("vlm_all_providers_failed_using_fallback")
        return _fallback_label()
