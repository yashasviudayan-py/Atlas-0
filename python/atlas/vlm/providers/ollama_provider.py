"""Ollama VLM provider — wraps the existing OllamaClient.

This is the default provider (free, local, no API key required).
Requires a running Ollama server and a pulled vision model (e.g. moondream,
llava, bakllava).

Configuration (via ``configs/default.toml`` or ``ATLAS_VLM_*`` env vars)::

    [vlm]
    provider = "ollama"
    model_name = "moondream"
    ollama_host = "http://localhost:11434"
"""

from __future__ import annotations

import structlog

from atlas.vlm.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaModelError,
)

logger = structlog.get_logger(__name__)


class OllamaProvider:
    """VLM provider backed by a local Ollama server.

    Args:
        model_name: The Ollama model tag to use (e.g. ``"moondream"``).
        host: Base URL of the Ollama server.
        timeout_seconds: Per-request timeout.

    Example::

        provider = OllamaProvider(model_name="moondream")
        await provider.initialize()
        text = await provider.generate(image_bytes, prompt)
        await provider.close()
    """

    def __init__(
        self,
        model_name: str = "moondream",
        host: str = "http://localhost:11434",
        timeout_seconds: float = 30.0,
    ) -> None:
        self._model = model_name
        self._host = host
        self._timeout = timeout_seconds
        self._client: OllamaClient | None = None
        self._ready = False

    async def initialize(self) -> None:
        """Connect to Ollama and verify the model is available.

        Logs a warning and continues in degraded mode when Ollama is
        unreachable, so the rest of the system keeps running with fallback
        labels.
        """
        if self._ready:
            return

        self._client = OllamaClient(host=self._host, timeout_seconds=self._timeout)
        await self._client.__aenter__()

        try:
            available = await self._client.check_model(self._model)
            if not available:
                logger.info("ollama_provider_pulling_model", model=self._model)
                await self._client.pull_model(self._model)
            logger.info("ollama_provider_ready", model=self._model, host=self._host)
        except OllamaConnectionError:
            logger.warning(
                "ollama_provider_unreachable",
                host=self._host,
                note="Labels will use fallbacks until Ollama is available.",
            )
            # Don't set _ready=False — we'll attempt generation anyway and
            # let generate() return an empty string on failure.
        except OllamaModelError:
            logger.warning(
                "ollama_provider_model_pull_failed",
                model=self._model,
            )

        self._ready = True

    async def generate(self, image_bytes: bytes, prompt: str) -> str:
        """Send image + prompt to Ollama and return the raw response text.

        Args:
            image_bytes: JPEG or PNG encoded image.
            prompt: Text instruction for the model.

        Returns:
            Raw text from Ollama; empty string on connection failure.

        Raises:
            RuntimeError: If :meth:`initialize` was not called.
        """
        if not self._ready:
            raise RuntimeError("OllamaProvider.initialize() must be called first.")
        if self._client is None:
            logger.warning("ollama_provider_client_none_returning_empty")
            return ""

        try:
            return await self._client.generate(
                model=self._model,
                prompt=prompt,
                image_bytes=image_bytes,
            )
        except Exception as exc:
            logger.warning("ollama_provider_generate_failed", error=str(exc))
            return ""

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._ready = False
