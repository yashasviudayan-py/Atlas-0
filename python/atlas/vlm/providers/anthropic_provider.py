"""Anthropic (Claude) VLM provider.

Uses the ``anthropic`` Python SDK to call Claude's vision API.
No local GPU or Ollama required — just an API key.

Setup::

    pip install "atlas-0[claude]"   # installs anthropic SDK
    export ANTHROPIC_API_KEY=sk-ant-...
    export ATLAS_VLM_PROVIDER=claude

Configuration (``configs/default.toml`` or ``ATLAS_VLM_*`` env vars)::

    [vlm]
    provider = "claude"
    claude_model = "claude-sonnet-4-6"   # or claude-opus-4-6 for highest quality
    max_tokens = 256
    temperature = 0.1
    timeout_seconds = 30.0

API key is read exclusively from the ``ANTHROPIC_API_KEY`` environment
variable — never from the config file (security rule).
"""

from __future__ import annotations

import base64
import os

import structlog

logger = structlog.get_logger(__name__)

_IMPORT_ERROR_MSG = (
    "The 'anthropic' package is required to use the Claude provider.\n"
    'Install it with:  pip install "atlas-0[claude]"\n'
    "Or directly:      pip install anthropic>=0.40.0"
)


class AnthropicProvider:
    """VLM provider backed by the Anthropic Claude API.

    Args:
        model: Claude model ID (e.g. ``"claude-sonnet-4-6"``).
        max_tokens: Maximum tokens in the model response.
        temperature: Sampling temperature (0.0-1.0).
        timeout_seconds: Per-request timeout passed to the SDK.

    Example::

        provider = AnthropicProvider(model="claude-sonnet-4-6")
        await provider.initialize()
        text = await provider.generate(image_bytes, prompt)
        await provider.close()
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 256,
        temperature: float = 0.1,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout_seconds
        self._client: object | None = None
        self._ready = False

    async def initialize(self) -> None:
        """Validate the Anthropic API key and create the async client.

        Raises:
            ImportError: If the ``anthropic`` package is not installed.
            RuntimeError: If ``ANTHROPIC_API_KEY`` is not set.
        """
        if self._ready:
            return

        try:
            import anthropic  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(_IMPORT_ERROR_MSG) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set.\n"
                "Get a key at https://console.anthropic.com and set:\n"
                "  export ANTHROPIC_API_KEY=sk-ant-..."
            )

        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=self._timeout,
        )
        self._ready = True
        logger.info("anthropic_provider_ready", model=self._model)

    async def generate(self, image_bytes: bytes, prompt: str) -> str:
        """Send image + prompt to Claude and return the response text.

        The image is sent as a base64-encoded ``image`` content block.
        JPEG and PNG are both supported by the Anthropic API.

        Args:
            image_bytes: JPEG or PNG encoded image.
            prompt: Text instruction for the model.

        Returns:
            Raw text content from Claude's response.

        Raises:
            RuntimeError: If :meth:`initialize` was not called.
            ImportError: If the ``anthropic`` package is not installed.
            Any ``anthropic.APIError`` subclass on API failure.
        """
        if not self._ready or self._client is None:
            raise RuntimeError("AnthropicProvider.initialize() must be called first.")

        try:
            import anthropic  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(_IMPORT_ERROR_MSG) from exc

        b64 = base64.b64encode(image_bytes).decode("ascii")
        # Detect media type from magic bytes; default to jpeg.
        media_type = "image/png" if image_bytes[:4] == b"\x89PNG" else "image/jpeg"

        try:
            response = await self._client.messages.create(  # type: ignore[union-attr]
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
        except anthropic.APIError as exc:
            logger.warning(
                "anthropic_provider_api_error",
                model=self._model,
                status=getattr(exc, "status_code", "unknown"),
                error=str(exc),
            )
            return ""

        # Extract text from the first content block.
        blocks = getattr(response, "content", [])
        for block in blocks:
            if getattr(block, "type", "") == "text":
                text: str = block.text
                logger.debug(
                    "anthropic_provider_generate_done",
                    model=self._model,
                    response_len=len(text),
                )
                return text

        logger.warning("anthropic_provider_no_text_block", model=self._model)
        return ""

    async def close(self) -> None:
        """Close the Anthropic async client (releases connection pool)."""
        if self._client is not None:
            import contextlib

            with contextlib.suppress(Exception):
                await self._client.close()  # type: ignore[union-attr]
            self._client = None
        self._ready = False
