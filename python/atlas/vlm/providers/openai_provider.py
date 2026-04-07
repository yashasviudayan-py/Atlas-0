"""OpenAI (GPT-4o) VLM provider.

Uses the ``openai`` Python SDK to call GPT-4o's vision API.
No local GPU or Ollama required — just an API key.

Setup::

    pip install "atlas-0[openai]"   # installs openai SDK
    export OPENAI_API_KEY=sk-...
    export ATLAS_VLM_PROVIDER=openai

Configuration (``configs/default.toml`` or ``ATLAS_VLM_*`` env vars)::

    [vlm]
    provider = "openai"
    openai_model = "gpt-4o"   # or gpt-4o-mini for lower cost
    max_tokens = 256
    temperature = 0.1
    timeout_seconds = 30.0

API key is read exclusively from the ``OPENAI_API_KEY`` environment
variable — never from the config file (security rule).
"""

from __future__ import annotations

import base64
import os

import structlog

logger = structlog.get_logger(__name__)

_IMPORT_ERROR_MSG = (
    "The 'openai' package is required to use the OpenAI provider.\n"
    'Install it with:  pip install "atlas-0[openai]"\n'
    "Or directly:      pip install openai>=1.50.0"
)


class OpenAIProvider:
    """VLM provider backed by the OpenAI GPT-4o vision API.

    Args:
        model: OpenAI model ID (e.g. ``"gpt-4o"`` or ``"gpt-4o-mini"``).
        max_tokens: Maximum tokens in the model response.
        temperature: Sampling temperature (0.0-2.0).
        timeout_seconds: Per-request timeout passed to the SDK.

    Example::

        provider = OpenAIProvider(model="gpt-4o")
        await provider.initialize()
        text = await provider.generate(image_bytes, prompt)
        await provider.close()
    """

    def __init__(
        self,
        model: str = "gpt-4o",
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
        """Validate the OpenAI API key and create the async client.

        Raises:
            ImportError: If the ``openai`` package is not installed.
            RuntimeError: If ``OPENAI_API_KEY`` is not set.
        """
        if self._ready:
            return

        try:
            import openai  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(_IMPORT_ERROR_MSG) from exc

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set.\n"
                "Get a key at https://platform.openai.com and set:\n"
                "  export OPENAI_API_KEY=sk-..."
            )

        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            timeout=self._timeout,
        )
        self._ready = True
        logger.info("openai_provider_ready", model=self._model)

    async def generate(self, image_bytes: bytes, prompt: str) -> str:
        """Send image + prompt to GPT-4o and return the response text.

        The image is sent as a base64 data URL in the ``image_url`` content
        block. JPEG and PNG are both supported.

        Args:
            image_bytes: JPEG or PNG encoded image.
            prompt: Text instruction for the model.

        Returns:
            Raw text content from GPT-4o's response.

        Raises:
            RuntimeError: If :meth:`initialize` was not called.
            ImportError: If the ``openai`` package is not installed.
            Any ``openai.APIError`` subclass on API failure.
        """
        if not self._ready or self._client is None:
            raise RuntimeError("OpenAIProvider.initialize() must be called first.")

        try:
            import openai  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(_IMPORT_ERROR_MSG) from exc

        b64 = base64.b64encode(image_bytes).decode("ascii")
        media_type = "image/png" if image_bytes[:4] == b"\x89PNG" else "image/jpeg"
        data_url = f"data:{media_type};base64,{b64}"

        try:
            response = await self._client.chat.completions.create(  # type: ignore[union-attr]
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url, "detail": "low"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
        except openai.APIError as exc:
            logger.warning(
                "openai_provider_api_error",
                model=self._model,
                status=getattr(exc, "status_code", "unknown"),
                error=str(exc),
            )
            return ""

        choices = getattr(response, "choices", [])
        if choices:
            text: str = choices[0].message.content or ""
            logger.debug(
                "openai_provider_generate_done",
                model=self._model,
                response_len=len(text),
            )
            return text

        logger.warning("openai_provider_no_choices", model=self._model)
        return ""

    async def close(self) -> None:
        """Close the OpenAI async client (releases connection pool)."""
        if self._client is not None:
            import contextlib

            with contextlib.suppress(Exception):
                await self._client.close()  # type: ignore[union-attr]
            self._client = None
        self._ready = False
