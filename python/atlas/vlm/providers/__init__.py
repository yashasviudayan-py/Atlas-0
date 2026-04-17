"""VLM provider factory and public exports.

This module is the single entry point for creating a :class:`VLMProvider`
from a :class:`~atlas.vlm.inference.VLMConfig`. Callers never import the
individual provider modules directly — they call :func:`get_provider` and
work against the :class:`VLMProvider` protocol.

Supported providers
-------------------
``"ollama"`` (default)
    Local Ollama server. Free, no API key. Requires Ollama running locally
    with a vision model pulled (e.g. ``moondream``, ``llava``).

``"claude"``
    Anthropic Claude vision API (``claude-sonnet-4-6`` by default).
    Requires ``pip install "atlas-0[claude]"`` and ``ANTHROPIC_API_KEY`` env var.
    Best label quality.

``"openai"``
    OpenAI GPT-4o vision API. Requires ``pip install "atlas-0[openai]"``
    and ``OPENAI_API_KEY`` env var.

Quick-switch via environment::

    # Use Claude
    ATLAS_VLM_PROVIDER=claude ANTHROPIC_API_KEY=sk-ant-... python -m atlas.api.server

    # Use OpenAI
    ATLAS_VLM_PROVIDER=openai OPENAI_API_KEY=sk-... python -m atlas.api.server

    # Use Ollama (default, no env vars needed)
    python -m atlas.api.server
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from atlas.vlm.providers.base import VLMProvider

if TYPE_CHECKING:
    from atlas.vlm.inference import VLMConfig

__all__ = ["VLMProvider", "get_provider"]

_KNOWN_PROVIDERS = ("ollama", "claude", "openai")


def get_provider(config: VLMConfig, *, provider_name: str | None = None) -> VLMProvider:
    """Create and return a :class:`VLMProvider` for the given config.

    The provider is *not* initialized here — callers must ``await
    provider.initialize()`` before calling ``generate()``.

    Args:
        config: VLM configuration object. The ``provider`` field selects the
            backend; other fields (model name, timeout, etc.) are forwarded
            to the appropriate provider.

    Returns:
        An uninitialized :class:`VLMProvider` instance.

    Raises:
        ValueError: If ``config.provider`` is not one of the known providers.
        ImportError: If the required SDK for the selected provider is not
            installed (raised lazily — only when the provider is actually used).

    Example::

        config = VLMConfig(provider="claude", claude_model="claude-sonnet-4-6")
        provider = get_provider(config)
        await provider.initialize()
        text = await provider.generate(image_bytes, prompt)
        await provider.close()
    """
    provider_name = (provider_name or config.provider).lower().strip()

    if provider_name == "ollama":
        from atlas.vlm.providers.ollama_provider import OllamaProvider

        return OllamaProvider(
            model_name=config.model_name,
            host=config.ollama_host,
            timeout_seconds=config.timeout_seconds,
        )

    if provider_name == "claude":
        from atlas.vlm.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            model=config.claude_model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout_seconds=config.timeout_seconds,
        )

    if provider_name == "openai":
        from atlas.vlm.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(
            model=config.openai_model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout_seconds=config.timeout_seconds,
        )

    raise ValueError(
        f"Unknown VLM provider: {config.provider!r}. "
        f"Valid options: {', '.join(_KNOWN_PROVIDERS)}.\n"
        f"Set via config: vlm.provider = \"ollama\"  (or claude / openai)\n"
        f"Set via env:    ATLAS_VLM_PROVIDER=claude"
    )
