"""VLM provider protocol — the interface every backend must satisfy.

All providers expose three async methods:
  - ``initialize()``  — connect, validate credentials, pull/check model.
  - ``generate()``    — send image + prompt, return raw text response.
  - ``close()``       — release connections and resources.

The protocol is structural (``runtime_checkable``) so duck-typed objects work
without explicit subclassing, which makes test mocking straightforward.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VLMProvider(Protocol):
    """Interface that every VLM backend must implement.

    Providers are async-first. Callers must ``await initialize()`` before
    calling ``generate()``, and ``await close()`` when done.

    Example::

        provider: VLMProvider = get_provider(config)
        await provider.initialize()
        text = await provider.generate(image_bytes, prompt)
        await provider.close()
    """

    async def initialize(self) -> None:
        """Connect to the backend and verify it is ready for inference.

        For Ollama: checks the model is pulled locally.
        For Claude/OpenAI: validates the API key with a lightweight call.

        Must be called before :meth:`generate`. May be called multiple times
        safely (subsequent calls are no-ops if already initialized).
        """
        ...

    async def generate(self, image_bytes: bytes, prompt: str) -> str:
        """Run vision-language inference on *image_bytes* with *prompt*.

        Args:
            image_bytes: JPEG or PNG encoded image of the scene region.
            prompt: Text instruction sent to the model.

        Returns:
            Raw text response from the model (may include chain-of-thought
            before the JSON payload — callers are responsible for parsing).

        Raises:
            RuntimeError: If :meth:`initialize` has not been called.
            Any provider-specific connection or API error.
        """
        ...

    async def close(self) -> None:
        """Release all resources held by this provider.

        Safe to call even if :meth:`initialize` was never called or failed.
        """
        ...
