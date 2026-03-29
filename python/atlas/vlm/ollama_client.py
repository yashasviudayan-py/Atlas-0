"""Async Ollama HTTP client for Atlas-0 VLM integration.

Wraps the Ollama REST API with async httpx, robust error handling,
and structured logging.
"""

from __future__ import annotations

import base64
import json

import httpx
import structlog

logger = structlog.get_logger(__name__)

_GENERATE_PATH = "/api/generate"
_TAGS_PATH = "/api/tags"
_PULL_PATH = "/api/pull"


class OllamaConnectionError(Exception):
    """Raised when the Ollama server cannot be reached."""


class OllamaModelError(Exception):
    """Raised for model-related errors (not found, pull failed)."""


class OllamaResponseError(Exception):
    """Raised when the Ollama response is malformed or unexpected."""


class OllamaClient:
    """Async HTTP client for the Ollama REST API.

    Must be used as an async context manager to ensure the underlying
    ``httpx.AsyncClient`` is properly closed.

    Args:
        host: Base URL of the Ollama server, e.g. ``"http://localhost:11434"``.
        timeout_seconds: Per-request timeout in seconds.

    Example::

        async with OllamaClient("http://localhost:11434") as client:
            available = await client.check_model("moondream")
            if not available:
                await client.pull_model("moondream")
            text = await client.generate(
                model="moondream",
                prompt="Describe this object.",
                image_bytes=jpeg_bytes,
            )
    """

    def __init__(self, host: str, timeout_seconds: float = 30.0) -> None:
        self._host = host.rstrip("/")
        self._timeout = timeout_seconds
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> OllamaClient:
        self._client = httpx.AsyncClient(base_url=self._host, timeout=self._timeout)
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "OllamaClient must be used as an async context manager "
                "(async with OllamaClient(...) as client:)"
            )
        return self._client

    async def check_model(self, name: str) -> bool:
        """Return ``True`` if *name* is present in the local Ollama registry.

        Args:
            name: Model name, e.g. ``"moondream"``.  The tag suffix (e.g.
                ``:latest``) is stripped for comparison.

        Returns:
            ``True`` if the model is available locally.

        Raises:
            OllamaConnectionError: If the Ollama server is unreachable.
        """
        client = self._get_client()
        try:
            response = await client.get(_TAGS_PATH)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(f"Cannot connect to Ollama at {self._host}") from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaConnectionError(
                f"Ollama /api/tags returned HTTP {exc.response.status_code}"
            ) from exc

        data = response.json()
        base_name = name.split(":")[0]
        present = any(m.get("name", "").split(":")[0] == base_name for m in data.get("models", []))
        logger.debug("ollama_check_model", model=name, present=present)
        return present

    async def pull_model(self, name: str) -> None:
        """Pull *name* from the Ollama registry.

        Streams pull progress until completion.

        Args:
            name: Model name to pull.

        Raises:
            OllamaConnectionError: If the Ollama server is unreachable.
            OllamaModelError: If the pull reports an error.
        """
        client = self._get_client()
        logger.info("ollama_pulling_model", model=name)
        try:
            async with client.stream(
                "POST", _PULL_PATH, json={"name": name, "stream": True}
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("error"):
                        raise OllamaModelError(f"Pull failed for {name!r}: {chunk['error']}")
                    status = chunk.get("status", "")
                    if status:
                        logger.debug("ollama_pull_progress", model=name, status=status)
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(f"Cannot connect to Ollama at {self._host}") from exc
        logger.info("ollama_model_ready", model=name)

    async def generate(
        self,
        model: str,
        prompt: str,
        image_bytes: bytes | None = None,
    ) -> str:
        """Send a generation request and return the full response text.

        Args:
            model: Model name to use for inference.
            prompt: Text prompt sent to the model.
            image_bytes: Optional JPEG or PNG image bytes.  When provided, the
                image is base64-encoded and attached as a vision input.

        Returns:
            The response string from Ollama (``response`` field of the JSON
            reply).

        Raises:
            OllamaConnectionError: If the server is not reachable.
            OllamaResponseError: If the response is malformed or times out.
        """
        client = self._get_client()

        payload: dict[str, object] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        if image_bytes is not None:
            payload["images"] = [base64.b64encode(image_bytes).decode("ascii")]

        try:
            response = await client.post(_GENERATE_PATH, json=payload)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(f"Cannot connect to Ollama at {self._host}") from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaResponseError(
                f"Ollama returned HTTP {exc.response.status_code}: " f"{exc.response.text[:200]}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OllamaResponseError(f"Ollama request timed out: {exc}") from exc

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise OllamaResponseError(
                f"Non-JSON response from Ollama: {response.text[:200]}"
            ) from exc

        text: str = data.get("response", "")
        logger.debug(
            "ollama_generate_done",
            model=model,
            prompt_len=len(prompt),
            response_len=len(text),
        )
        return text
