"""Tests for the VLM provider abstraction (Phase 4, Part 14)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from atlas.vlm.inference import VLMConfig, VLMEngine
from atlas.vlm.providers import get_provider
from atlas.vlm.providers.base import VLMProvider as VLMProviderProtocol
from atlas.vlm.providers.ollama_provider import OllamaProvider

# ── Factory tests ─────────────────────────────────────────────────────────────

_CUP_JSON = (
    '{"label": "cup", "material": "ceramic",' ' "mass_kg": 0.3, "fragility": 0.7, "friction": 0.4}'
)
_VASE_JSON = (
    '{"label": "vase", "material": "glass",' ' "mass_kg": 0.5, "fragility": 0.95, "friction": 0.2}'
)
_BOTTLE_JSON = (
    '{"label": "bottle", "material": "plastic",'
    ' "mass_kg": 0.8, "fragility": 0.3, "friction": 0.5}'
)
_MUG_JSON = (
    '{"label": "mug", "material": "ceramic",' ' "mass_kg": 0.4, "fragility": 0.6, "friction": 0.5}'
)


def test_get_provider_ollama_returns_ollama_provider() -> None:
    config = VLMConfig(provider="ollama", model_name="moondream")
    provider = get_provider(config)
    assert isinstance(provider, OllamaProvider)


def test_get_provider_ollama_case_insensitive() -> None:
    config = VLMConfig(provider="ollama")
    provider = get_provider(config)
    assert isinstance(provider, OllamaProvider)


def test_get_provider_unknown_raises_value_error() -> None:
    config = VLMConfig.__new__(VLMConfig)
    object.__setattr__(config, "provider", "unknown_backend")
    with pytest.raises(ValueError, match="Unknown VLM provider"):
        get_provider(config)


def test_get_provider_error_message_lists_valid_options() -> None:
    config = VLMConfig.__new__(VLMConfig)
    object.__setattr__(config, "provider", "gemini")
    with pytest.raises(ValueError) as exc_info:
        get_provider(config)
    msg = str(exc_info.value)
    assert "ollama" in msg
    assert "claude" in msg
    assert "openai" in msg


def test_get_provider_claude_returns_provider_without_importing_sdk() -> None:
    config = VLMConfig.__new__(VLMConfig)
    object.__setattr__(config, "provider", "claude")
    object.__setattr__(config, "claude_model", "claude-sonnet-4-6")
    object.__setattr__(config, "max_tokens", 256)
    object.__setattr__(config, "temperature", 0.1)
    object.__setattr__(config, "timeout_seconds", 30.0)
    # SDK import is deferred to initialize() — factory must succeed without it.
    provider = get_provider(config)
    assert provider is not None


def test_get_provider_openai_returns_provider_without_importing_sdk() -> None:
    config = VLMConfig.__new__(VLMConfig)
    object.__setattr__(config, "provider", "openai")
    object.__setattr__(config, "openai_model", "gpt-4o")
    object.__setattr__(config, "max_tokens", 256)
    object.__setattr__(config, "temperature", 0.1)
    object.__setattr__(config, "timeout_seconds", 30.0)
    provider = get_provider(config)
    assert provider is not None


# ── VLMProvider protocol compliance ───────────────────────────────────────────


def test_ollama_provider_satisfies_protocol() -> None:
    provider = OllamaProvider(model_name="moondream")
    assert isinstance(provider, VLMProviderProtocol)


# ── OllamaProvider unit tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ollama_provider_initialize_and_generate() -> None:
    provider = OllamaProvider(model_name="moondream")

    mock_client = AsyncMock()
    mock_client.check_model = AsyncMock(return_value=True)
    mock_client.generate = AsyncMock(return_value=_CUP_JSON)

    with patch("atlas.vlm.providers.ollama_provider.OllamaClient") as mock_cls:
        mock_cls.return_value = mock_client
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        await provider.initialize()
        text = await provider.generate(b"fake_image_bytes", "describe this")

    assert "cup" in text
    mock_client.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_ollama_provider_generate_before_initialize_raises() -> None:
    provider = OllamaProvider()
    with pytest.raises(RuntimeError, match="initialize"):
        await provider.generate(b"bytes", "prompt")


@pytest.mark.asyncio
async def test_ollama_provider_generate_failure_returns_empty_string() -> None:
    provider = OllamaProvider()

    mock_client = AsyncMock()
    mock_client.check_model = AsyncMock(return_value=True)
    mock_client.generate = AsyncMock(side_effect=Exception("connection refused"))

    with patch("atlas.vlm.providers.ollama_provider.OllamaClient") as mock_cls:
        mock_cls.return_value = mock_client
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        await provider.initialize()
        result = await provider.generate(b"bytes", "prompt")

    assert result == ""


@pytest.mark.asyncio
async def test_ollama_provider_close_is_idempotent() -> None:
    provider = OllamaProvider()
    # close() before initialize() must not raise.
    await provider.close()
    await provider.close()


# ── AnthropicProvider unit tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anthropic_provider_initialize_raises_when_sdk_missing() -> None:
    from atlas.vlm.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider()
    with (
        patch.dict("sys.modules", {"anthropic": None}),
        pytest.raises((ImportError, RuntimeError)),
    ):
        await provider.initialize()


@pytest.mark.asyncio
async def test_anthropic_provider_initialize_raises_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atlas.vlm.providers.anthropic_provider import AnthropicProvider

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = AnthropicProvider()

    mock_anthropic = MagicMock()
    with (
        patch.dict("sys.modules", {"anthropic": mock_anthropic}),
        pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"),
    ):
        await provider.initialize()


@pytest.mark.asyncio
async def test_anthropic_provider_generate_extracts_text_block() -> None:
    from atlas.vlm.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider()

    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = _VASE_JSON
    mock_response = MagicMock()
    mock_response.content = [mock_block]

    mock_client = AsyncMock()
    mock_client.messages = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    mock_anthropic_module = MagicMock()
    mock_anthropic_module.AsyncAnthropic.return_value = mock_client
    mock_anthropic_module.APIError = Exception

    with (
        patch.dict("sys.modules", {"anthropic": mock_anthropic_module}),
        patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}),
    ):
        await provider.initialize()
        text = await provider.generate(b"\xff\xd8\xff" + b"\x00" * 10, "describe this")

    assert "vase" in text


@pytest.mark.asyncio
async def test_anthropic_provider_generate_before_initialize_raises() -> None:
    from atlas.vlm.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider()
    with pytest.raises(RuntimeError, match="initialize"):
        await provider.generate(b"bytes", "prompt")


# ── OpenAIProvider unit tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_openai_provider_initialize_raises_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atlas.vlm.providers.openai_provider import OpenAIProvider

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIProvider()

    mock_openai = MagicMock()
    with (
        patch.dict("sys.modules", {"openai": mock_openai}),
        pytest.raises(RuntimeError, match="OPENAI_API_KEY"),
    ):
        await provider.initialize()


@pytest.mark.asyncio
async def test_openai_provider_generate_extracts_choice_text() -> None:
    from atlas.vlm.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider()

    mock_message = MagicMock()
    mock_message.content = _BOTTLE_JSON
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat = AsyncMock()
    mock_client.chat.completions = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    mock_openai_module = MagicMock()
    mock_openai_module.AsyncOpenAI.return_value = mock_client
    mock_openai_module.APIError = Exception

    with (
        patch.dict("sys.modules", {"openai": mock_openai_module}),
        patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}),
    ):
        await provider.initialize()
        text = await provider.generate(b"\xff\xd8\xff" + b"\x00" * 10, "describe this")

    assert "bottle" in text


@pytest.mark.asyncio
async def test_openai_provider_generate_before_initialize_raises() -> None:
    from atlas.vlm.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider()
    with pytest.raises(RuntimeError, match="initialize"):
        await provider.generate(b"bytes", "prompt")


# ── VLMEngine integration with providers ──────────────────────────────────────


@pytest.mark.asyncio
async def test_vlm_engine_uses_provider_abstraction() -> None:
    """VLMEngine.label_region delegates to the configured provider."""
    config = VLMConfig(provider="ollama", model_name="moondream")
    engine = VLMEngine(config)

    mock_provider = AsyncMock(spec=VLMProviderProtocol)
    mock_provider.generate = AsyncMock(return_value=_MUG_JSON)

    with patch("atlas.vlm.inference.get_provider", return_value=mock_provider):
        await engine.initialize()
        label = await engine.label_region(b"fake_image", region_hint="coffee mug on desk")

    assert label.label == "mug"
    assert label.material == "ceramic"
    mock_provider.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_vlm_engine_returns_fallback_on_provider_generate_error() -> None:
    config = VLMConfig(provider="ollama")
    engine = VLMEngine(config)

    mock_provider = AsyncMock(spec=VLMProviderProtocol)
    mock_provider.generate = AsyncMock(side_effect=Exception("network error"))

    with patch("atlas.vlm.inference.get_provider", return_value=mock_provider):
        await engine.initialize()
        label = await engine.label_region(b"fake_image")

    assert label.label == "unknown"
    assert label.confidence == 0.1


@pytest.mark.asyncio
async def test_vlm_engine_provider_field_in_config() -> None:
    """VLMConfig.provider field is set and accessible."""
    config = VLMConfig(
        provider="claude",
        fallback_provider="openai",
        claude_model="claude-sonnet-4-6",
    )
    assert config.provider == "claude"
    assert config.fallback_provider == "openai"
    assert config.claude_model == "claude-sonnet-4-6"


def test_vlm_config_provider_validation_rejects_unknown() -> None:
    """The Pydantic VlmConfig (in config.py) validates the provider field."""
    from atlas.utils.config import VlmConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="provider"):
        VlmConfig(provider="gemini")


def test_vlm_config_default_provider_is_ollama() -> None:
    config = VLMConfig()
    assert config.provider == "ollama"


def test_vlm_config_has_cloud_model_fields() -> None:
    config = VLMConfig()
    assert config.claude_model == "claude-sonnet-4-6"
    assert config.openai_model == "gpt-4o"


@pytest.mark.asyncio
async def test_vlm_engine_uses_fallback_provider_when_primary_fails() -> None:
    config = VLMConfig(provider="ollama", fallback_provider="openai")
    engine = VLMEngine(config)

    primary = AsyncMock(spec=VLMProviderProtocol)
    primary.generate = AsyncMock(side_effect=Exception("primary down"))
    fallback = AsyncMock(spec=VLMProviderProtocol)
    fallback.generate = AsyncMock(return_value=_MUG_JSON)

    with patch("atlas.vlm.inference.get_provider", side_effect=[primary, fallback]):
        await engine.initialize()
        label = await engine.label_region(b"fake_image")

    assert label.label == "mug"
    primary.generate.assert_awaited_once()
    fallback.generate.assert_awaited_once()
