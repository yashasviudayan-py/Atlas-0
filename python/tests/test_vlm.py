"""Unit tests for Phase-2 Part-6: VLM Integration & Semantic Labeling."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from atlas.vlm.inference import (
    SemanticLabel,
    VLMConfig,
    VLMEngine,
    _fallback_label,
    _parse_label_response,
)
from atlas.vlm.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
)
from atlas.vlm.prompts import (
    LABEL_REGION_V1,
    LABEL_REGION_V2,
    MATERIAL_DEFAULTS,
)
from atlas.vlm.region_extractor import BoundingBox, RegionExtractor
from atlas.world_model.label_store import LabelStore

# ── Prompt tests ──────────────────────────────────────────────────────────────


def test_prompt_template_build_inserts_hint() -> None:
    prompt = LABEL_REGION_V1.build(region_hint="shelf item")
    assert "shelf item" in prompt


def test_prompt_template_build_empty_hint() -> None:
    prompt = LABEL_REGION_V1.build(region_hint="")
    assert "{region_hint}" not in prompt


def test_label_region_v1_contains_json_instruction() -> None:
    prompt = LABEL_REGION_V1.build(region_hint="test")
    assert "JSON" in prompt
    assert "label" in prompt
    assert "material" in prompt
    assert "mass_kg" in prompt
    assert "fragility" in prompt
    assert "friction" in prompt


def test_label_region_v2_contains_chain_of_thought() -> None:
    prompt = LABEL_REGION_V2.build(region_hint="test")
    assert "step by step" in prompt


def test_prompt_template_frozen() -> None:
    with pytest.raises((AttributeError, TypeError)):
        LABEL_REGION_V1.version = "v99"  # type: ignore[misc]


def test_material_defaults_coverage() -> None:
    for material, (mass, frag, fric) in MATERIAL_DEFAULTS.items():
        assert mass > 0, f"{material}: mass must be positive"
        assert 0.0 <= frag <= 1.0, f"{material}: fragility out of range"
        assert 0.0 <= fric <= 1.0, f"{material}: friction out of range"


# ── _parse_label_response tests ───────────────────────────────────────────────


def test_parse_valid_json_response() -> None:
    text = (
        '{"label": "coffee mug", "material": "ceramic", '
        '"mass_kg": 0.35, "fragility": 0.6, "friction": 0.45}'
    )
    result = _parse_label_response(text)
    assert result is not None
    assert result.label == "coffee mug"
    assert result.material == "ceramic"
    assert result.mass_kg == pytest.approx(0.35)
    assert result.fragility == pytest.approx(0.6)
    assert result.friction == pytest.approx(0.45)
    assert result.confidence == pytest.approx(0.8)


def test_parse_embedded_json_with_prose() -> None:
    text = (
        "Looking at the object, I can see it is a glass bottle.\n"
        'The answer is: {"label": "bottle", "material": "glass", '
        '"mass_kg": 0.5, "fragility": 0.85, "friction": 0.25}'
    )
    result = _parse_label_response(text)
    assert result is not None
    assert result.label == "bottle"
    assert result.material == "glass"


def test_parse_chain_of_thought_uses_last_json() -> None:
    # v2-style: prose first, JSON on the last line.
    text = (
        '1. It is a {"label": "fake"}\n'
        "Actually let me reconsider.\n"
        '{"label": "laptop", "material": "plastic", "mass_kg": 1.8, '
        '"fragility": 0.4, "friction": 0.35}'
    )
    result = _parse_label_response(text)
    assert result is not None
    assert result.label == "laptop"


def test_parse_uses_material_defaults_for_missing_numeric_fields() -> None:
    text = '{"label": "plank", "material": "wood"}'
    result = _parse_label_response(text)
    assert result is not None
    expected_mass, expected_frag, expected_fric = MATERIAL_DEFAULTS["wood"]
    assert result.mass_kg == pytest.approx(expected_mass)
    assert result.fragility == pytest.approx(expected_frag)
    assert result.friction == pytest.approx(expected_fric)


def test_parse_clamps_fragility_above_one() -> None:
    text = '{"label": "egg", "material": "ceramic", "mass_kg": 0.06, "fragility": 5.0, "friction": 0.4}'  # noqa: E501
    result = _parse_label_response(text)
    assert result is not None
    assert result.fragility == pytest.approx(1.0)


def test_parse_clamps_fragility_below_zero() -> None:
    text = (
        '{"label": "rock", "material": "stone", "mass_kg": 2.0, "fragility": -0.5, "friction": 0.6}'
    )
    result = _parse_label_response(text)
    assert result is not None
    assert result.fragility == pytest.approx(0.0)


def test_parse_clamps_mass_below_minimum() -> None:
    text = '{"label": "dust", "material": "unknown", "mass_kg": 0.0, "fragility": 0.5, "friction": 0.5}'  # noqa: E501
    result = _parse_label_response(text)
    assert result is not None
    assert result.mass_kg >= 0.001


def test_parse_returns_none_for_empty_string() -> None:
    assert _parse_label_response("") is None


def test_parse_returns_none_for_plain_text() -> None:
    assert _parse_label_response("This is a chair made of wood.") is None


def test_parse_returns_none_for_bad_json() -> None:
    assert _parse_label_response("{label: glass}") is None


def test_fallback_label_has_low_confidence() -> None:
    label = _fallback_label()
    assert label.confidence == pytest.approx(0.1)
    assert label.label == "unknown"


# ── VLMEngine tests ───────────────────────────────────────────────────────────


def test_vlm_engine_raises_if_not_initialized() -> None:
    engine = VLMEngine()
    with pytest.raises(RuntimeError, match="not initialized"):
        import asyncio

        asyncio.get_event_loop().run_until_complete(engine.label_region(b"fake"))


@pytest.mark.asyncio
async def test_vlm_engine_label_region_returns_fallback_on_connection_error() -> None:
    engine = VLMEngine(VLMConfig(ollama_host="http://localhost:11434"))

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(side_effect=OllamaConnectionError("no server"))

    with patch("atlas.vlm.inference.get_provider", return_value=mock_provider):
        await engine.initialize()

    label = await engine.label_region(b"fake_image_bytes")
    assert label.confidence == pytest.approx(0.1)
    assert label.label == "unknown"


@pytest.mark.asyncio
async def test_vlm_engine_label_region_parses_valid_response() -> None:
    engine = VLMEngine()
    valid_response = json.dumps(
        {"label": "vase", "material": "ceramic", "mass_kg": 0.4, "fragility": 0.8, "friction": 0.35}
    )

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=valid_response)

    with patch("atlas.vlm.inference.get_provider", return_value=mock_provider):
        await engine.initialize()

    label = await engine.label_region(b"fake_jpeg", region_hint="on table")
    assert label.label == "vase"
    assert label.material == "ceramic"
    assert label.confidence == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_vlm_engine_returns_fallback_on_parse_failure() -> None:
    engine = VLMEngine()

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value="I cannot determine what this is.")

    with patch("atlas.vlm.inference.get_provider", return_value=mock_provider):
        await engine.initialize()

    label = await engine.label_region(b"fake_jpeg")
    assert label.label == "unknown"
    assert label.confidence == pytest.approx(0.1)


# ── LabelStore tests ──────────────────────────────────────────────────────────


def _make_label(label: str = "chair", confidence: float = 0.7) -> SemanticLabel:
    return SemanticLabel(
        label=label,
        material="wood",
        mass_kg=5.0,
        fragility=0.2,
        friction=0.6,
        confidence=confidence,
    )


def test_label_store_insert_new_entry() -> None:
    store = LabelStore()
    accepted = store.update(1, _make_label("chair", 0.7))
    assert accepted is True
    assert len(store) == 1
    entry = store.get(1)
    assert entry is not None
    assert entry.label.label == "chair"
    assert entry.update_count == 1


def test_label_store_higher_confidence_replaces() -> None:
    store = LabelStore()
    store.update(1, _make_label("chair", 0.5))
    accepted = store.update(1, _make_label("stool", 0.9))
    assert accepted is True
    assert store.get(1).label.label == "stool"  # type: ignore[union-attr]


def test_label_store_lower_confidence_rejected() -> None:
    store = LabelStore()
    store.update(1, _make_label("chair", 0.8))
    accepted = store.update(1, _make_label("bench", 0.5))
    assert accepted is False
    assert store.get(1).label.label == "chair"  # type: ignore[union-attr]


def test_label_store_equal_confidence_rejected() -> None:
    store = LabelStore()
    store.update(1, _make_label("chair", 0.7))
    accepted = store.update(1, _make_label("stool", 0.7))
    assert accepted is False
    assert store.get(1).label.label == "chair"  # type: ignore[union-attr]


def test_label_store_update_count_increments() -> None:
    store = LabelStore()
    store.update(5, _make_label("lamp", 0.4))
    store.update(5, _make_label("light", 0.9))
    entry = store.get(5)
    assert entry is not None
    assert entry.update_count == 2


def test_label_store_get_missing_returns_none() -> None:
    store = LabelStore()
    assert store.get(999) is None


def test_label_store_remove_existing() -> None:
    store = LabelStore()
    store.update(3, _make_label())
    removed = store.remove(3)
    assert removed is True
    assert store.get(3) is None
    assert len(store) == 0


def test_label_store_remove_missing_returns_false() -> None:
    store = LabelStore()
    assert store.remove(99) is False


def test_label_store_clear() -> None:
    store = LabelStore()
    store.update(1, _make_label())
    store.update(2, _make_label())
    store.clear()
    assert len(store) == 0
    assert store.all_entries() == []


def test_label_store_all_entries_snapshot() -> None:
    store = LabelStore()
    store.update(10, _make_label("cup", 0.6))
    store.update(20, _make_label("mug", 0.7))
    entries = store.all_entries()
    ids = {e.object_id for e in entries}
    assert ids == {10, 20}


def test_label_store_to_proto_list_returns_list() -> None:
    store = LabelStore()
    store.update(1, _make_label("cup", 0.8))
    # Should return a list (possibly empty if protobuf not wired, but never raises).
    result = store.to_proto_list()
    assert isinstance(result, list)


# ── RegionExtractor tests ─────────────────────────────────────────────────────

_GAUSSIAN_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("opacity", "<f4"),
        ("r", "<f4"),
        ("g", "<f4"),
        ("b", "<f4"),
    ]
)


def _make_snapshot(
    positions: list[tuple[float, float, float]],
    opacity: float = 0.5,
    color: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> MagicMock:
    from atlas.utils.shared_mem import CameraPose, MapSnapshot

    n = len(positions)
    gaussians = np.zeros(n, dtype=_GAUSSIAN_DTYPE)
    for i, (x, y, z) in enumerate(positions):
        gaussians[i]["x"] = x
        gaussians[i]["y"] = y
        gaussians[i]["z"] = z
        gaussians[i]["opacity"] = opacity
        gaussians[i]["r"] = color[0]
        gaussians[i]["g"] = color[1]
        gaussians[i]["b"] = color[2]

    return MapSnapshot(
        frame_id=1,
        timestamp_ns=0,
        pose=CameraPose(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        gaussians=gaussians,
    )


def test_region_extractor_empty_snapshot() -> None:
    snap = _make_snapshot([])
    extractor = RegionExtractor(min_samples=2)
    regions = extractor.extract_regions(snap)
    assert regions == []


def test_region_extractor_all_transparent() -> None:
    snap = _make_snapshot([(0.0, 0.0, 0.0)] * 20, opacity=0.0)
    extractor = RegionExtractor(min_opacity=0.1)
    regions = extractor.extract_regions(snap)
    assert regions == []


def test_region_extractor_single_cluster() -> None:
    positions = [(float(i) * 0.01, 0.0, float(i) * 0.01) for i in range(50)]
    snap = _make_snapshot(positions, opacity=0.8)
    extractor = RegionExtractor(eps_metres=0.5, min_samples=2, max_regions=5)
    regions = extractor.extract_regions(snap)
    # At minimum, all Gaussians should be reachable as one region.
    assert len(regions) >= 1
    total = sum(r.gaussian_count for r in regions)
    assert total == 50


def test_region_extractor_returns_jpeg_image() -> None:
    positions = [(float(i) * 0.01, 0.0, float(i) * 0.01) for i in range(30)]
    snap = _make_snapshot(positions, opacity=0.8)
    extractor = RegionExtractor(eps_metres=0.5, min_samples=2)
    regions = extractor.extract_regions(snap)
    assert len(regions) >= 1
    # JPEG magic bytes: FF D8 FF
    assert regions[0].image_bytes[:2] == b"\xff\xd8"


def test_region_extractor_sorted_by_gaussian_count() -> None:
    # Two tight clusters of different sizes, far apart.
    cluster_a = [(0.0 + i * 0.005, 0.0, 0.0) for i in range(40)]
    cluster_b = [(10.0 + i * 0.005, 0.0, 0.0) for i in range(20)]
    snap = _make_snapshot(cluster_a + cluster_b, opacity=0.9)
    extractor = RegionExtractor(eps_metres=0.1, min_samples=5)
    regions = extractor.extract_regions(snap)
    if len(regions) >= 2:
        assert regions[0].gaussian_count >= regions[1].gaussian_count


def test_region_extractor_respects_max_regions() -> None:
    # Many small well-separated clusters.
    positions = [(float(i) * 5.0, 0.0, 0.0) for i in range(100)]
    snap = _make_snapshot(positions * 5, opacity=0.8)
    extractor = RegionExtractor(eps_metres=0.1, min_samples=2, max_regions=3)
    regions = extractor.extract_regions(snap)
    assert len(regions) <= 3


def test_bounding_box_center() -> None:
    bbox = BoundingBox(0.0, 0.0, 0.0, 2.0, 4.0, 6.0)
    cx, cy, cz = bbox.center
    assert cx == pytest.approx(1.0)
    assert cy == pytest.approx(2.0)
    assert cz == pytest.approx(3.0)


def test_bounding_box_size() -> None:
    bbox = BoundingBox(1.0, 2.0, 3.0, 4.0, 6.0, 9.0)
    sx, sy, sz = bbox.size
    assert sx == pytest.approx(3.0)
    assert sy == pytest.approx(4.0)
    assert sz == pytest.approx(6.0)


# ── OllamaClient tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ollama_client_check_model_present() -> None:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "models": [{"name": "moondream:latest"}, {"name": "llama2:7b"}]
    }
    mock_response.raise_for_status = MagicMock()

    import httpx

    async def _mock_get(path: str, **_: object) -> MagicMock:
        return mock_response

    client = OllamaClient("http://localhost:11434")
    client._client = MagicMock(spec=httpx.AsyncClient)
    client._client.get = AsyncMock(return_value=mock_response)

    result = await client.check_model("moondream")
    assert result is True


@pytest.mark.asyncio
async def test_ollama_client_check_model_absent() -> None:
    import httpx

    mock_response = MagicMock()
    mock_response.json.return_value = {"models": [{"name": "llama2:7b"}]}
    mock_response.raise_for_status = MagicMock()

    client = OllamaClient("http://localhost:11434")
    client._client = MagicMock(spec=httpx.AsyncClient)
    client._client.get = AsyncMock(return_value=mock_response)

    result = await client.check_model("moondream")
    assert result is False


@pytest.mark.asyncio
async def test_ollama_client_check_model_connection_error() -> None:
    import httpx

    client = OllamaClient("http://localhost:11434")
    client._client = MagicMock(spec=httpx.AsyncClient)
    client._client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

    with pytest.raises(OllamaConnectionError):
        await client.check_model("moondream")


@pytest.mark.asyncio
async def test_ollama_client_generate_returns_response_text() -> None:
    import httpx

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": '{"label": "cup"}'}
    mock_response.raise_for_status = MagicMock()

    client = OllamaClient("http://localhost:11434")
    client._client = MagicMock(spec=httpx.AsyncClient)
    client._client.post = AsyncMock(return_value=mock_response)

    result = await client.generate(model="moondream", prompt="Describe this.")
    assert result == '{"label": "cup"}'


@pytest.mark.asyncio
async def test_ollama_client_generate_raises_on_timeout() -> None:
    import httpx

    client = OllamaClient("http://localhost:11434")
    client._client = MagicMock(spec=httpx.AsyncClient)
    client._client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

    with pytest.raises(OllamaResponseError):
        await client.generate(model="moondream", prompt="test")


def test_ollama_client_get_client_raises_without_context() -> None:
    client = OllamaClient("http://localhost:11434")
    with pytest.raises(RuntimeError, match="context manager"):
        client._get_client()
