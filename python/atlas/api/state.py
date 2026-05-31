"""Shared runtime state, configuration, and singletons for the API layer.

This module owns the canonical mutable state and configuration objects used
across the API. Routers and service modules import the objects defined here so
that every layer mutates the *same* dicts/instances (e.g. ``_state`` and
``_upload_jobs``). It must not import sibling service modules (``jobs``,
``analytics``, ``pipeline``) to keep the import graph acyclic.
"""

from __future__ import annotations

import pathlib
from typing import Any

from atlas.api.overlay import CameraParams, OverlayBuilder
from atlas.api.upload_store import UploadStore
from atlas.utils.config import load_config
from atlas.vlm.inference import VLMConfig
from atlas.world_model.agent import WorldModelAgent
from atlas.world_model.query_parser import QueryParser
from atlas.world_model.relationships import SemanticObject
from atlas.world_model.risk_aggregator import RiskAggregator

# ── Static frontend directory ───────────────────────────────────────────────
# Serve the Three.js AR overlay frontend from the repo's frontend/ directory.
_FRONTEND_DIR = pathlib.Path(__file__).parents[3] / "frontend"

# ── Application state ─────────────────────────────────────────────────────────
# Stored in a dict to avoid `global` statements (PLW0603).
_state: dict[str, Any] = {}
_query_parser = QueryParser()
_runtime_cfg = load_config()
_api_cfg = _runtime_cfg.api
_upload_cfg = _runtime_cfg.uploads
_evaluation_cfg = _runtime_cfg.evaluation

_upload_store = UploadStore(
    pathlib.Path(_upload_cfg.storage_dir),
    artifact_backend=_upload_cfg.artifact_backend,
    artifact_base_url=_upload_cfg.artifact_base_url,
    artifact_object_dir=(
        pathlib.Path(_upload_cfg.artifact_object_dir) if _upload_cfg.artifact_object_dir else None
    ),
    save_original_uploads=_upload_cfg.save_original_uploads,
    max_persisted_jobs=_upload_cfg.max_persisted_jobs,
    retention_days=_upload_cfg.retention_days,
    max_storage_bytes=_upload_cfg.max_storage_bytes,
)

# ── Upload job constants ──────────────────────────────────────────────────────
_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"})
_VIDEO_TYPES = frozenset({"video/mp4", "video/quicktime", "video/webm", "video/x-msvideo"})
_CLOUD_PROVIDERS = frozenset({"openai", "claude"})

# In-memory job store (keyed by job_id).
_upload_jobs: dict[str, dict[str, Any]] = _upload_store.load_jobs()


def _get_agent() -> WorldModelAgent:
    """FastAPI dependency: lazily create and return the singleton agent."""
    if "agent" not in _state:
        _state["agent"] = WorldModelAgent()
    return _state["agent"]


def _set_agent(agent: WorldModelAgent) -> None:
    """Replace the singleton agent — used by tests to inject a mock."""
    _state["agent"] = agent


def _get_aggregator() -> RiskAggregator:
    """Return the singleton :class:`~atlas.world_model.risk_aggregator.RiskAggregator`."""
    if "aggregator" not in _state:
        _state["aggregator"] = RiskAggregator()
    return _state["aggregator"]  # type: ignore[return-value]


def _get_overlay_builder() -> OverlayBuilder:
    """Return the singleton :class:`~atlas.api.overlay.OverlayBuilder`."""
    if "overlay_builder" not in _state:
        _state["overlay_builder"] = OverlayBuilder(camera=CameraParams())
    return _state["overlay_builder"]  # type: ignore[return-value]


def _build_runtime_vlm_config() -> VLMConfig:
    """Build a :class:`VLMConfig` from the active Atlas runtime config."""
    vlm = load_config().vlm
    return VLMConfig(
        provider=vlm.provider,
        fallback_provider=vlm.fallback_provider,
        model_name=vlm.model_name,
        ollama_host=vlm.ollama_host,
        claude_model=vlm.claude_model,
        openai_model=vlm.openai_model,
        max_tokens=vlm.max_tokens,
        temperature=vlm.temperature,
        timeout_seconds=vlm.timeout_seconds,
    )


def _current_objects(agent: WorldModelAgent) -> list[SemanticObject]:
    """Return the best available object list for API responses."""
    objects = agent.get_objects_sync()
    return objects or agent.build_objects_from_store()
