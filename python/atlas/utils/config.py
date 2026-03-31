"""Atlas-0 configuration loader.

Loads ``configs/default.toml`` and applies ``ATLAS_`` prefixed environment
variable overrides.  All sections are validated via Pydantic models so
callers get typed, IDE-friendly config objects with clear error messages.

Override format
---------------
Environment variable names follow the pattern::

    ATLAS_<SECTION>[_<SUBSECTION>]_<KEY>=<value>

Examples::

    ATLAS_API_PORT=9000          → api.port = 9000
    ATLAS_VLM_MODEL_NAME=llava   → vlm.model_name = "llava"
    ATLAS_CAMERA_INTRINSICS_FX=500.0  → camera.intrinsics.fx = 500.0

Values are coerced to the type of the field's default value.  Booleans
accept ``"true"``/``"1"``/``"yes"`` (case-insensitive) as truthy.

Usage::

    from atlas.utils.config import load_config
    cfg = load_config()
    print(cfg.api.port)          # 8420 (or env override)
    print(cfg.vlm.model_name)    # "moondream"
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger(__name__)

# Default config file path (resolved relative to this file's repo root).
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "default.toml"


# ── Section models ─────────────────────────────────────────────────────────────


class StreamConfig(BaseModel):
    """Video ingestion pipeline configuration."""

    target_fps: int = 60
    frame_width: int = 1280
    frame_height: int = 720
    buffer_size: int = 4
    device: str = "0"


class CameraIntrinsicsConfig(BaseModel):
    """Pinhole camera intrinsic parameters."""

    fx: float = 458.654
    fy: float = 457.296
    cx: float = 367.215
    cy: float = 248.375


class CameraDistortionConfig(BaseModel):
    """Radial and tangential distortion coefficients (OpenCV convention)."""

    k1: float = -0.283408
    k2: float = 0.073959
    p1: float = 0.000194
    p2: float = 0.000018


class CameraConfig(BaseModel):
    """Camera hardware and calibration configuration."""

    slam_width: int = 640
    slam_height: int = 480
    intrinsics: CameraIntrinsicsConfig = Field(default_factory=CameraIntrinsicsConfig)
    distortion: CameraDistortionConfig = Field(default_factory=CameraDistortionConfig)


class SlamConfig(BaseModel):
    """SLAM pipeline tuning parameters."""

    min_features: int = 100
    max_gaussians: int = 500_000
    keyframe_translation_threshold: float = 0.1
    keyframe_rotation_threshold: float = 0.1
    optimization_iterations: int = 10
    enable_loop_closure: bool = True

    @field_validator("min_features", "max_gaussians", "optimization_iterations")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"must be positive, got {v}")
        return v


class GaussianOptimizerConfig(BaseModel):
    """Adam learning rates for differentiable 3DGS optimization."""

    lr_opacity: float = 0.05
    lr_color: float = 0.01
    lr_position: float = 0.0001
    lr_covariance: float = 0.0005
    min_opacity: float = 0.005


class GaussianConfig(BaseModel):
    """Gaussian map initialization and pruning parameters."""

    initial_opacity: float = 0.5
    min_scale: float = 0.001
    max_scale: float = 1.0
    optimizer: GaussianOptimizerConfig = Field(default_factory=GaussianOptimizerConfig)

    @field_validator("initial_opacity")
    @classmethod
    def _unit_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"opacity must be in [0, 1], got {v}")
        return v


class PhysicsConfig(BaseModel):
    """Physics simulation parameters."""

    timestep: float = 0.001
    gravity: float = 9.81
    max_steps: int = 1_000
    rest_threshold: float = 0.01

    @field_validator("timestep")
    @classmethod
    def _positive_float(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"timestep must be positive, got {v}")
        return v


class VlmConfig(BaseModel):
    """VLM / Ollama inference configuration."""

    model_name: str = "moondream"
    ollama_host: str = "http://localhost:11434"
    max_tokens: int = 256
    temperature: float = 0.1
    timeout_seconds: float = 10.0

    @field_validator("temperature")
    @classmethod
    def _valid_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {v}")
        return v


class WorldModelConfig(BaseModel):
    """World model agent loop parameters."""

    assessment_interval_seconds: float = 1.0
    max_concurrent_queries: int = 3
    risk_threshold: float = 0.3

    @field_validator("risk_threshold")
    @classmethod
    def _unit_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"risk_threshold must be in [0, 1], got {v}")
        return v


class ApiConfig(BaseModel):
    """FastAPI server configuration."""

    host: str = "0.0.0.0"
    port: int = 8420
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    @field_validator("port")
    @classmethod
    def _valid_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"port must be in [1, 65535], got {v}")
        return v


class IpcConfig(BaseModel):
    """Rust↔Python shared-memory IPC parameters."""

    mmap_path: str = "/tmp/atlas.mmap"
    max_gaussians: int = 100_000
    snapshot_interval_keyframes: int = 5

    @field_validator("max_gaussians", "snapshot_interval_keyframes")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"must be positive, got {v}")
        return v


class AtlasConfig(BaseModel):
    """Top-level Atlas-0 configuration.

    All fields have defaults matching ``configs/default.toml``, so
    ``AtlasConfig()`` always produces a valid config even without a file.

    Example::

        cfg = load_config()
        print(cfg.api.port)        # 8420
        print(cfg.vlm.model_name)  # "moondream"
    """

    stream: StreamConfig = Field(default_factory=StreamConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    slam: SlamConfig = Field(default_factory=SlamConfig)
    gaussian: GaussianConfig = Field(default_factory=GaussianConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    vlm: VlmConfig = Field(default_factory=VlmConfig)
    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    ipc: IpcConfig = Field(default_factory=IpcConfig)


# ── Loader ────────────────────────────────────────────────────────────────────


def load_config(config_path: Path | None = None) -> AtlasConfig:
    """Load Atlas-0 configuration from TOML file with env var overrides.

    Reads the TOML file at *config_path* (defaulting to
    ``configs/default.toml`` relative to the repo root), applies any
    ``ATLAS_*`` environment variable overrides, then validates the result
    against :class:`AtlasConfig`.

    Args:
        config_path: Path to the TOML config file.  Defaults to
            ``configs/default.toml`` relative to the repository root.

    Returns:
        Validated :class:`AtlasConfig` instance.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
        pydantic.ValidationError: If any field fails validation.

    Example::

        cfg = load_config()
        cfg = load_config(Path("/etc/atlas/custom.toml"))
    """
    path = config_path or _DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.warning("config_file_not_found", path=str(path), using_defaults=True)
        data: dict[str, Any] = {}
    else:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        logger.info("config_loaded", path=str(path))

    _apply_env_overrides(data)
    config = AtlasConfig.model_validate(data)
    logger.debug("config_validated", api_port=config.api.port, vlm_model=config.vlm.model_name)
    return config


# ── Environment override helpers ──────────────────────────────────────────────


def _apply_env_overrides(data: dict[str, Any]) -> None:
    """Scan ``ATLAS_*`` env vars and apply them to *data* in-place.

    Args:
        data: TOML data dict to modify.
    """
    prefix = "ATLAS_"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        # e.g. ATLAS_API_PORT -> ["api", "port"]
        # e.g. ATLAS_CAMERA_INTRINSICS_FX -> ["camera", "intrinsics", "fx"]
        parts = env_key[len(prefix) :].lower().split("_")
        applied = _set_nested(data, parts, env_val)
        if applied:
            logger.debug("env_override_applied", env_key=env_key, value=env_val)
        else:
            logger.warning("env_override_unmatched", env_key=env_key)


def _set_nested(data: dict[str, Any], parts: list[str], value: str) -> bool:
    """Recursively navigate *data* using *parts* and set the leaf to *value*.

    Uses a greedy left-to-right match: it tries progressively longer
    underscore-joined prefixes until a matching key is found, then recurses
    into the sub-dict or sets the leaf value.

    Args:
        data: Current dict level to search in.
        parts: Remaining lowercase name fragments.
        value: Raw string value from the environment.

    Returns:
        ``True`` if the path was matched and set, ``False`` otherwise.
    """
    if not parts:
        return False

    for i in range(1, len(parts) + 1):
        key = "_".join(parts[:i])
        if key not in data:
            continue
        if i == len(parts):
            # Leaf: coerce and set.
            data[key] = _coerce(value, data[key])
            return True
        if isinstance(data[key], dict):
            return _set_nested(data[key], parts[i:], value)
    return False


def _coerce(raw: str, existing: Any) -> Any:
    """Coerce *raw* string to the type of *existing*.

    Args:
        raw: Raw string value from the environment.
        existing: Current value (used to determine the target type).

    Returns:
        Value cast to the same type as *existing*, or the raw string if the
        type is unknown.
    """
    if isinstance(existing, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(existing, int):
        return int(raw)
    if isinstance(existing, float):
        return float(raw)
    if isinstance(existing, list):
        # Comma-separated list of strings.
        return [item.strip() for item in raw.split(",") if item.strip()]
    return raw
