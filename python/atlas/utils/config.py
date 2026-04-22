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
    """VLM inference configuration.

    The ``provider`` field selects the backend:

    - ``"ollama"`` (default) — local Ollama server. Free, no API key.
      Uses ``model_name`` and ``ollama_host``.
    - ``"claude"`` — Anthropic Claude vision API. Needs ``ANTHROPIC_API_KEY``
      env var. Uses ``claude_model``. Install: ``pip install "atlas-0[claude]"``.
    - ``"openai"`` — OpenAI GPT-4o vision API. Needs ``OPENAI_API_KEY`` env var.
      Uses ``openai_model``. Install: ``pip install "atlas-0[openai]"``.

    API keys are read from environment variables only — never stored in config.
    """

    provider: str = "ollama"
    fallback_provider: str | None = None
    model_name: str = "moondream"
    ollama_host: str = "http://localhost:11434"
    claude_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-4o"
    max_tokens: int = 256
    temperature: float = 0.1
    timeout_seconds: float = 10.0

    @field_validator("provider")
    @classmethod
    def _valid_provider(cls, v: str) -> str:
        allowed = {"ollama", "claude", "openai"}
        if v.lower() not in allowed:
            raise ValueError(f"provider must be one of {allowed}, got {v!r}")
        return v.lower()

    @field_validator("fallback_provider")
    @classmethod
    def _valid_fallback_provider(cls, v: str | None) -> str | None:
        if v is None:
            return None
        token = v.strip().lower()
        if not token:
            return None
        allowed = {"ollama", "claude", "openai"}
        if token not in allowed:
            raise ValueError(f"fallback_provider must be one of {allowed}, got {v!r}")
        return token

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
    access_token: str | None = None
    allow_unauthenticated_loopback: bool = True
    enable_job_listing: bool = False

    @field_validator("port")
    @classmethod
    def _valid_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"port must be in [1, 65535], got {v}")
        return v

    @field_validator("access_token")
    @classmethod
    def _empty_token_to_none(cls, v: str | None) -> str | None:
        token = (v or "").strip()
        return token or None


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


class UploadsConfig(BaseModel):
    """Upload/report persistence configuration."""

    storage_dir: str = ".atlas/uploads"
    worker_mode: str = "in_process"
    worker_poll_seconds: float = 0.5
    worker_claim_ttl_seconds: float = 300.0
    worker_heartbeat_seconds: float = 10.0
    worker_stale_after_seconds: float = 45.0
    artifact_backend: str = "local_fs"
    artifact_base_url: str | None = None
    artifact_object_dir: str | None = None
    save_original_uploads: bool = False
    max_persisted_jobs: int = 200
    retention_days: int = 14
    max_upload_bytes: int = 75_000_000
    max_video_duration_seconds: float = 75.0
    max_concurrent_jobs: int = 2
    max_queue_depth: int = 24
    max_job_attempts: int = 2
    job_timeout_seconds: float = 180.0
    max_storage_bytes: int = 1_500_000_000
    min_scan_quality_score: float = 0.42
    min_motion_coverage: float = 0.2
    min_saliency_coverage: float = 0.22
    min_frames_for_room_report: int = 3
    redact_text_heavy_regions: bool = True
    text_density_threshold: float = 0.52
    max_redacted_regions_per_frame: int = 2
    strict_startup_checks: bool = False
    job_failure_log_limit: int = 20

    @field_validator(
        "max_persisted_jobs",
        "retention_days",
        "max_upload_bytes",
        "max_concurrent_jobs",
        "max_queue_depth",
        "max_job_attempts",
        "max_storage_bytes",
        "max_redacted_regions_per_frame",
        "min_frames_for_room_report",
        "job_failure_log_limit",
    )
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"must be positive, got {v}")
        return v

    @field_validator(
        "max_video_duration_seconds",
        "job_timeout_seconds",
        "worker_poll_seconds",
        "worker_claim_ttl_seconds",
        "worker_heartbeat_seconds",
        "worker_stale_after_seconds",
        "min_scan_quality_score",
        "min_motion_coverage",
        "min_saliency_coverage",
    )
    @classmethod
    def _positive_float(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"must be positive, got {v}")
        return v

    @field_validator(
        "text_density_threshold",
        "min_scan_quality_score",
        "min_motion_coverage",
        "min_saliency_coverage",
    )
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"must be in [0, 1], got {v}")
        return v

    @field_validator("artifact_backend")
    @classmethod
    def _valid_artifact_backend(cls, v: str) -> str:
        token = v.strip().lower()
        allowed = {"local_fs", "object_store_fs"}
        if token not in allowed:
            raise ValueError(f"artifact_backend must be one of {allowed}, got {v!r}")
        return token

    @field_validator("worker_mode")
    @classmethod
    def _valid_worker_mode(cls, v: str) -> str:
        token = v.strip().lower()
        allowed = {"in_process", "external"}
        if token not in allowed:
            raise ValueError(f"worker_mode must be one of {allowed}, got {v!r}")
        return token

    @field_validator("artifact_base_url")
    @classmethod
    def _normalize_artifact_base_url(cls, v: str | None) -> str | None:
        token = (v or "").strip()
        if not token:
            return None
        return token.rstrip("/")

    @field_validator("artifact_object_dir")
    @classmethod
    def _normalize_artifact_object_dir(cls, v: str | None) -> str | None:
        token = (v or "").strip()
        return token or None


class EvaluationConfig(BaseModel):
    """Evaluation corpus and release-gate targets."""

    target_corpus_size: int = 50
    min_reviewed_jobs: int = 8
    min_benchmark_match_rate: float = 0.75
    max_false_positive_job_rate: float = 0.35
    max_missed_hazard_rate: float = 0.2
    min_avg_review_coverage: float = 0.65

    @field_validator("target_corpus_size", "min_reviewed_jobs")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"must be positive, got {v}")
        return v

    @field_validator(
        "min_benchmark_match_rate",
        "max_false_positive_job_rate",
        "max_missed_hazard_rate",
        "min_avg_review_coverage",
    )
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"must be in [0, 1], got {v}")
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
    uploads: UploadsConfig = Field(default_factory=UploadsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


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
    path = _resolve_config_path(config_path)

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


def _resolve_config_path(config_path: Path | None) -> Path:
    """Resolve the config path, honoring ``ATLAS_CONFIG`` when present.

    Args:
        config_path: Explicit path passed by the caller.

    Returns:
        The config path to load.
    """
    if config_path is not None:
        return config_path

    env_path = os.environ.get("ATLAS_CONFIG")
    if env_path:
        return Path(env_path).expanduser()

    return _DEFAULT_CONFIG_PATH


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
