"""Unit tests for atlas.utils.config — typed TOML config loader.

Covers:
- Default values when no file exists.
- Loading from the real configs/default.toml.
- ATLAS_ env var overrides for scalars, booleans, floats, and lists.
- Nested override paths (camera.intrinsics.fx, gaussian.optimizer.lr_opacity).
- Unmatched env var is ignored without raising.
- Pydantic validation errors on invalid values.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from atlas.utils.config import (
    ApiConfig,
    AtlasConfig,
    IpcConfig,
    SlamConfig,
    VlmConfig,
    _apply_env_overrides,
    _coerce,
    _set_nested,
    load_config,
)
from pydantic import ValidationError

# ── _coerce ───────────────────────────────────────────────────────────────────


class TestCoerce:
    def test_bool_true_variants(self) -> None:
        for val in ("true", "True", "TRUE", "1", "yes", "on"):
            assert _coerce(val, False) is True

    def test_bool_false_variants(self) -> None:
        for val in ("false", "0", "no", "off", ""):
            assert _coerce(val, True) is False

    def test_int(self) -> None:
        assert _coerce("9000", 8420) == 9000
        assert isinstance(_coerce("9000", 8420), int)

    def test_float(self) -> None:
        assert _coerce("3.14", 0.0) == pytest.approx(3.14)
        assert isinstance(_coerce("1.0", 0.0), float)

    def test_list(self) -> None:
        result = _coerce("http://a.com, http://b.com", ["http://x.com"])
        assert result == ["http://a.com", "http://b.com"]

    def test_list_single(self) -> None:
        assert _coerce("http://only.com", []) == ["http://only.com"]

    def test_string_passthrough(self) -> None:
        assert _coerce("llava", "moondream") == "llava"


# ── _set_nested ───────────────────────────────────────────────────────────────


class TestSetNested:
    def _data(self) -> dict:
        return {
            "api": {"port": 8420, "host": "0.0.0.0"},
            "camera": {
                "slam_width": 640,
                "intrinsics": {"fx": 458.0, "fy": 457.0},
            },
        }

    def test_flat_key(self) -> None:
        d = self._data()
        assert _set_nested(d, ["api", "port"], "9000") is True
        assert d["api"]["port"] == 9000

    def test_nested_two_levels(self) -> None:
        d = self._data()
        assert _set_nested(d, ["camera", "intrinsics", "fx"], "500.0") is True
        assert d["camera"]["intrinsics"]["fx"] == pytest.approx(500.0)

    def test_underscore_key(self) -> None:
        d = self._data()
        assert _set_nested(d, ["camera", "slam", "width"], "320") is True
        assert d["camera"]["slam_width"] == 320

    def test_unmatched_returns_false(self) -> None:
        d = self._data()
        assert _set_nested(d, ["nonexistent", "key"], "val") is False

    def test_empty_parts(self) -> None:
        d = self._data()
        assert _set_nested(d, [], "val") is False


# ── _apply_env_overrides ──────────────────────────────────────────────────────


class TestApplyEnvOverrides:
    def test_scalar_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATLAS_API_PORT", "9999")
        data: dict = {"api": {"port": 8420}}
        _apply_env_overrides(data)
        assert data["api"]["port"] == 9999

    def test_nested_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATLAS_CAMERA_INTRINSICS_FX", "600.0")
        data: dict = {"camera": {"intrinsics": {"fx": 458.0}}}
        _apply_env_overrides(data)
        assert data["camera"]["intrinsics"]["fx"] == pytest.approx(600.0)

    def test_bool_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATLAS_SLAM_ENABLE_LOOP_CLOSURE", "false")
        data: dict = {"slam": {"enable_loop_closure": True}}
        _apply_env_overrides(data)
        assert data["slam"]["enable_loop_closure"] is False

    def test_unmatched_key_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATLAS_NONEXISTENT_KEY", "val")
        data: dict = {"api": {"port": 8420}}
        _apply_env_overrides(data)  # must not raise
        assert data == {"api": {"port": 8420}}

    def test_non_atlas_prefix_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTHER_API_PORT", "1234")
        data: dict = {"api": {"port": 8420}}
        _apply_env_overrides(data)
        assert data["api"]["port"] == 8420


# ── load_config ───────────────────────────────────────────────────────────────


class TestLoadConfig:
    def test_default_values_without_file(self, tmp_path: Path) -> None:
        """When config file is missing, defaults should be used."""
        missing = tmp_path / "no_such_file.toml"
        cfg = load_config(missing)
        assert isinstance(cfg, AtlasConfig)
        assert cfg.api.port == 8420
        assert cfg.vlm.model_name == "moondream"
        assert cfg.ipc.max_gaussians == 100_000

    def test_loads_default_toml(self) -> None:
        """The real configs/default.toml should parse without errors."""
        cfg = load_config()
        assert cfg.api.port == 8420
        assert cfg.vlm.model_name == "moondream"
        assert cfg.slam.max_gaussians == 500_000
        assert cfg.camera.slam_width == 640
        assert cfg.ipc.mmap_path == "/tmp/atlas.mmap"
        assert cfg.uploads.storage_dir == ".atlas/uploads"
        assert cfg.uploads.save_original_uploads is False
        assert cfg.uploads.retention_days == 14
        assert cfg.uploads.max_upload_bytes == 75_000_000
        assert cfg.uploads.max_queue_depth == 24
        assert cfg.uploads.max_job_attempts == 2
        assert cfg.uploads.job_timeout_seconds == 180.0
        assert cfg.uploads.max_storage_bytes == 1_500_000_000
        assert cfg.api.enable_job_listing is False

    def test_env_atlas_config_selects_runtime_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_file = tmp_path / "runtime.toml"
        cfg_file.write_bytes(b"[api]\nport = 7771\n")
        monkeypatch.setenv("ATLAS_CONFIG", str(cfg_file))

        cfg = load_config()

        assert cfg.api.port == 7771

    def test_custom_toml(self, tmp_path: Path) -> None:
        toml_content = b"""
[api]
port = 7777
host = "127.0.0.1"

[vlm]
model_name = "llava"
ollama_host = "http://localhost:11434"
max_tokens = 512
temperature = 0.2
timeout_seconds = 30.0
"""
        cfg_file = tmp_path / "custom.toml"
        cfg_file.write_bytes(toml_content)
        cfg = load_config(cfg_file)
        assert cfg.api.port == 7777
        assert cfg.api.host == "127.0.0.1"
        assert cfg.vlm.model_name == "llava"

    def test_env_override_takes_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        toml_content = b"[api]\nport = 7777\n"
        cfg_file = tmp_path / "cfg.toml"
        cfg_file.write_bytes(toml_content)
        monkeypatch.setenv("ATLAS_API_PORT", "8888")
        cfg = load_config(cfg_file)
        assert cfg.api.port == 8888

    def test_env_override_nested(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATLAS_CAMERA_INTRINSICS_FX", "999.9")
        cfg = load_config()
        assert cfg.camera.intrinsics.fx == pytest.approx(999.9)

    def test_env_override_vlm_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATLAS_VLM_MODEL_NAME", "bakllava")
        cfg = load_config()
        assert cfg.vlm.model_name == "bakllava"

    def test_env_override_vlm_fallback_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATLAS_VLM_FALLBACK_PROVIDER", "openai")
        cfg = load_config()
        assert cfg.vlm.fallback_provider == "openai"

    def test_env_override_slam_bool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATLAS_SLAM_ENABLE_LOOP_CLOSURE", "false")
        cfg = load_config()
        assert cfg.slam.enable_loop_closure is False

    def test_env_override_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATLAS_API_CORS_ORIGINS", "http://a.com,http://b.com")
        cfg = load_config()
        assert cfg.api.cors_origins == ["http://a.com", "http://b.com"]


# ── Pydantic validation ───────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_port(self) -> None:
        with pytest.raises(ValidationError):
            ApiConfig(port=0)

    def test_invalid_port_too_high(self) -> None:
        with pytest.raises(ValidationError):
            ApiConfig(port=99999)

    def test_invalid_opacity(self) -> None:
        from atlas.utils.config import GaussianConfig

        with pytest.raises(ValidationError):
            GaussianConfig(initial_opacity=1.5)

    def test_invalid_temperature(self) -> None:
        with pytest.raises(ValidationError):
            VlmConfig(temperature=3.0)

    def test_invalid_fallback_provider(self) -> None:
        with pytest.raises(ValidationError):
            VlmConfig(fallback_provider="gemini")

    def test_invalid_risk_threshold(self) -> None:
        from atlas.utils.config import WorldModelConfig

        with pytest.raises(ValidationError):
            WorldModelConfig(risk_threshold=-0.1)

    def test_invalid_slam_min_features(self) -> None:
        with pytest.raises(ValidationError):
            SlamConfig(min_features=0)

    def test_invalid_ipc_max_gaussians(self) -> None:
        with pytest.raises(ValidationError):
            IpcConfig(max_gaussians=-1)

    def test_invalid_timestep(self) -> None:
        from atlas.utils.config import PhysicsConfig

        with pytest.raises(ValidationError):
            PhysicsConfig(timestep=0.0)

    def test_invalid_max_persisted_jobs(self) -> None:
        from atlas.utils.config import UploadsConfig

        with pytest.raises(ValidationError):
            UploadsConfig(max_persisted_jobs=0)

    def test_invalid_max_upload_bytes(self) -> None:
        from atlas.utils.config import UploadsConfig

        with pytest.raises(ValidationError):
            UploadsConfig(max_upload_bytes=0)

    def test_invalid_max_video_duration(self) -> None:
        from atlas.utils.config import UploadsConfig

        with pytest.raises(ValidationError):
            UploadsConfig(max_video_duration_seconds=0.0)

    def test_invalid_max_queue_depth(self) -> None:
        from atlas.utils.config import UploadsConfig

        with pytest.raises(ValidationError):
            UploadsConfig(max_queue_depth=0)

    def test_invalid_max_job_attempts(self) -> None:
        from atlas.utils.config import UploadsConfig

        with pytest.raises(ValidationError):
            UploadsConfig(max_job_attempts=0)

    def test_invalid_job_timeout(self) -> None:
        from atlas.utils.config import UploadsConfig

        with pytest.raises(ValidationError):
            UploadsConfig(job_timeout_seconds=0.0)

    def test_invalid_max_storage_bytes(self) -> None:
        from atlas.utils.config import UploadsConfig

        with pytest.raises(ValidationError):
            UploadsConfig(max_storage_bytes=0)

    def test_invalid_text_density_threshold(self) -> None:
        from atlas.utils.config import UploadsConfig

        with pytest.raises(ValidationError):
            UploadsConfig(text_density_threshold=1.5)


# ── Default.toml round-trip ───────────────────────────────────────────────────


class TestDefaultTomlRoundTrip:
    def test_toml_matches_model_defaults(self) -> None:
        """Keys in default.toml must all be accepted by AtlasConfig."""
        toml_path = Path(__file__).parent.parent.parent / "configs" / "default.toml"
        assert toml_path.exists(), f"default.toml not found at {toml_path}"
        with open(toml_path, "rb") as fh:
            raw = tomllib.load(fh)
        # Must not raise.
        cfg = AtlasConfig.model_validate(raw)
        assert cfg.api.port == 8420
