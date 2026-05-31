"""Protected operator diagnostics: settings and storage pruning."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request

from atlas.api.analytics import (
    _aggregate_evaluation_metrics,
    _aggregate_product_metrics,
    _build_beta_inbox,
    _operator_access_descriptor,
    _operator_system_summary,
    _provider_runtime_summary,
    _require_private_access,
)
from atlas.api.helpers import _utc_now_iso
from atlas.api.jobs import (
    _effective_active_worker_count,
    _job_status_counts,
    _refresh_operational_metrics,
)
from atlas.api.models import OperatorSettingsResponse, StoragePruneResponse
from atlas.api.state import _state, _upload_cfg, _upload_store

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/operator/settings", response_model=OperatorSettingsResponse)
def operator_settings(request: Request) -> OperatorSettingsResponse:
    """Return protected operator diagnostics and upload-policy visibility."""
    _require_private_access(request)
    counts = _job_status_counts()
    _refresh_operational_metrics()
    return OperatorSettingsResponse(
        access=_operator_access_descriptor(),
        uploads={
            "worker_mode": _upload_cfg.worker_mode,
            "worker_poll_seconds": _upload_cfg.worker_poll_seconds,
            "worker_claim_ttl_seconds": _upload_cfg.worker_claim_ttl_seconds,
            "worker_heartbeat_seconds": _upload_cfg.worker_heartbeat_seconds,
            "worker_stale_after_seconds": _upload_cfg.worker_stale_after_seconds,
            "artifact_backend": _upload_cfg.artifact_backend,
            "artifact_base_url": _upload_cfg.artifact_base_url,
            "artifact_object_dir": _upload_cfg.artifact_object_dir,
            "save_original_uploads": _upload_cfg.save_original_uploads,
            "retention_days": _upload_cfg.retention_days,
            "max_upload_bytes": _upload_cfg.max_upload_bytes,
            "max_video_duration_seconds": _upload_cfg.max_video_duration_seconds,
            "max_concurrent_jobs": _upload_cfg.max_concurrent_jobs,
            "max_queue_depth": _upload_cfg.max_queue_depth,
            "max_job_attempts": _upload_cfg.max_job_attempts,
            "job_timeout_seconds": _upload_cfg.job_timeout_seconds,
            "max_storage_bytes": _upload_cfg.max_storage_bytes,
            "strict_startup_checks": _upload_cfg.strict_startup_checks,
        },
        queue={
            "queued_jobs": counts["queued"],
            "processing_jobs": counts["processing"],
            "completed_jobs": counts["complete"],
            "failed_jobs": counts["error"],
            "worker_count": _effective_active_worker_count(),
            "configured_capacity": _upload_cfg.max_concurrent_jobs,
        },
        storage=_upload_store.storage_summary(),
        system=_operator_system_summary(),
        providers=_provider_runtime_summary(),
        evaluation=_aggregate_evaluation_metrics(),
        product=_aggregate_product_metrics(),
        beta_inbox=_build_beta_inbox(),
    )


@router.post("/operator/storage/prune", response_model=StoragePruneResponse)
def prune_operator_storage(request: Request) -> StoragePruneResponse:
    """Run retention/storage pruning immediately for operator diagnostics."""
    _require_private_access(request)
    result = _upload_store.prune()
    _state["last_prune_at"] = _utc_now_iso()
    _refresh_operational_metrics()
    return StoragePruneResponse(pruned_at=str(_state["last_prune_at"]), **result)
