"""Pydantic request/response models for the Atlas-0 API.

Extracted from :mod:`atlas.api.server` to keep the FastAPI app module focused
on routing and behaviour.  These are pure data definitions with no dependency
on application state, so they can be imported freely by the server and tests.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response model for GET /health."""

    status: str
    slam_active: bool
    vlm_active: bool
    frame_count: int
    object_count: int
    risk_count: int
    risks_stale_seconds: float
    deployment_ready: bool = False
    worker_mode: str = "in_process"
    warnings: list[str] = []


class SpatialQuery(BaseModel):
    """Request body for POST /query."""

    query: str
    max_results: int = 5


class SpatialQueryResult(BaseModel):
    """A single result returned by POST /query."""

    object_label: str
    position: list[float]
    confidence: float
    risk_level: float
    description: str


class ObjectInfo(BaseModel):
    """Physical and spatial metadata for one labeled object."""

    object_id: int
    label: str
    material: str
    mass_kg: float
    fragility: float
    friction: float
    confidence: float
    position: list[float]
    relationships: list[str]


class RiskInfo(BaseModel):
    """Summary of one risk entry."""

    object_id: int
    object_label: str
    position: list[float]
    risk_score: float
    description: str


class SceneState(BaseModel):
    """Full snapshot of the current scene."""

    object_count: int
    objects: list[ObjectInfo]
    risk_count: int
    risks: list[RiskInfo]
    point_cloud: list[list[float]] = []
    """Pseudo-depth point cloud from uploaded images — each entry is [x, y, z, r, g, b]
    where rgb is normalised 0-1.  Used by the 3DGS frontend to render upload-derived
    structure in world space."""


class OperatorAccessResponse(BaseModel):
    """Public-facing access policy used by the hosted frontend."""

    requires_token: bool
    allow_unauthenticated_loopback: bool
    enable_job_listing: bool
    public_demo: bool = False
    mode: str


class OperatorSettingsResponse(BaseModel):
    """Protected operator diagnostics for hosted beta deployments."""

    access: dict[str, Any]
    uploads: dict[str, Any]
    queue: dict[str, Any]
    storage: dict[str, Any]
    system: dict[str, Any]
    providers: dict[str, Any]
    evaluation: dict[str, Any]
    product: dict[str, Any]
    beta_inbox: dict[str, Any]


class PrivacyPolicyResponse(BaseModel):
    """Public privacy posture exposed to the hosted frontend."""

    retention_days: int
    save_original_uploads: bool
    delete_supported: bool
    text_redaction_enabled: bool
    summary: str
    details: list[str]
    artifact_backend: str = "local_fs"


class UploadGuidanceResponse(BaseModel):
    """Public upload limits and capture guidance for the hosted frontend."""

    max_upload_bytes: int
    max_video_duration_seconds: float
    recommended_duration_seconds: dict[str, int]
    accepted_extensions: list[str]
    accepted_media_prefixes: list[str]
    one_room_only: bool
    checklist: list[str]
    retry_guidance: list[str]


class TrustProofResponse(BaseModel):
    """Privacy-safe public proof signals for product trust UX."""

    completed_scans: int
    rejected_or_downgraded_scans: int
    evidence_backed_reports: int
    useful_feedback_count: int
    negative_feedback_count: int
    eval_ready_reports: int
    sample_report_available: bool
    known_limits: list[str]
    proof_points: list[dict[str, str]]
    privacy_notes: list[str]


class ProductEventRequest(BaseModel):
    """One lightweight public product analytics event."""

    event_name: str
    surface: str | None = None
    job_id: str | None = None
    sample_key: str | None = None
    audience_mode: str | None = None
    room_labeled: bool | None = None
    room_label: str | None = None
    session_id: str | None = None
    client_ts: str | None = None
    referrer: str | None = None
    utm_source: str | None = None
    utm_campaign: str | None = None
    persona: str | None = None
    use_case: str | None = None
    referral_code: str | None = None
    mission_id: str | None = None
    challenge_id: str | None = None
    file_type: str | None = None
    file_size: int | None = None
    reason: str | None = None


class WaitlistRequest(BaseModel):
    """One public waitlist signup submission."""

    email: str
    use_case: str | None = None
    name: str | None = None
    notes: str | None = None
    source: str | None = None
    audience_mode: str | None = None
    persona: str | None = None
    referral_code: str | None = None


class WaitlistResponse(BaseModel):
    """Public response returned after a waitlist signup."""

    accepted: bool
    waitlist_count: int
    message: str


class StoragePruneResponse(BaseModel):
    """Protected response returned after a manual storage prune."""

    deleted_jobs: int
    bytes_reclaimed: int
    remaining_jobs: int
    remaining_bytes: int
    pruned_at: str


class EvalCandidateRequest(BaseModel):
    """Operator request to export a reviewed report as an eval candidate."""

    label: str | None = None
    note: str | None = None


class OverlayRiskEntry(BaseModel):
    """Richer risk payload sent over the WebSocket delta stream.

    Carries physics + heuristic merged scores, trajectory data, and
    pre-built overlay primitives ready for the Three.js renderer.
    """

    object_id: int
    object_label: str
    position: list[float] | None
    combined_score: float
    physics_score: float
    heuristic_score: float
    risk_type: str
    impact_point: list[float] | None
    trajectory_points: list[list[float]] | None
    description: str
    overlay: dict[str, Any]


class RiskDeltaMessage(BaseModel):
    """Delta update message pushed over ``/ws/risks``.

    Only items that changed since the previous tick are included.
    """

    added: list[OverlayRiskEntry]
    updated: list[OverlayRiskEntry]
    removed: list[int]


class UploadJobStatus(BaseModel):
    """Status of a media upload and analysis job."""

    job_id: str
    filename: str
    room_label: str | None = None
    is_sample: bool = False
    sample_key: str | None = None
    audience_mode: str | None = None
    status: str  # queued | processing | complete | error
    stage: str  # upload | ingest | vlm | risk | complete
    progress: float
    objects: list[dict[str, Any]] | None = None
    risks: list[dict[str, Any]] | None = None
    fix_first: list[dict[str, Any]] | None = None
    weekend_fix_list: list[dict[str, Any]] | None = None
    summary: dict[str, Any] | None = None
    recommendations: list[dict[str, Any]] | None = None
    evidence_frames: list[dict[str, Any]] | None = None
    scan_quality: dict[str, Any] | None = None
    trust_notes: list[str] | None = None
    scene_source: str | None = None
    finding_feedback: list[dict[str, Any]] | None = None
    feedback_summary: dict[str, int] | None = None
    evaluation_summary: dict[str, Any] | None = None
    human_evaluation: dict[str, Any] | None = None
    room_history: list[dict[str, Any]] | None = None
    room_comparison: dict[str, Any] | None = None
    room_wins: list[dict[str, Any]] | None = None
    finding_follow_up: list[dict[str, Any]] | None = None
    resolution_summary: dict[str, Any] | None = None
    report_url: str | None = None
    share_url: str | None = None
    error: str | None = None
    artifacts: dict[str, Any] | None = None
    attempt_count: int = 0
    expires_at: str | None = None
    queued_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class FindingFeedbackRequest(BaseModel):
    """One user feedback event for a specific report finding."""

    hazard_code: str
    verdict: str
    object_id: str | None = None
    note: str | None = None


class FindingFollowUpRequest(BaseModel):
    """Persist a lightweight follow-up status for one report finding."""

    hazard_code: str
    status: str
    object_id: str | None = None
    note: str | None = None


class JobEvaluationRequest(BaseModel):
    """One human review verdict for a completed report."""

    status: str
    benchmark_label: str | None = None
    missed_hazards: list[str] | None = None
    note: str | None = None
