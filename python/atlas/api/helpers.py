"""Pure, stateless helper functions for the Atlas-0 API.

Extracted from :mod:`atlas.api.server`.  Every function here is a pure
transformation of its arguments with no dependency on application state,
configuration, or the upload store, which keeps them trivially testable and
reusable.  :mod:`atlas.api.server` re-imports them to preserve call sites.
"""

from __future__ import annotations

import base64
import uuid
from datetime import UTC, datetime
from typing import Any

_TRACEPARENT_VERSION = "00"

_PUBLIC_PRODUCT_EVENTS = {
    "beta_onboarding_started",
    "beta_invite_copied",
    "before_after_card_copied",
    "capture_coach_checked",
    "capture_mode_changed",
    "confidence_inspector_opened",
    "cta_start_scan",
    "daily_mission_completed",
    "daily_mission_started",
    "field_note_expanded",
    "first_run_started",
    "fix_guide_opened",
    "fix_checklist_toggled",
    "fix_library_opened",
    "fix_plan_copied",
    "fix_quest_completed",
    "fix_today_copied",
    "home_bingo_task_completed",
    "home_journal_opened",
    "home_pulse_opened",
    "landing_section_viewed",
    "live_capture_coach_started",
    "live_capture_quality_checked",
    "mystery_mode_started",
    "personal_mode_selected",
    "pdf_export_clicked",
    "one_thing_today_completed",
    "one_thing_today_started",
    "offline_upload_queued",
    "offline_upload_retried",
    "post_report_feedback_submitted",
    "pwa_offline_ready",
    "privacy_receipt_copied",
    "privacy_receipt_opened",
    "report_share_card_copied",
    "report_answer_copied",
    "report_question_asked",
    "room_map_preview_opened",
    "room_compare_opened",
    "room_care_calendar_opened",
    "room_care_task_completed",
    "room_care_week_regenerated",
    "room_health_timeline_opened",
    "room_passport_opened",
    "room_personality_viewed",
    "room_playbook_started",
    "sample_cta_clicked",
    "sample_gallery_opened",
    "sample_journey_opened",
    "sample_report_opened",
    "share_card_style_changed",
    "share_card_studio_copied",
    "evidence_privacy_toggled",
    "trust_dashboard_opened",
    "room_win_card_shared",
    "weekly_recap_copied",
    "weekly_challenge_completed",
    "room_win_copied",
    "room_reminder_clicked",
    "room_ritual_completed",
    "room_ritual_started",
    "rescan_prompt_clicked",
    "same_room_rescan_started",
    "scan_preflight_failed",
    "settings_accessibility_changed",
    "settings_data_cleared",
    "settings_default_scan_changed",
    "settings_feedback_clicked",
    "settings_daily_value_changed",
    "settings_local_backup_exported",
    "settings_local_backup_imported",
    "seasonal_pack_started",
    "seasonal_pack_selected",
    "smart_rescan_coach_opened",
    "settings_motion_changed",
    "settings_report_preferences_changed",
    "settings_sample_opened",
    "settings_theme_changed",
    "fix_verification_started",
    "fix_verification_copied",
    "evidence_frame_focused",
    "evidence_story_opened",
    "confidence_explainer_opened",
    "welcome_tour_completed",
    "upload_started",
    "upload_completed",
    "report_viewed",
    "report_theme_changed",
    "report_share_copied",
    "report_pdf_downloaded",
    "waitlist_submitted",
}


def _safe_request_id(value: str | None) -> str:
    """Return a bounded request ID safe to echo in headers and logs."""
    token = "".join(ch for ch in str(value or "") if ch.isalnum() or ch in {"-", "_"})
    return token[:80] or uuid.uuid4().hex


def _trace_id_from_traceparent(value: str | None) -> str:
    """Extract a W3C trace ID or generate a fresh one."""
    parts = str(value or "").split("-")
    if len(parts) >= 4 and len(parts[1]) == 32:
        trace_id = parts[1].lower()
        if trace_id != "0" * 32 and all(ch in "0123456789abcdef" for ch in trace_id):
            return trace_id
    return uuid.uuid4().hex


def _traceparent(trace_id: str) -> str:
    """Build a minimal W3C traceparent header for downstream correlation."""
    return f"{_TRACEPARENT_VERSION}-{trace_id}-{uuid.uuid4().hex[:16]}-01"


def _fmt_pos(pos: tuple[float, float, float]) -> str:
    """Format a 3D position as ``(x.xx, y.yy, z.zz)``."""
    return f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"


def _normalize_public_event_name(value: str | None) -> str | None:
    """Normalize one public-facing product event name."""
    event_name = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return event_name if event_name in _PUBLIC_PRODUCT_EVENTS else None


def _bounded_text(value: str | None, *, max_len: int) -> str | None:
    """Collapse whitespace and keep public product metadata safely bounded."""
    text = _collapsed_text(value)
    if not text:
        return None
    return text[:max_len]


def _collapsed_text(value: str | None) -> str | None:
    """Collapse whitespace in optional user-provided text."""
    text = " ".join(str(value or "").strip().split())
    return text or None


def _normalize_waitlist_email(value: str | None) -> str | None:
    """Normalize and lightly validate one waitlist email address."""
    email = str(value or "").strip().lower()
    if not email or len(email) > 160 or "@" not in email:
        return None
    local, _sep, domain = email.partition("@")
    if not local or "." not in domain or domain.startswith(".") or domain.endswith("."):
        return None
    return email


def _normalize_follow_up_status(value: str | None) -> str | None:
    """Normalize one persisted finding follow-up status."""
    status = str(value or "").strip().lower()
    if status in {"resolved", "monitor", "ignored"}:
        return status
    return None


def _mask_waitlist_email(email: str | None) -> str:
    """Return a privacy-safe email preview for operator beta triage."""
    value = str(email or "").strip().lower()
    local, sep, domain = value.partition("@")
    if not sep or not local or not domain:
        return "unknown"
    visible = local[:2] if len(local) > 2 else local[:1]
    return f"{visible}***@{domain}"


def _iso_sort_key(job: dict[str, Any]) -> str:
    """Return a stable descending sort key for job recency."""
    return str(job.get("completed_at") or job.get("queued_at") or job.get("job_id") or "")


def _normalize_room_label(value: str | None) -> str | None:
    """Normalize a user-facing room label for storage and comparison."""
    label = " ".join(str(value or "").strip().split())
    if not label:
        return None
    return label


def _finding_key(hazard: dict[str, Any]) -> str:
    """Build a stable identifier for one report finding."""
    return (
        f"{hazard.get('object_id') or hazard.get('object_label') or 'finding'}:"
        f"{hazard.get('hazard_code') or 'unknown'}"
    )


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def _point_cloud_centroid(points: list[list[float]]) -> tuple[float, float, float]:
    """Estimate a stable object position from an upload-derived point cloud."""
    if not points:
        return (0.0, 0.8, 1.5)

    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    zs = [float(p[2]) for p in points]
    return (
        round(sum(xs) / len(xs), 3),
        round(sum(ys) / len(ys), 3),
        round(sum(zs) / len(zs), 3),
    )


def _encode_data_url(content: bytes, mime_type: str) -> str:
    """Encode raw media bytes as a data URL for inline evidence previews."""
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _risk_severity(score: float) -> str:
    """Convert a numeric score into a user-facing severity bucket."""
    if score >= 0.78:
        return "critical"
    if score >= 0.58:
        return "high"
    if score >= 0.35:
        return "moderate"
    return "low"


def _location_label(position: tuple[float, float, float]) -> str:
    """Describe an approximate room zone from an estimated object position."""
    x, _y, z = position
    horizontal = "center"
    depth = "middle"

    if x < -0.8:
        horizontal = "left"
    elif x > 0.8:
        horizontal = "right"

    if z < 0.8:
        depth = "front"
    elif z > 1.8:
        depth = "back"

    if horizontal == "center" and depth == "middle":
        return "center area"
    return f"{depth}-{horizontal}".replace("-center", "")
