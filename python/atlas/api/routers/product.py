"""Public product surface: access policy, privacy, guidance, events, waitlist."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from atlas.api.analytics import (
    _operator_access_descriptor,
    _public_privacy_descriptor,
    _public_trust_proof_descriptor,
    _public_upload_guidance_descriptor,
    _request_host,
)
from atlas.api.helpers import (
    _bounded_text,
    _collapsed_text,
    _normalize_public_event_name,
    _normalize_waitlist_email,
    _utc_now_iso,
)
from atlas.api.metrics import waitlist_signup_total
from atlas.api.models import (
    OperatorAccessResponse,
    PrivacyPolicyResponse,
    ProductEventRequest,
    TrustProofResponse,
    UploadGuidanceResponse,
    WaitlistRequest,
    WaitlistResponse,
)
from atlas.api.state import _upload_store
from atlas.world_model.hazards import normalize_audience_mode

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/operator/access", response_model=OperatorAccessResponse)
def operator_access() -> OperatorAccessResponse:
    """Expose the minimal hosted-access policy needed by the frontend."""
    return OperatorAccessResponse(**_operator_access_descriptor())


@router.get("/product/privacy", response_model=PrivacyPolicyResponse)
def product_privacy() -> PrivacyPolicyResponse:
    """Expose user-visible privacy and deletion controls."""
    return PrivacyPolicyResponse(**_public_privacy_descriptor())


@router.get("/product/upload-guidance", response_model=UploadGuidanceResponse)
def product_upload_guidance() -> UploadGuidanceResponse:
    """Expose upload limits and capture guidance before a user submits a scan."""
    return UploadGuidanceResponse(**_public_upload_guidance_descriptor())


@router.get("/product/trust-proof", response_model=TrustProofResponse)
def product_trust_proof() -> TrustProofResponse:
    """Expose aggregate-only proof signals for public product trust UX."""
    return TrustProofResponse(**_public_trust_proof_descriptor())


@router.post("/product/events", status_code=204)
def record_product_event(payload: ProductEventRequest, request: Request) -> Response:
    """Record one lightweight public product event for funnel analysis."""
    event_name = _normalize_public_event_name(payload.event_name)
    if event_name is None:
        raise HTTPException(status_code=400, detail="Unknown product event name.")

    _upload_store.append_product_event(
        {
            "event_name": event_name,
            "surface": str(payload.surface or "").strip() or None,
            "job_id": str(payload.job_id or "").strip() or None,
            "sample_key": str(payload.sample_key or "").strip() or None,
            "audience_mode": normalize_audience_mode(payload.audience_mode),
            "room_labeled": bool(payload.room_labeled),
            "room_label": _bounded_text(payload.room_label, max_len=80),
            "session_id": _bounded_text(payload.session_id, max_len=80),
            "client_ts": _bounded_text(payload.client_ts, max_len=64),
            "referrer": _bounded_text(payload.referrer, max_len=240),
            "utm_source": _bounded_text(payload.utm_source, max_len=80),
            "utm_campaign": _bounded_text(payload.utm_campaign, max_len=120),
            "persona": _bounded_text(payload.persona, max_len=80),
            "use_case": _bounded_text(payload.use_case, max_len=120),
            "referral_code": _bounded_text(payload.referral_code, max_len=80),
            "mission_id": _bounded_text(payload.mission_id, max_len=80),
            "challenge_id": _bounded_text(payload.challenge_id, max_len=80),
            "file_type": _bounded_text(payload.file_type, max_len=80),
            "file_size": max(int(payload.file_size or 0), 0),
            "reason": _bounded_text(payload.reason, max_len=180),
            "host": _request_host(request),
            "created_at": _utc_now_iso(),
        }
    )
    return Response(status_code=204)


@router.post("/product/waitlist", response_model=WaitlistResponse)
def join_waitlist(payload: WaitlistRequest, request: Request) -> WaitlistResponse:
    """Capture a public beta-interest submission."""
    email = _normalize_waitlist_email(payload.email)
    if email is None:
        raise HTTPException(status_code=400, detail="Enter a valid email address.")

    name = _collapsed_text(payload.name) or ""
    use_case = _collapsed_text(payload.use_case) or ""
    notes = _collapsed_text(payload.notes) or ""
    source = _bounded_text(payload.source, max_len=80) or "hero_waitlist"
    audience_mode = normalize_audience_mode(payload.audience_mode)
    persona = _bounded_text(payload.persona, max_len=80)
    referral_code = _bounded_text(payload.referral_code, max_len=80)
    if len(name) > 80:
        raise HTTPException(status_code=400, detail="Name must be 80 characters or fewer.")
    if len(use_case) > 120:
        raise HTTPException(status_code=400, detail="Use case must be 120 characters or fewer.")
    if len(notes) > 280:
        raise HTTPException(
            status_code=400,
            detail="Waitlist note must be 280 characters or fewer.",
        )

    existing_entries = _upload_store.load_waitlist_entries()
    for existing in existing_entries:
        if str(existing.get("email") or "").lower() == email:
            _upload_store.append_product_event(
                {
                    "event_name": "waitlist_submitted",
                    "surface": source,
                    "job_id": None,
                    "sample_key": None,
                    "audience_mode": audience_mode,
                    "room_labeled": False,
                    "room_label": None,
                    "session_id": None,
                    "client_ts": None,
                    "referrer": None,
                    "utm_source": None,
                    "utm_campaign": None,
                    "persona": persona,
                    "use_case": use_case or None,
                    "referral_code": referral_code,
                    "host": _request_host(request),
                    "created_at": _utc_now_iso(),
                    "deduped": True,
                }
            )
            return WaitlistResponse(
                accepted=True,
                waitlist_count=len(existing_entries),
                message=(
                    "You're already on the list. We'll use your original signup"
                    " for the next beta wave."
                ),
            )

    entry = {
        "email": email,
        "name": name or None,
        "use_case": use_case or None,
        "notes": notes or None,
        "source": source,
        "audience_mode": audience_mode,
        "persona": persona,
        "referral_code": referral_code,
        "host": _request_host(request),
        "created_at": _utc_now_iso(),
    }
    _upload_store.append_waitlist_entry(entry)
    waitlist_signup_total.inc()
    _upload_store.append_product_event(
        {
            "event_name": "waitlist_submitted",
            "surface": source,
            "job_id": None,
            "sample_key": None,
            "audience_mode": audience_mode,
            "room_labeled": False,
            "room_label": None,
            "session_id": None,
            "client_ts": None,
            "referrer": None,
            "utm_source": None,
            "utm_campaign": None,
            "persona": persona,
            "use_case": use_case or None,
            "referral_code": referral_code,
            "host": _request_host(request),
            "created_at": entry["created_at"],
        }
    )
    waitlist_count = len(_upload_store.load_waitlist_entries())
    return WaitlistResponse(
        accepted=True,
        waitlist_count=waitlist_count,
        message="You're on the list. We'll use this to shape the next beta wave.",
    )
