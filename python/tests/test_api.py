"""Tests for the Atlas-0 API server."""

import pytest
from atlas.api.server import _upload_jobs, app
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_upload_jobs():
    """Keep upload job state isolated across tests."""
    snapshot = dict(_upload_jobs)
    _upload_jobs.clear()
    yield
    _upload_jobs.clear()
    _upload_jobs.update(snapshot)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_spatial_query_empty():
    response = client.post("/query", json={"query": "where is the cup?"})
    assert response.status_code == 200
    assert response.json() == []


def test_upload_job_status_exposes_report_fields():
    _upload_jobs["job12345"] = {
        "job_id": "job12345",
        "filename": "living-room.mp4",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [
            {
                "object_id": "obj-1",
                "hazard_code": "top_heavy_tipping",
                "object_label": "Floor lamp",
                "risk_score": 0.82,
                "description": "Tall lamp leaning near a walkway.",
                "what": "Tall lamp leaning near a walkway.",
                "why_it_matters": "Tall items can tip when bumped.",
                "what_to_do_next": "Move it away from the walkway or secure the base.",
                "severity": "high",
                "location_label": "front-right zone",
                "confidence": 0.79,
                "confidence_label": "strong",
                "reasoning": {
                    "support_summary": "3 supporting observations across the scan",
                    "signals": ["slenderness: 2.1"],
                    "grounding_confidence": 0.76,
                },
                "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
                "latest_feedback": None,
            }
        ],
        "fix_first": [
            {
                "title": "Top-heavy tipping risk",
                "action": "Move it away from the walkway or secure the base.",
                "why": "Tall items can tip when bumped.",
                "location": "front-right zone",
                "severity": "high",
                "confidence_label": "strong",
            }
        ],
        "summary": {
            "filename": "living-room.mp4",
            "hazard_count": 1,
            "object_count": 1,
            "scene_source": "heuristic_estimate",
            "confidence_label": "Approximate spatial grounding",
            "scan_quality_label": "Fair",
            "scan_quality_score": 0.62,
            "top_hazard_label": "Floor lamp",
            "top_severity": "high",
        },
        "recommendations": [
            {
                "title": "Stabilize tall lamp",
                "action": "Move it away from the walkway or secure the base.",
                "why": "Tall items can tip when bumped.",
                "location": "front-right zone",
                "priority": "high",
            }
        ],
        "evidence_frames": [
            {
                "image_url": "data:image/jpeg;base64,ZmFrZQ==",
                "caption": "Lamp near walkway",
                "confidence": 0.82,
            }
        ],
        "scan_quality": {
            "status": "fair",
            "score": 0.62,
            "usable": True,
            "warnings": ["The walkthrough covers too little motion to ground objects confidently."],
            "retry_guidance": [
                "Move slowly across the room so objects appear from more than one viewpoint."
            ],
            "metrics": {"frame_count": 5},
        },
        "trust_notes": [
            "Locations are approximate because upload-side geometry is still heuristic."
        ],
        "scene_source": "heuristic_estimate",
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": "/reports/job12345.pdf",
        "error": None,
    }

    response = client.get("/jobs/job12345")
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["top_hazard_label"] == "Floor lamp"
    assert data["fix_first"][0]["title"] == "Top-heavy tipping risk"
    assert data["recommendations"][0]["priority"] == "high"
    assert data["scan_quality"]["status"] == "fair"
    assert data["evidence_frames"][0]["caption"] == "Lamp near walkway"
    assert data["report_url"] == "/reports/job12345.pdf"
    assert data["scene_source"] == "heuristic_estimate"


def test_report_pdf_missing_job_returns_404():
    response = client.get("/reports/missing.pdf")
    assert response.status_code == 404


def test_report_pdf_download_for_completed_job():
    _upload_jobs["jobpdf01"] = {
        "job_id": "jobpdf01",
        "filename": "nursery.mp4",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [
            {
                "object_id": "obj-2",
                "hazard_code": "unstable_stack",
                "object_label": "Book stack",
                "risk_score": 0.61,
                "description": "Stack is leaning near the crib.",
                "what": "Stack is leaning near the crib.",
                "why_it_matters": "Loose stacks can slide or fall.",
                "what_to_do_next": "Move the heaviest books lower or flatten the pile.",
                "severity": "medium",
                "location_label": "left wall",
                "confidence": 0.71,
                "confidence_label": "approximate",
                "reasoning": {
                    "support_summary": "2 supporting observations across the scan",
                    "signals": ["position variance: 0.14"],
                    "grounding_confidence": 0.67,
                },
                "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
                "latest_feedback": None,
            }
        ],
        "fix_first": [
            {
                "title": "Unstable stack",
                "action": "Move the heaviest books lower or flatten the pile.",
                "why": "Loose stacks can slide or fall.",
                "location": "left wall",
                "severity": "medium",
                "confidence_label": "approximate",
            }
        ],
        "summary": {
            "filename": "nursery.mp4",
            "hazard_count": 1,
            "object_count": 2,
            "scene_source": "heuristic_estimate",
            "confidence_label": "Approximate spatial grounding",
            "scan_quality_label": "Good",
            "scan_quality_score": 0.79,
            "top_hazard_label": "Book stack",
            "top_severity": "medium",
        },
        "recommendations": [
            {
                "title": "Lower the stack",
                "action": "Move the heaviest books lower or flatten the pile.",
                "why": "Loose stacks can slide or fall.",
                "location": "left wall",
                "priority": "medium",
            }
        ],
        "evidence_frames": [],
        "scan_quality": {
            "status": "good",
            "score": 0.79,
            "usable": True,
            "warnings": [],
            "retry_guidance": [],
        },
        "trust_notes": ["Treat scene positions as approximate."],
        "scene_source": "heuristic_estimate",
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": "/reports/jobpdf01.pdf",
        "error": None,
    }

    response = client.get("/reports/jobpdf01.pdf")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert "atlas0-report-jobpdf01.pdf" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF-")


def test_record_job_feedback_updates_job_and_finding() -> None:
    _upload_jobs["jobfeed1"] = {
        "job_id": "jobfeed1",
        "filename": "hallway.mp4",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "objects": [],
        "risks": [
            {
                "object_id": "track-01",
                "hazard_code": "walkway_clutter",
                "hazard_title": "Walkway clutter",
                "object_label": "Box",
                "risk_score": 0.58,
                "severity": "moderate",
                "location_label": "front-center",
                "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
                "latest_feedback": None,
            }
        ],
        "fix_first": [],
        "summary": {"filename": "hallway.mp4", "hazard_count": 1, "object_count": 1},
        "recommendations": [],
        "evidence_frames": [],
        "scan_quality": {"status": "fair", "score": 0.61, "usable": True, "warnings": []},
        "trust_notes": [],
        "scene_source": "estimated_multiview",
        "finding_feedback": [],
        "feedback_summary": {"useful": 0, "wrong": 0, "duplicate": 0},
        "report_url": "/reports/jobfeed1.pdf",
        "error": None,
    }

    response = client.post(
        "/jobs/jobfeed1/feedback",
        json={
            "hazard_code": "walkway_clutter",
            "object_id": "track-01",
            "verdict": "useful",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["feedback_summary"]["useful"] == 1
    assert data["finding_feedback"][0]["verdict"] == "useful"
    assert data["risks"][0]["latest_feedback"] == "useful"
    assert data["risks"][0]["feedback_summary"]["useful"] == 1
