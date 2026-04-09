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
                "object_label": "Floor lamp",
                "risk_score": 0.82,
                "description": "Tall lamp leaning near a walkway.",
                "severity": "high",
                "location_label": "front-right zone",
            }
        ],
        "summary": {
            "filename": "living-room.mp4",
            "hazard_count": 1,
            "object_count": 1,
            "scene_source": "heuristic_estimate",
            "confidence_label": "Approximate spatial grounding",
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
        "trust_notes": [
            "Locations are approximate because upload-side geometry is still heuristic."
        ],
        "scene_source": "heuristic_estimate",
        "report_url": "/reports/job12345.pdf",
        "error": None,
    }

    response = client.get("/jobs/job12345")
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["top_hazard_label"] == "Floor lamp"
    assert data["recommendations"][0]["priority"] == "high"
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
                "object_label": "Book stack",
                "risk_score": 0.61,
                "description": "Stack is leaning near the crib.",
                "severity": "medium",
                "location_label": "left wall",
            }
        ],
        "summary": {
            "filename": "nursery.mp4",
            "hazard_count": 1,
            "object_count": 2,
            "scene_source": "heuristic_estimate",
            "confidence_label": "Approximate spatial grounding",
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
        "trust_notes": ["Treat scene positions as approximate."],
        "scene_source": "heuristic_estimate",
        "report_url": "/reports/jobpdf01.pdf",
        "error": None,
    }

    response = client.get("/reports/jobpdf01.pdf")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert "atlas0-report-jobpdf01.pdf" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF-")
