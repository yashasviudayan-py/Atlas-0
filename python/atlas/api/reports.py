"""Report rendering for the Atlas-0 API.

Extracted from :mod:`atlas.api.server`.  Turns a finished job dict into a
self-contained PDF report.  Pure transformation of its argument (no
application state), so :mod:`atlas.api.server` re-imports it.
"""

from __future__ import annotations

from typing import Any

from atlas.world_model.hazards import audience_mode_label


def _build_pdf_report(job: dict[str, Any]) -> bytes:
    """Generate a compact PDF report without extra runtime dependencies."""
    summary = job.get("summary") or {}
    risks = job.get("risks") or []
    fix_first = job.get("fix_first") or []
    weekend_fix_list = job.get("weekend_fix_list") or []
    recommendations = job.get("recommendations") or []
    scan_quality = job.get("scan_quality") or {}
    trust_notes = job.get("trust_notes") or []
    evaluation_summary = job.get("evaluation_summary") or {}
    room_wins = job.get("room_wins") or []

    lines = [
        "ATLAS-0 Room Safety Report",
        f"Scan file: {summary.get('filename', 'unknown')}",
        (
            "Audience mode: "
            f"{summary.get('audience_label', audience_mode_label(job.get('audience_mode')))}"
        ),
        f"Hazards found: {summary.get('hazard_count', 0)}",
        f"Objects detected: {summary.get('object_count', 0)}",
        f"Scene source: {summary.get('scene_source', 'unknown')}",
        f"Report posture: {summary.get('report_posture', 'screening')}",
        (
            "Scan quality: "
            f"{str(scan_quality.get('status', 'unknown')).upper()} "
            f"({int(float(scan_quality.get('score', 0.0)) * 100)} / 100)"
        ),
        f"Coverage: {summary.get('coverage_label', 'Unknown')}",
        summary.get(
            "screening_statement",
            (
                "This report flags likely hazards from the uploaded scan."
                " It does not certify that the room is safe."
            ),
        ),
        "",
        "Fix first:",
    ]

    if fix_first:
        for action in fix_first[:3]:
            lines.append(f"- {action.get('title', 'Action')}: {action.get('action', '')}")
    else:
        lines.append("- No high-priority actions were generated.")

    lines.extend(["", "Weekend fix list:"])
    if weekend_fix_list:
        for item in weekend_fix_list[:3]:
            lines.append(
                f"- {item.get('title', 'Weekend fix')} ({item.get('effort', '20-30 minutes')}): "
                f"{item.get('task', '')}"
            )
    else:
        lines.append("- No weekend fix list was generated.")

    lines.extend(
        [
            "",
            "Top hazards:",
        ]
    )

    if risks:
        for risk in risks[:5]:
            lines.append(
                f"- {risk.get('hazard_title', risk.get('object_label', 'Object'))} "
                f"({str(risk.get('severity', 'low')).upper()}): "
                f"{risk.get('what', risk.get('description', ''))}"
            )
    else:
        lines.append(
            "- No high-confidence hazards were detected in this scan."
            " This is not a safety clearance."
        )

    lines.extend(["", "Recommended actions:"])
    if recommendations:
        for rec in recommendations[:5]:
            lines.append(f"- {rec.get('title', 'Action')}: {rec.get('action', '')}")
    else:
        lines.append("- No follow-up actions were generated.")

    if scan_quality.get("warnings"):
        lines.extend(["", "Scan quality warnings:"])
        for warning in scan_quality["warnings"][:3]:
            lines.append(f"- {warning}")

    if trust_notes:
        lines.extend(["", "Trust notes:"])
        for note in trust_notes[:3]:
            lines.append(f"- {note}")

    if room_wins:
        lines.extend(["", "Positive signs in this scan:"])
        for win in room_wins[:3]:
            lines.append(f"- {win.get('title', 'Positive sign')}: {win.get('detail', '')}")

    if evaluation_summary:
        lines.extend(
            [
                "",
                "Review loop:",
                f"- {evaluation_summary.get('summary', 'No review summary available.')}",
                (
                    f"- Precision proxy: "
                    f"{int(float(evaluation_summary.get('precision_proxy', 0.0)) * 100)} / 100"
                ),
                (
                    f"- Recall proxy: "
                    f"{int(float(evaluation_summary.get('recall_proxy', 0.0)) * 100)} / 100"
                ),
            ]
        )

    max_lines = 34
    visible_lines = lines[:max_lines]

    def _pdf_escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    y = 790
    content_lines: list[str] = []
    for index, line in enumerate(visible_lines):
        font_size = 18 if index == 0 else 11
        content_lines.append(f"BT /F1 {font_size} Tf 48 {y} Td ({_pdf_escape(line)}) Tj ET")
        y -= 24 if index == 0 else 17

    stream = "\n".join(content_lines).encode("latin-1", "replace")
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        (
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
        ),
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        (
            f"5 0 obj << /Length {len(stream)} >> stream\n".encode("ascii")
            + stream
            + b"\nendstream endobj\n"
        ),
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf)
