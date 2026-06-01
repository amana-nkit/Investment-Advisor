"""
utilities/pdf_export.py
Professional PDF generation using ReportLab's Paragraph/Story API.
Replaces the original line[:100] hack that caused word-splitting.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

from utilities.logger import get_logger

log = get_logger(__name__)


def _build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title":    ParagraphStyle("Title2",    fontSize=18, fontName="Helvetica-Bold",
                                   spaceAfter=6, textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER),
        "subtitle": ParagraphStyle("Subtitle2", fontSize=11, fontName="Helvetica",
                                   spaceAfter=12, textColor=colors.HexColor("#4a4a6a"), alignment=TA_CENTER),
        "h2":       ParagraphStyle("H2",        fontSize=13, fontName="Helvetica-Bold",
                                   spaceBefore=14, spaceAfter=4, textColor=colors.HexColor("#1a1a2e")),
        "body":     ParagraphStyle("Body2",     fontSize=10, fontName="Helvetica",
                                   spaceAfter=4, leading=15),
        "bullet":   ParagraphStyle("Bullet2",   fontSize=10, fontName="Helvetica",
                                   spaceAfter=3, leftIndent=14, bulletIndent=4, leading=14),
        "disclaimer": ParagraphStyle("Disc",    fontSize=8,  fontName="Helvetica-Oblique",
                                     spaceAfter=4, textColor=colors.HexColor("#888888"), leading=11),
    }


def create_pdf(proposal_dict: dict, output_path: Optional[str] = None) -> str:
    """
    Generate a professional A4 PDF from a structured proposal dict.
    Returns the path to the created file.
    """
    if output_path is None:
        output_path = os.path.join(
            tempfile.gettempdir(), f"proposal_{uuid.uuid4().hex[:8]}.pdf"
        )

    styles = _build_styles()
    story = []

    # --- Header ---
    story.append(Paragraph("Investment Proposal", styles["title"]))
    story.append(Paragraph(
        f"Prepared for: {proposal_dict.get('customer_name', 'Client')} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Date: {proposal_dict.get('prepared_date', '')}",
        styles["subtitle"]
    ))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=10))

    # --- Key metrics row ---
    metrics = [
        ["Risk Appetite", "Horizon", "Health Score", "Expected Return"],
        [
            proposal_dict.get("risk_appetite", "—"),
            f"{proposal_dict.get('investment_horizon_years', '—')} yrs",
            f"{proposal_dict.get('financial_health_score', '—')}/10",
            f"{proposal_dict.get('expected_annual_return_pct', '—'):.1f}%"
            if isinstance(proposal_dict.get('expected_annual_return_pct'), (int, float)) else "—",
        ]
    ]
    tbl = Table(metrics, colWidths=[42*mm, 38*mm, 38*mm, 42*mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 9),
        ("BACKGROUND",  (0, 1), (-1, 1), colors.HexColor("#f0f0f8")),
        ("FONTNAME",    (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 1), (-1, 1), 11),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#1a1a2e"), colors.HexColor("#f0f0f8")]),
        ("BOX",         (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("INNERGRID",   (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    # --- Executive summary ---
    story.append(Paragraph("Executive Summary", styles["h2"]))
    story.append(Paragraph(proposal_dict.get("executive_summary", ""), styles["body"]))
    story.append(Spacer(1, 8))

    # --- Asset allocation ---
    story.append(Paragraph("Asset Allocation", styles["h2"]))
    alloc_data = [["Asset Class", "Allocation", "Rationale"]]
    for a in proposal_dict.get("asset_allocation", []):
        alloc_data.append([
            a.get("asset_class", ""),
            f"{a.get('percentage', 0):.1f}%",
            a.get("rationale", ""),
        ])
    if len(alloc_data) > 1:
        at = Table(alloc_data, colWidths=[55*mm, 25*mm, 90*mm])
        at.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f8fc")]),
            ("BOX",         (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("INNERGRID",   (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
            ("ALIGN",       (1, 0), (1, -1), "CENTER"),
            ("TOPPADDING",  (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(at)
    story.append(Spacer(1, 8))

    # --- Recommended products ---
    story.append(Paragraph("Recommended Products", styles["h2"]))
    for product in proposal_dict.get("recommended_products", []):
        story.append(Paragraph(f"• {product}", styles["bullet"]))
    story.append(Spacer(1, 8))

    # --- Risks & mitigation ---
    story.append(Paragraph("Key Risks & Mitigation", styles["h2"]))
    risks = proposal_dict.get("key_risks", [])
    mitigations = proposal_dict.get("mitigation_strategies", [])
    for i, risk in enumerate(risks):
        story.append(Paragraph(f"<b>Risk:</b> {risk}", styles["body"]))
        if i < len(mitigations):
            story.append(Paragraph(f"<b>Mitigation:</b> {mitigations[i]}", styles["bullet"]))
    story.append(Spacer(1, 12))

    # --- Disclaimer ---
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=6))
    story.append(Paragraph(proposal_dict.get("disclaimer", ""), styles["disclaimer"]))

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )
    doc.build(story)
    log.info("PDF generated: %s", output_path)
    return output_path
