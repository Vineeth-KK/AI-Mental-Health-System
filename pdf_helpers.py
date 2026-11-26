# pdf_helpers.py
"""Reusable styling, typography, and layout helpers for CareAI Technical Documentation PDF."""

import html
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, KeepTogether

USABLE_WIDTH = 504  # 612 (letter width) - 108 (54pt left + 54pt right margins)

# Palette Definitions
COLOR_PRIMARY = colors.HexColor("#0F172A")     # Slate 900 (Deep Text)
COLOR_SECONDARY = colors.HexColor("#334155")   # Slate 700 (Body Subdued)
COLOR_BRAND_BLUE = colors.HexColor("#0284C7")  # Sky 600 (Primary Brand)
COLOR_ACCENT_WARM = colors.HexColor("#D97706") # Amber 600 (Secondary Accent)
COLOR_SUCCESS = colors.HexColor("#059669")     # Emerald 600 (Tips / Restorative)
COLOR_DANGER = colors.HexColor("#DC2626")      # Red 600 (Alerts / Crisis)
COLOR_BG_CARD = colors.HexColor("#F8FAFC")     # Slate 50 (Table / Box Fill)
COLOR_BORDER = colors.HexColor("#CBD5E1")      # Slate 300 (Subtle Border)
COLOR_HEADER_BG = colors.HexColor("#1E293B")   # Slate 800 (Table Headers)

# Initialize Base Stylesheet
_base_styles = getSampleStyleSheet()

STYLES = {
    "CoverTitle": ParagraphStyle(
        "CoverTitle",
        parent=_base_styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        textColor=colors.HexColor("#0F172A"),
        alignment=0,
        spaceAfter=10
    ),
    "CoverSubtitle": ParagraphStyle(
        "CoverSubtitle",
        parent=_base_styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=17,
        textColor=colors.HexColor("#334155"),
        spaceAfter=16
    ),
    "H1": ParagraphStyle(
        "H1",
        parent=_base_styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=COLOR_HEADER_BG,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    ),
    "H2": ParagraphStyle(
        "H2",
        parent=_base_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=COLOR_BRAND_BLUE,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    ),
    "H3": ParagraphStyle(
        "H3",
        parent=_base_styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=COLOR_SECONDARY,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    ),
    "Body": ParagraphStyle(
        "Body",
        parent=_base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=11.8,
        textColor=COLOR_PRIMARY,
        spaceAfter=5
    ),
    "BodyBold": ParagraphStyle(
        "BodyBold",
        parent=_base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.2,
        leading=11.8,
        textColor=COLOR_PRIMARY,
        spaceAfter=5
    ),
    "Bullet": ParagraphStyle(
        "Bullet",
        parent=_base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=11.5,
        textColor=COLOR_PRIMARY,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2.5
    ),
    "TableHeader": ParagraphStyle(
        "TableHeader",
        parent=_base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.8,
        leading=10,
        textColor=colors.white,
        alignment=0
    ),
    "TableCell": ParagraphStyle(
        "TableCell",
        parent=_base_styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.8,
        textColor=COLOR_PRIMARY
    ),
    "CodeText": ParagraphStyle(
        "CodeText",
        parent=_base_styles["Normal"],
        fontName="Courier",
        fontSize=6.8,
        leading=8.5,
        textColor=colors.HexColor("#0F172A")
    ),
    "CalloutText": ParagraphStyle(
        "CalloutText",
        parent=_base_styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=11,
        textColor=COLOR_PRIMARY
    ),
    "QAQuestion": ParagraphStyle(
        "QAQuestion",
        parent=_base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=7,
        spaceAfter=2,
        keepWithNext=True
    ),
    "QAAnswer": ParagraphStyle(
        "QAAnswer",
        parent=_base_styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=11.2,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=5
    )
}


def sanitize_text(text: str) -> str:
    """Escapes XML entities so reportlab Paragraph doesn't crash."""
    if not text:
        return ""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    text = text.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
    text = text.replace("&lt;font", "<font").replace("&lt;/font&gt;", "</font>")
    return text


def p(text: str) -> Paragraph:
    return Paragraph(sanitize_text(text), STYLES["Body"])


def p_bold(text: str) -> Paragraph:
    return Paragraph(sanitize_text(text), STYLES["BodyBold"])


def h1(text: str) -> Paragraph:
    return Paragraph(sanitize_text(text), STYLES["H1"])


def h2(text: str) -> Paragraph:
    return Paragraph(sanitize_text(text), STYLES["H2"])


def h3(text: str) -> Paragraph:
    return Paragraph(sanitize_text(text), STYLES["H3"])


def bullet(text: str) -> Paragraph:
    clean = sanitize_text(text)
    return Paragraph(f"• {clean}", STYLES["Bullet"])


def callout(title: str, text: str, kind: str = "info") -> Table:
    """Creates a professional callout box with a colored left accent border."""
    border_color = COLOR_BRAND_BLUE
    bg_color = colors.HexColor("#F0F9FF")

    if kind in ["warning", "danger"]:
        border_color = COLOR_DANGER
        bg_color = colors.HexColor("#FEF2F2")
    elif kind in ["tip", "success"]:
        border_color = COLOR_SUCCESS
        bg_color = colors.HexColor("#ECFDF5")
    elif kind in ["note", "accent"]:
        border_color = COLOR_ACCENT_WARM
        bg_color = colors.HexColor("#FFFBEB")

    content = [
        Paragraph(f"<b>{sanitize_text(title)}</b>", STYLES["CalloutText"]),
        Spacer(1, 2),
        Paragraph(sanitize_text(text), STYLES["CalloutText"])
    ]
    t = Table([[content]], colWidths=[USABLE_WIDTH])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_color),
        ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("LINEBEFORE", (0, 0), (0, -1), 3.5, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def code_box(code_str: str) -> Table:
    """
    Formats monospace code or architecture diagrams inside a clean box.
    Uses multi-row table architecture so long blocks can paginate across pages naturally!
    """
    lines = code_str.strip().split("\n")
    row_data = []
    for line in lines:
        clean_line = sanitize_text(line).replace(" ", "&nbsp;")
        if not clean_line.strip():
            clean_line = "&nbsp;"
        p_line = Paragraph(clean_line, STYLES["CodeText"])
        row_data.append([p_line])

    t = Table(row_data, colWidths=[USABLE_WIDTH])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.75, COLOR_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 0.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def create_table(headers, data, col_widths=None) -> Table:
    """Creates a beautifully styled data table with headers and zebra striping."""
    if not col_widths:
        w = USABLE_WIDTH / len(headers)
        col_widths = [w] * len(headers)

    header_cells = [Paragraph(f"<b>{sanitize_text(h)}</b>", STYLES["TableHeader"]) for h in headers]
    table_rows = [header_cells]

    for row in data:
        row_cells = []
        for cell in row:
            if isinstance(cell, str):
                row_cells.append(Paragraph(sanitize_text(cell), STYLES["TableCell"]))
            else:
                row_cells.append(cell)
        table_rows.append(row_cells)

    t = Table(table_rows, colWidths=col_widths, repeatRows=1)
    ts = [
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]

    for i in range(1, len(table_rows)):
        if i % 2 == 0:
            ts.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F8FAFC")))
        else:
            ts.append(("BACKGROUND", (0, i), (-1, i), colors.white))

    t.setStyle(TableStyle(ts))
    return t
