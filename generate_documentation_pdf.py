# generate_documentation_pdf.py
"""Master build script for CareAI Complete Technical Documentation PDF."""

import os
import sys
import time

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, PageBreak
from reportlab.pdfgen import canvas
from reportlab.lib import colors

from pdf_sections_part1 import (
    build_cover_and_toc, build_section_1, build_section_2,
    build_section_3, build_section_4, build_section_5
)
from pdf_sections_part2 import (
    build_section_6, build_section_7, build_section_8,
    build_section_9, build_section_10
)
from pdf_sections_part3 import (
    build_section_11, build_section_12, build_section_13, build_section_14
)
from pdf_sections_part4 import (
    build_section_15, build_section_16
)


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas that computes total pages and prints running headers and footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return  # Suppress running header and footer on the cover page

        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#475569"))

        # Running Top Header
        self.drawString(54, 752, "CareAI: Dynamic Mental Health System | Technical Documentation")
        self.setFont("Helvetica", 7.5)
        self.drawRightString(letter[0] - 54, 752, "Engineering Reference & Architecture Specification")

        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 744, letter[0] - 54, 744)

        # Running Bottom Footer
        self.line(54, 46, letter[0] - 54, 46)
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(54, 34, "CareAI Academic & Engineering Handbook | Semester 7 LLM Project")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 34, page_str)
        self.restoreState()


def generate_pdf():
    output_pdf = "Complete_Project_Technical_Documentation.pdf"
    print(f"[1/4] Initializing PDF document builder for: {output_pdf}")

    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    story = []

    print("[2/4] Assembling content sections...")
    # Cover & Table of Contents
    story.extend(build_cover_and_toc())

    # Section 1: Overview
    story.extend(build_section_1())
    story.append(PageBreak())

    # Section 2: Architecture
    story.extend(build_section_2())
    story.append(PageBreak())

    # Section 3: Tech Stack
    story.extend(build_section_3())
    story.append(PageBreak())

    # Section 4: Module-by-Module
    story.extend(build_section_4())
    story.append(PageBreak())

    # Section 5: Working Mechanism
    story.extend(build_section_5())
    story.append(PageBreak())

    # Section 6: AI/ML & Algorithms
    story.extend(build_section_6())
    story.append(PageBreak())

    # Section 7: Database Architecture
    story.extend(build_section_7())
    story.append(PageBreak())

    # Section 8: API & Backend
    story.extend(build_section_8())
    story.append(PageBreak())

    # Section 9: Frontend Architecture
    story.extend(build_section_9())
    story.append(PageBreak())

    # Section 10: Folder Structure
    story.extend(build_section_10())
    story.append(PageBreak())

    # Section 11: Installation & Setup
    story.extend(build_section_11())
    story.append(PageBreak())

    # Section 12: Security & Performance
    story.extend(build_section_12())
    story.append(PageBreak())

    # Section 13: Challenges & Solutions
    story.extend(build_section_13())
    story.append(PageBreak())

    # Section 14: Advantages & Future Scope
    story.extend(build_section_14())
    story.append(PageBreak())

    # Section 15: Technical Interview Handbook
    story.extend(build_section_15())
    story.append(PageBreak())

    # Section 16: Final Summary & Pitches
    story.extend(build_section_16())

    print(f"[3/4] Building PDF with {len(story)} flowable elements...")
    start_time = time.time()
    doc.build(story, canvasmaker=NumberedCanvas)
    elapsed = time.time() - start_time

    file_size_kb = os.path.getsize(output_pdf) / 1024
    print(f"[4/4] SUCCESS! Generated '{output_pdf}' ({file_size_kb:.1f} KB) in {elapsed:.2f} seconds.")
    return output_pdf


if __name__ == "__main__":
    generate_pdf()
