from __future__ import annotations

from importlib import import_module
from io import BytesIO
import re
import textwrap
from typing import Literal

_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename(name: str) -> str:
    cleaned = _FILENAME_RE.sub("-", name.strip().lower()).strip("-")
    return cleaned or "saulgpt-report"


def _build_pdf(title: str, content: str) -> bytes:
    reportlab_canvas = import_module("reportlab.pdfgen.canvas")
    reportlab_pagesizes = import_module("reportlab.lib.pagesizes")

    buffer = BytesIO()
    canvas = reportlab_canvas.Canvas(buffer, pagesize=reportlab_pagesizes.A4)
    width, height = reportlab_pagesizes.A4
    margin_x = 48
    margin_y = 56

    text_obj = canvas.beginText(margin_x, height - margin_y)
    text_obj.setFont("Helvetica-Bold", 14)
    text_obj.textLine(title)
    text_obj.moveCursor(0, 10)
    text_obj.setFont("Helvetica", 11)

    def flush_page() -> None:
        nonlocal text_obj
        canvas.drawText(text_obj)
        canvas.showPage()
        text_obj = canvas.beginText(margin_x, height - margin_y)
        text_obj.setFont("Helvetica", 11)

    max_chars = max(50, int((width - (2 * margin_x)) / 6))
    for paragraph in content.splitlines():
        wrapped = textwrap.wrap(paragraph, width=max_chars) or [""]
        for line in wrapped:
            if text_obj.getY() <= margin_y:
                flush_page()
            text_obj.textLine(line)
        if text_obj.getY() <= margin_y:
            flush_page()
        text_obj.textLine("")

    canvas.drawText(text_obj)
    canvas.save()
    return buffer.getvalue()


def _build_docx(title: str, content: str) -> bytes:
    docx_module = import_module("docx")
    document_class = getattr(docx_module, "Document")

    document = document_class()
    document.add_heading(title, level=1)
    for paragraph in content.split("\n\n"):
        text = paragraph.strip()
        if text:
            document.add_paragraph(text)

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def build_report_export(
    title: str,
    content: str,
    fmt: Literal["pdf", "docx"],
    filename: str | None = None,
) -> tuple[bytes, str, str]:
    export_title = title.strip() or "SaulGPT Report"
    base_filename = _sanitize_filename(filename or export_title)

    if fmt == "pdf":
        return (
            _build_pdf(export_title, content),
            "application/pdf",
            f"{base_filename}.pdf",
        )
    if fmt == "docx":
        return (
            _build_docx(export_title, content),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            f"{base_filename}.docx",
        )
    raise ValueError("Unsupported format. Use pdf or docx.")
