"""Tests for PDF loading and validation."""

from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from rag.loader import load_pdf


def test_loads_valid_pdf(sample_pdf_path: Path) -> None:
    """Load text and one-based page metadata from the sample PDF."""
    documents = load_pdf(sample_pdf_path)

    assert documents
    assert any("Sample PDF" in document.page_content for document in documents)
    assert documents[0].metadata["page"] == 1


def test_missing_file_raises_value_error(tmp_path: Path) -> None:
    """Reject a path that does not exist."""
    with pytest.raises(ValueError, match="does not exist"):
        load_pdf(tmp_path / "missing.pdf")


def test_non_pdf_raises_value_error(tmp_path: Path) -> None:
    """Reject files without a PDF extension."""
    text_file = tmp_path / "document.txt"
    text_file.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a PDF"):
        load_pdf(text_file)


def test_pdf_without_extractable_text_raises_value_error(tmp_path: Path) -> None:
    """Reject a valid PDF page containing no selectable text."""
    scanned_pdf = tmp_path / "scanned.pdf"
    pdf = canvas.Canvas(str(scanned_pdf))
    pdf.showPage()
    pdf.save()

    with pytest.raises(ValueError, match="no extractable text"):
        load_pdf(scanned_pdf)
