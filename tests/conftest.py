"""Shared fixtures for the rag-pdf-chat test suite."""

from pathlib import Path

import pytest


@pytest.fixture
def sample_pdf_path() -> Path:
    """Return the generated text PDF used by loader tests."""
    return Path(__file__).parent / "fixtures" / "sample.pdf"
