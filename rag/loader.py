"""PDF loading and validation."""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf(path: str | Path) -> list[Document]:
    """Load a text-based PDF and normalize its page metadata to one-based pages."""
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise ValueError(f"PDF file does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The uploaded file must be a PDF.")

    try:
        documents = PyPDFLoader(str(pdf_path)).load()
    except Exception as exc:
        logger.exception("Could not read PDF file %s", pdf_path)
        raise ValueError("The file could not be read as a valid PDF.") from exc

    if not any(document.page_content.strip() for document in documents):
        raise ValueError("The PDF contains no extractable text.")

    for document in documents:
        page = document.metadata.get("page")
        if isinstance(page, int):
            document.metadata["page"] = page + 1
    return documents
