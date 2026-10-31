"""Tests for recursive document splitting."""

from langchain_core.documents import Document

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE
from rag.splitter import split_documents


def test_chunks_do_not_exceed_configured_size() -> None:
    """Keep every generated chunk within the configured maximum."""
    chunks = split_documents([Document(page_content="x" * 2500)])

    assert chunks
    assert all(len(chunk.page_content) <= CHUNK_SIZE for chunk in chunks)


def test_chunks_have_configured_overlap() -> None:
    """Retain the configured overlap between adjacent chunks."""
    chunks = split_documents([Document(page_content="x" * 2500)])

    assert len(chunks) > 1
    assert chunks[0].page_content[-CHUNK_OVERLAP:] == chunks[1].page_content[:CHUNK_OVERLAP]


def test_empty_input_returns_empty_list() -> None:
    """Return no chunks for an empty document collection."""
    assert split_documents([]) == []
