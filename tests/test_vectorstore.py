"""Tests for Chroma construction and retrieval."""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.vectorstore import build_vectorstore


class KeywordEmbeddings(Embeddings):
    """Map two known terms to deterministic orthogonal vectors."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Create deterministic vectors for the supplied texts."""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Create a deterministic vector for a query."""
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        """Map text mentioning apple or ocean to a fixed unit vector."""
        if "apple" in text.lower():
            return [1.0, 0.0]
        return [0.0, 1.0]


def test_builds_store_and_retrieves_relevant_chunk() -> None:
    """Build a cosine store and retrieve the expected known-query match."""
    chunks = [
        Document(page_content="An apple is a fruit.", metadata={"page": 1}),
        Document(page_content="The ocean is vast.", metadata={"page": 2}),
    ]

    store = build_vectorstore(chunks, embeddings=KeywordEmbeddings())
    results = store.similarity_search("apple", k=1)

    assert results[0].page_content == "An apple is a fruit."
    assert results[0].metadata["page"] == 1


def test_empty_chunks_are_rejected() -> None:
    """Reject an empty collection instead of constructing a useless store."""
    try:
        build_vectorstore([], embeddings=KeywordEmbeddings())
    except ValueError as exc:
        assert "At least one" in str(exc)
    else:
        raise AssertionError("Expected an empty chunk collection to fail")
