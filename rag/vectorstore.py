"""Chroma vector store construction with retrying embeddings."""

import logging
from collections.abc import Sequence
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)
_TRANSIENT_ERRORS = (TimeoutError, ConnectionError)


@lru_cache(maxsize=1)
def create_local_embeddings() -> HuggingFaceEmbeddings:
    """Load and cache the configured local Hugging Face embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


class RetryingEmbeddings(Embeddings):
    """Retry transient failures from an embeddings implementation."""

    def __init__(self, embeddings: Embeddings) -> None:
        """Store the embeddings implementation to wrap."""
        self._embeddings = embeddings

    @retry(
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=2),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents, retrying transient API failures."""
        return self._embeddings.embed_documents(texts)

    @retry(
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=2),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def embed_query(self, text: str) -> list[float]:
        """Embed a query, retrying transient API failures."""
        return self._embeddings.embed_query(text)


def build_vectorstore(
    chunks: Sequence[Document],
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Build an in-memory cosine-similarity Chroma store from document chunks."""
    if not chunks:
        raise ValueError("At least one document chunk is required.")

    if embeddings is None:
        embeddings = create_local_embeddings()

    retrying_embeddings = RetryingEmbeddings(embeddings)
    try:
        return Chroma.from_documents(
            documents=list(chunks),
            embedding=retrying_embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
    except _TRANSIENT_ERRORS:
        logger.exception("Local embedding model failed after retry attempts")
        raise
