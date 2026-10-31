"""Tests for retrieval embedding failures."""

from pytest_mock import MockerFixture

from rag.chain import answer
from rag.config import MODEL_ERROR_MESSAGE


def test_query_embedding_failure_returns_friendly_error(
    mocker: MockerFixture,
) -> None:
    """Return a friendly error if query embedding retries are exhausted."""
    store = mocker.Mock()
    store.similarity_search_with_relevance_scores.side_effect = TimeoutError()

    assert answer("A question", store) == (MODEL_ERROR_MESSAGE, [])
