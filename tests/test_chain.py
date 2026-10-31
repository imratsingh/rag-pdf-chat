"""Tests for grounded answer generation."""

import os

import pytest
from langchain_core.documents import Document
from pytest_mock import MockerFixture

import rag.chain as chain_module
from rag.chain import answer
from rag.config import (
    EMPTY_QUESTION_MESSAGE,
    MIN_SCORE,
    MODEL_ERROR_MESSAGE,
    NO_ANSWER_MESSAGE,
)


class SearchStore:
    """Return configured scored documents for chain tests."""

    def __init__(self, results: list[tuple[Document, float]]) -> None:
        """Keep the results that retrieval should return."""
        self.results = results
        self.calls = 0

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        """Return the configured search results."""
        self.calls += 1
        return self.results[:k]


def _good_store() -> SearchStore:
    """Create a store returning a single relevant source document."""
    return SearchStore(
        [(Document(page_content="The project began in 2020.", metadata={"page": 3}), 0.9)]
    )


def test_empty_questions_return_prompt_without_api_calls(
    mocker: MockerFixture,
) -> None:
    """Reject empty and whitespace questions before retrieval or LLM calls."""
    store = mocker.Mock()
    qa_model = mocker.Mock()

    assert answer("", store, qa_model) == (EMPTY_QUESTION_MESSAGE, [])
    assert answer("   \t", store, qa_model) == (EMPTY_QUESTION_MESSAGE, [])
    store.similarity_search_with_relevance_scores.assert_not_called()
    qa_model.assert_not_called()


def test_irrelevant_results_return_no_answer_without_llm(
    mocker: MockerFixture,
) -> None:
    """Skip model invocation when all retrieved scores are below threshold."""
    store = SearchStore(
        [(Document(page_content="Unrelated text.", metadata={"page": 1}), MIN_SCORE - 0.01)]
    )
    qa_model = mocker.Mock()

    assert answer("What is the answer?", store, qa_model) == (NO_ANSWER_MESSAGE, [])
    qa_model.assert_not_called()


def test_relevant_results_include_context_and_return_sources(
    mocker: MockerFixture,
) -> None:
    """Pass retrieved context to the model and report its source page."""
    store = _good_store()
    qa_model = mocker.Mock(
        return_value={
            "answer": "The project began in 2020.",
            "score": 0.9,
            "start": 0,
            "end": 28,
        }
    )

    response, sources = answer("When did the project begin?", store, qa_model)

    assert response == "The project began in 2020."
    assert sources == [3]
    qa_model.assert_called_once_with(
        question="When did the project begin?",
        context="The project began in 2020.",
    )


def test_low_cosine_scores_use_matching_keywords(
    mocker: MockerFixture,
) -> None:
    """Keep an exact-term passage when local embedding scores are below threshold."""
    document = Document(
        page_content="The Challenge: replies slow down and order details get lost.",
        metadata={"page": 1},
    )
    store = SearchStore([(document, 0.07)])
    qa_model = mocker.Mock(
        return_value={
            "answer": "Replies slow down and order details get lost.",
            "score": 0.8,
            "start": 0,
            "end": 46,
        }
    )

    response, sources = answer("The Challenge details", store, qa_model)

    assert response == "Replies slow down and order details get lost."
    assert sources == [1]
    qa_model.assert_called_once()


def test_heading_question_returns_matching_pdf_section_without_llm(
    mocker: MockerFixture,
) -> None:
    """Return all text under a matching heading directly from the PDF."""
    document = Document(
        page_content=(
            "Introduction\nOpening details.\n"
            "The Challenge\nReplies slow down and order details are lost.\n"
            "What We Built\nThe bakery uses an order dashboard."
        ),
        metadata={"page": 1},
    )
    store = SearchStore([(document, 0.07)])
    qa_model = mocker.Mock()

    response, sources = answer("The Challenge give", store, qa_model)

    assert response == "Replies slow down and order details are lost."
    assert sources == [1]
    qa_model.assert_not_called()


def test_heading_extraction_stops_at_uppercase_heading(
    mocker: MockerFixture,
) -> None:
    """Do not append the following all-caps section to a heading answer."""
    document = Document(
        page_content=(
            "PROFESSIONAL SUMMARY\n"
            "AI/ML Engineer with experience building automation.\n"
            "PROFESSIONAL EXPERIENCE\n"
            "Intern AI Developer at Example."
        ),
        metadata={"page": 1},
    )
    store = SearchStore([(document, 0.1)])
    qa_model = mocker.Mock()

    response, sources = answer("What is the professional summary?", store, qa_model)

    assert response == "AI/ML Engineer with experience building automation."
    assert sources == [1]
    qa_model.assert_not_called()


def test_model_comparison_question_returns_evidence_sentence(
    mocker: MockerFixture,
) -> None:
    """Return all compared model names instead of the nearby project heading."""
    document = Document(
        page_content=(
            "Customer Churn Analytics — End-to-End Data Science Project\n"
            "Built and compared 3 predictive models (Logistic Regression: 77.25%, "
            "Random Forest: 76%, Deep Learning: 77.25% accuracy)."
        ),
        metadata={"page": 1},
    )
    store = SearchStore([(document, 0.7)])
    qa_model = mocker.Mock()

    response, sources = answer(
        "Which three models did they compare in the churn prediction project?",
        store,
        qa_model,
    )

    assert response == (
        "Built and compared 3 predictive models (Logistic Regression: 77.25%, "
        "Random Forest: 76%, Deep Learning: 77.25% accuracy)."
    )
    assert sources == [1]
    qa_model.assert_not_called()


def test_degree_and_university_question_returns_education_entry(
    mocker: MockerFixture,
) -> None:
    """Retrieve a low-scoring education chunk and return its credential entry."""
    document = Document(
        page_content=(
            "EDUCATION\n"
            "B.Tech, Computer Science and Engineering — RGPV, Bhopal\n"
            "Jun 2022 – Jun 2026\n"
            "CGPA: 7.67 | First Division with Honours"
        ),
        metadata={"page": 1},
    )
    store = SearchStore([(document, 0.24)])
    qa_model = mocker.Mock()

    response, sources = answer(
        "What degree did they earn, and from which university?",
        store,
        qa_model,
    )

    assert response == "B.Tech, Computer Science and Engineering — RGPV, Bhopal"
    assert sources == [1]
    qa_model.assert_not_called()


def test_heading_match_ignores_generic_shared_words(
    mocker: MockerFixture,
) -> None:
    """Do not match a bakery heading to an unrelated owner-name question."""
    document = Document(
        page_content="What the Bakery Can Do\nTake orders and send review requests.",
        metadata={"page": 1},
    )
    store = SearchStore([(document, 0.07)])
    qa_model = mocker.Mock(
        return_value={
            "answer": "Take orders",
            "score": 0.001,
            "start": 0,
            "end": 11,
        }
    )

    response, sources = answer("What is the bakery owner name?", store, qa_model)

    assert response == NO_ANSWER_MESSAGE
    assert sources == []
    qa_model.assert_called_once()


def test_low_qa_confidence_returns_exact_fallback(
    mocker: MockerFixture,
) -> None:
    """Reject an answer span with confidence below the configured threshold."""
    store = _good_store()
    qa_model = mocker.Mock(
        return_value={
            "answer": "2020",
            "score": 0.01,
            "start": 25,
            "end": 29,
        }
    )

    response, sources = answer("What skills does it mention?", store, qa_model)

    assert response == NO_ANSWER_MESSAGE
    assert sources == []


def test_transient_error_is_retried_and_then_succeeds(
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry a transient QA error and return a later successful response."""
    monkeypatch.setattr(chain_module._invoke_qa.retry, "sleep", lambda _: None)
    store = _good_store()
    qa_model = mocker.Mock()
    qa_model.side_effect = [
        ConnectionError("temporary local model failure"),
        ConnectionError("temporary local model failure"),
        {"answer": "The project began in 2020.", "score": 0.9, "start": 0, "end": 28},
    ]

    response, sources = answer("When did the project begin?", store, qa_model)

    assert response == "The project began in 2020."
    assert sources == [3]
    assert qa_model.call_count == 3


def test_final_transient_error_returns_friendly_error(
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a friendly message after all five QA retries fail."""
    monkeypatch.setattr(chain_module._invoke_qa.retry, "sleep", lambda _: None)
    store = _good_store()
    qa_model = mocker.Mock()
    qa_model.side_effect = ConnectionError("local model unavailable")

    response, sources = answer("When did the project begin?", store, qa_model)

    assert response == MODEL_ERROR_MESSAGE
    assert sources == []
    assert qa_model.call_count == 5


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_HF_INTEGRATION") != "1",
    reason="Set RUN_HF_INTEGRATION=1 to download and test local Hugging Face models",
)
def test_optional_local_hugging_face_integration() -> None:
    """Check heading extraction and unsupported questions with real embeddings."""
    from rag.splitter import split_documents
    from rag.vectorstore import build_vectorstore

    documents = [
        Document(
            page_content=(
                "The Challenge\nSmall bakeries get most of their orders by chat. "
                "When the shop is busy, replies slow down, order details (name, "
                "item, date, time) get lost in long conversations, and after "
                "delivery nobody asks for a review or follows up with an offer.\n"
                "What We Built\nThe bakery uses an order dashboard."
            ),
            metadata={"page": 1},
        )
    ]
    store = build_vectorstore(split_documents(documents))
    response, sources = answer("The Challenge give", store)

    assert "replies slow down" in response.lower()
    assert "order details" in response.lower()
    assert "get lost" in response.lower()
    assert sources == [1]

    unsupported_response, unsupported_sources = answer(
        "What is the bakery owner's name?",
        store,
    )
    assert unsupported_response == NO_ANSWER_MESSAGE
    assert unsupported_sources == []
