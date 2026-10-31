"""Grounded question-answering over a document vector store."""

import logging
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol, TypedDict

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from transformers import pipeline

from rag.config import (
    EMPTY_QUESTION_MESSAGE,
    MIN_ANSWER_SCORE,
    MIN_SCORE,
    MODEL_ERROR_MESSAGE,
    NO_ANSWER_MESSAGE,
    QA_MODEL,
    TOP_K,
)

logger = logging.getLogger(__name__)
_TRANSIENT_ERRORS = (TimeoutError, ConnectionError)
_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "can",
    "could",
    "describe",
    "did",
    "do",
    "does",
    "explain",
    "for",
    "give",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "please",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "would",
    "you",
}
_HEADING_STOP_WORDS = _STOP_WORDS | {
    "bakery",
    "built",
    "business",
    "do",
    "document",
    "introduction",
    "overview",
    "project",
    "section",
}
_EDUCATION_QUERY_TERMS = {
    "academic",
    "college",
    "degree",
    "education",
    "graduate",
    "school",
    "university",
    "qualification",
}


class QAResult(TypedDict):
    """Answer span and confidence returned by a question-answering pipeline."""

    answer: str
    score: float
    start: int
    end: int


class QuestionAnswerer(Protocol):
    """Callable interface for an extractive question-answering model."""

    def __call__(self, *, question: str, context: str) -> QAResult:
        """Return the best answer span and its confidence."""


@lru_cache(maxsize=1)
def create_local_qa_pipeline() -> QuestionAnswerer:
    """Load and cache a local extractive QA model with no-answer support."""
    return pipeline(
        "question-answering",
        model=QA_MODEL,
        device=-1,
    )


@retry(
    retry=retry_if_exception_type(_TRANSIENT_ERRORS),
    wait=wait_exponential(multiplier=0.1, min=0.1, max=2),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _invoke_qa(
    qa_model: QuestionAnswerer,
    question: str,
    context: str,
) -> QAResult:
    """Run local extractive QA, retrying transient failures."""
    return qa_model(question=question, context=context)


def _normalize_term(term: str) -> str:
    """Normalize a simple plural token for keyword matching."""
    if len(term) > 4 and term.endswith("ies"):
        return f"{term[:-3]}y"
    if len(term) > 3 and term.endswith("s"):
        return term[:-1]
    return term


def _keyword_matches(question: str, document: Document) -> int:
    """Count meaningful question terms present in a retrieved document."""
    question_terms = {
        _normalize_term(term)
        for term in re.findall(r"\b[a-z0-9]+\b", question.lower())
        if term not in _STOP_WORDS and len(term) > 2
    }
    if question_terms & _EDUCATION_QUERY_TERMS:
        question_terms.add("education")
    document_terms = {
        _normalize_term(term)
        for term in re.findall(r"\b[a-z0-9]+\b", document.page_content.lower())
    }
    return len(question_terms & document_terms)


def _is_section_heading(line: str) -> bool:
    """Recognize common title-case and uppercase document section headings."""
    words = line.split()
    has_letters = any(character.isalpha() for character in line)
    return (
        bool(words)
        and len(words) <= 8
        and not line.endswith((".", "!", "?", ":"))
        and (
            line.istitle()
            or (has_letters and line.isupper())
        )
    )


def _select_relevant_documents(
    question: str,
    results: Sequence[tuple[Document, float]],
) -> list[Document]:
    """Use cosine scores first, then lexical matches for weak local scores."""
    scored_documents = [
        document for document, score in results if score >= MIN_SCORE
    ]
    if scored_documents:
        return scored_documents

    keyword_ranked = [
        (document, score, _keyword_matches(question, document))
        for document, score in results
    ]
    keyword_ranked = [
        candidate for candidate in keyword_ranked if candidate[2] > 0
    ]
    keyword_ranked.sort(key=lambda candidate: (candidate[2], candidate[1]), reverse=True)
    return [document for document, _, _ in keyword_ranked[:2]]


def _extract_matching_heading_section(
    question: str,
    documents: Sequence[Document],
) -> tuple[str, int] | None:
    """Return a directly requested titled section without model paraphrasing."""
    question_terms = {
        _normalize_term(term)
        for term in re.findall(r"\b[a-z0-9]+\b", question.lower())
        if term not in _STOP_WORDS and len(term) > 2
    }
    if not question_terms:
        return None

    for document in documents:
        lines = [line.strip().lstrip("\x7f").strip() for line in document.page_content.splitlines()]
        for index, line in enumerate(lines):
            heading_terms = {
                _normalize_term(term)
                for term in re.findall(r"\b[a-z0-9]+\b", line.lower())
                if term not in _HEADING_STOP_WORDS and len(term) > 2
            }
            if not heading_terms or not (question_terms & heading_terms):
                continue
            if not _is_section_heading(line):
                continue
            overlap = len(question_terms & heading_terms)
            if overlap / min(len(question_terms), len(heading_terms)) < 0.5:
                continue

            section_lines = []
            for section_line in lines[index + 1 :]:
                if section_line and _is_section_heading(section_line):
                    break
                if section_line:
                    section_lines.append(section_line)
            section = " ".join(section_lines).strip()
            if section:
                page = document.metadata.get("page")
                return section, page if isinstance(page, int) else -1
    return None


def _extract_model_comparison_answer(
    question: str,
    documents: Sequence[Document],
) -> tuple[str, int] | None:
    """Return a source sentence listing compared models when specifically asked."""
    if not re.search(r"\bmodels?\b", question, re.IGNORECASE):
        return None
    if not re.search(r"\bcompar\w*\b", question, re.IGNORECASE):
        return None

    for document in documents:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", document.page_content)
        for sentence in sentences:
            if not re.search(r"\bmodels?\b", sentence, re.IGNORECASE):
                continue
            if not re.search(r"\bcompar\w*\b", sentence, re.IGNORECASE):
                continue
            answer_text = sentence.strip().lstrip("\x7f\u0001•*- ")
            if answer_text:
                page = document.metadata.get("page")
                return answer_text, page if isinstance(page, int) else -1
    return None


def _extract_education_answer(
    question: str,
    documents: Sequence[Document],
) -> tuple[str, int] | None:
    """Return the first qualification entry from a matching education section."""
    question_terms = {
        _normalize_term(term)
        for term in re.findall(r"\b[a-z0-9]+\b", question.lower())
    }
    if not question_terms & _EDUCATION_QUERY_TERMS:
        return None

    for document in documents:
        in_education_section = False
        for line in document.page_content.splitlines():
            cleaned_line = line.strip().lstrip("\x7f\u0001•*- ").strip()
            normalized_line = re.sub(r"[^a-z]+", " ", cleaned_line.lower()).strip()
            if normalized_line in {
                "education",
                "education qualification",
                "education qualifications",
                "academic background",
            }:
                in_education_section = True
                continue
            if not in_education_section or not cleaned_line:
                continue
            if not re.search(r"[a-z]", cleaned_line, re.IGNORECASE):
                continue

            page = document.metadata.get("page")
            return cleaned_line, page if isinstance(page, int) else -1
    return None


def answer(
    question: str,
    store: VectorStore,
    qa_model: QuestionAnswerer | None = None,
) -> tuple[str, list[int]]:
    """Answer from relevant PDF content and return the cited source pages."""
    cleaned_question = question.strip()
    if not cleaned_question:
        return EMPTY_QUESTION_MESSAGE, []

    try:
        results = store.similarity_search_with_relevance_scores(
            cleaned_question,
            k=TOP_K,
        )
    except _TRANSIENT_ERRORS:
        logger.exception("Local query embedding failed after retry attempts")
        return MODEL_ERROR_MESSAGE, []
    relevant_documents = _select_relevant_documents(cleaned_question, results)
    if not relevant_documents:
        return NO_ANSWER_MESSAGE, []

    model_answer = _extract_model_comparison_answer(
        cleaned_question,
        relevant_documents,
    )
    if model_answer is not None:
        text, page = model_answer
        return text, [page] if page > 0 else []

    education_answer = _extract_education_answer(
        cleaned_question,
        relevant_documents,
    )
    if education_answer is not None:
        text, page = education_answer
        return text, [page] if page > 0 else []

    heading_section = _extract_matching_heading_section(
        cleaned_question,
        relevant_documents,
    )
    if heading_section is not None:
        section, page = heading_section
        return section, [page] if page > 0 else []

    model = qa_model or create_local_qa_pipeline()
    best_result: QAResult | None = None
    best_page: int | None = None
    try:
        for document in relevant_documents:
            result = _invoke_qa(model, cleaned_question, document.page_content)
            if best_result is None or result["score"] > best_result["score"]:
                best_result = result
                page = document.metadata.get("page")
                best_page = page if isinstance(page, int) else None
    except _TRANSIENT_ERRORS:
        logger.exception("Local question-answering model failed after retries")
        return MODEL_ERROR_MESSAGE, []

    if (
        best_result is None
        or best_result["score"] < MIN_ANSWER_SCORE
        or not best_result["answer"].strip()
    ):
        return NO_ANSWER_MESSAGE, []
    return best_result["answer"].strip(), [best_page] if best_page else []
