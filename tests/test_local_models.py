"""Tests for configuring local Hugging Face model factories."""

from pytest_mock import MockerFixture

import rag.chain as chain_module
import rag.vectorstore as vectorstore_module
from rag.config import EMBEDDING_MODEL, QA_MODEL


def test_local_embedding_factory_uses_configured_hugging_face_model(
    mocker: MockerFixture,
) -> None:
    """Create local embeddings with the configured model and normalized vectors."""
    factory = mocker.patch(
        "rag.vectorstore.HuggingFaceEmbeddings",
        return_value=mocker.Mock(),
    )
    vectorstore_module.create_local_embeddings.cache_clear()

    vectorstore_module.create_local_embeddings()

    factory.assert_called_once_with(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore_module.create_local_embeddings.cache_clear()


def test_local_qa_factory_uses_configured_hugging_face_model(
    mocker: MockerFixture,
) -> None:
    """Create the configured extractive question-answering pipeline."""
    factory = mocker.patch(
        "rag.chain.pipeline",
        return_value=mocker.Mock(),
    )
    chain_module.create_local_qa_pipeline.cache_clear()

    chain_module.create_local_qa_pipeline()

    factory.assert_called_once_with(
        "question-answering",
        model=QA_MODEL,
        device=-1,
    )
    chain_module.create_local_qa_pipeline.cache_clear()
