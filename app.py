"""Streamlit user interface for chatting with an uploaded PDF."""

import hashlib
import logging
import tempfile
from pathlib import Path

import streamlit as st

from rag.chain import answer
from rag.config import MODEL_ERROR_MESSAGE
from rag.loader import load_pdf
from rag.splitter import split_documents
from rag.vectorstore import build_vectorstore

logger = logging.getLogger(__name__)


def _reset_document_state() -> None:
    """Clear the active PDF and its conversation from the current session."""
    st.session_state.vectorstore = None
    st.session_state.document_hash = None
    st.session_state.chat_history = []


def _render_history() -> None:
    """Render saved user and assistant messages with source pages."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message["sources"]:
                pages = ", ".join(str(page) for page in message["sources"])
                st.caption(f"Source pages: {pages}")


def main() -> None:
    """Render the PDF upload and grounded chat interface."""
    st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
    st.title("Chat with your PDF")

    if "vectorstore" not in st.session_state:
        _reset_document_state()

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is None:
        st.info("Upload a PDF to start a conversation.")
        return

    pdf_bytes = uploaded_file.getvalue()
    current_hash = hashlib.sha256(pdf_bytes).hexdigest()
    if current_hash != st.session_state.document_hash:
        _reset_document_state()
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                pdf_path = Path(temporary_directory) / "upload.pdf"
                pdf_path.write_bytes(pdf_bytes)
                documents = load_pdf(pdf_path)
        except ValueError as exc:
            st.error(str(exc))
            return

        try:
            chunks = split_documents(documents)
            st.session_state.vectorstore = build_vectorstore(chunks)
            st.session_state.document_hash = current_hash
        except Exception:
            logger.exception("Could not prepare uploaded PDF")
            st.error(MODEL_ERROR_MESSAGE)
            return

    _render_history()
    question = st.chat_input("Ask a question about your PDF")
    if not question:
        return

    st.session_state.chat_history.append(
        {"role": "user", "content": question, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Looking through your PDF..."):
            try:
                response, sources = answer(question, st.session_state.vectorstore)
            except Exception:
                logger.exception("Could not answer question from uploaded PDF")
                response, sources = MODEL_ERROR_MESSAGE, []
        st.markdown(response)
        if sources:
            st.caption("Source pages: " + ", ".join(str(page) for page in sources))
    st.session_state.chat_history.append(
        {"role": "assistant", "content": response, "sources": sources}
    )


if __name__ == "__main__":
    main()
