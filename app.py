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

APP_STYLES = """
<style>
    :root {
        color-scheme: dark;
        --ink: #f4f5fb;
        --muted: #9aa3b5;
        --panel: rgba(17, 23, 36, 0.88);
        --line: rgba(156, 173, 202, 0.15);
    }

    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(80, 68, 153, 0.20), transparent 34rem),
            radial-gradient(ellipse at 100% 20%, rgba(48, 105, 139, 0.12), transparent 28rem),
            #090c13;
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"], footer { visibility: hidden; }
    .block-container {
        max-width: 1040px;
        padding: 2rem 2rem 4rem;
    }
    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4.5rem;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        color: var(--ink);
        font-size: 0.97rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    .brand-icon {
        display: grid;
        width: 2.25rem;
        height: 2.25rem;
        place-items: center;
        border: 1px solid rgba(188, 177, 255, 0.35);
        border-radius: 0.75rem;
        background: linear-gradient(145deg, #302a59, #171b2b);
        color: #c9c0ff;
        font-size: 1.1rem;
    }
    .privacy-badge, .eyebrow {
        color: #b7adff;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }
    .privacy-badge {
        padding: 0.48rem 0.75rem;
        border: 1px solid rgba(156, 140, 255, 0.24);
        border-radius: 999px;
        background: rgba(96, 79, 181, 0.10);
        letter-spacing: 0.06em;
        text-transform: none;
    }
    .hero { margin-bottom: 2rem; }
    .hero h1 {
        margin: 0.55rem 0 0.65rem;
        color: var(--ink);
        font-size: clamp(2.2rem, 5vw, 3.4rem);
        font-weight: 750;
        letter-spacing: -0.055em;
        line-height: 1.08;
    }
    .hero p, .section-description {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.65;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 1.15rem;
        background: var(--panel);
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.18);
    }
    .panel-heading {
        margin: 0 0 0.25rem;
        color: var(--ink);
        font-size: 1rem;
        font-weight: 650;
    }
    .section-description { margin: 0 0 1rem; font-size: 0.88rem; }
    [data-testid="stFileUploader"] {
        padding: 0.45rem;
        border: 1px dashed rgba(167, 154, 255, 0.38);
        border-radius: 0.85rem;
        background: rgba(116, 99, 204, 0.06);
    }
    [data-testid="stFileUploader"] section {
        border: 0;
        background: transparent;
    }
    [data-testid="stFileUploader"] button,
    [data-testid="stChatInput"] {
        border-radius: 0.8rem;
    }
    [data-testid="stFileUploader"] button {
        border: 1px solid rgba(180, 169, 255, 0.35);
        background: rgba(119, 101, 218, 0.18);
        color: var(--ink);
    }
    [data-testid="stFileUploader"] button:hover {
        border-color: #b5a8ff;
        color: white;
    }
    .upload-hint {
        margin: 0.75rem 0 0;
        color: var(--muted);
        font-size: 0.78rem;
    }
    .document-status {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin: 1.15rem 0 0;
        padding: 0.8rem 0.9rem;
        border: 1px solid rgba(100, 204, 164, 0.19);
        border-radius: 0.75rem;
        background: rgba(39, 130, 96, 0.10);
        color: #bcebd7;
        font-size: 0.88rem;
    }
    .chat-panel-heading {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--line);
    }
    .ready-label { color: #8dd9b8; font-size: 0.76rem; }
    .welcome-card {
        margin: 1.3rem 0 0.4rem;
        padding: 1.15rem;
        border: 1px solid rgba(156, 140, 255, 0.14);
        border-radius: 0.9rem;
        background: rgba(108, 91, 190, 0.08);
    }
    .welcome-card strong { color: var(--ink); }
    .welcome-card p {
        margin: 0.35rem 0 0;
        color: var(--muted);
        font-size: 0.87rem;
        line-height: 1.6;
    }
    [data-testid="stChatMessage"] {
        margin-top: 0.8rem;
        padding: 0.95rem 1rem;
        border: 1px solid var(--line);
        border-radius: 0.9rem;
        background: rgba(22, 28, 43, 0.75);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        border-color: rgba(156, 140, 255, 0.17);
        background: rgba(96, 79, 181, 0.11);
    }
    [data-testid="stChatInput"] {
        border: 1px solid rgba(156, 140, 255, 0.24);
        background: #111624;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(181, 168, 255, 0.8);
        box-shadow: 0 0 0 1px rgba(181, 168, 255, 0.18);
    }
    [data-testid="stCaptionContainer"] { color: #a9b5ca; }
    @media (max-width: 640px) {
        .block-container { padding: 1.2rem 1rem 3rem; }
        .brand-bar { margin-bottom: 3rem; }
    }
</style>
"""


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
    st.markdown(APP_STYLES, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="brand-bar">
            <div class="brand"><span class="brand-icon">▤</span> Papertrail</div>
            <div class="privacy-badge">● &nbsp;Private, local AI</div>
        </div>
        <div class="hero">
            <div class="eyebrow">Your document, understood</div>
            <h1>Have a conversation<br>with your PDF.</h1>
            <p>Drop in a document, ask what matters, and get answers grounded in its pages.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "vectorstore" not in st.session_state:
        _reset_document_state()

    with st.container(border=True):
        st.markdown(
            """
            <div class="panel-heading">Add your document</div>
            <p class="section-description">Choose a text-based PDF to start asking questions.</p>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            label_visibility="collapsed",
            key="pdf_upload",
        )
        st.markdown(
            '<p class="upload-hint">PDF only · Your document is processed on this computer.</p>',
            unsafe_allow_html=True,
        )
    if uploaded_file is None:
        st.markdown(
            """
            <div class="chat-panel">
                <div class="chat-panel-heading">
                    <div class="panel-heading">Your conversation</div>
                    <div class="privacy-badge">Waiting for a PDF</div>
                </div>
                <div class="welcome-card">
                    <strong>Start with a document</strong>
                    <p>Once your PDF is ready, ask about its key ideas, people, dates, or any specific detail.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
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

    st.markdown(
        '<div class="document-status"><span>✓</span><span><strong>PDF ready</strong></span></div>',
        unsafe_allow_html=True,
    )
    st.caption(uploaded_file.name)
    with st.container(border=True):
        st.markdown(
            """
            <div class="chat-panel-heading">
                <div class="panel-heading">Your conversation</div>
                <div class="ready-label">● Document ready</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not st.session_state.chat_history:
            st.markdown(
                """
                <div class="welcome-card">
                    <strong>What would you like to know?</strong>
                    <p>Ask a question in the box below. Answers include page references, and the app will say when it cannot find evidence in the PDF.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
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
