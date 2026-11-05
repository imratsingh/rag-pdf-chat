# rag-pdf-chat

A Streamlit application for asking questions about a PDF. Answers are grounded
strictly in retrieved PDF text; when that text does not support an answer, the
app responds exactly: **Ye jawab document mein nahi mila.**

## Features

- Custom dark interface with a focused PDF upload area and chat workspace.
- Upload a PDF and chat with its extracted text.
- Cite the one-based PDF page numbers used to answer each question.
- Reject invalid, missing, and text-free (for example, scanned) PDFs.
- Keep each uploaded document's vector store and conversation in Streamlit
  session state; re-uploading the same file does not rebuild the store.
- Use local Hugging Face `sentence-transformers/all-MiniLM-L6-v2` embeddings
  and `deepset/roberta-base-squad2` for extractive question answering.
- Return exact answer spans from PDF text and use the model's no-answer
  confidence to reject unsupported questions.
- Run without an API key; models are downloaded from Hugging Face on first use
  and then run locally.
- Cache the loaded embedding and chat models for reuse.

## Architecture and pipeline

1. `app.py` receives the upload and manages the UI and session state.
2. `rag.loader.load_pdf` extracts page text with LangChain's `PyPDFLoader`,
   validates that text is available, and numbers pages starting at 1.
3. `rag.splitter.split_documents` makes overlapping text chunks while
   preserving page metadata.
4. `rag.vectorstore.build_vectorstore` embeds chunks locally and creates an
   in-memory Chroma collection using cosine distance.
5. `rag.chain.answer` retrieves relevant chunks. If local cosine scores fall
   below their cloud-model threshold, it tries keyword matches. Questions
   matching a PDF heading return that section directly; other questions use
   local extractive QA and a confidence threshold, with source page citations.

## Setup

Requires Python 3.11 or newer.

```powershell
cd rag-pdf-chat
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The app downloads the configured models from Hugging Face the first time they
are needed. You do not need an OpenAI key, Hugging Face token, or `.env` file.
Downloads require an internet connection and additional disk space; local
inference also needs sufficient RAM. The QA pipeline uses the CPU by default,
so answers may take time on slower computers. Transformers is pinned below
version 5 because the question-answering pipeline used here is not available
in Transformers 5.

## Run the app

```powershell
streamlit run app.py
```

Upload a PDF and enter questions in the chat input. The assistant shows source
pages below each answer.

## Run tests

```powershell
pip install -r requirements-dev.txt
pytest -q
```

The default test suite uses a generated local PDF and deterministic fake
embeddings; it does not download or run models. One test is marked
`integration` and is skipped unless `RUN_HF_INTEGRATION=1` is set. Run it
explicitly in PowerShell to download and exercise the real local models:

```powershell
$env:RUN_HF_INTEGRATION = "1"
pytest -q -m integration
```

## Design choices

- **Chunk size 800, overlap 100:** keeps passages small enough for focused
  retrieval while retaining nearby context across chunk boundaries.
- **Top k 4:** supplies a small set of relevant passages to the model without
  unnecessarily expanding the prompt.
- **Minimum relevance score 0.3:** filters weak cosine matches. Local
  MiniLM scores on the sample PDF are lower, so keyword matching recovers
  obvious heading/topic queries when the semantic cutoff is not met.
- **Minimum QA score 0.1:** rejects weak extractive answer spans so unsupported
  questions get the exact no-answer message.
- **Cosine similarity:** a common comparison for normalized semantic
  embeddings and explicitly configured on the Chroma collection.
- **One-based page metadata:** page citations match the page labels readers see
  in PDF viewers.
- **Local Hugging Face models:** model inference happens on this computer,
  rather than through a hosted model API. Model weights are cached locally.
- **In-memory vector store:** keeps the demo simple and avoids persisting
  private document content to disk. The store is rebuilt for a different file
  upload in a session.

## Edge cases handled

- Missing files, non-PDF files, malformed PDFs, and PDFs with no extractable
  text produce clear validation errors.
- Empty and whitespace-only questions return `Pehle koi sawaal likho.` without
  retrieval or API calls.
- No retrieved passages above the score threshold or with matching keywords
  returns the exact no-answer message without running QA.
- Transient model timeouts and connection failures are retried up to five
  times. Model download or runtime errors produce a friendly user-facing
  message; UI exceptions are logged without displaying a traceback.
- A new upload clears the previous conversation and store before processing,
  so questions cannot accidentally use an earlier PDF.

## Future improvements

- Add OCR for scanned PDFs and support additional document formats.
- Add streaming responses, multi-PDF collections, and document deletion.
- Persist indexes securely for larger or repeat workloads.
- Add retrieval evaluation and configurable thresholds based on a test
  corpus.
- Add OCR, retrieval evaluation, and support for more local model choices.
