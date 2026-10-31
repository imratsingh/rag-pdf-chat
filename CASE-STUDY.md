# Case Study: rag-pdf-chat

## Project overview

**rag-pdf-chat** is a local-first application that lets a user upload a PDF and
ask questions about its contents in a chat interface. It retrieves passages
from the uploaded document, answers using those passages, and cites the source
page. When the document does not contain enough information, it returns:

> Ye jawab document mein nahi mila.

The project is built with Python, Streamlit, LangChain, ChromaDB, and local
Hugging Face models. It does not require an OpenAI API key or a Hugging Face
access token.

## Problem

Finding a specific fact in a long PDF can be slow. General-purpose chatbots may
also produce plausible-sounding answers that are not supported by the
document. The project aims to make document questions easier to ask while
keeping responses tied to the uploaded PDF and showing where the information
came from.

## Solution

The application processes a PDF into searchable passages and uses local
embedding and extractive question-answering models to answer questions. It
keeps the active document index and conversation in the user's Streamlit
session. Source-page citations let the user check answers against the PDF.

## How it works

1. **Upload:** The user selects a PDF in the Streamlit interface.
2. **Extract:** PyPDFLoader extracts text page by page. Invalid or text-free
   documents are rejected with a clear error.
3. **Split:** The text is divided into chunks of up to 800 characters, with
   100 characters of overlap. Page metadata is retained.
4. **Index:** `sentence-transformers/all-MiniLM-L6-v2` creates local
   embeddings. ChromaDB stores the chunks in an in-memory collection using
   cosine similarity.
5. **Retrieve:** The app searches for up to four relevant chunks. Keyword
   matching can recover obvious matches when local semantic scores are low.
6. **Answer:** For supported questions, `deepset/roberta-base-squad2` extracts
   an answer span from the retrieved text. Direct source extraction handles
   matching document sections, model-comparison sentences, and education
   entries.
7. **Cite or abstain:** The app shows source page numbers with the answer. If
   the evidence is missing or confidence is too low, it displays the exact
   no-answer message.

## Key design decisions

- **Local inference:** Embeddings and question answering run on the user's
  computer after model downloads. PDF text is not sent to an OpenAI service.
- **Evidence-first answering:** The QA model receives retrieved PDF text, not
  unrestricted general knowledge. Low-confidence answers are rejected.
- **Cosine similarity:** The embedding model outputs normalized vectors, for
  which cosine similarity is a suitable retrieval measure.
- **Small, overlapping chunks:** Chunks keep retrieval focused; overlap helps
  retain context at chunk boundaries.
- **In-memory indexing:** The vector store is not persisted to disk, keeping
  the demo simple and avoiding local index files containing document text.
- **One-based citations:** Page numbers match the numbering users see in PDF
  viewers.

## Results

- Users can ask questions conversationally and see the relevant PDF pages.
- Empty questions are handled without model inference.
- Missing, non-PDF, malformed, and text-free PDFs produce user-facing errors
  rather than raw tracebacks.
- Unsupported questions receive the defined no-answer response instead of a
  confident guess.
- The resume-question regression scenarios now return the compared churn
  models and the degree/university entry from the source PDF.
- The automated test suite was verified with **24 passed and 1 skipped**; the
  skipped test is the optional local-model integration test. Ruff checks
  passed.

## Limitations

- Scanned PDFs do not contain extractable text; OCR is not currently included.
- Model downloads require an internet connection the first time, and local
  inference requires enough disk space and memory.
- The QA model runs on CPU by default, which may be slow on some computers.
- Extractive QA works best when the answer is stated clearly in the PDF.
  Complex reasoning across distant passages may require additional
  improvements.
- The vector store is in memory and must be rebuilt when the app session or
  uploaded document changes.

## Running the project

From the project directory in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Run the automated tests with:

```powershell
python -m pytest -q
```

The app downloads its configured Hugging Face models on first use. No API key
or `.env` file is needed.

## Possible next steps

- Add OCR for scanned PDFs.
- Evaluate retrieval and answer quality against a larger question-and-answer
  dataset.
- Support multiple PDFs in a single conversation.
- Add optional secure persistence for users who need to reuse indexes.
- Improve handling of multi-part questions and answers requiring evidence
  across multiple pages.
