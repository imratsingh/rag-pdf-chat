"""Application configuration for local Hugging Face models."""

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4
MIN_SCORE = 0.3
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL = "deepset/roberta-base-squad2"
MIN_ANSWER_SCORE = 0.1
NO_ANSWER_MESSAGE = "Ye jawab document mein nahi mila."
EMPTY_QUESTION_MESSAGE = "Pehle koi sawaal likho."
MODEL_ERROR_MESSAGE = (
    "Could not load a local Hugging Face model. Check your internet connection "
    "for the first download and ensure your computer has enough memory."
)
