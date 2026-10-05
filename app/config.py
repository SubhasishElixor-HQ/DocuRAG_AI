import os
from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")

DOCUMENTS_PATH = "documents"
VECTORSTORE_PATH = "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG retrieval settings
# Retrieve up to 6 candidate chunks.
# This gives us a threshold for deciding whether a retrieved result is relevant enough.
RETRIEVAL_K = 6
MAX_RELEVANCE_DISTANCE = 1.2