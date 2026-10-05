import os
from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("HuggingFace_API_Key")

DOCUMENTS_PATH = "documents"
VECTORSTORE_PATH = "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG retrieval settings
RETRIEVAL_K = 6
MAX_RELEVANCE_DISTANCE = 1.2