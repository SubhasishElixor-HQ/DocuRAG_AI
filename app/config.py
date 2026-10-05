import os
from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("HUGGINGFACEHUB_API_KEY")

PDF_PATH = "documents/machine_learning.pdf"
VECTORSTORE_PATH = "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"