from langchain_community.document_loaders import PyPDFLoader
from app.config import PDF_PATH

def load_pdf():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    return documents

# dbms_notes.pdf
#       ↓
# PyPDFLoader
#       ↓
# documents