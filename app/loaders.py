from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from app.config import DOCUMENTS_PATH


def load_all_pdfs():

    all_documents = []
    # Find every PDF inside the documents folder.
    pdf_files = Path(DOCUMENTS_PATH).glob("*.pdf")

    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))

        documents = loader.load()

        all_documents.extend(documents)

    return all_documents