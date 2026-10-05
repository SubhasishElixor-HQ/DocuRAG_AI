from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from app.config import DOCUMENTS_PATH


def load_all_pdfs():

    all_documents = []

    pdf_files = Path(DOCUMENTS_PATH).glob("*.pdf")

    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))

        documents = loader.load()

        for document in documents:

            document.metadata["document_name"] = pdf_file.name
            document.metadata["document_type"] = "pdf"

        all_documents.extend(documents)

    return all_documents