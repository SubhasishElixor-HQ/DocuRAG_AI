from langchain_community.vectorstores import FAISS

from app.config import VECTORSTORE_PATH


def create_vectorstore(chunks, embeddings):

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(
        VECTORSTORE_PATH
    )

    return vectorstore