from app.config import RETRIEVAL_K, MAX_RELEVANCE_DISTANCE


def create_retriever(vectorstore, document_name=None):

    search_kwargs = {
        "k": RETRIEVAL_K
    }

    if document_name:
        search_kwargs["filter"] = {
            "document_name": document_name
        }

    retriever = vectorstore.as_retriever(
        search_kwargs=search_kwargs
    )

    return retriever


def retrieve_documents(
    vectorstore,
    question,
    document_name=None
):
    """
    Retrieve documents using similarity scores.
    """

    if document_name:
        results = vectorstore.similarity_search_with_score(
            question,
            k=RETRIEVAL_K,
            filter={
                "document_name": document_name
            }
        )
    else:
        results = vectorstore.similarity_search_with_score(
            question,
            k=RETRIEVAL_K
        )

    relevant_documents = []

    for document, score in results:

        if score <= MAX_RELEVANCE_DISTANCE:
            relevant_documents.append(
                document
            )

    return relevant_documents