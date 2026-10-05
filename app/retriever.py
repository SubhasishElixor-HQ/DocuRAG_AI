def create_retriever(vectorstore, document_name=None):

    search_kwargs = {
        "k": 4
    }

    if document_name:
        search_kwargs["filter"] = {
            "document_name": document_name
        }

    retriever = vectorstore.as_retriever(
        search_kwargs=search_kwargs
    )

    return retriever