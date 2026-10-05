from langchain_core.prompts import ChatPromptTemplate


def create_rag_chain(retriever, llm):

    prompt = ChatPromptTemplate.from_template(
        """
You are a document question-answering assistant.

Answer the question ONLY using the provided context.

Rules:
1. Use only the provided context.
2. Do not use outside knowledge.
3. Do not invent information.
4. If the answer is not in the context, say:
   "I could not find this information in the uploaded documents."
5. Give a concise and clear answer.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    def ask(question):

        documents = retriever.invoke(question)

        if not documents:

            return {
                "answer": (
                    "I could not find this information "
                    "in the uploaded documents."
                ),
                "sources": []
            }

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        messages = prompt.format_messages(
            context=context,
            question=question
        )

        response = llm.invoke(messages)

        sources = []

        seen_sources = set()

        for document in documents:

            document_name = document.metadata.get(
                "document_name",
                "Unknown document"
            )

            page = document.metadata.get(
                "page",
                0
            )

            source_key = (
                document_name,
                page
            )

            if source_key not in seen_sources:

                sources.append({
                    "document": document_name,
                    "page": page + 1
                })

                seen_sources.add(source_key)

        return {
            "answer": response.content,
            "sources": sources
        }

    return ask