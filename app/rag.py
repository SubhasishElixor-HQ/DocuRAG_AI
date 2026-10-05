from langchain_core.prompts import ChatPromptTemplate


def create_rag_chain(retriever, llm):

    prompt = ChatPromptTemplate.from_template(
        """
You are a document question-answering assistant.

Answer the question ONLY using the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not present in the context, say:
   "I could not find this information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    def ask(question):

        documents = retriever.invoke(question)

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

        for document in documents:

            source = document.metadata.get("source")
            page = document.metadata.get("page")

            sources.append({
                "source": source,
                "page": page
            })

        return {
            "answer": response.content,
            "sources": sources
        }

    return ask