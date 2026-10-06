from langchain_core.prompts import ChatPromptTemplate

from app.hybrid_search import HybridSearch
from app.reranker import Reranker


def create_rag_chain(
    documents,
    vectorstore,
    llm,
    selected_document=None
):

    # =======================================
    # HYBRID SEARCH
    # =======================================

    hybrid_search = HybridSearch(
        documents,
        vectorstore
    )

    # =======================================
    # RERANKER
    # =======================================

    reranker = Reranker()

    # =======================================
    # PROMPT
    # =======================================

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful research paper assistant.

Answer the user's question using ONLY
the provided context.

If the answer cannot be found in the
provided context, say:

"I could not find the answer in the
provided documents."

Do not invent information.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # =======================================
    # RAG FUNCTION
    # =======================================

    def rag(question):

        # -----------------------------------
        # STEP 1
        # Hybrid retrieval
        # -----------------------------------

        candidates = hybrid_search.search(
            question,
            k=10
        )

        # -----------------------------------
        # STEP 2
        # Filter selected document
        # -----------------------------------

        if selected_document:

            candidates = [
                document
                for document in candidates
                if document.metadata.get(
                    "document_name"
                ) == selected_document
            ]

        # -----------------------------------
        # STEP 3
        # RERANK
        # -----------------------------------

        reranked_results = reranker.rerank(
            question,
            candidates,
            top_k=5
        )

        # -----------------------------------
        # STEP 4
        # Get final documents
        # -----------------------------------

        final_documents = [
            document
            for document, score
            in reranked_results
        ]

        # -----------------------------------
        # STEP 5
        # Create context
        # -----------------------------------

        context = "\n\n".join(
            document.page_content
            for document in final_documents
        )

        # -----------------------------------
        # STEP 6
        # Create prompt
        # -----------------------------------

        formatted_prompt = prompt.format(
            context=context,
            question=question
        )

        # -----------------------------------
        # STEP 7
        # LLM
        # -----------------------------------

        response = llm.invoke(
            formatted_prompt
        )

        # -----------------------------------
        # STEP 8
        # Sources
        # -----------------------------------

        sources = []

        for document, score in reranked_results:

            sources.append(
                {
                    "document": document.metadata.get(
                        "document_name",
                        "Unknown"
                    ),
                    "page": document.metadata.get(
                        "page",
                        "Unknown"
                    ),
                    "score": float(score)
                }
            )

        # -----------------------------------
        # STEP 9
        # Return
        # -----------------------------------

        return {
            "answer": response.content,
            "sources": sources
        }

    return rag