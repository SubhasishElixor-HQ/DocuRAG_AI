from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        print("Loading reranker model...")

        self.model = CrossEncoder(
            model_name
        )

        print("Reranker ready.")

    def rerank(
        self,
        query,
        documents,
        top_k=5
    ):
        """
        Rerank retrieved documents
        according to query relevance.
        """

        # -----------------------------------
        # Create query-document pairs
        # -----------------------------------

        pairs = []

        for document in documents:

            pairs.append(
                (
                    query,
                    document.page_content
                )
            )

        # -----------------------------------
        # Calculate relevance scores
        # -----------------------------------

        scores = self.model.predict(
            pairs
        )

        # -----------------------------------
        # Combine document + score
        # -----------------------------------

        ranked_documents = list(
            zip(
                documents,
                scores
            )
        )

        # -----------------------------------
        # Sort by relevance
        # -----------------------------------

        ranked_documents.sort(
            key=lambda item: item[1],
            reverse=True
        )

        # -----------------------------------
        # Return top K
        # -----------------------------------

        return ranked_documents[:top_k]