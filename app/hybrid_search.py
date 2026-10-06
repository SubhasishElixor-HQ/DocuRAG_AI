from rank_bm25 import BM25Okapi


class HybridSearch:

    def __init__(
        self,
        documents,
        vectorstore
    ):
        """
        Create Hybrid Search.

        documents:
            All document chunks.

        vectorstore:
            Existing FAISS vector database.
        """

        self.documents = documents
        self.vectorstore = vectorstore

        # -----------------------------------
        # Create BM25 index
        # -----------------------------------

        tokenized_documents = [
            document.page_content.lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    # =======================================
    # VECTOR SEARCH
    # =======================================

    def vector_search(
        self,
        query,
        k=10
    ):
        """
        Semantic search using FAISS.
        """

        results = self.vectorstore.similarity_search(
            query,
            k=k
        )

        return results

    # =======================================
    # KEYWORD SEARCH
    # =======================================

    def keyword_search(
        self,
        query,
        k=10
    ):
        """
        Keyword search using BM25.
        """

        # Convert query into tokens
        query_tokens = query.lower().split()

        # Calculate BM25 scores
        scores = self.bm25.get_scores(
            query_tokens
        )

        # Sort document indexes
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indices[:k]:

            results.append(
                self.documents[index]
            )

        return results

    # =======================================
    # RRF
    # =======================================

    def reciprocal_rank_fusion(
        self,
        vector_results,
        keyword_results,
        k=5
    ):
        """
        Combine vector and keyword rankings
        using Reciprocal Rank Fusion.
        """

        RRF_K = 60

        scores = {}

        # -----------------------------------
        # Vector results
        # -----------------------------------

        for rank, document in enumerate(
            vector_results,
            start=1
        ):

            document_key = (
                document.page_content,
                document.metadata.get(
                    "document_name"
                ),
                document.metadata.get(
                    "page"
                )
            )

            score = 1 / (
                RRF_K + rank
            )

            if document_key not in scores:

                scores[document_key] = {
                    "document": document,
                    "score": 0
                }

            scores[document_key]["score"] += score

        # -----------------------------------
        # Keyword results
        # -----------------------------------

        for rank, document in enumerate(
            keyword_results,
            start=1
        ):

            document_key = (
                document.page_content,
                document.metadata.get(
                    "document_name"
                ),
                document.metadata.get(
                    "page"
                )
            )

            score = 1 / (
                RRF_K + rank
            )

            if document_key not in scores:

                scores[document_key] = {
                    "document": document,
                    "score": 0
                }

            scores[document_key]["score"] += score

        # -----------------------------------
        # Sort
        # -----------------------------------

        ranked_results = sorted(
            scores.values(),
            key=lambda item: item["score"],
            reverse=True
        )

        # Return documents only
        return [
            item["document"]
            for item in ranked_results[:k]
        ]

    # =======================================
    # HYBRID SEARCH
    # =======================================

    def search(
        self,
        query,
        k=5
    ):
        """
        Complete Hybrid Search:

        Query
          ↓
        Vector Search
          +
        BM25
          ↓
        RRF
          ↓
        Top K Documents
        """

        # Get more results from both systems
        vector_results = self.vector_search(
            query,
            k=10
        )

        keyword_results = self.keyword_search(
            query,
            k=10
        )

        # Combine results
        final_results = self.reciprocal_rank_fusion(
            vector_results,
            keyword_results,
            k=k
        )

        return final_results