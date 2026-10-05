from app.loaders import load_all_pdfs
from app.splitter import split_documents
from app.embeddings import create_embeddings
from app.vectorstore import create_vectorstore
from app.retriever import create_retriever
from app.llm import create_llm
from app.rag import create_rag_chain


def main():

    print("\n======================================")
    print("      RAG V2 - Multiple Documents")
    print("======================================\n")

    print("1. Loading PDF documents...")

    documents = load_all_pdfs()

    print(f"Loaded {len(documents)} pages.\n")


    print("2. Splitting documents...")

    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks.\n")


    print("3. Creating embedding model...")

    embeddings = create_embeddings()

    print("Embeddings ready.\n")


    print("4. Creating FAISS vector database...")

    vectorstore = create_vectorstore(
        chunks,
        embeddings
    )

    print("Vector database created.\n")


    print("5. Creating retriever...")

    retriever = create_retriever(
        vectorstore
    )

    print("Retriever ready.\n")


    print("6. Loading LLM...")

    llm = create_llm()

    print("LLM ready.\n")


    rag = create_rag_chain(
        retriever,
        llm
    )


    print("======================================")
    print("RAG chatbot is ready!")
    print("Type 'exit' to stop.")
    print("======================================")


    while True:

        question = input("\nAsk a question: ")

        if question.lower().strip() in ["exit", "quit"]:

            print("Goodbye!")

            break


        result = rag(question)


        print("\nAnswer:")
        print(result["answer"])


        print("\nSources:")

        seen_sources = set()


        for source in result["sources"]:

            source_name = source["source"]
            page = source["page"] + 1

            source_key = (
                source_name,
                page
            )


            if source_key not in seen_sources:

                print(
                    f"- {source_name} "
                    f"(page {page})"
                )

                seen_sources.add(
                    source_key
                )


if __name__ == "__main__":
    main()