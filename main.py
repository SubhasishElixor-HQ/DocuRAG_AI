from app.loaders import load_all_pdfs
from app.splitter import split_documents
from app.embeddings import create_embeddings
from app.vectorstore import create_vectorstore
from app.llm import create_llm
from app.rag import create_rag_chain


def main():

    print("\n======================================")
    print("       RAG V4 - Better Retrieval")
    print("======================================\n")


    # 1. Load PDF documents
    print("1. Loading PDF documents...")

    documents = load_all_pdfs()

    print(f"Loaded {len(documents)} pages.\n")


    # 2. Split documents into chunks
    print("2. Splitting documents...")

    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks.\n")


    # 3. Create embedding model
    print("3. Creating embedding model...")

    embeddings = create_embeddings()

    print("Embeddings ready.\n")


    # 4. Create FAISS vector database
    print("4. Creating FAISS vector database...")

    vectorstore = create_vectorstore(
        chunks,
        embeddings
    )

    print("Vector database created.\n")


    # 5. Load LLM
    print("5. Loading LLM...")

    llm = create_llm()

    print("LLM ready.\n")


    # 6. Create RAG chain
    print("6. Creating RAG chain...")

    rag = create_rag_chain(
        vectorstore,
        llm
    )

    print("RAG chain ready.\n")


    # 7. Start chatbot
    print("======================================")
    print("       RAG V4 chatbot is ready!")
    print("======================================")
    print("Type 'exit' to stop.")
    print("======================================")


    while True:

        question = input("\nAsk a question: ").strip()


        # Exit chatbot
        if question.lower() in ["exit", "quit"]:

            print("Goodbye!")

            break


        # Empty question
        if not question:

            print("Please enter a question.")

            continue


        # Ask RAG system
        result = rag(question)


        # Display answer
        print("\nAnswer:")

        print(result["answer"])


        # Display sources
        print("\nSources:")


        if not result["sources"]:

            print("No relevant sources found.")


        else:

            for source in result["sources"]:

                print(
                    f"- {source['document']} "
                    f"(Page {source['page']})"
                )


if __name__ == "__main__":
    main()