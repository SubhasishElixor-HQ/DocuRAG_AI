from app.loaders import load_pdf
from app.splitter import split_documents
from app.embeddings import create_embeddings
from app.vectorstore import create_vectorstore
from app.retriever import create_retriever
from app.llm import create_llm
from app.rag import create_rag_chain


def main():

    print("Loading PDF...")

    documents = load_pdf()

    print(f"Loaded {len(documents)} pages.")

    print("Splitting documents...")

    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings...")

    embeddings = create_embeddings()

    print("Creating vector store...")

    vectorstore = create_vectorstore(
        chunks,
        embeddings
    )

    print("Creating retriever...")

    retriever = create_retriever(vectorstore)

    print("Loading LLM...")

    llm = create_llm()

    rag = create_rag_chain(
        retriever,
        llm
    )

    print("\nRAG chatbot is ready!")

    while True:

        question = input("\nAsk a question: ")

        if question.lower() in ["exit", "quit"]:
            break

        result = rag(question)

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources:")

        for source in result["sources"]:
            print(
                f"- {source['source']} "
                f"(page {source['page'] + 1})"
            )


if __name__ == "__main__":
    main()