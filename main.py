from app.loaders import load_all_pdfs
from app.splitter import split_documents
from app.embeddings import create_embeddings
from app.vectorstore import create_vectorstore
from app.llm import create_llm
from app.rag import create_rag_chain


def main():

    print("\n======================================")
    print("       RAG V4 - Hybrid Search")
    print("======================================\n")

   
    # 1. LOAD PDF
   

    print("1. Loading PDF documents...")

    documents = load_all_pdfs()

    print(
        f"Loaded {len(documents)} pages.\n"
    )

    
    # 2. SPLIT DOCUMENTS
   

    print("2. Splitting documents...")

    chunks = split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks.\n"
    )

    
    # 3. EMBEDDINGS
    

    print("3. Creating embedding model...")

    embeddings = create_embeddings()

    print(
        "Embeddings ready.\n"
    )

   
    # 4. FAISS
   

    print(
        "4. Creating FAISS vector database..."
    )

    vectorstore = create_vectorstore(
        chunks,
        embeddings
    )

    print(
        "Vector database created.\n"
    )

    
    # 5. LLM
    

    print("5. Loading LLM...")

    llm = create_llm()

    print(
        "LLM ready.\n"
    )

    
    # 6. DOCUMENT LIST
   

    print("======================================")
    print("Available documents:")
    print("======================================")

    document_names = sorted(
        {
            document.metadata[
                "document_name"
            ]
            for document in documents
        }
    )

    for index, document_name in enumerate(
        document_names,
        start=1
    ):

        print(
            f"{index}. {document_name}"
        )

    print("\nEnter document number to filter.")
    print("Enter 0 to search all documents.")

  
    # DOCUMENT SELECTION
   

    while True:

        choice = input(
            "\nSelect document: "
        ).strip()

        # EXIT
        

        if choice.lower() == "exit":

            print("Goodbye!")

            break

        # VALIDATION
       

        if not choice.isdigit():

            print(
                "Please enter a valid number."
            )

            continue

        choice = int(choice)

       
        # ALL DOCUMENTS
       

        if choice == 0:

            selected_document = None

        
        # SPECIFIC DOCUMENT
        

        elif 1 <= choice <= len(
            document_names
        ):

            selected_document = (
                document_names[
                    choice - 1
                ]
            )

        else:

            print(
                "Invalid document number."
            )

            continue

        print(
            f"\nSelected: "
            f"{selected_document or 'ALL DOCUMENTS'}"
        )

        
        # CREATE RAG
       

        rag = create_rag_chain(
            documents,
            vectorstore,
            llm,
            selected_document
        )

        print(
            "\nYou can now ask questions."
        )

        print(
            "Type 'back' to select another document."
        )

       
        # QUESTION LOOP
        

        while True:

            question = input(
                "\nAsk a question: "
            ).strip()

            
            # BACK
            
            if question.lower() == "back":

                break

            
            # EXIT
            

            if question.lower() == "exit":

                print("Goodbye!")

                return

            
            # EMPTY QUESTION
            

            if not question:

                print(
                    "Please enter a question."
                )

                continue

            
            # RAG
            

            result = rag(
                question
            )

            
            # ANSWER
           
            print("\nAnswer:")

            print(
                result["answer"]
            )

            # SOURCES
           

            print("\nSources:")

            if not result["sources"]:

                print(
                    "No sources found."
                )

            else:

                for source in result[
                    "sources"
                ]:

                    print(
                        f"- {source['document']} "
                        f"— Page {source['page']}"
                    )


if __name__ == "__main__":
    main()