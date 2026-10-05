from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    return chunks

# PDF
#  ↓
# Page 1
# Page 2
# Page 3
#  ↓
# Chunks
#  ├── Chunk 1
#  ├── Chunk 2
#  ├── Chunk 3
#  ├── Chunk 4
#  └── ...