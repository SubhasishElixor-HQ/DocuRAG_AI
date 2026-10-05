from langchain_huggingface import HuggingFaceEmbeddings
from app.config import EMBEDDING_MODEL


def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return embeddings

# "What is normalization?"
#           ↓
#      Embedding Model
#           ↓
# [0.12, -0.42, 0.73, ...]