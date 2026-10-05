from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from app.config import LLM_API_KEY


def create_llm():
    
    llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
    )
    model = ChatHuggingFace(llm=llm)

    return model