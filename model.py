from rag_flow import RAGFlow
from langchain_ollama import ChatOllama
import os

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

llama = ChatOllama(
    model="llama3.2:3b",
    base_url=OLLAMA_URL,
    verbose=False
)

classifier = ChatOllama(
    model="llama3.2:3b",
    base_url=OLLAMA_URL,
    temperature=0,
    num_predict=15,
    reasoning=False,
    num_ctx=3000,
    verbose=False,
)

params = {
    "query_model": llama,
    "answer_model": llama,
    "titles_limit": 5,
    "top_k_titles": 5,
    "chunk_size": 200,
    "chunk_overlap_fraction": 0.2,
    "candidate_k": 20,
    "top_k_chunks": 10,
    "classifier": classifier,
}

rag = RAGFlow(params)
