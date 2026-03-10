from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false

from langchain_ollama import ChatOllama

from app.core.config import settings
from app.services import vector_store
from app.services.rag_service import LegalSearchRetriever, create_rag_chain


def get_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.1,
    )


def get_vector_store():
    return vector_store


def get_rag_chain(k: int = 5):
    store = get_vector_store()
    retriever = LegalSearchRetriever(search_fn=store.search_similar, k=k)
    return create_rag_chain(get_llm(), retriever)
