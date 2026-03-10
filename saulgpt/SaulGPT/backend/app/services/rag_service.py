from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUntypedBaseClass=false, reportMissingParameterType=false, reportImplicitStringConcatenation=false, reportImplicitOverride=false

import re
from collections.abc import Callable
from typing import cast

from langchain_classic.chains import create_retrieval_chain  # pyright: ignore[reportAny]
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever

from app.models.schemas import SearchResult

SYSTEM_PROMPT = (
    "You are SaulGPT, a specialized legal AI assistant for Indian law. Use the "
    "retrieved context as the primary authority for your answer. You may use "
    "general legal knowledge only to improve explanation, structure, or plain "
    "language, but never to override or contradict the retrieved text. Always cite "
    "the specific section and act when the context provides them. "
    "Treat these abbreviations as fixed: BNS = Bharatiya Nyaya Sanhita, BNSS = "
    "Bharatiya Nagarik Suraksha Sanhita, IPC = Indian Penal Code, Constitution = "
    "Constitution of India. Never expand them to unrelated laws or acronyms. If "
    "the retrieved context includes the cited act or section, do not claim it is "
    "missing. For equivalence questions, compare the retrieved source and target "
    "provisions by title and subject matter and answer cautiously as the closest "
    "available match. For FIR or procedure questions, prefer retrieved BNSS "
    "procedural sections over generic legal knowledge. If the context is partial, "
    "say which parts are supported by retrieved text and which parts are general "
    "legal guidance that should be verified. If the context truly does not contain "
    "relevant information, say so clearly. When the user asks for a "
    "legal draft, application, notice, complaint, or template, you may provide a "
    "neutral informational sample grounded only in the retrieved context. Keep it "
    "general, use placeholders for facts, avoid personalizing strategy, and cite "
    "the supporting section and act for the clauses you include. Do not invent "
    "procedural powers, grounds, or filing requirements that are not stated in the "
    "retrieved text. If the available context covers only the offence and not the "
    "procedure for the requested filing, say so explicitly and use a placeholder "
    "for the missing procedural provision instead of asserting one. If the "
    "retrieved context is too thin for a reliable draft, say that you can only "
    "provide a limited template based on the available sections. You are NOT a "
    "lawyer - provide informational guidance only and say it is not legal advice."
)

DOCUMENT_PROMPT = PromptTemplate.from_template(
    """Source Type: {source_type}
Document Type: {document_type}
Filename: {filename}
Act: {act_name} ({act})
Section: {section}
Title: {title}
Source: {source}
Content:
{page_content}"""
)

_ACT_NAMES = {
    "BNS": "Bharatiya Nyaya Sanhita",
    "BNSS": "Bharatiya Nagarik Suraksha Sanhita",
    "BSA": "Bharatiya Sakshya Adhiniyam",
    "IPC": "Indian Penal Code",
    "CONSTITUTION": "Constitution of India",
}


def _act_name(act: str | None) -> str:
    if act is None:
        return "Unknown Act"
    return _ACT_NAMES.get(act.upper(), act)


class LegalSearchRetriever(BaseRetriever):
    search_fn: Callable[[str, int, str | None], list[SearchResult]]
    k: int = 5
    min_score: float = 0.55

    def _get_relevant_documents(
        self, query: str, *, run_manager: object
    ) -> list[Document]:
        _ = run_manager
        results = self.search_fn(query, self.k, None)
        documents: list[Document] = []

        for result in results:
            metadata = dict(result.metadata)
            source_type = str(metadata.get("source_type") or "statutory")
            score_threshold = self.min_score if source_type != "uploaded" else 0.35
            if result.score < score_threshold:
                continue

            metadata["score"] = result.score
            metadata["source_type"] = source_type
            metadata.setdefault("document_type", "statutory")
            metadata.setdefault("filename", "")
            metadata.setdefault("act", "")
            metadata.setdefault("section", "")
            metadata.setdefault("title", "Untitled provision")
            metadata.setdefault("source", "")
            metadata["act_name"] = _act_name(metadata.get("act"))
            documents.append(Document(page_content=result.content, metadata=metadata))

        return documents


_LEGAL_REF_RE = re.compile(
    r"Section\s+\d+|Article\s+\d+|BNS|IPC|Constitution",
    re.IGNORECASE,
)

_LEGAL_QUERY_RE = re.compile(
    r"\b(?:law|section|act|court|legal|crime|punishment|bail|fir)\b",
    re.IGNORECASE,
)

_NO_REF_DISCLAIMER = (
    "\n\n*Note: This response could not find relevant statutory references. "
    "Please verify with official legal sources.*"
)


def maybe_append_disclaimer(answer: str, question: str) -> str | None:
    """Return the disclaimer suffix if the answer lacks statutory references
    and the question appears legal in nature.  Returns ``None`` otherwise."""
    if _LEGAL_REF_RE.search(answer):
        return None
    if not _LEGAL_QUERY_RE.search(question):
        return None
    return _NO_REF_DISCLAIMER


def create_rag_chain(llm, retriever: BaseRetriever) -> object:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Context:\n{context}\n\n"
                "Question: {input}\n\n"
                "Response rules:\n"
                "- Treat the provided context as the primary legal authority.\n"
                "- You may use general legal knowledge only to explain or organize the answer.\n"
                "- Never contradict the provided context.\n"
                "- If a point is not directly supported by the context, label it as general guidance that requires verification.\n"
                "- Cite each legal basis with section and act.\n"
                "- For draft or template requests, use this order: Retrieved legal basis; "
                "Limits of available context; Informational template.\n"
                "- Do not turn an offence section into a bail provision or filing power.\n"
                "- If the context does not include the filing procedure, bail power, "
                "required grounds, conditions, or supporting documents, say that explicitly "
                "in Limits of available context and use bracketed placeholders for those "
                "parts instead of inventing them.\n"
                "- Keep any draft general, informational, and not legal advice.",
            ),
        ]
    )
    question_answer_chain = cast(
        object,
        create_stuff_documents_chain(
            llm,
            prompt,
            document_prompt=DOCUMENT_PROMPT,
            document_separator="\n\n---\n\n",
        ),
    )
    return cast(object, create_retrieval_chain(retriever, question_answer_chain))
