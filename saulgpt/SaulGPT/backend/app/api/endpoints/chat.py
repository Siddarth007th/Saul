"""Streaming chat endpoint using Vercel AI SDK Data Stream Protocol."""

from __future__ import annotations

# pyright: reportMissingImports=false

import json
import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ai_datastream.stream_parts import DataStreamPart, DataStreamText

from app.dependencies import get_rag_chain
from app.models.schemas import ChatRequest
from app.services.rag_service import maybe_append_disclaimer

logger = logging.getLogger(__name__)

router = APIRouter()

DATA_STREAM_HEADERS = {
    "Content-Type": "text/plain; charset=utf-8",
    "X-Vercel-AI-Data-Stream": "v1",
}

_GREETING_RE = re.compile(
    r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|thanks?|thank you)$",
    re.IGNORECASE,
)
_META_RE = re.compile(
    r"(?:what can you do|who are you|help|how do you work|what do you help with)",
    re.IGNORECASE,
)
_LEGAL_RE = re.compile(
    r"\b(?:law|legal|section|article|act|code|bns|bnss|bsa|ipc|constitution|fir|bail|court|judge|petition|complaint|notice|affidavit|agreement|draft|template|offence|offense|crime|punishment|arrest|police|evidence|procedure|theft|murder|defamation|cyber)\b",
    re.IGNORECASE,
)


def _stream_text_response(message: str):
    yield DataStreamText(message).format()


def _route_message(message: str) -> tuple[str, str | None]:
    normalized = " ".join(message.strip().split())

    if _GREETING_RE.fullmatch(normalized):
        return (
            "direct",
            "Hello. I am SaulGPT, a legal research assistant for Indian law. "
            "Ask me about BNS, BNSS, BSA, IPC, constitutional provisions, "
            "FIR procedure, bail, or legal drafting.",
        )

    if _META_RE.search(normalized):
        return (
            "direct",
            "I help with Indian legal research and drafting support. I can "
            "answer questions about statutes, sections, procedure, citations, "
            "and provide limited legal templates grounded in retrieved legal text.",
        )

    if _LEGAL_RE.search(normalized):
        return ("rag", None)

    return (
        "direct",
        "I am limited to Indian legal research and legal drafting support. "
        "Please ask a legal question about statutes, procedure, citations, or legal documents.",
    )


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    user_messages = [m for m in request.messages if m.role == "user"]

    if not user_messages or not user_messages[-1].content.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    last_message = user_messages[-1].content

    # Long query truncation
    if len(last_message) > 2000:
        last_message = last_message[:2000]

    route, direct_response = _route_message(last_message)

    if route == "direct" and direct_response is not None:
        return StreamingResponse(
            _stream_text_response(direct_response),
            media_type="text/event-stream",
            headers=DATA_STREAM_HEADERS,
        )

    # Context window protection: fewer retrieval docs for long queries
    k = 3 if len(last_message) > 1000 else 5

    async def generate():  # noqa: C901
        try:
            chain: Any = get_rag_chain(k=k)
        except Exception as exc:
            logger.error("Failed to initialise RAG chain: %s", exc)
            raise HTTPException(status_code=503, detail="Service unavailable") from exc

        context_docs: list[Any] = []
        full_answer: list[str] = []

        try:
            async for chunk in chain.astream({"input": last_message}):
                if "answer" in chunk and chunk["answer"]:
                    full_answer.append(chunk["answer"])
                    yield DataStreamText(chunk["answer"]).format()
                if "context" in chunk:
                    context_docs = chunk["context"]
        except ConnectionRefusedError as exc:
            logger.error("Ollama connection refused: %s", exc)
            yield DataStreamPart(
                "e",
                json.dumps({"error": "Ollama is unavailable (503)"}),
            ).format()
            return
        except Exception as exc:
            exc_type = type(exc).__name__
            exc_module = type(exc).__module__ or ""

            # httpx.ConnectError or similar connection failures → Ollama down
            if "ConnectError" in exc_type or "ConnectionRefused" in exc_type:
                logger.error("Ollama connection error: %s", exc)
                yield DataStreamPart(
                    "e",
                    json.dumps({"error": "Ollama is unavailable (503)"}),
                ).format()
                return

            # psycopg2.OperationalError → pgvector down
            if "OperationalError" in exc_type and "psycopg2" in exc_module:
                logger.error("pgvector connection error: %s", exc)
                yield DataStreamPart(
                    "e",
                    json.dumps({"error": "pgvector is unavailable (503)"}),
                ).format()
                return

            logger.exception("Unexpected error during streaming")
            yield DataStreamPart(
                "e",
                json.dumps({"error": "Internal server error"}),
            ).format()
            return

        disclaimer = maybe_append_disclaimer("".join(full_answer), last_message)
        if disclaimer:
            yield DataStreamText(disclaimer).format()

        for doc in context_docs:
            citation = {
                "type": "citation",
                "data": {
                    "title": doc.metadata.get("title", ""),
                    "section": doc.metadata.get("section", ""),
                    "act": doc.metadata.get("act", ""),
                    "source_type": doc.metadata.get("source_type", ""),
                    "document_type": doc.metadata.get("document_type", ""),
                    "filename": doc.metadata.get("filename", ""),
                    "content": doc.page_content[:200],
                },
            }
            yield DataStreamPart("d", citation).format()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=DATA_STREAM_HEADERS,
    )
