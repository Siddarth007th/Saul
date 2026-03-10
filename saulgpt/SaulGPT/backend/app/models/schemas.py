"""Pydantic schemas for SaulGPT API."""

from typing import Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    messages: list[ChatMessage]


class SearchRequest(BaseModel):
    """Request body for search endpoint."""

    query: str
    top_k: int = 5
    filter_act: str | None = None  # "BNS", "IPC", "Constitution"


class SearchResult(BaseModel):
    """A single search result."""

    content: str
    metadata: dict
    score: float


class SearchResponse(BaseModel):
    """Response body for search endpoint."""

    results: list[SearchResult]
    query: str


class HealthResponse(BaseModel):
    """Response body for health check endpoint."""

    status: str
    ollama: bool
    pgvector: bool


class LegalDocumentChunk(BaseModel):
    """A chunk of a legal document."""

    content: str
    metadata: dict


class DocumentIngestionItem(BaseModel):
    """Result for one file in an ingestion request."""

    status: Literal["ingested", "failed"]
    filename: str
    document_type: str | None = None
    document_id: str | None = None
    chunks: int = 0
    detail: str


class DocumentIngestionResponse(BaseModel):
    """Response body for document ingestion endpoint."""

    total_files: int
    ingested_files: int
    failed_files: int
    total_chunks: int
    results: list[DocumentIngestionItem]


class ReportExportRequest(BaseModel):
    """Request body for report export endpoint."""

    title: str = "SaulGPT Report"
    content: str = Field(min_length=1)
    format: Literal["pdf", "docx"] = "pdf"
    filename: str | None = None
