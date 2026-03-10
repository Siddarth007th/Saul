from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from io import BytesIO
from pathlib import Path
import re
from typing import Any, Protocol, TypedDict, cast
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.services.vector_store import get_embedding_model

_SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
}
_SUPPORTED_CONTENT_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
}
_CHUNK_SIZE = 900
_CHUNK_OVERLAP = 120
_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE,
    chunk_overlap=_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)
_EMBEDDING_BATCH_SIZE = 32


@dataclass
class UploadedFilePayload:
    filename: str
    content_type: str | None
    data: bytes


class EmbeddingVector(Protocol):
    def tolist(self) -> list[float]: ...


class BatchEmbeddingModel(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[EmbeddingVector]: ...


class CursorProtocol(Protocol):
    def __enter__(self) -> CursorProtocol: ...

    def __exit__(
        self, exc_type: object, exc_value: object, traceback: object
    ) -> None: ...

    def execute(
        self, query: str, params: tuple[object, ...] | None = None
    ) -> object: ...


class ConnectionProtocol(Protocol):
    autocommit: bool

    def cursor(self) -> CursorProtocol: ...

    def commit(self) -> None: ...

    def close(self) -> None: ...


class ConnectionFactory(Protocol):
    def __call__(self, dsn: str) -> ConnectionProtocol: ...


class ExecuteValues(Protocol):
    def __call__(
        self,
        cur: CursorProtocol,
        sql: str,
        argslist: list[tuple[str, str, str]],
        *,
        template: str,
    ) -> object: ...


class IngestionRow(TypedDict):
    content: str
    metadata: dict[str, object]


class IngestionResult(TypedDict):
    status: str
    filename: str
    document_type: str | None
    document_id: str | None
    chunks: int
    detail: str


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_document_type(filename: str, content_type: str | None) -> str:
    extension = Path(filename).suffix.lower()
    if extension in _SUPPORTED_EXTENSIONS:
        return _SUPPORTED_EXTENSIONS[extension]

    if content_type is not None and content_type in _SUPPORTED_CONTENT_TYPES:
        return _SUPPORTED_CONTENT_TYPES[content_type]

    raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")


def _extract_pdf_text(data: bytes) -> str:
    pypdf_module = import_module("pypdf")
    reader_class = getattr(pypdf_module, "PdfReader")
    reader = reader_class(BytesIO(data))
    page_texts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        cleaned = text.strip()
        if cleaned:
            page_texts.append(cleaned)
    return "\n\n".join(page_texts)


def _extract_docx_text(data: bytes) -> str:
    docx_module = import_module("docx")
    document_class = cast(Any, getattr(docx_module, "Document"))
    document = document_class(BytesIO(data))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _extract_txt_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _extract_text(document_type: str, data: bytes) -> str:
    if document_type == "pdf":
        return _extract_pdf_text(data)
    if document_type == "docx":
        return _extract_docx_text(data)
    return _extract_txt_text(data)


def _chunk_text(text: str) -> list[str]:
    if len(text) <= _CHUNK_SIZE:
        return [text]
    return [chunk.strip() for chunk in _TEXT_SPLITTER.split_text(text) if chunk.strip()]


def _insert_chunks(chunk_rows: list[IngestionRow]) -> None:
    if not chunk_rows:
        return

    model = cast(BatchEmbeddingModel, get_embedding_model())
    psycopg2_module = import_module("psycopg2")
    connect = cast(ConnectionFactory, getattr(psycopg2_module, "connect"))
    extras_module = import_module("psycopg2.extras")
    execute_values = cast(ExecuteValues, getattr(extras_module, "execute_values"))

    conn = connect(settings.DATABASE_URL)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            _ = cur.execute(
                "CREATE INDEX IF NOT EXISTS legal_document_chunks_source_type_idx "
                "ON legal_document_chunks ((metadata->>'source_type'))"
            )
            _ = cur.execute(
                "CREATE INDEX IF NOT EXISTS legal_document_chunks_filename_idx "
                "ON legal_document_chunks ((metadata->>'filename'))"
            )
            for start in range(0, len(chunk_rows), _EMBEDDING_BATCH_SIZE):
                batch = chunk_rows[start : start + _EMBEDDING_BATCH_SIZE]
                texts = [row["content"] for row in batch]
                embeddings = model.encode(
                    texts,
                    batch_size=min(_EMBEDDING_BATCH_SIZE, len(texts)),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                rows = [
                    (
                        row["content"],
                        json.dumps(row["metadata"]),
                        _vector_literal(embedding.tolist()),
                    )
                    for row, embedding in zip(batch, embeddings)
                ]
                execute_values(
                    cur,
                    "INSERT INTO legal_document_chunks (content, metadata, embedding) VALUES %s",
                    rows,
                    template="(%s, %s::jsonb, %s::vector)",
                )
        conn.commit()
    finally:
        conn.close()


def ingest_documents(files: list[UploadedFilePayload]) -> dict[str, object]:
    now_iso = datetime.now(timezone.utc).isoformat()
    chunk_rows: list[IngestionRow] = []
    results: list[IngestionResult] = []
    total_chunks = 0
    ingested_files = 0

    for file in files:
        filename = file.filename.strip() or "uploaded_document"
        try:
            document_type = _detect_document_type(filename, file.content_type)
            raw_text = _extract_text(document_type, file.data)
            text = _normalize_text(raw_text)
            if not text:
                raise ValueError("No extractable text found in file.")

            chunks = _chunk_text(text)
            if not chunks:
                raise ValueError("Document text could not be chunked.")

            document_id = str(uuid4())
            title = Path(filename).stem or "Uploaded document"
            for index, chunk in enumerate(chunks):
                metadata: dict[str, object] = {
                    "source_type": "uploaded",
                    "document_id": document_id,
                    "filename": filename,
                    "document_type": document_type,
                    "mime_type": file.content_type or "",
                    "title": title,
                    "source": "user_upload",
                    "type": "uploaded_document",
                    "act": "",
                    "section": "",
                    "chunk_index": index,
                    "uploaded_at": now_iso,
                }
                chunk_rows.append({"content": chunk, "metadata": metadata})

            chunk_count = len(chunks)
            total_chunks += chunk_count
            ingested_files += 1
            results.append(
                {
                    "status": "ingested",
                    "filename": filename,
                    "document_type": document_type,
                    "document_id": document_id,
                    "chunks": chunk_count,
                    "detail": "Ingested successfully.",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "status": "failed",
                    "filename": filename,
                    "document_type": None,
                    "document_id": None,
                    "chunks": 0,
                    "detail": str(exc),
                }
            )

    if chunk_rows:
        _insert_chunks(chunk_rows)

    failed_files = len(files) - ingested_files
    return {
        "total_files": len(files),
        "ingested_files": ingested_files,
        "failed_files": failed_files,
        "total_chunks": total_chunks,
        "results": results,
    }
