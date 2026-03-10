#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Protocol, TypedDict, cast

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
BATCH_SIZE = 32

sys.path.insert(0, str(SCRIPT_DIR.parent))

from app.core.config import settings
from app.services.vector_store import get_embedding_model


class Chunk(TypedDict):
    content: str
    metadata: dict[str, object]


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

    def execute(self, query: str) -> object: ...

    def fetchone(self) -> tuple[int] | None: ...

    def fetchall(self) -> list[tuple[str, int]]: ...


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


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def load_chunks() -> list[Chunk]:
    with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
        return [cast(Chunk, json.loads(line)) for line in handle]


def main() -> None:
    print("Loading embedding model...")
    model = cast(BatchEmbeddingModel, get_embedding_model())

    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = load_chunks()
    print(f"Total chunks: {len(chunks)}")

    psycopg2_module = import_module("psycopg2")
    connect = cast(ConnectionFactory, getattr(psycopg2_module, "connect"))
    extras_module = import_module("psycopg2.extras")
    execute_values = cast(ExecuteValues, getattr(extras_module, "execute_values"))

    conn = connect(settings.DATABASE_URL)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE legal_document_chunks RESTART IDENTITY")

            inserted = 0
            for start in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[start : start + BATCH_SIZE]
                texts = [chunk["content"] for chunk in batch]
                embeddings = model.encode(
                    texts,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                rows = [
                    (
                        chunk["content"],
                        json.dumps(chunk["metadata"]),
                        _vector_literal(embedding.tolist()),
                    )
                    for chunk, embedding in zip(batch, embeddings)
                ]
                execute_values(
                    cur,
                    "INSERT INTO legal_document_chunks (content, metadata, embedding) VALUES %s",
                    rows,
                    template="(%s, %s::jsonb, %s::vector)",
                )

                inserted += len(batch)
                if inserted % 500 == 0 or inserted == len(chunks):
                    print(f"Inserted {inserted}/{len(chunks)} chunks...")

            conn.commit()
            print(f"Done. Inserted {inserted} chunks.")

            cur.execute("SELECT count(*) FROM legal_document_chunks")
            count_row = cur.fetchone()
            count = count_row[0] if count_row is not None else 0
            print(f"DB count: {count}")

            cur.execute(
                "SELECT metadata->>'act', count(*) FROM legal_document_chunks GROUP BY metadata->>'act' ORDER BY metadata->>'act'"
            )
            acts = cur.fetchall()
            print("Acts:")
            for act, act_count in acts:
                print(f"  {act}: {act_count}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
