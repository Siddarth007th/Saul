from __future__ import annotations

from importlib import import_module
import re
from typing import Protocol, cast

from app.core.config import settings
from app.models.schemas import SearchResult


class EmbeddingVector(Protocol):
    def tolist(self) -> list[float]: ...


class EmbeddingModel(Protocol):
    def encode(
        self,
        sentences: str,
        *,
        normalize_embeddings: bool = False,
    ) -> EmbeddingVector: ...


class EmbeddingModelFactory(Protocol):
    def __call__(self, model_name: str) -> EmbeddingModel: ...


class CursorProtocol(Protocol):
    def __enter__(self) -> CursorProtocol: ...

    def __exit__(
        self, exc_type: object, exc_value: object, traceback: object
    ) -> None: ...

    def execute(
        self, query: str, params: tuple[object, ...] | None = None
    ) -> object: ...

    def fetchall(self) -> list[tuple[str, dict[str, object], float]]: ...


class ConnectionProtocol(Protocol):
    def cursor(self) -> CursorProtocol: ...

    def close(self) -> None: ...


class ConnectionFactory(Protocol):
    def __call__(self, dsn: str) -> ConnectionProtocol: ...


_model: EmbeddingModel | None = None

_ACT_PATTERN = re.compile(r"\b(BNSS|BSA|BNS|IPC|CONSTITUTION)\b", re.IGNORECASE)
_SECTION_PATTERN = re.compile(
    r"(?:section|sec\.?|§|article)\s*(\d+[A-Z]?)", re.IGNORECASE
)
_FIR_PROCEDURE_QUERY_PATTERN = re.compile(
    r"\b(?:fir|investigation|complaint|procedure)\b", re.IGNORECASE
)
_STOPWORDS = {
    "what",
    "under",
    "with",
    "find",
    "relevant",
    "section",
    "sections",
    "procedure",
    "draft",
    "application",
    "equivalent",
    "file",
    "bns",
    "bnss",
    "bsa",
    "ipc",
}
_QUERY_SYNONYMS = {
    "fir": [
        "first",
        "information",
        "report",
        "cognizable",
        "offence",
        "informant",
        "police",
        "station",
    ],
    "cyber": ["electronic", "online"],
    "defamation": ["defamation"],
}
_STATUTORY_SOURCE_CLAUSE = (
    "(metadata->>'source_type' IS NULL OR metadata->>'source_type' = 'statutory')"
)
_UPLOADED_SOURCE_CLAUSE = "metadata->>'source_type' = 'uploaded'"


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def get_embedding_model() -> EmbeddingModel:
    global _model
    if _model is None:
        sentence_transformers = import_module("sentence_transformers")
        model_class = cast(
            EmbeddingModelFactory,
            getattr(sentence_transformers, "SentenceTransformer"),
        )
        _model = model_class(settings.EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> list[float]:
    model = get_embedding_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def _extract_acts(query: str) -> list[str]:
    acts: list[str] = []
    for match in _ACT_PATTERN.finditer(query):
        act = match.group(1).upper()
        if act not in acts:
            acts.append(act)
    return acts


def _extract_section(query: str) -> str | None:
    match = _SECTION_PATTERN.search(query)
    if match is None:
        return None
    return match.group(1).upper()


def _keyword_terms(query: str) -> list[str]:
    terms: list[str] = []
    tokens = cast(list[str], re.findall(r"[A-Za-z]{3,}", query.lower()))
    for token in tokens:
        if token in _STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
        for synonym in _QUERY_SYNONYMS.get(token, []):
            if synonym not in terms:
                terms.append(synonym)
    return terms


def _row_key(content: str, metadata: dict[str, object]) -> tuple[object, ...]:
    return (
        metadata.get("source_type"),
        metadata.get("filename"),
        metadata.get("act"),
        metadata.get("section"),
        metadata.get("chunk_index"),
        content[:120],
    )


def _fetch_exact_section_matches(
    cur: CursorProtocol, acts: list[str], section: str, limit: int
) -> list[tuple[str, dict[str, object], float]]:
    rows: list[tuple[str, dict[str, object], float]] = []
    for act in acts:
        _ = cur.execute(
            "SELECT content, metadata, 1.0 AS score FROM legal_document_chunks"
            " WHERE "
            + _STATUTORY_SOURCE_CLAUSE
            + " AND metadata->>'act' = %s AND metadata->>'section' = %s"
            " ORDER BY COALESCE((metadata->>'chunk_index')::int, 0) LIMIT %s",
            (act, section, limit),
        )
        rows.extend(cur.fetchall())
    return rows


def _fetch_statutory_keyword_matches(
    cur: CursorProtocol, terms: list[str], acts: list[str] | None, limit: int
) -> list[tuple[str, dict[str, object], float]]:
    if not terms:
        return []

    clauses: list[str] = [_STATUTORY_SOURCE_CLAUSE]
    where_params: list[object] = []

    if acts:
        clauses.append("metadata->>'act' = ANY(%s)")
        where_params.append(acts)

    match_clauses: list[str] = []
    score_clauses: list[str] = []
    score_params: list[object] = []
    for term in terms:
        like_term = f"%{term}%"
        match_clauses.append("lower(metadata->>'title') LIKE %s")
        where_params.append(like_term)
        match_clauses.append("lower(content) LIKE %s")
        where_params.append(like_term)
        score_clauses.append(
            "CASE WHEN lower(metadata->>'title') LIKE %s THEN 2 ELSE 0 END"
        )
        score_params.append(like_term)
        score_clauses.append("CASE WHEN lower(content) LIKE %s THEN 1 ELSE 0 END")
        score_params.append(like_term)

    clauses.append("(" + " OR ".join(match_clauses) + ")")
    params = [*score_params, *where_params, limit]

    _ = cur.execute(
        "SELECT content, metadata, ("
        + " + ".join(score_clauses)
        + ")::float AS score FROM legal_document_chunks WHERE "
        + " AND ".join(clauses)
        + " ORDER BY score DESC, COALESCE((metadata->>'chunk_index')::int, 0)"
        + " LIMIT %s",
        tuple(params),
    )
    return cur.fetchall()


def _fetch_uploaded_keyword_matches(
    cur: CursorProtocol, terms: list[str], limit: int
) -> list[tuple[str, dict[str, object], float]]:
    if not terms:
        return []

    clauses: list[str] = [_UPLOADED_SOURCE_CLAUSE]
    where_params: list[object] = []
    match_clauses: list[str] = []
    score_clauses: list[str] = []
    score_params: list[object] = []

    for term in terms:
        like_term = f"%{term}%"
        match_clauses.append("lower(metadata->>'filename') LIKE %s")
        where_params.append(like_term)
        match_clauses.append("lower(metadata->>'title') LIKE %s")
        where_params.append(like_term)
        match_clauses.append("lower(content) LIKE %s")
        where_params.append(like_term)
        score_clauses.append(
            "CASE WHEN lower(metadata->>'filename') LIKE %s THEN 2 ELSE 0 END"
        )
        score_params.append(like_term)
        score_clauses.append(
            "CASE WHEN lower(metadata->>'title') LIKE %s THEN 2 ELSE 0 END"
        )
        score_params.append(like_term)
        score_clauses.append("CASE WHEN lower(content) LIKE %s THEN 1 ELSE 0 END")
        score_params.append(like_term)

    clauses.append("(" + " OR ".join(match_clauses) + ")")
    params = [*score_params, *where_params, limit]
    _ = cur.execute(
        "SELECT content, metadata, ("
        + " + ".join(score_clauses)
        + ")::float AS score FROM legal_document_chunks WHERE "
        + " AND ".join(clauses)
        + " ORDER BY score DESC, COALESCE((metadata->>'chunk_index')::int, 0)"
        + " LIMIT %s",
        tuple(params),
    )
    return cur.fetchall()


def _fetch_statutory_vector_matches(
    cur: CursorProtocol, embedding: str, top_k: int, acts: list[str] | None
) -> list[tuple[str, dict[str, object], float]]:
    if acts:
        _ = cur.execute(
            "SELECT content, metadata, 1 - (embedding <=> %s::vector) AS score"
            " FROM legal_document_chunks WHERE "
            + _STATUTORY_SOURCE_CLAUSE
            + " AND metadata->>'act' = ANY(%s)"
            " ORDER BY embedding <=> %s::vector LIMIT %s",
            (embedding, acts, embedding, top_k),
        )
    else:
        _ = cur.execute(
            "SELECT content, metadata, 1 - (embedding <=> %s::vector) AS score"
            " FROM legal_document_chunks WHERE "
            + _STATUTORY_SOURCE_CLAUSE
            + " ORDER BY embedding <=> %s::vector LIMIT %s",
            (embedding, embedding, top_k),
        )
    return cur.fetchall()


def _fetch_uploaded_vector_matches(
    cur: CursorProtocol, embedding: str, top_k: int
) -> list[tuple[str, dict[str, object], float]]:
    _ = cur.execute(
        "SELECT content, metadata, 1 - (embedding <=> %s::vector) AS score"
        " FROM legal_document_chunks WHERE "
        + _UPLOADED_SOURCE_CLAUSE
        + " ORDER BY embedding <=> %s::vector LIMIT %s",
        (embedding, embedding, top_k),
    )
    return cur.fetchall()


def _fetch_section_title(cur: CursorProtocol, act: str, section: str) -> str | None:
    _ = cur.execute(
        "SELECT metadata->>'title' FROM legal_document_chunks WHERE "
        + _STATUTORY_SOURCE_CLAUSE
        + " AND metadata->>'act' = %s AND metadata->>'section' = %s"
        " ORDER BY COALESCE((metadata->>'chunk_index')::int, 0) LIMIT 1",
        (act, section),
    )
    rows = cur.fetchall()
    if not rows:
        return None
    return cast(str | None, rows[0][0])


def _search_statutory_rows(
    cur: CursorProtocol,
    *,
    embedding: str,
    top_k: int,
    mentioned_acts: list[str],
    candidate_acts: list[str],
    is_equivalent_query: bool,
    is_fir_or_procedure_query: bool,
    section: str | None,
    keyword_terms: list[str],
) -> list[tuple[str, dict[str, object], float]]:
    max_results = max(top_k * 2, 10)
    rows: list[tuple[str, dict[str, object], float]] = []
    seen: set[tuple[object, ...]] = set()

    def add_results(candidates: list[tuple[str, dict[str, object], float]]) -> None:
        for content, metadata, score in candidates:
            key = _row_key(content, metadata)
            if key in seen:
                continue
            seen.add(key)
            rows.append((content, metadata, score))
            if len(rows) >= max_results:
                return

    if section and candidate_acts:
        exact_acts = candidate_acts
        if is_equivalent_query and mentioned_acts:
            exact_acts = [mentioned_acts[0]]
        add_results(_fetch_exact_section_matches(cur, exact_acts, section, max_results))

    if len(rows) < max_results and is_fir_or_procedure_query and "BNSS" in candidate_acts:
        add_results(_fetch_exact_section_matches(cur, ["BNSS"], "173", max_results))

    if len(rows) < max_results and is_equivalent_query:
        if len(mentioned_acts) >= 2 and section:
            source_act = mentioned_acts[0]
            target_act = mentioned_acts[-1]
            source_title = _fetch_section_title(cur, source_act, section)
            if source_title:
                add_results(
                    _fetch_statutory_keyword_matches(
                        cur,
                        _keyword_terms(source_title),
                        [target_act],
                        max_results,
                    )
                )

    if len(rows) < max_results:
        keyword_acts = candidate_acts or None
        if is_equivalent_query and len(mentioned_acts) >= 2:
            keyword_acts = [mentioned_acts[-1]]
        add_results(
            _fetch_statutory_keyword_matches(
                cur,
                keyword_terms,
                keyword_acts,
                max_results,
            )
        )

    if len(rows) < max_results:
        add_results(
            _fetch_statutory_vector_matches(
                cur,
                embedding,
                max(max_results * 2, 20),
                candidate_acts or None,
            )
        )

    return rows


def _search_uploaded_rows(
    cur: CursorProtocol, *, embedding: str, top_k: int, keyword_terms: list[str]
) -> list[tuple[str, dict[str, object], float]]:
    max_results = max(top_k * 2, 10)
    rows: list[tuple[str, dict[str, object], float]] = []
    seen: set[tuple[object, ...]] = set()

    def add_results(candidates: list[tuple[str, dict[str, object], float]]) -> None:
        for content, metadata, score in candidates:
            key = _row_key(content, metadata)
            if key in seen:
                continue
            seen.add(key)
            rows.append((content, metadata, score))
            if len(rows) >= max_results:
                return

    if keyword_terms:
        add_results(_fetch_uploaded_keyword_matches(cur, keyword_terms, max_results))

    if len(rows) < max_results:
        add_results(
            _fetch_uploaded_vector_matches(cur, embedding, max(max_results * 2, 20))
        )

    return rows


def search_similar(
    query: str, top_k: int = 5, filter_act: str | None = None
) -> list[SearchResult]:
    embedding = _vector_literal(embed_text(query))
    mentioned_acts = _extract_acts(query)
    candidate_acts = [filter_act] if filter_act else mentioned_acts.copy()
    is_equivalent_query = bool(re.search(r"\bequivalent\b", query, re.IGNORECASE))
    is_fir_or_procedure_query = bool(_FIR_PROCEDURE_QUERY_PATTERN.search(query))
    if is_fir_or_procedure_query:
        if "BNSS" not in candidate_acts:
            candidate_acts.append("BNSS")
    section = _extract_section(query)
    keyword_terms = _keyword_terms(query)
    psycopg2_module = import_module("psycopg2")
    connect = cast(ConnectionFactory, getattr(psycopg2_module, "connect"))
    conn = connect(settings.DATABASE_URL)
    try:
        with conn.cursor() as cur:
            statutory_rows = _search_statutory_rows(
                cur,
                embedding=embedding,
                top_k=top_k,
                mentioned_acts=mentioned_acts,
                candidate_acts=candidate_acts,
                is_equivalent_query=is_equivalent_query,
                is_fir_or_procedure_query=is_fir_or_procedure_query,
                section=section,
                keyword_terms=keyword_terms,
            )
            uploaded_rows = _search_uploaded_rows(
                cur,
                embedding=embedding,
                top_k=top_k,
                keyword_terms=keyword_terms,
            )

        deduped: list[tuple[str, dict[str, object], float]] = []
        seen: set[tuple[object, ...]] = set()
        for content, metadata, score in statutory_rows + uploaded_rows:
            key = _row_key(content, metadata)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((content, metadata, score))

        deduped.sort(key=lambda row: row[2], reverse=True)
        return [
            SearchResult(content=row[0], metadata=row[1], score=float(row[2]))
            for row in deduped[:top_k]
        ]
    finally:
        conn.close()
