from pathlib import Path
from datetime import datetime, timezone
import os
import re
from threading import Lock
from typing import Any, Dict, List, Optional, Set

BASE_DIR = Path(__file__).resolve().parent
LAW_FILE = BASE_DIR / "laws.txt"
REPORT_FILE = BASE_DIR / "past_reports.txt"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
MIN_HYBRID_SCORE = 0.18
LEXICAL_WEIGHT = 0.35
SEMANTIC_WEIGHT = 0.65
USE_EMBEDDINGS = os.getenv("SAULGPT_USE_EMBEDDINGS", "0") == "1"

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "for", "on", "and", "or", "with", "by",
    "what", "how", "do", "does", "is", "are", "can", "i", "my", "we", "our", "you",
    "vs", "under", "as", "at", "from",
    # High-frequency legal boilerplate tokens that harm ranking precision.
    "section", "act", "law", "code", "rules", "procedure", "indian", "india",
}

TOKEN_NORMALIZATION = {
    "rental": "rent",
    "tenancy": "tenant",
    "ownership": "owner",
    "landlord": "landlord",
    "landowners": "owner",
    "mutation": "mutation",
}


def _tokenize(text: str) -> Set[str]:
    raw_tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    tokens: Set[str] = set()
    for token in raw_tokens:
        if len(token) <= 1 or token in STOPWORDS:
            continue
        norm = TOKEN_NORMALIZATION.get(token, token)
        if norm and norm not in STOPWORDS:
            tokens.add(norm)
    return tokens


def _load_laws() -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []

    for line in LAW_FILE.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue

        parts = [part.strip() for part in raw.split("|")]
        if len(parts) != 4:
            continue

        section, text, source, effective_date = parts
        if not section or not text:
            continue

        docs.append(
            {
                "kind": "law",
                "id": section,
                "section": section,
                "text": text,
                "source": source,
                "effective_date": effective_date,
                "_token_blob": f"{section} {text}",
            }
        )

    return docs


def _load_reports() -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    if not REPORT_FILE.exists():
        return docs

    for line in REPORT_FILE.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue

        parts = [part.strip() for part in raw.split("|")]
        if len(parts) != 6:
            continue

        report_id, title, summary, style_notes, source, report_date = parts
        text = f"{title}. {summary}. Style: {style_notes}"

        docs.append(
            {
                "kind": "report",
                "id": report_id,
                "section": title,
                "text": text,
                "source": source,
                "effective_date": report_date,
                "_token_blob": f"{title} {summary} {style_notes}",
            }
        )

    return docs


DOCUMENTS = _load_laws() + _load_reports()
if not DOCUMENTS:
    raise RuntimeError("No legal entries found in laws.txt/past_reports.txt")

DOC_TOKENS = [_tokenize(doc["_token_blob"]) for doc in DOCUMENTS]
UPLOADED_DOCUMENTS: List[Dict[str, str]] = []
UPLOADED_DOC_TOKENS: List[Set[str]] = []
_UPLOAD_LOCK = Lock()

_MODEL: Optional[Any] = None
_INDEX: Optional[Any] = None
_EMBED_READY = False


def add_uploaded_document_chunks(
    filename: str, document_type: str, chunks: List[str]
) -> int:
    if not chunks:
        return 0

    uploaded_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _UPLOAD_LOCK:
        start = len(UPLOADED_DOCUMENTS) + 1
        for offset, chunk in enumerate(chunks):
            doc_id = f"UPLOAD-{start + offset}"
            safe_text = " ".join(chunk.split()).strip()
            doc = {
                "kind": "uploaded",
                "id": doc_id,
                "section": filename,
                "text": safe_text,
                "source": f"user_upload:{document_type}",
                "effective_date": uploaded_at,
                "_token_blob": f"{filename} {safe_text}",
            }
            UPLOADED_DOCUMENTS.append(doc)
            UPLOADED_DOC_TOKENS.append(_tokenize(doc["_token_blob"]))
    return len(chunks)


def _get_corpus_snapshot() -> tuple[List[Dict[str, str]], List[Set[str]]]:
    with _UPLOAD_LOCK:
        if not UPLOADED_DOCUMENTS:
            return DOCUMENTS, DOC_TOKENS
        docs = DOCUMENTS + [dict(d) for d in UPLOADED_DOCUMENTS]
        tokens = DOC_TOKENS + [set(t) for t in UPLOADED_DOC_TOKENS]
        return docs, tokens


def _ensure_embeddings() -> bool:
    global _MODEL, _INDEX, _EMBED_READY

    if _EMBED_READY:
        return _MODEL is not None and _INDEX is not None

    _EMBED_READY = True
    if not USE_EMBEDDINGS:
        return False

    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBED_MODEL_NAME)
        corpus = [f"{d['section']}. {d['text']}" for d in DOCUMENTS]
        emb = model.encode(corpus, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

        _MODEL = model
        _INDEX = index
        return True
    except Exception as exc:
        print(f"[legal_rag] Embeddings unavailable; using lexical retrieval only: {exc}")
        _MODEL = None
        _INDEX = None
        return False


def _lexical_score(query_tokens: Set[str], doc_tokens: Set[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(doc_tokens))
    return overlap / max(1, len(query_tokens))


def search_knowledge(query: str, top_k: int = 6) -> List[Dict[str, str]]:
    clean_query = " ".join((query or "").split()).strip()
    if not clean_query:
        return []

    query_tokens = _tokenize(clean_query)
    ranked: List[Dict[str, str]] = []

    embeddings_ready = _ensure_embeddings()

    # Always compute lexical score as baseline.
    all_docs, all_tokens = _get_corpus_snapshot()

    lexical_scores: Dict[int, float] = {}
    for idx, tokens in enumerate(all_tokens):
        lexical_scores[idx] = _lexical_score(query_tokens, tokens)

    if not embeddings_ready:
        for idx, doc in enumerate(all_docs):
            lexical = lexical_scores[idx]
            if lexical <= 0:
                continue
            result = dict(doc)
            result.pop("_token_blob", None)
            result["score"] = round(lexical, 4)
            ranked.append(result)

        ranked.sort(key=lambda d: d["score"], reverse=True)
        return ranked[:top_k]

    # Hybrid semantic + lexical branch.
    import faiss

    query_emb = _MODEL.encode([clean_query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_emb)

    candidate_k = max(top_k * 3, 12)
    candidate_k = min(candidate_k, len(DOCUMENTS))
    semantic_scores, indices = _INDEX.search(query_emb, candidate_k)

    for semantic_score, idx in zip(semantic_scores[0], indices[0]):
        if idx < 0:
            continue

        lexical = lexical_scores[int(idx)]
        hybrid = (SEMANTIC_WEIGHT * float(semantic_score)) + (LEXICAL_WEIGHT * lexical)
        if hybrid < MIN_HYBRID_SCORE:
            continue

        doc = dict(DOCUMENTS[int(idx)])
        doc.pop("_token_blob", None)
        doc["score"] = round(hybrid, 4)
        ranked.append(doc)

    # Uploaded docs are retrieved lexically so user uploads are always searchable.
    base_count = len(DOCUMENTS)
    for idx in range(base_count, len(all_docs)):
        lexical = lexical_scores[idx]
        if lexical <= 0:
            continue
        doc = dict(all_docs[idx])
        doc.pop("_token_blob", None)
        doc["score"] = round(lexical, 4)
        ranked.append(doc)

    ranked.sort(key=lambda d: d["score"], reverse=True)
    return ranked[:top_k]


def search_law(query: str, top_k: int = 4) -> List[Dict[str, str]]:
    results = search_knowledge(query, top_k=max(top_k + 3, 8))
    filtered = [doc for doc in results if doc.get("kind") == "law"]
    return filtered[:top_k]
