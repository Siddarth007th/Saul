from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import math
import os
import re
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent
LAW_FILES = [
    BASE_DIR / "laws.txt",
    BASE_DIR / "laws_expanded.txt",
]
REPORT_FILE = BASE_DIR / "past_reports.txt"

EMBED_MODEL_NAME = os.getenv("SAULGPT_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDING_MODE = os.getenv("SAULGPT_USE_EMBEDDINGS", "auto").strip().lower()
EMBED_LOCAL_ONLY = os.getenv("SAULGPT_EMBED_LOCAL_ONLY", "1").strip().lower() not in {"0", "false", "off", "no"}
MIN_RANK_SCORE = float(os.getenv("SAULGPT_MIN_RETRIEVAL_SCORE", "0.15"))
LEXICAL_BLEND_WEIGHT = 0.76
SEMANTIC_BLEND_WEIGHT = 0.24

STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "for",
    "on",
    "and",
    "or",
    "with",
    "by",
    "what",
    "how",
    "do",
    "does",
    "is",
    "are",
    "can",
    "i",
    "my",
    "me",
    "we",
    "our",
    "you",
    "your",
    "vs",
    "under",
    "as",
    "at",
    "from",
    "about",
    "into",
    "this",
    "that",
    "these",
    "those",
    "law",
    "laws",
    "legal",
    "act",
    "acts",
    "code",
    "rules",
    "rule",
    "procedure",
    "indian",
    "india",
}

TOKEN_NORMALIZATION = {
    "rental": "rent",
    "rentals": "rent",
    "tenancy": "tenant",
    "tenancies": "tenant",
    "owners": "owner",
    "ownership": "owner",
    "landowners": "owner",
    "landlord": "landlord",
    "landlords": "landlord",
    "deposits": "deposit",
    "wages": "wage",
    "salary": "wage",
    "salaries": "wage",
    "refunded": "refund",
    "refunds": "refund",
    "scammed": "scam",
    "fraudulent": "fraud",
    "forgeries": "forgery",
    "documents": "document",
    "records": "record",
    "witnesses": "witness",
    "timelines": "timeline",
    "chronological": "timeline",
    "proofs": "proof",
    "complaints": "complaint",
    "drafting": "draft",
    "reports": "report",
    "lawyerly": "lawyer",
    "advocates": "advocate",
    "harassed": "harassment",
    "hacking": "hack",
    "hacked": "hack",
}

QUERY_SYNONYMS = {
    "rent": {"rental", "tenant", "tenancy", "landlord", "lease", "deposit", "eviction"},
    "owner": {"ownership", "title", "deed", "registry", "mutation", "possession", "land"},
    "wage": {"salary", "employment", "employer", "employee", "dues", "payslip", "settlement"},
    "fraud": {"scam", "cheating", "forgery", "deception", "fake", "impersonation"},
    "property": {"land", "owner", "title", "deed", "possession", "mutation", "registry"},
    "consumer": {"defect", "service", "refund", "warranty", "replacement", "seller"},
    "family": {"maintenance", "custody", "inheritance", "marriage", "domestic", "support"},
    "contract": {"agreement", "breach", "payment", "refund", "vendor", "invoice"},
    "cyber": {"upi", "otp", "phishing", "hack", "account", "online", "digital"},
    "report": {"draft", "complaint", "summary", "timeline", "evidence", "witness", "format"},
    "document": {"proof", "evidence", "receipt", "invoice", "agreement", "record", "paper"},
    "witness": {"eyewitness", "statement", "testimony", "present"},
}

DOMAIN_HINTS = {
    "property": {"property", "land", "owner", "tenant", "landlord", "lease", "title", "deed", "mutation", "rent"},
    "employment": {"salary", "wage", "employment", "employer", "employee", "dues", "termination", "hr", "settlement"},
    "consumer": {"consumer", "defect", "product", "service", "refund", "warranty", "seller", "replacement"},
    "family": {"family", "maintenance", "custody", "marriage", "inheritance", "domestic", "separation", "support"},
    "cyber": {"cyber", "upi", "otp", "phishing", "hack", "account", "online", "digital", "scam"},
    "fraud": {"fraud", "cheating", "forgery", "scam", "deception", "impersonation", "fake"},
    "contract": {"contract", "agreement", "breach", "payment", "vendor", "settlement", "invoice"},
}


@dataclass
class DocFeatures:
    section_tf: Counter[str]
    text_tf: Counter[str]
    source_tf: Counter[str]
    combined_tf: Counter[str]
    token_set: Set[str]
    bigrams: Set[str]
    chargrams: Set[str]
    normalized_section: str
    normalized_text: str
    section_len: int
    text_len: int
    source_len: int


@dataclass
class QueryPlan:
    clean_query: str
    original_tokens: Tuple[str, ...]
    query_weights: Dict[str, float]
    normalized_query: str
    bigrams: Set[str]
    chargrams: Set[str]
    domain: Optional[str]
    wants_uploaded: bool
    wants_documents: bool
    wants_guidance: bool
    wants_report_style: bool
    wants_general_info: bool
    wants_comparison: bool
    wants_process: bool
    wants_checklist: bool


@dataclass
class CorpusState:
    docs: List[Dict[str, str]]
    features: List[DocFeatures]
    df: Counter[str]
    avg_section_len: float
    avg_text_len: float
    avg_source_len: float
    key: Tuple[int, int]

    @property
    def doc_count(self) -> int:
        return len(self.docs)


def _embedding_requested() -> bool:
    return EMBEDDING_MODE not in {"0", "false", "off", "no"}


USE_EMBEDDINGS = _embedding_requested()


def _stem_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _normalize_token(token: str) -> str:
    normalized = token.lower().strip()
    normalized = TOKEN_NORMALIZATION.get(normalized, normalized)
    normalized = _stem_token(normalized)
    normalized = TOKEN_NORMALIZATION.get(normalized, normalized)
    return normalized


def _tokenize_terms(text: str, drop_stopwords: bool = True) -> List[str]:
    raw_tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    tokens: List[str] = []
    for raw in raw_tokens:
        normalized = _normalize_token(raw)
        if len(normalized) <= 1:
            continue
        if drop_stopwords and normalized in STOPWORDS:
            continue
        tokens.append(normalized)
    return tokens


def _tokenize(text: str) -> Set[str]:
    return set(_tokenize_terms(text))


def _ngrams(tokens: Sequence[str], size: int) -> Set[str]:
    if len(tokens) < size:
        return set()
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def _chargrams(text: str, size: int = 3) -> Set[str]:
    cleaned = re.sub(r"[^a-z0-9]+", "", text.lower())
    if not cleaned:
        return set()
    if len(cleaned) <= size:
        return {cleaned}
    return {cleaned[i : i + size] for i in range(len(cleaned) - size + 1)}


def _normalized_token_text(tokens: Sequence[str]) -> str:
    return " ".join(tokens).strip()


def _dominant_domain(tokens: Set[str]) -> Optional[str]:
    best_domain: Optional[str] = None
    best_score = 0
    for domain, hints in DOMAIN_HINTS.items():
        score = len(tokens.intersection(hints))
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain


def _expand_query_weights(tokens: Sequence[str], domain: Optional[str]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for token in tokens:
        weights[token] = max(weights.get(token, 0.0), 1.0)

    for token in tokens:
        for key, values in QUERY_SYNONYMS.items():
            if token == key or token in values:
                weights[key] = max(weights.get(key, 0.0), 0.95)
                for value in values:
                    weights[value] = max(weights.get(value, 0.0), 0.42)

    if domain:
        for hint in DOMAIN_HINTS.get(domain, set()):
            weights[hint] = max(weights.get(hint, 0.0), 0.24)

    return weights


def _query_plan(clean_query: str) -> QueryPlan:
    raw_tokens = tuple(dict.fromkeys(_tokenize_terms(clean_query)))
    domain = _dominant_domain(set(raw_tokens))
    weighted_tokens = _expand_query_weights(raw_tokens, domain)
    normalized_query = _normalized_token_text(raw_tokens)
    lower = clean_query.lower()

    return QueryPlan(
        clean_query=clean_query,
        original_tokens=raw_tokens,
        query_weights=weighted_tokens,
        normalized_query=normalized_query,
        bigrams=_ngrams(raw_tokens, 2),
        chargrams=_chargrams(clean_query),
        domain=domain,
        wants_uploaded=any(
            token in lower
            for token in ("upload", "uploaded", "document", "documents", "file", "attached", "attachment")
        ),
        wants_documents=any(
            token in lower
            for token in ("what documents", "which documents", "checklist", "proof", "evidence", "records")
        ),
        wants_guidance=any(
            token in lower
            for token in ("how", "what should", "next step", "guide", "timeline", "witness", "preserve", "what happens")
        ),
        wants_report_style=any(
            token in lower
            for token in ("report", "draft", "format", "lawyer style", "past report", "previous report", "complaint")
        ),
        wants_general_info=any(
            token in lower
            for token in (
                "explain",
                "overview",
                "difference",
                "what is",
                "what are",
                "rights",
                "remedies",
                "general law",
                "general legal",
                "vs",
                "versus",
                "owner rights",
                "tenant rights",
            )
        ),
        wants_comparison=any(
            token in lower
            for token in ("difference", "compare", "comparison", "vs", "versus", "owner rights", "tenant rights")
        ),
        wants_process=any(
            token in lower
            for token in (
                "process",
                "procedure",
                "what happens",
                "how to proceed",
                "what next",
                "steps",
                "complaint",
                "fir",
                "bail",
            )
        ),
        wants_checklist=any(
            token in lower
            for token in ("checklist", "what documents", "which documents", "what proof", "what evidence")
        ),
    )


def _sanitize_row_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _load_laws() -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str, str, str]] = set()

    for law_file in LAW_FILES:
        if not law_file.exists():
            continue
        for line in law_file.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue

            parts = [part.strip() for part in raw.split("|")]
            if len(parts) != 4:
                continue

            section, text, source, effective_date = parts
            if not section or not text:
                continue

            key = (section, text, source, effective_date)
            if key in seen:
                continue
            seen.add(key)

            docs.append(
                {
                    "kind": "law",
                    "id": section,
                    "section": section,
                    "text": _sanitize_row_text(text),
                    "source": source,
                    "effective_date": effective_date,
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
        text = _sanitize_row_text(f"{title}. {summary}. Style: {style_notes}")
        docs.append(
            {
                "kind": "report",
                "id": report_id,
                "section": title,
                "text": text,
                "source": source,
                "effective_date": report_date,
            }
        )

    return docs


DOCUMENTS = _load_laws() + _load_reports()
if not DOCUMENTS:
    raise RuntimeError("No legal entries found in configured knowledge files/past_reports.txt")

UPLOADED_DOCUMENTS: List[Dict[str, str]] = []
_UPLOAD_LOCK = Lock()
_STATE_LOCK = Lock()
_EMBED_LOCK = Lock()
_UPLOAD_VERSION = 0
_STATE_CACHE: Optional[CorpusState] = None
_STATE_KEY: Optional[Tuple[int, int]] = None
_EMBED_MODEL: Optional[Any] = None
_EMBED_INDEX: Optional[Any] = None
_EMBED_KEY: Optional[Tuple[int, int]] = None
_EMBED_WARNING_SHOWN = False


def add_uploaded_document_chunks(filename: str, document_type: str, chunks: List[str]) -> int:
    global _UPLOAD_VERSION

    if not chunks:
        return 0

    uploaded_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _UPLOAD_LOCK:
        start = len(UPLOADED_DOCUMENTS) + 1
        for offset, chunk in enumerate(chunks):
            safe_text = _sanitize_row_text(chunk)
            if not safe_text:
                continue
            doc_id = f"UPLOAD-{start + offset}"
            UPLOADED_DOCUMENTS.append(
                {
                    "kind": "uploaded",
                    "id": doc_id,
                    "section": filename,
                    "text": safe_text,
                    "source": f"user_upload:{document_type}",
                    "effective_date": uploaded_at,
                }
            )
        _UPLOAD_VERSION += 1
    return len(chunks)


def _get_corpus_snapshot() -> Tuple[List[Dict[str, str]], Tuple[int, int]]:
    with _UPLOAD_LOCK:
        uploaded = [dict(doc) for doc in UPLOADED_DOCUMENTS]
        version = _UPLOAD_VERSION
    docs = DOCUMENTS + uploaded
    return docs, (len(docs), version)


def _build_doc_features(doc: Dict[str, str]) -> DocFeatures:
    section_terms = _tokenize_terms(str(doc.get("section", "")))
    text_terms = _tokenize_terms(str(doc.get("text", "")))
    source_terms = _tokenize_terms(str(doc.get("source", "")))

    section_tf = Counter(section_terms)
    text_tf = Counter(text_terms)
    source_tf = Counter(source_terms)
    combined_tf = section_tf + text_tf + source_tf
    token_set = set(combined_tf.keys())
    sequence = section_terms + text_terms

    return DocFeatures(
        section_tf=section_tf,
        text_tf=text_tf,
        source_tf=source_tf,
        combined_tf=combined_tf,
        token_set=token_set,
        bigrams=_ngrams(sequence, 2),
        chargrams=_chargrams(f"{doc.get('section', '')}. {doc.get('text', '')}"),
        normalized_section=_normalized_token_text(section_terms),
        normalized_text=_normalized_token_text(text_terms),
        section_len=max(1, len(section_terms)),
        text_len=max(1, len(text_terms)),
        source_len=max(1, len(source_terms)),
    )


def _build_corpus_state(docs: List[Dict[str, str]], key: Tuple[int, int]) -> CorpusState:
    features = [_build_doc_features(doc) for doc in docs]
    df: Counter[str] = Counter()
    for feature in features:
        df.update(feature.token_set)

    avg_section = sum(feature.section_len for feature in features) / max(1, len(features))
    avg_text = sum(feature.text_len for feature in features) / max(1, len(features))
    avg_source = sum(feature.source_len for feature in features) / max(1, len(features))

    return CorpusState(
        docs=docs,
        features=features,
        df=df,
        avg_section_len=max(1.0, avg_section),
        avg_text_len=max(1.0, avg_text),
        avg_source_len=max(1.0, avg_source),
        key=key,
    )


def _get_corpus_state() -> CorpusState:
    global _STATE_CACHE, _STATE_KEY

    docs, key = _get_corpus_snapshot()
    with _STATE_LOCK:
        if _STATE_CACHE is not None and _STATE_KEY == key:
            return _STATE_CACHE
        _STATE_CACHE = _build_corpus_state(docs, key)
        _STATE_KEY = key
        return _STATE_CACHE


def _idf(state: CorpusState, term: str) -> float:
    df = state.df.get(term, 0)
    return math.log(1.0 + ((state.doc_count - df + 0.5) / (df + 0.5)))


def _bm25(
    query_weights: Dict[str, float],
    term_counter: Counter[str],
    doc_len: int,
    avg_len: float,
    state: CorpusState,
    k1: float = 1.6,
    b: float = 0.75,
) -> float:
    score = 0.0
    length_norm = k1 * (1.0 - b + b * (doc_len / max(1.0, avg_len)))
    for term, query_weight in query_weights.items():
        tf = term_counter.get(term, 0)
        if not tf:
            continue
        idf = _idf(state, term)
        score += query_weight * idf * ((tf * (k1 + 1.0)) / (tf + length_norm))
    return score


def _tfidf_overlap(query_weights: Dict[str, float], feature: DocFeatures, state: CorpusState) -> float:
    dot = 0.0
    query_norm = 0.0
    doc_norm = 0.0
    for term, query_weight in query_weights.items():
        idf = _idf(state, term)
        q = query_weight * idf
        query_norm += q * q
        tf = feature.combined_tf.get(term, 0)
        if not tf:
            continue
        d = (1.0 + math.log(tf)) * idf
        dot += q * d
        doc_norm += d * d
    if dot <= 0.0 or query_norm <= 0.0 or doc_norm <= 0.0:
        return 0.0
    return dot / math.sqrt(query_norm * doc_norm)


def _coverage_score(plan: QueryPlan, feature: DocFeatures) -> float:
    if not plan.original_tokens:
        return 0.0
    overlap = len(set(plan.original_tokens).intersection(feature.token_set))
    return overlap / max(1, len(set(plan.original_tokens)))


def _phrase_score(plan: QueryPlan, feature: DocFeatures) -> float:
    if not plan.original_tokens:
        return 0.0

    score = 0.0
    if plan.normalized_query and len(plan.original_tokens) >= 2:
        if plan.normalized_query in feature.normalized_section:
            score += 1.0
        elif plan.normalized_query in feature.normalized_text:
            score += 0.75

    if plan.bigrams:
        overlap = len(plan.bigrams.intersection(feature.bigrams))
        score += 0.45 * (overlap / max(1, len(plan.bigrams)))

    return min(score, 1.0)


def _char_similarity(plan: QueryPlan, feature: DocFeatures) -> float:
    if not plan.chargrams or not feature.chargrams:
        return 0.0
    overlap = len(plan.chargrams.intersection(feature.chargrams))
    union = len(plan.chargrams.union(feature.chargrams))
    if union == 0:
        return 0.0
    return overlap / union


def _doc_matches_domain(doc: Dict[str, str], domain: Optional[str]) -> bool:
    if not domain:
        return False
    hints = DOMAIN_HINTS.get(domain, set())
    blob = f"{doc.get('section', '')} {doc.get('text', '')} {doc.get('source', '')}".lower()
    return any(hint in blob for hint in hints)


def _doc_profile(doc: Dict[str, str]) -> Dict[str, bool]:
    title = str(doc.get("section", "")).lower()
    source = str(doc.get("source", "")).lower()
    blob = f"{title} {source}"
    return {
        "is_uploaded": str(doc.get("kind", "")).lower() == "uploaded",
        "is_report": str(doc.get("kind", "")).lower() == "report",
        "is_statute": bool(
            re.match(r"^(section|order)\b", title)
            or " india code " in f" {source} "
            or " act " in f" {blob} "
            or " code " in f" {blob} "
        ),
        "is_procedure": title.startswith("procedure -"),
        "is_guide": "guide" in title or "playbook" in source,
        "is_checklist": "checklist" in title or "document list" in title,
        "is_overview": "overview" in title or "positions" in title or "primer" in title,
        "is_drafting": "draft" in title or "format" in title or "complaint" in title or "report" in title,
    }


def _score_with_boost(base_score: float, doc: Dict[str, str], plan: QueryPlan) -> float:
    boosted = base_score
    kind = str(doc.get("kind", "")).lower()
    section_lower = str(doc.get("section", "")).lower()
    source_lower = str(doc.get("source", "")).lower()
    profile = _doc_profile(doc)

    if kind == "uploaded":
        boosted += 0.08
        if plan.wants_uploaded:
            boosted += 0.12

    if plan.wants_report_style and kind == "report":
        boosted += 0.16
    elif plan.wants_report_style and any(token in section_lower or token in source_lower for token in ("draft", "report", "format", "procedure")):
        boosted += 0.08

    if plan.wants_documents and any(
        token in section_lower or token in source_lower
        for token in ("checklist", "proof", "document", "procedure", "guide")
    ):
        boosted += 0.08

    if plan.wants_guidance and any(
        token in section_lower or token in source_lower
        for token in ("guide", "procedure", "playbook", "checklist", "draft")
    ):
        boosted += 0.07

    if plan.wants_general_info and kind == "law":
        boosted += 0.03
        if profile["is_guide"] or profile["is_overview"] or profile["is_checklist"]:
            boosted += 0.10
        if profile["is_procedure"]:
            boosted -= 0.06
        if profile["is_statute"]:
            boosted -= 0.08

    if plan.wants_comparison and (profile["is_overview"] or profile["is_guide"]):
        boosted += 0.08

    if plan.wants_process:
        if profile["is_procedure"] or profile["is_drafting"]:
            boosted += 0.10
        if profile["is_statute"] and not profile["is_procedure"]:
            boosted -= 0.03

    if plan.wants_checklist:
        if profile["is_checklist"] or ("proof" in section_lower) or ("document" in section_lower):
            boosted += 0.12

    if plan.wants_report_style and profile["is_drafting"]:
        boosted += 0.10
    if plan.wants_report_style and profile["is_statute"]:
        boosted -= 0.04

    if _doc_matches_domain(doc, plan.domain):
        boosted += 0.08
    elif plan.domain:
        boosted -= 0.01

    for token in plan.original_tokens[:3]:
        if token and token in section_lower:
            boosted += 0.02

    if any(token in plan.original_tokens for token in ("owner", "title", "deed", "mutation", "proof")):
        if any(token in section_lower for token in ("ownership", "title", "deed", "mutation", "proof", "registry")):
            boosted += 0.12
        if "encroachment" in section_lower and "encroachment" not in plan.original_tokens:
            boosted -= 0.07

    if any(token in plan.original_tokens for token in ("rent", "tenant", "landlord", "lease")) and plan.wants_general_info:
        if any(token in section_lower for token in ("rent control", "rental law", "tenant", "lease", "positions", "overview")):
            boosted += 0.11
        if "arrears" in section_lower and "arrears" not in plan.original_tokens:
            boosted -= 0.05
        if "ownership" in section_lower and not any(token in section_lower for token in ("tenant", "rent", "lease")):
            boosted -= 0.05

    if any(token in plan.original_tokens for token in ("wage", "salary", "employment")) and plan.wants_general_info:
        if "employment guide" in section_lower or "salary rights" in section_lower or "overview" in section_lower:
            boosted += 0.10
        if profile["is_statute"]:
            boosted -= 0.05

    if any(token in plan.original_tokens for token in ("owner", "title", "deed", "mutation", "land")):
        if any(token in section_lower for token in ("rental law", "tenant", "rent control")) and "rent" not in plan.original_tokens:
            boosted -= 0.04

    return max(boosted, 0.0)


def _ensure_embeddings(state: CorpusState) -> bool:
    global _EMBED_MODEL, _EMBED_INDEX, _EMBED_KEY, _EMBED_WARNING_SHOWN

    if not _embedding_requested():
        return False

    with _EMBED_LOCK:
        if _EMBED_INDEX is not None and _EMBED_KEY == state.key:
            return True

        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            if _EMBED_MODEL is None:
                _EMBED_MODEL = SentenceTransformer(
                    EMBED_MODEL_NAME,
                    local_files_only=EMBED_LOCAL_ONLY,
                )

            corpus = [
                f"{doc.get('section', '')}. {doc.get('text', '')}. Source: {doc.get('source', '')}"
                for doc in state.docs
            ]
            embeddings = _EMBED_MODEL.encode(
                corpus,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)

            _EMBED_INDEX = index
            _EMBED_KEY = state.key
            return True
        except Exception as exc:
            if not _EMBED_WARNING_SHOWN:
                print(f"[legal_rag] Embeddings unavailable; using lexical retrieval only: {exc}")
                _EMBED_WARNING_SHOWN = True
            _EMBED_INDEX = None
            _EMBED_KEY = None
            return False


def _semantic_scores(plan: QueryPlan, state: CorpusState, top_k: int) -> Dict[int, float]:
    if not _ensure_embeddings(state):
        return {}

    import faiss

    query_embedding = _EMBED_MODEL.encode(
        [plan.clean_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    candidate_k = min(max(top_k * 8, 24), state.doc_count)
    similarities, indices = _EMBED_INDEX.search(query_embedding, candidate_k)

    positive_scores = [float(score) for score in similarities[0] if float(score) > 0.0]
    max_score = max(positive_scores) if positive_scores else 1.0
    semantic: Dict[int, float] = {}

    for raw_score, idx in zip(similarities[0], indices[0]):
        if idx < 0:
            continue
        score = max(0.0, float(raw_score))
        if score <= 0.0:
            continue
        semantic[int(idx)] = score / max_score

    return semantic


def _base_scores(plan: QueryPlan, state: CorpusState) -> Dict[int, float]:
    lexical_scores: Dict[int, float] = {}
    tfidf_scores: Dict[int, float] = {}
    coverage_scores: Dict[int, float] = {}
    phrase_scores: Dict[int, float] = {}
    char_scores: Dict[int, float] = {}

    for idx, feature in enumerate(state.features):
        section_score = _bm25(plan.query_weights, feature.section_tf, feature.section_len, state.avg_section_len, state, b=0.35)
        text_score = _bm25(plan.query_weights, feature.text_tf, feature.text_len, state.avg_text_len, state, b=0.82)
        source_score = _bm25(plan.query_weights, feature.source_tf, feature.source_len, state.avg_source_len, state, b=0.25)
        lexical_scores[idx] = (1.7 * section_score) + (1.15 * text_score) + (0.45 * source_score)
        tfidf_scores[idx] = _tfidf_overlap(plan.query_weights, feature, state)
        coverage_scores[idx] = _coverage_score(plan, feature)
        phrase_scores[idx] = _phrase_score(plan, feature)
        char_scores[idx] = _char_similarity(plan, feature)

    max_lexical = max(lexical_scores.values()) if lexical_scores else 1.0
    max_tfidf = max(tfidf_scores.values()) if tfidf_scores else 1.0

    scores: Dict[int, float] = {}
    for idx in range(state.doc_count):
        lexical_component = lexical_scores[idx] / max(1.0, max_lexical)
        tfidf_component = tfidf_scores[idx] / max(1.0, max_tfidf)
        scores[idx] = (
            0.50 * lexical_component
            + 0.18 * tfidf_component
            + 0.14 * coverage_scores[idx]
            + 0.12 * phrase_scores[idx]
            + 0.06 * char_scores[idx]
        )
    return scores


def _result_similarity(left: DocFeatures, right: DocFeatures) -> float:
    token_union = len(left.token_set.union(right.token_set))
    token_score = 0.0
    if token_union:
        token_score = len(left.token_set.intersection(right.token_set)) / token_union

    bigram_union = len(left.bigrams.union(right.bigrams))
    bigram_score = 0.0
    if bigram_union:
        bigram_score = len(left.bigrams.intersection(right.bigrams)) / bigram_union

    return (0.72 * token_score) + (0.28 * bigram_score)


def _diverse_rerank(candidates: List[Dict[str, float]], state: CorpusState, top_k: int) -> List[Dict[str, float]]:
    selected: List[Dict[str, float]] = []
    remaining = list(candidates)
    source_counts: Counter[str] = Counter()

    while remaining and len(selected) < top_k:
        best: Optional[Dict[str, float]] = None
        best_adjusted = -1.0

        for candidate in remaining:
            idx = int(candidate["idx"])
            source = str(state.docs[idx].get("source", ""))
            source_penalty = 0.03 * source_counts[source]
            similarity_penalty = 0.0
            if selected:
                similarity_penalty = max(
                    _result_similarity(state.features[idx], state.features[int(item["idx"])])
                    for item in selected
                )

            adjusted = float(candidate["score"]) - (0.14 * similarity_penalty) - source_penalty
            candidate["adjusted"] = adjusted
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best = candidate

        if best is None:
            break

        remaining.remove(best)
        idx = int(best["idx"])
        if selected:
            duplicate_level = max(
                _result_similarity(state.features[idx], state.features[int(item["idx"])])
                for item in selected
            )
            if duplicate_level >= 0.94:
                continue

        selected.append(best)
        source_counts[str(state.docs[idx].get("source", ""))] += 1

    return selected


def search_knowledge(query: str, top_k: int = 8) -> List[Dict[str, str]]:
    clean_query = " ".join((query or "").split()).strip()
    if not clean_query:
        return []

    state = _get_corpus_state()
    plan = _query_plan(clean_query)
    base_scores = _base_scores(plan, state)
    semantic_scores = _semantic_scores(plan, state, top_k=top_k)

    candidates: List[Dict[str, float]] = []
    for idx, doc in enumerate(state.docs):
        base_score = base_scores[idx]
        semantic_score = semantic_scores.get(idx, 0.0)
        blended = base_score
        if semantic_scores:
            blended = (LEXICAL_BLEND_WEIGHT * base_score) + (SEMANTIC_BLEND_WEIGHT * semantic_score)
        score = _score_with_boost(blended, doc, plan)
        if score >= MIN_RANK_SCORE:
            candidates.append({"idx": float(idx), "score": score})

    if not candidates:
        fallback = sorted(
            (
                {"idx": float(idx), "score": _score_with_boost(score, state.docs[idx], plan)}
                for idx, score in base_scores.items()
            ),
            key=lambda item: item["score"],
            reverse=True,
        )[: max(top_k, 5)]
        candidates = [item for item in fallback if item["score"] > 0.0]

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = _diverse_rerank(candidates[: max(top_k * 4, 18)], state, top_k)

    results: List[Dict[str, str]] = []
    for item in selected:
        idx = int(item["idx"])
        doc = dict(state.docs[idx])
        doc["score"] = round(float(item["score"]), 4)
        results.append(doc)
    return results


def search_law(query: str, top_k: int = 4) -> List[Dict[str, str]]:
    results = search_knowledge(query, top_k=max(top_k + 4, 8))
    filtered = [doc for doc in results if doc.get("kind") == "law"]
    return filtered[:top_k]
