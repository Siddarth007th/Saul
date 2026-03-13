import os
import re
import time
from importlib import import_module
from io import BytesIO
from collections import defaultdict, deque
from typing import Any, Dict, List, Literal, Optional, Tuple
#
import requests
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from legal_rag import add_uploaded_document_chunks, search_knowledge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "index.html")

app = FastAPI(title="SaulGPT API", version="3.4.0")

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500,null",
    ).split(",")
    if origin.strip()
]
ALLOW_ALL_ORIGINS = os.getenv("ALLOW_ALL_ORIGINS", "1") == "1"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_ALL_ORIGINS else ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
MODEL_CANDIDATES = [
    name.strip()
    for name in os.getenv(
        "OLLAMA_MODEL_CANDIDATES",
        "mistral,llama3.1:8b-instruct,qwen2.5:7b-instruct",
    ).split(",")
    if name.strip()
]
if OLLAMA_MODEL not in MODEL_CANDIDATES:
    MODEL_CANDIDATES.insert(0, OLLAMA_MODEL)
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45"))
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "520"))
GENERATE_MAX_TOKENS = int(os.getenv("GENERATE_MAX_TOKENS", "900"))
REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "30"))
API_KEY = os.getenv("SAULGPT_API_KEY")
MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
CHAT_CACHE_TTL_SECONDS = int(os.getenv("CHAT_CACHE_TTL_SECONDS", "900"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "10485760"))  # 10 MB
OLLAMA_CHAT_MODE = os.getenv("USE_OLLAMA_CHAT", "auto").strip().lower()
MAX_FOLLOWUP_QUESTIONS = int(os.getenv("MAX_FOLLOWUP_QUESTIONS", "2"))
OLLAMA_DISCOVERY_TTL_SECONDS = int(os.getenv("OLLAMA_DISCOVERY_TTL_SECONDS", "45"))
OLLAMA_TAGS_URL = (
    OLLAMA_URL.rsplit("/api/generate", 1)[0] + "/api/tags"
    if "/api/generate" in OLLAMA_URL
    else OLLAMA_URL
)

SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt"}

_rate_limit_store: Dict[str, deque] = defaultdict(deque)
_chat_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_ollama_available_cache: Optional[bool] = None
_ollama_last_check_at = 0.0

LAW_FOCUS_GUIDANCE = (
    "I focus on Indian legal information. If you describe the issue in legal terms, "
    "I can explain the likely legal category and general context."
)

NON_ADVICE_NOTE = (
    "This is general legal information, not legal advice or a lawyer-client opinion."
)

_CONTEXT_DEPENDENT_QUERY_RE = re.compile(
    r"\b(?:this|that|it|these|those|same case|same issue|as above|next|then|what next|how to proceed|continue|prepare|deeper|more details|clarify)\b",
    re.IGNORECASE,
)
_DATE_MARKER_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|20\d{2}|yesterday|today|last|since|on\s+\d{1,2}\b)\b",
    re.IGNORECASE,
)
_PARTY_MARKER_RE = re.compile(
    r"\b(?:landlord|tenant|owner|buyer|seller|employer|employee|company|police|neighbor|partner|client|vendor)\b",
    re.IGNORECASE,
)
_DOCUMENT_MARKER_RE = re.compile(
    r"\b(?:agreement|contract|email|message|receipt|invoice|screenshot|recording|document|proof|payslip|statement)\b",
    re.IGNORECASE,
)
_GOAL_MARKER_RE = re.compile(
    r"\b(?:refund|compensation|salary|payment|return|eviction|possession|settlement|action|resolution|recovery)\b",
    re.IGNORECASE,
)
_REPORT_INTENT_RE = re.compile(
    r"\b(?:report|draft|format|complaint draft|prepare report|case summary|legal notice draft|petition draft|fir draft|application draft|make report|write complaint|draft for me)\b",
    re.IGNORECASE,
)
_REPORT_GENERATE_RE = re.compile(
    r"\b(?:generate|create|produce|finalize|make)\b.*\b(?:report|summary|draft)\b|\bready for report\b|\bgenerate now\b",
    re.IGNORECASE,
)
_REPORT_EXIT_RE = re.compile(
    r"\b(?:stop|cancel|exit|skip|close)\b.*\b(?:report|draft|intake|format)\b|\bno report\b",
    re.IGNORECASE,
)
_DATE_CAPTURE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4}|20\d{2}|yesterday|today|last week|last month|last year|this month|this year)\b",
    re.IGNORECASE,
)
_AMOUNT_CAPTURE_RE = re.compile(r"(?:₹|rs\.?|inr)\s*[\d,]+", re.IGNORECASE)
_LOCATION_CAPTURE_RE = re.compile(
    r"\b(?:at|in|near)\s+([A-Za-z][A-Za-z .,-]{2,45}?)(?=$|[.;]|,|\s+and\s+|\s+but\s+|\s+where\s+)",
    re.IGNORECASE,
)
_PARTY_HINT_RE = re.compile(
    r"\b(?:landlord|tenant|owner|buyer|seller|employer|employee|company|police|neighbor|partner|accused|complainant)\b",
    re.IGNORECASE,
)
_REPORT_INTAKE_MARKER = "To prepare your report in Indian format"

_REPORT_COMMON_SLOT_QUESTIONS: Dict[str, str] = {
    "incident_date": "What is the incident date (or date range)?",
    "incident_location": "Where did this happen (city/locality/state)?",
    "parties": "Who are the main parties involved and their roles?",
    "incident_summary": "What exactly happened in 3-5 lines (chronological order)?",
    "timeline": "Can you share a short timeline of key events and dates?",
    "harm_or_amount": "What loss/harm occurred (amount, impact, or risk)?",
    "evidence": "What documents/evidence do you currently have?",
    "witnesses": "Are there any witnesses? If yes, who and what did they observe?",
    "desired_outcome": "What result do you want from this report?",
}

_REPORT_CATEGORY_SLOT_QUESTIONS: Dict[str, Dict[str, str]] = {
    "Property or tenancy issue": {
        "property_details": "What is the property detail (address/type/unit)?",
        "agreement_status": "Do you have rent/lease terms and key dates (start/end/vacate)?",
    },
    "Employment dispute": {
        "employment_details": "What is your role, employer, and employment period?",
        "dues_details": "What dues are pending (salary/benefits) and for which months?",
    },
    "Fraud or cheating concern": {
        "counterparty_details": "Who is the counterparty (person/platform/account identifier)?",
        "transaction_details": "What transactions/events occurred (date, mode, amount)?",
    },
    "Consumer issue": {
        "seller_service_details": "Who is the seller/service provider and what was promised?",
        "purchase_timeline": "When was the purchase/service and what went wrong?",
    },
    "Cyber or online harm": {
        "platform_details": "Which app/platform/account was involved?",
        "digital_trail": "Do you have screenshots, transaction IDs, or account alerts?",
    },
}

SLOT_LABELS: Dict[str, str] = {
    "incident_date": "Incident date",
    "incident_location": "Incident location",
    "parties": "Parties involved",
    "incident_summary": "Background summary",
    "timeline": "Timeline of events",
    "harm_or_amount": "Loss or amount",
    "evidence": "Document evidence",
    "witnesses": "Witness information",
    "desired_outcome": "Desired outcome",
    "property_details": "Property details",
    "agreement_status": "Agreement or tenancy status",
    "employment_details": "Employment details",
    "dues_details": "Pending dues details",
    "counterparty_details": "Counterparty details",
    "transaction_details": "Transaction details",
    "seller_service_details": "Seller/service details",
    "purchase_timeline": "Purchase timeline",
    "platform_details": "Platform/account details",
    "digital_trail": "Digital trail",
}

_REPORT_REQUIRED_BY_CATEGORY: Dict[str, List[str]] = {
    "Property or tenancy issue": ["property_details", "agreement_status"],
    "Employment dispute": ["employment_details", "dues_details"],
    "Fraud or cheating concern": ["counterparty_details", "transaction_details"],
    "Consumer issue": ["seller_service_details", "purchase_timeline"],
    "Cyber or online harm": ["platform_details", "digital_trail"],
}

_REPORT_SLOT_PRIORITY: List[str] = [
    "incident_summary",
    "parties",
    "incident_date",
    "incident_location",
    "timeline",
    "harm_or_amount",
    "evidence",
    "witnesses",
    "desired_outcome",
    "property_details",
    "agreement_status",
    "employment_details",
    "dues_details",
    "counterparty_details",
    "transaction_details",
    "seller_service_details",
    "purchase_timeline",
    "platform_details",
    "digital_trail",
]

WORKFLOW_STAGE_LABELS: Dict[int, str] = {
    1: "Stage 1 - Identify legal category",
    2: "Stage 2 - Collect key facts",
    3: "Stage 3 - Generate structured report",
    4: "Stage 4 - Suggest high-level next steps",
}

WORKFLOW_BASE_REQUIRED_SLOTS: List[str] = [
    "incident_summary",
    "parties",
]

WORKFLOW_OPTIONAL_SIGNAL_SLOTS: List[str] = [
    "timeline",
    "harm_or_amount",
    "desired_outcome",
    "evidence",
    "witnesses",
]

LAW_KEYWORDS = {
    "law", "legal", "case", "ipc", "crpc", "evidence", "fir", "complaint", "petition",
    "bail", "section", "court", "judge", "advocate", "lawyer", "police", "crime", "fraud",
    "forgery", "theft", "defamation", "conspiracy", "intimidation", "contract", "notice",
    "litigation", "arrest", "rights", "remedy", "punishment", "offence", "offense",
    "charge", "cheating", "breach", "trust", "property", "civil", "criminal", "procedure",
    "appeal", "warrant", "summons", "injunction", "damages", "liability", "mediation",
    "divorce", "custody", "maintenance", "anticipatory", "quash", "affidavit", "agreement",
    "draft", "strategy", "accused", "respondent", "complainant", "chargesheet", "charge-sheet",
    "citizen", "civilian", "tenant", "landlord", "property", "consumer", "employment",
    "employer", "employee", "salary", "wage", "wages", "workplace",
    "service", "notice", "agreement", "family", "inheritance", "succession",
    "rent", "rental", "lease", "owner", "ownership", "land", "title", "deed",
    "mutation", "registry", "registration", "encumbrance", "partition", "probate", "will",
    "eviction", "tenancy", "stamp", "conveyance", "sublease", "allotment",
    "refund", "defect", "defective", "product", "seller", "buyer", "warranty",
    "upi", "otp", "phishing", "hacked", "scam", "cyber", "online", "digital",
    "harassment", "maintenance", "custody", "inheritance", "salary", "dues",
}

LEGAL_CATEGORIES: Dict[str, set[str]] = {
    "Property or tenancy issue": {
        "property", "tenant", "tenancy", "landlord", "rent", "lease", "owner",
        "ownership", "eviction", "land", "deed", "mutation", "registry", "title",
        "sublease", "allotment", "flat", "house", "apartment", "plot", "real estate",
        "rera", "builder", "possession", "encumbrance", "stamp duty", "registration",
        "conveyance", "partition", "probate", "will", "inheritance", "succession",
        "trespass", "dispossession", "mortgage",
    },
    "Contract or payment dispute": {
        "contract", "agreement", "payment", "deposit", "refund", "invoice",
        "breach", "dues", "advance", "service",
        "cheque", "bounce", "dishonour", "negotiable", "promissory",
        "arbitration", "settlement", "vendor", "supplier", "buyer",
        "seller", "transaction", "deal",
    },
    "Fraud or cheating concern": {
        "fraud", "cheat", "cheating", "forgery", "scam", "deception", "fake",
        "misrepresentation", "impersonation", "ponzi", "embezzlement",
        "420", "criminal breach", "trust", "dishonest", "upi", "otp",
    },
    "Consumer issue": {
        "consumer", "defect", "deficiency", "warranty", "product", "seller",
        "ecommerce", "replacement", "return", "refund", "amazon", "flipkart",
        "service provider", "complaint forum", "district commission",
    },
    "Employment dispute": {
        "employment", "salary", "termination", "dismissal", "workplace", "hr",
        "harassment", "wages", "bonus", "notice period", "pf", "gratuity",
        "labour", "industrial", "retrenchment", "unfair dismissal", "employer",
        "employee", "unpaid", "pending pay",
    },
    "Family or relationship dispute": {
        "divorce", "custody", "maintenance", "marriage", "domestic", "inheritance",
        "succession", "family", "will", "alimony", "dowry", "matrimonial",
        "guardian", "adoption", "child", "spouse", "husband", "wife",
    },
    "Cyber or online harm": {
        "cyber", "online", "phishing", "hacked", "social media", "digital",
        "data", "account", "otp", "upi", "payment fraud", "bank fraud",
        "email", "whatsapp", "instagram", "deepfake", "blackmail", "extortion online",
        "it act", "cybercrime", "identity theft",
        "scam", "impersonation",
    },
    "Defamation or reputation issue": {
        "defamation", "reputation", "false statement", "public post", "libel", "slander",
        "character", "honour", "dignity",
    },
    "Traffic or road incident": {
        "traffic", "road", "vehicle", "accident", "driving", "license", "challan",
        "motor", "car", "truck", "mva", "insurance claim", "hit and run",
    },
    "FIR and police complaint": {
        "fir", "first information report", "police complaint", "police station",
        "register complaint", "lodge complaint", "section 154", "cognizable",
        "non-cognizable", "chargesheet", "charge sheet", "challan",
        "investigation", "arrest", "zero fir", "complaint to police",
        "magistrate", "crpc", "bnss",
    },
    "Bail and custody": {
        "bail", "anticipatory bail", "regular bail", "remand", "custody",
        "detention", "arrested", "locked up", "jail", "prison",
        "section 436", "section 437", "section 438", "section 439",
        "bailable", "non-bailable", "surety",
    },
    "General criminal concern": {
        "crime", "offence", "offense", "threat", "violence", "assault", "police",
        "arrest", "intimidation", "murder", "theft", "robbery", "kidnapping",
        "rape", "harassment", "criminal", "ipc", "bns", "warrant", "quash",
        "acquittal", "conviction", "sentence", "punishment", "accused",
    },
}


class CaseRequest(BaseModel):
    case_type: str = Field(min_length=3, max_length=120)
    incident: str = Field(min_length=10, max_length=3000)
    amount: Optional[str] = Field(default="", max_length=80)

    @field_validator("case_type", "incident", "amount", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> str:
        return _normalize_text(value)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=3000)

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, value: Any) -> str:
        return _normalize_text(value)


class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=3000)
    history: List[ChatTurn] = Field(default_factory=list, max_length=20)

    @field_validator("message", mode="before")
    @classmethod
    def normalize_message(cls, value: Any) -> str:
        return _normalize_text(value)


class Citation(BaseModel):
    kind: str
    id: str
    section: str
    text: str
    source: str
    effective_date: str
    score: float


class GenerateResponse(BaseModel):
    draft: str
    citations: List[Citation]
    disclaimer: str


class CaseWorkflow(BaseModel):
    enabled: bool = False
    stage: Literal[1, 2, 3, 4] = 1
    stage_label: str = WORKFLOW_STAGE_LABELS[1]
    legal_category: str = "General legal issue"
    ready_for_report: bool = False
    report_generated: bool = False
    missing_fields: List[str] = Field(default_factory=list)
    collected_facts: Dict[str, str] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    reply: str
    citations: List[Citation]
    redirected_to_law: bool
    disclaimer: str
    case_workflow: Optional[CaseWorkflow] = None


class IngestFileResult(BaseModel):
    filename: str
    document_type: Optional[str] = None
    status: Literal["ingested", "failed"]
    chunks: int = 0
    detail: str


class IngestResponse(BaseModel):
    total_files: int
    ingested_files: int
    failed_files: int
    total_chunks: int
    results: List[IngestFileResult]


class ReportExportRequest(BaseModel):
    title: str = "SaulGPT Legal Report"
    content: str = Field(min_length=1, max_length=20000)
    format: Literal["pdf", "docx", "txt"] = "pdf"
    filename: Optional[str] = None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("must be a string")
    return " ".join(value.split()).strip()


def _check_api_key(x_api_key: Optional[str]) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


def _check_rate_limit(client_id: str) -> None:
    now = time.time()
    window_start = now - 60
    bucket = _rate_limit_store[client_id]

    while bucket and bucket[0] < window_start:
        bucket.popleft()

    if len(bucket) >= REQUESTS_PER_MINUTE:
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    bucket.append(now)


def _is_law_topic(text: str) -> bool:
    lower = text.lower()
    words = set(re.findall(r"[a-zA-Z]+", lower))
    if words.intersection(LAW_KEYWORDS):
        return True

    # Handle real-life citizen case narratives without explicit legal keywords.
    if re.search(r"\b(i|my|me|we|our)\b", lower) and re.search(
        r"\b(partner|landlord|tenant|deposit|money|payment|loan|took|stole|fraud|cheat|threat|forgery|notice|police|case)\b",
        lower,
    ):
        return True

    legal_patterns = [
        r"section\s+\d+",
        r"ipc\s*\d*",
        r"fir",
        r"criminal",
        r"civil",
        r"rental?\s+law",
        r"rent\s+act",
        r"owner(ship)?",
        r"land",
        r"property",
        r"title\s+deed",
        r"mutation",
        r"tenant|tenancy|landlord|lease|eviction",
        r"legal\s+notice",
        r"what\s+is\s+the\s+law",
        r"can\s+i\s+file",
        r"how\s+to\s+file",
        r"draft\s+(complaint|petition|notice|report)",
        r"accused",
    ]
    if any(re.search(pattern, lower) for pattern in legal_patterns):
        return True

    # Bias toward legal for citizen-law style questions.
    if re.search(r"\b(act|law|rule|code|regulation|right|remedy|owner|tenant|rent|land)\b", lower):
        return True
    return False


def _sanitize_reference_text(text: str, collapse_whitespace: bool = True) -> str:
    cleaned = re.sub(r"\b(?:section|sec\.?|article)\s*\d+[a-zA-Z]?\b", "a relevant provision", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:ipc|crpc|cpc|bns|bnss|bsa)\s*\d*[a-zA-Z]?\b", "the legal framework", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bIndian Penal Code\b", "criminal law framework", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bCode of Criminal Procedure\b", "criminal procedure framework", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b[A-Z][A-Za-z&' .-]{1,80}\s+(?:Act|Code|Rules?|Regulations?)\b",
        "the relevant legal framework",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b\d{1,4}[A-Za-z]?\s+of\s+the\s+[A-Za-z&' .-]{1,80}(?:Act|Code|Rules?|Regulations?)\b",
        "a relevant legal provision",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bthe relevant legal framework\s+\d{4}\b",
        "the relevant legal framework",
        cleaned,
        flags=re.IGNORECASE,
    )
    if collapse_whitespace:
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
    else:
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _context_dependent_query(message: str) -> bool:
    lower = message.lower().strip()
    if len(lower) <= 55 and _CONTEXT_DEPENDENT_QUERY_RE.search(lower):
        return True
    if len(lower.split()) <= 14 and lower.endswith("?"):
        return True
    if len(lower.split()) <= 5 and any(
        token in lower for token in ("next", "proceed", "what now", "then", "same")
    ):
        return True
    return False


def _recent_user_messages(history: List[ChatTurn], limit: int = 5) -> List[str]:
    return [turn.content for turn in history if turn.role == "user"][-limit:]


def _history_has_legal_context(history: List[ChatTurn]) -> bool:
    return any(_is_law_topic(turn.content) for turn in history if turn.role == "user")


def _compose_retrieval_query(message: str, history: List[ChatTurn]) -> str:
    recent = _recent_user_messages(history, limit=4)
    if not recent:
        return message
    recent_text = " ".join(recent[-2:]).lower()
    message_category, message_score = _category_from_text(message.lower())
    recent_category, recent_score = _category_from_text(recent_text)
    if message_score >= 2 and recent_score >= 2 and message_category != recent_category:
        return message
    if _context_dependent_query(message):
        return f"{' '.join(recent)} {message}".strip()
    return f"{message} {' '.join(recent[-2:])}".strip()


def _style_variant(message: str, history: List[ChatTurn]) -> int:
    turn_index = len(_recent_user_messages(history, limit=12))
    message_weight = sum(ord(char) for char in message[:80])
    return (turn_index + message_weight) % 3


def _with_article(phrase: str) -> str:
    text = phrase.strip()
    if not text:
        return text
    article = "an" if text[0].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {text}"


def _missing_detail_questions(
    message: str, history: List[ChatTurn], category: str, has_uploaded: bool
) -> List[str]:
    context_blob = " ".join(_recent_user_messages(history, limit=5) + [message]).strip()
    lower = context_blob.lower()
    questions: List[str] = []

    if not _DATE_MARKER_RE.search(lower):
        questions.append("When did this start, and what are the key dates?")
    if not _PARTY_MARKER_RE.search(lower):
        questions.append("Who are the main parties involved, and what is each role?")
    if not _DOCUMENT_MARKER_RE.search(lower):
        questions.append("What documents or message records do you already have?")
    if not _GOAL_MARKER_RE.search(lower):
        questions.append("What outcome are you trying to achieve in this matter?")

    if category == "Property or tenancy issue" and not has_uploaded:
        questions.append("Do you have the rent/lease terms and payment proof available?")
    if category == "Employment dispute" and not has_uploaded:
        questions.append("Do you have offer letter, salary records, or HR communication?")
    if category == "Fraud or cheating concern" and not has_uploaded:
        questions.append("Do you have transaction IDs, account details, or communication trail?")

    if len(context_blob.split()) > 45:
        return questions[: min(2, MAX_FOLLOWUP_QUESTIONS)]
    return questions[:MAX_FOLLOWUP_QUESTIONS]


def _report_intake_in_progress(history: List[ChatTurn]) -> bool:
    return any(
        turn.role == "assistant"
        and (
            _REPORT_INTAKE_MARKER.lower() in turn.content.lower()
            or WORKFLOW_STAGE_LABELS[2].lower() in turn.content.lower()
        )
        for turn in history[-8:]
    )


def _report_mode_requested(message: str, history: List[ChatTurn]) -> bool:
    lower = message.lower()
    if _REPORT_EXIT_RE.search(lower):
        return False
    if _REPORT_INTENT_RE.search(lower):
        return True
    if _report_intake_in_progress(history):
        return True
    return False


def _workflow_mode_requested(message: str, history: List[ChatTurn]) -> bool:
    lower = message.lower()
    if _REPORT_EXIT_RE.search(lower):
        return False
    if _report_mode_requested(message, history):
        return True
    if any(
        turn.role == "assistant"
        and (
            _REPORT_INTAKE_MARKER.lower() in turn.content.lower()
            or WORKFLOW_STAGE_LABELS[2].lower() in turn.content.lower()
            or "structured legal intake report" in turn.content.lower()
            or WORKFLOW_STAGE_LABELS[3].lower() in turn.content.lower()
        )
        for turn in history[-10:]
    ):
        return True
    return False


def _all_slot_questions_for_category(category: str) -> Dict[str, str]:
    questions = dict(_REPORT_COMMON_SLOT_QUESTIONS)
    questions.update(_REPORT_CATEGORY_SLOT_QUESTIONS.get(category, {}))
    return questions


def _report_intake_rounds(history: List[ChatTurn]) -> int:
    return sum(
        1
        for turn in history[-12:]
        if turn.role == "assistant"
        and (
            _REPORT_INTAKE_MARKER.lower() in turn.content.lower()
            or WORKFLOW_STAGE_LABELS[2].lower() in turn.content.lower()
        )
    )


def _asked_slot_counts(history: List[ChatTurn], category: str) -> Dict[str, int]:
    questions = _all_slot_questions_for_category(category)
    counts: Dict[str, int] = {slot: 0 for slot in questions}
    for turn in history:
        if turn.role != "assistant":
            continue
        lower = turn.content.lower()
        for slot, question in questions.items():
            if question.lower() in lower:
                counts[slot] = counts.get(slot, 0) + 1
    return counts


def _explicitly_unknown(text: str) -> bool:
    lower = text.lower()
    return bool(
        re.search(
            r"\b(?:unknown|not sure|not available|dont know|don't know|na|n/a|none|no idea)\b",
            lower,
        )
    )


def _last_requested_slots(history: List[ChatTurn], category: str) -> List[str]:
    questions = _all_slot_questions_for_category(category)
    for turn in reversed(history):
        if turn.role != "assistant":
            continue
        lower = turn.content.lower()
        slots = [slot for slot, question in questions.items() if question.lower() in lower]
        if slots:
            return slots
    return []


def _apply_direct_reply_slots(
    message: str,
    history: List[ChatTurn],
    category: str,
    slots: Dict[str, str],
) -> None:
    requested_slots = _last_requested_slots(history, category)
    if not requested_slots:
        return

    raw = message.strip()
    lower = raw.lower()
    if not raw or raw.endswith("?"):
        return

    unknown = _explicitly_unknown(raw)
    short_answer = len(raw.split()) <= 8
    date_match = _DATE_CAPTURE_RE.search(raw)
    amount_match = _AMOUNT_CAPTURE_RE.search(raw)

    for slot in requested_slots:
        if _slot_filled(slots.get(slot)):
            continue

        if unknown:
            if slot == "witnesses":
                slots[slot] = "No witnesses identified"
            elif slot == "evidence":
                slots[slot] = "No documentary evidence available yet"
            else:
                slots[slot] = "Not provided by user"
            continue

        if slot == "incident_date":
            if date_match:
                slots[slot] = date_match.group(0).strip()
            elif short_answer:
                slots[slot] = raw[:80]
        elif slot == "incident_location":
            candidate = re.sub(r"^(?:at|in|near)\s+", "", raw, flags=re.IGNORECASE).strip(" ,.-")
            if candidate:
                slots[slot] = candidate[:120]
        elif slot == "witnesses":
            if re.search(r"^(?:no|none)\b", lower):
                slots[slot] = "No witnesses identified"
            else:
                slots[slot] = raw[:220]
        elif slot == "evidence":
            if re.search(r"^(?:no|none)\b", lower):
                slots[slot] = "No documentary evidence available yet"
            elif re.search(r"^(?:yes|available|have it|i have it)\b", lower):
                slots[slot] = "User confirms documents or records are available"
            else:
                slots[slot] = raw[:220]
        elif slot == "harm_or_amount":
            if amount_match:
                slots[slot] = amount_match.group(0).upper()
            else:
                slots[slot] = raw[:160]
        elif slot in {"incident_summary", "timeline", "parties"} and short_answer:
            continue
        else:
            slots[slot] = raw[:240]


def _slot_label(slot: str) -> str:
    return SLOT_LABELS.get(slot, slot.replace("_", " ").title())


def _missing_field_labels(slots: List[str]) -> List[str]:
    return [_slot_label(slot) for slot in slots]


def _last_group_match(pattern: str, text: str) -> Optional[str]:
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        for part in value:
            if part:
                return str(part).strip()
        return None
    return str(value).strip()


def _extract_report_slots(
    message: str, history: List[ChatTurn], category: str, retrieval: List[Dict[str, Any]]
) -> Dict[str, str]:
    user_messages = _recent_user_messages(history, limit=14) + [message]
    joined = "\n".join(user_messages).strip()
    lower = joined.lower()
    slots: Dict[str, str] = {}

    date_match = _DATE_CAPTURE_RE.search(joined)
    if date_match:
        slots["incident_date"] = date_match.group(0).strip()

    location_match = _LOCATION_CAPTURE_RE.search(joined)
    if location_match:
        slots["incident_location"] = location_match.group(1).strip(" ,.-")

    parties: List[str] = []
    if re.search(r"\b(i|my|me|we|our)\b", lower):
        parties.append("User/complainant")
    role_hits = [hit.group(0).lower() for hit in _PARTY_HINT_RE.finditer(joined)]
    for role in role_hits:
        if role not in parties:
            parties.append(role)
    if not parties and category == "Property or tenancy issue":
        parties.extend(["tenant", "landlord"])
    if not parties and category == "Employment dispute":
        parties.extend(["employee", "employer"])
    if parties:
        slots["parties"] = ", ".join(parties[:5])

    summary_candidate = ""
    for msg in reversed(user_messages):
        if len(msg.split()) < 12 or _REPORT_INTENT_RE.search(msg.lower()):
            continue
        if _DATE_CAPTURE_RE.search(msg) or re.search(
            r"\b(happened|occurred|started|withheld|refused|terminated|received|paid|took|threat|vacated|delivered|failed)\b",
            msg,
            re.IGNORECASE,
        ):
            summary_candidate = msg
            break
    if not summary_candidate:
        long_candidates = [
            msg for msg in user_messages if len(msg.split()) >= 12 and not _REPORT_INTENT_RE.search(msg.lower())
        ]
        if long_candidates:
            summary_candidate = max(long_candidates, key=len)
    if not summary_candidate and len(joined.split()) >= 18:
        summary_candidate = " ".join(" ".join(user_messages[-3:]).split()[:90])
    if summary_candidate:
        slots["incident_summary"] = summary_candidate[:500]

    timeline_fragments: List[str] = []
    for msg in user_messages[-6:]:
        if _DATE_CAPTURE_RE.search(msg) or re.search(
            r"\b(then|after|before|later|next|on|during|when)\b", msg, re.IGNORECASE
        ):
            timeline_fragments.append(msg.strip())
    if timeline_fragments:
        slots["timeline"] = " | ".join(timeline_fragments[:3])[:500]

    amount_match = _AMOUNT_CAPTURE_RE.search(joined)
    if amount_match:
        slots["harm_or_amount"] = amount_match.group(0).upper()
    else:
        harm_match = re.search(
            r"([^.!\n]*\b(loss|damage|harm|injury|pending|unpaid|deposit|dues)\b[^.!\n]*)",
            joined,
            re.IGNORECASE,
        )
        if harm_match:
            slots["harm_or_amount"] = harm_match.group(1).strip()

    evidence_markers: List[str] = []
    if re.search(
        r"\b(agreement|contract|receipt|invoice|email|message|screenshot|recording|payslip|statement|offer letter|attendance|hr|bank transfer|bank statement|whatsapp|chat|payment receipt)\b",
        joined,
        re.IGNORECASE,
    ):
        evidence_markers.append("Documents mentioned in chat")
    if re.search(r"\b(i have|available|attached|uploaded)\b", joined, re.IGNORECASE):
        evidence_markers.append("User indicates supporting records are available")
    uploaded_sources = [
        str(doc.get("source", "")).strip()
        for doc in retrieval
        if doc.get("kind") == "uploaded" and str(doc.get("source", "")).strip()
    ]
    if uploaded_sources:
        seen: List[str] = []
        for source in uploaded_sources:
            if source not in seen:
                seen.append(source)
        evidence_markers.append(f"Uploaded files: {', '.join(seen[:3])}")
    if re.search(r"\b(no evidence|no document|no proof|not available)\b", lower):
        evidence_markers.append("No documentary evidence available yet")
    if evidence_markers:
        slots["evidence"] = ", ".join(evidence_markers)
    if "evidence" in slots:
        slots["document_evidence"] = slots["evidence"]

    witness_match = re.search(
        r"([^.!\n]*\b(witness|witnesses|saw|seen by|eyewitness|present at scene)\b[^.!\n]*)",
        joined,
        re.IGNORECASE,
    )
    if witness_match:
        slots["witnesses"] = witness_match.group(1).strip()
    elif re.search(r"\b(no witness|no witnesses|none witnessed)\b", lower):
        slots["witnesses"] = "No witnesses identified"

    outcome_match = re.search(
        r"\b(refund|compensation|salary release|dues|return of deposit|eviction|compliance|recovery|resolution|settlement|clarification)\b",
        joined,
        re.IGNORECASE,
    )
    if outcome_match:
        slots["desired_outcome"] = outcome_match.group(1)
    else:
        user_goal = re.search(
            r"\b(?:want|need|seeking|looking for)\s+([^.!\n]{4,120})",
            joined,
            re.IGNORECASE,
        )
        if user_goal:
            slots["desired_outcome"] = user_goal.group(1).strip()

    if category == "Property or tenancy issue":
        property_value = _last_group_match(
            r"([^.!\n]*\b(property|flat|house|apartment|shop|unit|plot|land)\b[^.!\n]*)",
            joined,
        )
        if property_value:
            slots["property_details"] = property_value
        agreement_value = _last_group_match(
            r"([^.!\n]*\b(agreement|lease|rent term|vacated|vacate|tenancy)\b[^.!\n]*)",
            joined,
        )
        if agreement_value:
            slots["agreement_status"] = agreement_value
    elif category == "Employment dispute":
        role_value = _last_group_match(
            r"([^.!\n]*\b(employer|employee|company|role|designation|joining|work as|working as)\b[^.!\n]*)",
            joined,
        )
        if role_value:
            slots["employment_details"] = role_value
        dues_value = _last_group_match(
            r"([^.!\n]*\b(salary|dues|unpaid|pending|month)\b[^.!\n]*)",
            joined,
        )
        if dues_value:
            slots["dues_details"] = dues_value
    elif category == "Fraud or cheating concern":
        counterparty_value = _last_group_match(
            r"([^.!\n]*\b(account|number|id|person|agent|broker|platform)\b[^.!\n]*)",
            joined,
        )
        if counterparty_value:
            slots["counterparty_details"] = counterparty_value
        txn_value = _last_group_match(
            r"([^.!\n]*\b(transaction|transfer|upi|bank|payment|wallet)\b[^.!\n]*)",
            joined,
        )
        if txn_value:
            slots["transaction_details"] = txn_value
    elif category == "Consumer issue":
        seller_value = _last_group_match(
            r"([^.!\n]*\b(seller|provider|company|store|service)\b[^.!\n]*)",
            joined,
        )
        if seller_value:
            slots["seller_service_details"] = seller_value
        timeline_value = _last_group_match(
            r"([^.!\n]*\b(order|purchase|delivery|defect|warranty|refund)\b[^.!\n]*)",
            joined,
        )
        if timeline_value:
            slots["purchase_timeline"] = timeline_value
    elif category == "Cyber or online harm":
        platform_value = _last_group_match(
            r"([^.!\n]*\b(app|platform|account|website|social media|email)\b[^.!\n]*)",
            joined,
        )
        if platform_value:
            slots["platform_details"] = platform_value
        trail_value = _last_group_match(
            r"([^.!\n]*\b(screenshot|otp|transaction id|alert|log)\b[^.!\n]*)",
            joined,
        )
        if trail_value:
            slots["digital_trail"] = trail_value

    _apply_direct_reply_slots(message=message, history=history, category=category, slots=slots)

    if not slots.get("incident_location"):
        stripped_message = re.sub(r"^(?:at|in|near)\s+", "", message.strip(), flags=re.IGNORECASE).strip(" ,.-")
        if stripped_message and len(stripped_message.split()) <= 5 and not message.strip().endswith("?"):
            if any(slot == "incident_location" for slot in _last_requested_slots(history, category)):
                slots["incident_location"] = stripped_message[:120]
        if not slots.get("incident_location"):
            first_clause = re.split(r"[.!?\n]", message.strip(), maxsplit=1)[0].strip(" ,.-")
            if (
                first_clause
                and len(first_clause.split()) <= 4
                and re.fullmatch(r"[A-Za-z][A-Za-z .,-]{1,80}", first_clause)
            ):
                slots["incident_location"] = first_clause[:120]

    if re.search(r"\b(not sure|unknown|not available|dont know|don't know|na|n/a)\b", lower):
        slots.setdefault("incident_date", "Not specified by user")
        slots.setdefault("incident_location", "Not specified by user")

    return slots


def _slot_filled(value: Optional[str], min_chars: int = 3) -> bool:
    if not value:
        return False
    return len(value.strip()) >= min_chars


def _missing_report_slots(category: str, slots: Dict[str, str]) -> List[str]:
    missing: List[str] = []
    for slot in WORKFLOW_BASE_REQUIRED_SLOTS:
        if not _slot_filled(slots.get(slot)):
            missing.append(slot)

    if not (_slot_filled(slots.get("timeline")) or _slot_filled(slots.get("incident_date"))):
        missing.append("incident_date")

    category_slots = _REPORT_REQUIRED_BY_CATEGORY.get(category, [])
    if category_slots and not any(_slot_filled(slots.get(slot)) for slot in category_slots):
        missing.append(category_slots[0])

    if not (
        _slot_filled(slots.get("harm_or_amount"))
        or _slot_filled(slots.get("desired_outcome"))
        or _slot_filled(slots.get("evidence"))
    ):
        missing.append("desired_outcome")

    if not _slot_filled(slots.get("incident_location")) and not any(
        _slot_filled(slots.get(slot)) for slot in category_slots
    ):
        missing.append("incident_location")

    seen: set[str] = set()
    ordered: List[str] = []
    for slot in missing:
        if slot in seen:
            continue
        seen.add(slot)
        ordered.append(slot)
    return ordered


def _resolve_missing_report_slots(
    category: str,
    slots: Dict[str, str],
    history: List[ChatTurn],
) -> List[str]:
    missing = _missing_report_slots(category, slots)
    asked_counts = _asked_slot_counts(history, category)

    resolved_missing: List[str] = []
    for slot in missing:
        if _slot_filled(slots.get(slot)):
            continue
        if asked_counts.get(slot, 0) >= 2:
            slots[slot] = "Not provided by user"
            continue
        resolved_missing.append(slot)
    return resolved_missing


def _has_minimum_facts_for_report(
    category: str,
    slots: Dict[str, str],
    missing_slots: List[str],
    message: str,
) -> bool:
    score_slots = [
        "incident_summary",
        "parties",
        "incident_date",
        "incident_location",
        "timeline",
        "harm_or_amount",
        "desired_outcome",
        "evidence",
        "witnesses",
    ]
    score = sum(1 for slot in score_slots if _slot_filled(slots.get(slot)))
    has_core = _slot_filled(slots.get("incident_summary")) and _slot_filled(slots.get("parties"))
    has_case_detail = any(
        _slot_filled(slots.get(slot))
        for slot in _REPORT_REQUIRED_BY_CATEGORY.get(category, [])
    ) or any(
        _slot_filled(slots.get(slot))
        for slot in ("harm_or_amount", "desired_outcome", "evidence", "incident_location")
    )
    has_context = any(
        _slot_filled(slots.get(slot))
        for slot in ("incident_date", "timeline", "incident_location")
    )
    wants_report_now = bool(_REPORT_GENERATE_RE.search(message.lower()))

    if has_core and has_context and has_case_detail and score >= 5:
        return True
    if wants_report_now and has_core and (has_case_detail or has_context) and score >= 4:
        return True
    if has_core and has_case_detail and len(missing_slots) <= 2 and score >= 4:
        return True
    return False


def _should_force_report_generation(
    category: str,
    slots: Dict[str, str],
    message: str,
    history: List[ChatTurn],
) -> bool:
    has_core = _slot_filled(slots.get("incident_summary")) and _slot_filled(slots.get("parties"))
    if not has_core:
        return False

    intake_rounds = _report_intake_rounds(history)
    score = sum(
        1
        for slot in (
            "incident_summary",
            "parties",
            "incident_date",
            "incident_location",
            "timeline",
            "harm_or_amount",
            "desired_outcome",
            "evidence",
            *_REPORT_REQUIRED_BY_CATEGORY.get(category, []),
        )
        if _slot_filled(slots.get(slot))
    )
    requested_now = bool(_REPORT_GENERATE_RE.search(message.lower()))

    if requested_now and intake_rounds >= 1 and score >= 4:
        return True
    if intake_rounds >= 2 and score >= 4:
        return True
    return False


def _finalize_report_slots(category: str, slots: Dict[str, str]) -> Dict[str, str]:
    finalized = dict(slots)
    defaults = {
        "incident_date": "Not specified by user",
        "incident_location": "Not specified by user",
        "timeline": "Timeline not fully specified by user",
        "harm_or_amount": "Loss or amount not clearly specified",
        "evidence": "No document evidence specified by user",
        "witnesses": "No witness information provided",
        "desired_outcome": "Desired outcome not clearly specified",
    }
    for slot, value in defaults.items():
        if not _slot_filled(finalized.get(slot)):
            finalized[slot] = value

    for slot in _REPORT_REQUIRED_BY_CATEGORY.get(category, []):
        if not _slot_filled(finalized.get(slot)):
            finalized[slot] = "Not provided by user"

    return finalized


def _next_report_questions(
    category: str, missing_slots: List[str], history: List[ChatTurn]
) -> List[str]:
    question_bank = dict(_REPORT_COMMON_SLOT_QUESTIONS)
    question_bank.update(_REPORT_CATEGORY_SLOT_QUESTIONS.get(category, {}))
    asked_counts = _asked_slot_counts(history, category)

    ordered_missing: List[str] = []
    for slot in _REPORT_SLOT_PRIORITY:
        if slot in missing_slots and slot not in ordered_missing:
            ordered_missing.append(slot)
    for slot in missing_slots:
        if slot not in ordered_missing:
            ordered_missing.append(slot)

    questions: List[str] = []
    for slot in ordered_missing:
        if asked_counts.get(slot, 0) >= 2:
            continue
        question = question_bank.get(slot)
        if question:
            questions.append(question)
        if len(questions) >= 2:
            break
    if not questions and missing_slots:
        questions.append("Please share the pending factual details so I can complete the report draft.")
    return questions


def _report_slots_summary(slots: Dict[str, str]) -> str:
    labels = [
        ("incident_date", "Incident date"),
        ("incident_location", "Location"),
        ("parties", "Parties"),
        ("incident_summary", "Incident summary"),
        ("timeline", "Timeline"),
        ("harm_or_amount", "Loss/amount"),
        ("evidence", "Evidence available"),
        ("witnesses", "Witness information"),
        ("desired_outcome", "Desired outcome"),
        ("property_details", "Property details"),
        ("agreement_status", "Agreement/rent status"),
        ("employment_details", "Employment details"),
        ("dues_details", "Pending dues"),
        ("counterparty_details", "Counterparty details"),
        ("transaction_details", "Transaction details"),
        ("seller_service_details", "Seller/service details"),
        ("purchase_timeline", "Purchase timeline"),
        ("platform_details", "Platform/account details"),
        ("digital_trail", "Digital trail"),
    ]
    lines = []
    for key, label in labels:
        value = slots.get(key)
        if value:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines) if lines else "- No essential details captured yet."


def _build_report_intake_reply(
    category: str,
    slots: Dict[str, str],
    questions: List[str],
    missing_slots: List[str],
) -> str:
    compact_keys = [
        "incident_summary",
        "parties",
        "incident_date",
        "incident_location",
        "evidence",
        "witnesses",
        "harm_or_amount",
        "desired_outcome",
    ]
    compact_lines: List[str] = []
    for key in compact_keys:
        value = slots.get(key)
        if value:
            compact_lines.append(f"- {_slot_label(key)}: {str(value)[:120]}")
        if len(compact_lines) >= 5:
            break
    compact_summary = "\n".join(compact_lines) if compact_lines else "- No core facts captured yet."

    if not questions:
        return (
            f"{WORKFLOW_STAGE_LABELS[3]}\n"
            f"Category: {category}\n"
            "Sufficient facts captured. Generating report now."
        )

    numbered_questions = "\n".join(
        f"{index}. {question}" for index, question in enumerate(questions, start=1)
    )
    missing_compact = ", ".join(_slot_label(slot) for slot in missing_slots[:4]) or "None"
    return (
        f"{WORKFLOW_STAGE_LABELS[2]}\n"
        f"Category: {category}\n"
        f"I have already captured these facts:\n{compact_summary}\n\n"
        f"To finish the report, I still need: {missing_compact}\n"
        "Reply with short bullets or one compact paragraph:\n"
        f"{numbered_questions}\n\n"
        "Once I have that, I will move straight to the report."
    )


def _is_placeholder_value(value: Optional[str]) -> bool:
    if not value:
        return True
    normalized = value.strip().lower()
    placeholders = (
        "not provided by user",
        "not specified by user",
        "timeline not fully specified by user",
        "loss or amount not clearly specified",
        "no document evidence specified by user",
        "no witness information provided",
        "desired outcome not clearly specified",
        "no documentary evidence available yet",
        "no witnesses identified",
    )
    return any(phrase in normalized for phrase in placeholders)


def _report_matter_focus(category: str, slots: Dict[str, str]) -> str:
    text_blob = " ".join(str(slots.get(key, "")) for key in slots).lower()
    if category == "Property or tenancy issue":
        if "deposit" in text_blob:
            return "Rental deposit dispute"
        if any(token in text_blob for token in ("ownership", "title", "mutation", "deed", "land")):
            return "Ownership / title matter"
        if any(token in text_blob for token in ("eviction", "vacate", "tenant", "landlord")):
            return "Tenancy / possession dispute"
        return "Property / tenancy matter"
    if category == "Employment dispute":
        if any(token in text_blob for token in ("salary", "unpaid", "dues", "payslip")):
            return "Unpaid salary / dues matter"
        if any(token in text_blob for token in ("termination", "dismissal", "relieving")):
            return "Employment separation dispute"
        return "Employment matter"
    if category == "Consumer issue":
        if any(token in text_blob for token in ("defect", "defective", "warranty")):
            return "Defective product or service dispute"
        if "refund" in text_blob:
            return "Refund dispute"
        return "Consumer matter"
    if category == "Fraud or cheating concern":
        return "Fraud / cheating matter"
    if category == "Cyber or online harm":
        if any(token in text_blob for token in ("upi", "otp", "bank", "account")):
            return "Online payment / account harm matter"
        return "Cyber / online harm matter"
    if category == "Family or relationship dispute":
        return "Family / relationship matter"
    if category == "Contract or payment dispute":
        return "Contract / payment matter"
    if category == "FIR and police complaint":
        return "Police complaint intake matter"
    if category == "Bail and custody":
        return "Custody / bail support matter"
    return category


def _report_title(category: str, slots: Dict[str, str]) -> str:
    focus = _report_matter_focus(category, slots)
    return f"SaulGPT Structured Legal Intake Report - {focus}"


def _report_case_snapshot(category: str, slots: Dict[str, str]) -> List[str]:
    snapshot = [
        f"- Matter focus: {_report_matter_focus(category, slots)}",
        f"- Parties: {slots.get('parties', 'Not fully specified')}",
        f"- Location: {slots.get('incident_location', 'Not specified by user')}",
        f"- Date / period: {slots.get('incident_date', 'Not specified by user')}",
    ]
    if not _is_placeholder_value(slots.get("harm_or_amount")):
        snapshot.append(f"- Reported loss / amount: {slots.get('harm_or_amount')}")
    if not _is_placeholder_value(slots.get("desired_outcome")):
        snapshot.append(f"- User objective: {slots.get('desired_outcome')}")
    return snapshot


def _category_specific_heading(category: str) -> str:
    headings = {
        "Property or tenancy issue": "Property / Tenancy Details",
        "Employment dispute": "Employment Details",
        "Fraud or cheating concern": "Counterparty and Transaction Details",
        "Consumer issue": "Seller / Service Details",
        "Cyber or online harm": "Digital Account and Trail Details",
        "Family or relationship dispute": "Relationship Context",
        "Contract or payment dispute": "Contract / Payment Context",
    }
    return headings.get(category, "Category-Specific Details")


def _category_specific_lines(category: str, slots: Dict[str, str]) -> List[str]:
    slot_map = {
        "Property or tenancy issue": [
            ("property_details", "Property details"),
            ("agreement_status", "Agreement / tenancy status"),
        ],
        "Employment dispute": [
            ("employment_details", "Employment details"),
            ("dues_details", "Pending dues details"),
        ],
        "Fraud or cheating concern": [
            ("counterparty_details", "Counterparty details"),
            ("transaction_details", "Transaction details"),
        ],
        "Consumer issue": [
            ("seller_service_details", "Seller / service details"),
            ("purchase_timeline", "Purchase timeline"),
        ],
        "Cyber or online harm": [
            ("platform_details", "Platform / account details"),
            ("digital_trail", "Digital trail"),
        ],
    }
    lines: List[str] = []
    for slot, label in slot_map.get(category, []):
        value = slots.get(slot)
        if value and not _is_placeholder_value(value):
            lines.append(f"- {label}: {value}")
    if not lines:
        lines.append("- No additional category-specific detail was captured.")
    return lines


def _report_issue_observations(category: str, slots: Dict[str, str]) -> List[str]:
    observations: List[str] = []
    if category == "Property or tenancy issue":
        observations.append("- The matter appears to turn on possession or tenancy record, money trail, and move-out / notice chronology.")
        if "deposit" in " ".join(str(slots.get(key, "")) for key in slots).lower():
            observations.append("- Security deposit handling appears to be a central factual issue.")
    elif category == "Employment dispute":
        observations.append("- The matter appears to turn on proof of employment, work performed, and month-wise dues trail.")
        observations.append("- Internal communication with HR or management is likely to be central.")
    elif category == "Fraud or cheating concern":
        observations.append("- The matter appears to turn on what representation was made, how it was relied on, and the resulting loss.")
        observations.append("- Identity trail and transaction chronology will likely carry significant weight.")
    elif category == "Consumer issue":
        observations.append("- The matter appears to turn on what was promised, what was delivered, and how the defect or deficiency was documented.")
    elif category == "Cyber or online harm":
        observations.append("- The matter appears to turn on digital traceability, transaction identifiers, and speed of evidence preservation.")
    elif category == "Family or relationship dispute":
        observations.append("- The matter appears to turn on relationship records, shared household or dependency history, and chronology of events.")
    elif category == "Contract or payment dispute":
        observations.append("- The matter appears to turn on the underlying promise, breach event, payment or performance trail, and measurable loss.")
    else:
        observations.append(f"- The matter appears to turn on {_category_key_factors(category)}.")

    if not _is_placeholder_value(slots.get("evidence")):
        observations.append("- There is at least some documentary or digital material already identified in support of the case narrative.")
    return observations


def _report_information_gaps(category: str, slots: Dict[str, str]) -> List[str]:
    gaps: List[str] = []
    tracked_slots = [
        ("incident_location", "Exact incident location is not yet clearly stated."),
        ("timeline", "A tighter date-wise timeline would improve the report."),
        ("witnesses", "Witness position is not yet clear."),
        ("desired_outcome", "The user’s intended outcome is not yet clearly stated."),
    ]
    for slot, message in tracked_slots:
        if _is_placeholder_value(slots.get(slot)):
            gaps.append(f"- {message}")

    for slot in _REPORT_REQUIRED_BY_CATEGORY.get(category, []):
        if _is_placeholder_value(slots.get(slot)):
            gaps.append(f"- {_slot_label(slot)} is still incomplete.")

    if not gaps:
        gaps.append("- No major factual gaps identified from the current intake record.")
    return gaps


def _build_indian_report_draft(
    category: str, slots: Dict[str, str], retrieval: List[Dict[str, Any]]
) -> str:
    context_points = [f"- {text}" for text in _neutral_retrieval_points(retrieval, limit=3)]
    if not context_points:
        context_points.append("- Context based on user-provided facts.")

    report_title = _report_title(category, slots)
    timeline_text = slots.get("timeline") or slots.get("incident_date") or "Not specified by user"
    witness_text = slots.get("witnesses") or "No witness information provided."
    document_text = slots.get("evidence") or "No document evidence specified."
    background = slots.get("incident_summary") or "Background details were partially provided by the user."
    amount_impact = slots.get("harm_or_amount") or "Impact or amount not clearly specified."
    snapshot_lines = _report_case_snapshot(category, slots)
    category_lines = _category_specific_lines(category, slots)
    issue_lines = _report_issue_observations(category, slots)
    gap_lines = _report_information_gaps(category, slots)
    objective_line = None if _is_placeholder_value(slots.get("desired_outcome")) else slots.get("desired_outcome")

    return (
        f"Title\n{report_title}\n\n"
        "Legal Category\n"
        f"{category}\n\n"
        "Case Snapshot\n"
        + "\n".join(snapshot_lines)
        + "\n\n"
        "Background of the Issue\n"
        f"- Parties: {slots.get('parties', 'Not fully specified')}\n"
        f"- Location: {slots.get('incident_location', 'Not specified')}\n"
        f"- Background summary: {background}\n"
        f"- Reported impact/loss: {amount_impact}\n\n"
        f"{_category_specific_heading(category)}\n"
        + "\n".join(category_lines)
        + "\n\n"
        "Key Issues Observed\n"
        + "\n".join(issue_lines)
        + "\n\n"
        "Summary of Evidence\n"
        + "\n".join(context_points)
        + "\n\n"
        "Witness Information\n"
        f"- {witness_text}\n\n"
        "Document Evidence\n"
        f"- {document_text}\n"
        + "\n\n"
        "Timeline of Events\n"
        f"- {timeline_text}\n\n"
        "General Legal Context\n"
        f"{_category_context(category)}\n\n"
        + (
            "User Objective\n"
            f"- {objective_line}\n\n"
            if objective_line
            else ""
        )
        + "Information Gaps / Clarifications\n"
        + "\n".join(gap_lines)
        + "\n\n"
        "Suggested High-Level Next Steps\n"
        + _general_next_steps(category)
        + "\n\n"
        "Disclaimer\n"
        f"{NON_ADVICE_NOTE}"
    )


def _workflow_collected_facts(slots: Dict[str, str]) -> Dict[str, str]:
    preferred_order = [
        "incident_summary",
        "parties",
        "incident_date",
        "incident_location",
        "timeline",
        "evidence",
        "witnesses",
        "harm_or_amount",
        "desired_outcome",
        "property_details",
        "agreement_status",
        "employment_details",
        "dues_details",
        "counterparty_details",
        "transaction_details",
        "seller_service_details",
        "purchase_timeline",
        "platform_details",
        "digital_trail",
    ]
    facts: Dict[str, str] = {}
    for slot in preferred_order:
        value = slots.get(slot)
        if _slot_filled(value):
            facts[_slot_label(slot)] = str(value).strip()
    return facts


def _build_stage1_reply(category: str) -> str:
    return (
        f"{WORKFLOW_STAGE_LABELS[1]}\n"
        f"Category: {category}\n"
        f"{_category_context(category)}\n"
        f"Next: {WORKFLOW_STAGE_LABELS[2]}"
    )


def _build_stage4_next_steps(category: str) -> str:
    steps = [line for line in _general_next_steps(category).splitlines() if line.strip()]
    return (
        f"{WORKFLOW_STAGE_LABELS[4]}\n"
        "Suggested high-level next steps:\n"
        + "\n".join(steps[:3])
    )


def _workflow_stage(
    report_generated: bool,
    ready_for_report: bool,
    has_any_fact: bool,
    intake_active: bool,
) -> int:
    if report_generated:
        return 4
    if ready_for_report:
        return 3
    if has_any_fact or intake_active:
        return 2
    return 1

def _category_from_text(text: str) -> tuple[str, int]:
    best_category = "General legal issue"
    best_score = 0
    for category, keywords in LEGAL_CATEGORIES.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_category = category
    return best_category, best_score


def _last_category_from_history(history: List[ChatTurn]) -> Optional[str]:
    for turn in reversed(history):
        if turn.role != "assistant":
            continue
        lower = turn.content.lower()
        for category in LEGAL_CATEGORIES:
            if category.lower() in lower:
                return category
    return None


def _infer_legal_category(message: str, retrieval: List[Dict[str, Any]]) -> str:
    message_lower = message.lower()
    message_category, message_score = _category_from_text(message_lower)
    if message_score >= 1:
        return message_category

    combined = f"{message_lower} {message_lower} {message_lower}"
    for doc in retrieval[:4]:
        combined += f" {str(doc.get('text', '')).lower()} {str(doc.get('section', '')).lower()}"
    combined_category, _ = _category_from_text(combined)
    return combined_category


def _category_context(category: str) -> str:
    contexts = {
        "Property or tenancy issue": "This usually concerns rights and obligations around possession, use of property, rent terms, and documentary proof of ownership or occupancy.",
        "Contract or payment dispute": "This usually concerns promises between parties, whether terms were fulfilled, and whether money or services were delivered as agreed.",
        "Fraud or cheating concern": "This usually concerns alleged deception, false representation, and financial or practical loss caused by that conduct.",
        "Consumer issue": "This usually concerns product or service quality, fair trade expectations, and grievance resolution channels for buyers.",
        "Employment dispute": "This usually concerns working conditions, wages/benefits, termination process, and employer-employee obligations.",
        "Family or relationship dispute": "This usually concerns personal status, support obligations, family rights, and documentary family records.",
        "Cyber or online harm": "This usually concerns digital evidence, unauthorized access, online impersonation, or misuse of personal information.",
        "Defamation or reputation issue": "This usually concerns allegedly harmful statements and whether they caused reputational impact.",
        "Traffic or road incident": "This usually concerns compliance with road-safety rules, incident facts, and responsibility allocation.",
        "FIR and police complaint": (
            "This usually concerns how serious incidents are formally recorded by the police, what factual detail matters at the reporting stage, "
            "and how chronology, witnesses, and documentary proof affect the complaint process."
        ),
        "Bail and custody": (
            "This usually concerns custody status, release conditions, risk factors considered by courts, and whether the available facts and records support release."
        ),
        "General criminal concern": "This usually concerns allegations of wrongful conduct, evidence quality, and procedural safeguards under Indian criminal law.",
    }
    return contexts.get(category, "This appears to involve a legal disagreement where facts, documents, and timeline are central to understanding the issue.")



def _document_observations(retrieval: List[Dict[str, Any]]) -> str:
    uploaded_items = [d for d in retrieval if d.get("kind") == "uploaded"][:2]
    if not uploaded_items:
        return ""

    observations: List[str] = []
    for item in uploaded_items:
        text = _sanitize_reference_text(str(item.get("text", "")))
        if text:
            observations.append(f"- Uploaded document: {text[:180]}")
    if not observations:
        return ""
    return "Document observations:\n" + "\n".join(observations) + "\n\n"


def _retrieval_citations(retrieval: List[Dict[str, Any]], top_k: int = 3) -> List[Citation]:
    citations: List[Citation] = []
    seen: set[tuple[str, str]] = set()

    for item in retrieval:
        kind = str(item.get("kind", "")).strip() or "reference"
        item_id = str(item.get("id", "")).strip() or str(item.get("section", "")).strip()
        if not item_id:
            continue

        score = float(item.get("score", 0.0) or 0.0)
        minimum = 0.12 if kind in {"uploaded", "report"} else 0.17
        if score < minimum:
            continue

        dedupe_key = (kind, item_id)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        text = _sanitize_reference_text(str(item.get("text", "")))
        citations.append(
            Citation(
                kind=kind,
                id=item_id,
                section=str(item.get("section", "")).strip() or item_id,
                text=text[:220],
                source=str(item.get("source", "")).strip() or "unknown",
                effective_date=str(item.get("effective_date", "")).strip(),
                score=round(score, 4),
            )
        )
        if len(citations) >= top_k:
            break

    return citations


def _general_next_steps(category: str) -> str:
    common = [
        "- Build a dated timeline of events in plain language.",
        "- Gather relevant documents (communications, receipts, agreements, identity records, and proof of payment if applicable).",
        "- Verify process details on official government portals or authority websites.",
        "- Discuss the facts with a qualified lawyer or legal-aid service before taking decisions.",
    ]

    if category == "Cyber or online harm":
        common.insert(2, "- Preserve screenshots, transaction logs, account alerts, and device/email headers.")
    elif category == "Property or tenancy issue":
        common.insert(2, "- Organize property papers, rent records, correspondence, and proof of possession/occupancy.")
    elif category == "Employment dispute":
        common.insert(2, "- Compile offer letters, salary records, HR emails, attendance logs, and any policy documents.")

    return "\n".join(common)


def _category_key_factors(category: str) -> str:
    factors = {
        "Property or tenancy issue": "possession, the written or oral tenancy/title record, payment trail, and communication around notice, deposit, or possession",
        "Contract or payment dispute": "the promise made, proof of payment or performance, breach chronology, and measurable loss",
        "Fraud or cheating concern": "what representation was made, how the user relied on it, the money or property trail, and evidence of deception",
        "Consumer issue": "what was promised, what was delivered, when the defect was reported, and the complaint record",
        "Employment dispute": "proof of employment, actual work performed, the pay or benefit trail, and employer communication",
        "Family or relationship dispute": "relationship records, timeline of events, financial dependence, living arrangement, and communication history",
        "Cyber or online harm": "the digital trail, account or transaction identifiers, chronology of the incident, and preservation of raw evidence",
        "Defamation or reputation issue": "what was said, where it was published, who saw it, and how the statement caused harm",
        "Traffic or road incident": "the incident sequence, vehicle details, identity of persons involved, medical or repair records, and witness or camera material",
        "FIR and police complaint": "clear chronology, identity details, seriousness of the allegation, immediate supporting material, and consistency of the complaint narrative",
        "Bail and custody": "custody status, seriousness of allegations, personal ties, conduct record, and risk concerns around release",
        "General criminal concern": "the allegation, timeline, available proof, witness support, and procedural safeguards",
    }
    return factors.get(category, "the facts, the timeline, the documents, and whether the available material supports the user’s version clearly")


def _comparison_context(category: str) -> Optional[str]:
    if category == "Property or tenancy issue":
        return (
            "In broad terms, tenant-side issues usually concern possession, peaceful use, notice, deposit handling, and protection from arbitrary eviction. "
            "Owner-side issues usually concern title, rent recovery, damage claims, and lawful recovery of possession."
        )
    if category == "Employment dispute":
        return (
            "In broad terms, employee-side issues usually concern unpaid dues, unfair process, or workplace treatment. "
            "Employer-side issues usually concern performance record, policy compliance, and proof of communication."
        )
    return None


def _document_focus(category: str) -> str:
    mapping = {
        "Property or tenancy issue": "Most useful records are the agreement or title papers, payment proof, possession or occupancy proof, municipal or tax records, and message history.",
        "Contract or payment dispute": "Most useful records are the agreement, invoice or payment proof, delivery or performance record, and all notice or follow-up communication.",
        "Fraud or cheating concern": "Most useful records are transaction proof, account or identity details, screenshots, call or message history, and any document showing the false representation.",
        "Consumer issue": "Most useful records are the invoice, warranty or policy terms, defect photos, complaint history, and seller or platform responses.",
        "Employment dispute": "Most useful records are the offer letter, payslips, bank credits, attendance material, HR mails, and settlement or termination communication.",
        "Family or relationship dispute": "Most useful records are relationship proof, communication records, financial records, shared household records, and any chronology of incidents.",
        "Cyber or online harm": "Most useful records are transaction IDs, account screenshots, alerts, raw emails or chats, device logs, and platform or bank complaint references.",
        "FIR and police complaint": "Most useful records are a date-wise chronology, identity details, medical or damage proof where relevant, witness details, and supporting digital or physical records.",
    }
    return mapping.get(category, "Most useful records are the core documents, payment trail, communication history, and a dated timeline.")


def _is_action_heavy_text(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:file|register|lodge|send|approach|apply|call|report|complaint|petition|suit|legal notice|court|magistrate)\b",
            text,
            re.IGNORECASE,
        )
    )


def _neutral_retrieval_points(retrieval: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    preferred: List[str] = []
    fallback: List[str] = []

    for item in retrieval:
        if item.get("kind") not in {"uploaded", "law"}:
            continue
        text = _sanitize_reference_text(str(item.get("text", "")))
        if not text:
            continue
        if _is_action_heavy_text(text):
            fallback.append(text[:180])
        else:
            preferred.append(text[:180])

    return (preferred + fallback)[:limit]


def _retrieval_takeaways(retrieval: List[Dict[str, Any]], limit: int = 2) -> List[str]:
    takeaways: List[str] = []
    for text in _neutral_retrieval_points(retrieval, limit=max(limit + 2, 4)):
        takeaways.append(text[:190])
        if len(takeaways) >= limit:
            break
    return takeaways


def _high_level_steps_line(category: str) -> str:
    if category == "Cyber or online harm":
        return "At a high level, preserve the digital trail immediately, keep a clean incident timeline, and verify the reporting path on official channels."
    if category == "Property or tenancy issue":
        return "At a high level, organize the property or tenancy papers first, map the payment and possession timeline, and separate facts from assumptions."
    if category == "Employment dispute":
        return "At a high level, organize employment records, map the dues month by month, and keep the communication trail clean and dated."
    return "At a high level, organize the timeline first, gather the main records, and verify the relevant official process before taking decisions."


def _build_information_reply(
    message: str, retrieval: List[Dict[str, Any]], history: Optional[List[ChatTurn]] = None
) -> str:
    history = history or []
    category = _infer_legal_category(message, retrieval)
    if _context_dependent_query(message):
        prior_category = _last_category_from_history(history)
        if prior_category:
            category = prior_category
    lower_message = message.lower()
    asks_comparison = bool(re.search(r"\b(vs|versus|difference|compare|comparison)\b", lower_message))
    asks_general_info = bool(
        re.search(
            r"\b(explain|overview|difference|what is|what are|how do i prove|what documents should|what details should|what should i do|checklist|preserve|guide)\b",
            lower_message,
        )
    )
    asks_for_intake = (
        (_needs_more_facts(message) or _context_dependent_query(message))
        and not asks_general_info
    )
    questions = []
    if asks_for_intake:
        questions = _missing_detail_questions(
            message=message,
            history=history,
            category=category,
            has_uploaded=any(item.get("kind") == "uploaded" for item in retrieval),
        )[:2]

    asks_steps = bool(re.search(r"\b(next step|next steps|what should i do|how should i proceed|guide)\b", lower_message))
    asks_documents = bool(re.search(r"\b(document|documents|checklist|proof|evidence|records)\b", lower_message))
    asks_report = bool(re.search(r"\b(report|draft|format|complaint)\b", lower_message))
    has_uploaded = any(item.get("kind") == "uploaded" for item in retrieval)
    takeaways = _retrieval_takeaways(retrieval, limit=2)
    variant = _style_variant(message, history)
    opening_lines = [
        f"This falls under {category.lower()}.",
        f"On these facts, the issue is mainly {category.lower()}.",
        f"The legal category here is {category.lower()}.",
    ]

    lines = [opening_lines[variant]]

    comparison_line = _comparison_context(category) if asks_comparison else None
    if comparison_line:
        lines.append(comparison_line)
    else:
        lines.append(f"In practice, this usually turns on {_category_key_factors(category)}.")

    if takeaways:
        lines.append(f"Relevant context: {takeaways[0]}")
        if len(takeaways) > 1 and (asks_general_info or asks_documents or asks_steps):
            lines.append(f"Also relevant: {takeaways[1]}")

    if has_uploaded and (_context_dependent_query(message) or asks_report or asks_documents):
        lines.append("I can also work from the uploaded material already in the case record, not just the statutory database.")

    if asks_documents:
        lines.append(_document_focus(category))
    elif asks_steps:
        lines.append(_high_level_steps_line(category))
    elif asks_general_info:
        lines.append(_category_context(category))

    if questions:
        lines.append("To make this specific, I only need:")
        for index, question in enumerate(questions, start=1):
            lines.append(f"{index}. {question}")
    elif asks_report:
        lines.append("If you want, I can turn this into a structured case report and only flag the facts that are still missing.")
    else:
        closing_lines = [
            "If you want, I can go one level deeper on the legal context or switch into report-building mode.",
            "If useful, I can now narrow this down with one or two focused questions or draft a report from the current facts.",
            "If you want to continue, I can either explain the legal position further or convert this into a structured case summary.",
        ]
        lines.append(closing_lines[variant])

    if asks_steps or asks_documents or questions or asks_report:
        lines.append(NON_ADVICE_NOTE)

    return "\n".join(lines)


def _detect_document_type(filename: str) -> str:
    _, ext = os.path.splitext(filename.lower())
    if ext in SUPPORTED_UPLOAD_EXTENSIONS:
        return ext.lstrip(".")
    raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")


def _extract_txt_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _extract_pdf_text(data: bytes) -> str:
    try:
        pypdf_module = import_module("pypdf")
    except ImportError as exc:
        raise RuntimeError("PDF parsing dependency missing. Install pypdf.") from exc
    reader = pypdf_module.PdfReader(BytesIO(data))
    parts: List[str] = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _extract_docx_text(data: bytes) -> str:
    try:
        docx_module = import_module("docx")
    except ImportError as exc:
        raise RuntimeError("DOCX parsing dependency missing. Install python-docx.") from exc
    document = docx_module.Document(BytesIO(data))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _extract_upload_text(filename: str, data: bytes) -> tuple[str, str]:
    document_type = _detect_document_type(filename)
    if document_type == "txt":
        text = _extract_txt_text(data)
    elif document_type == "pdf":
        text = _extract_pdf_text(data)
    else:
        text = _extract_docx_text(data)
    normalized = re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
    return document_type, normalized


def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip().lower()).strip("-")
    return cleaned or "saulgpt-report"


def _build_pdf_bytes(title: str, content: str) -> bytes:
    try:
        canvas_module = import_module("reportlab.pdfgen.canvas")
        pagesizes_module = import_module("reportlab.lib.pagesizes")
    except ImportError as exc:
        raise RuntimeError("PDF export dependency missing. Install reportlab.") from exc

    buffer = BytesIO()
    canvas = canvas_module.Canvas(buffer, pagesize=pagesizes_module.A4)
    width, height = pagesizes_module.A4
    margin_x, margin_y = 48, 56
    text_obj = canvas.beginText(margin_x, height - margin_y)
    text_obj.setFont("Helvetica-Bold", 14)
    text_obj.textLine(title)
    text_obj.moveCursor(0, 10)
    text_obj.setFont("Helvetica", 11)

    max_chars = max(50, int((width - (2 * margin_x)) / 6))
    for paragraph in content.splitlines():
        wrapped = [paragraph[i : i + max_chars] for i in range(0, len(paragraph), max_chars)]
        if not wrapped:
            wrapped = [""]
        for line in wrapped:
            if text_obj.getY() <= margin_y:
                canvas.drawText(text_obj)
                canvas.showPage()
                text_obj = canvas.beginText(margin_x, height - margin_y)
                text_obj.setFont("Helvetica", 11)
            text_obj.textLine(line)
        text_obj.textLine("")
    canvas.drawText(text_obj)
    canvas.save()
    return buffer.getvalue()


def _build_docx_bytes(title: str, content: str) -> bytes:
    try:
        docx_module = import_module("docx")
    except ImportError as exc:
        raise RuntimeError("DOCX export dependency missing. Install python-docx.") from exc

    document = docx_module.Document()
    document.add_heading(title, level=1)
    for paragraph in content.split("\n\n"):
        text = paragraph.strip()
        if text:
            document.add_paragraph(text)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _build_txt_bytes(title: str, content: str) -> bytes:
    text = f"{title}\n\n{content}".strip() + "\n"
    return text.encode("utf-8")


def _build_context_block(context_docs: List[Dict[str, Any]]) -> str:
    if not context_docs:
        return "No directly matching references were retrieved."

    context_lines = []
    for i, doc in enumerate(context_docs, start=1):
        safe_text = _sanitize_reference_text(str(doc.get("text", "")))
        context_lines.append(
            f"{i}. ({str(doc.get('kind', 'reference')).upper()}) {safe_text} "
            f"(source: {doc.get('source', 'unknown')}, score: {doc.get('score', 0)})"
        )
    return "\n".join(context_lines)


def _ollama_chat_enabled() -> bool:
    global _ollama_available_cache, _ollama_last_check_at

    if OLLAMA_CHAT_MODE in {"0", "false", "off", "no"}:
        return False
    if OLLAMA_CHAT_MODE in {"1", "true", "on", "yes"}:
        return True

    now = time.time()
    if (
        _ollama_available_cache is not None
        and now - _ollama_last_check_at < OLLAMA_DISCOVERY_TTL_SECONDS
    ):
        return _ollama_available_cache

    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=1.0)
        _ollama_available_cache = response.ok
    except requests.RequestException:
        _ollama_available_cache = False
    _ollama_last_check_at = now
    return bool(_ollama_available_cache)


def _call_ollama_text(prompt: str, num_predict: int) -> str:
    last_error: Optional[Exception] = None

    for model_name in MODEL_CANDIDATES:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": num_predict,
            },
        }

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
                response.raise_for_status()
                raw = response.json().get("response", "").strip()
                if raw:
                    return raw
                raise ValueError(f"empty model response from {model_name}")
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == MAX_RETRIES:
                    break
                time.sleep(0.4 * (attempt + 1))

    raise HTTPException(status_code=502, detail=f"model call failed: {last_error}")


def _chat_prompt(message: str, history: List[ChatTurn], context_docs: List[Dict[str, Any]]) -> str:
    recent_turns = history[-8:]
    history_lines = [f"{turn.role.upper()}: {turn.content}" for turn in recent_turns]
    history_block = "\n".join(history_lines) if history_lines else "No prior context."

    return f"""
You are SaulGPT, a highly capable Indian-law legal information and drafting assistant.
Follow these rules strictly:
- Do NOT provide specific legal advice.
- Do NOT cite exact sections, statute numbers, or legal provisions.
- Do NOT instruct the user to take specific legal action.
- Do NOT present yourself as a lawyer or legal authority.
- Sound like a sharp legal analyst: direct, grounded, clear, interactive, and concise.
- Respond with ChatGPT-like quality: adapt to the user, avoid template language, and do not pad the answer.
- If the user message depends on prior context, use earlier details.
- If the user asks a broad legal question, answer it first instead of interrogating the user.
- If the user is discussing a real case, identify what legally matters and ask only the next missing facts.
- Ask at most two focused follow-up questions only when required.
- Do not repeat questions already answered in the conversation.
- If the user wants a report and the facts are already sufficient, generate it instead of asking more questions.
- Prefer short paragraphs. Use bullets only for checklists or stepwise guidance.
- Use uploaded documents as part of the case record when they are present in the references.

Use these references first:
{_build_context_block(context_docs)}

Conversation so far:
{history_block}

User message:
{message}

Output format:
- Start with the direct answer, not labels.
- Identify the likely legal category naturally in the answer.
- Use retrieved material when relevant.
- Ask 0-2 numbered follow-up questions only if missing facts are essential.
- Keep it concise, informative, and human.
- Keep total length under 220 words unless the user asks for a report.
""".strip()


def _build_generate_prompt(data: CaseRequest, context_docs: List[Dict[str, Any]]) -> str:
    return f"""
You are SaulGPT, a highly capable Indian-law legal information and drafting assistant.
Provide grounded, practical legal information in a professional tone.
Do not provide specific legal advice.
Do not cite exact section numbers or statute provisions.
Do not recommend specific legal actions.
Be direct and concise. Avoid boilerplate and unnecessary disclaimer text.

Retrieved legal context:
{_build_context_block(context_docs)}

Case details:
- Case type: {data.case_type}
- Incident: {data.incident}
- Amount involved: {data.amount or "Not specified"}

Write a direct answer that:
- identifies the likely legal category
- explains the general legal context in plain English
- gives high-level guidance only
- gives short next steps only if useful
- sounds natural, not robotic
""".strip()


def _fallback_chat_reply(
    message: str, retrieval: List[Dict[str, Any]], history: Optional[List[ChatTurn]] = None
) -> str:
    return _build_information_reply(message, retrieval, history)


def _needs_more_facts(message: str) -> bool:
    lower = message.lower()
    scenario_markers = {
        "my ", "i ", "we ", "our ", "accused", "incident", "happened", "paid", "payment",
        "took", "threat", "dispute", "arrest", "transaction", "landlord", "tenant",
        "notice received", "refusing", "stopped responding", "police refused",
    }
    if any(marker in lower for marker in scenario_markers):
        return True
    if re.search(r"\b(i|my|me|we|our)\b", lower):
        return True
    # General doctrinal/citizen-law Q&A usually does not require a missing-facts block.
    general_markers = {
        "explain", "difference", "what is", "what are", "overview", "define",
        "meaning", "penalty", "procedure", "rights", "remedies", "law for",
    }
    if any(marker in lower for marker in general_markers):
        return False
    return False


def _fallback_draft(data: CaseRequest, retrieval: List[Dict[str, Any]]) -> str:
    query = f"{data.case_type}. {data.incident}. Amount: {data.amount or 'Not specified'}"
    return _build_information_reply(query, retrieval, history=[])


def _clean_reply(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = _sanitize_reference_text(cleaned, collapse_whitespace=False)
    cleaned = re.sub(
        r"\b(file|register|lodge|initiate)\b[^.\n]*\b(complaint|fir|case|petition|suit)\b",
        "contact the appropriate authority after getting professional legal advice",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


def _finalize_chat_reply(
    message: str,
    reply: str,
    history: Optional[List[ChatTurn]] = None,
    retrieval: Optional[List[Dict[str, Any]]] = None,
) -> str:
    cleaned = _clean_reply(reply)
    if len(cleaned.split()) < 18:
        cleaned = _build_information_reply(message, retrieval or [], history or [])
    return cleaned


def _chat_cache_key(message: str, history: List[ChatTurn]) -> str:
    recent = history[-6:]
    hist = "|".join(f"{t.role}:{t.content}" for t in recent)
    return f"{message.strip().lower()}||{hist.strip().lower()}"


def _get_cached_chat(message: str, history: List[ChatTurn]) -> Optional[ChatResponse]:
    key = _chat_cache_key(message, history)
    hit = _chat_cache.get(key)
    if not hit:
        return None
    ts, payload = hit
    if time.time() - ts > CHAT_CACHE_TTL_SECONDS:
        _chat_cache.pop(key, None)
        return None
    return ChatResponse(**payload)


def _set_cached_chat(message: str, history: List[ChatTurn], response: ChatResponse) -> None:
    key = _chat_cache_key(message, history)
    _chat_cache[key] = (time.time(), response.model_dump())


def _is_reply_grounded(reply: str, retrieval: List[Dict[str, Any]]) -> bool:
    if not retrieval:
        return True
    lower = reply.lower()
    for doc in retrieval[:5]:
        doc_id = str(doc.get("id", "")).lower()
        section = str(doc.get("section", "")).lower()
        if doc_id and doc_id in lower:
            return True
        if section and section in lower:
            return True
        numbers = re.findall(r"\d+[a-zA-Z]?", f"{doc_id} {section}")
        if any(n.lower() in lower for n in numbers):
            return True
    return False


def _disclaimer() -> str:
    return (
        "SaulGPT provides general legal information only and does not provide legal advice. "
        "For decisions in your specific matter, consult a qualified lawyer."
    )


@app.get("/")
def home() -> FileResponse:
    return FileResponse(INDEX_FILE)

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_candidates": MODEL_CANDIDATES,
        "timeout_seconds": OLLAMA_TIMEOUT_SECONDS,
        "chat_cache_ttl_seconds": CHAT_CACHE_TTL_SECONDS,
        "ollama_chat_mode": OLLAMA_CHAT_MODE,
        "ollama_chat_enabled": _ollama_chat_enabled(),
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(data: CaseRequest, request: Request, x_api_key: Optional[str] = Header(default=None)) -> GenerateResponse:
    _check_api_key(x_api_key)
    client_id = request.client.host if request.client else "unknown"
    _check_rate_limit(client_id)

    query = f"{data.case_type} {data.incident} {data.amount or ''}".strip()
    retrieval = search_knowledge(query, top_k=7)
    draft = _build_information_reply(query, retrieval, history=[])
    if _ollama_chat_enabled():
        try:
            llm_text = _call_ollama_text(
                _build_generate_prompt(data, retrieval),
                num_predict=GENERATE_MAX_TOKENS,
            )
            if len(llm_text.split()) >= 35:
                draft = llm_text
        except HTTPException:
            draft = _fallback_draft(data, retrieval)

    return GenerateResponse(
        draft=_clean_reply(draft),
        citations=_retrieval_citations(retrieval),
        disclaimer=_disclaimer(),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(data: ChatRequest, request: Request, x_api_key: Optional[str] = Header(default=None)) -> ChatResponse:
    _check_api_key(x_api_key)
    client_id = request.client.host if request.client else "unknown"
    _check_rate_limit(client_id)

    cached = _get_cached_chat(data.message, data.history)
    if cached:
        return cached

    legal_intent = _is_law_topic(data.message)
    if not legal_intent and _report_intake_in_progress(data.history):
        legal_intent = True
    if not legal_intent and _context_dependent_query(data.message):
        legal_intent = _history_has_legal_context(data.history)
    if not legal_intent and _history_has_legal_context(data.history):
        if len(data.message.split()) <= 14 or data.message.strip().endswith("?"):
            legal_intent = True

    if not legal_intent:
        response = ChatResponse(
            reply=(
                f"{LAW_FOCUS_GUIDANCE} "
                "You can ask things like: 'What type of legal issue is this?', "
                "'What is the general legal context?', or 'What documents should I gather?'"
            ),
            citations=[],
            redirected_to_law=True,
            disclaimer=_disclaimer(),
            case_workflow=CaseWorkflow(
                enabled=False,
                stage=1,
                stage_label=WORKFLOW_STAGE_LABELS[1],
                legal_category="General legal issue",
                ready_for_report=False,
                report_generated=False,
                missing_fields=[],
                collected_facts={},
            ),
        )
        _set_cached_chat(data.message, data.history, response)
        return response

    retrieval_query = _compose_retrieval_query(data.message, data.history)
    retrieval = search_knowledge(retrieval_query, top_k=9)
    workflow_enabled = _workflow_mode_requested(data.message, data.history)
    if workflow_enabled:
        category = _infer_legal_category(data.message, retrieval)
        prior_category = _last_category_from_history(data.history)
        if prior_category and (
            _report_intake_in_progress(data.history)
            or _context_dependent_query(data.message)
            or _report_mode_requested(data.message, data.history)
        ):
            category = prior_category

        slots = _extract_report_slots(
            message=data.message,
            history=data.history,
            category=category,
            retrieval=retrieval,
        )
        missing_slots = _resolve_missing_report_slots(
            category=category,
            slots=slots,
            history=data.history,
        )
        ready_for_report = _has_minimum_facts_for_report(
            category=category,
            slots=slots,
            missing_slots=missing_slots,
            message=data.message,
        )
        requested_report_now = bool(_REPORT_GENERATE_RE.search(data.message.lower()))
        force_report_generation = _should_force_report_generation(
            category=category,
            slots=slots,
            message=data.message,
            history=data.history,
        )
        report_generated = False
        facts = _workflow_collected_facts(slots)
        has_any_fact = bool(facts)

        if ready_for_report or force_report_generation:
            report_generated = True
            finalized_slots = _finalize_report_slots(category, slots)
            report_text = _build_indian_report_draft(category, finalized_slots, retrieval)
            reply = (
                f"{WORKFLOW_STAGE_LABELS[3]}\n\n"
                f"{report_text}\n\n"
                f"{_build_stage4_next_steps(category)}"
            )
            missing_slots = []
            facts = _workflow_collected_facts(finalized_slots)
        elif requested_report_now:
            questions = _next_report_questions(category, missing_slots, data.history)
            reply = _build_report_intake_reply(
                category=category,
                slots=slots,
                questions=questions,
                missing_slots=missing_slots,
            )
        elif has_any_fact or _report_mode_requested(data.message, data.history):
            questions = _next_report_questions(category, missing_slots, data.history)
            reply = _build_report_intake_reply(
                category=category,
                slots=slots,
                questions=questions,
                missing_slots=missing_slots,
            )
        else:
            reply = _build_stage1_reply(category)
            if missing_slots:
                seed_questions = _next_report_questions(category, missing_slots, data.history)
                if seed_questions:
                    reply = (
                        f"{reply}\n\n"
                        "To begin intake, please share:\n"
                        + "\n".join(
                            f"{index}. {question}"
                            for index, question in enumerate(seed_questions, start=1)
                        )
                    )

        stage = _workflow_stage(
            report_generated=report_generated,
            ready_for_report=ready_for_report,
            has_any_fact=has_any_fact,
            intake_active=(has_any_fact or _report_mode_requested(data.message, data.history)),
        )
        workflow = CaseWorkflow(
            enabled=True,
            stage=stage,
            stage_label=WORKFLOW_STAGE_LABELS[stage],
            legal_category=category,
            ready_for_report=ready_for_report,
            report_generated=report_generated,
            missing_fields=_missing_field_labels(missing_slots),
            collected_facts=facts,
        )

        response = ChatResponse(
            reply=_clean_reply(reply),
            citations=_retrieval_citations(retrieval),
            redirected_to_law=False,
            disclaimer=_disclaimer(),
            case_workflow=workflow,
        )
        _set_cached_chat(data.message, data.history, response)
        return response

    reply = _build_information_reply(data.message, retrieval, data.history)

    if _ollama_chat_enabled():
        try:
            llm_text = _call_ollama_text(
                _chat_prompt(data.message, data.history, retrieval),
                num_predict=CHAT_MAX_TOKENS,
            )
            if len(llm_text.split()) >= 35:
                reply = llm_text
        except HTTPException:
            reply = _fallback_chat_reply(data.message, retrieval, data.history)

    response = ChatResponse(
        reply=_finalize_chat_reply(data.message, reply, data.history, retrieval),
        citations=_retrieval_citations(retrieval),
        redirected_to_law=False,
        disclaimer=_disclaimer(),
        case_workflow=CaseWorkflow(
            enabled=False,
            stage=1,
            stage_label=WORKFLOW_STAGE_LABELS[1],
            legal_category=_infer_legal_category(data.message, retrieval),
            ready_for_report=False,
            report_generated=False,
            missing_fields=[],
            collected_facts={},
        ),
    )
    _set_cached_chat(data.message, data.history, response)
    return response


@app.post("/api/chat", response_model=ChatResponse)
def chat_api_alias(
    data: ChatRequest, request: Request, x_api_key: Optional[str] = Header(default=None)
) -> ChatResponse:
    return chat(data, request, x_api_key)


@app.post("/documents/ingest", response_model=IngestResponse)
@app.post("/api/documents/ingest", response_model=IngestResponse)
async def documents_ingest(
    request: Request,
    files: List[UploadFile] = File(...),
    x_api_key: Optional[str] = Header(default=None),
) -> IngestResponse:
    _check_api_key(x_api_key)
    client_id = request.client.host if request.client else "unknown"
    _check_rate_limit(client_id)

    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    results: List[IngestFileResult] = []
    ingested_files = 0
    total_chunks = 0

    for file in files:
        filename = (file.filename or "uploaded_document.txt").strip()
        try:
            raw = await file.read()
            if not raw:
                raise ValueError("File is empty.")
            if len(raw) > MAX_UPLOAD_BYTES:
                raise ValueError("File is too large. Max supported size is 10 MB.")

            document_type, text = _extract_upload_text(filename, raw)
            if not text:
                raise ValueError("No readable text found in file.")

            chunks = _chunk_text(text)
            if not chunks:
                raise ValueError("Could not split document into chunks.")

            added = add_uploaded_document_chunks(filename, document_type, chunks)
            ingested_files += 1
            total_chunks += added
            results.append(
                IngestFileResult(
                    filename=filename,
                    document_type=document_type,
                    status="ingested",
                    chunks=added,
                    detail="Indexed successfully.",
                )
            )
        except Exception as exc:
            results.append(
                IngestFileResult(
                    filename=filename,
                    document_type=None,
                    status="failed",
                    chunks=0,
                    detail=str(exc),
                )
            )

    if ingested_files == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents were ingested. Check file type and parsing dependencies.",
        )

    return IngestResponse(
        total_files=len(files),
        ingested_files=ingested_files,
        failed_files=len(files) - ingested_files,
        total_chunks=total_chunks,
        results=results,
    )


@app.post("/reports/export")
@app.post("/api/reports/export")
def export_report(
    data: ReportExportRequest,
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
) -> Response:
    _check_api_key(x_api_key)
    client_id = request.client.host if request.client else "unknown"
    _check_rate_limit(client_id)

    title = data.title.strip() or "SaulGPT Legal Report"
    content = data.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Report content cannot be empty.")

    base_name = _safe_filename(data.filename or title)

    try:
        if data.format == "pdf":
            payload = _build_pdf_bytes(title, content)
            media_type = "application/pdf"
            filename = f"{base_name}.pdf"
        elif data.format == "docx":
            payload = _build_docx_bytes(title, content)
            media_type = (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            filename = f"{base_name}.docx"
        else:
            payload = _build_txt_bytes(title, content)
            media_type = "text/plain; charset=utf-8"
            filename = f"{base_name}.txt"
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(
        content=payload,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
