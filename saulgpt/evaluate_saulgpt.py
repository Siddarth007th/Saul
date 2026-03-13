import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


STATUTE_REPLY_RE = re.compile(
    r"\b(?:section|sec\.?|article|order)\s+\d+[a-zA-Z]?\b|\b(?:ipc|crpc|cpc|bns|bnss|bsa|ni act|it act)\b",
    re.IGNORECASE,
)


@dataclass
class RetrievalCase:
    name: str
    query: str
    expected_sections: Tuple[str, ...]
    expected_top_n: int = 3


@dataclass
class ChatCase:
    name: str
    prompt: str
    expected_category_terms: Tuple[str, ...]
    expected_terms: Tuple[str, ...] = ()
    min_citations: int = 1
    expect_redirect: bool = False
    max_words: int = 260


@dataclass
class WorkflowTurn:
    role: str
    content: str


@dataclass
class WorkflowCase:
    name: str
    first_message: str
    followup_message: str
    expected_category: str
    expected_fact_keys: Tuple[str, ...]
    required_report_headings: Tuple[str, ...] = (
        "Title",
        "Legal Category",
        "Background of the Issue",
        "Summary of Evidence",
        "Witness Information",
        "Document Evidence",
        "Timeline of Events",
        "General Legal Context",
        "Suggested High-Level Next Steps",
        "Disclaimer",
    )


RETRIEVAL_CASES: List[RetrievalCase] = [
    RetrievalCase(
        name="rental_overview",
        query="explain rental laws in india",
        expected_sections=("Property Guide - Rental Law Overview", "Rent Control (State-specific)"),
    ),
    RetrievalCase(
        name="tenant_vs_owner",
        query="difference between tenant rights and owner rights",
        expected_sections=("Property Guide - Tenant vs Owner Positions",),
    ),
    RetrievalCase(
        name="land_ownership",
        query="how do i prove ownership of my land",
        expected_sections=("Property Guide - Land Ownership Overview", "Ownership Proof Checklist"),
    ),
    RetrievalCase(
        name="salary_overview",
        query="what is the general law around unpaid salary",
        expected_sections=("Employment Guide - Unpaid Salary", "Employment Guide - Salary Rights Overview"),
    ),
]


CHAT_CASES: List[ChatCase] = [
    ChatCase(
        name="rental_overview_chat",
        prompt="Explain rental laws in India.",
        expected_category_terms=("property", "tenancy", "tenant", "landlord"),
        expected_terms=("agreement", "deposit", "notice"),
        min_citations=1,
    ),
    ChatCase(
        name="owner_tenant_comparison_chat",
        prompt="Difference between tenant rights and owner rights",
        expected_category_terms=("property", "tenancy", "tenant", "owner"),
        expected_terms=("tenant", "owner"),
        min_citations=1,
    ),
    ChatCase(
        name="salary_chat",
        prompt="What is the general law around unpaid salary?",
        expected_category_terms=("employment", "salary", "wage", "dues"),
        expected_terms=("employment", "salary", "records"),
        min_citations=1,
    ),
    ChatCase(
        name="non_legal_redirect",
        prompt="Tell me a joke about cricket.",
        expected_category_terms=(),
        min_citations=0,
        expect_redirect=True,
        max_words=120,
    ),
]


WORKFLOW_CASES: List[WorkflowCase] = [
    WorkflowCase(
        name="property_report_flow",
        first_message="My landlord is not returning my deposit. Please prepare a report.",
        followup_message="Bengaluru. I vacated on 1 Feb 2026. The landlord is Rajesh. I have the rent agreement, bank transfer proof, and WhatsApp chats.",
        expected_category="Property or tenancy issue",
        expected_fact_keys=("Incident location", "Incident date", "Document evidence", "Property details"),
    ),
    WorkflowCase(
        name="employment_report_flow",
        first_message="My employer has not paid my salary for three months. Generate a report.",
        followup_message="Chennai. I work as a sales manager. Salary from December 2025 to February 2026 is unpaid. I have offer letter, payslips, attendance proof, and HR emails.",
        expected_category="Employment dispute",
        expected_fact_keys=("Incident location", "Incident date", "Employment details", "Pending dues details"),
    ),
]


def _load_local_client():
    from fastapi.testclient import TestClient

    from saulgpt_api import app

    return TestClient(app)


def _word_count(text: str) -> int:
    return len(text.split())


def _has_any(text: str, terms: Sequence[str]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms)


def _record_result(
    suite: List[Dict[str, Any]],
    name: str,
    score: int,
    max_score: int,
    notes: List[str],
    extra: Optional[Dict[str, Any]] = None,
) -> int:
    payload: Dict[str, Any] = {
        "name": name,
        "score": score,
        "max_score": max_score,
        "notes": notes,
    }
    if extra:
        payload.update(extra)
    suite.append(payload)
    return score


def _evaluate_retrieval() -> Tuple[List[Dict[str, Any]], int, int]:
    from legal_rag import search_knowledge

    results: List[Dict[str, Any]] = []
    total_score = 0
    max_score = len(RETRIEVAL_CASES) * 3

    for case in RETRIEVAL_CASES:
        started = time.perf_counter()
        hits = search_knowledge(case.query, top_k=5)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        sections = [str(item.get("section", "")) for item in hits]
        score = 0
        notes: List[str] = []

        top_sections = sections[: case.expected_top_n]
        if any(expected in top_sections for expected in case.expected_sections):
            score += 2
        else:
            notes.append("expected overview result not in top ranking")

        if hits and float(hits[0].get("score", 0.0) or 0.0) >= 0.45:
            score += 1
        else:
            notes.append("top retrieval score weak")

        total_score += _record_result(
            results,
            case.name,
            score,
            3,
            notes,
            {
                "query": case.query,
                "latency_ms": elapsed_ms,
                "top_sections": sections[:5],
            },
        )

    return results, total_score, max_score


def _post_chat_http(base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def _post_chat_local(client, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = client.post("/api/chat", json=payload)
    response.raise_for_status()
    return response.json()


def _evaluate_chat(client, base_url: Optional[str]) -> Tuple[List[Dict[str, Any]], int, int]:
    results: List[Dict[str, Any]] = []
    total_score = 0
    max_score = len(CHAT_CASES) * 6

    for case in CHAT_CASES:
        payload = {"message": case.prompt, "history": []}
        started = time.perf_counter()
        data = _post_chat_http(base_url, payload) if base_url else _post_chat_local(client, payload)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        reply = str(data.get("reply", ""))
        citations = data.get("citations", [])
        redirected = bool(data.get("redirected_to_law"))
        score = 0
        notes: List[str] = []

        if case.expect_redirect:
            if redirected:
                score += 2
            else:
                notes.append("non-legal prompt did not redirect")
            if "error" not in reply.lower():
                score += 1
            else:
                notes.append("redirect reply sounds like an error")
            if _word_count(reply) <= case.max_words:
                score += 1
            else:
                notes.append("redirect reply too long")
            if citations == []:
                score += 1
            else:
                notes.append("redirect reply should not include citations")
            if "legal" in reply.lower():
                score += 1
            else:
                notes.append("redirect reply missing legal framing")
        else:
            if not redirected:
                score += 1
            else:
                notes.append("legal prompt was redirected")

            if citations and len(citations) >= case.min_citations:
                score += 1
            else:
                notes.append("insufficient citations")

            if not STATUTE_REPLY_RE.search(reply):
                score += 1
            else:
                notes.append("reply cites exact statutory references")

            if _has_any(reply, case.expected_category_terms):
                score += 1
            else:
                notes.append("reply missing expected legal category/context")

            if not case.expected_terms or _has_any(reply, case.expected_terms):
                score += 1
            else:
                notes.append("reply missing expected practical/legal details")

            if 35 <= _word_count(reply) <= case.max_words:
                score += 1
            else:
                notes.append("reply length outside target range")

        total_score += _record_result(
            results,
            case.name,
            score,
            6,
            notes,
            {
                "prompt": case.prompt,
                "latency_ms": elapsed_ms,
                "redirected_to_law": redirected,
                "citations": len(citations),
                "reply_preview": reply[:320],
            },
        )

    return results, total_score, max_score


def _evaluate_workflow(client, base_url: Optional[str]) -> Tuple[List[Dict[str, Any]], int, int]:
    results: List[Dict[str, Any]] = []
    total_score = 0
    max_score = len(WORKFLOW_CASES) * 9

    for case in WORKFLOW_CASES:
        score = 0
        notes: List[str] = []
        history: List[WorkflowTurn] = []

        first_payload = {"message": case.first_message, "history": []}
        first = _post_chat_http(base_url, first_payload) if base_url else _post_chat_local(client, first_payload)
        history.extend(
            [
                WorkflowTurn(role="user", content=case.first_message),
                WorkflowTurn(role="assistant", content=str(first.get("reply", ""))),
            ]
        )
        first_workflow = first.get("case_workflow") or {}
        first_stage = int(first_workflow.get("stage", 0) or 0)
        if first_stage == 2 and not bool(first_workflow.get("report_generated")):
            score += 2
        else:
            notes.append("first workflow step did not remain in intake stage")

        followup_payload = {
            "message": case.followup_message,
            "history": [{"role": item.role, "content": item.content} for item in history],
        }
        second = _post_chat_http(base_url, followup_payload) if base_url else _post_chat_local(client, followup_payload)
        workflow = second.get("case_workflow") or {}
        reply = str(second.get("reply", ""))
        facts = workflow.get("collected_facts") or {}

        if workflow.get("legal_category") == case.expected_category:
            score += 1
        else:
            notes.append("workflow category incorrect")

        if int(workflow.get("stage", 0) or 0) == 4 and bool(workflow.get("report_generated")):
            score += 2
        else:
            notes.append("follow-up did not generate final report")

        if all(heading in reply for heading in case.required_report_headings):
            score += 1
        else:
            notes.append("report missing required headings")

        missing_fact_keys = [key for key in case.expected_fact_keys if key not in facts]
        if not missing_fact_keys:
            score += 1
        else:
            notes.append(f"expected facts missing: {', '.join(missing_fact_keys)}")

        if not STATUTE_REPLY_RE.search(reply):
            score += 1
        else:
            notes.append("report includes exact statutory references")

        if "Once I have that, I will move straight to the report." not in reply:
            score += 1
        else:
            notes.append("final report still contains intake copy")

        total_score += _record_result(
            results,
            case.name,
            score,
            9,
            notes,
            {
                "first_stage": first_stage,
                "final_stage": workflow.get("stage"),
                "report_generated": workflow.get("report_generated"),
                "legal_category": workflow.get("legal_category"),
                "facts": facts,
                "reply_preview": reply[:420],
            },
        )

    return results, total_score, max_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SaulGPT retrieval, chat quality, and workflow behavior.")
    parser.add_argument("--base-url", default="", help="Optional running backend URL. Defaults to in-process TestClient.")
    parser.add_argument("--out", default="eval_report.json", help="Report output path")
    args = parser.parse_args()

    client = None
    base_url = args.base_url.strip() or None
    if base_url is None:
        client = _load_local_client()

    retrieval_results, retrieval_score, retrieval_max = _evaluate_retrieval()
    chat_results, chat_score, chat_max = _evaluate_chat(client, base_url)
    workflow_results, workflow_score, workflow_max = _evaluate_workflow(client, base_url)

    total_score = retrieval_score + chat_score + workflow_score
    max_score = retrieval_max + chat_max + workflow_max
    percent = round((total_score / max_score) * 100, 2) if max_score else 0.0

    report = {
        "mode": "http" if base_url else "local_testclient",
        "base_url": base_url or "local",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_score": total_score,
            "max_score": max_score,
            "percent": percent,
        },
        "retrieval": {
            "score": retrieval_score,
            "max_score": retrieval_max,
            "cases": retrieval_results,
        },
        "chat": {
            "score": chat_score,
            "max_score": chat_max,
            "cases": chat_results,
        },
        "workflow": {
            "score": workflow_score,
            "max_score": workflow_max,
            "cases": workflow_results,
        },
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Evaluation complete: {total_score}/{max_score} ({percent}%)")
    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
