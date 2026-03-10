import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple

import requests


@dataclass
class EvalCase:
    prompt: str
    expect_legal: bool = True


TEST_CASES: List[EvalCase] = [
    EvalCase("Explain the difference between cheating and criminal breach of trust under Indian criminal law."),
    EvalCase("What details should be included in an FIR-style complaint for document forgery?"),
    EvalCase("My partner diverted client funds. Which sections may apply and what should I do first?"),
    EvalCase("How should I preserve WhatsApp and email evidence for a defamation/intimidation complaint?"),
    EvalCase("Give me a lawyer-style report structure for a property fraud matter."),
    EvalCase("Can I file both criminal complaint and civil recovery for the same fraudulent transaction?"),
    EvalCase("What documents should I attach with a police complaint for cheating?"),
    EvalCase("Draft strategy: accused took INR 8 lakh and stopped responding."),
    EvalCase("What immediate legal steps should I take after receiving threats from the accused?"),
    EvalCase("Tell me a joke about cricket.", expect_legal=False),
]


def score_reply(reply: str, citations_count: int, expect_legal: bool) -> Tuple[int, List[str]]:
    score = 0
    notes: List[str] = []
    lower = reply.lower()

    if expect_legal:
        if len(reply.strip()) >= 80:
            score += 1
        else:
            notes.append("too short")

        headings = ["short answer", "applicable law", "practical next steps", "missing facts"]
        heading_hits = sum(1 for h in headings if h in lower)
        if heading_hits >= 2:
            score += 1
        else:
            notes.append("weak structure")

        legal_terms = ["section", "ipc", "fir", "complaint", "legal", "evidence", "court", "offence", "offense"]
        term_hits = sum(1 for t in legal_terms if t in lower)
        if term_hits >= 2 or citations_count > 0:
            score += 1
        else:
            notes.append("low legal grounding")

        step_terms = ["next", "step", "file", "attach", "prepare", "request", "collect"]
        if any(t in lower for t in step_terms):
            score += 1
        else:
            notes.append("no actionable steps")

        if "outside my legal scope" not in lower:
            score += 1
        else:
            notes.append("incorrect diversion")

        return score, notes

    # Non-legal case scoring: reward graceful diversion and legal redirection.
    if "outside my legal scope" in lower:
        score += 2
    else:
        notes.append("missed diversion")

    if "legal" in lower and ("you can ask" in lower or "reframe" in lower):
        score += 2
    else:
        notes.append("missing legal redirection guidance")

    if len(reply.strip()) >= 40:
        score += 1
    else:
        notes.append("too short")

    return score, notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SaulGPT legal chat quality")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="SaulGPT backend base URL")
    parser.add_argument("--out", default="eval_report.json", help="Report output path")
    args = parser.parse_args()

    report = {
        "base_url": args.base_url,
        "cases": [],
        "total_score": 0,
        "max_score": len(TEST_CASES) * 5,
    }

    for case in TEST_CASES:
        payload = {"message": case.prompt, "history": []}
        try:
            response = requests.post(f"{args.base_url}/chat", json=payload, timeout=120)
            data = response.json()
        except Exception as exc:
            report["cases"].append({
                "prompt": case.prompt,
                "status": "failed",
                "error": str(exc),
                "score": 0,
            })
            continue

        if not response.ok:
            report["cases"].append({
                "prompt": case.prompt,
                "status": "failed",
                "error": data.get("detail", f"HTTP {response.status_code}"),
                "score": 0,
            })
            continue

        reply = data.get("reply", "")
        citations = data.get("citations", [])
        score, notes = score_reply(reply, len(citations), case.expect_legal)

        report["cases"].append({
            "prompt": case.prompt,
            "status": "ok",
            "score": score,
            "max": 5,
            "notes": notes,
            "citations": len(citations),
            "reply_preview": reply[:280],
        })
        report["total_score"] += score

    report["percent"] = round((report["total_score"] / report["max_score"]) * 100, 2) if report["max_score"] else 0

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Evaluation complete: {report['total_score']}/{report['max_score']} ({report['percent']}%)")
    print(f"Saved report: {args.out}")


if __name__ == "__main__":
    main()
