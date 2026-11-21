"""Check every drafted claim against the retrieved evidence.     OWNER: B (retrieval + LLM)

Contract:
    verify(draft, retrieval) -> {
        "ok": bool,
        "claims": [{..claim.., "verified": bool, "score": float, "why": str}],
        "problems": [str],
        "grounding_rate": float,          # verified claims / all claims (1.0 if no claims)
    }

A claim passes when its ticket_id is one of the retrieved tickets AND its excerpt appears in
that ticket's answer (fuzzy match ≥ MIN_MATCH, to allow whitespace/punctuation drift).
The reply is also scanned for promises (refund, amounts, guarantees) that no evidence contains.
"""
from __future__ import annotations

import re
from typing import Any

from . import rules

MIN_MATCH = 88.0


def _match(excerpt: str, answer: str) -> float:
    excerpt, answer = _norm(excerpt), _norm(answer)
    if not excerpt or not answer:
        return 0.0
    if excerpt in answer:
        return 100.0
    try:
        from rapidfuzz import fuzz
        return float(fuzz.partial_ratio(excerpt, answer))
    except ImportError:
        from difflib import SequenceMatcher
        m = SequenceMatcher(None, excerpt, answer).find_longest_match(0, len(excerpt), 0, len(answer))
        return 100.0 * m.size / len(excerpt)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def verify(draft: dict, retrieval: list[dict]) -> dict[str, Any]:
    answers = {r["ticket_id"]: r.get("answer") or "" for r in retrieval}
    checked, problems = [], []
    for claim in draft.get("claims") or []:
        tid = claim.get("ticket_id", "")
        if tid not in answers:
            checked.append({**claim, "verified": False, "score": 0.0, "why": "ticket ID was not retrieved"})
            problems.append(f"cites unknown ticket {tid}")
            continue
        score = _match(claim.get("excerpt", ""), answers[tid])
        ok = score >= MIN_MATCH
        checked.append({**claim, "verified": ok, "score": round(score, 1),
                        "why": "excerpt found in cited answer" if ok else "excerpt not found in cited answer"})
        if not ok:
            problems.append(f"excerpt not supported by {tid}")

    evidence = " ".join(answers.values())
    for promise in rules.promise_hits(draft.get("reply", "")):
        if promise not in rules.promise_hits(evidence):
            problems.append(f"reply mentions '{promise}' but no retrieved case does")
    if draft.get("error"):
        problems.append(f"drafting failed: {draft['error']}")
    if draft.get("unsupported_requests"):
        problems.append("customer asked for: " + "; ".join(draft["unsupported_requests"])[:200])

    n_ok = sum(1 for c in checked if c["verified"])
    return {"ok": not problems, "claims": checked, "problems": problems,
            "grounding_rate": round(n_ok / len(checked), 3) if checked else 1.0}
