"""Escalation & safety gate: plain code, never the LLM.          OWNER: C (gate + dashboard)

Contract:
    decide(text, classification, retrieval, thresholds) -> {
        "route": "page_now" | "human_review" | "draft_ready",
        "badge": "Escalate to On-Call Engineer" | "Human Review" | "Auto-Draft Ready",
        "reasons": [str, ...],          # every rule that fired, human-readable
        "impact": [str, ...],           # impact rules that fired (clinical, outage, ...)
        "promises": [str, ...],         # promise rules that fired (refund, amount, ...)
    }
    after_verification(decision, verification) -> decision, demoted to human_review if a claim failed

Urgency and evidence are separate: PAGE NOW wins over everything; HUMAN REVIEW collects
low confidence, weak evidence and unsupported promises.
"""
from __future__ import annotations

from typing import Any

from . import rules

BADGES = {
    "page_now": "Escalate to On-Call Engineer",
    "human_review": "Human Review",
    "draft_ready": "Auto-Draft Ready",
}
SEVERE_IMPACT = {"clinical", "outage", "security", "data_loss"}


def impact_pages(impact: list[str]) -> list[str]:
    """Severe impact categories that page on their own (shared with scripts/train.py)."""
    severe = [i for i in impact if i in SEVERE_IMPACT]
    if severe and ("urgency" in impact or len(severe) >= 2 or "security" in severe):
        return severe
    return []


def min_conf(thresholds: dict, head: str) -> float:
    return float(thresholds.get(f"min_confidence_{head}", thresholds.get("min_confidence", 0.55)))


def label_pages(t: dict, p: dict, thresholds: dict) -> bool:
    """Page on predicted labels only when the classifier is confident in both; low-confidence
    guesses are left to the calibrated critical-incident detector and the impact rules."""
    return (t.get("label") == "Incident" and p.get("label") == "high"
            and t.get("confidence", 0) >= min_conf(thresholds, "type")
            and p.get("confidence", 0) >= min_conf(thresholds, "priority"))


def decide(text: str, classification: dict, retrieval: list[dict], thresholds: dict) -> dict[str, Any]:
    page, review = [], []
    impact = rules.impact_hits(text)
    promises = rules.promise_hits(text)

    t, p = classification.get("type", {}), classification.get("priority", {})
    if label_pages(t, p, thresholds):
        page.append(f"predicted Incident + high priority ({t['confidence']:.0%} / {p['confidence']:.0%})")
    crit = float(classification.get("critical_score") or 0.0)
    if crit >= thresholds["critical"]:
        page.append(f"critical-incident score {crit:.2f} ≥ {thresholds['critical']:.2f}")
    severe = impact_pages(impact)
    if severe:
        page.append("impact: " + ", ".join(severe + (["urgent wording"] if "urgency" in impact else [])))

    for head in ("type", "queue", "priority"):
        c = classification.get(head, {})
        if c.get("label") is None:
            review.append(f"no trained {head} model")
        elif c["confidence"] < min_conf(thresholds, head):
            review.append(f"low {head} confidence {c['confidence']:.0%} < {min_conf(thresholds, head):.0%}")
    top_sim = max((r.get("similarity") or 0.0 for r in retrieval), default=0.0)
    if not retrieval:
        review.append("no historical precedent retrieved")
    elif top_sim < thresholds["min_similarity"]:
        review.append(f"weak precedent: best similarity {top_sim:.2f} < {thresholds['min_similarity']:.2f}")
    if promises:
        review.append("asks for commitments the AI may not grant: " + ", ".join(promises))

    route = "page_now" if page else "human_review" if review else "draft_ready"
    return {"route": route, "badge": BADGES[route], "reasons": page + review or ["all checks passed"],
            "impact": impact, "promises": promises}


def after_verification(decision: dict, verification: dict) -> dict:
    if decision["route"] == "draft_ready" and not verification.get("ok", False):
        decision = {**decision, "route": "human_review", "badge": BADGES["human_review"],
                    "reasons": [r for r in decision["reasons"] if r != "all checks passed"]
                    + [f"draft failed verification: {p}" for p in verification.get("problems", [])]}
    return decision
