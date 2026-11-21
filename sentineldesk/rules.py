"""Deterministic rules that run outside any model.

1. IMPACT rules: phrases that signal a critical incident (clinical impact, full outage,
   security breach, data loss). Negated mentions ("no data was lost") are ignored.
2. PROMISE rules: requests for refunds, amounts, guarantees, SLAs or legal commitments.
   The AI must not grant these unless the retrieved evidence explicitly supports them.
Both lists cover English and German.
"""
from __future__ import annotations

import re

NEGATION = re.compile(
    r"(?i)\b(no|not|never|without|isn'?t|wasn'?t|aren'?t|don'?t|doesn'?t|kein\w*|nicht|nie|ohne)\b[\w\s,']{0,20}$"
)

IMPACT = [
    ("clinical", re.compile(
        r"(?i)\b(patient\w*|hospital\w*|clinic\w*|clinician\w*|doctor\w*|ärzt\w*|arzt|krankenhaus\w*|klinik\w*|"
        r"emr|ehr|telemedicine|telemedizin|medical records?|patientenakte\w*)\b")),
    ("outage", re.compile(
        r"(?i)\b(outage|down for (?:all|every)\w*|(?:is|are|went|gone) down|not working for (?:all|any|every)\w*|"
        r"nobody can|no one can|all users|entire (?:company|team|office|site)|company-wide|site-wide|"
        r"ausfall|komplett ausgefallen|niemand kann|alle (?:nutzer|mitarbeiter|benutzer))\b")),
    ("security", re.compile(
        r"(?i)\b(breach\w*|hack(?:ed|er)?|ransomware|malware|phishing|unauthori[sz]ed access|compromised|"
        r"data leak\w*|sicherheitsvorfall|angriff|gehackt|datenleck)\b")),
    ("data_loss", re.compile(
        r"(?i)\b(data loss|lost (?:all )?(?:our |the )?data|deleted (?:all|our)|corrupt\w*|datenverlust|daten verloren)\b")),
    ("urgency", re.compile(
        r"(?i)\b(urgent\w*|immediate\w*|asap|emergency|critical|dringend\w*|sofort\w*|notfall|kritisch\w*)\b")),
]

PROMISE = [
    ("refund", re.compile(r"(?i)\b(refund\w*|money back|reimburs\w*|chargeback|rückerstattung\w*|erstatt\w*|geld zurück)\b")),
    ("amount", re.compile(r"(?i)(?:[$€£]\s?\d[\d.,]*|\b\d[\d.,]*\s?(?:usd|eur|euro|dollars?|€|\$))")),
    ("guarantee", re.compile(r"(?i)\b(guarantee\w*|promise\w*|assure\w*|garantie\w*|versprech\w*|zusicher\w*)\b")),
    ("sla", re.compile(r"(?i)\b(sla|service level|uptime commitment|compensation|entschädigung|kompensation)\b")),
    ("legal", re.compile(r"(?i)\b(lawsuit|legal action|lawyer|attorney|gdpr|dsgvo|anwalt|klage|rechtlich\w*)\b")),
]


def _hits(rules: list, text: str) -> list[str]:
    found: list[str] = []
    for name, pattern in rules:
        for m in pattern.finditer(text or ""):
            if NEGATION.search(text[max(0, m.start() - 30):m.start()]):
                continue
            found.append(name)
            break
    return found


def impact_hits(text: str) -> list[str]:
    """Names of impact rules that fire on the ticket text, e.g. ["clinical", "outage"]."""
    return _hits(IMPACT, text)


def promise_hits(text: str) -> list[str]:
    """Names of promise rules that fire, e.g. ["refund", "amount", "guarantee"]."""
    return _hits(PROMISE, text)
