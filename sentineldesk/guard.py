"""Mask personal data and credentials before ticket text reaches the LLM (bonus: PII guardrail).

Covers SSNs, Luhn-checked cards, credentials, API keys and bank numbers, plus what support
tickets contain most: emails, phone numbers, IP addresses, IBANs.
Technical content (product names, error codes, hostnames) is kept on purpose.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
IPV6 = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,3})?\b")
PHONE = re.compile(r"(?<![\w.])(?:\+|00)?\d{1,3}?[\s./-]?\(?\d{2,5}\)?(?:[\s./-]?\d{2,5}){2,4}(?![\w.])")
SSN = re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
CREDENTIAL = re.compile(
    r"(?i)\b(password|passwort|passcode|pwd|pin|verification code|security code|one[- ]time code|cvv|cvc)"
    r"\b(\s*(?:is|was|ist|=|:)\s*|\s+)([^\s,;]{3,})"
)
API_KEY = re.compile(
    r"\b(?:sk-ant-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}"
    r"|xox[abprs]-[A-Za-z0-9-]{10,})\b"
)
ACCOUNT = re.compile(r"(?i)\b(account|acct|konto|kundennummer|customer)\s*(?:number|no\.?|nr\.?|#)?\s*(?:is|:)?\s*(\d{6,17})\b")


def luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass
class Redaction:
    text: str
    kinds: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.kinds)


def redact(text: str) -> Redaction:
    if not isinstance(text, str):
        return Redaction("", ["unreadable"])
    kinds: list[str] = []
    value = text[:12000]

    def mark(kind: str) -> str:
        if kind not in kinds:
            kinds.append(kind)
        return f"[{kind.upper()}]"

    value = API_KEY.sub(lambda m: mark("api_key"), value)
    value = EMAIL.sub(lambda m: mark("email"), value)
    value = IBAN.sub(lambda m: mark("iban"), value)
    value = SSN.sub(lambda m: mark("ssn"), value)

    def card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return mark("card_number") if 13 <= len(digits) <= 19 and luhn_ok(digits) else m.group(0)

    value = CARD.sub(card, value)
    value = ACCOUNT.sub(lambda m: f"{m.group(1)} {mark('account_number')}", value)
    value = IPV6.sub(lambda m: mark("ip_address"), value)
    value = IPV4.sub(lambda m: mark("ip_address"), value)
    value = CREDENTIAL.sub(lambda m: f"{m.group(1)}{m.group(2)}{mark('credential')}", value)

    def phone(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return mark("phone") if 8 <= len(digits) <= 15 else m.group(0)

    value = PHONE.sub(phone, value)
    return Redaction(value, kinds)
