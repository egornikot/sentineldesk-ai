"""Normalise the dataset's anonymisation placeholders in historical answers (idea from the akshita branch).

The answers contain "<tel_num>", "<Name>", "<Tel-Nummer>", "<acc_num>" ... in ~15 spellings. Precedents are
shown and given to the LLM with one canonical token each, and a raw "<...>" left in a draft is flagged.
"""
from __future__ import annotations

import re

PLACEHOLDER_RE = re.compile(r"<([^<>\n]{1,30})>")
TOKENS = {"name": "[name]", "phone": "[phone number]", "account": "[account number]", "email": "[email]",
          "amount": "[amount]", "link": "[link]", "other": "[detail]"}


def _token(inner: str) -> str:
    s = inner.lower().replace("-", "_").replace(" ", "_")
    if s in ("br", "br/", "br_/"):
        return "\n"
    if "name" in s:
        return TOKENS["name"]
    if "tel" in s or "phone" in s or "nummer" in s and "konto" not in s:
        return TOKENS["phone"]
    if "acc" in s or "konto" in s or "account" in s:
        return TOKENS["account"]
    if "mail" in s:
        return TOKENS["email"]
    if "amount" in s or "betrag" in s or "price" in s:
        return TOKENS["amount"]
    if "url" in s or "link" in s:
        return TOKENS["link"]
    return TOKENS["other"]


def clean_answer(text: str | None) -> str:
    return PLACEHOLDER_RE.sub(lambda m: _token(m.group(1)), text or "").strip()


def raw_placeholders(text: str | None) -> list[str]:
    return [m.group(0) for m in PLACEHOLDER_RE.finditer(text or "")]
