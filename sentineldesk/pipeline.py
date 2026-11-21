"""Runs every stage for one ticket and records per-stage latency.   OWNER: C (gate + dashboard)

    pipe = Pipeline.from_settings(Settings())
    result = pipe.run(subject, body)        # -> the interface contract in docs/PLAN.md
"""
from __future__ import annotations

import re
import time
from typing import Any, Optional

from . import gate, guard, verify
from .classify import Classifier
from .config import MODELS_DIR, Settings
from .data import ticket_text
from .generate import draft_reply
from .llm import make_llm
from .retrieve import Retriever

GERMAN_HINTS = re.compile(r"(?i)\b(und|nicht|ist|ich|wir|sie|der|die|das|bitte|danke|können|funktioniert|mit|für|auf)\b")


def detect_language(text: str) -> str:
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        code = detect(text)
        return code if code in ("en", "de") else "en"
    except Exception:
        words = max(1, len(text.split()))
        return "de" if len(GERMAN_HINTS.findall(text)) / words > 0.08 else "en"


def _walk(est):
    """Yield every estimator nested in Pipelines / FeatureUnions."""
    yield est
    for _, sub in getattr(est, "steps", []) or []:
        yield from _walk(sub)
    for _, sub, *_ in getattr(est, "transformer_list", []) or []:
        yield from _walk(sub)


class Pipeline:
    def __init__(self, settings: Settings, classifier: Classifier, retriever: Retriever, llm: Any) -> None:
        self.settings, self.classifier, self.retriever, self.llm = settings, classifier, retriever, llm

    @classmethod
    def from_settings(cls, settings: Settings, warm: bool = True) -> "Pipeline":
        pipe = cls(settings, Classifier.load(MODELS_DIR), Retriever.load(MODELS_DIR, settings.embedding_model),
                   make_llm(settings))
        if warm:
            pipe.warm_up()
        return pipe

    def warm_up(self) -> None:
        """Load the embedding model once, share it between classifier and retriever, run one dummy ticket."""
        from .features import Embedder
        shared = None
        for model in self.classifier.models.values():
            for step in _walk(model):
                if isinstance(step, Embedder):
                    if shared is None:
                        shared = step._model()
                    step._st = shared
        if shared is not None and self.retriever.ready:
            self.retriever._encoder = shared
        if self.classifier.models:
            self.classifier.predict("warm up", "warm up")
        if self.retriever.ready:
            self.retriever.search("warm up", "warm up", 1)

    def status(self) -> dict:
        return {"classifier": self.classifier.engine, "retriever": "ready" if self.retriever.ready else "no index",
                "llm": self.llm.name, "model": getattr(self.llm, "model", ""), "thresholds": self.settings.thresholds}

    def run(self, subject: Optional[str], body: str) -> dict[str, Any]:
        t: dict[str, float] = {}
        start = time.perf_counter()

        def lap(name: str, t0: float) -> None:
            t[name] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        subj, bod = guard.redact(subject or ""), guard.redact(body or "")
        language = detect_language(ticket_text(subject, body))
        ticket = {"subject": subj.text, "body": bod.text, "language": language}
        lap("preprocess", t0)

        t0 = time.perf_counter()
        classification = self.classifier.predict(subj.text, bod.text)
        lap("classify", t0)

        t0 = time.perf_counter()
        retrieval = self.retriever.search(subj.text, bod.text, self.settings.top_k)
        lap("retrieve", t0)

        t0 = time.perf_counter()
        decision = gate.decide(ticket_text(subj.text, bod.text), classification, retrieval, self.settings.thresholds)
        lap("gate", t0)

        draft, verification = None, None
        if retrieval:
            t0 = time.perf_counter()
            draft = draft_reply(self.llm, ticket, retrieval, decision)
            lap("generate", t0)
            t0 = time.perf_counter()
            verification = verify.verify(draft, retrieval)
            draft["claims"] = verification["claims"]
            decision = gate.after_verification(decision, verification)
            lap("verify", t0)

        t["total"] = round((time.perf_counter() - start) * 1000, 1)
        return {
            "ticket": ticket,
            "pii": {"masked": subj.changed or bod.changed, "found": sorted(set(subj.kinds + bod.kinds))},
            "classification": classification,
            "retrieval": retrieval,
            "decision": decision,
            "draft": draft,
            "verification": verification and {k: verification[k] for k in ("ok", "problems", "grounding_rate")},
            "timings_ms": t,
            "engine": self.status(),
        }
