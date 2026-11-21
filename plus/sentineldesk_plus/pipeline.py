"""SentinelDesk+ pipeline: the original stages plus twin vote, burst radar, placeholder cleanup and PII view."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from sentineldesk import gate, guard, verify
from sentineldesk.classify import HEADS, Classifier
from sentineldesk.config import Settings
from sentineldesk.data import ticket_text
from sentineldesk.generate import draft_reply
from sentineldesk.llm import make_llm
from sentineldesk.pipeline import Pipeline, detect_language
from sentineldesk.retrieve import Retriever

from . import core
from .answers import clean_answer, raw_placeholders
from .burst import BurstRadar

BURST_DEMO = [
    ("VPN down for all staff", "Since 9am nobody in our office can connect to the VPN. The client times out for every user.", "en"),
    ("VPN outage at our clinic", "The VPN stopped working for the whole clinic this morning, staff cannot reach internal systems.", "en"),
    ("VPN-Ausfall im Büro", "Seit heute Morgen kann sich niemand im Büro mit dem VPN verbinden, alle Verbindungen brechen ab.", "de"),
]


class PlusPipeline(Pipeline):
    def __init__(self, settings: Settings, classifier: Classifier, retriever: Retriever, llm: Any, cfg: dict) -> None:
        super().__init__(settings, classifier, retriever, llm)
        self.cfg = cfg
        self.thresholds = {**settings.thresholds, **cfg["thresholds"]}
        self.radar = BurstRadar()
        self.labels = {h: core.neighbor_labels(retriever.meta, h) for h in core.ALL_HEADS} if retriever.ready else {}

    @classmethod
    def build(cls, settings: Settings, models_dir: Path, cfg_path: Path, warm: bool = True) -> "PlusPipeline":
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        pipe = cls(settings, Classifier.load(models_dir), Retriever.load(models_dir, settings.embedding_model),
                   make_llm(settings), cfg)
        if warm:
            pipe.warm_up()
        return pipe

    def status(self) -> dict:
        return {**super().status(), "version": "plus", "thresholds": self.thresholds,
                "twin_vote": {"k": self.cfg["k"], "alpha": {h: self.cfg["heads"][h]["alpha"] for h in core.ALL_HEADS}}}

    def _classify(self, text: str, q: np.ndarray, sims: np.ndarray) -> dict:
        k = self.cfg["k"]
        top = np.argsort(-sims)[:k]
        idx, s = top[None, :], sims[top][None, :]
        out: dict[str, Any] = {}
        for head in core.ALL_HEADS:
            pipe = self.classifier.models[head]
            classes = list(pipe.classes_)
            p_clf = pipe.predict_proba([text])
            p_knn = core.knn_proba(idx, s, self.labels[head], classes, k, self.cfg["tau"])
            h = self.cfg["heads"][head]
            p = core.blend(p_clf, p_knn, h["alpha"], h.get("T", 1.0))[0]
            if head == "critical":
                out["critical_score"] = round(float(p[classes.index(1)]), 3)
                continue
            i, ic, ik = int(p.argmax()), int(p_clf[0].argmax()), int(p_knn[0].argmax())
            out[head] = {"label": str(classes[i]), "confidence": round(float(p[i]), 3),
                         "classifier": {"label": str(classes[ic]), "confidence": round(float(p_clf[0][ic]), 3)},
                         "twins": {"label": str(classes[ik]), "share": round(float(p_knn[0][ik]), 3), "k": k},
                         "alpha": h["alpha"]}
        out["engine"] = "trained+twins"
        return out

    def classify_text(self, subject: str, body: str) -> tuple[dict, np.ndarray, np.ndarray]:
        text = ticket_text(subject, body)
        q = self.retriever.encode([text])[0]
        sims = self.retriever.vectors @ q
        return self._classify(text, q, sims), q, sims

    def simulate_burst(self) -> None:
        for subject, body, lang in BURST_DEMO:
            c, q, _ = self.classify_text(subject, body)
            self.radar.observe(q, subject, body, lang, incident_like=True)

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
        classification, q, sims = self.classify_text(subj.text, bod.text)
        lap("classify", t0)

        t0 = time.perf_counter()
        retrieval = []
        for i in np.argsort(-sims)[: self.settings.top_k]:
            rec = self.retriever.meta.iloc[int(i)].to_dict()
            rec = {key: (None if v != v else v) for key, v in rec.items()}
            rec["answer"] = clean_answer(rec.get("answer"))
            rec["similarity"] = round(float(sims[i]), 3)
            retrieval.append(rec)
        lap("retrieve", t0)

        t0 = time.perf_counter()
        incident_like = classification["type"]["label"] in ("Incident", "Problem")
        burst = self.radar.observe(q, subj.text, bod.text, language, incident_like)
        decision = gate.decide(ticket_text(subj.text, bod.text), classification, retrieval, self.thresholds)
        if burst:
            reason = (f"outage burst {burst['incident_id']}: {burst['count']} similar incident reports in the last "
                      f"{self.radar.window_s // 60} min ({', '.join(burst['languages']).upper()})")
            decision["reasons"] = [reason] + decision["reasons"]
            if decision["route"] != "page_now":
                decision.update(route="page_now", badge=gate.BADGES["page_now"])
        lap("gate", t0)

        draft, verification = None, None
        if retrieval:
            t0 = time.perf_counter()
            draft = draft_reply(self.llm, ticket, retrieval, decision)
            lap("generate", t0)
            t0 = time.perf_counter()
            verification = verify.verify(draft, retrieval)
            leftover = raw_placeholders(draft.get("reply")) + raw_placeholders(draft.get("customer_ack"))
            if leftover:
                verification["problems"].append(f"draft contains raw placeholders: {', '.join(sorted(set(leftover)))}")
                verification["ok"] = False
            draft["claims"] = verification["claims"]
            decision = gate.after_verification(decision, verification)
            lap("verify", t0)

        t["total"] = round((time.perf_counter() - start) * 1000, 1)
        return {
            "ticket": ticket,
            "pii": {"masked": subj.changed or bod.changed, "found": sorted(set(subj.kinds + bod.kinds)),
                    "sent_to_llm": {"subject": subj.text, "body": bod.text}},
            "classification": classification,
            "retrieval": retrieval,
            "decision": decision,
            "burst": burst,
            "draft": draft,
            "verification": verification and {k: verification[k] for k in ("ok", "problems", "grounding_rate")},
            "timings_ms": t,
            "engine": self.status(),
        }
