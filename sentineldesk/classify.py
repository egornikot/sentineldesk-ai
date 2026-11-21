"""Type / queue / priority classifiers + critical-incident detector.   OWNER: A (data + models)

Contract:
    Classifier.load(models_dir) -> Classifier   (works even if no models are trained yet)
    clf.predict(subject, body)  -> {
        "type":     {"label": str | None, "confidence": float},
        "queue":    {"label": str | None, "confidence": float},
        "priority": {"label": str | None, "confidence": float},
        "critical_score": float,          # P(high-priority Incident), 0..1
        "engine": "trained" | "fallback",
    }

scripts/train.py saves models/classifier.joblib = {"type": pipe, "queue": pipe, "priority": pipe, "critical": pipe},
each an sklearn Pipeline (e.g. TfidfVectorizer + calibrated LogisticRegression) taking ticket_text().

Until models exist, a fallback returns no labels, zero confidence and a rule-based critical
score, so the gate escalates (low confidence) and the dashboard still runs end to end.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from . import rules
from .data import ticket_text

HEADS = ("type", "queue", "priority")


class Classifier:
    def __init__(self, models: Optional[dict] = None) -> None:
        self.models = models or {}

    @classmethod
    def load(cls, models_dir: Path) -> "Classifier":
        path = Path(models_dir) / "classifier.joblib"
        if not path.is_file():
            return cls()
        import joblib
        return cls(joblib.load(path))

    @property
    def engine(self) -> str:
        return "trained" if self.models else "fallback"

    def predict(self, subject: Optional[str], body: str) -> dict[str, Any]:
        text = ticket_text(subject, body)
        if not self.models:
            hits = rules.impact_hits(text)
            score = min(1.0, 0.25 * len([h for h in hits if h != "urgency"]) + (0.15 if "urgency" in hits else 0.0))
            out = {h: {"label": None, "confidence": 0.0} for h in HEADS}
            out.update(critical_score=round(score, 3), engine="fallback")
            return out
        out: dict[str, Any] = {}
        for head in HEADS:
            pipe = self.models[head]
            proba = pipe.predict_proba([text])[0]
            i = int(proba.argmax())
            out[head] = {"label": str(pipe.classes_[i]), "confidence": round(float(proba[i]), 3)}
        crit = self.models.get("critical")
        if crit is not None:
            classes = list(crit.classes_)
            out["critical_score"] = round(float(crit.predict_proba([text])[0][classes.index(1)]), 3)
        else:
            out["critical_score"] = 0.0
        out["engine"] = "trained"
        return out
