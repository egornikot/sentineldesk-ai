"""Twin vote + calibration + shared evaluation helpers for SentinelDesk+."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentineldesk import gate, rules
from sentineldesk.classify import HEADS

ALL_HEADS = HEADS + ("critical",)


# ---------------------------------------------------------------------------------------------- twin vote
def topk(E: np.ndarray, V: np.ndarray, k: int, chunk: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Indices and similarities of the k nearest index rows for every query row (chunked)."""
    idx = np.zeros((len(E), k), dtype=np.int64)
    sims = np.zeros((len(E), k), dtype=np.float32)
    for s in range(0, len(E), chunk):
        S = E[s:s + chunk] @ V.T
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, 1)
        order = np.argsort(-ps, axis=1)
        idx[s:s + chunk] = np.take_along_axis(part, order, 1)
        sims[s:s + chunk] = np.take_along_axis(ps, order, 1)
    return idx, sims


def neighbor_labels(meta: pd.DataFrame, head: str) -> np.ndarray:
    if head == "critical":
        return ((meta["type"] == "Incident") & (meta["priority"] == "high")).astype(int).to_numpy()
    return meta[head].to_numpy(dtype=object)


def knn_proba(idx: np.ndarray, sims: np.ndarray, labels: np.ndarray, classes: list, k: int, tau: float) -> np.ndarray:
    """Similarity-weighted vote of the k nearest train tickets, lightly smoothed."""
    idx, sims = idx[:, :k], sims[:, :k]
    w = np.exp((sims - sims[:, :1]) / tau)
    lab = labels[idx]
    P = np.stack([(w * (lab == c)).sum(1) for c in classes], 1) / w.sum(1, keepdims=True)
    return (P + 0.01) / (1 + 0.01 * len(classes))


def blend(p_clf: np.ndarray, p_knn: np.ndarray, alpha: float, T: float = 1.0) -> np.ndarray:
    p = (1 - alpha) * p_clf + alpha * p_knn
    if T != 1.0:
        z = np.log(np.clip(p, 1e-9, 1)) / T
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
    return p


def fit_temperature(P: np.ndarray, y_idx: np.ndarray) -> float:
    best, best_nll = 1.0, np.inf
    for T in np.round(np.arange(0.3, 3.01, 0.05), 2):
        Q = blend(P, P, 0.0, float(T))
        nll = -np.mean(np.log(np.clip(Q[np.arange(len(Q)), y_idx], 1e-9, 1)))
        if nll < best_nll:
            best, best_nll = float(T), nll
    return best


# ----------------------------------------------------------------------------------------- predictions
def embed_and_classify(df: pd.DataFrame, clf, retr) -> dict[str, Any]:
    """Embeds texts once (shared with embedding-based classifier heads); returns raw classifier probabilities."""
    from sentineldesk.features import Embedder
    from sentineldesk.pipeline import _walk
    texts = df["text"].tolist()
    E = retr.encode(texts)
    cache = dict(zip(texts, E))
    for model in clf.models.values():
        for step in _walk(model):
            if isinstance(step, Embedder):
                step.cache = cache
    out = {"E": E}
    for head in ALL_HEADS:
        pipe = clf.models[head]
        out[head] = (pipe.predict_proba(texts), list(pipe.classes_))
    for model in clf.models.values():
        for step in _walk(model):
            if isinstance(step, Embedder):
                step.cache = None
    return out


def plus_probs(raw: dict, idx: np.ndarray, sims: np.ndarray, meta: pd.DataFrame, cfg: dict) -> dict:
    """Blended + temperature-scaled probabilities for every head, using a tuned cfg."""
    out = {}
    for head in ALL_HEADS:
        P, classes = raw[head]
        h = cfg["heads"][head]
        K = knn_proba(idx, sims, neighbor_labels(meta, head), classes, cfg["k"], cfg["tau"])
        out[head] = (blend(P, K, h["alpha"], h.get("T", 1.0)), classes)
    return out


# ---------------------------------------------------------------------------------------------- metrics
def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return round(float(total), 4)


def evaluate(df: pd.DataFrame, probs: dict, thresholds: dict, top1: np.ndarray,
             nn_idx: np.ndarray | None = None, meta: pd.DataFrame | None = None) -> dict:
    """All headline metrics for one model on one labelled split."""
    from sklearn.metrics import f1_score
    m: dict[str, Any] = {"n": int(len(df))}
    labels, confs = {}, {}
    for head in HEADS:
        P, classes = probs[head]
        lab = np.array(classes, dtype=object)[P.argmax(1)]
        conf = P.max(1)
        y = df[head].to_numpy(dtype=object)
        labels[head], confs[head] = lab, conf
        m[f"f1_{head}"] = round(float(f1_score(y, lab, average="macro")), 4)
        m[f"acc_{head}"] = round(float((y == lab).mean()), 4)
        m[f"ece_{head}"] = ece(conf, y == lab)
        blank = df["subject_blank"].to_numpy(dtype=bool)
        if blank.any() and (~blank).any():
            m[f"f1_{head}_null_subject"] = round(float(f1_score(y[blank], lab[blank], average="macro")), 4)
    Pc, cc = probs["critical"]
    crit = Pc[:, cc.index(1)]
    y_crit = df["critical"].to_numpy(dtype=int) == 1
    label_rule = ((labels["type"] == "Incident") & (labels["priority"] == "high")
                  & (confs["type"] >= gate.min_conf(thresholds, "type"))
                  & (confs["priority"] >= gate.min_conf(thresholds, "priority")))
    impact = np.array([bool(gate.impact_pages(rules.impact_hits(t))) for t in df["text"]])
    paged = label_rule | (crit >= thresholds["critical"]) | impact
    m["incident_recall"] = round(float(paged[y_crit].mean()), 4)
    m["other_paged"] = round(float(paged[~y_crit].mean()), 4)
    m["incident_precision"] = round(float(y_crit[paged].mean()), 4) if paged.any() else 0.0
    routes = []
    for i, text in enumerate(df["text"]):
        c = {h: {"label": labels[h][i], "confidence": float(confs[h][i])} for h in HEADS}
        c["critical_score"] = float(crit[i])
        routes.append(gate.decide(text, c, [{"similarity": float(top1[i])}], thresholds)["route"])
    rr = pd.Series(routes).value_counts(normalize=True)
    m["routes"] = {k: round(float(rr.get(k, 0.0)), 4) for k in ("page_now", "human_review", "draft_ready")}
    if nn_idx is not None and meta is not None:
        nl = meta["language"].to_numpy(dtype=object)[nn_idx[:, :5]]
        nq = meta["queue"].to_numpy(dtype=object)[nn_idx[:, :5]]
        ql = df["language"].to_numpy(dtype=object)[:, None]
        m["cross_lingual_top5_pct"] = round(float((nl != ql).any(1).mean() * 100), 2)
        m["same_queue_hit_at5_pct"] = round(float((nq == df["queue"].to_numpy(dtype=object)[:, None]).any(1).mean() * 100), 2)
        m["top1_similarity_median"] = round(float(np.median(top1)), 4)
    return m


def tune_thresholds(dev: pd.DataFrame, probs: dict, top1: np.ndarray, meta: pd.DataFrame, nn_idx: np.ndarray,
                    target_acc: float = 0.75, target_recall: float = 0.90, sim_agreement: float = 0.85) -> dict:
    th: dict[str, float] = {}
    labels, confs = {}, {}
    for head in HEADS:
        P, classes = probs[head]
        lab, conf = np.array(classes, dtype=object)[P.argmax(1)], P.max(1)
        labels[head], confs[head] = lab, conf
        correct = lab == dev[head].to_numpy(dtype=object)
        cut = 0.9
        for c in np.arange(0.30, 0.91, 0.01):
            mask = conf >= c
            if mask.sum() >= 50 and correct[mask].mean() >= target_acc:
                cut = float(c)
                break
        th[f"min_confidence_{head}"] = round(cut, 2)
    th["min_confidence"] = 0.55
    Pc, cc = probs["critical"]
    crit = Pc[:, cc.index(1)]
    y = dev["critical"].to_numpy(dtype=int) == 1
    label_rule = ((labels["type"] == "Incident") & (labels["priority"] == "high")
                  & (confs["type"] >= th["min_confidence_type"]) & (confs["priority"] >= th["min_confidence_priority"]))
    impact = np.array([bool(gate.impact_pages(rules.impact_hits(t))) for t in dev["text"]])
    chosen = 0.05
    for t in np.arange(0.95, 0.04, -0.01):
        if (label_rule | (crit >= t) | impact)[y].mean() >= target_recall:
            chosen = float(t)
            break
    th["critical"] = round(chosen, 2)
    # akshita: the weakest precedent we trust = smallest similarity where the nearest train ticket has the
    # dev ticket's true queue >= 85% of the time
    queue_ok = meta["queue"].to_numpy(dtype=object)[nn_idx[:, 0]] == dev["queue"].to_numpy(dtype=object)
    sim = float(np.percentile(top1, 10))
    for c in np.round(np.arange(0.50, 0.99, 0.01), 2):
        mask = top1 >= c
        if mask.mean() < 0.10:
            break
        if queue_ok[mask].mean() >= sim_agreement:
            sim = float(c)
            break
    th["min_similarity"] = round(sim, 3)
    return th


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
