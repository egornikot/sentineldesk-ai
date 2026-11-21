"""Tune the SentinelDesk+ layer on DEV only (the base classifiers and index are reused, test is never read).

    py plus/train_plus.py                                  # standard split  -> plus/models/plus.json
    py plus/train_plus.py --models plus/honest/models --split plus/honest/processed --out plus/honest/models/plus.json

For every head: alpha = how much to trust the twin vote vs the classifier, then a temperature for calibration.
Shared k (neighbours) and tau (vote sharpness) are chosen by the summed dev macro-F1 of type/queue/priority.
Then the gate thresholds are re-tuned for the new probabilities (same targets as the original).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PLUS = Path(__file__).resolve().parent
ROOT = PLUS.parent
sys.path[:0] = [str(ROOT), str(PLUS)]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score  # noqa: E402

from sentineldesk.classify import Classifier, HEADS  # noqa: E402
from sentineldesk.config import Settings  # noqa: E402
from sentineldesk.retrieve import Retriever  # noqa: E402
from sentineldesk_plus import core  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--split", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--out", default=str(PLUS / "models" / "plus.json"))
    ap.add_argument("--target-recall", type=float, default=0.92,
                    help="dev recall target; 2 points above the 90%% goal as a safety margin (Helios uses 95%%)")
    a = ap.parse_args()
    s = Settings()
    clf, retr = Classifier.load(Path(a.models)), Retriever.load(Path(a.models), s.embedding_model)
    dev = pd.read_parquet(Path(a.split) / "dev.parquet")
    t0 = time.time()
    raw = core.embed_and_classify(dev, clf, retr)
    idx, sims = core.topk(raw["E"], retr.vectors, 30)
    print(f"dev {len(dev)}: classified + neighbours in {time.time() - t0:.0f}s")

    ys = {h: (dev[h].to_numpy(dtype=object) if h != "critical" else dev["critical"].to_numpy(dtype=int)) for h in core.ALL_HEADS}
    base = {h: round(f1_score(ys[h], np.array(raw[h][1], dtype=object)[raw[h][0].argmax(1)], average="macro"), 4) for h in HEADS}
    print("classifier only, dev macro-F1:", base)

    best = None
    for k in (5, 10, 20, 30):
        for tau in (0.005, 0.01, 0.02, 0.05):
            heads, total = {}, 0.0
            for h in core.ALL_HEADS:
                P, classes = raw[h]
                K = core.knn_proba(idx, sims, core.neighbor_labels(retr.meta, h), classes, k, tau)
                scores = []
                for alpha in np.round(np.arange(0, 1.01, 0.1), 1):
                    Q = core.blend(P, K, float(alpha))
                    if h == "critical":
                        sc = average_precision_score(ys[h], Q[:, classes.index(1)])
                    else:
                        sc = f1_score(ys[h], np.array(classes, dtype=object)[Q.argmax(1)], average="macro")
                    scores.append((sc, float(alpha)))
                sc, alpha = max(scores)
                heads[h] = {"alpha": alpha, "dev_score": round(float(sc), 4)}
                if h != "critical":
                    total += sc
            if best is None or total > best[0]:
                best = (total, k, tau, heads)
            print(f"  k={k:<3} tau={tau:<6} sum F1 {total:.4f}  " + "  ".join(f"{h}:a={v['alpha']}" for h, v in heads.items()))
    _, k, tau, heads = best
    cfg = {"k": k, "tau": tau, "heads": heads}
    for h in HEADS:  # temperature per head on the blended probabilities
        P, classes = raw[h]
        K = core.knn_proba(idx, sims, core.neighbor_labels(retr.meta, h), classes, k, tau)
        Q = core.blend(P, K, heads[h]["alpha"])
        y_idx = np.array([classes.index(v) for v in ys[h]])
        heads[h]["T"] = core.fit_temperature(Q, y_idx)
    heads["critical"]["T"] = 1.0
    probs = core.plus_probs(raw, idx, sims, retr.meta, cfg)
    top1 = sims[:, 0]
    cfg["thresholds"] = core.tune_thresholds(dev, probs, top1, retr.meta, idx, target_recall=a.target_recall)
    cfg["dev_classifier_only_f1"] = base
    cfg["dev_metrics"] = core.evaluate(dev, probs, cfg["thresholds"], top1, idx, retr.meta)
    core.save_json(Path(a.out), cfg)
    print(f"\nchosen k={k} tau={tau}: " + ", ".join(f"{h} alpha={v['alpha']} T={v['T']}" for h, v in heads.items()))
    print("thresholds:", cfg["thresholds"])
    print("dev:", {kk: v for kk, v in cfg["dev_metrics"].items() if kk.startswith(("f1_", "incident", "other", "ece_type"))})
    print("wrote", a.out)


if __name__ == "__main__":
    main()
