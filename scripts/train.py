"""Train classifiers, tune gate thresholds, build the retrieval index.   OWNERS: A (models), B (index)

    py scripts/train.py

Reads ONLY data/processed/train.parquet (fit) and dev.parquet (model selection + thresholds).
Never touches test.

Model selection, per head (type, queue, priority, critical), on dev:
  tfidf_svm : word 1-2 grams + char 3-5 grams      -> calibrated LinearSVC
  emb_lr    : multilingual sentence embeddings      -> LogisticRegression
  combo_svm : tfidf + embeddings                    -> calibrated LinearSVC
  The candidate with the best dev macro-F1 (critical: best dev average precision) is kept.
  All probabilities are calibrated (CalibratedClassifierCV / LR), so "confidence" means something.

Writes:
  models/classifier.joblib    {"type","queue","priority","critical"} raw-text -> predict_proba pipelines
  models/thresholds.json      critical, min_confidence_<head>, min_similarity
  models/selection.json       dev scores of every candidate (goes into the README / pitch)
  models/index.npy + index_meta.parquet   embeddings of TRAIN tickets with a non-empty answer
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse import csr_matrix, hstack  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from sentineldesk import data, gate, rules  # noqa: E402
from sentineldesk.config import MODELS_DIR, Settings  # noqa: E402
from sentineldesk.features import Embedder  # noqa: E402

HEADS = ["type", "queue", "priority", "critical"]
EMB_WEIGHT = 1.0
TARGET_RECALL = 0.90      # high-priority incidents the gate must catch on dev
TARGET_ACCURACY = 0.75    # accuracy required above a head's confidence threshold
WEAK_EVIDENCE_PCT = 10    # bottom X% of dev top-1 similarities count as "weak precedent"


def tfidf_union() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300_000, sublinear_tf=True)),
    ])


def svm() -> CalibratedClassifierCV:
    return CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3, method="sigmoid")


def lr() -> LogisticRegression:
    return LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000)


def score(head: str, clf, X, y) -> float:
    if head == "critical":
        return average_precision_score(y, clf.predict_proba(X)[:, list(clf.classes_).index(1)])
    return f1_score(y, clf.predict(X), average="macro")


def main() -> None:
    settings = Settings()
    train, dev = data.load_split("train"), data.load_split("dev")
    print(f"train {len(train)} | dev {len(dev)}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    embedder = Embedder(settings.embedding_model, weight=EMB_WEIGHT)
    embedder.cache = {}
    cache_file = MODELS_DIR / "emb_cache.npz"
    if cache_file.is_file():
        z = np.load(cache_file, allow_pickle=True)
        if z["model"].item() == settings.embedding_model:
            embedder.cache.update(zip(z["texts"].tolist(), z["vecs"]))
    E_tr = embedder.encode(train["text"].tolist(), show_progress=True)
    E_dev = embedder.encode(dev["text"].tolist(), show_progress=True)
    texts = list(embedder.cache)
    np.savez(cache_file, texts=np.array(texts, dtype=object), vecs=np.vstack([embedder.cache[t] for t in texts]),
             model=np.array(settings.embedding_model, dtype=object))
    print(f"embeddings done in {time.time() - t0:.0f}s, dim {E_tr.shape[1]}")

    t0 = time.time()
    union = tfidf_union().fit(train["text"])
    T_tr, T_dev = union.transform(train["text"]), union.transform(dev["text"])
    C_tr = hstack([T_tr, csr_matrix(E_tr * EMB_WEIGHT)]).tocsr()
    C_dev = hstack([T_dev, csr_matrix(E_dev * EMB_WEIGHT)]).tocsr()
    print(f"tf-idf features {T_tr.shape[1]} in {time.time() - t0:.0f}s")

    feats = {"tfidf_svm": (T_tr, T_dev, svm), "emb_lr": (E_tr, E_dev, lr), "combo_svm": (C_tr, C_dev, svm)}
    models, selection = {}, {}
    for head in HEADS:
        y_tr, y_dev = _labels(train, head), _labels(dev, head)
        results = {}
        for name, (Xtr, Xdev, make) in feats.items():
            t0 = time.time()
            clf = make().fit(Xtr, y_tr)
            results[name] = (score(head, clf, Xdev, y_dev), clf)
            print(f"  {head:<9} {name:<10} dev {'AP' if head == 'critical' else 'macro-F1'} "
                  f"{results[name][0]:.4f}  ({time.time() - t0:.0f}s)")
        best = max(results, key=lambda k: results[k][0])
        selection[head] = {"chosen": best, **{k: round(v[0], 4) for k, v in results.items()}}
        clf = results[best][1]
        if best == "tfidf_svm":
            steps = [("features", union)]
        elif best == "emb_lr":
            steps = [("features", embedder)]
        else:
            steps = [("features", FeatureUnion([("tfidf", union), ("emb", embedder)]))]
        models[head] = Pipeline(steps + [("clf", clf)])
        print(f"  -> {head}: {best}")

    # ---- thresholds on dev -------------------------------------------------------------
    def proba(head, X_by_name):
        name = selection[head]["chosen"]
        return models[head].steps[-1][1].predict_proba(X_by_name[name])

    dev_X = {"tfidf_svm": T_dev, "emb_lr": E_dev, "combo_svm": C_dev}
    thresholds = {}
    preds, confs = {}, {}
    for head in ("type", "queue", "priority"):
        P = proba(head, dev_X)
        clf = models[head].steps[-1][1]
        conf, lab = P.max(1), clf.classes_[P.argmax(1)]
        preds[head], confs[head] = lab, conf
        correct = lab == _labels(dev, head)
        cut = 0.9
        for c in np.arange(0.30, 0.91, 0.01):
            mask = conf >= c
            if mask.sum() >= 50 and correct[mask].mean() >= TARGET_ACCURACY:
                cut = float(c)
                break
        thresholds[f"min_confidence_{head}"] = round(cut, 2)
        print(f"  min_confidence_{head} = {cut:.2f}  (covers {np.mean(conf >= cut):.0%} of dev)")

    crit_clf = models["critical"].steps[-1][1]
    crit = proba("critical", dev_X)[:, list(crit_clf.classes_).index(1)]
    label_rule = ((preds["type"] == "Incident") & (preds["priority"] == "high")
                  & (confs["type"] >= thresholds["min_confidence_type"])
                  & (confs["priority"] >= thresholds["min_confidence_priority"]))
    impact_rule = np.array([_impact_pages(t) for t in dev["text"]])
    y_crit = _labels(dev, "critical") == 1
    chosen_t = 0.05
    for t in np.arange(0.95, 0.04, -0.01):
        paged = label_rule | (crit >= t) | impact_rule
        if paged[y_crit].mean() >= TARGET_RECALL:
            chosen_t = float(t)
            break
    paged = label_rule | (crit >= chosen_t) | impact_rule
    thresholds["critical"] = round(chosen_t, 2)
    print(f"  critical = {chosen_t:.2f}: dev recall {paged[y_crit].mean():.1%}, "
          f"other tickets paged {paged[~y_crit].mean():.1%} (labels-only recall {label_rule[y_crit].mean():.1%})")

    # ---- retrieval index (train tickets with an answer) ---------------------------------
    kb = train[train["answer"] != ""].reset_index(drop=True)
    vecs = embedder.encode(kb["text"])
    np.save(MODELS_DIR / "index.npy", vecs.astype(np.float32))
    kb[["ticket_id", "subject", "body", "answer", "type", "queue", "priority", "language"]].to_parquet(
        MODELS_DIR / "index_meta.parquet", index=False)
    top1 = (E_dev @ vecs.T).max(1)
    thresholds["min_similarity"] = round(float(np.percentile(top1, WEAK_EVIDENCE_PCT)), 3)
    print(f"  index {len(kb)} tickets | min_similarity = {thresholds['min_similarity']} "
          f"(dev top-1 median {np.median(top1):.3f})")

    joblib.dump(models, MODELS_DIR / "classifier.joblib", compress=3)
    (MODELS_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (MODELS_DIR / "selection.json").write_text(json.dumps(selection, indent=2))
    print("\nselection:", json.dumps(selection, indent=1))
    print("thresholds:", thresholds)


def _labels(df, head: str) -> np.ndarray:
    """Plain numpy labels (pandas 3 arrow strings break sklearn's CV indexing)."""
    return df[head].to_numpy(dtype=int if head == "critical" else object)


def _impact_pages(text: str) -> bool:
    return bool(gate.impact_pages(rules.impact_hits(text)))


if __name__ == "__main__":
    main()
