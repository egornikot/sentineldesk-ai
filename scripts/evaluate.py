"""Holdout evaluation on the 20% test split.                  OWNER: A

    py scripts/evaluate.py                 # classifier + gate metrics, latency with the mock LLM
    py scripts/evaluate.py --live 40       # also grounding rate with Claude on 40 test tickets (costs API calls)

The ONLY script that reads test labels. Prediction sees subject + body only; labels are used
afterwards for scoring.

Writes:
  reports/metrics.json  ({"headline": {...}, ...}; the dashboard shows "headline")
  reports/metrics.md    (paste into README / slides)
  reports/calibration.png
  reports/predictions/sentineldesk_ml_test.csv   predictions only (no true labels), in the shared
                        format from docs/COMPARE.md, so other models (e.g. the Gemini classifier)
                        can be compared and ensembled ticket by ticket
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, f1_score  # noqa: E402

from sentineldesk import data, gate, rules  # noqa: E402
from sentineldesk.classify import HEADS, Classifier  # noqa: E402
from sentineldesk.config import MODELS_DIR, REPORTS_DIR, Settings  # noqa: E402
from sentineldesk.pipeline import Pipeline  # noqa: E402
from sentineldesk.retrieve import Retriever  # noqa: E402


TEST_VECS = None  # test embeddings, computed once in predict_test


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    edges = np.linspace(0, 1, bins + 1)
    total, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc, avg = correct[m].mean(), conf[m].mean()
        total += m.mean() * abs(acc - avg)
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "accuracy": round(float(acc), 3),
                      "confidence": round(float(avg), 3)})
    return round(float(total), 4), table


def predict_test(test: pd.DataFrame, clf: Classifier, retr: Retriever) -> pd.DataFrame:
    """Embeds the test texts ONCE and shares the vectors with every embedding-based model."""
    from sentineldesk.features import Embedder
    from sentineldesk.pipeline import _walk
    global TEST_VECS
    texts = test["text"].tolist()
    vecs = TEST_VECS = retr.encode(texts)
    cache = dict(zip(texts, vecs))
    for model in clf.models.values():
        for step in _walk(model):
            if isinstance(step, Embedder):
                step.cache = cache
    out = pd.DataFrame({"ticket_id": test["ticket_id"], "row": test["row"]})
    for head in HEADS:
        pipe = clf.models[head]
        P = pipe.predict_proba(texts)
        out[f"pred_{head}"] = pipe.classes_[P.argmax(1)]
        out[f"conf_{head}"] = P.max(1).round(4)
        for j, c in enumerate(pipe.classes_):
            out[f"p_{head}__{c}"] = P[:, j].round(4)
    crit = clf.models["critical"]
    out["critical_score"] = crit.predict_proba(texts)[:, list(crit.classes_).index(1)].round(4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=int, default=0, help="grounding sample size with Claude (0 = skip)")
    ap.add_argument("--latency-sample", type=int, default=100)
    args = ap.parse_args()

    settings = Settings()
    test = data.load_split("test")
    train = data.load_split("train")
    clf = Classifier.load(MODELS_DIR)
    if clf.engine != "trained":
        raise SystemExit("No trained models: run scripts/train.py first.")
    th = settings.thresholds
    print(f"test {len(test)} tickets | thresholds {th}")

    t0 = time.time()
    retr = Retriever.load(MODELS_DIR, settings.embedding_model)
    pred = predict_test(test, clf, retr)
    print(f"classified test in {time.time() - t0:.1f}s")

    # ---- classification -----------------------------------------------------------------
    cls_metrics, reports = {}, {}
    for head in ("type", "queue", "priority"):
        y, p = test[head].to_numpy(dtype=object), pred[f"pred_{head}"].to_numpy(dtype=object)
        majority = train[head].value_counts().idxmax()
        cls_metrics[head] = {
            "macro_f1": round(f1_score(y, p, average="macro"), 4),
            "accuracy": round(accuracy_score(y, p), 4),
            "baseline_majority_macro_f1": round(f1_score(y, [majority] * len(y), average="macro"), 4),
        }
        reports[head] = classification_report(y, p, digits=3, zero_division=0)
        conf = pred[f"conf_{head}"].values
        cls_metrics[head]["ece"], cls_metrics[head]["calibration"] = ece(conf, y == p)
        cut = gate.min_conf(th, head)
        cls_metrics[head]["confident_share"] = round(float((conf >= cut).mean()), 4)
        cls_metrics[head]["accuracy_when_confident"] = round(float((y == p)[conf >= cut].mean()), 4)

    # ---- high-priority incident recall (two tiers + rules) --------------------------------
    y_crit = test["critical"].to_numpy(dtype=int) == 1
    label_rule = ((pred["pred_type"] == "Incident").to_numpy(dtype=bool)
                  & (pred["pred_priority"] == "high").to_numpy(dtype=bool)
                  & (pred["conf_type"].to_numpy(dtype=float) >= gate.min_conf(th, "type"))
                  & (pred["conf_priority"].to_numpy(dtype=float) >= gate.min_conf(th, "priority")))
    det_rule = pred["critical_score"].to_numpy(dtype=float) >= th["critical"]
    impact_rule = np.array([bool(gate.impact_pages(rules.impact_hits(t))) for t in test["text"]])
    tiers = {}
    for name, mask in (("labels_only", label_rule), ("labels_plus_detector", label_rule | det_rule),
                       ("full_gate", label_rule | det_rule | impact_rule)):
        tiers[name] = {"recall": round(float(mask[y_crit].mean()), 4), "caught": int(mask[y_crit].sum()),
                       "of": int(y_crit.sum()), "other_flagged_rate": round(float(mask[~y_crit].mean()), 4),
                       "other_flagged": int(mask[~y_crit].sum()), "other_total": int((~y_crit).sum())}

    # ---- routes on the whole test split (retrieval similarity from the index) ------------
    top1 = (TEST_VECS @ retr.vectors.T).max(1)
    routes = []
    for i, row in enumerate(test.itertuples()):
        c = {h: {"label": pred.at[i, f"pred_{h}"], "confidence": float(pred.at[i, f"conf_{h}"])}
             for h in ("type", "queue", "priority")}
        c["critical_score"] = float(pred.at[i, "critical_score"])
        routes.append(gate.decide(row.text, c, [{"similarity": float(top1[i])}], th)["route"])
    pred["route"] = routes
    route_rates = pd.Series(routes).value_counts(normalize=True).round(4).to_dict()

    # ---- latency (+ optional grounding with Claude) ---------------------------------------
    # --live N: N tickets through Claude (latency + grounding); otherwise the mock LLM
    n = args.live if args.live else args.latency_sample
    sample = test.sample(n=min(n, len(test)), random_state=1)
    settings.llm_mode = "anthropic" if (args.live and settings.anthropic_key) else "mock"
    pipe = Pipeline.from_settings(settings)
    timings, grounding = [], []
    for row in sample.itertuples():
        r = pipe.run(row.subject, row.body)
        timings.append(r["timings_ms"])
        if r["verification"] and r["draft"] and r["draft"]["claims"]:
            grounding.append(r["verification"]["grounding_rate"])
    lat = pd.DataFrame(timings)
    latency = {col: {"p50": round(float(lat[col].quantile(.5)), 1), "p95": round(float(lat[col].quantile(.95)), 1)}
               for col in lat.columns}

    headline = {
        "macro_f1_type": cls_metrics["type"]["macro_f1"],
        "macro_f1_queue": cls_metrics["queue"]["macro_f1"],
        "macro_f1_priority": cls_metrics["priority"]["macro_f1"],
        "high_priority_incident_recall": tiers["full_gate"]["recall"],
        "recall_labels_only": tiers["labels_only"]["recall"],
        "other_tickets_paged": tiers["full_gate"]["other_flagged_rate"],
        "ece_type": cls_metrics["type"]["ece"],
        "auto_draft_ready_share": route_rates.get("draft_ready", 0.0),
        "latency_p50_ms (" + pipe.llm.name + ")": latency["total"]["p50"],
        "latency_p95_ms (" + pipe.llm.name + ")": latency["total"]["p95"],
        "grounding_rate (" + pipe.llm.name + ")": round(float(np.mean(grounding)), 4) if grounding else None,
        "test_tickets": len(test),
    }
    metrics = {"headline": headline, "classification": cls_metrics, "high_priority_recall": tiers,
               "routes": route_rates, "latency_ms": latency, "thresholds": th,
               "selection": json.loads((MODELS_DIR / "selection.json").read_text())}

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "predictions").mkdir(exist_ok=True)
    pred.to_csv(REPORTS_DIR / "predictions" / "sentineldesk_ml_test.csv", index=False)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_markdown(metrics, reports)
    plot_calibration(cls_metrics)
    print(json.dumps(headline, indent=2))
    print("\nhigh-priority recall tiers:", json.dumps(tiers, indent=1))
    print("routes:", route_rates)


def write_markdown(m: dict, reports: dict) -> None:
    h, t = m["headline"], m["high_priority_recall"]
    lines = ["# SentinelDesk holdout metrics", "",
             f"Test split: {h['test_tickets']} tickets (20%, grouped by duplicate text, never seen in training).", "",
             "## Classification", "", "| Head | Macro-F1 | Accuracy | Majority baseline F1 | ECE | Model |", "|---|---|---|---|---|---|"]
    for head, c in m["classification"].items():
        lines.append(f"| {head} | **{c['macro_f1']:.3f}** | {c['accuracy']:.3f} | {c['baseline_majority_macro_f1']:.3f} | "
                     f"{c['ece']:.3f} | {m['selection'][head]['chosen']} |")
    lines += ["", "## High-priority incident recall", "",
              "| Rule | Caught | Recall | Other tickets flagged |", "|---|---|---|---|"]
    for name, r in t.items():
        lines.append(f"| {name.replace('_', ' ')} | {r['caught']}/{r['of']} | **{r['recall']:.1%}** | "
                     f"{r['other_flagged']}/{r['other_total']} ({r['other_flagged_rate']:.1%}) |")
    lines += ["", "## Routes", "", ", ".join(f"{k}: {v:.1%}" for k, v in m["routes"].items()), "",
              "## Latency (ms)", "", "| Stage | p50 | p95 |", "|---|---|---|"]
    lines += [f"| {k} | {v['p50']} | {v['p95']} |" for k, v in m["latency_ms"].items()]
    lines += ["", "## Model selection (dev split)", "", "```", json.dumps(m["selection"], indent=1), "```", ""]
    for head, rep in reports.items():
        lines += [f"### {head} per-class report", "", "```", rep, "```", ""]
    (REPORTS_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def plot_calibration(cls_metrics: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#999", label="perfect")
    for head, c in cls_metrics.items():
        xs = [b["confidence"] for b in c["calibration"]]
        ys = [b["accuracy"] for b in c["calibration"]]
        ax.plot(xs, ys, marker="o", label=f"{head} (ECE {c['ece']:.3f})")
    ax.set_xlabel("confidence"), ax.set_ylabel("accuracy"), ax.set_title("Calibration on test split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "calibration.png", dpi=120)


if __name__ == "__main__":
    main()
