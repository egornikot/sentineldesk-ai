"""Score every version on the held-out TEST tickets and write the comparison the dashboard shows.

    py plus/evaluate_plus.py

Standard split (data/processed, models/) and, if trained, the paraphrase-grouped honest split
(plus/honest/processed, plus/honest/models/). For each: the original model (classifier only, original
thresholds) and SentinelDesk+ (twin vote + temperature, re-tuned thresholds). The teammate's Helios v3 numbers
come from their published artifacts/holdout_metrics.json (their own 5,718-ticket split, duplicates grouped).
Writes plus/reports/compare.json, metrics_plus.json, COMPARE_RESULTS.md.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PLUS = Path(__file__).resolve().parent
ROOT = PLUS.parent
sys.path[:0] = [str(ROOT), str(PLUS)]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from sentineldesk.classify import Classifier, HEADS  # noqa: E402
from sentineldesk.config import Settings  # noqa: E402
from sentineldesk.retrieve import Retriever  # noqa: E402
from sentineldesk_plus import core  # noqa: E402

REPORTS = PLUS / "reports"
SPLITS = {
    "standard": (ROOT / "data" / "processed", ROOT / "models", PLUS / "models" / "plus.json"),
    "honest": (PLUS / "honest" / "processed", PLUS / "honest" / "models", PLUS / "honest" / "models" / "plus.json"),
}


def score_split(name: str, s: Settings) -> dict | None:
    split_dir, models_dir, cfg_path = SPLITS[name]
    if not ((models_dir / "classifier.joblib").is_file() and cfg_path.is_file()):
        print(f"[{name}] skipped (not trained)")
        return None
    test = pd.read_parquet(split_dir / "test.parquet")
    clf, retr = Classifier.load(models_dir), Retriever.load(models_dir, s.embedding_model)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    orig_th = {"critical": 0.4, "min_confidence": 0.55, "min_similarity": 0.5,
               **json.loads((models_dir / "thresholds.json").read_text())}
    t0 = time.time()
    raw = core.embed_and_classify(test, clf, retr)
    idx, sims = core.topk(raw["E"], retr.vectors, 30)
    top1 = sims[:, 0]
    orig = core.evaluate(test, {h: raw[h] for h in core.ALL_HEADS}, orig_th, top1, idx, retr.meta)
    plus = core.evaluate(test, core.plus_probs(raw, idx, sims, retr.meta, cfg), cfg["thresholds"], top1, idx, retr.meta)
    twins = {}
    for h in HEADS:
        classes = raw[h][1]
        K = core.knn_proba(idx, sims, core.neighbor_labels(retr.meta, h), classes, cfg["k"], cfg["tau"])
        twins[f"f1_{h}"] = round(float(f1_score(test[h].to_numpy(dtype=object),
                                                np.array(classes, dtype=object)[K.argmax(1)], average="macro")), 4)
    near = float(np.mean(top1 >= 0.95))
    print(f"[{name}] {len(test)} test tickets in {time.time() - t0:.0f}s | test tickets with a train near-copy: {near:.1%}")
    for label, m in (("original", orig), ("plus", plus), ("twins only", twins)):
        print(f"   {label:<10} " + "  ".join(f"{k}={v}" for k, v in m.items() if k.startswith(("f1_", "incident_recall", "other_paged", "ece_type")) and "null" not in k))
    return {"original": orig, "plus": plus, "twins_only": twins, "near_copy_share": round(near, 4), "cfg": cfg,
            "original_thresholds": orig_th}


def latency(s: Settings, n: int = 40) -> dict:
    from sentineldesk.config import MODELS_DIR
    from sentineldesk_plus.pipeline import PlusPipeline
    s.llm_mode = "mock"
    pipe = PlusPipeline.build(s, MODELS_DIR, PLUS / "models" / "plus.json")
    test = pd.read_parquet(ROOT / "data" / "processed" / "test.parquet").sample(n, random_state=3)
    tot, cls = [], []
    for row in test.itertuples():
        r = pipe.run(row.subject, row.body)
        tot.append(r["timings_ms"]["total"])
        cls.append(r["timings_ms"]["classify"] + r["timings_ms"]["retrieve"])
    return {"total_p50": round(float(np.median(tot)), 1), "total_p95": round(float(np.percentile(tot, 95)), 1),
            "classify_retrieve_p50": round(float(np.median(cls)), 1)}


def helios_row() -> dict | None:
    f = REPORTS / "helios_v3_holdout_metrics.json"
    if not f.is_file():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    hp = d["high_priority_incident"]
    return {"name": "Teammate v3 (Helios)", "note": "their own 5,718-ticket split (dups grouped)",
            "f1_type": d["heads"]["type"]["macro_f1"], "f1_queue": d["heads"]["queue"]["macro_f1"],
            "f1_priority": d["heads"]["priority"]["macro_f1"], "incident_recall": hp["recall"],
            "other_paged": hp.get("other_tickets_paged_pct", 0) / 100, "ece_type": d["heads"]["type"].get("ece"),
            "draft_ready": None}


def row(name: str, note: str, m: dict) -> dict:
    return {"name": name, "note": note, **{k: m.get(k) for k in ("f1_type", "f1_queue", "f1_priority", "incident_recall",
                                                                 "other_paged", "ece_type")},
            "draft_ready": m.get("routes", {}).get("draft_ready")}


def main() -> None:
    s = Settings()
    res = {name: score_split(name, s) for name in SPLITS}
    std, hon = res["standard"], res["honest"]
    rows = [row("Original (ours)", "standard split", std["original"]),
            row("SentinelDesk+ (combined)", "standard split", std["plus"])]
    h = helios_row()
    if h:
        rows.append(h)
    if hon:
        rows += [row("Original (ours)", "honest split: no rewrites shared with train", hon["original"]),
                 row("SentinelDesk+ (combined)", "honest split: no rewrites shared with train", hon["plus"])]
    columns = [{"key": "f1_type", "label": "Type F1"}, {"key": "f1_queue", "label": "Queue F1"},
               {"key": "f1_priority", "label": "Priority F1"},
               {"key": "incident_recall", "label": "Incident recall", "pct": True},
               {"key": "other_paged", "label": "Others paged", "pct": True, "lower": True},
               {"key": "ece_type", "label": "ECE type", "lower": True},
               {"key": "draft_ready", "label": "Auto-draft", "pct": True}]
    foot = (f"Standard split: {std['near_copy_share']:.0%} of test tickets have a near-copy (cos≥0.95) in train"
            + (f"; honest split: {hon['near_copy_share']:.0%}" if hon else "") + ". Bold = best in column.")
    compare = {"columns": columns, "rows": rows, "footnote": foot, "standard": std, "honest": hon}
    lat = latency(Settings())
    p = std["plus"]
    headline = {"macro_f1_type": p["f1_type"], "macro_f1_queue": p["f1_queue"], "macro_f1_priority": p["f1_priority"],
                "high_priority_incident_recall": p["incident_recall"], "other_tickets_paged": p["other_paged"],
                "ece_type": p["ece_type"], "auto_draft_ready_share": p["routes"]["draft_ready"],
                "cross_lingual_top5_pct": p.get("cross_lingual_top5_pct"),
                "priority_f1_null_subject": p.get("f1_priority_null_subject"),
                "latency_p50_ms (mock)": lat["total_p50"], "test_tickets": p["n"]}
    if hon:
        headline["honest_split_priority_f1"] = hon["plus"]["f1_priority"]
    core.save_json(REPORTS / "compare.json", compare)
    core.save_json(REPORTS / "metrics_plus.json", {"headline": headline, "latency_ms": lat, "standard": std, "honest": hon})
    md = ["# SentinelDesk versions compared on held-out test tickets", "",
          "| Version | Split | " + " | ".join(c["label"] for c in columns) + " |", "|---|---|" + "---|" * len(columns)]
    for r in rows:
        md.append(f"| {r['name']} | {r['note']} | " + " | ".join(
            "–" if r.get(c["key"]) is None else (f"{r[c['key']] * 100:.1f}%" if c.get("pct") else f"{r[c['key']]:.3f}")
            for c in columns) + " |")
    md += ["", foot, "", f"SentinelDesk+ latency (mock LLM): p50 {lat['total_p50']} ms, classify+retrieve p50 "
           f"{lat['classify_retrieve_p50']} ms.", "", "Twin vote only (copy the nearest train tickets' labels), macro-F1:",
           f"- standard split: {std['twins_only']}"] + ([f"- honest split: {hon['twins_only']}"] if hon else [])
    (REPORTS / "COMPARE_RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
