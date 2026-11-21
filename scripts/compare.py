"""Score every prediction file in reports/predictions/ on the test split and try ensembles.

    py scripts/compare.py [--base sentineldesk_ml]

Format: docs/COMPARE.md. Writes reports/compare.md. Reads test labels (evaluation only).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from sentineldesk import data, gate  # noqa: E402
from sentineldesk.config import REPORTS_DIR, Settings  # noqa: E402

HEADS = ("type", "queue", "priority")


def score(df: pd.DataFrame, truth: pd.DataFrame) -> dict:
    m = truth.merge(df, on="ticket_id", how="inner")
    out = {"n": len(m)}
    for h in HEADS:
        out[f"macro_f1_{h}"] = round(f1_score(m[h], m[f"pred_{h}"], average="macro"), 4)
    y = m["critical"] == 1
    paged = (m["pred_type"] == "Incident") & (m["pred_priority"] == "high")
    if "critical_score" in m and m["critical_score"].notna().all():
        paged |= m["critical_score"] >= Settings().thresholds["critical"]
    out["hp_incident_recall"] = round(float(paged[y].mean()), 4)
    out["other_flagged_rate"] = round(float(paged[~y].mean()), 4)
    if "latency_ms" in m:
        out["latency_p50_ms"] = round(float(m["latency_ms"].median()), 1)
    return out


def cascade(base: pd.DataFrame, other: pd.DataFrame, th: dict) -> pd.DataFrame:
    m = base.merge(other, on="ticket_id", suffixes=("", "_o"))
    out = pd.DataFrame({"ticket_id": m["ticket_id"]})
    for h in HEADS:
        keep = m[f"conf_{h}"] >= gate.min_conf(th, h)
        out[f"pred_{h}"] = np.where(keep, m[f"pred_{h}"], m[f"pred_{h}_o"])
        out[f"conf_{h}"] = np.where(keep, m[f"conf_{h}"], m[f"conf_{h}_o"])
    if "critical_score" in m:
        out["critical_score"] = m["critical_score"]
    out.attrs["llm_share"] = {h: round(float((m[f"conf_{h}"] < gate.min_conf(th, h)).mean()), 3) for h in HEADS}
    return out


def prob_average(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame | None:
    m = a.merge(b, on="ticket_id", suffixes=("", "_o"))
    out = pd.DataFrame({"ticket_id": m["ticket_id"]})
    for h in HEADS:
        cols = [c for c in a.columns if c.startswith(f"p_{h}__")]
        if not cols or not all(c + "_o" in m for c in cols):
            return None
        P = (m[cols].values + m[[c + "_o" for c in cols]].values) / 2
        classes = [c.split("__", 1)[1] for c in cols]
        out[f"pred_{h}"] = np.array(classes)[P.argmax(1)]
        out[f"conf_{h}"] = P.max(1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sentineldesk_ml")
    args = ap.parse_args()
    truth = data.load_split("test")[["ticket_id", "type", "queue", "priority", "critical"]]
    files = {p.stem.removesuffix("_test"): pd.read_csv(p) for p in sorted((REPORTS_DIR / "predictions").glob("*_test.csv"))}
    if not files:
        raise SystemExit("No prediction files in reports/predictions/")
    th = Settings().thresholds
    results = {name: score(df, truth) for name, df in files.items()}

    base = files.get(args.base)
    for name, other in files.items():
        if base is None or name == args.base:
            continue
        c = cascade(base, other, th)
        results[f"cascade({args.base} -> {name})"] = {**score(c, truth), "llm_share": c.attrs["llm_share"]}
        avg = prob_average(base, other)
        if avg is not None:
            results[f"prob_avg({args.base}, {name})"] = score(avg, truth)
        both = base.merge(other, on="ticket_id", suffixes=("", "_o"))
        either = both.copy()
        hp_o = (both["pred_type_o"] == "Incident") & (both["pred_priority_o"] == "high")
        either.loc[hp_o, ["pred_type", "pred_priority"]] = ["Incident", "high"]
        results[f"safety_or({args.base}, {name})"] = score(either, truth)
    for a, b in itertools.combinations(files, 2):
        m = files[a].merge(files[b], on="ticket_id", suffixes=("", "_o"))
        results[f"agreement({a}, {b})"] = {h: round(float((m[f"pred_{h}"] == m[f"pred_{h}_o"]).mean()), 3) for h in HEADS}

    lines = ["# Classifier comparison (test split)", "", "```", json.dumps(results, indent=1), "```"]
    (REPORTS_DIR / "compare.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
