"""Load, clean, dedupe and split the ticket CSV.            OWNER: A (data + models)

    load_raw(path)   -> DataFrame with the official columns + "row" (0-based line in the CSV)
    clean(df)        -> blank subjects -> "", text stripped, empty bodies dropped,
                        adds "text" (ticket_text) and "dup_key" (normalized subject+body)
    split(df)        -> (train, dev, test) = 72 / 8 / 20 %, grouped by dup_key (duplicates never
                        cross splits) and stratified by type|priority; ticket_id "TKT-00000".. assigned
    save_splits / load_split -> data/processed/{train,dev,test}.parquet
    splits/split_assignment.csv (committed) maps CSV row -> ticket_id -> split, so every model on
    the team (incl. the Gemini classifier) is scored on exactly the same test tickets.

Leakage rules: training code reads train (+ dev for tuning). test labels/answers are read ONLY by
scripts/evaluate.py.
"""
from __future__ import annotations

import re
from pathlib import Path

from .config import DATA_DIR, ROOT

PROCESSED = DATA_DIR / "processed"
ASSIGNMENT = ROOT / "splits" / "split_assignment.csv"
LABELS = ["type", "queue", "priority"]
TEXT_COLS = ["subject", "body", "answer"]


def normalize(text: str) -> str:
    return re.sub(r"[^\wäöüß ]+", "", re.sub(r"\s+", " ", str(text or "").lower())).strip()


def ticket_text(subject: str | None, body: str | None) -> str:
    """Single place that defines how subject + body become model input. Use it everywhere."""
    subject = (subject or "").strip() if isinstance(subject, str) else ""
    body = (body or "").strip() if isinstance(body, str) else ""
    return f"{subject}\n{body}" if subject else body


def load_raw(path: Path):
    import pandas as pd
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.insert(0, "row", range(len(df)))
    return df


def clean(df):
    df = df.copy()
    for col in TEXT_COLS + LABELS + ["language"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[df["body"] != ""].reset_index(drop=True)
    df["text"] = [ticket_text(s, b) for s, b in zip(df["subject"], df["body"])]
    df["dup_key"] = [normalize(s) + "||" + normalize(b) for s, b in zip(df["subject"], df["body"])]
    df["subject_blank"] = df["subject"] == ""
    df["critical"] = ((df["type"] == "Incident") & (df["priority"] == "high")).astype(int)
    return df


def split(df, seed: int = 42):
    from sklearn.model_selection import StratifiedGroupKFold

    strat = df["type"] + "|" + df["priority"]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(outer.split(df, strat, groups=df["dup_key"]))
    rest = df.iloc[rest_idx]
    inner = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    train_idx, dev_idx = next(inner.split(rest, strat.iloc[rest_idx], groups=rest["dup_key"]))

    df = df.copy()
    df["split"] = "test"
    df.loc[rest.index[train_idx], "split"] = "train"
    df.loc[rest.index[dev_idx], "split"] = "dev"
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
    parts = {name: df[df["split"] == name].reset_index(drop=True) for name in ("train", "dev", "test")}
    assert not set(parts["test"]["dup_key"]) & set(parts["train"]["dup_key"]), "duplicate text crosses train/test"
    return parts["train"], parts["dev"], parts["test"]


def save_splits(train, dev, test) -> None:
    import pandas as pd
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("dev", dev), ("test", test)):
        frame.to_parquet(PROCESSED / f"{name}.parquet", index=False)
    ASSIGNMENT.parent.mkdir(parents=True, exist_ok=True)
    allrows = pd.concat([train, dev, test])[["row", "ticket_id", "split"]].sort_values("row")
    allrows.to_csv(ASSIGNMENT, index=False)


def load_split(name: str):
    import pandas as pd
    return pd.read_parquet(PROCESSED / f"{name}.parquet")
