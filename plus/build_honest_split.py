"""Paraphrase-grouped ("honest") split: rewrites of the same scenario never cross train/dev/test.

    py plus/build_honest_split.py [--threshold 0.9]

The dataset rewrites each scenario many times; a split that only removes exact duplicates (ours, and every
random split) leaves ~42% of test tickets with a near-copy (cosine >= 0.95) in train. Here tickets are linked
when their multilingual embeddings have cosine >= threshold, connected components become groups, and a
stratified GROUP split keeps every group on one side. Ticket IDs stay the same as the standard split.
Writes plus/honest/processed/{train,dev,test}.parquet and plus/honest/split_assignment.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PLUS = Path(__file__).resolve().parent
ROOT = PLUS.parent
sys.path[:0] = [str(ROOT), str(PLUS)]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentineldesk import data  # noqa: E402
from sentineldesk.config import MODELS_DIR, Settings  # noqa: E402

OUT = PLUS / "honest"


def embeddings(df: pd.DataFrame) -> np.ndarray:
    cache = {}
    f = MODELS_DIR / "emb_cache.npz"
    if f.is_file():
        z = np.load(f, allow_pickle=True)
        cache = dict(zip(z["texts"].tolist(), z["vecs"]))
    missing = [t for t in dict.fromkeys(df["text"]) if t not in cache]
    if missing:
        from sentence_transformers import SentenceTransformer
        print(f"embedding {len(missing)} texts ...", flush=True)
        st = SentenceTransformer(Settings().embedding_model)
        cache.update(zip(missing, st.encode(missing, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)))
    return np.vstack([cache[t] for t in df["text"]]).astype(np.float32)


def components(E: np.ndarray, thr: float, chunk: int = 2000) -> np.ndarray:
    parent = np.arange(len(E))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for s in range(0, len(E), chunk):
        S = E[s:s + chunk] @ E.T
        rows, cols = np.nonzero(S >= thr)
        for r, c in zip(rows + s, cols):
            if r < c:
                a, b = find(r), find(c)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return np.array([find(i) for i in range(len(E))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = pd.concat([data.load_split(n) for n in ("train", "dev", "test")], ignore_index=True)
    E = embeddings(df)
    comp = components(E, a.threshold)
    sizes = pd.Series(comp).value_counts()
    print(f"threshold {a.threshold}: {len(sizes)} groups | largest {sizes.iloc[0]} ({sizes.iloc[0] / len(df):.1%}) | "
          f"singletons {(sizes == 1).sum()} | tickets in groups >1: {(sizes[sizes > 1].sum()) / len(df):.1%}")
    if sizes.iloc[0] / len(df) > 0.15:
        sys.exit("largest group too big to split fairly: raise --threshold")
    df["group"] = comp

    from sklearn.model_selection import StratifiedGroupKFold
    strat = df["type"] + "|" + df["priority"]
    rest_idx, test_idx = next(StratifiedGroupKFold(5, shuffle=True, random_state=a.seed).split(df, strat, df["group"]))
    rest = df.iloc[rest_idx]
    tr, dv = next(StratifiedGroupKFold(10, shuffle=True, random_state=a.seed).split(rest, strat.iloc[rest_idx], rest["group"]))
    df["split"] = "test"
    df.loc[rest.index[tr], "split"] = "train"
    df.loc[rest.index[dv], "split"] = "dev"
    assert not set(df.loc[df.split == "test", "group"]) & set(df.loc[df.split != "test", "group"])

    test_E, train_E = E[(df.split == "test").to_numpy()], E[(df.split == "train").to_numpy()]
    near = np.concatenate([(test_E[s:s + 2000] @ train_E.T).max(1) for s in range(0, len(test_E), 2000)])
    print(f"test tickets with a train near-copy (cos>=0.95): {np.mean(near >= 0.95):.1%} (standard split: ~42%)")
    (OUT / "processed").mkdir(parents=True, exist_ok=True)
    for name in ("train", "dev", "test"):
        part = df[df.split == name].drop(columns=["group"]).reset_index(drop=True)
        part.to_parquet(OUT / "processed" / f"{name}.parquet", index=False)
        print(f"  {name:<5} {len(part)}")
    df[["row", "ticket_id", "split"]].sort_values("row").to_csv(OUT / "split_assignment.csv", index=False)


if __name__ == "__main__":
    main()
