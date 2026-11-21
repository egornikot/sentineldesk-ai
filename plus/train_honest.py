"""Retrain the ORIGINAL model recipe (scripts/train.py, unchanged) on the paraphrase-grouped split.

    py plus/train_honest.py        -> plus/honest/models/  (classifier, thresholds, index)

Redirects the original code's data/model folders at runtime instead of editing it.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PLUS = Path(__file__).resolve().parent
ROOT = PLUS.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import sentineldesk.config as config  # noqa: E402
import sentineldesk.data as data  # noqa: E402

HONEST = PLUS / "honest"
config.MODELS_DIR = HONEST / "models"
data.PROCESSED = HONEST / "processed"
config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
if (ROOT / "models" / "emb_cache.npz").is_file() and not (config.MODELS_DIR / "emb_cache.npz").is_file():
    shutil.copy(ROOT / "models" / "emb_cache.npz", config.MODELS_DIR / "emb_cache.npz")  # same texts, faster

import train  # noqa: E402  (scripts/train.py, imported after the redirect)

if __name__ == "__main__":
    train.main()
