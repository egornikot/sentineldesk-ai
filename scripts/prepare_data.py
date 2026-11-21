"""Download, clean, dedupe and split the dataset.            OWNER: A        (Hour 1)

    py scripts/prepare_data.py

1. Downloads ONE file (settings.hf_file) into data/raw/ if it isn't there yet.
   Not load_dataset("Tobi-Bueck/customer-support-tickets"): that merges every CSV in the repo,
   and those files share thousands of duplicate tickets, which would leak across the split.
2. data.clean -> data.split (grouped by duplicate text, stratified by type|priority)
3. Saves data/processed/*.parquet and splits/split_assignment.csv (committed, shared with the team)
4. Prints a profile (blank subjects, class balance per split) for the README / judges.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentineldesk import data  # noqa: E402
from sentineldesk.config import DATA_DIR, Settings  # noqa: E402


def download(settings: Settings) -> Path:
    target = DATA_DIR / "raw" / settings.hf_file
    if target.is_file():
        return target
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=settings.hf_dataset, filename=settings.hf_file, repo_type="dataset",
                           local_dir=str(DATA_DIR / "raw"))
    return Path(path)


def main() -> None:
    settings = Settings()
    raw = data.load_raw(download(settings))
    df = data.clean(raw)
    train, dev, test = data.split(df)
    data.save_splits(train, dev, test)

    print(f"raw rows {len(raw)} | after cleaning {len(df)} | blank subjects {int(df['subject_blank'].sum())}")
    print(f"train {len(train)} | dev {len(dev)} | test {len(test)}")
    for col in ("type", "priority", "queue", "language"):
        print(f"\n{col}:")
        for name, part in (("train", train), ("dev", dev), ("test", test)):
            dist = part[col].value_counts(normalize=True).round(3).to_dict()
            print(f"  {name:<5} {dist}")
    for name, part in (("train", train), ("dev", dev), ("test", test)):
        print(f"high-priority incidents in {name}: {int(part['critical'].sum())} ({part['critical'].mean():.1%})")
    print(f"\nwrote {data.PROCESSED} and {data.ASSIGNMENT}")


if __name__ == "__main__":
    main()
