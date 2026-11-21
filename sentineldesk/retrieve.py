"""Similar-ticket retrieval over the TRAIN split only.        OWNER: B (retrieval + LLM)

Contract:
    Retriever.load(models_dir, embedding_model) -> Retriever (works with no index yet)
    r.search(subject, body, k=5) -> [
        {"ticket_id", "similarity", "subject", "body", "answer", "type", "queue", "priority", "language"}, ...
    ]  sorted by similarity, highest first

scripts/train.py writes:
    models/index.npy        float32 [n, d], L2-normalized embeddings of ticket_text() for train rows
    models/index_meta.parquet   the columns above for the same rows, same order
Multilingual embeddings give cross-language matching (DE ticket -> EN precedent) for free.
NEVER add dev/test rows to the index.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .data import ticket_text


class Retriever:
    def __init__(self, vectors=None, meta=None, embedding_model: str = "") -> None:
        self.vectors = vectors
        self.meta = meta
        self.embedding_model = embedding_model
        self._encoder = None

    @classmethod
    def load(cls, models_dir: Path, embedding_model: str) -> "Retriever":
        vec_path, meta_path = Path(models_dir) / "index.npy", Path(models_dir) / "index_meta.parquet"
        if not (vec_path.is_file() and meta_path.is_file()):
            return cls(embedding_model=embedding_model)
        import numpy as np
        import pandas as pd
        return cls(np.load(vec_path), pd.read_parquet(meta_path), embedding_model)

    @property
    def ready(self) -> bool:
        return self.vectors is not None and self.meta is not None

    def encode(self, texts: list[str]):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.embedding_model)
        return self._encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def search(self, subject: Optional[str], body: str, k: int = 5) -> list[dict[str, Any]]:
        if not self.ready:
            return []
        import numpy as np
        q = self.encode([ticket_text(subject, body)])[0]
        sims = self.vectors @ q
        top = np.argsort(-sims)[:k]
        rows = []
        for i in top:
            rec = self.meta.iloc[int(i)].to_dict()
            rec = {key: (None if v != v else v) for key, v in rec.items()}  # NaN -> None
            rec["similarity"] = round(float(sims[i]), 3)
            rows.append(rec)
        return rows
