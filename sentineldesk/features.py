"""Shared feature extractors so trained pipelines take raw ticket text at inference time.

Embedder wraps a sentence-transformers model as an sklearn transformer. The model itself is
not pickled (loaded lazily on first use), and train.py can prefill `cache` with precomputed
vectors so embeddings are computed once and reused by the classifiers and the retrieval index.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Embedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 weight: float = 1.0, batch_size: int = 64):
        self.model_name = model_name
        self.weight = weight
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def _model(self):
        if getattr(self, "_st", None) is None:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self.model_name)
        return self._st

    def encode(self, texts, show_progress: bool = False) -> np.ndarray:
        texts = list(texts)
        cache = getattr(self, "cache", None) or {}
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        if missing:
            vecs = self._model().encode(missing, batch_size=self.batch_size, normalize_embeddings=True,
                                        convert_to_numpy=True, show_progress_bar=show_progress)
            if getattr(self, "cache", None) is not None:
                self.cache.update(zip(missing, vecs))
            else:
                cache = dict(zip(missing, vecs))
        lookup = self.cache if getattr(self, "cache", None) is not None else cache
        return np.vstack([lookup[t] for t in texts]).astype(np.float32)

    def transform(self, X):
        return self.encode(X) * self.weight

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_st", None)
        state.pop("cache", None)
        return state
