"""Outage burst radar (idea from Helios): several similar incident reports in a short window = one major incident.

EN and DE reports cluster together because the embeddings are multilingual.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime

import numpy as np


class BurstRadar:
    # 0.72: same-outage reports score 0.74-0.87 with the multilingual MiniLM, unrelated tickets <= 0.25
    def __init__(self, window_s: int = 7200, min_similarity: float = 0.72, min_size: int = 3, max_items: int = 500):
        self.window_s, self.min_similarity, self.min_size = window_s, min_similarity, min_size
        self.items: deque = deque(maxlen=max_items)
        self.lock = threading.Lock()
        self.counter = 0
        self.seq = 0

    def _prune(self, now: float) -> None:
        while self.items and now - self.items[0]["t"] > self.window_s:
            self.items.popleft()

    def observe(self, vec: np.ndarray, subject: str, body: str, language: str, incident_like: bool) -> dict | None:
        now = time.time()
        with self.lock:
            self._prune(now)
            self.seq += 1
            item = {"t": now, "vec": vec, "id": f"LIVE-{self.seq:04d}", "subject": subject or "(no subject)",
                    "snippet": (body or "")[:90], "language": language, "incident_like": incident_like, "incident": None}
            members = []
            for it in self.items:
                s = float(it["vec"] @ vec)
                if s >= self.min_similarity and it["incident_like"]:
                    members.append((it, s))
            self.items.append(item)
            if not incident_like or len(members) + 1 < self.min_size:
                return None
            incident = next((it["incident"] for it, _ in members if it["incident"]), None)
            if incident is None:
                self.counter += 1
                incident = f"MIR-{datetime.now():%Y%m%d}-{self.counter:02d}"
            for it, _ in members:
                it["incident"] = incident
            item["incident"] = incident
            return self._describe(incident, sims={id(it): s for it, s in members})

    def _describe(self, incident: str, sims: dict | None = None) -> dict:
        group = [it for it in self.items if it["incident"] == incident]
        return {"incident_id": incident, "count": len(group),
                "languages": sorted({it["language"] for it in group}),
                "members": [{"id": it["id"], "subject": it["subject"], "snippet": it["snippet"], "language": it["language"],
                             "similarity": round(sims.get(id(it), 1.0), 3) if sims else None} for it in group]}

    def snapshot(self) -> dict:
        with self.lock:
            self._prune(time.time())
            ids = list(dict.fromkeys(it["incident"] for it in self.items if it["incident"]))
            return {"active": [self._describe(i) for i in ids], "tracked": len(self.items),
                    "window_minutes": self.window_s // 60, "min_similarity": self.min_similarity, "min_size": self.min_size}

    def reset(self) -> None:
        with self.lock:
            self.items.clear()
