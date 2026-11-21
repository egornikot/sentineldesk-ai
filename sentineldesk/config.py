"""Settings from environment variables and an optional .env file (stdlib only).

Real environment variables win over .env, and a
provider whose key is missing falls back to the offline mock so the app always starts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


class Settings:
    def __init__(self) -> None:
        load_dotenv()
        self.anthropic_key = _env("ANTHROPIC_API_KEY")
        self.model = _env("ANTHROPIC_MODEL", "claude-opus-5-5")
        self.effort = _env("ANTHROPIC_EFFORT", "low")  # low keeps drafting latency demo-friendly
        self.gemini_key = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
        self.gemini_model = _env("GEMINI_MODEL", "gemini-3.5-flash-lite")  # ~1 s drafts; benchmarked in scripts/bench_llm.py
        self.gemini_thinking = _env("GEMINI_THINKING", "low")
        self.gemini_fallbacks = [m.strip() for m in _env(
            "GEMINI_FALLBACK_MODELS", "gemini-3.8-flash").split(",") if m.strip()]
        mode = _env("LLM_MODE").lower()
        if mode not in {"gemini", "anthropic", "mock"}:
            mode = "gemini" if self.gemini_key else "anthropic" if self.anthropic_key else "mock"
        # never pick a provider whose key is missing: fall through to the next one, then the mock
        if mode == "gemini" and not self.gemini_key:
            mode = "anthropic" if self.anthropic_key else "mock"
        if mode == "anthropic" and not self.anthropic_key:
            mode = "gemini" if self.gemini_key else "mock"
        self.llm_mode = mode

        self.hf_dataset = _env("HF_DATASET", "Tobi-Bueck/customer-support-tickets")
        self.hf_file = _env("HF_FILE", "aa_dataset-tickets-multi-lang-5-2-50-version.csv")
        self.embedding_model = _env("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.top_k = int(_env("TOP_K", "5") or 5)

        self.host = _env("HOST", "127.0.0.1")
        self.port = int(_env("PORT", "8000") or 8000)

        # Gate thresholds: .env values, overridden by models/thresholds.json written by scripts/train.py
        self.thresholds = {
            "critical": _float("CRITICAL_THRESHOLD", 0.40),
            "min_confidence": _float("MIN_CONFIDENCE", 0.55),
            "min_similarity": _float("MIN_SIMILARITY", 0.50),
        }
        tuned = MODELS_DIR / "thresholds.json"
        if tuned.is_file():
            try:
                self.thresholds.update({k: float(v) for k, v in json.loads(tuned.read_text()).items()})
            except (ValueError, TypeError):
                pass
