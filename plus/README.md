# SentinelDesk+ (the main version)

The main version of SentinelDesk. It is built **on top of** the original `sentineldesk/` package, which is kept
as the baseline (`py run.py`, port 8000). This version runs with `py plus/run_plus.py` on port 8001.

It keeps our core (calibrated classifiers, the three-route gate, quote-verified citations, PII masking before any
model sees text) and adds the best ideas from the three team builds:

| Added | Where the idea came from | What it does |
|---|---|---|
| **Twin vote** | new (data finding) | The dataset rewrites each scenario many times. The 5 most similar *training* tickets vote on type/queue/priority/critical; the vote is blended with the classifier (weights tuned on dev). |
| **Temperature scaling** | Helios v3, akshita branch | Per-head calibration on dev, so a 80% confidence means ~80% correct. |
| **Outage burst radar** | Helios | ≥3 similar incident reports (EN + DE) within 2 h become one major incident `MIR-…`; new matching tickets are escalated automatically. "Simulate outage burst" button on the dashboard. |
| **"What left your network"** | Helios | Shows the exact masked text the LLM received (`[EMAIL]`, `[PHONE]`, …). |
| **Placeholder cleanup** | akshita branch | The dataset's `<tel_num>`, `<name>`, `<Tel-Nummer>` … are normalised in precedents; a raw placeholder in a draft is flagged. |
| **Precedent threshold by queue agreement** | akshita branch | "Weak precedent" = below the similarity where the nearest past ticket has the right queue ≥85% of the time. |
| **Honest split + extra metrics** | new; null-subject & cross-lingual from Helios | Paraphrase-grouped split (no rewrite of a test scenario in train), null-subject slice, cross-lingual retrieval share. |

## Results

See [reports/COMPARE_RESULTS.md](reports/COMPARE_RESULTS.md) (also shown on the dashboard, "All versions, same test tickets").

**The data finding behind it.** On the standard split, 42% of test tickets have a near-copy (cosine ≥ 0.95) in
training, and that near-copy has the same priority 98% of the time. A random or exact-duplicate split therefore
rewards memorising rewrites. We report both:
- **standard split**: comparable with every other team's numbers;
- **honest split**: rewrites grouped (0% near-copies), the number that predicts performance on genuinely new tickets.

## Run it

```powershell
py plus/train_plus.py            # tune twin vote + temperature + thresholds on dev  (~2 min, uses models/)
py plus/evaluate_plus.py         # score all versions on test -> plus/reports/
py plus/run_plus.py              # dashboard at http://localhost:8001
```

Honest split (optional, ~15 min): `py plus/build_honest_split.py`, `py plus/train_honest.py`,
`py plus/train_plus.py --models plus/honest/models --split plus/honest/processed --out plus/honest/models/plus.json`.

Public link: `go_public.bat` serves it at https://egornikot.github.io/sentineldesk-ai/ (the main page) from this computer;
when it is off, the page replays recorded runs (`py plus/build_static_plus.py`).
