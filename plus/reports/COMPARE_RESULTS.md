# SentinelDesk results on held-out test tickets

| Model | Split | Type F1 | Queue F1 | Priority F1 | Incident recall | Others paged | ECE type | Auto-draft |
|---|---|---|---|---|---|---|---|---|
| SentinelDesk | standard split | 0.929 | 0.782 | 0.818 | 91.5% | 16.5% | 0.013 | 54.0% |
| Ablation: classifiers only (no twin vote / calibration) | standard split | 0.859 | 0.571 | 0.587 | 89.9% | 26.6% | 0.032 | 9.0% |
| SentinelDesk | honest split: no rewrites shared with train | 0.815 | 0.367 | 0.458 | 96.8% | 45.3% | 0.009 | 2.4% |
| Ablation: classifiers only (no twin vote / calibration) | honest split: no rewrites shared with train | 0.821 | 0.299 | 0.389 | 95.9% | 45.0% | 0.017 | 0.0% |

Standard split: 41% of test tickets have a near-copy (cos≥0.95) in train; honest split: 0%.

Latency (mock LLM): p50 122.5 ms, classify+retrieve p50 117.5 ms.

Twin vote only (copy the nearest train tickets' labels), macro-F1:
- standard split: {'f1_type': 0.9197, 'f1_queue': 0.7556, 'f1_priority': 0.8098}
- honest split: {'f1_type': 0.7654, 'f1_queue': 0.3502, 'f1_priority': 0.4577}
