# SentinelDesk holdout metrics

Test split: 5718 tickets (20%, grouped by duplicate text, never seen in training).

## Classification

| Head | Macro-F1 | Accuracy | Majority baseline F1 | ECE | Model |
|---|---|---|---|---|---|
| type | **0.859** | 0.853 | 0.143 | 0.032 | tfidf_svm |
| queue | **0.571** | 0.576 | 0.045 | 0.103 | combo_svm |
| priority | **0.587** | 0.615 | 0.191 | 0.065 | tfidf_svm |

## High-priority incident recall

| Rule | Caught | Recall | Other tickets flagged |
|---|---|---|---|
| labels only | 507/986 | **51.4%** | 227/4732 (4.8%) |
| labels plus detector | 869/986 | **88.1%** | 994/4732 (21.0%) |
| full gate | 886/986 | **89.9%** | 1260/4732 (26.6%) |

## Routes

human_review: 53.4%, page_now: 37.5%, draft_ready: 9.0%

## Latency (ms)

| Stage | p50 | p95 |
|---|---|---|
| preprocess | 4.8 | 7.4 |
| classify | 83.7 | 123.4 |
| retrieve | 27.7 | 50.7 |
| gate | 0.3 | 0.5 |
| generate | 0.2 | 0.3 |
| verify | 0.1 | 0.2 |
| total | 116.8 | 190.2 |

## Model selection (dev split)

```
{
 "type": {
  "chosen": "tfidf_svm",
  "tfidf_svm": 0.8659,
  "emb_lr": 0.8063,
  "combo_svm": 0.8649
 },
 "queue": {
  "chosen": "combo_svm",
  "tfidf_svm": 0.5885,
  "emb_lr": 0.2727,
  "combo_svm": 0.5932
 },
 "priority": {
  "chosen": "tfidf_svm",
  "tfidf_svm": 0.5797,
  "emb_lr": 0.4355,
  "combo_svm": 0.5736
 },
 "critical": {
  "chosen": "combo_svm",
  "tfidf_svm": 0.6774,
  "emb_lr": 0.5036,
  "combo_svm": 0.6842
 }
}
```

### type per-class report

```
              precision    recall  f1-score   support

      Change      0.990     0.978     0.984       584
    Incident      0.793     0.868     0.829      2293
     Problem      0.696     0.572     0.628      1202
     Request      0.993     0.995     0.994      1639

    accuracy                          0.853      5718
   macro avg      0.868     0.853     0.859      5718
weighted avg      0.850     0.853     0.850      5718

```

### queue per-class report

```
                                 precision    recall  f1-score   support

           Billing and Payments      0.842     0.760     0.799       567
               Customer Service      0.516     0.486     0.500       885
                General Inquiry      0.933     0.341     0.500        82
                Human Resources      0.933     0.365     0.525       115
                     IT Support      0.610     0.400     0.483       695
                Product Support      0.513     0.486     0.499      1055
          Returns and Exchanges      0.739     0.398     0.517       284
            Sales and Pre-Sales      0.736     0.426     0.540       190
Service Outages and Maintenance      0.793     0.670     0.726       200
              Technical Support      0.517     0.757     0.615      1645

                       accuracy                          0.576      5718
                      macro avg      0.713     0.509     0.570      5718
                   weighted avg      0.602     0.576     0.571      5718

```

### priority per-class report

```
              precision    recall  f1-score   support

        high      0.627     0.678     0.652      2236
         low      0.669     0.371     0.477      1179
      medium      0.590     0.679     0.631      2303

    accuracy                          0.615      5718
   macro avg      0.629     0.576     0.587      5718
weighted avg      0.621     0.615     0.607      5718

```
