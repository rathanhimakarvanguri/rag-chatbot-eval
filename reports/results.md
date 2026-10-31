# Evaluation report

- Corpus: `data/corpus` (42 chunks)
- Questions: 44 (36 answerable, 8 unanswerable)
- Generator: `extractive`, embedder: `lsa`, k=3, support threshold=0.3

## Retrievers compared (gate on)

| retriever | hit@1 | hit@3 | mrr | ndcg@3 | token_f1 | key_fact_recall | citation_precision | faithfulness | answered_unanswerable | abstain_f1 | false_abstain_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.889 | 0.944 | 0.917 | 0.924 | 0.465 | 0.806 | 0.938 | 1.000 | 0.125 | 0.737 | 0.111 |
| dense[lsa] | 0.889 | 0.972 | 0.926 | 0.938 | 0.461 | 0.806 | 0.968 | 1.000 | 0.125 | 0.700 | 0.139 |
| hybrid[bm25+lsa] | 0.889 | 0.944 | 0.917 | 0.924 | 0.465 | 0.806 | 0.938 | 1.000 | 0.125 | 0.737 | 0.111 |

## Effect of the abstention gate

| retriever | gate | token_f1 | answered_unanswerable | false_abstain_rate |
|---|---|---|---|---|
| bm25 | off | 0.502 | 1.000 | 0.000 |
| bm25 | on | 0.465 | 0.125 | 0.111 |
| dense[lsa] | off | 0.492 | 1.000 | 0.000 |
| dense[lsa] | on | 0.461 | 0.125 | 0.139 |
| hybrid[bm25+lsa] | off | 0.502 | 1.000 | 0.000 |
| hybrid[bm25+lsa] | on | 0.465 | 0.125 | 0.111 |

## Threshold sweep (hybrid[bm25+lsa])

| threshold | abstain_f1 | answered_unanswerable | false_abstain_rate | token_f1 |
|---|---|---|---|---|
| 0.00 | 0.000 | 1.000 | 0.000 | 0.502 |
| 0.05 | 0.000 | 1.000 | 0.000 | 0.502 |
| 0.10 | 0.222 | 0.875 | 0.000 | 0.502 |
| 0.15 | 0.364 | 0.750 | 0.028 | 0.501 |
| 0.20 | 0.364 | 0.750 | 0.028 | 0.501 |
| 0.25 | 0.615 | 0.500 | 0.028 | 0.501 |
| 0.30 | 0.737 | 0.125 | 0.111 | 0.465 |
| 0.35 | 0.737 | 0.125 | 0.111 | 0.465 |
| 0.40 | 0.700 | 0.125 | 0.139 | 0.453 |
| 0.45 | 0.609 | 0.125 | 0.222 | 0.414 |
| 0.50 | 0.583 | 0.125 | 0.250 | 0.396 |
| 0.55 | 0.560 | 0.125 | 0.278 | 0.375 |
| 0.60 | 0.560 | 0.125 | 0.278 | 0.375 |

## By question type (hybrid[bm25+lsa])

| type | n | hit@1 | hit@3 | token_f1 |
|---|---|---|---|---|
| factual | 27 | 1.000 | 1.000 | 0.499 |
| paraphrase | 9 | 0.556 | 0.778 | 0.360 |
