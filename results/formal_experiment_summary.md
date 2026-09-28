# Formal Experiment Summary

- Source files matched: `*formal*experiment*.json`
- Rows summarized: `36`
- CSV output: `formal_experiment_summary.csv`

## Quick Takeaways
- Best overall setting by macro F1: `private_formal_50 / chexagent / labels_only / single_vlm` with macro F1 `0.457`.
- Model average `chexagent`: macro F1 `0.224`, exact match `31.8%`, hallucination FP ratio `0.165`, normal-case accuracy `59.3%`.
- Model average `medgemma-27b`: macro F1 `0.187`, exact match `48.0%`, hallucination FP ratio `0.055`, normal-case accuracy `93.3%`.
- Model average `medgemma-4b`: macro F1 `0.178`, exact match `49.2%`, hallucination FP ratio `0.050`, normal-case accuracy `91.7%`.
- Workflow average `single_vlm`: macro F1 `0.197`, exact match `41.3%`, hallucination FP ratio `0.118`, abnormal-case partial match `38.2%`.
- Workflow average `multi_agent`: macro F1 `0.196`, exact match `44.7%`, hallucination FP ratio `0.062`, abnormal-case partial match `32.2%`.
- Prompt average `structured`: macro F1 `0.211`, exact match `42.5%`, hallucination FP ratio `0.084`.
- Prompt average `concise`: macro F1 `0.198`, exact match `41.5%`, hallucination FP ratio `0.098`.
- Prompt average `labels_only`: macro F1 `0.182`, exact match `45.0%`, hallucination FP ratio `0.088`.

## Best Per Dataset
- `mimic_formal_50` best macro F1: `medgemma-27b / structured / multi_agent` -> macro F1 `0.335`, exact match `46.0%`.
- `private_formal_50` best macro F1: `chexagent / labels_only / single_vlm` -> macro F1 `0.457`, exact match `46.0%`.

## Full Table

| Dataset | Model | Prompt | Workflow | Exact | Partial | Macro F1 | Hall FP | Normal Acc | Abnormal Partial | Avg Latency | Avg Tokens | Avg Pred Labels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mimic_formal_50 | chexagent | structured | single_vlm | 6.0% | 30.0% | 0.118 | 0.351 | 12.0% | 48.0% | 1.32s | 22.2 | 4.24 |
| mimic_formal_50 | chexagent | structured | multi_agent | 24.0% | 32.0% | 0.160 | 0.089 | 40.0% | 24.0% | 6.30s | 144.5 | 1.32 |
| mimic_formal_50 | chexagent | concise | single_vlm | 6.0% | 28.0% | 0.093 | 0.380 | 12.0% | 44.0% | 1.44s | 23.9 | 4.56 |
| mimic_formal_50 | chexagent | concise | multi_agent | 38.0% | 42.0% | 0.127 | 0.073 | 72.0% | 12.0% | 3.96s | 70.9 | 1.26 |
| mimic_formal_50 | chexagent | labels_only | single_vlm | 6.0% | 28.0% | 0.094 | 0.367 | 12.0% | 44.0% | 1.37s | 23.3 | 4.42 |
| mimic_formal_50 | chexagent | labels_only | multi_agent | 32.0% | 46.0% | 0.127 | 0.125 | 64.0% | 28.0% | 2.92s | 49.7 | 1.86 |
| mimic_formal_50 | medgemma-27b | structured | single_vlm | 50.0% | 60.0% | 0.138 | 0.047 | 100.0% | 20.0% | 4.86s | 104.3 | 1.12 |
| mimic_formal_50 | medgemma-27b | structured | multi_agent | 46.0% | 74.0% | 0.335 | 0.053 | 92.0% | 56.0% | 6.27s | 135.1 | 1.40 |
| mimic_formal_50 | medgemma-27b | concise | single_vlm | 52.0% | 66.0% | 0.263 | 0.051 | 100.0% | 32.0% | 3.41s | 72.9 | 1.26 |
| mimic_formal_50 | medgemma-27b | concise | multi_agent | 42.0% | 62.0% | 0.211 | 0.085 | 80.0% | 44.0% | 4.17s | 87.5 | 1.64 |
| mimic_formal_50 | medgemma-4b | concise | single_vlm | 52.0% | 76.0% | 0.210 | 0.049 | 100.0% | 52.0% | 0.28s | 17.5 | 1.30 |
| mimic_formal_50 | medgemma-4b | concise | multi_agent | 44.0% | 66.0% | 0.236 | 0.058 | 88.0% | 44.0% | 0.67s | 53.5 | 1.30 |
| mimic_formal_50 | medgemma-27b | labels_only | single_vlm | 50.0% | 60.0% | 0.171 | 0.042 | 100.0% | 20.0% | 0.45s | 5.8 | 1.12 |
| mimic_formal_50 | medgemma-27b | labels_only | multi_agent | 50.0% | 68.0% | 0.248 | 0.056 | 96.0% | 40.0% | 3.51s | 73.2 | 1.36 |
| mimic_formal_50 | medgemma-4b | labels_only | single_vlm | 52.0% | 70.0% | 0.252 | 0.047 | 96.0% | 44.0% | 0.20s | 7.5 | 1.24 |
| mimic_formal_50 | medgemma-4b | labels_only | multi_agent | 48.0% | 66.0% | 0.228 | 0.053 | 92.0% | 40.0% | 0.80s | 67.8 | 1.32 |
| mimic_formal_50 | medgemma-4b | structured | single_vlm | 50.0% | 68.0% | 0.239 | 0.047 | 96.0% | 40.0% | 0.76s | 30.2 | 1.22 |
| mimic_formal_50 | medgemma-4b | structured | multi_agent | 48.0% | 64.0% | 0.188 | 0.053 | 96.0% | 32.0% | 0.70s | 56.1 | 1.26 |
| private_formal_50 | chexagent | structured | single_vlm | 36.0% | 78.0% | 0.436 | 0.147 | 64.0% | 92.0% | 0.95s | 13.7 | 2.62 |
| private_formal_50 | chexagent | structured | multi_agent | 52.0% | 62.0% | 0.317 | 0.038 | 96.0% | 28.0% | 4.58s | 54.2 | 1.06 |
| private_formal_50 | chexagent | concise | single_vlm | 32.0% | 74.0% | 0.429 | 0.182 | 56.0% | 92.0% | 1.11s | 15.8 | 2.98 |
| private_formal_50 | chexagent | concise | multi_agent | 54.0% | 62.0% | 0.217 | 0.044 | 100.0% | 24.0% | 2.95s | 36.9 | 1.12 |
| private_formal_50 | chexagent | labels_only | single_vlm | 46.0% | 82.0% | 0.457 | 0.129 | 84.0% | 80.0% | 1.00s | 11.9 | 2.46 |
| private_formal_50 | chexagent | labels_only | multi_agent | 50.0% | 56.0% | 0.118 | 0.049 | 100.0% | 12.0% | 2.61s | 21.9 | 1.10 |
| private_formal_50 | medgemma-27b | structured | single_vlm | 50.0% | 54.0% | 0.125 | 0.042 | 100.0% | 8.0% | 4.02s | 85.6 | 1.00 |
| private_formal_50 | medgemma-27b | structured | multi_agent | 46.0% | 62.0% | 0.174 | 0.049 | 88.0% | 36.0% | 5.77s | 123.7 | 1.16 |
| private_formal_50 | medgemma-27b | concise | single_vlm | 50.0% | 52.0% | 0.092 | 0.053 | 100.0% | 4.0% | 2.88s | 60.8 | 1.10 |
| private_formal_50 | medgemma-27b | concise | multi_agent | 36.0% | 64.0% | 0.216 | 0.089 | 64.0% | 64.0% | 3.46s | 71.8 | 1.66 |
| private_formal_50 | medgemma-4b | concise | single_vlm | 50.0% | 56.0% | 0.120 | 0.051 | 84.0% | 28.0% | 0.25s | 13.3 | 1.12 |
| private_formal_50 | medgemma-4b | concise | multi_agent | 42.0% | 54.0% | 0.158 | 0.058 | 80.0% | 28.0% | 0.62s | 48.9 | 1.20 |
| private_formal_50 | medgemma-27b | labels_only | single_vlm | 52.0% | 54.0% | 0.112 | 0.042 | 100.0% | 8.0% | 0.41s | 5.1 | 1.00 |
| private_formal_50 | medgemma-27b | labels_only | multi_agent | 52.0% | 60.0% | 0.163 | 0.053 | 100.0% | 20.0% | 3.25s | 67.1 | 1.20 |
| private_formal_50 | medgemma-4b | labels_only | single_vlm | 52.0% | 52.0% | 0.084 | 0.047 | 96.0% | 8.0% | 0.18s | 6.4 | 1.04 |
| private_formal_50 | medgemma-4b | labels_only | multi_agent | 50.0% | 56.0% | 0.128 | 0.049 | 92.0% | 20.0% | 0.76s | 64.2 | 1.10 |
| private_formal_50 | medgemma-4b | structured | single_vlm | 52.0% | 56.0% | 0.118 | 0.047 | 88.0% | 24.0% | 0.35s | 20.0 | 1.08 |
| private_formal_50 | medgemma-4b | structured | multi_agent | 50.0% | 60.0% | 0.179 | 0.045 | 92.0% | 28.0% | 0.63s | 50.4 | 1.10 |

## Suspicious Rows To Inspect

No suspicious rows detected by the current heuristic.
