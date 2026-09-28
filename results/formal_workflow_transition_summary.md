# Workflow Transition Summary

Counts below quantify how multi-agent changes behavior relative to single-VLM for the same cases.

| Dataset | Model | Prompt | Partial Recovery | Partial Regression | Exact Recovery | Exact Regression | Introduced FP on Normal | Reduced FP on Normal | Abnormal->Normal Collapse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mimic_formal_50 | chexagent | concise | 15 | 8 | 16 | 0 | 0 | 15 | 12 |
| mimic_formal_50 | chexagent | labels_only | 13 | 4 | 13 | 0 | 0 | 13 | 4 |
| mimic_formal_50 | chexagent | structured | 7 | 6 | 9 | 0 | 0 | 7 | 4 |
| mimic_formal_50 | medgemma-27b | concise | 7 | 9 | 1 | 6 | 5 | 0 | 0 |
| mimic_formal_50 | medgemma-27b | labels_only | 6 | 2 | 1 | 1 | 1 | 0 | 0 |
| mimic_formal_50 | medgemma-27b | structured | 9 | 2 | 0 | 2 | 2 | 0 | 1 |
| mimic_formal_50 | medgemma-4b | concise | 3 | 8 | 0 | 4 | 3 | 0 | 2 |
| mimic_formal_50 | medgemma-4b | labels_only | 3 | 5 | 1 | 3 | 1 | 0 | 4 |
| mimic_formal_50 | medgemma-4b | structured | 0 | 2 | 0 | 1 | 0 | 0 | 1 |
| private_formal_50 | chexagent | concise | 11 | 17 | 12 | 1 | 0 | 11 | 18 |
| private_formal_50 | chexagent | labels_only | 5 | 18 | 4 | 2 | 0 | 4 | 18 |
| private_formal_50 | chexagent | structured | 8 | 16 | 10 | 2 | 0 | 8 | 13 |
| private_formal_50 | medgemma-27b | concise | 15 | 9 | 2 | 9 | 9 | 0 | 0 |
| private_formal_50 | medgemma-27b | labels_only | 4 | 1 | 1 | 1 | 0 | 0 | 1 |
| private_formal_50 | medgemma-27b | structured | 8 | 4 | 1 | 3 | 3 | 0 | 0 |
| private_formal_50 | medgemma-4b | concise | 3 | 4 | 0 | 4 | 1 | 0 | 1 |
| private_formal_50 | medgemma-4b | labels_only | 3 | 1 | 0 | 1 | 1 | 0 | 0 |
| private_formal_50 | medgemma-4b | structured | 3 | 1 | 1 | 2 | 0 | 1 | 2 |