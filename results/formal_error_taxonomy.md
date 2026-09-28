# Formal Error Taxonomy

This table partitions case outcomes into mutually interpretable categories to avoid relying on exact match alone.

| Dataset | Model | Prompt | Workflow | Exact | Abnormal->Normal | Normal->Abnormal | Undercall | Overcall | Mixed Partial | Wrong Type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mimic_formal_50 | chexagent | concise | multi_agent | 38.0% | 32.0% | 14.0% | 0.0% | 2.0% | 2.0% | 12.0% |
| mimic_formal_50 | chexagent | concise | single_vlm | 6.0% | 8.0% | 44.0% | 0.0% | 22.0% | 0.0% | 20.0% |
| mimic_formal_50 | chexagent | labels_only | multi_agent | 32.0% | 12.0% | 18.0% | 0.0% | 10.0% | 4.0% | 24.0% |
| mimic_formal_50 | chexagent | labels_only | single_vlm | 6.0% | 8.0% | 44.0% | 0.0% | 22.0% | 0.0% | 20.0% |
| mimic_formal_50 | chexagent | structured | multi_agent | 24.0% | 18.0% | 30.0% | 0.0% | 2.0% | 6.0% | 20.0% |
| mimic_formal_50 | chexagent | structured | single_vlm | 6.0% | 12.0% | 44.0% | 0.0% | 22.0% | 2.0% | 14.0% |
| mimic_formal_50 | medgemma-27b | concise | multi_agent | 42.0% | 2.0% | 10.0% | 0.0% | 4.0% | 16.0% | 26.0% |
| mimic_formal_50 | medgemma-27b | concise | single_vlm | 52.0% | 16.0% | 0.0% | 4.0% | 2.0% | 8.0% | 18.0% |
| mimic_formal_50 | medgemma-27b | labels_only | multi_agent | 50.0% | 22.0% | 2.0% | 2.0% | 2.0% | 14.0% | 8.0% |
| mimic_formal_50 | medgemma-27b | labels_only | single_vlm | 50.0% | 38.0% | 0.0% | 6.0% | 2.0% | 2.0% | 2.0% |
| mimic_formal_50 | medgemma-27b | structured | multi_agent | 46.0% | 10.0% | 4.0% | 14.0% | 2.0% | 12.0% | 12.0% |
| mimic_formal_50 | medgemma-27b | structured | single_vlm | 50.0% | 30.0% | 0.0% | 6.0% | 0.0% | 4.0% | 10.0% |
| mimic_formal_50 | medgemma-4b | concise | multi_agent | 44.0% | 12.0% | 6.0% | 4.0% | 4.0% | 14.0% | 16.0% |
| mimic_formal_50 | medgemma-4b | concise | single_vlm | 52.0% | 16.0% | 0.0% | 2.0% | 6.0% | 16.0% | 8.0% |
| mimic_formal_50 | medgemma-4b | labels_only | multi_agent | 48.0% | 20.0% | 4.0% | 8.0% | 2.0% | 8.0% | 10.0% |
| mimic_formal_50 | medgemma-4b | labels_only | single_vlm | 52.0% | 24.0% | 2.0% | 0.0% | 4.0% | 14.0% | 4.0% |
| mimic_formal_50 | medgemma-4b | structured | multi_agent | 48.0% | 20.0% | 2.0% | 2.0% | 4.0% | 10.0% | 14.0% |
| mimic_formal_50 | medgemma-4b | structured | single_vlm | 50.0% | 26.0% | 2.0% | 2.0% | 0.0% | 16.0% | 4.0% |
| private_formal_50 | chexagent | concise | multi_agent | 54.0% | 38.0% | 0.0% | 0.0% | 2.0% | 6.0% | 0.0% |
| private_formal_50 | chexagent | concise | single_vlm | 32.0% | 2.0% | 22.0% | 0.0% | 32.0% | 10.0% | 2.0% |
| private_formal_50 | chexagent | labels_only | multi_agent | 50.0% | 44.0% | 0.0% | 0.0% | 2.0% | 4.0% | 0.0% |
| private_formal_50 | chexagent | labels_only | single_vlm | 46.0% | 10.0% | 8.0% | 0.0% | 28.0% | 8.0% | 0.0% |
| private_formal_50 | chexagent | structured | multi_agent | 52.0% | 28.0% | 2.0% | 4.0% | 0.0% | 6.0% | 8.0% |
| private_formal_50 | chexagent | structured | single_vlm | 36.0% | 2.0% | 18.0% | 2.0% | 32.0% | 8.0% | 2.0% |
| private_formal_50 | medgemma-27b | concise | multi_agent | 36.0% | 4.0% | 18.0% | 2.0% | 14.0% | 12.0% | 14.0% |
| private_formal_50 | medgemma-27b | concise | single_vlm | 50.0% | 32.0% | 0.0% | 2.0% | 0.0% | 0.0% | 16.0% |
| private_formal_50 | medgemma-27b | labels_only | multi_agent | 52.0% | 30.0% | 0.0% | 2.0% | 2.0% | 4.0% | 10.0% |
| private_formal_50 | medgemma-27b | labels_only | single_vlm | 52.0% | 44.0% | 0.0% | 2.0% | 0.0% | 0.0% | 2.0% |
| private_formal_50 | medgemma-27b | structured | multi_agent | 46.0% | 16.0% | 6.0% | 8.0% | 2.0% | 6.0% | 16.0% |
| private_formal_50 | medgemma-27b | structured | single_vlm | 50.0% | 46.0% | 0.0% | 4.0% | 0.0% | 0.0% | 0.0% |
| private_formal_50 | medgemma-4b | concise | multi_agent | 42.0% | 24.0% | 10.0% | 2.0% | 4.0% | 6.0% | 12.0% |
| private_formal_50 | medgemma-4b | concise | single_vlm | 50.0% | 26.0% | 8.0% | 0.0% | 4.0% | 2.0% | 10.0% |
| private_formal_50 | medgemma-4b | labels_only | multi_agent | 50.0% | 30.0% | 4.0% | 0.0% | 2.0% | 4.0% | 10.0% |
| private_formal_50 | medgemma-4b | labels_only | single_vlm | 52.0% | 42.0% | 2.0% | 0.0% | 0.0% | 0.0% | 4.0% |
| private_formal_50 | medgemma-4b | structured | multi_agent | 50.0% | 32.0% | 4.0% | 4.0% | 4.0% | 2.0% | 4.0% |
| private_formal_50 | medgemma-4b | structured | single_vlm | 52.0% | 32.0% | 6.0% | 0.0% | 2.0% | 2.0% | 6.0% |