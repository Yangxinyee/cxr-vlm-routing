# Decision-Time Routing Evaluation

| Config | Strategy | Macro F1 | Latency(s) | Tokens | Multi(%) |
| --- | --- | ---: | ---: | ---: | ---: |
| mimic_formal_50__MedGemma-27B__structured | always_multi | 0.335 | 6.27 | 135.1 | 100.0 |
| mimic_formal_50__MedGemma-27B__structured | always_single | 0.138 | 4.86 | 104.3 | 0.0 |
| mimic_formal_50__MedGemma-27B__structured | heuristic_a | 0.138 | 4.86 | 104.3 | 0.0 |
| mimic_formal_50__MedGemma-27B__structured | heuristic_b | 0.331 | 5.26 | 112.5 | 40.0 |
| mimic_formal_50__MedGemma-27B__structured | learned_5fold | 0.162 | 4.78 | 102.2 | 10.0 |
| mimic_formal_50__MedGemma-27B__structured | oracle | 0.338 | 5.08 | 109.0 | 20.0 |
| private_formal_50__CheXagent-8B__labels_only | always_multi | 0.118 | 2.61 | 21.9 | 100.0 |
| private_formal_50__CheXagent-8B__labels_only | always_single | 0.457 | 1.00 | 11.9 | 0.0 |
| private_formal_50__CheXagent-8B__labels_only | heuristic_a | 0.457 | 1.00 | 11.9 | 0.0 |
| private_formal_50__CheXagent-8B__labels_only | heuristic_b | 0.488 | 1.31 | 15.0 | 18.0 |
| private_formal_50__CheXagent-8B__labels_only | learned_5fold | 0.463 | 1.03 | 12.0 | 2.0 |
| private_formal_50__CheXagent-8B__labels_only | oracle | 0.488 | 1.17 | 13.6 | 10.0 |